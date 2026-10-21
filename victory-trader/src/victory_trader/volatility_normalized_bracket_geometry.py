"""Request 206: volatility-normalized bracket geometry bridge.

No predictive model. Replays fixed and volatility-normalized full-position
brackets from the same earliest executable calibration shadow state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .attention_replay import MINUTE_MS
from .causal_second_risk_compatible_entry import (
    BASE_SCENARIO,
    LATENCY_MS,
    SecondStore,
    _bars_from_seconds,
)
from .causal_first_passage_directional_edge import (
    prepare_directional_states,
)
from .execution_costs import modeled_sell_fill
from .shadow_entry_repricing_decomposition import EPISODE_KEYS

REQUEST_ID = 206
LOOKBACK_MS = 5 * MINUTE_MS
MIN_ACTIVE_MINUTES = 3
R_FLOOR_PCT = 3.0
R_CAP_PCT = 8.0
MAX_HOLD_MS = 30 * MINUTE_MS
EXIT_PAD_MS = 10_000

POLICIES = {
    "fixed_tight": ("fixed", 3.0, 5.0),
    "fixed_runner": ("fixed", 3.0, 10.0),
    "adaptive_2R": ("adaptive", 1.0, 2.0),
    "adaptive_3R": ("adaptive", 1.0, 3.0),
}


@dataclass(frozen=True)
class BracketResult:
    status: str
    reason: str
    net_return_pct: float | None
    exit_fill_t: int | None
    stop_pct: float
    take_pct: float


def causal_r_pct(
    seconds: pd.DataFrame,
    decision_t: int,
) -> float:
    t = pd.to_numeric(seconds["t"], errors="coerce")
    window = seconds.loc[
        t.ge(int(decision_t) - LOOKBACK_MS)
        & t.lt(int(decision_t))
    ].copy()
    if window.empty:
        return np.nan

    window["_minute"] = (
        pd.to_numeric(window["t"], errors="coerce")
        // MINUTE_MS
    )
    ranges: list[float] = []
    for _, group in window.groupby("_minute", sort=True):
        high = pd.to_numeric(
            group["h"],
            errors="coerce",
        ).max()
        low = pd.to_numeric(
            group["l"],
            errors="coerce",
        ).min()
        if (
            pd.notna(high)
            and pd.notna(low)
            and np.isfinite(float(high))
            and np.isfinite(float(low))
            and float(high) > 0
            and float(low) > 0
            and float(high) >= float(low)
        ):
            ranges.append(
                float(
                    (
                        float(high) / float(low)
                        - 1.0
                    )
                    * 100.0
                )
            )

    if len(ranges) < MIN_ACTIVE_MINUTES:
        return np.nan
    raw = float(
        np.median(
            np.asarray(ranges[-5:], dtype=float)
        )
    )
    return float(
        np.clip(
            raw,
            R_FLOOR_PCT,
            R_CAP_PCT,
        )
    )


def replay_bracket(
    row: dict[str, object],
    store: SecondStore,
    *,
    stop_pct: float,
    take_pct: float,
) -> BracketResult:
    fill_t = pd.to_numeric(
        pd.Series([row.get("entry_fill_t")]),
        errors="coerce",
    ).iloc[0]
    entry = pd.to_numeric(
        pd.Series([row.get("entry_modeled_price")]),
        errors="coerce",
    ).iloc[0]
    if (
        pd.isna(fill_t)
        or pd.isna(entry)
        or not np.isfinite(float(entry))
        or float(entry) <= 0
    ):
        return BracketResult(
            "entry_unavailable",
            "missing_entry_fill",
            None,
            None,
            float(stop_pct),
            float(take_pct),
        )

    day = str(row["trading_day"])
    ticker = str(row["ticker"]).upper()
    fill_t_int = int(fill_t)
    seconds = store.seconds(day, ticker)
    path_end = (
        fill_t_int
        + MAX_HOLD_MS
        + LATENCY_MS
        + EXIT_PAD_MS
    )
    t = pd.to_numeric(
        seconds["t"],
        errors="coerce",
    )
    path = seconds.loc[
        t.ge(fill_t_int)
        & t.le(path_end)
    ].copy()
    bars = _bars_from_seconds(
        path,
        ticker=ticker,
        intervals=store.halts(day),
    )

    stop_level = float(entry) * (
        1.0 - float(stop_pct) / 100.0
    )
    take_level = float(entry) * (
        1.0 + float(take_pct) / 100.0
    )
    pending: tuple[str, int] | None = None

    def close_at(
        reason: str,
        bar_open: float,
        bar_t: int,
    ) -> BracketResult:
        sell = modeled_sell_fill(
            float(bar_open),
            BASE_SCENARIO,
        )
        sell *= (
            1.0
            - BASE_SCENARIO.sell_fee_bps
            / 10_000.0
        )
        net = (
            sell / float(entry) - 1.0
        ) * 100.0
        return BracketResult(
            "closed",
            reason,
            float(net),
            int(bar_t),
            float(stop_pct),
            float(take_pct),
        )

    for bar in bars:
        if bar.halted:
            continue

        if (
            pending is not None
            and int(bar.t)
            >= int(pending[1]) + LATENCY_MS
        ):
            return close_at(
                pending[0],
                float(bar.o),
                int(bar.t),
            )

        trigger_t = int(bar.t) + 1_000
        if pending is not None:
            continue

        stop_hit = float(bar.low) <= stop_level
        take_hit = float(bar.h) >= take_level
        if stop_hit and take_hit:
            return BracketResult(
                "ambiguous",
                "stop_and_take_same_second",
                None,
                None,
                float(stop_pct),
                float(take_pct),
            )
        if stop_hit:
            pending = ("stop", trigger_t)
        elif take_hit:
            pending = ("take", trigger_t)
        elif (
            trigger_t
            >= fill_t_int + MAX_HOLD_MS
        ):
            pending = ("time_cap", trigger_t)

    if pending is not None:
        return BracketResult(
            "unresolved",
            f"{pending[0]}_unfilled",
            None,
            None,
            float(stop_pct),
            float(take_pct),
        )
    return BracketResult(
        "unresolved",
        "no_exit_trigger_or_cap_fill",
        None,
        None,
        float(stop_pct),
        float(take_pct),
    )


def choose_episode_states(
    states: pd.DataFrame,
    store: SecondStore,
) -> tuple[pd.DataFrame, dict[str, int]]:
    work = states.sort_values(
        [*EPISODE_KEYS, "state_t"],
        kind="stable",
    ).copy()
    chosen: list[pd.Series] = []
    skipped_no_r = 0
    skipped_no_fill = 0

    for _, group in work.groupby(
        EPISODE_KEYS,
        sort=False,
    ):
        admitted = False
        for _, row in group.iterrows():
            seconds = store.seconds(
                str(row["trading_day"]),
                str(row["ticker"]),
            )
            r_pct = causal_r_pct(
                seconds,
                int(row["state_t"]),
            )
            if not np.isfinite(r_pct):
                skipped_no_r += 1
                continue
            if (
                str(row["replay_status"])
                == "entry_unavailable"
            ):
                skipped_no_fill += 1
                continue
            selected = row.copy()
            selected["adaptive_r_pct"] = float(
                r_pct
            )
            chosen.append(selected)
            admitted = True
            break
        if admitted:
            continue

    selected = (
        pd.DataFrame(chosen)
        if chosen
        else work.iloc[0:0].copy()
    )
    return selected, {
        "episodes_total": int(
            work.loc[:, EPISODE_KEYS]
            .drop_duplicates()
            .shape[0]
        ),
        "episodes_admitted": int(len(selected)),
        "skipped_state_no_r": int(
            skipped_no_r
        ),
        "skipped_state_no_fill": int(
            skipped_no_fill
        ),
    }


def replay_policies(
    selected: pd.DataFrame,
    store: SecondStore,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in selected.to_dict("records"):
        r_pct = float(row["adaptive_r_pct"])
        for policy, (
            mode,
            stop_value,
            take_value,
        ) in POLICIES.items():
            if mode == "adaptive":
                stop_pct = (
                    r_pct * float(stop_value)
                )
                take_pct = (
                    r_pct * float(take_value)
                )
            else:
                stop_pct = float(stop_value)
                take_pct = float(take_value)

            result = replay_bracket(
                row,
                store,
                stop_pct=stop_pct,
                take_pct=take_pct,
            )
            records.append(
                {
                    **{
                        key: row[key]
                        for key in EPISODE_KEYS
                    },
                    "state_t": int(row["state_t"]),
                    "minutes_held": float(
                        row["minutes_held"]
                    ),
                    "adaptive_r_pct": r_pct,
                    "policy": policy,
                    "stop_pct": stop_pct,
                    "take_pct": take_pct,
                    "status": result.status,
                    "exit_reason": result.reason,
                    "net_return_pct": (
                        result.net_return_pct
                    ),
                    "exit_fill_t": (
                        result.exit_fill_t
                    ),
                    "realized_r_multiple": (
                        float(
                            result.net_return_pct
                            / stop_pct
                        )
                        if result.net_return_pct
                        is not None
                        and stop_pct > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def _day_mean(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    work = frame.loc[
        frame["status"].eq("closed")
    ].copy()
    work[column] = pd.to_numeric(
        work[column],
        errors="coerce",
    )
    work = work.dropna(subset=[column])
    if work.empty:
        return pd.Series(dtype=float)
    return work.groupby(
        work["trading_day"].astype(str),
        sort=True,
    )[column].mean()


def summarize(
    frame: pd.DataFrame,
) -> dict[str, object]:
    total = int(len(frame))
    closed = frame.loc[
        frame["status"].eq("closed")
    ].copy()
    returns = pd.to_numeric(
        closed["net_return_pct"],
        errors="coerce",
    ).dropna()
    closed = closed.loc[returns.index]
    r_mult = pd.to_numeric(
        closed["realized_r_multiple"],
        errors="coerce",
    ).dropna()
    wins = returns.loc[returns.gt(0)]
    losses = returns.loc[returns.lt(0)]
    daily = _day_mean(
        frame,
        "net_return_pct",
    )
    daily_r = _day_mean(
        frame,
        "realized_r_multiple",
    )
    reasons = frame[
        "exit_reason"
    ].astype(str)
    stop = pd.to_numeric(
        frame["stop_pct"],
        errors="coerce",
    )
    return {
        "episodes": total,
        "closed": int(len(closed)),
        "closed_coverage": (
            float(len(closed) / total)
            if total
            else None
        ),
        "status_counts": {
            str(k): int(v)
            for k, v in frame["status"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "mean_net_return_pct": (
            float(returns.mean())
            if len(returns)
            else None
        ),
        "day_balanced_net_return_pct": (
            float(daily.mean())
            if len(daily)
            else None
        ),
        "positive_days": int(
            daily.gt(0).sum()
        ),
        "positive_rate": (
            float(returns.gt(0).mean())
            if len(returns)
            else None
        ),
        "mean_winner_pct": (
            float(wins.mean())
            if len(wins)
            else None
        ),
        "mean_loser_pct": (
            float(losses.mean())
            if len(losses)
            else None
        ),
        "payoff_ratio": (
            float(
                wins.mean()
                / abs(losses.mean())
            )
            if len(wins)
            and len(losses)
            and float(losses.mean()) != 0
            else None
        ),
        "severe_loss_rate": (
            float(returns.le(-5.0).mean())
            if len(returns)
            else None
        ),
        "mean_realized_r": (
            float(r_mult.mean())
            if len(r_mult)
            else None
        ),
        "day_balanced_realized_r": (
            float(daily_r.mean())
            if len(daily_r)
            else None
        ),
        "stop_exit_rate": float(
            reasons.eq("stop").mean()
        ) if total else None,
        "take_exit_rate": float(
            reasons.eq("take").mean()
        ) if total else None,
        "time_cap_exit_rate": float(
            reasons.eq("time_cap").mean()
        ) if total else None,
        "stop_pct": {
            "mean": (
                float(stop.mean())
                if stop.notna().any()
                else None
            ),
            "median": (
                float(stop.median())
                if stop.notna().any()
                else None
            ),
            "p10": (
                float(stop.quantile(0.10))
                if stop.notna().any()
                else None
            ),
            "p90": (
                float(stop.quantile(0.90))
                if stop.notna().any()
                else None
            ),
        },
        "by_day_mean_pct": {
            str(k): float(v)
            for k, v in daily.items()
        },
    }


def matched_difference(
    adaptive: pd.DataFrame,
    fixed: pd.DataFrame,
) -> dict[str, object]:
    keys = EPISODE_KEYS
    left = adaptive.loc[
        adaptive["status"].eq("closed"),
        [*keys, "net_return_pct", "realized_r_multiple"],
    ].rename(
        columns={
            "net_return_pct": "adaptive_return",
            "realized_r_multiple": "adaptive_r",
        }
    )
    right = fixed.loc[
        fixed["status"].eq("closed"),
        [*keys, "net_return_pct", "realized_r_multiple"],
    ].rename(
        columns={
            "net_return_pct": "fixed_return",
            "realized_r_multiple": "fixed_r",
        }
    )
    merged = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return {
            "matched": 0,
            "mean_return_difference_pct": None,
            "day_balanced_return_difference_pct": None,
            "mean_r_difference": None,
            "day_balanced_r_difference": None,
            "positive_improvement_days": 0,
        }

    merged["return_diff"] = (
        pd.to_numeric(
            merged["adaptive_return"],
            errors="coerce",
        )
        - pd.to_numeric(
            merged["fixed_return"],
            errors="coerce",
        )
    )
    merged["r_diff"] = (
        pd.to_numeric(
            merged["adaptive_r"],
            errors="coerce",
        )
        - pd.to_numeric(
            merged["fixed_r"],
            errors="coerce",
        )
    )
    daily_return = merged.groupby(
        merged["trading_day"].astype(str),
        sort=True,
    )["return_diff"].mean()
    daily_r = merged.groupby(
        merged["trading_day"].astype(str),
        sort=True,
    )["r_diff"].mean()
    return {
        "matched": int(len(merged)),
        "mean_return_difference_pct": float(
            merged["return_diff"].mean()
        ),
        "day_balanced_return_difference_pct": float(
            daily_return.mean()
        ),
        "mean_r_difference": float(
            merged["r_diff"].mean()
        ),
        "day_balanced_r_difference": float(
            daily_r.mean()
        ),
        "positive_improvement_days": int(
            daily_return.gt(0).sum()
        ),
        "by_day_return_difference_pct": {
            str(k): float(v)
            for k, v in daily_return.items()
        },
    }


def evaluate(
    calibration_path: Path,
    history_scan_path: Path,
    output_path: Path,
) -> int:
    positions = pd.read_parquet(
        calibration_path
    )
    scan = pd.read_parquet(
        history_scan_path
    )
    days = set(
        positions["trading_day"].astype(str)
    )
    scan = scan.loc[
        scan["trading_day"]
        .astype(str)
        .isin(days)
    ].copy()

    store = SecondStore()
    states = prepare_directional_states(
        positions,
        scan,
        store,
    )
    selected, state_audit = (
        choose_episode_states(
            states,
            store,
        )
    )
    replay = replay_policies(
        selected,
        store,
    )

    summaries = {
        policy: summarize(
            replay.loc[
                replay["policy"].eq(policy)
            ].copy()
        )
        for policy in POLICIES
    }
    comparisons = {
        "adaptive_2R_minus_fixed_tight": (
            matched_difference(
                replay.loc[
                    replay["policy"].eq(
                        "adaptive_2R"
                    )
                ],
                replay.loc[
                    replay["policy"].eq(
                        "fixed_tight"
                    )
                ],
            )
        ),
        "adaptive_3R_minus_fixed_runner": (
            matched_difference(
                replay.loc[
                    replay["policy"].eq(
                        "adaptive_3R"
                    )
                ],
                replay.loc[
                    replay["policy"].eq(
                        "fixed_runner"
                    )
                ],
            )
        ),
    }

    passes: dict[str, bool] = {}
    for adaptive, fixed, key in (
        (
            "adaptive_2R",
            "fixed_tight",
            "adaptive_2R_minus_fixed_tight",
        ),
        (
            "adaptive_3R",
            "fixed_runner",
            "adaptive_3R_minus_fixed_runner",
        ),
    ):
        a = summaries[adaptive]
        f = summaries[fixed]
        diff = comparisons[key]
        passes[adaptive] = bool(
            a["closed_coverage"] is not None
            and float(a["closed_coverage"])
            >= 0.90
            and diff[
                "day_balanced_return_difference_pct"
            ]
            is not None
            and float(
                diff[
                    "day_balanced_return_difference_pct"
                ]
            )
            >= 0.50
            and diff[
                "day_balanced_r_difference"
            ]
            is not None
            and float(
                diff[
                    "day_balanced_r_difference"
                ]
            )
            >= 0.20
            and int(
                diff[
                    "positive_improvement_days"
                ]
            )
            >= 6
            and a["severe_loss_rate"]
            is not None
            and f["severe_loss_rate"]
            is not None
            and float(
                a["severe_loss_rate"]
            )
            <= float(
                f["severe_loss_rate"]
            )
        )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "fresh_evaluated": False,
        "promotion_eligible": False,
        "evaluation_partition": (
            "request171_calibration_only"
        ),
        "risk_definition": {
            "lookback_minutes": 5,
            "minimum_active_minutes": (
                MIN_ACTIVE_MINUTES
            ),
            "raw_statistic": (
                "median completed-minute high/low range pct"
            ),
            "r_floor_pct": R_FLOOR_PCT,
            "r_cap_pct": R_CAP_PCT,
        },
        "state_audit": state_audit,
        "adaptive_r_distribution": {
            "mean": float(
                pd.to_numeric(
                    selected["adaptive_r_pct"],
                    errors="coerce",
                ).mean()
            ),
            "median": float(
                pd.to_numeric(
                    selected["adaptive_r_pct"],
                    errors="coerce",
                ).median()
            ),
            "p10": float(
                pd.to_numeric(
                    selected["adaptive_r_pct"],
                    errors="coerce",
                ).quantile(0.10)
            ),
            "p90": float(
                pd.to_numeric(
                    selected["adaptive_r_pct"],
                    errors="coerce",
                ).quantile(0.90)
            ),
        },
        "summaries": summaries,
        "matched_comparisons": comparisons,
        "geometry_pass_by_policy": passes,
        "geometry_bridge_pass": bool(
            any(passes.values())
        ),
        "halt_feed_by_day": dict(
            sorted(
                store.halt_status.items()
            )
        ),
        "second_client_stats": (
            store.client.stats.to_dict()
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--history-scan",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.calibration,
        args.history_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
