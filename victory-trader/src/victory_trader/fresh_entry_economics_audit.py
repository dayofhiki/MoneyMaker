"""Request 181: fresh entry-economics decomposition audit.

Diagnostic-only reuse of Request-178 BUY-gated POSITION episodes. This module
does not change entry or exit policy. It decomposes the first executable
post-entry minute into raw price movement versus modeled BASE execution drag,
then audits a one-minute delayed-entry counterfactual using exact minute-1 and
minute-2 executable opens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fresh_consensus_hurdle_value import FRESH_DAYS
from .selected_hot_position_value_observability import _base_return

REQUEST_ID = 181
BOOTSTRAP_SEED = 20261081
BOOTSTRAP_SAMPLES = 10_000
SEVERE_LOSS_PCT = -5.0
EPISODE_KEYS = ["trading_day", "ticker", "hot_t"]
PRICE_BINS = (-np.inf, 1.0, 2.0, 5.0, 10.0, np.inf)
PRICE_LABELS = ("<1", "1-2", "2-5", "5-10", ">=10")


def _gross_return(entry: object, exit_: object) -> float:
    e = pd.to_numeric(pd.Series([entry]), errors="coerce").iloc[0]
    x = pd.to_numeric(pd.Series([exit_]), errors="coerce").iloc[0]
    if (
        pd.isna(e)
        or pd.isna(x)
        or not np.isfinite(float(e))
        or not np.isfinite(float(x))
        or float(e) <= 0
        or float(x) <= 0
    ):
        return np.nan
    return (float(x) / float(e) - 1.0) * 100.0


def _day_balanced(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return None
    daily = (
        pd.DataFrame(
            {
                "day": frame.loc[valid, "trading_day"].astype(str),
                "value": values.loc[valid].to_numpy(dtype=float),
            }
        )
        .groupby("day", sort=True)["value"]
        .mean()
    )
    return float(daily.mean()) if len(daily) else None


def _bootstrap_daily_difference(
    frame: pd.DataFrame,
) -> dict[str, float | int | None]:
    values = pd.to_numeric(
        frame["delayed_minus_immediate_base_pct"],
        errors="coerce",
    )
    valid = values.notna()
    daily = (
        pd.DataFrame(
            {
                "day": frame.loc[valid, "trading_day"].astype(str),
                "value": values.loc[valid].to_numpy(dtype=float),
            }
        )
        .groupby("day", sort=True)["value"]
        .mean()
        .to_numpy(dtype=float)
    )
    if len(daily) < 2:
        return {
            "days": int(len(daily)),
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": float(daily.mean()) if len(daily) else None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(
        0,
        len(daily),
        size=(BOOTSTRAP_SAMPLES, len(daily)),
    )
    draws = daily[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(daily)),
        "samples": BOOTSTRAP_SAMPLES,
        "mean_pct": float(daily.mean()),
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def build_episode_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    records: list[dict[str, object]] = []
    total_episodes = int(frame.loc[:, EPISODE_KEYS].drop_duplicates().shape[0])
    minute1_episodes = 0
    minute2_episodes = 0

    for _, group in frame.groupby(EPISODE_KEYS, sort=False):
        work = group.copy()
        work["_minute"] = pd.to_numeric(
            work["minutes_held"],
            errors="coerce",
        )
        work = work.loc[
            work["_minute"].notna() & work["_minute"].mod(1).eq(0)
        ].copy()
        work["_minute"] = work["_minute"].astype(int)
        by_minute = {
            int(row["_minute"]): row
            for _, row in work.sort_values("_minute", kind="stable").iterrows()
        }
        row1 = by_minute.get(1)
        if row1 is None:
            continue

        entry = pd.to_numeric(
            pd.Series([row1.get("entry_open")]),
            errors="coerce",
        ).iloc[0]
        minute1_open = pd.to_numeric(
            pd.Series([row1.get("exit_reference_open")]),
            errors="coerce",
        ).iloc[0]
        immediate_base = pd.to_numeric(
            pd.Series([row1.get("exit_now_base_return_pct")]),
            errors="coerce",
        ).iloc[0]
        if (
            pd.isna(entry)
            or pd.isna(minute1_open)
            or pd.isna(immediate_base)
            or float(entry) <= 0
            or float(minute1_open) <= 0
        ):
            continue
        minute1_episodes += 1

        immediate_gross = _gross_return(entry, minute1_open)
        immediate_drag = (
            float(immediate_gross) - float(immediate_base)
            if pd.notna(immediate_gross)
            else np.nan
        )

        row2 = by_minute.get(2)
        minute2_open = np.nan
        delayed_gross = np.nan
        delayed_base = np.nan
        delayed_drag = np.nan
        improvement = np.nan
        if row2 is not None:
            minute2_open = pd.to_numeric(
                pd.Series([row2.get("exit_reference_open")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(minute2_open) and float(minute2_open) > 0:
                minute2_episodes += 1
                delayed_gross = _gross_return(minute1_open, minute2_open)
                delayed_base = _base_return(
                    float(minute1_open),
                    float(minute2_open),
                )
                if pd.notna(delayed_gross) and pd.notna(delayed_base):
                    delayed_drag = float(delayed_gross) - float(delayed_base)
                    improvement = float(delayed_base) - float(immediate_base)

        records.append(
            {
                "trading_day": str(row1["trading_day"]),
                "ticker": str(row1["ticker"]).upper(),
                "hot_t": int(row1["hot_t"]),
                "entry_open": float(entry),
                "minute1_open": float(minute1_open),
                "minute2_open": (
                    float(minute2_open) if pd.notna(minute2_open) else np.nan
                ),
                "immediate_gross_pct": float(immediate_gross),
                "immediate_base_pct": float(immediate_base),
                "immediate_execution_drag_pct": float(immediate_drag),
                "delayed_gross_pct": (
                    float(delayed_gross) if pd.notna(delayed_gross) else np.nan
                ),
                "delayed_base_pct": (
                    float(delayed_base) if pd.notna(delayed_base) else np.nan
                ),
                "delayed_execution_drag_pct": (
                    float(delayed_drag) if pd.notna(delayed_drag) else np.nan
                ),
                "delayed_minus_immediate_base_pct": (
                    float(improvement) if pd.notna(improvement) else np.nan
                ),
            }
        )

    audit = pd.DataFrame(records)
    if len(audit):
        audit["entry_price_bin"] = pd.cut(
            audit["entry_open"],
            bins=PRICE_BINS,
            labels=PRICE_LABELS,
            right=False,
        ).astype(str)

    return audit, {
        "total_position_episodes": total_episodes,
        "minute1_exact_episodes": minute1_episodes,
        "minute2_exact_episodes": minute2_episodes,
        "minute1_coverage": (
            float(minute1_episodes / total_episodes)
            if total_episodes
            else None
        ),
        "minute2_coverage": (
            float(minute2_episodes / total_episodes)
            if total_episodes
            else None
        ),
        "minute2_given_minute1_coverage": (
            float(minute2_episodes / minute1_episodes)
            if minute1_episodes
            else None
        ),
    }


def _summary(frame: pd.DataFrame) -> dict[str, object]:
    immediate_gross = pd.to_numeric(
        frame["immediate_gross_pct"],
        errors="coerce",
    )
    immediate_base = pd.to_numeric(
        frame["immediate_base_pct"],
        errors="coerce",
    )
    immediate_drag = pd.to_numeric(
        frame["immediate_execution_drag_pct"],
        errors="coerce",
    )
    delayed_gross = pd.to_numeric(
        frame["delayed_gross_pct"],
        errors="coerce",
    )
    delayed_base = pd.to_numeric(
        frame["delayed_base_pct"],
        errors="coerce",
    )
    delayed_drag = pd.to_numeric(
        frame["delayed_execution_drag_pct"],
        errors="coerce",
    )
    improvement = pd.to_numeric(
        frame["delayed_minus_immediate_base_pct"],
        errors="coerce",
    )

    valid_immediate = immediate_base.notna() & immediate_gross.notna()
    valid_delayed = delayed_base.notna() & delayed_gross.notna()

    return {
        "rows": int(len(frame)),
        "immediate_gross_mean_pct": (
            float(immediate_gross.mean()) if immediate_gross.notna().any() else None
        ),
        "immediate_base_mean_pct": (
            float(immediate_base.mean()) if immediate_base.notna().any() else None
        ),
        "immediate_execution_drag_mean_pct": (
            float(immediate_drag.mean()) if immediate_drag.notna().any() else None
        ),
        "immediate_day_balanced_gross_mean_pct": _day_balanced(
            frame, "immediate_gross_pct"
        ),
        "immediate_day_balanced_base_mean_pct": _day_balanced(
            frame, "immediate_base_pct"
        ),
        "immediate_day_balanced_drag_mean_pct": _day_balanced(
            frame, "immediate_execution_drag_pct"
        ),
        "immediate_positive_base_rate": (
            float(immediate_base.loc[valid_immediate].gt(0).mean())
            if valid_immediate.any()
            else None
        ),
        "immediate_severe_loss_rate": (
            float(
                immediate_base.loc[valid_immediate]
                .le(SEVERE_LOSS_PCT)
                .mean()
            )
            if valid_immediate.any()
            else None
        ),
        "positive_gross_but_negative_base_rate": (
            float(
                (
                    immediate_gross.loc[valid_immediate].gt(0)
                    & immediate_base.loc[valid_immediate].lt(0)
                ).mean()
            )
            if valid_immediate.any()
            else None
        ),
        "delayed_rows": int(valid_delayed.sum()),
        "delayed_gross_mean_pct": (
            float(delayed_gross.loc[valid_delayed].mean())
            if valid_delayed.any()
            else None
        ),
        "delayed_base_mean_pct": (
            float(delayed_base.loc[valid_delayed].mean())
            if valid_delayed.any()
            else None
        ),
        "delayed_execution_drag_mean_pct": (
            float(delayed_drag.loc[valid_delayed].mean())
            if valid_delayed.any()
            else None
        ),
        "delayed_day_balanced_gross_mean_pct": _day_balanced(
            frame, "delayed_gross_pct"
        ),
        "delayed_day_balanced_base_mean_pct": _day_balanced(
            frame, "delayed_base_pct"
        ),
        "delayed_day_balanced_drag_mean_pct": _day_balanced(
            frame, "delayed_execution_drag_pct"
        ),
        "delayed_positive_base_rate": (
            float(delayed_base.loc[valid_delayed].gt(0).mean())
            if valid_delayed.any()
            else None
        ),
        "delayed_severe_loss_rate": (
            float(delayed_base.loc[valid_delayed].le(SEVERE_LOSS_PCT).mean())
            if valid_delayed.any()
            else None
        ),
        "delayed_minus_immediate_base_mean_pct": (
            float(improvement.mean()) if improvement.notna().any() else None
        ),
        "delayed_minus_immediate_day_balanced_pct": _day_balanced(
            frame, "delayed_minus_immediate_base_pct"
        ),
        "delayed_minus_immediate_bootstrap": _bootstrap_daily_difference(frame),
    }


def _group_summaries(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, group in frame.groupby(column, sort=True, observed=True):
        result[str(key)] = _summary(group)
    return result


def _dominant_bottleneck(summary: dict[str, object]) -> str:
    gross = summary["immediate_gross_mean_pct"]
    drag = summary["immediate_execution_drag_mean_pct"]
    if gross is None or drag is None:
        return "unknown"
    gross = float(gross)
    drag = float(drag)
    if gross >= 0:
        return "mixed"
    adverse = abs(gross)
    if drag > adverse:
        return "execution-dominant"
    if adverse > drag:
        return "price-move-dominant"
    return "mixed"


def evaluate(
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(position_paths) != len(FRESH_DAYS):
        raise ValueError("request 181 requires complete request 178 position shards")
    fresh = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    found_days = sorted(fresh["trading_day"].astype(str).unique())
    if found_days != list(FRESH_DAYS):
        raise ValueError(
            f"request 181 expected {FRESH_DAYS}, found {found_days}"
        )

    audit, coverage = build_episode_audit(fresh)
    if audit.empty:
        raise ValueError("request 181 produced no auditable episodes")

    summary = _summary(audit)
    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "policy_changed": False,
        "promotion_eligible": False,
        "coverage": coverage,
        "summary": summary,
        "dominant_first_minute_bottleneck": _dominant_bottleneck(summary),
        "by_day": _group_summaries(audit, "trading_day"),
        "by_entry_price_bin": _group_summaries(audit, "entry_price_bin"),
        "interpretation_contract": {
            "execution_dominant": (
                "mean execution drag exceeds absolute adverse gross move"
            ),
            "price_move_dominant": (
                "absolute adverse gross move exceeds mean execution drag"
            ),
            "mixed": "otherwise or pooled gross movement is positive",
            "robust_wait_signal": (
                "delayed-minus-immediate bootstrap 95% lower bound > 0"
            ),
        },
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    audit.to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fresh_dir,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
