"""Request 199: causal first-passage directional-edge diagnostic.

FIT-only training, chronological calibration-only evaluation. No Request-178
fresh outcomes are opened.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .causal_second_risk_compatible_entry import (
    BASE_SCENARIO,
    ENTRY_EXPIRY_MS,
    LATENCY_MS,
    RULES,
    SecondStore,
    _bars_from_seconds,
    _enrich_rich_second,
    _state_filter,
)
from .future_cost_cover_state_observability import (
    _day_weights,
    _safe_ap,
    _safe_auc,
)
from .execution_costs import modeled_buy_fill
from .market_regime_position_value import attach_market_regime, build_market_regime
from .rich_second_position_value import RICH_SECOND_FEATURES
from .shadow_entry_repricing_decomposition import FEATURES, add_shadow_features

REQUEST_ID = 199
MODEL_SEED = 20261130
QUANTILES = (0.80, 0.90, 0.95)


@dataclass(frozen=True)
class Head:
    model: HistGradientBoostingClassifier
    columns: tuple[str, ...]


def _columns(frame: pd.DataFrame, *, include_rich: bool) -> tuple[str, ...]:
    excluded = set() if include_rich else set(RICH_SECOND_FEATURES)
    return tuple(
        column
        for column in FEATURES
        if column not in excluded
        and column in frame.columns
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    )


def _x(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def prepare_directional_states(
    positions: pd.DataFrame,
    scan: pd.DataFrame,
    store: SecondStore,
) -> pd.DataFrame:
    """Build the exact Request-197 causal feature state plus entry fill only.

    First-passage diagnostics do not need a full stop/take/trailing replay before
    labeling. This preserves entry semantics while avoiding two redundant full
    trajectory replays per state.
    """
    base = _state_filter(positions)
    base = add_shadow_features(base)
    regime = build_market_regime(scan)
    base = attach_market_regime(base, regime)
    base = _enrich_rich_second(base, store)

    statuses: list[str] = []
    fill_times: list[float] = []
    modeled_prices: list[float] = []

    for row in base.to_dict("records"):
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        decision_t = int(row["state_t"])
        seconds = store.seconds(day, ticker)
        windows = store.halts(day)
        start_t = decision_t + LATENCY_MS
        end_t = start_t + ENTRY_EXPIRY_MS
        path = seconds.loc[
            pd.to_numeric(seconds["t"], errors="coerce").between(
                start_t,
                end_t,
                inclusive="both",
            )
        ].copy()
        bars = _bars_from_seconds(
            path,
            ticker=ticker,
            intervals=windows,
        )
        eligible = next(
            (
                bar
                for bar in bars
                if not bar.halted
                and start_t <= int(bar.t) <= end_t
            ),
            None,
        )
        if eligible is None:
            statuses.append("entry_unavailable")
            fill_times.append(np.nan)
            modeled_prices.append(np.nan)
            continue
        statuses.append("entry_filled")
        fill_times.append(float(eligible.t))
        modeled_prices.append(
            float(modeled_buy_fill(float(eligible.o), BASE_SCENARIO))
        )

    base["replay_status"] = statuses
    base["entry_fill_t"] = fill_times
    base["entry_modeled_price"] = modeled_prices
    return base


def add_first_passage(
    frame: pd.DataFrame,
    store: SecondStore,
    *,
    take_pct: float,
    stop_pct: float = 3.0,
    max_hold_ms: int = 30 * 60_000,
) -> pd.DataFrame:
    result = frame.copy()
    statuses: list[str] = []
    proxy: list[float] = []

    for row in result.to_dict("records"):
        if str(row["replay_status"]) == "entry_unavailable":
            statuses.append("entry_unavailable")
            proxy.append(np.nan)
            continue

        fill_t = pd.to_numeric(
            pd.Series([row.get("entry_fill_t")]),
            errors="coerce",
        ).iloc[0]
        entry_modeled = pd.to_numeric(
            pd.Series([row.get("entry_modeled_price")]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(fill_t) or pd.isna(entry_modeled):
            statuses.append("entry_unavailable")
            proxy.append(np.nan)
            continue

        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        seconds = store.seconds(day, ticker)
        cap_t = int(fill_t) + int(max_hold_ms)
        path = seconds.loc[
            pd.to_numeric(seconds["t"], errors="coerce").between(
                int(fill_t),
                cap_t,
                inclusive="both",
            )
        ].copy()
        bars = _bars_from_seconds(
            path,
            ticker=ticker,
            intervals=store.halts(day),
        )
        stop_level = float(entry_modeled) * (1.0 - stop_pct / 100.0)
        take_level = float(entry_modeled) * (1.0 + take_pct / 100.0)
        outcome = "neither"
        for bar in bars:
            if bar.halted:
                continue
            stop_hit = bar.low <= stop_level
            take_hit = bar.h >= take_level
            if stop_hit and take_hit:
                outcome = "ambiguous"
                break
            if stop_hit:
                outcome = "stop_first"
                break
            if take_hit:
                outcome = "take_first"
                break

        statuses.append(outcome)
        if outcome == "take_first":
            proxy.append(float(take_pct))
        elif outcome == "stop_first":
            proxy.append(-float(stop_pct))
        elif outcome == "neither":
            proxy.append(0.0)
        else:
            proxy.append(np.nan)

    result["first_passage_status"] = statuses
    result["barrier_proxy_pct"] = proxy
    result["take_before_stop"] = (
        result["first_passage_status"].eq("take_first")
        .where(
            ~result["first_passage_status"].isin(
                ["entry_unavailable", "ambiguous"]
            )
        )
    )
    return result


def fit_head(
    frame: pd.DataFrame,
    *,
    include_rich: bool,
    seed: int,
) -> Head:
    y = pd.to_numeric(frame["take_before_stop"], errors="coerce")
    valid = y.notna()
    train = frame.loc[valid].copy()
    actual = y.loc[valid].astype(int)
    if len(train) < 500 or actual.nunique() < 2:
        raise ValueError(
            "request 199 insufficient first-passage training support"
        )
    columns = _columns(train, include_rich=include_rich)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _x(train, columns),
        actual,
        sample_weight=_day_weights(train),
    )
    return Head(model=model, columns=columns)


def predict(frame: pd.DataFrame, head: Head) -> np.ndarray:
    return head.model.predict_proba(_x(frame, head.columns))[:, 1]


def classifier_metrics(
    frame: pd.DataFrame,
    score_column: str,
) -> dict[str, object]:
    y = pd.to_numeric(frame["take_before_stop"], errors="coerce")
    s = pd.to_numeric(frame[score_column], errors="coerce")
    valid = y.notna() & s.notna()
    work = frame.loc[valid].copy()
    y = y.loc[valid]
    s = s.loc[valid]

    decisive = work["first_passage_status"].isin(
        ["take_first", "stop_first"]
    )
    decisive_index = work.index[decisive]
    by_day: dict[str, object] = {}
    for day, group in work.groupby(work["trading_day"].astype(str), sort=True):
        idx = group.index
        by_day[str(day)] = {
            "rows": int(len(group)),
            "take_rate": float(y.loc[idx].mean()) if len(group) else None,
            "auc": _safe_auc(y.loc[idx], s.loc[idx]),
        }

    return {
        "rows": int(valid.sum()),
        "take_rate": float(y.mean()) if int(valid.sum()) else None,
        "auc": _safe_auc(y, s),
        "average_precision": _safe_ap(y, s),
        "decisive_rows": int(len(decisive_index)),
        "decisive_auc": _safe_auc(
            y.loc[decisive_index],
            s.loc[decisive_index],
        ),
        "decisive_average_precision": _safe_ap(
            y.loc[decisive_index],
            s.loc[decisive_index],
        ),
        "by_day": by_day,
    }


def _day_balanced_proxy(frame: pd.DataFrame) -> float | None:
    proxy = pd.to_numeric(frame["barrier_proxy_pct"], errors="coerce")
    valid = proxy.notna()
    work = frame.loc[valid].copy()
    if work.empty:
        return None
    work["_proxy"] = proxy.loc[valid]
    daily = work.groupby(work["trading_day"].astype(str), sort=True)["_proxy"].mean()
    return float(daily.mean()) if len(daily) else None


def top_groups(frame: pd.DataFrame) -> dict[str, object]:
    score = pd.to_numeric(frame["full_score"], errors="coerce")
    outputs: dict[str, object] = {}
    for q in QUANTILES:
        cutoff = float(score.dropna().quantile(q))
        part = frame.loc[score.ge(cutoff)].copy()
        status = part["first_passage_status"].astype(str)
        outputs[f"p{int(q * 100)}"] = {
            "score_cutoff": cutoff,
            "states": int(len(part)),
            "episodes": int(
                part[["trading_day", "ticker", "hot_t"]]
                .drop_duplicates()
                .shape[0]
            ),
            "take_first_rate": float(status.eq("take_first").mean()),
            "stop_first_rate": float(status.eq("stop_first").mean()),
            "neither_rate": float(status.eq("neither").mean()),
            "ambiguous_rate": float(status.eq("ambiguous").mean()),
            "entry_unavailable_rate": float(
                status.eq("entry_unavailable").mean()
            ),
            "barrier_proxy_mean_pct": (
                float(
                    pd.to_numeric(
                        part["barrier_proxy_pct"],
                        errors="coerce",
                    ).mean()
                )
                if len(part)
                else None
            ),
            "day_balanced_barrier_proxy_pct": _day_balanced_proxy(part),
        }
    return outputs


def evaluate_rule(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    store: SecondStore,
    *,
    rule_name: str,
    seed_offset: int,
) -> dict[str, object]:
    take_pct = RULES[rule_name].take_pct
    fit_labeled = add_first_passage(
        fit,
        store,
        take_pct=take_pct,
    )
    cal_labeled = add_first_passage(
        calibration,
        store,
        take_pct=take_pct,
    )

    full = fit_head(
        fit_labeled,
        include_rich=True,
        seed=MODEL_SEED + seed_offset,
    )
    no_rich = fit_head(
        fit_labeled,
        include_rich=False,
        seed=MODEL_SEED + 10 + seed_offset,
    )
    cal_labeled["full_score"] = predict(cal_labeled, full)
    cal_labeled["no_rich_score"] = predict(cal_labeled, no_rich)

    full_metrics = classifier_metrics(cal_labeled, "full_score")
    no_rich_metrics = classifier_metrics(cal_labeled, "no_rich_score")
    full_auc = full_metrics["auc"]
    no_rich_auc = no_rich_metrics["auc"]

    return {
        "take_pct": float(take_pct),
        "status_counts": {
            str(k): int(v)
            for k, v in cal_labeled["first_passage_status"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "full": full_metrics,
        "no_rich_second": no_rich_metrics,
        "full_minus_no_rich_auc": (
            float(full_auc) - float(no_rich_auc)
            if full_auc is not None and no_rich_auc is not None
            else None
        ),
        "top_groups_full_score": top_groups(cal_labeled),
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_path)
    cal_positions = pd.read_parquet(calibration_path)
    scan = pd.read_parquet(history_scan_path)

    fit_days = set(fit_positions["trading_day"].astype(str))
    cal_days = set(cal_positions["trading_day"].astype(str))
    fit_scan = scan.loc[scan["trading_day"].astype(str).isin(fit_days)].copy()
    cal_scan = scan.loc[scan["trading_day"].astype(str).isin(cal_days)].copy()

    store = SecondStore()
    fit_states = prepare_directional_states(
        fit_positions,
        fit_scan,
        store,
    )
    cal_states = prepare_directional_states(
        cal_positions,
        cal_scan,
        store,
    )

    diagnostics: dict[str, object] = {}
    for i, rule_name in enumerate(RULES):
        diagnostics[rule_name] = evaluate_rule(
            fit_states,
            cal_states,
            store,
            rule_name=rule_name,
            seed_offset=i,
        )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "fresh_evaluated": False,
        "promotion_eligible": False,
        "base_scenario": BASE_SCENARIO.name,
        "training_partition": "request171_fit_only",
        "evaluation_partition": "request171_calibration_only",
        "diagnostics": diagnostics,
        "halt_feed_by_day": dict(sorted(store.halt_status.items())),
        "second_client_stats": store.client.stats.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--history-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
