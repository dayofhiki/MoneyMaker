"""Request280: causal pre-HOT ticker-history and market-context ablation.

Request279 showed that downstream-aligned supervision improves the candidate
model but does not create a positive selected subset. Request280 keeps that
target and the downstream trading policy frozen, then changes only the
HOT-time representation.

Three representations are compared on identical May5-8 leave-one-day-out
folds and the same train-derived top-20% selection contract:

1. Request279 static HOT representation;
2. static + exact 1/3/5-minute pre-HOT ticker trajectory;
3. ticker trajectory + same-time cross-sectional market/crowding context.

Every added feature is available no later than HOT time. No entry, exit,
selection fraction, or trading date is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .causal_minute_controller_rebuild import causal_execution_scan
from .downstream_aligned_candidate import (
    MIN_RESOLUTION_OR_CASH,
    SELECTION_FRACTION,
    _safe_spearman,
    attach_downstream_target,
    candidate_columns,
    fit_pair,
    selection_metrics,
    threshold_from_training,
)
from .full_hot_fixed_policy_value import attach_fixed_value, xframe
from .joint_pullback_admission import (
    FIXED_MAX_HOLD_MINUTES,
    FIXED_STOP_LOSS_PCT,
    FIXED_TRAIL_PCT,
)
from .lagged_minute_context import CROSSFIT_DAYS
from .pullback_trailing_exit import (
    apply_exit_rule,
    build_pullback_episodes,
)
from .second_path_attention_probe import (
    BASELINE_FEATURES,
    MINUTE_MS,
    _annotate_scan,
)

REQUEST_ID = 280
LAGS = (1, 3, 5)
TRACKED_TICKER_FEATURES = tuple(BASELINE_FEATURES)
MIN_SELECTED = 100
MIN_POSITIVE_DAYS = 3
MIN_VALUE_SPEARMAN = 0.25
MAX_SEVERE_TRADE_RATE = 0.25

MARKET_FEATURES = (
    "market_universe_size",
    "market_return_median_pct",
    "market_return_q90_pct",
    "market_runner_share_ge5",
    "market_runner_share_ge10",
    "market_body_positive_share",
    "market_body_median_pct",
    "market_range_median_pct",
    "market_log_volume_median",
    "market_log_transactions_median",
    "market_attention_score_median",
)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _build_ticker_context_from_annotated(
    annotated: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    annotated = annotated.sort_values(
        ["trading_day", "ticker", "t"],
        kind="stable",
    ).copy()
    available = [
        name for name in TRACKED_TICKER_FEATURES
        if name in annotated.columns
    ]
    group = annotated.groupby(
        ["trading_day", "ticker"],
        sort=False,
    )
    created: list[str] = []
    for lag in LAGS:
        lag_t = group["t"].shift(lag)
        exact = (
            _numeric(annotated["t"]) - _numeric(lag_t)
        ).eq(lag * MINUTE_MS)
        for name in available:
            lag_name = f"prehot_lag{lag}_{name}"
            delta_name = f"prehot_delta{lag}_{name}"
            lag_value = _numeric(group[name].shift(lag)).where(exact)
            current = _numeric(annotated[name])
            annotated[lag_name] = lag_value
            annotated[delta_name] = current - lag_value
            created.extend([lag_name, delta_name])
    context = annotated.loc[
        :,
        ["trading_day", "ticker", "t", *created],
    ].copy()
    return context, tuple(created)


def build_ticker_context(
    raw_scan: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    annotated = _annotate_scan(raw_scan)
    return _build_ticker_context_from_annotated(annotated)


def _build_market_context_from_annotated(
    annotated: pd.DataFrame,
) -> pd.DataFrame:
    annotated = annotated.copy()
    annotated["_ret"] = _numeric(
        annotated["return_from_previous_close_pct"]
    )
    annotated["_body"] = _numeric(
        annotated["minute_body_return_pct"]
    )
    annotated["_range"] = _numeric(
        annotated["minute_range_pct"]
    )
    annotated["_logv"] = _numeric(
        annotated["log_minute_volume"]
    )
    annotated["_logn"] = _numeric(
        annotated["log_minute_transactions"]
    )
    annotated["_attention"] = _numeric(
        annotated["attention_score"]
    )
    annotated["_runner5"] = annotated["_ret"].ge(5.0)
    annotated["_runner10"] = annotated["_ret"].ge(10.0)
    annotated["_body_positive"] = annotated["_body"].gt(0.0)

    rows: list[dict[str, float | int | str]] = []
    for (day, timestamp), group in annotated.groupby(
        ["trading_day", "t"],
        sort=False,
    ):
        ret = group["_ret"].dropna()
        rows.append({
            "trading_day": str(day),
            "t": int(timestamp),
            "market_universe_size": int(
                group.ticker.astype(str).nunique()
            ),
            "market_return_median_pct": (
                float(ret.median()) if len(ret) else np.nan
            ),
            "market_return_q90_pct": (
                float(ret.quantile(0.90)) if len(ret) else np.nan
            ),
            "market_runner_share_ge5": float(
                group["_runner5"].mean()
            ),
            "market_runner_share_ge10": float(
                group["_runner10"].mean()
            ),
            "market_body_positive_share": float(
                group["_body_positive"].mean()
            ),
            "market_body_median_pct": float(
                group["_body"].median()
            ),
            "market_range_median_pct": float(
                group["_range"].median()
            ),
            "market_log_volume_median": float(
                group["_logv"].median()
            ),
            "market_log_transactions_median": float(
                group["_logn"].median()
            ),
            "market_attention_score_median": float(
                group["_attention"].median()
            ),
        })
    return pd.DataFrame(rows)


def build_market_context(
    raw_scan: pd.DataFrame,
) -> pd.DataFrame:
    annotated = _annotate_scan(raw_scan)
    return _build_market_context_from_annotated(annotated)


def attach_context(
    candidates: pd.DataFrame,
    raw_scan: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
]:
    annotated = _annotate_scan(raw_scan)
    ticker_context, ticker_names = (
        _build_ticker_context_from_annotated(annotated)
    )
    market_context = _build_market_context_from_annotated(annotated)
    result = candidates.merge(
        ticker_context,
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        market_context,
        on=["trading_day", "t"],
        how="left",
        validate="many_to_one",
    )
    market_names = tuple(
        name for name in MARKET_FEATURES
        if name in result.columns
    )
    return result, ticker_names, market_names


def usable_context_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        name
        for name in columns
        if name in frame
        and _numeric(frame[name]).notna().any()
    )


def crossfit_scores(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    prefix: str,
) -> pd.DataFrame:
    parts = []
    for fold_index, holdout in enumerate(CROSSFIT_DAYS):
        train_days = tuple(
            day for day in CROSSFIT_DAYS if day != holdout
        )
        train = frame.loc[
            frame.trading_day.astype(str).isin(train_days)
        ].copy()
        train = train.loc[
            _numeric(train.downstream_value_pct).notna()
        ].copy()
        test = frame.loc[
            frame.trading_day.astype(str).eq(holdout)
        ].copy()
        if test.empty:
            raise ValueError(
                f"Request280 missing held-out day {holdout}"
            )
        pair = fit_pair(
            train,
            "downstream_value_pct",
            columns,
            20261380 + fold_index * 10,
        )
        train_pred = pair.regressor.predict(
            xframe(train, columns)
        )
        threshold = threshold_from_training(
            train_pred,
            fraction=SELECTION_FRACTION,
        )
        test_x = xframe(test, columns)
        value = pair.regressor.predict(test_x)
        positive = pair.classifier.predict_proba(test_x)[:, 1]
        part = test.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
        ]].copy()
        part[f"{prefix}_predicted_value_pct"] = value
        part[f"{prefix}_positive_probability"] = positive
        part[f"{prefix}_threshold"] = threshold
        part[f"selected_{prefix}"] = value >= threshold
        part["fold_holdout_day"] = holdout
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def representation_diagnostics(
    labeled: pd.DataFrame,
    scores: pd.DataFrame,
    prefix: str,
) -> dict:
    frame = labeled.loc[:, [
        "trading_day",
        "ticker",
        "hot_t",
        "downstream_value_pct",
    ]].merge(
        scores,
        on=["trading_day", "ticker", "hot_t"],
        how="inner",
        validate="one_to_one",
    )
    target = _numeric(frame.downstream_value_pct)
    valid = target.notna()
    frame = frame.loc[valid].copy()
    target = _numeric(frame.downstream_value_pct)
    positive = target.gt(0).astype(int)
    probability = _numeric(
        frame[f"{prefix}_positive_probability"]
    )
    auc = (
        float(roc_auc_score(positive, probability))
        if positive.nunique() == 2
        else None
    )
    return {
        "rows": int(len(frame)),
        "value_spearman": _safe_spearman(
            target,
            frame[f"{prefix}_predicted_value_pct"],
        ),
        "positive_auc": auc,
    }


def context_coverage(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> dict:
    return {
        name: float(_numeric(frame[name]).notna().mean())
        for name in columns
    }


def evaluate(
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    raw_scan = pd.read_parquet(scan_path)
    raw_scan = raw_scan.loc[
        raw_scan.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    causal_scan = causal_execution_scan(raw_scan)
    first_hot = pd.read_parquet(first_hot_path).drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, causal_scan)
    used = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()

    episodes = build_pullback_episodes(used, causal_scan)
    policy = apply_exit_rule(
        episodes,
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name="request280_frozen_downstream",
    )
    labeled = attach_downstream_target(used, policy)
    enriched, ticker_names, market_names = attach_context(
        labeled,
        raw_scan,
    )

    base_columns = candidate_columns(enriched)
    ticker_extra = usable_context_columns(
        enriched,
        ticker_names,
    )
    market_extra = usable_context_columns(
        enriched,
        market_names,
    )
    ticker_columns = tuple(
        dict.fromkeys([*base_columns, *ticker_extra])
    )
    full_columns = tuple(
        dict.fromkeys([*ticker_columns, *market_extra])
    )

    base_scores = crossfit_scores(
        enriched,
        base_columns,
        "base",
    )
    ticker_scores = crossfit_scores(
        enriched,
        ticker_columns,
        "ticker",
    )
    market_scores = crossfit_scores(
        enriched,
        full_columns,
        "market",
    )

    base_metrics = selection_metrics(
        policy,
        base_scores,
        "selected_base",
    )
    ticker_metrics = selection_metrics(
        policy,
        ticker_scores,
        "selected_ticker",
    )
    market_metrics = selection_metrics(
        policy,
        market_scores,
        "selected_market",
    )
    base_diag = representation_diagnostics(
        enriched,
        base_scores,
        "base",
    )
    ticker_diag = representation_diagnostics(
        enriched,
        ticker_scores,
        "ticker",
    )
    market_diag = representation_diagnostics(
        enriched,
        market_scores,
        "market",
    )

    representations = {
        "base": {
            "metrics": base_metrics,
            "diagnostics": base_diag,
        },
        "ticker": {
            "metrics": ticker_metrics,
            "diagnostics": ticker_diag,
        },
        "market": {
            "metrics": market_metrics,
            "diagnostics": market_diag,
        },
    }
    enriched_names = ("ticker", "market")
    best_name = max(
        enriched_names,
        key=lambda name: (
            representations[name]["metrics"][
                "candidate_mean_pct"
            ]
            if representations[name]["metrics"][
                "candidate_mean_pct"
            ] is not None
            else -999.0
        ),
    )
    best = representations[best_name]
    best_mean = best["metrics"]["candidate_mean_pct"]
    base_mean = base_metrics["candidate_mean_pct"]
    gain_vs_base = (
        float(best_mean - base_mean)
        if best_mean is not None and base_mean is not None
        else None
    )
    checks = {
        "minimum_selected": (
            best["metrics"]["selected_candidates"]
            >= MIN_SELECTED
        ),
        "resolution_or_cash": (
            best["metrics"]["resolution_or_cash_rate"]
            >= MIN_RESOLUTION_OR_CASH
        ),
        "candidate_mean_positive": (
            best_mean is not None and best_mean > 0
        ),
        "day_balanced_mean_positive": (
            best["metrics"][
                "day_balanced_candidate_mean_pct"
            ] is not None
            and best["metrics"][
                "day_balanced_candidate_mean_pct"
            ] > 0
        ),
        "positive_days": (
            best["metrics"]["positive_candidate_mean_days"]
            >= MIN_POSITIVE_DAYS
        ),
        "value_spearman": (
            best["diagnostics"]["value_spearman"] is not None
            and best["diagnostics"]["value_spearman"]
            >= MIN_VALUE_SPEARMAN
        ),
        "gain_vs_base": (
            gain_vs_base is not None and gain_vs_base > 0
        ),
        "severe_trade_rate": (
            best["metrics"][
                "trade_severe_loss_rate_le_minus2"
            ] is not None
            and best["metrics"][
                "trade_severe_loss_rate_le_minus2"
            ] <= MAX_SEVERE_TRADE_RATE
        ),
    }
    gate = bool(all(checks.values()))

    rows = base_scores.merge(
        ticker_scores.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "ticker_predicted_value_pct",
            "ticker_positive_probability",
            "ticker_threshold",
            "selected_ticker",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="inner",
        validate="one_to_one",
    ).merge(
        market_scores.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "market_predicted_value_pct",
            "market_positive_probability",
            "market_threshold",
            "selected_market",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="inner",
        validate="one_to_one",
    )

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "selection_fraction": SELECTION_FRACTION,
        "fixed_downstream_policy": {
            "entry": "first causal 2% pullback",
            "hard_stop_loss_pct": FIXED_STOP_LOSS_PCT,
            "trailing_drawdown_pct": FIXED_TRAIL_PCT,
            "max_hold_minutes": FIXED_MAX_HOLD_MINUTES,
            "missing_exact_deadline": "unresolved",
        },
        "feature_counts": {
            "base": len(base_columns),
            "ticker_extra": len(ticker_extra),
            "ticker_total": len(ticker_columns),
            "market_extra": len(market_extra),
            "market_total": len(full_columns),
        },
        "ticker_context_coverage": context_coverage(
            enriched,
            ticker_extra,
        ),
        "market_context_coverage": context_coverage(
            enriched,
            market_extra,
        ),
        "representations": representations,
        "best_enriched_representation": best_name,
        "best_candidate_mean_gain_vs_base_pct": gain_vs_base,
        "checks": checks,
        "representation_economic_gate_pass": gate,
        "next_boundary": (
            "validate the chosen pre-HOT representation on May11-20 "
            "and then combine it with Request277 pullback risk context"
            if gate
            else (
                "the remaining bottleneck is not simple pre-HOT history "
                "or broad market regime; inspect candidate outcome modes "
                "and conditional risk before changing execution rules"
            )
        ),
        "timing_contract": (
            "ticker lags require exact completed-minute spacing; market "
            "context uses only the cross-section at the HOT timestamp; "
            "future timestamps are never consumed"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows.to_parquet(rows_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
