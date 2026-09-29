"""Request281: decompose pre-HOT candidate value into conditional outcomes.

Request280 showed useful signal in exact pre-HOT ticker history, but a single
candidate-value target is dominated by cash-zero episodes. This experiment
keeps the frozen downstream trading policy and ticker-history representation,
then separates:

1. probability the fixed 2% pullback entry occurs;
2. conditional trade return given an entered and resolved trade;
3. conditional positive-trade probability;
4. conditional severe-loss probability.

The deployable candidate score is the causal expected value
P(pullback) * E[trade return | pullback]. No future-best label, post-HOT
feature, or held-out-day calibration is used.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
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
from .prehot_context_ablation import (
    build_ticker_context,
    usable_context_columns,
)
from .pullback_trailing_exit import (
    apply_exit_rule,
    build_pullback_episodes,
)

REQUEST_ID = 281
MIN_TRADE_TRAIN_ROWS = 120
MIN_SELECTED = 100
MIN_POSITIVE_DAYS = 3
MAX_SEVERE_TRADE_RATE = 0.25


@dataclass(frozen=True)
class ConditionalHeads:
    pullback: HistGradientBoostingClassifier
    trade_value: HistGradientBoostingRegressor
    trade_positive: HistGradientBoostingClassifier
    trade_severe: HistGradientBoostingClassifier


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    days = frame.trading_day.astype(str)
    counts = days.map(days.value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(float)
    return weights / float(np.mean(weights))


def attach_ticker_history(
    candidates: pd.DataFrame,
    raw_scan: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    context, names = build_ticker_context(raw_scan)
    enriched = candidates.merge(
        context,
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
    )
    extra = usable_context_columns(enriched, names)
    return enriched, extra


def fit_conditional_heads(
    train: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
) -> ConditionalHeads:
    x_all = xframe(train, columns)
    entered = train.entered.astype(bool).to_numpy(int)
    if np.unique(entered).size != 2:
        raise ValueError("Request281 pullback target lacks both classes")

    pullback = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )
    pullback.fit(
        x_all,
        entered,
        sample_weight=_day_weights(train),
    )

    trade = train.loc[
        train.entered.astype(bool)
        & train.resolved.astype(bool)
        & _numeric(train.trade_return_pct).notna()
    ].copy()
    if len(trade) < MIN_TRADE_TRAIN_ROWS:
        raise ValueError(
            f"Request281 only {len(trade)} resolved trade rows"
        )
    y = _numeric(trade.trade_return_pct).to_numpy(float)
    positive = (y > 0).astype(int)
    severe = (y <= -2.0).astype(int)
    if np.unique(positive).size != 2:
        raise ValueError("Request281 trade-positive target lacks classes")
    if np.unique(severe).size != 2:
        raise ValueError("Request281 severe target lacks classes")

    low, high = np.quantile(y, [0.01, 0.99])
    trade_value = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed + 1,
    )
    trade_positive = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed + 2,
    )
    trade_severe = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed + 3,
    )
    x_trade = xframe(trade, columns)
    weights = _day_weights(trade)
    trade_value.fit(
        x_trade,
        np.clip(y, low, high),
        sample_weight=weights,
    )
    trade_positive.fit(
        x_trade,
        positive,
        sample_weight=weights,
    )
    trade_severe.fit(
        x_trade,
        severe,
        sample_weight=weights,
    )
    return ConditionalHeads(
        pullback,
        trade_value,
        trade_positive,
        trade_severe,
    )


def score_conditional(
    heads: ConditionalHeads,
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    result = frame.loc[:, [
        "trading_day",
        "ticker",
        "hot_t",
    ]].copy()
    x = xframe(frame, columns)
    pullback = heads.pullback.predict_proba(x)[:, 1]
    trade_value = heads.trade_value.predict(x)
    positive = heads.trade_positive.predict_proba(x)[:, 1]
    severe = heads.trade_severe.predict_proba(x)[:, 1]

    result["pullback_probability"] = pullback
    result["conditional_trade_value_pct"] = trade_value
    result["conditional_positive_probability"] = positive
    result["conditional_severe_probability"] = severe
    result["expected_candidate_value_pct"] = pullback * trade_value
    result["positive_probability_mass"] = pullback * positive
    result["severe_probability_mass"] = pullback * severe
    return result


def crossfit_predictions(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    parts = []
    for fold_index, holdout in enumerate(CROSSFIT_DAYS):
        train_days = tuple(
            day for day in CROSSFIT_DAYS if day != holdout
        )
        train = frame.loc[
            frame.trading_day.astype(str).isin(train_days)
        ].copy()
        test = frame.loc[
            frame.trading_day.astype(str).eq(holdout)
        ].copy()
        if test.empty:
            raise ValueError(
                f"Request281 missing held-out day {holdout}"
            )

        direct_train = train.loc[
            _numeric(train.downstream_value_pct).notna()
        ].copy()
        direct = fit_pair(
            direct_train,
            "downstream_value_pct",
            columns,
            20261410 + fold_index * 10,
        )
        conditional = fit_conditional_heads(
            train,
            columns,
            20261420 + fold_index * 10,
        )

        train_x = xframe(train, columns)
        direct_train_score = direct.regressor.predict(train_x)
        conditional_train_score = score_conditional(
            conditional,
            train,
            columns,
        ).expected_candidate_value_pct.to_numpy(float)

        direct_threshold = threshold_from_training(
            direct_train_score,
            fraction=SELECTION_FRACTION,
        )
        conditional_threshold = threshold_from_training(
            conditional_train_score,
            fraction=SELECTION_FRACTION,
        )

        test_x = xframe(test, columns)
        out = score_conditional(
            conditional,
            test,
            columns,
        )
        out["direct_predicted_value_pct"] = (
            direct.regressor.predict(test_x)
        )
        out["direct_positive_probability"] = (
            direct.classifier.predict_proba(test_x)[:, 1]
        )
        out["direct_threshold"] = direct_threshold
        out["conditional_threshold"] = conditional_threshold
        out["selected_direct"] = (
            out.direct_predicted_value_pct >= direct_threshold
        )
        out["selected_conditional"] = (
            out.expected_candidate_value_pct
            >= conditional_threshold
        )
        out["fold_holdout_day"] = holdout
        parts.append(out)

    return pd.concat(parts, ignore_index=True)


def prediction_diagnostics(
    labeled: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict:
    frame = labeled.loc[:, [
        "trading_day",
        "ticker",
        "hot_t",
        "entered",
        "resolved",
        "downstream_value_pct",
        "trade_return_pct",
    ]].merge(
        predictions,
        on=["trading_day", "ticker", "hot_t"],
        how="inner",
        validate="one_to_one",
    )

    downstream = _numeric(frame.downstream_value_pct)
    resolved = downstream.notna()
    econ = frame.loc[resolved].copy()
    y = _numeric(econ.downstream_value_pct)
    positive = y.gt(0).astype(int)

    direct_auc = (
        float(
            roc_auc_score(
                positive,
                _numeric(econ.direct_positive_probability),
            )
        )
        if positive.nunique() == 2
        else None
    )
    conditional_auc = (
        float(
            roc_auc_score(
                positive,
                _numeric(econ.positive_probability_mass),
            )
        )
        if positive.nunique() == 2
        else None
    )

    entered = frame.entered.astype(bool)
    pullback_auc = (
        float(
            roc_auc_score(
                entered.astype(int),
                _numeric(frame.pullback_probability),
            )
        )
        if entered.nunique() == 2
        else None
    )

    trades = frame.loc[
        entered
        & frame.resolved.astype(bool)
        & _numeric(frame.trade_return_pct).notna()
    ].copy()
    trade_y = _numeric(trades.trade_return_pct)
    trade_positive = trade_y.gt(0).astype(int)
    trade_severe = trade_y.le(-2.0).astype(int)

    trade_positive_auc = (
        float(
            roc_auc_score(
                trade_positive,
                _numeric(
                    trades.conditional_positive_probability
                ),
            )
        )
        if trade_positive.nunique() == 2
        else None
    )
    trade_severe_auc = (
        float(
            roc_auc_score(
                trade_severe,
                _numeric(
                    trades.conditional_severe_probability
                ),
            )
        )
        if trade_severe.nunique() == 2
        else None
    )

    return {
        "resolved_or_cash_rows": int(resolved.sum()),
        "entered_rows": int(entered.sum()),
        "resolved_trade_rows": int(len(trades)),
        "direct_value_spearman": _safe_spearman(
            y,
            econ.direct_predicted_value_pct,
        ),
        "conditional_expected_value_spearman": _safe_spearman(
            y,
            econ.expected_candidate_value_pct,
        ),
        "direct_positive_auc": direct_auc,
        "conditional_positive_mass_auc": conditional_auc,
        "pullback_occurrence_auc": pullback_auc,
        "conditional_trade_value_spearman": _safe_spearman(
            trade_y,
            trades.conditional_trade_value_pct,
        ),
        "conditional_trade_positive_auc": trade_positive_auc,
        "conditional_trade_severe_auc": trade_severe_auc,
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
    first_hot = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    first_hot = attach_fixed_value(first_hot, causal_scan)

    episodes = build_pullback_episodes(first_hot, causal_scan)
    policy = apply_exit_rule(
        episodes,
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name="request281_frozen_downstream",
    )
    labeled = attach_downstream_target(first_hot, policy)
    enriched, ticker_extra = attach_ticker_history(
        labeled,
        raw_scan,
    )
    base_columns = candidate_columns(enriched)
    columns = tuple(
        dict.fromkeys([*base_columns, *ticker_extra])
    )

    predictions = crossfit_predictions(
        enriched,
        columns,
    )
    direct_metrics = selection_metrics(
        policy,
        predictions,
        "selected_direct",
    )
    conditional_metrics = selection_metrics(
        policy,
        predictions,
        "selected_conditional",
    )
    diagnostics = prediction_diagnostics(
        enriched,
        predictions,
    )

    direct_mean = direct_metrics["candidate_mean_pct"]
    conditional_mean = conditional_metrics["candidate_mean_pct"]
    gain = (
        float(conditional_mean - direct_mean)
        if direct_mean is not None
        and conditional_mean is not None
        else None
    )
    checks = {
        "minimum_selected": (
            conditional_metrics["selected_candidates"]
            >= MIN_SELECTED
        ),
        "resolution_or_cash": (
            conditional_metrics["resolution_or_cash_rate"]
            >= MIN_RESOLUTION_OR_CASH
        ),
        "candidate_mean_positive": (
            conditional_mean is not None
            and conditional_mean > 0
        ),
        "day_balanced_mean_positive": (
            conditional_metrics[
                "day_balanced_candidate_mean_pct"
            ] is not None
            and conditional_metrics[
                "day_balanced_candidate_mean_pct"
            ] > 0
        ),
        "positive_days": (
            conditional_metrics["positive_candidate_mean_days"]
            >= MIN_POSITIVE_DAYS
        ),
        "gain_vs_direct": (
            gain is not None and gain > 0
        ),
        "severe_trade_rate": (
            conditional_metrics[
                "trade_severe_loss_rate_le_minus2"
            ] is not None
            and conditional_metrics[
                "trade_severe_loss_rate_le_minus2"
            ] <= MAX_SEVERE_TRADE_RATE
        ),
    }
    gate = bool(all(checks.values()))

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "selection_fraction": SELECTION_FRACTION,
        "feature_count": len(columns),
        "ticker_history_feature_count": len(ticker_extra),
        "fixed_downstream_policy": {
            "entry": "first causal 2% pullback",
            "hard_stop_loss_pct": FIXED_STOP_LOSS_PCT,
            "trailing_drawdown_pct": FIXED_TRAIL_PCT,
            "max_hold_minutes": FIXED_MAX_HOLD_MINUTES,
            "no_pullback": "cash_zero",
            "missing_exact_deadline": "unresolved",
        },
        "support": {
            "candidate_rows": int(len(enriched)),
            "entered_rows": int(
                enriched.entered.astype(bool).sum()
            ),
            "resolved_trade_rows": int(
                (
                    enriched.entered.astype(bool)
                    & enriched.resolved.astype(bool)
                    & _numeric(
                        enriched.trade_return_pct
                    ).notna()
                ).sum()
            ),
            "cash_no_pullback_rows": int(
                (~enriched.entered.astype(bool)).sum()
            ),
            "unresolved_entered_rows": int(
                (
                    enriched.entered.astype(bool)
                    & ~enriched.resolved.astype(bool)
                ).sum()
            ),
        },
        "diagnostics": diagnostics,
        "direct_downstream_top20": direct_metrics,
        "conditional_expected_value_top20": conditional_metrics,
        "candidate_mean_gain_vs_direct_pct": gain,
        "checks": checks,
        "conditional_value_gate_pass": gate,
        "next_boundary": (
            "combine this pre-HOT selector with Request277's causal "
            "pullback-state severe-loss veto, then validate unchanged "
            "on May11-20"
            if gate
            else (
                "use the conditional severe-loss head as a preregistered "
                "risk veto before expected-value ranking; do not alter "
                "entry/exit mechanics"
            )
        ),
        "interpretation": (
            "Cash-zero episodes train the pullback-occurrence head rather "
            "than dominating the trade-return regressor. Conditional trade "
            "heads train only on entered resolved episodes. Unresolved "
            "entries are never relabeled as zero."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    predictions.to_parquet(rows_output, index=False)
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
