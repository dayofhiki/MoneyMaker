"""Request279: downstream-aligned pre-HOT candidate value diagnostic.

The experiment keeps the pre-HOT feature boundary and model family fixed while
changing the candidate target. The historical baseline learns the old
fixed-first-WATCH value. The aligned model learns the realized economic result
of one frozen causal downstream policy:

HOT -> wait for the first causal 2% pullback -> fixed stop/trailing/max-hold.

No-pullback candidates are cash with zero return. Entered candidates whose
exact deadline cannot be resolved remain unknown. May5-8 are evaluated only
through leave-one-trading-day-out predictions. No May11-20 or June dates are
opened by this request.
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
from .full_hot_fixed_policy_value import (
    attach_fixed_value,
    usable_columns,
    xframe,
)
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

REQUEST_ID = 279
SELECTION_FRACTION = 0.20
MIN_SELECTED = 100
MIN_RESOLUTION_OR_CASH = 0.95
MIN_POSITIVE_DAYS = 3
MIN_GAIN_VS_OLD_TARGET_PCT = 0.10
MAX_SEVERE_TRADE_RATE = 0.25

FORBIDDEN_FEATURES = {
    "fixed_first_watch_value_pct",
    "fixed_entry_elapsed_minutes",
    "downstream_value_pct",
    "economic_return_pct",
    "trade_return_pct",
    "resolved",
    "entered",
    "exit_t",
    "exit_reason",
}


@dataclass(frozen=True)
class Pair:
    regressor: HistGradientBoostingRegressor
    classifier: HistGradientBoostingClassifier


def attach_downstream_target(
    first_hot: pd.DataFrame,
    policy: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the frozen policy outcome to the HOT-time candidate row."""
    frame = first_hot.copy()
    frame["hot_t"] = pd.to_numeric(
        frame["t"],
        errors="raise",
    ).astype("int64")
    labels = policy.loc[:, [
        "trading_day",
        "ticker",
        "hot_t",
        "entered",
        "resolved",
        "economic_return_pct",
        "trade_return_pct",
        "exit_t",
        "exit_reason",
    ]].copy()
    labels["ticker"] = labels.ticker.astype(str).str.upper()
    frame["ticker"] = frame.ticker.astype(str).str.upper()
    merged = frame.merge(
        labels,
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="one_to_one",
    )
    if merged.resolved.isna().any():
        raise ValueError("Request279 candidate/policy merge lost rows")
    merged["downstream_value_pct"] = np.nan
    resolved = merged.resolved.astype(bool)
    merged.loc[
        resolved,
        "downstream_value_pct",
    ] = pd.to_numeric(
        merged.loc[resolved, "economic_return_pct"],
        errors="coerce",
    )
    return merged


def candidate_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns = usable_columns(frame)
    overlap = FORBIDDEN_FEATURES.intersection(columns)
    if overlap:
        raise ValueError(
            "Request279 target leakage in pre-HOT features: "
            + ",".join(sorted(overlap))
        )
    if not columns:
        raise ValueError("Request279 found no pre-HOT candidate features")
    return columns


def _weights(frame: pd.DataFrame) -> np.ndarray:
    days = frame.trading_day.astype(str)
    counts = days.map(days.value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(float)
    return weights / float(np.mean(weights))


def fit_pair(
    train: pd.DataFrame,
    target_column: str,
    columns: tuple[str, ...],
    seed: int,
) -> Pair:
    target = pd.to_numeric(
        train[target_column],
        errors="coerce",
    )
    fit = train.loc[target.notna()].copy()
    y = pd.to_numeric(
        fit[target_column],
        errors="coerce",
    ).to_numpy(float)
    if len(fit) < 400:
        raise ValueError(
            f"Request279 {target_column} support only {len(fit)}"
        )
    positive = (y > 0).astype(int)
    if np.unique(positive).size != 2:
        raise ValueError(
            f"Request279 {target_column} lacks positive classes"
        )
    low, high = np.quantile(y, [0.005, 0.995])
    weights = _weights(fit)
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed + 1,
    )
    regressor.fit(
        xframe(fit, columns),
        np.clip(y, low, high),
        sample_weight=weights,
    )
    classifier.fit(
        xframe(fit, columns),
        positive,
        sample_weight=weights,
    )
    return Pair(regressor, classifier)


def predict_pair(
    pair: Pair,
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    prefix: str,
) -> pd.DataFrame:
    result = frame.copy()
    x = xframe(result, columns)
    result[f"{prefix}_predicted_value_pct"] = pair.regressor.predict(x)
    result[f"{prefix}_positive_probability"] = (
        pair.classifier.predict_proba(x)[:, 1]
    )
    return result


def threshold_from_training(
    predicted_values: np.ndarray | pd.Series,
    fraction: float = SELECTION_FRACTION,
) -> float:
    values = np.asarray(predicted_values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 100:
        raise ValueError("Request279 threshold support too small")
    return float(np.quantile(values, 1.0 - float(fraction)))


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
        common = (
            pd.to_numeric(
                train.fixed_first_watch_value_pct,
                errors="coerce",
            ).notna()
            & pd.to_numeric(
                train.downstream_value_pct,
                errors="coerce",
            ).notna()
        )
        train = train.loc[common].copy()
        test = frame.loc[
            frame.trading_day.astype(str).eq(holdout)
        ].copy()
        if test.empty:
            raise ValueError(f"Request279 missing held-out day {holdout}")

        old_pair = fit_pair(
            train,
            "fixed_first_watch_value_pct",
            columns,
            20261379 + fold_index * 10,
        )
        aligned_pair = fit_pair(
            train,
            "downstream_value_pct",
            columns,
            20261380 + fold_index * 10,
        )

        train_old = old_pair.regressor.predict(
            xframe(train, columns)
        )
        train_aligned = aligned_pair.regressor.predict(
            xframe(train, columns)
        )
        old_threshold = threshold_from_training(train_old)
        aligned_threshold = threshold_from_training(train_aligned)

        scored = predict_pair(
            old_pair,
            test,
            columns,
            "old_target",
        )
        scored = predict_pair(
            aligned_pair,
            scored,
            columns,
            "aligned",
        )
        scored["old_target_threshold"] = old_threshold
        scored["aligned_threshold"] = aligned_threshold
        scored["selected_old_target"] = (
            scored.old_target_predicted_value_pct
            >= old_threshold
        )
        scored["selected_aligned"] = (
            scored.aligned_predicted_value_pct
            >= aligned_threshold
        )
        scored["fold_holdout_day"] = holdout
        scored["fold_train_days"] = ",".join(train_days)
        parts.append(scored)
    result = pd.concat(parts, ignore_index=True)
    if sorted(result.trading_day.astype(str).unique()) != sorted(
        CROSSFIT_DAYS
    ):
        raise ValueError("Request279 crossfit output lost a day")
    return result


def _safe_spearman(
    actual: pd.Series,
    predicted: pd.Series,
) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna()
    if int(valid.sum()) < 20:
        return None
    value = a.loc[valid].corr(p.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def prediction_diagnostics(predictions: pd.DataFrame) -> dict:
    target = pd.to_numeric(
        predictions.downstream_value_pct,
        errors="coerce",
    )
    valid = target.notna()
    rows = predictions.loc[valid].copy()
    y = pd.to_numeric(
        rows.downstream_value_pct,
        errors="coerce",
    )
    positive = y.gt(0).astype(int)

    def auc(column: str) -> float | None:
        if positive.nunique() != 2:
            return None
        return float(
            roc_auc_score(
                positive,
                pd.to_numeric(rows[column], errors="coerce"),
            )
        )

    return {
        "resolved_or_cash_rows": int(len(rows)),
        "coverage": (
            float(valid.mean()) if len(predictions) else 0.0
        ),
        "population_mean_pct": (
            float(y.mean()) if len(y) else None
        ),
        "population_positive_rate": (
            float(positive.mean()) if len(positive) else None
        ),
        "old_target_value_spearman_to_downstream": _safe_spearman(
            y,
            rows.old_target_predicted_value_pct,
        ),
        "aligned_value_spearman_to_downstream": _safe_spearman(
            y,
            rows.aligned_predicted_value_pct,
        ),
        "old_target_positive_auc_to_downstream": auc(
            "old_target_positive_probability"
        ),
        "aligned_positive_auc_to_downstream": auc(
            "aligned_positive_probability"
        ),
    }


def selection_metrics(
    policy: pd.DataFrame,
    predictions: pd.DataFrame,
    selection_column: str | None,
) -> dict:
    decisions = policy.copy()
    if selection_column is None:
        chosen = predictions.loc[:, [
            "trading_day", "ticker", "hot_t",
        ]].copy()
        chosen["selected"] = True
    else:
        chosen = predictions.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            selection_column,
        ]].rename(columns={selection_column: "selected"})
    decisions = decisions.merge(
        chosen,
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="one_to_one",
    )
    decisions["selected"] = (
        decisions.selected.fillna(False).astype(bool)
    )
    decisions["decision_resolved"] = (
        ~decisions.selected
        | decisions.resolved.astype(bool)
    )
    decisions["decision_return_pct"] = 0.0
    active_resolved = (
        decisions.selected
        & decisions.resolved.astype(bool)
    )
    decisions.loc[
        active_resolved,
        "decision_return_pct",
    ] = pd.to_numeric(
        decisions.loc[
            active_resolved,
            "economic_return_pct",
        ],
        errors="coerce",
    )
    decisions.loc[
        ~decisions.decision_resolved,
        "decision_return_pct",
    ] = np.nan

    resolved = decisions.loc[
        decisions.decision_resolved
    ].copy()
    selected = decisions.loc[
        decisions.selected
    ].copy()
    trade = selected.loc[
        selected.entered.astype(bool)
        & selected.resolved.astype(bool)
    ].copy()
    daily = resolved.groupby(
        resolved.trading_day.astype(str),
        sort=True,
    ).decision_return_pct.mean()

    trade_return = pd.to_numeric(
        trade.trade_return_pct,
        errors="coerce",
    )
    return {
        "candidate_episodes": int(len(decisions)),
        "selected_candidates": int(decisions.selected.sum()),
        "selection_rate": (
            float(decisions.selected.mean())
            if len(decisions)
            else 0.0
        ),
        "selected_entries": int(
            (
                decisions.selected
                & decisions.entered.astype(bool)
            ).sum()
        ),
        "unresolved_selected_entries": int(
            (
                decisions.selected
                & decisions.entered.astype(bool)
                & ~decisions.resolved.astype(bool)
            ).sum()
        ),
        "resolution_or_cash_rate": (
            float(decisions.decision_resolved.mean())
            if len(decisions)
            else 0.0
        ),
        "candidate_mean_pct": (
            float(
                pd.to_numeric(
                    resolved.decision_return_pct,
                    errors="coerce",
                ).mean()
            )
            if len(resolved)
            else None
        ),
        "day_balanced_candidate_mean_pct": (
            float(daily.mean()) if len(daily) else None
        ),
        "positive_candidate_mean_days": int((daily > 0).sum()),
        "trade_count": int(len(trade)),
        "trade_mean_pct": (
            float(trade_return.mean())
            if len(trade_return)
            else None
        ),
        "trade_positive_rate": (
            float(trade_return.gt(0).mean())
            if len(trade_return)
            else None
        ),
        "trade_severe_loss_rate_le_minus2": (
            float(trade_return.le(-2).mean())
            if len(trade_return)
            else None
        ),
        "by_day": {
            str(day): {
                "candidates": int(len(part)),
                "selected": int(part.selected.sum()),
                "candidate_mean_pct": float(
                    pd.to_numeric(
                        part.loc[
                            part.decision_resolved,
                            "decision_return_pct",
                        ],
                        errors="coerce",
                    ).mean()
                ),
            }
            for day, part in decisions.groupby(
                decisions.trading_day.astype(str),
                sort=True,
            )
        },
    }


def evaluate(
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    raw_scan = pd.read_parquet(scan_path)
    scan = causal_execution_scan(raw_scan)
    first_hot = pd.read_parquet(first_hot_path).drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, scan)

    used = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    episodes = build_pullback_episodes(used, scan)
    policy = apply_exit_rule(
        episodes,
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name="request279_frozen_downstream",
    )
    labeled = attach_downstream_target(used, policy)
    columns = candidate_columns(labeled)
    predictions = crossfit_predictions(labeled, columns)

    raw_metrics = selection_metrics(
        policy,
        predictions,
        None,
    )
    old_metrics = selection_metrics(
        policy,
        predictions,
        "selected_old_target",
    )
    aligned_metrics = selection_metrics(
        policy,
        predictions,
        "selected_aligned",
    )
    diagnostics = prediction_diagnostics(predictions)

    gain_vs_old = None
    if (
        aligned_metrics["candidate_mean_pct"] is not None
        and old_metrics["candidate_mean_pct"] is not None
    ):
        gain_vs_old = float(
            aligned_metrics["candidate_mean_pct"]
            - old_metrics["candidate_mean_pct"]
        )

    checks = {
        "minimum_selected_candidates": (
            aligned_metrics["selected_candidates"] >= MIN_SELECTED
        ),
        "resolution_or_cash": (
            aligned_metrics["resolution_or_cash_rate"]
            >= MIN_RESOLUTION_OR_CASH
        ),
        "candidate_mean_positive": (
            aligned_metrics["candidate_mean_pct"] is not None
            and aligned_metrics["candidate_mean_pct"] > 0
        ),
        "day_balanced_mean_positive": (
            aligned_metrics[
                "day_balanced_candidate_mean_pct"
            ] is not None
            and aligned_metrics[
                "day_balanced_candidate_mean_pct"
            ] > 0
        ),
        "positive_days": (
            aligned_metrics["positive_candidate_mean_days"]
            >= MIN_POSITIVE_DAYS
        ),
        "gain_vs_old_target": (
            gain_vs_old is not None
            and gain_vs_old >= MIN_GAIN_VS_OLD_TARGET_PCT
        ),
        "severe_trade_rate": (
            aligned_metrics[
                "trade_severe_loss_rate_le_minus2"
            ] is not None
            and aligned_metrics[
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
        "feature_boundary": (
            "same pre-HOT ENTRY_FEATURES for both old-target and "
            "downstream-aligned models"
        ),
        "fixed_downstream_policy": {
            "entry": "first causal 2% pullback",
            "hard_stop_loss_pct": FIXED_STOP_LOSS_PCT,
            "trailing_drawdown_pct": FIXED_TRAIL_PCT,
            "max_hold_minutes": FIXED_MAX_HOLD_MINUTES,
            "no_pullback": "cash_zero",
            "missing_exact_deadline": "unresolved",
        },
        "support": {
            "candidate_rows": int(len(labeled)),
            "downstream_resolved_or_cash_rows": int(
                pd.to_numeric(
                    labeled.downstream_value_pct,
                    errors="coerce",
                ).notna().sum()
            ),
            "entered_rows": int(
                labeled.entered.astype(bool).sum()
            ),
            "unresolved_entered_rows": int(
                (
                    labeled.entered.astype(bool)
                    & ~labeled.resolved.astype(bool)
                ).sum()
            ),
            "feature_count": len(columns),
        },
        "prediction_diagnostics": diagnostics,
        "admit_all_baseline": raw_metrics,
        "old_target_top20": old_metrics,
        "downstream_aligned_top20": aligned_metrics,
        "candidate_mean_gain_vs_old_target_pct": gain_vs_old,
        "checks": checks,
        "representation_economic_gate_pass": gate,
        "next_boundary": (
            "combine downstream-aligned pre-HOT selection with "
            "Request277 strictly lagged pullback risk context"
            if gate
            else (
                "add causal pre-HOT temporal/regime context before "
                "changing entry or exit rules"
            )
        ),
        "interpretation": (
            "The old and aligned models share the same HOT-time feature "
            "boundary, model family, crossfit days, and 20% train-derived "
            "selection fraction. Their supervised target is the isolated "
            "difference. Thresholds are derived only from each fold's "
            "training days, never from the held-out day's future candidates."
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
