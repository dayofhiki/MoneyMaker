"""Request 191: causal future cost-cover state observability.

Tests whether additional causal POSITION-path information can identify states
from which at least one future exact executable BASE exit through the inherited
30-minute research cap still covers modeled trading costs.

This is an observability diagnostic only. It does not execute a HOLD/EXIT
policy and therefore does not claim account-level profitability.
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
from sklearn.metrics import average_precision_score, roc_auc_score

from .decomposed_cost_aware_entry_surplus import FRESH_DAYS
from .market_regime_position_value import (
    MARKET_REGIME_FEATURES,
    attach_market_regime,
    build_market_regime,
)
from .rich_second_position_value import RICH_SECOND_FEATURES
from .selected_hot_position_value_observability import MODEL_FEATURES

REQUEST_ID = 191
CLASSIFIER_SEED = 20261094
REGRESSOR_SEED = 20261095
PROBABILITY_GATE_QUANTILE = 0.75
EARLY_MAX_MINUTE = 10

MIN_AUC = 0.60
MIN_VALUE_SPEARMAN = 0.15
MIN_EARLY_AUC = 0.60
MIN_EARLY_VALUE_SPEARMAN = 0.15
MIN_POSITIVE_SPEARMAN_DAYS = 4
MIN_SELECTED_PREVALENCE_UPLIFT = 0.10
MIN_SELECTED_VALUE_UPLIFT_PCT = 0.50
MIN_POSITIVE_SELECTED_DAYS = 4

FEATURES = tuple(
    dict.fromkeys(
        [
            *MODEL_FEATURES,
            *RICH_SECOND_FEATURES,
            *MARKET_REGIME_FEATURES,
        ]
    )
)


@dataclass(frozen=True)
class FutureCostCoverModels:
    classifier: HistGradientBoostingClassifier
    regressor: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    probability_gate: float
    regression_offset: float
    winsor_low: float
    winsor_high: float


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    days = frame["trading_day"].astype(str)
    counts = days.map(days.value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(dtype=float)
    mean = float(np.mean(weights))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("request 191 invalid equal-day weights")
    return weights / mean


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    future = pd.to_numeric(
        result["best_future_base_return_pct"],
        errors="coerce",
    )
    result["future_cost_coverable"] = future.gt(0).where(
        future.notna()
    )
    return result


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def _safe_auc(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    y = y.loc[valid].astype(int)
    if len(y) < 20 or y.nunique() < 2:
        return None
    return float(
        roc_auc_score(
            y,
            s.loc[valid].astype(float),
        )
    )


def _safe_ap(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    y = y.loc[valid].astype(int)
    if len(y) < 20 or int(y.sum()) == 0:
        return None
    return float(
        average_precision_score(
            y,
            s.loc[valid].astype(float),
        )
    )


def _safe_spearman(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    value = y.loc[valid].corr(
        s.loc[valid],
        method="spearman",
    )
    return None if pd.isna(value) else float(value)


def train_models(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> FutureCostCoverModels:
    fit = add_targets(fit)
    calibration = add_targets(calibration)

    fit_value = pd.to_numeric(
        fit["best_future_base_return_pct"],
        errors="coerce",
    )
    fit_valid = fit_value.notna()
    train = fit.loc[fit_valid].copy()
    y_value = fit_value.loc[fit_valid].astype(float)
    y_class = y_value.gt(0).astype(int)
    if len(train) < 1000 or y_class.nunique() < 2:
        raise ValueError("request 191 insufficient fit support")

    columns = tuple(
        column
        for column in FEATURES
        if column in train.columns
        and pd.to_numeric(
            train[column],
            errors="coerce",
        ).notna().any()
    )
    if not columns:
        raise ValueError("request 191 has no usable causal features")

    weights = _day_weights(train)
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=CLASSIFIER_SEED,
    )
    classifier.fit(
        _feature_frame(train, columns),
        y_class,
        sample_weight=weights,
    )

    low, high = np.quantile(
        y_value.to_numpy(dtype=float),
        [0.005, 0.995],
    )
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=REGRESSOR_SEED,
    )
    regressor.fit(
        _feature_frame(train, columns),
        y_value.clip(lower=low, upper=high),
        sample_weight=weights,
    )

    cal_value = pd.to_numeric(
        calibration["best_future_base_return_pct"],
        errors="coerce",
    )
    cal_valid = cal_value.notna()
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 500:
        raise ValueError("request 191 insufficient calibration support")

    cal_x = _feature_frame(cal, columns)
    cal_probability = classifier.predict_proba(cal_x)[:, 1]
    probability_gate = float(
        np.quantile(
            cal_probability,
            PROBABILITY_GATE_QUANTILE,
        )
    )

    raw_value = regressor.predict(cal_x)
    residual = (
        cal_value.loc[cal_valid].to_numpy(dtype=float)
        - raw_value
    )
    regression_offset = float(
        np.average(
            residual,
            weights=_day_weights(cal),
        )
    )

    return FutureCostCoverModels(
        classifier=classifier,
        regressor=regressor,
        columns=columns,
        probability_gate=probability_gate,
        regression_offset=regression_offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def score_states(
    frame: pd.DataFrame,
    models: FutureCostCoverModels,
) -> pd.DataFrame:
    scored = add_targets(frame)
    x = _feature_frame(scored, models.columns)
    scored["future_cost_cover_probability"] = (
        models.classifier.predict_proba(x)[:, 1]
    )
    scored["predicted_best_future_base_return_pct"] = (
        models.regressor.predict(x) + models.regression_offset
    )
    scored["selected"] = (
        pd.to_numeric(
            scored["future_cost_cover_probability"],
            errors="coerce",
        )
        .ge(models.probability_gate)
    )
    return scored


def _metrics(frame: pd.DataFrame) -> dict[str, object]:
    value = pd.to_numeric(
        frame["best_future_base_return_pct"],
        errors="coerce",
    )
    probability = pd.to_numeric(
        frame["future_cost_cover_probability"],
        errors="coerce",
    )
    prediction = pd.to_numeric(
        frame["predicted_best_future_base_return_pct"],
        errors="coerce",
    )
    valid = value.notna() & probability.notna() & prediction.notna()
    part = frame.loc[valid].copy()
    value = value.loc[valid]
    probability = probability.loc[valid]
    prediction = prediction.loc[valid]
    y = value.gt(0).astype(int)

    selected_mask = part["selected"].fillna(False).astype(bool)
    selected = part.loc[selected_mask].copy()
    selected_value = pd.to_numeric(
        selected["best_future_base_return_pct"],
        errors="coerce",
    )

    prevalence = float(y.mean()) if len(y) else None
    selected_positive_rate = (
        float(selected_value.gt(0).mean())
        if len(selected_value)
        else None
    )
    all_mean = float(value.mean()) if len(value) else None
    selected_mean = (
        float(selected_value.mean())
        if len(selected_value)
        else None
    )
    prevalence_uplift = (
        float(selected_positive_rate - prevalence)
        if selected_positive_rate is not None
        and prevalence is not None
        else None
    )
    value_uplift = (
        float(selected_mean - all_mean)
        if selected_mean is not None
        and all_mean is not None
        else None
    )

    return {
        "rows": int(len(frame)),
        "evaluable_rows": int(valid.sum()),
        "future_cost_cover_prevalence": prevalence,
        "classifier_auc": _safe_auc(y, probability),
        "classifier_average_precision": _safe_ap(
            y,
            probability,
        ),
        "value_spearman": _safe_spearman(
            value,
            prediction,
        ),
        "selected_rows": int(len(selected)),
        "selected_rate": (
            float(len(selected) / len(part))
            if len(part)
            else None
        ),
        "selected_future_cost_cover_rate": selected_positive_rate,
        "selected_minus_all_prevalence_uplift": (
            prevalence_uplift
        ),
        "selected_future_best_mean_pct": selected_mean,
        "all_state_future_best_mean_pct": all_mean,
        "selected_minus_all_value_uplift_pct": value_uplift,
    }


def _by_day(
    scored: pd.DataFrame,
) -> tuple[dict[str, object], int, int]:
    result: dict[str, object] = {}
    positive_spearman_days = 0
    positive_selected_days = 0

    for day in FRESH_DAYS:
        part = scored.loc[
            scored["trading_day"].astype(str).eq(day)
        ].copy()
        metrics = _metrics(part)
        spearman = metrics["value_spearman"]
        selected_mean = metrics["selected_future_best_mean_pct"]
        if spearman is not None and float(spearman) > 0:
            positive_spearman_days += 1
        if selected_mean is not None and float(selected_mean) > 0:
            positive_selected_days += 1
        result[day] = metrics

    return (
        result,
        positive_spearman_days,
        positive_selected_days,
    )


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    history_scan = pd.read_parquet(history_scan_path)

    position_paths = sorted(
        fresh_dir.glob("*-positions.parquet")
    )
    scan_paths = sorted(
        fresh_dir.glob("*-scan.parquet")
    )
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError(
            "request 191 requires complete request 178 fresh shards"
        )

    fresh = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    found_days = sorted(
        fresh["trading_day"].astype(str).unique()
    )
    if found_days != list(FRESH_DAYS):
        raise ValueError(
            f"request 191 expected {FRESH_DAYS}, found {found_days}"
        )

    historical_days = set(
        fit["trading_day"].astype(str)
    ) | set(
        calibration["trading_day"].astype(str)
    )
    hist_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(
            historical_days
        )
    ].copy()

    hist_regime = build_market_regime(hist_scan)
    fresh_regime = build_market_regime(fresh_scan)
    fit = attach_market_regime(fit, hist_regime)
    calibration = attach_market_regime(
        calibration,
        hist_regime,
    )
    fresh = attach_market_regime(
        fresh,
        fresh_regime,
    )

    models = train_models(fit, calibration)
    scored = score_states(fresh, models)

    overall = _metrics(scored)
    held = pd.to_numeric(
        scored["minutes_held"],
        errors="coerce",
    )
    early = _metrics(
        scored.loc[
            held.notna()
            & held.ge(1)
            & held.le(EARLY_MAX_MINUTE)
        ].copy()
    )
    by_day, positive_spearman_days, positive_selected_days = (
        _by_day(scored)
    )

    context_coverage = float(
        scored.loc[:, MARKET_REGIME_FEATURES]
        .notna()
        .any(axis=1)
        .mean()
    )

    gate = bool(
        overall["classifier_auc"] is not None
        and float(overall["classifier_auc"]) >= MIN_AUC
        and overall["value_spearman"] is not None
        and float(overall["value_spearman"])
        >= MIN_VALUE_SPEARMAN
        and early["classifier_auc"] is not None
        and float(early["classifier_auc"]) >= MIN_EARLY_AUC
        and early["value_spearman"] is not None
        and float(early["value_spearman"])
        >= MIN_EARLY_VALUE_SPEARMAN
        and positive_spearman_days
        >= MIN_POSITIVE_SPEARMAN_DAYS
        and overall[
            "selected_minus_all_prevalence_uplift"
        ]
        is not None
        and float(
            overall[
                "selected_minus_all_prevalence_uplift"
            ]
        )
        >= MIN_SELECTED_PREVALENCE_UPLIFT
        and overall[
            "selected_minus_all_value_uplift_pct"
        ]
        is not None
        and float(
            overall[
                "selected_minus_all_value_uplift_pct"
            ]
        )
        >= MIN_SELECTED_VALUE_UPLIFT_PCT
        and positive_selected_days
        >= MIN_POSITIVE_SELECTED_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "policy_executed": False,
        "account_simulation_applicable": False,
        "research_cap_minutes": 30,
        "target_semantics": (
            "best exact executable BASE exit strictly after the "
            "current POSITION state and before the 30m research cap"
        ),
        "feature_count": len(models.columns),
        "market_context_coverage": context_coverage,
        "calibration_probability_p75_gate": (
            models.probability_gate
        ),
        "overall": overall,
        "early_minutes_1_to_10": early,
        "positive_value_spearman_days": int(
            positive_spearman_days
        ),
        "positive_selected_future_best_days": int(
            positive_selected_days
        ),
        "by_day": by_day,
        "model_diagnostics": {
            "regression_offset_pct": (
                models.regression_offset
            ),
            "winsor_low_pct": models.winsor_low,
            "winsor_high_pct": models.winsor_high,
        },
        "frozen_gate": {
            "min_auc": MIN_AUC,
            "min_value_spearman": MIN_VALUE_SPEARMAN,
            "min_early_auc": MIN_EARLY_AUC,
            "min_early_value_spearman": (
                MIN_EARLY_VALUE_SPEARMAN
            ),
            "min_positive_value_spearman_days": (
                MIN_POSITIVE_SPEARMAN_DAYS
            ),
            "min_selected_prevalence_uplift": (
                MIN_SELECTED_PREVALENCE_UPLIFT
            ),
            "min_selected_value_uplift_pct": (
                MIN_SELECTED_VALUE_UPLIFT_PCT
            ),
            "min_positive_selected_days": (
                MIN_POSITIVE_SELECTED_DAYS
            ),
        },
        "development_bridge_pass": gate,
        "next_step_if_pass": (
            "compose with request189 capture timing into an "
            "honest recurrent executable HOLD/EXIT controller"
        ),
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_output_path.parent.mkdir(
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
    scored.to_parquet(
        scored_output_path,
        index=False,
        compression="zstd",
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
    parser.add_argument("--fit", type=Path, required=True)
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
        "--fresh-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--scored-output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
