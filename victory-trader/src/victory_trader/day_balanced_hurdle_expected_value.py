"""Request 177: equal-trading-day weighted hurdle expected value.

No new dates. Keep Request-176 features, target and EV>0 semantic boundary fixed.
Change only training/calibration weighting so each trading day contributes equal
total weight within each head. This targets day-level robustness without using
June outcomes to tune a threshold or feature subset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .market_regime_hurdle_expected_value import (
    MIN_CONTEXT_COVERAGE,
    MIN_EV_SPEARMAN,
    MIN_GOOD_DAYS,
    MIN_SELECTED_RATE,
    MIN_SELECTED_REALIZED_MEAN_PCT,
    MagnitudeHead,
    _evaluate,
    _predict_magnitude,
)
from .market_regime_position_value import (
    MARKET_REGIME_FEATURES,
    attach_market_regime,
    build_market_regime,
)
from .market_regime_positive_value import (
    PositiveValueClassifier,
    _feature_frame,
)
from .selected_hot_position_value_observability import (
    apply_excess_target,
    fit_minute_baselines,
)

REQUEST_ID = 177
RANDOM_SEED = 20261077


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    days = frame["trading_day"].astype(str)
    counts = days.value_counts()
    weights = days.map(lambda day: 1.0 / float(counts.loc[day])).to_numpy(dtype=float)
    mean = float(np.mean(weights))
    return weights / mean if mean > 0 else weights


def train_day_balanced_classifier(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> PositiveValueClassifier:
    fit_target = pd.to_numeric(
        fit["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    cal_target = pd.to_numeric(
        calibration["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    fit_valid = fit_target.notna()
    cal_valid = cal_target.notna()
    train = fit.loc[fit_valid].copy()
    cal = calibration.loc[cal_valid].copy()
    y_fit = fit_target.loc[fit_valid].gt(0).astype(int)
    y_cal = cal_target.loc[cal_valid].gt(0).astype(int)

    if (
        len(train) < 1000
        or len(cal) < 500
        or y_fit.nunique() < 2
        or y_cal.nunique() < 2
    ):
        raise ValueError("request 177 insufficient classifier support")

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=RANDOM_SEED,
    )
    model.fit(
        _feature_frame(train),
        y_fit,
        sample_weight=_day_weights(train),
    )

    raw = np.clip(
        model.predict_proba(_feature_frame(cal))[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    platt.fit(
        logits,
        y_cal,
        sample_weight=_day_weights(cal),
    )
    return PositiveValueClassifier(
        model=model,
        platt=platt,
        feature_columns=tuple(_feature_frame(train).columns),
    )


def _day_balanced_offset(
    frame: pd.DataFrame,
    target: pd.Series,
    prediction: np.ndarray,
) -> float:
    temp = pd.DataFrame(
        {
            "day": frame["trading_day"].astype(str).to_numpy(),
            "residual": target.to_numpy(dtype=float) - np.asarray(prediction, dtype=float),
        }
    )
    return float(temp.groupby("day", sort=True)["residual"].mean().mean())


def train_day_balanced_magnitude(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    positive_class: bool,
    seed: int,
) -> MagnitudeHead:
    fit_target = pd.to_numeric(
        fit["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    cal_target = pd.to_numeric(
        calibration["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    if positive_class:
        fit_mask = fit_target.gt(0)
        cal_mask = cal_target.gt(0)
    else:
        fit_mask = fit_target.le(0)
        cal_mask = cal_target.le(0)

    train = fit.loc[fit_mask].copy()
    y = fit_target.loc[fit_mask].astype(float)
    cal = calibration.loc[cal_mask].copy()
    y_cal = cal_target.loc[cal_mask].astype(float)
    if len(train) < 500 or len(cal) < 200:
        raise ValueError("request 177 insufficient magnitude support")

    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _feature_frame(train),
        y.clip(lower=low, upper=high),
        sample_weight=_day_weights(train),
    )
    raw = model.predict(_feature_frame(cal))
    offset = _day_balanced_offset(cal, y_cal, raw)
    return MagnitudeHead(
        model=model,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
        positive_class=positive_class,
    )


def _predict_probability(
    frame: pd.DataFrame,
    fitted: PositiveValueClassifier,
) -> np.ndarray:
    raw = np.clip(
        fitted.model.predict_proba(_feature_frame(frame))[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    return fitted.platt.predict_proba(logits)[:, 1]


def _score(
    frame: pd.DataFrame,
    classifier: PositiveValueClassifier,
    positive_head: MagnitudeHead,
    nonpositive_head: MagnitudeHead,
) -> pd.DataFrame:
    scored = frame.copy()
    p = _predict_probability(scored, classifier)
    win = _predict_magnitude(scored, positive_head)
    loss = _predict_magnitude(scored, nonpositive_head)
    scored["positive_value_probability"] = p
    scored["predicted_positive_magnitude_pct"] = win
    scored["predicted_nonpositive_magnitude_pct"] = loss
    scored["hurdle_expected_value_pct"] = p * win + (1.0 - p) * loss
    return scored


def run(
    fit_path: Path,
    calibration_path: Path,
    evaluation_path: Path,
    scan_path: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    evaluation = pd.read_parquet(evaluation_path)
    scan = pd.read_parquet(scan_path)

    wanted = set(fit["trading_day"].astype(str)) | set(
        calibration["trading_day"].astype(str)
    ) | set(evaluation["trading_day"].astype(str))
    scan = scan.loc[scan["trading_day"].astype(str).isin(wanted)].copy()
    regime = build_market_regime(scan)
    fit = attach_market_regime(fit, regime)
    calibration = attach_market_regime(calibration, regime)
    evaluation = attach_market_regime(evaluation, regime)

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(calibration, baselines)
    evaluation = apply_excess_target(evaluation, baselines)

    classifier = train_day_balanced_classifier(fit, calibration)
    positive_head = train_day_balanced_magnitude(
        fit,
        calibration,
        positive_class=True,
        seed=RANDOM_SEED + 1,
    )
    nonpositive_head = train_day_balanced_magnitude(
        fit,
        calibration,
        positive_class=False,
        seed=RANDOM_SEED + 2,
    )

    scored = _score(
        evaluation,
        classifier,
        positive_head,
        nonpositive_head,
    )
    metrics = _evaluate(scored)
    context_coverage = float(
        evaluation.loc[:, MARKET_REGIME_FEATURES].notna().any(axis=1).mean()
    )

    gate = bool(
        context_coverage >= MIN_CONTEXT_COVERAGE
        and metrics["ev_spearman"] is not None
        and float(metrics["ev_spearman"]) >= MIN_EV_SPEARMAN
        and metrics["selected_rate"] is not None
        and float(metrics["selected_rate"]) >= MIN_SELECTED_RATE
        and metrics["selected_realized_excess_mean_pct"] is not None
        and float(metrics["selected_realized_excess_mean_pct"])
        >= MIN_SELECTED_REALIZED_MEAN_PCT
        and metrics["selected_day_balanced_excess_mean_pct"] is not None
        and float(metrics["selected_day_balanced_excess_mean_pct"]) > 0
        and metrics["selected_day_bootstrap"]["ci_low_pct"] is not None
        and float(metrics["selected_day_bootstrap"]["ci_low_pct"]) > 0
        and metrics["good_days"] >= MIN_GOOD_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "change_from_request176": (
            "equal total trading-day sample weight in classifier, Platt "
            "calibration, magnitude regression and magnitude calibration offset"
        ),
        "policy_boundary": "hurdle_expected_value_pct > 0",
        "market_context_coverage": context_coverage,
        "metrics": metrics,
        "head_diagnostics": {
            "platt_coefficient": float(classifier.platt.coef_[0, 0]),
            "platt_intercept": float(classifier.platt.intercept_[0]),
            "positive_offset": positive_head.offset,
            "nonpositive_offset": nonpositive_head.offset,
        },
        "frozen_gate": {
            "min_context_coverage": MIN_CONTEXT_COVERAGE,
            "min_ev_spearman": MIN_EV_SPEARMAN,
            "min_selected_rate": MIN_SELECTED_RATE,
            "min_selected_realized_mean_pct": MIN_SELECTED_REALIZED_MEAN_PCT,
            "min_good_days": MIN_GOOD_DAYS,
            "bootstrap_ci_low_must_be_positive": True,
        },
        "promotion_gate_pass": gate,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scored.to_parquet(scored_output_path, index=False, compression="zstd")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-output", type=Path, required=True)
    args = parser.parse_args()
    return run(
        args.fit,
        args.calibration,
        args.evaluation,
        args.scan,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
