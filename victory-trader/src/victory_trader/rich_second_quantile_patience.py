"""Request 172: uncertainty-aware rich-second patience value.

No new dates. Reuse Request-171 rich-second phase artifacts. Fit a one-sided
20th-quantile model for excess remaining option value, then conformally correct
its lower bound on chronological calibration only. A state is a conservative
patience candidate only when the calibrated lower bound is > 0.

This is still an observability/patience-set diagnostic, not an executable
trajectory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .rich_second_position_value import RICH_SECOND_FEATURES
from .selected_hot_position_value_observability import (
    MODEL_FEATURES,
    OPTION_SEED,
    apply_excess_target,
    fit_minute_baselines,
)

REQUEST_ID = 172
LOWER_QUANTILE = 0.20
MIN_EVAL_BOUND_COVERAGE = 0.70
MIN_SELECTED_RATE = 0.05
MIN_SELECTED_POSITIVE_RATE = 0.55
MIN_POSITIVE_DAYS = 5
BOOTSTRAP_SEED = 20261072
BOOTSTRAP_SAMPLES = 10_000


@dataclass(frozen=True)
class QuantilePatienceModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    conformal_correction: float
    winsor_low: float
    winsor_high: float


def _columns() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys([*MODEL_FEATURES, *RICH_SECOND_FEATURES])
    )


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reindex(columns=_columns()).apply(
        pd.to_numeric,
        errors="coerce",
    )


def train_quantile_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> QuantilePatienceModel:
    target = pd.to_numeric(
        fit["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError("request 172 insufficient fit support")

    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=LOWER_QUANTILE,
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=OPTION_SEED + 19,
    )
    model.fit(
        _feature_frame(train),
        y.clip(lower=low, upper=high),
    )

    cal_target = pd.to_numeric(
        calibration["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    cal_valid = cal_target.notna()
    if int(cal_valid.sum()) < 500:
        raise ValueError("request 172 insufficient calibration support")
    raw = model.predict(_feature_frame(calibration.loc[cal_valid]))
    residual = cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    correction = float(
        np.quantile(residual, LOWER_QUANTILE, method="lower")
    )
    return QuantilePatienceModel(
        model=model,
        feature_columns=_columns(),
        conformal_correction=correction,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_lower_bound(
    frame: pd.DataFrame,
    fitted: QuantilePatienceModel,
) -> np.ndarray:
    return (
        fitted.model.predict(_feature_frame(frame))
        + fitted.conformal_correction
    )


def _bootstrap_day_means(
    selected: pd.DataFrame,
) -> dict[str, float | int | None]:
    values = pd.to_numeric(
        selected["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    temp = selected.loc[values.notna(), ["trading_day"]].copy()
    temp["value"] = values.loc[values.notna()].to_numpy()
    daily = (
        temp.groupby(temp["trading_day"].astype(str), sort=True)["value"]
        .mean()
        .to_numpy(dtype=float)
    )
    if len(daily) < 2:
        return {
            "days": int(len(daily)),
            "samples": BOOTSTRAP_SAMPLES,
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
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(daily)),
        "samples": BOOTSTRAP_SAMPLES,
        "ci_low_pct": float(lo),
        "ci_high_pct": float(hi),
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    evaluation_path: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    evaluation = pd.read_parquet(evaluation_path)

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(calibration, baselines)
    evaluation = apply_excess_target(evaluation, baselines)

    fitted = train_quantile_model(fit, calibration)
    scored = evaluation.copy()
    scored["conservative_excess_value_lower_pct"] = predict_lower_bound(
        scored, fitted
    )

    target = pd.to_numeric(
        scored["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    lower = pd.to_numeric(
        scored["conservative_excess_value_lower_pct"],
        errors="coerce",
    )
    valid = target.notna() & lower.notna()
    eval_bound_coverage = float(
        target.loc[valid].ge(lower.loc[valid]).mean()
    )
    selected = scored.loc[
        valid & lower.gt(0)
    ].copy()
    selected_target = pd.to_numeric(
        selected["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    selected_rate = (
        float(len(selected) / int(valid.sum()))
        if int(valid.sum())
        else None
    )
    selected_mean = (
        float(selected_target.mean()) if len(selected) else None
    )
    selected_positive_rate = (
        float(selected_target.gt(0).mean()) if len(selected) else None
    )
    daily = (
        pd.DataFrame(
            {
                "day": selected["trading_day"].astype(str),
                "value": selected_target,
            }
        )
        .dropna()
        .groupby("day", sort=True)["value"]
        .agg(["size", "mean"])
    )
    day_balanced = (
        float(daily["mean"].mean()) if len(daily) else None
    )
    bootstrap = _bootstrap_day_means(selected)

    by_day: dict[str, object] = {}
    positive_days = 0
    for day in sorted(scored["trading_day"].astype(str).unique()):
        part = scored.loc[
            scored["trading_day"].astype(str).eq(day)
        ].copy()
        ptarget = pd.to_numeric(
            part["excess_remaining_option_value_pct"],
            errors="coerce",
        )
        plower = pd.to_numeric(
            part["conservative_excess_value_lower_pct"],
            errors="coerce",
        )
        pvalid = ptarget.notna() & plower.notna()
        psel = part.loc[pvalid & plower.gt(0)].copy()
        vals = pd.to_numeric(
            psel["excess_remaining_option_value_pct"],
            errors="coerce",
        )
        mean = float(vals.mean()) if len(psel) else None
        pos = float(vals.gt(0).mean()) if len(psel) else None
        good = bool(mean is not None and mean > 0)
        positive_days += int(good)
        by_day[day] = {
            "rows": int(len(part)),
            "evaluable_rows": int(pvalid.sum()),
            "lower_bound_coverage": (
                float(ptarget.loc[pvalid].ge(plower.loc[pvalid]).mean())
                if int(pvalid.sum())
                else None
            ),
            "selected_rows": int(len(psel)),
            "selected_rate": (
                float(len(psel) / int(pvalid.sum()))
                if int(pvalid.sum())
                else None
            ),
            "selected_realized_excess_mean_pct": mean,
            "selected_positive_rate": pos,
        }

    gate = bool(
        eval_bound_coverage >= MIN_EVAL_BOUND_COVERAGE
        and selected_rate is not None
        and selected_rate >= MIN_SELECTED_RATE
        and selected_mean is not None
        and selected_mean > 0
        and selected_positive_rate is not None
        and selected_positive_rate >= MIN_SELECTED_POSITIVE_RATE
        and day_balanced is not None
        and day_balanced > 0
        and bootstrap["ci_low_pct"] is not None
        and float(bootstrap["ci_low_pct"]) > 0
        and positive_days >= MIN_POSITIVE_DAYS
    )

    summary = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "rich_second_features_changed": False,
        "lower_quantile": LOWER_QUANTILE,
        "conformal_correction_pct": fitted.conformal_correction,
        "evaluation_lower_bound_coverage": eval_bound_coverage,
        "selected_rows": int(len(selected)),
        "selected_rate": selected_rate,
        "selected_realized_excess_mean_pct": selected_mean,
        "selected_positive_rate": selected_positive_rate,
        "selected_day_balanced_excess_mean_pct": day_balanced,
        "selected_day_bootstrap": bootstrap,
        "positive_selected_excess_days": int(positive_days),
        "by_day": by_day,
        "model_diagnostics": {
            "feature_count": len(fitted.feature_columns),
            "winsor_low_pct": fitted.winsor_low,
            "winsor_high_pct": fitted.winsor_high,
        },
        "frozen_gate": {
            "min_eval_bound_coverage": MIN_EVAL_BOUND_COVERAGE,
            "min_selected_rate": MIN_SELECTED_RATE,
            "min_selected_positive_rate": MIN_SELECTED_POSITIVE_RATE,
            "min_positive_days": MIN_POSITIVE_DAYS,
            "bootstrap_ci_low_must_be_positive": True,
        },
        "patience_set_bridge_pass": gate,
        "interpretation": (
            "Development-only conservative patience-set diagnostic. "
            "No future value enters runtime features; no executable "
            "trajectory is claimed."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scored.to_parquet(
        scored_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.evaluation,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
