"""Request 175: positive-value calibration and prevalence drift audit.

No new dates. Reconstruct Request-174 state/model exactly and report how target
prevalence and calibrated probability distributions move from fit to
calibration to June evaluation. This request changes no policy or threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .market_regime_position_value import (
    attach_market_regime,
    build_market_regime,
)
from .market_regime_positive_value import (
    _feature_frame,
    train_classifier,
)
from .selected_hot_position_value_observability import (
    apply_excess_target,
    fit_minute_baselines,
)

REQUEST_ID = 175
HOLD_BUCKETS = (
    ("1-3", 1, 3),
    ("4-7", 4, 7),
    ("8-15", 8, 15),
    ("16-29", 16, 29),
)


def _safe_auc(y: pd.Series, p: np.ndarray) -> float | None:
    yy = pd.to_numeric(y, errors="coerce")
    valid = yy.notna() & np.isfinite(p)
    yy = yy.loc[valid].astype(int)
    pp = np.asarray(p)[valid.to_numpy()]
    if yy.nunique() < 2 or len(yy) < 20:
        return None
    return float(roc_auc_score(yy, pp))


def _phase_metrics(
    frame: pd.DataFrame,
    raw: np.ndarray,
    calibrated: np.ndarray,
) -> dict[str, object]:
    target = pd.to_numeric(
        frame["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    valid = target.notna() & np.isfinite(raw) & np.isfinite(calibrated)
    part = frame.loc[valid].copy()
    y = target.loc[valid].gt(0).astype(int)
    rawv = np.asarray(raw)[valid.to_numpy()]
    calv = np.asarray(calibrated)[valid.to_numpy()]

    buckets: dict[str, object] = {}
    held = pd.to_numeric(part["minutes_held"], errors="coerce")
    for label, low, high in HOLD_BUCKETS:
        mask = held.between(low, high, inclusive="both").to_numpy()
        if not mask.any():
            continue
        yy = y.to_numpy()[mask]
        pp = calv[mask]
        buckets[label] = {
            "rows": int(mask.sum()),
            "positive_rate": float(yy.mean()),
            "mean_probability": float(pp.mean()),
            "max_probability": float(pp.max()),
            "auc": (
                float(roc_auc_score(yy, pp))
                if len(np.unique(yy)) >= 2 and len(yy) >= 20
                else None
            ),
        }

    positive_values = target.loc[valid & target.gt(0)]
    negative_values = target.loc[valid & target.le(0)]
    return {
        "rows": int(valid.sum()),
        "positive_rate": float(y.mean()),
        "positive_target_mean_pct": (
            float(positive_values.mean()) if len(positive_values) else None
        ),
        "nonpositive_target_mean_pct": (
            float(negative_values.mean()) if len(negative_values) else None
        ),
        "raw_auc": _safe_auc(y, rawv),
        "calibrated_auc": _safe_auc(y, calv),
        "raw_probability_mean": float(rawv.mean()),
        "calibrated_probability_mean": float(calv.mean()),
        "calibrated_brier": float(brier_score_loss(y, calv)),
        "raw_gt_05_rate": float(np.mean(rawv > 0.5)),
        "calibrated_gt_05_rate": float(np.mean(calv > 0.5)),
        "raw_probability_quantiles": {
            str(q): float(np.quantile(rawv, q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "calibrated_probability_quantiles": {
            str(q): float(np.quantile(calv, q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "held_buckets": buckets,
    }


def _predict_pair(frame: pd.DataFrame, fitted) -> tuple[np.ndarray, np.ndarray]:
    raw = np.clip(
        fitted.model.predict_proba(_feature_frame(frame))[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    calibrated = fitted.platt.predict_proba(logits)[:, 1]
    return raw, calibrated


def _calibration_bin_edges(calibrated: np.ndarray) -> np.ndarray:
    edges = np.quantile(
        calibrated[np.isfinite(calibrated)],
        np.linspace(0, 1, 11),
    )
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_table(
    frame: pd.DataFrame,
    calibrated: np.ndarray,
    edges: np.ndarray,
) -> list[dict[str, object]]:
    target = pd.to_numeric(
        frame["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    valid = target.notna() & np.isfinite(calibrated)
    yy = target.loc[valid].gt(0).astype(int).to_numpy()
    pp = np.asarray(calibrated)[valid.to_numpy()]
    bins = np.digitize(pp, edges[1:-1], right=True)
    result: list[dict[str, object]] = []
    for index in range(10):
        mask = bins == index
        if not mask.any():
            result.append(
                {
                    "bin": index,
                    "rows": 0,
                    "mean_probability": None,
                    "positive_rate": None,
                }
            )
            continue
        result.append(
            {
                "bin": index,
                "rows": int(mask.sum()),
                "mean_probability": float(pp[mask].mean()),
                "positive_rate": float(yy[mask].mean()),
            }
        )
    return result


def run(
    fit_path: Path,
    calibration_path: Path,
    evaluation_path: Path,
    scan_path: Path,
    output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    evaluation = pd.read_parquet(evaluation_path)
    scan = pd.read_parquet(scan_path)

    wanted = set(
        fit["trading_day"].astype(str).unique()
    ) | set(calibration["trading_day"].astype(str).unique()) | set(
        evaluation["trading_day"].astype(str).unique()
    )
    scan = scan.loc[
        scan["trading_day"].astype(str).isin(wanted)
    ].copy()
    regime = build_market_regime(scan)
    fit = attach_market_regime(fit, regime)
    calibration = attach_market_regime(calibration, regime)
    evaluation = attach_market_regime(evaluation, regime)

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(calibration, baselines)
    evaluation = apply_excess_target(evaluation, baselines)

    fitted = train_classifier(fit, calibration)
    fit_raw, fit_cal = _predict_pair(fit, fitted)
    cal_raw, cal_cal = _predict_pair(calibration, fitted)
    eval_raw, eval_cal = _predict_pair(evaluation, fitted)

    edges = _calibration_bin_edges(cal_cal)
    by_day: dict[str, object] = {}
    for day in sorted(evaluation["trading_day"].astype(str).unique()):
        mask = evaluation["trading_day"].astype(str).eq(day).to_numpy()
        by_day[day] = _phase_metrics(
            evaluation.loc[mask].reset_index(drop=True),
            eval_raw[mask],
            eval_cal[mask],
        )

    phase = {
        "fit": _phase_metrics(fit, fit_raw, fit_cal),
        "calibration": _phase_metrics(
            calibration, cal_raw, cal_cal
        ),
        "evaluation": _phase_metrics(
            evaluation, eval_raw, eval_cal
        ),
    }
    cal_rate = phase["calibration"]["positive_rate"]
    eval_rate = phase["evaluation"]["positive_rate"]
    summary = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "policy_changed": False,
        "threshold_changed": False,
        "platt_coefficient": float(fitted.platt.coef_[0, 0]),
        "platt_intercept": float(fitted.platt.intercept_[0]),
        "phase": phase,
        "evaluation_by_day": by_day,
        "calibration_defined_probability_bins": {
            "calibration": _bin_table(
                calibration, cal_cal, edges
            ),
            "evaluation": _bin_table(
                evaluation, eval_cal, edges
            ),
        },
        "calibration_to_evaluation_positive_rate_shift_pp": float(
            (eval_rate - cal_rate) * 100.0
        ),
        "diagnosis_flags": {
            "base_rate_shift_abs_ge_5pp": bool(
                abs(eval_rate - cal_rate) >= 0.05
            ),
            "evaluation_ranking_above_random": bool(
                phase["evaluation"]["calibrated_auc"] is not None
                and phase["evaluation"]["calibrated_auc"] > 0.5
            ),
            "evaluation_semantic_boundary_sparse": bool(
                phase["evaluation"]["calibrated_gt_05_rate"] < 0.05
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(
        args.fit,
        args.calibration,
        args.evaluation,
        args.scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
