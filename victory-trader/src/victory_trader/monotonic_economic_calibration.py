"""Request 234: monotonic economic calibration of Request232 tradability.

Request232 learned a strong relative tradability ranking. Request233 showed that
its raw zero boundary is not an economic zero: probability-p75 plus raw
predicted value > 0 still selected negative-mean episodes.

Request234 keeps the Request232 learner frozen and calibrates only the scalar
predicted persistent-tradability value. Isotonic regression is fitted with
equal day weight. Calibration robustness is checked with leave-one-day-out
predictions across the eight already-opened calibration days. Only if that
out-of-day economic gate passes is a final map fitted on all calibration days
and the frozen June fresh block evaluated.

Admission remains simple and causal:
- Request232 probability must pass its frozen calibration p75 gate; and
- monotonic calibrated expected persistent tradability must be > 0.

No fresh outcome is used to choose the mapping or threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .economic_full_hot_admission import _selection_metrics
from .event_time_full_hot_tradability import (
    MIN_TARGET_COVERAGE,
    attach_full_hot_tradability,
    score_tradability,
    train_tradability,
)
from .extended_history_economic_opportunity import EXTENDED_CAL_DAYS
from .fresh_validated_attention_economic_opportunity import FRESH_EVAL_DAYS
from .hierarchical_attention_runtime import add_stage2_scores, fit_stage2
from .hot_economic_opportunity import _first_hot_feature_rows

REQUEST_ID = 234

MIN_SELECTED_RATE = 0.03
MAX_SELECTED_RATE = 0.25
MIN_SELECTED_ROWS = 40
MIN_SELECTED_POSITIVE_RATE = 0.35
MIN_CAL_POSITIVE_MEAN_DAYS = 5
MIN_FRESH_POSITIVE_MEAN_DAYS = 4
MIN_MEAN_GAIN_VS_REQUEST232_PCT = 0.25
MIN_MAP_ROWS = 800


def _fit_isotonic(
    frame: pd.DataFrame,
) -> IsotonicRegression:
    work = frame.copy()
    score = pd.to_numeric(
        work["predicted_persistent_tradability_pct"],
        errors="coerce",
    )
    target = pd.to_numeric(
        work["persistent_tradability_pct"],
        errors="coerce",
    )
    valid = score.notna() & target.notna()
    work = work.loc[valid].copy()
    score = pd.to_numeric(
        work["predicted_persistent_tradability_pct"],
        errors="coerce",
    )
    target = pd.to_numeric(
        work["persistent_tradability_pct"],
        errors="coerce",
    )
    if len(work) < MIN_MAP_ROWS:
        raise ValueError(
            f"request 234 isotonic map needs >= {MIN_MAP_ROWS} rows, "
            f"got {len(work)}"
        )
    if score.nunique() < 2:
        raise ValueError("request 234 isotonic score is constant")

    day = work["trading_day"].astype(str)
    counts = day.value_counts()
    sample_weight = day.map(
        lambda value: 1.0 / float(counts.loc[value])
    ).to_numpy(dtype=float)

    model = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
    )
    model.fit(
        score.to_numpy(dtype=float),
        target.to_numpy(dtype=float),
        sample_weight=sample_weight,
    )
    return model


def _apply_map(
    frame: pd.DataFrame,
    model: IsotonicRegression,
    output_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    score = pd.to_numeric(
        result["predicted_persistent_tradability_pct"],
        errors="coerce",
    )
    result[output_column] = np.nan
    valid = score.notna()
    if valid.any():
        result.loc[valid, output_column] = model.predict(
            score.loc[valid].to_numpy(dtype=float)
        )
    return result


def _cross_fitted_calibration(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["oof_calibrated_expected_value_pct"] = np.nan

    for held_out_day in EXTENDED_CAL_DAYS:
        fit_days = [
            day
            for day in EXTENDED_CAL_DAYS
            if day != held_out_day
        ]
        train = result.loc[
            result["trading_day"].astype(str).isin(fit_days)
        ].copy()
        held = result.loc[
            result["trading_day"].astype(str).eq(held_out_day)
        ].copy()
        model = _fit_isotonic(train)
        score = pd.to_numeric(
            held["predicted_persistent_tradability_pct"],
            errors="coerce",
        )
        valid = score.notna()
        if valid.any():
            result.loc[
                held.index[valid],
                "oof_calibrated_expected_value_pct",
            ] = model.predict(
                score.loc[valid].to_numpy(dtype=float)
            )
    return result


def _add_selection(
    frame: pd.DataFrame,
    value_column: str,
    selection_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    probability_selected = result[
        "tradability_selected"
    ].fillna(False).astype(bool)
    value = pd.to_numeric(
        result[value_column],
        errors="coerce",
    )
    result[selection_column] = (
        probability_selected
        & value.gt(0.0)
    )
    return result


def _economic_gate(
    candidate: dict[str, object],
    baseline: dict[str, object],
    required_positive_days: int,
) -> tuple[bool, dict[str, object]]:
    mean = candidate["selected_mean_pct"]
    day_mean = candidate["day_balanced_selected_mean_pct"]
    rate = candidate["selected_rate"]
    positive_rate = candidate["selected_positive_rate"]
    rows = int(candidate["selected_rows"])
    baseline_mean = baseline["selected_mean_pct"]
    gain = (
        float(mean) - float(baseline_mean)
        if mean is not None
        and baseline_mean is not None
        else None
    )
    checks = {
        "min_selected_rows": rows >= MIN_SELECTED_ROWS,
        "selected_rate_in_range": (
            rate is not None
            and MIN_SELECTED_RATE
            <= float(rate)
            <= MAX_SELECTED_RATE
        ),
        "selected_mean_positive": (
            mean is not None and float(mean) > 0
        ),
        "day_balanced_mean_positive": (
            day_mean is not None and float(day_mean) > 0
        ),
        "selected_positive_rate": (
            positive_rate is not None
            and float(positive_rate)
            >= MIN_SELECTED_POSITIVE_RATE
        ),
        "positive_mean_days": (
            int(candidate["positive_selected_mean_days"])
            >= required_positive_days
        ),
        "mean_gain_vs_request232": (
            gain is not None
            and gain >= MIN_MEAN_GAIN_VS_REQUEST232_PCT
        ),
    }
    return bool(all(checks.values())), {
        "checks": checks,
        "selected_mean_gain_vs_request232_pct": gain,
    }


def evaluate(
    stage2_candidates_path: Path,
    opportunity_candidates_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    stage2_candidates = pd.read_parquet(stage2_candidates_path)
    opportunity_candidates = pd.read_parquet(
        opportunity_candidates_path
    )
    opportunity_scan = pd.read_parquet(
        opportunity_scan_path
    )

    minute_model, second_model = fit_stage2(stage2_candidates)
    scored = add_stage2_scores(
        opportunity_candidates,
        minute_model,
        second_model,
    )
    first_hot, runtime_audit = _first_hot_feature_rows(
        scored,
        opportunity_scan,
    )
    first_hot = attach_full_hot_tradability(
        first_hot,
        opportunity_scan,
    )
    fitted = train_tradability(first_hot)
    scored_hot = score_tradability(first_hot, fitted)

    scored_hot = _cross_fitted_calibration(scored_hot)
    scored_hot = _add_selection(
        scored_hot,
        "oof_calibrated_expected_value_pct",
        "request234_oof_selected",
    )

    calibration_baseline = _selection_metrics(
        scored_hot,
        EXTENDED_CAL_DAYS,
        "tradability_selected",
    )
    calibration_candidate = _selection_metrics(
        scored_hot,
        EXTENDED_CAL_DAYS,
        "request234_oof_selected",
    )
    calibration_gate, calibration_gate_detail = _economic_gate(
        calibration_candidate,
        calibration_baseline,
        MIN_CAL_POSITIVE_MEAN_DAYS,
    )

    fresh_baseline = None
    fresh_candidate = None
    fresh_gate = False
    fresh_gate_detail = None

    if calibration_gate:
        calibration = scored_hot.loc[
            scored_hot["trading_day"]
            .astype(str)
            .isin(EXTENDED_CAL_DAYS)
        ].copy()
        final_map = _fit_isotonic(calibration)
        scored_hot = _apply_map(
            scored_hot,
            final_map,
            "final_calibrated_expected_value_pct",
        )
        scored_hot = _add_selection(
            scored_hot,
            "final_calibrated_expected_value_pct",
            "request234_fresh_selected",
        )
        fresh_baseline = _selection_metrics(
            scored_hot,
            FRESH_EVAL_DAYS,
            "tradability_selected",
        )
        fresh_candidate = _selection_metrics(
            scored_hot,
            FRESH_EVAL_DAYS,
            "request234_fresh_selected",
        )
        fresh_gate, fresh_gate_detail = _economic_gate(
            fresh_candidate,
            fresh_baseline,
            MIN_FRESH_POSITIVE_MEAN_DAYS,
        )

    fresh_coverage = (
        fresh_candidate["coverage"]
        if fresh_candidate is not None
        else None
    )
    coverage_promotion_ready = bool(
        fresh_coverage is not None
        and float(fresh_coverage) >= MIN_TARGET_COVERAGE
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "architecture": (
            "frozen Request232 tradability head plus day-balanced "
            "monotonic economic calibration"
        ),
        "mapping": {
            "input": "predicted_persistent_tradability_pct",
            "target": "persistent_tradability_pct",
            "model": "IsotonicRegression(increasing=True)",
            "day_balanced_sample_weight": True,
            "calibration_check": (
                "leave-one-calibration-day-out predictions"
            ),
            "final_map_fit": (
                "all eight calibration days, only after OOF gate passes"
            ),
            "economic_threshold_pct": 0.0,
            "also_requires_request232_probability_p75": True,
            "fresh_used_to_fit_or_choose_mapping": False,
        },
        "development_split": {
            "calibration_days": list(EXTENDED_CAL_DAYS),
            "fresh_eval_days": list(FRESH_EVAL_DAYS),
        },
        "runtime_audit": runtime_audit,
        "support": {
            "first_hot_rows": int(len(scored_hot)),
            "calibration_labeled_rows": int(
                scored_hot.loc[
                    scored_hot["trading_day"]
                    .astype(str)
                    .isin(EXTENDED_CAL_DAYS),
                    "persistent_tradability_pct",
                ].notna().sum()
            ),
        },
        "calibration_oof": {
            "request232_baseline": calibration_baseline,
            "monotonic_candidate": calibration_candidate,
            "gate": calibration_gate_detail,
            "economic_gate_pass": calibration_gate,
        },
        "fresh_evaluated": bool(calibration_gate),
        "fresh": (
            {
                "request232_baseline": fresh_baseline,
                "monotonic_candidate": fresh_candidate,
                "gate": fresh_gate_detail,
                "economic_gate_pass": fresh_gate,
            }
            if calibration_gate
            else None
        ),
        "coverage_promotion_ready": coverage_promotion_ready,
        "development_gate_pass": bool(
            calibration_gate and fresh_gate
        ),
        "promotion_gate_pass": False,
        "frozen_gate": {
            "min_selected_rows": MIN_SELECTED_ROWS,
            "min_selected_rate": MIN_SELECTED_RATE,
            "max_selected_rate": MAX_SELECTED_RATE,
            "min_selected_positive_rate": (
                MIN_SELECTED_POSITIVE_RATE
            ),
            "min_calibration_positive_mean_days": (
                MIN_CAL_POSITIVE_MEAN_DAYS
            ),
            "min_fresh_positive_mean_days": (
                MIN_FRESH_POSITIVE_MEAN_DAYS
            ),
            "min_mean_gain_vs_request232_pct": (
                MIN_MEAN_GAIN_VS_REQUEST232_PCT
            ),
            "promotion_min_target_coverage": (
                MIN_TARGET_COVERAGE
            ),
        },
        "next_boundary_if_pass": (
            "freeze economic admission and train a new causal "
            "ENTER-vs-WAIT controller only inside admitted episodes; "
            "repair target coverage separately before promotion."
        ),
        "next_boundary_if_fail_calibration": (
            "the frozen rank score does not support a stable positive "
            "economic boundary across calibration days; move to "
            "market-conditioned economic calibration rather than "
            "adding downstream timing logic."
        ),
        "next_boundary_if_fail_fresh": (
            "the monotonic economic boundary is development-stable but "
            "does not transfer to June; test market-conditioned "
            "calibration while keeping Request232 ranking frozen."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scored_hot.to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2-candidates",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--opportunity-candidates",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--opportunity-scan",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.stage2_candidates,
        args.opportunity_candidates,
        args.opportunity_scan,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
