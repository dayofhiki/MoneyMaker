"""Request234 fast path: evaluate monotonic calibration from frozen Request233 rows.

Request233's row artifact already contains the frozen Request232 tradability
scores, event-time labels and first-HOT metadata. Reusing it changes only data
transport, not the Request234 hypothesis, split, mapping, gate or fresh policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .economic_full_hot_admission import _selection_metrics
from .extended_history_economic_opportunity import EXTENDED_CAL_DAYS
from .fresh_validated_attention_economic_opportunity import FRESH_EVAL_DAYS
from .monotonic_economic_calibration import (
    MIN_CAL_POSITIVE_MEAN_DAYS,
    MIN_FRESH_POSITIVE_MEAN_DAYS,
    MIN_MEAN_GAIN_VS_REQUEST232_PCT,
    MIN_SELECTED_POSITIVE_RATE,
    MIN_SELECTED_RATE,
    MAX_SELECTED_RATE,
    MIN_SELECTED_ROWS,
    _add_selection,
    _apply_map,
    _cross_fitted_calibration,
    _economic_gate,
    _fit_isotonic,
)
from .event_time_full_hot_tradability import MIN_TARGET_COVERAGE

REQUEST_ID = 234
SOURCE_RUN = 36251239308


def evaluate_rows(
    scored_hot_path: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    scored_hot = pd.read_parquet(scored_hot_path)
    required = {
        "trading_day",
        "persistent_tradability_pct",
        "predicted_persistent_tradability_pct",
        "tradability_selected",
    }
    missing = required - set(scored_hot.columns)
    if missing:
        raise ValueError(
            f"request 234 frozen rows missing columns: {sorted(missing)}"
        )

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
        "source_run": SOURCE_RUN,
        "source_rows": (
            "Request233 frozen full-HOT row artifact; no score or label "
            "recomputation"
        ),
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
                "all eight calibration days only after OOF gate passes"
            ),
            "economic_threshold_pct": 0.0,
            "also_requires_request232_probability_p75": True,
            "fresh_used_to_fit_or_choose_mapping": False,
        },
        "development_split": {
            "calibration_days": list(EXTENDED_CAL_DAYS),
            "fresh_eval_days": list(FRESH_EVAL_DAYS),
        },
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
            "promotion_min_target_coverage": MIN_TARGET_COVERAGE,
        },
        "next_boundary_if_pass": (
            "freeze economic HOT admission and train a new causal "
            "ENTER-vs-WAIT controller only inside admitted episodes; "
            "repair target coverage separately before promotion."
        ),
        "next_boundary_if_fail_calibration": (
            "the global rank score cannot be translated into a stable "
            "positive boundary across calibration days; move to "
            "market-conditioned economic calibration."
        ),
        "next_boundary_if_fail_fresh": (
            "global monotonic calibration is development-stable but not "
            "June-stable; test market-conditioned calibration while "
            "keeping Request232 ranking frozen."
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
        "--scored-hot",
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
    return evaluate_rows(
        args.scored_hot,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
