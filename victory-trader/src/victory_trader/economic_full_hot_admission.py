"""Request 233: calibration-gated economic full-HOT admission.

Request232 established that a separate first-HOT tradability head contains
substantial ranking information, but the frozen classifier p75 gate still
selected a negative-mean population. Request233 does not add a new learner.
It asks whether the already-fitted Request232 head has enough absolute economic
calibration to reject "less bad" episodes.

Candidate admission requires BOTH:
1) positive-class probability >= the Request232 calibration p75 gate; and
2) calibrated predicted persistent tradability > 0.

The rule is fixed before fresh evaluation. Calibration economics must pass
before the June fresh block is formally evaluated. The target, features,
models, fit/calibration dates and fresh dates remain frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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

REQUEST_ID = 233

MIN_SELECTED_RATE = 0.03
MAX_SELECTED_RATE = 0.25
MIN_SELECTED_ROWS = 40
MIN_SELECTED_POSITIVE_RATE = 0.35
MIN_CAL_POSITIVE_MEAN_DAYS = 5
MIN_FRESH_POSITIVE_MEAN_DAYS = 4
MIN_MEAN_GAIN_VS_REQUEST232_PCT = 0.25


def apply_economic_admission(
    frame: pd.DataFrame,
    probability_gate: float,
) -> pd.DataFrame:
    result = frame.copy()
    probability = pd.to_numeric(
        result["tradability_positive_probability"],
        errors="coerce",
    )
    predicted_value = pd.to_numeric(
        result["predicted_persistent_tradability_pct"],
        errors="coerce",
    )
    result["request232_selected"] = probability.ge(
        float(probability_gate)
    )
    result["economic_admission_selected"] = (
        result["request232_selected"]
        & predicted_value.gt(0.0)
    )
    return result


def _selection_metrics(
    frame: pd.DataFrame,
    days: list[str] | tuple[str, ...],
    selection_column: str,
) -> dict[str, object]:
    subset = frame.loc[
        frame["trading_day"].astype(str).isin(days)
    ].copy()
    target = pd.to_numeric(
        subset["persistent_tradability_pct"],
        errors="coerce",
    )
    valid = target.notna()
    evaluable = subset.loc[valid].copy()
    target = pd.to_numeric(
        evaluable["persistent_tradability_pct"],
        errors="coerce",
    )
    selected = evaluable.loc[
        evaluable[selection_column]
        .fillna(False)
        .astype(bool)
    ].copy()
    selected_target = pd.to_numeric(
        selected["persistent_tradability_pct"],
        errors="coerce",
    )

    by_day: dict[str, object] = {}
    selected_day_means: list[float] = []
    positive_mean_days = 0
    days_with_selection = 0

    for day in days:
        day_all = evaluable.loc[
            evaluable["trading_day"].astype(str).eq(str(day))
        ].copy()
        chosen = day_all.loc[
            day_all[selection_column]
            .fillna(False)
            .astype(bool)
        ].copy()
        chosen_target = pd.to_numeric(
            chosen["persistent_tradability_pct"],
            errors="coerce",
        )
        chosen_mean = (
            float(chosen_target.mean())
            if len(chosen_target)
            else None
        )
        if chosen_mean is not None:
            selected_day_means.append(chosen_mean)
            days_with_selection += 1
            positive_mean_days += int(chosen_mean > 0)

        by_day[str(day)] = {
            "evaluable_rows": int(len(day_all)),
            "selected_rows": int(len(chosen)),
            "selected_rate": (
                float(len(chosen) / len(day_all))
                if len(day_all)
                else None
            ),
            "selected_mean_pct": chosen_mean,
            "selected_positive_rate": (
                float(chosen_target.gt(0).mean())
                if len(chosen_target)
                else None
            ),
        }

    population_mean = (
        float(target.mean())
        if len(target)
        else None
    )
    selected_mean = (
        float(selected_target.mean())
        if len(selected_target)
        else None
    )
    selected_positive_rate = (
        float(selected_target.gt(0).mean())
        if len(selected_target)
        else None
    )
    selected_rate = (
        float(len(selected) / len(evaluable))
        if len(evaluable)
        else None
    )
    coverage = (
        float(valid.mean())
        if len(subset)
        else None
    )
    day_balanced_mean = (
        float(np.mean(selected_day_means))
        if selected_day_means
        else None
    )

    return {
        "rows": int(len(subset)),
        "evaluable_rows": int(len(evaluable)),
        "coverage": coverage,
        "population_mean_pct": population_mean,
        "selected_rows": int(len(selected)),
        "selected_rate": selected_rate,
        "selected_mean_pct": selected_mean,
        "selected_positive_rate": selected_positive_rate,
        "day_balanced_selected_mean_pct": day_balanced_mean,
        "days_with_selection": days_with_selection,
        "positive_selected_mean_days": positive_mean_days,
        "by_day": by_day,
    }


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
            day_mean is not None
            and float(day_mean) > 0
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
    scored_hot = apply_economic_admission(
        scored_hot,
        fitted.probability_gate,
    )

    calibration_baseline = _selection_metrics(
        scored_hot,
        EXTENDED_CAL_DAYS,
        "request232_selected",
    )
    calibration_candidate = _selection_metrics(
        scored_hot,
        EXTENDED_CAL_DAYS,
        "economic_admission_selected",
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
        fresh_baseline = _selection_metrics(
            scored_hot,
            FRESH_EVAL_DAYS,
            "request232_selected",
        )
        fresh_candidate = _selection_metrics(
            scored_hot,
            FRESH_EVAL_DAYS,
            "economic_admission_selected",
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
        "hypothesis": (
            "Request232 already ranks persistent tradability well; "
            "requiring both high positive probability and positive "
            "calibrated predicted value may reject merely less-bad HOTs."
        ),
        "architecture": (
            "frozen Request232 tradability head plus a zero-parameter "
            "economic admission rule"
        ),
        "admission_rule": {
            "request232_probability_gate": float(
                fitted.probability_gate
            ),
            "probability_condition": (
                "tradability_positive_probability >= frozen "
                "Request232 calibration p75"
            ),
            "value_condition": (
                "predicted_persistent_tradability_pct > 0"
            ),
            "both_required": True,
            "fresh_used_to_choose_rule": False,
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
        "calibration": {
            "request232_baseline": calibration_baseline,
            "economic_candidate": calibration_candidate,
            "gate": calibration_gate_detail,
            "economic_gate_pass": calibration_gate,
        },
        "fresh_evaluated": bool(calibration_gate),
        "fresh": (
            {
                "request232_baseline": fresh_baseline,
                "economic_candidate": fresh_candidate,
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
            "freeze economic HOT admission, then train a new causal "
            "ENTER-vs-WAIT timing controller only inside economically "
            "admitted episodes; target coverage remains a separate "
            "promotion blocker until repaired."
        ),
        "next_boundary_if_fail_calibration": (
            "the Request232 score is rank-useful but its zero boundary "
            "is not economically calibrated; fit a calibration-only "
            "monotonic value map before changing representation."
        ),
        "next_boundary_if_fail_fresh": (
            "the absolute zero boundary does not transfer; test a "
            "calibration-only monotonic economic value map or a more "
            "stable market-conditioned admission threshold, without "
            "changing the underlying tradability learner."
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
