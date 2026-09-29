"""Request269: calibration-separated recurrent causal HOLD/EXIT controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .causal_hold_exit_controller import (
    _base_record,
    policy_metrics,
    static_policy_rows,
)
from .causal_hold_exit_signal import (
    build_position_states,
    fit_models,
    path_columns,
)
from .causal_minute_controller_rebuild import causal_execution_scan
from .full_hot_fixed_policy_value import attach_fixed_value
from .learned_pullback_entry import CONTROLLER_TRAIN_DAYS, TEST_DAYS, controller_x
from .rich_post_hot_state import score_and_select_candidates

REQUEST_ID = 269
MODEL_FIT_DAYS = ("2026-05-05", "2026-05-06", "2026-05-07")
CALIBRATION_DAY = "2026-05-08"
PROBABILITY_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
ADVANTAGE_THRESHOLDS = (0.00, 0.25, 0.50, 1.00)
MIN_CAL_GAIN_VS_FIRST = 0.25
MAX_CAL_SEVERE_DEGRADATION = 0.02

MIN_TEST_POSITIVE_RATE = 0.40
MAX_TEST_SEVERE_RATE = 0.15
MIN_TEST_POSITIVE_DAYS = 5
MIN_TEST_GAIN_VS_BEST_STATIC = 0.10


def dynamic_policy_rows(
    states: pd.DataFrame,
    reg,
    cls,
    columns: tuple[str, ...],
    *,
    probability_threshold: float,
    advantage_threshold: float,
) -> pd.DataFrame:
    work = states.copy()
    x = controller_x(work, columns)
    work["predicted_hold_advantage_pct"] = reg.predict(x)
    work["hold_probability"] = cls.predict_proba(x)[:, 1]
    rows = []
    for _, group in work.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        ordered = group.sort_values("state_t").reset_index(drop=True)
        if ordered.empty:
            continue
        chosen = ordered.iloc[-1]
        exit_reason = "terminal_cap"
        for _, state in ordered.iterrows():
            hold = (
                float(state.predicted_hold_advantage_pct)
                > float(advantage_threshold)
                and float(state.hold_probability)
                >= float(probability_threshold)
            )
            if not hold:
                chosen = state
                exit_reason = "model_exit"
                break
        record = _base_record(ordered, chosen, "dynamic")
        record["exit_reason"] = exit_reason
        record["probability_threshold"] = float(probability_threshold)
        record["advantage_threshold_pct"] = float(advantage_threshold)
        rows.append(record)
    return pd.DataFrame(rows)


def calibration_table(
    states: pd.DataFrame,
    reg,
    cls,
    columns: tuple[str, ...],
) -> tuple[dict | None, list[dict]]:
    first = policy_metrics(static_policy_rows(states, "first"))
    rows = []
    passing = []
    for probability_threshold in PROBABILITY_THRESHOLDS:
        for advantage_threshold in ADVANTAGE_THRESHOLDS:
            policy = dynamic_policy_rows(
                states,
                reg,
                cls,
                columns,
                probability_threshold=probability_threshold,
                advantage_threshold=advantage_threshold,
            )
            metrics = policy_metrics(policy)
            gain = (
                metrics["mean_pct"] - first["mean_pct"]
                if metrics["mean_pct"] is not None
                and first["mean_pct"] is not None
                else None
            )
            severe_degradation = (
                metrics["severe_loss_rate_le_minus2"]
                - first["severe_loss_rate_le_minus2"]
                if metrics["severe_loss_rate_le_minus2"] is not None
                and first["severe_loss_rate_le_minus2"] is not None
                else None
            )
            checks = {
                "gain_vs_first": (
                    gain is not None
                    and gain >= MIN_CAL_GAIN_VS_FIRST
                ),
                "severe_not_materially_worse": (
                    severe_degradation is not None
                    and severe_degradation
                    <= MAX_CAL_SEVERE_DEGRADATION
                ),
            }
            row = {
                "probability_threshold": float(probability_threshold),
                "advantage_threshold_pct": float(advantage_threshold),
                "metrics": metrics,
                "first_exit": first,
                "mean_gain_vs_first_pct": gain,
                "severe_rate_delta_vs_first": severe_degradation,
                "checks": checks,
                "passes": bool(all(checks.values())),
            }
            rows.append(row)
            if row["passes"]:
                passing.append(row)
    if not passing:
        return None, rows
    passing.sort(
        key=lambda row: (
            -(row["mean_gain_vs_first_pct"] or -999.0),
            row["metrics"]["severe_loss_rate_le_minus2"],
            -row["probability_threshold"],
            -row["advantage_threshold_pct"],
        )
    )
    return passing[0], rows


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    causal_scan = causal_execution_scan(scan)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, causal_scan)
    scored, train_threshold, _ = score_and_select_candidates(first_hot)

    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(CONTROLLER_TRAIN_DAYS)
        & scored.candidate_probability.ge(train_threshold)
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_probability.ge(train_threshold)
    ].copy()

    train_states = build_position_states(train_selected, causal_scan)
    test_states = build_position_states(test_selected, causal_scan)
    fit_states = train_states.loc[
        train_states.trading_day.astype(str).isin(MODEL_FIT_DAYS)
    ].copy()
    calibration_states = train_states.loc[
        train_states.trading_day.astype(str).eq(CALIBRATION_DAY)
    ].copy()

    columns = path_columns(fit_states)
    reg, cls = fit_models(fit_states, columns, 20261410)
    chosen, calibration = calibration_table(
        calibration_states,
        reg,
        cls,
        columns,
    )

    if chosen is None:
        result = {
            "request_id": REQUEST_ID,
            "development_only": True,
            "opens_new_dates": False,
            "promotion_eligible": False,
            "causal_reference_contract": True,
            "candidate_threshold_top20": train_threshold,
            "support": {
                "fit_position_states": int(len(fit_states)),
                "calibration_position_states": int(len(calibration_states)),
                "test_position_states": int(len(test_states)),
            },
            "calibration_pass": False,
            "calibration": calibration,
            "economic_gate_pass": False,
            "next_boundary": (
                "replace oracle-best-future labels with policy-consistent "
                "continuation-value targets"
            ),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0

    probability_threshold = float(chosen["probability_threshold"])
    advantage_threshold = float(chosen["advantage_threshold_pct"])
    dynamic_rows = dynamic_policy_rows(
        test_states,
        reg,
        cls,
        columns,
        probability_threshold=probability_threshold,
        advantage_threshold=advantage_threshold,
    )
    static_rows = {
        mode: static_policy_rows(test_states, mode)
        for mode in ("first", "fifth", "terminal")
    }
    dynamic = policy_metrics(dynamic_rows)
    statics = {
        mode: policy_metrics(rows)
        for mode, rows in static_rows.items()
    }
    best_static_mean = max(
        report["mean_pct"]
        for report in statics.values()
        if report["mean_pct"] is not None
    )
    gain_vs_best_static = dynamic["mean_pct"] - best_static_mean

    checks = {
        "dynamic_mean_positive": dynamic["mean_pct"] > 0,
        "day_balanced_mean_positive": (
            dynamic["day_balanced_mean_pct"] > 0
        ),
        "positive_rate": (
            dynamic["positive_rate"] >= MIN_TEST_POSITIVE_RATE
        ),
        "severe_loss_rate": (
            dynamic["severe_loss_rate_le_minus2"]
            <= MAX_TEST_SEVERE_RATE
        ),
        "positive_mean_days": (
            dynamic["positive_mean_days"] >= MIN_TEST_POSITIVE_DAYS
        ),
        "gain_vs_best_static": (
            gain_vs_best_static >= MIN_TEST_GAIN_VS_BEST_STATIC
        ),
    }

    all_rows = [dynamic_rows]
    for mode, rows in static_rows.items():
        tagged = rows.copy()
        tagged["policy"] = mode
        all_rows.append(tagged)

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "causal_reference_contract": True,
        "candidate_threshold_top20": train_threshold,
        "support": {
            "fit_position_states": int(len(fit_states)),
            "calibration_position_states": int(len(calibration_states)),
            "test_position_states": int(len(test_states)),
        },
        "calibration_pass": True,
        "chosen_thresholds": {
            "probability_threshold": probability_threshold,
            "advantage_threshold_pct": advantage_threshold,
        },
        "chosen_calibration": chosen,
        "calibration": calibration,
        "test_policies": {
            "dynamic_hold_exit": dynamic,
            "static_first_exit": statics["first"],
            "static_fifth_exit": statics["fifth"],
            "static_terminal_exit": statics["terminal"],
        },
        "best_static_mean_pct": best_static_mean,
        "mean_gain_vs_best_static_pct": gain_vs_best_static,
        "checks": checks,
        "economic_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "May5-7 fit the HOLD model, May8 chooses only the preregistered "
            "HOLD thresholds, and May11-20 is development evaluation. "
            "All prices are causal decision-time minute opens."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.concat(all_rows, ignore_index=True).to_parquet(
        rows_output,
        index=False,
    )
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
