"""Request 183: calibration-relative entry-value support diagnostic.

Reproduces Request 182 and replaces only the absolute predicted-value zero
boundary with score thresholds frozen from old calibration quantiles.
Development-only; no new dates are opened and no policy can be promoted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .causal_entry_action_value import (
    ENTER_SEED,
    WAIT_SEED,
    build_action_labels,
    build_entry_states,
    evaluate_policy,
    predict_value,
    train_value_model,
    _safe_spearman,
)
from .fresh_consensus_hurdle_value import FRESH_DAYS

REQUEST_ID = 183
SUPPORT_LEVELS = (0.80, 0.90, 0.95)
MIN_ACTION_VALUE_COVERAGE = 0.90
MIN_EXECUTED_RATE = 0.03
MAX_EXECUTED_RATE = 0.40
MIN_POSITIVE_DAYS = 3


def _score_models(
    frame: pd.DataFrame,
    enter_model,
    wait_model,
) -> pd.DataFrame:
    scored = frame.copy()
    scored["predicted_enter_now_base_pct"] = predict_value(
        scored,
        enter_model,
    )
    scored["predicted_wait_1m_base_pct"] = predict_value(
        scored,
        wait_model,
    )
    scored["best_predicted_value_pct"] = np.maximum(
        pd.to_numeric(
            scored["predicted_enter_now_base_pct"],
            errors="coerce",
        ),
        pd.to_numeric(
            scored["predicted_wait_1m_base_pct"],
            errors="coerce",
        ),
    )
    return scored


def _positive_trade_days(scored: pd.DataFrame) -> int:
    resolved = (
        scored["action_value_resolved"].fillna(False)
        & scored["executed"].fillna(False)
    )
    part = scored.loc[resolved].copy()
    if part.empty:
        return 0
    daily = (
        part.assign(
            _value=pd.to_numeric(
                part["policy_realized_base_pct"],
                errors="coerce",
            )
        )
        .groupby(part["trading_day"].astype(str), sort=True)["_value"]
        .mean()
        .dropna()
    )
    return int(daily.gt(0).sum())


def evaluate_support_level(
    fresh_scored: pd.DataFrame,
    *,
    threshold: float,
    label: str,
    enter_spearman: float | None,
    wait_spearman: float | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidate = fresh_scored.copy()
    candidate["raw_predicted_enter_now_base_pct"] = candidate[
        "predicted_enter_now_base_pct"
    ]
    candidate["raw_predicted_wait_1m_base_pct"] = candidate[
        "predicted_wait_1m_base_pct"
    ]

    # evaluate_policy chooses max(action values, 0). Subtracting the same
    # calibration-frozen threshold from both heads exactly implements
    # max(raw V_enter, raw V_wait) >= threshold while preserving argmax.
    candidate["predicted_enter_now_base_pct"] = (
        pd.to_numeric(
            candidate["raw_predicted_enter_now_base_pct"],
            errors="coerce",
        )
        - threshold
    )
    candidate["predicted_wait_1m_base_pct"] = (
        pd.to_numeric(
            candidate["raw_predicted_wait_1m_base_pct"],
            errors="coerce",
        )
        - threshold
    )
    evaluated, metrics = evaluate_policy(candidate)
    positive_days = _positive_trade_days(evaluated)

    executed_rate = metrics["executed_rate"]
    trade_day = metrics["executed_day_balanced_base_mean_pct"]
    all_day = metrics["all_opportunity_day_balanced_policy_value_pct"]
    diff_day = metrics["matched_policy_minus_current_day_balanced_pct"]
    diff_low = metrics["matched_difference_bootstrap"]["ci_low_pct"]
    severe = metrics["executed_severe_loss_rate"]
    comparator_severe = metrics["comparator_severe_loss_rate"]

    gate = bool(
        metrics["action_value_coverage"] is not None
        and float(metrics["action_value_coverage"])
        >= MIN_ACTION_VALUE_COVERAGE
        and executed_rate is not None
        and MIN_EXECUTED_RATE <= float(executed_rate) <= MAX_EXECUTED_RATE
        and trade_day is not None
        and float(trade_day) > 0
        and all_day is not None
        and float(all_day) > 0
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
        and severe is not None
        and comparator_severe is not None
        and float(severe) <= float(comparator_severe)
        and positive_days >= MIN_POSITIVE_DAYS
        and enter_spearman is not None
        and float(enter_spearman) > 0
        and wait_spearman is not None
        and float(wait_spearman) > 0
    )
    return evaluated, {
        "label": label,
        "calibration_threshold_pct": float(threshold),
        "positive_return_days": positive_days,
        "metrics": metrics,
        "development_support_gate_pass": gate,
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_path)
    cal_positions = pd.read_parquet(calibration_path)
    history_scan = pd.read_parquet(history_scan_path)

    fit_labels = build_action_labels(fit_positions)
    cal_labels = build_action_labels(cal_positions)
    historical_labels = pd.concat(
        [fit_labels, cal_labels],
        ignore_index=True,
    )
    historical_states = build_entry_states(
        history_scan,
        historical_labels,
    )
    fit_days = set(fit_labels["trading_day"].astype(str))
    cal_days = set(cal_labels["trading_day"].astype(str))
    fit = historical_states.loc[
        historical_states["trading_day"].astype(str).isin(fit_days)
    ].copy()
    calibration = historical_states.loc[
        historical_states["trading_day"].astype(str).isin(cal_days)
    ].copy()

    enter_model = train_value_model(
        fit,
        calibration,
        target_column="enter_now_base_pct",
        seed=ENTER_SEED,
    )
    wait_model = train_value_model(
        fit,
        calibration,
        target_column="wait_1m_base_pct",
        seed=WAIT_SEED,
    )
    cal_scored = _score_models(calibration, enter_model, wait_model)
    cal_best = pd.to_numeric(
        cal_scored["best_predicted_value_pct"],
        errors="coerce",
    ).dropna()
    if len(cal_best) < 150:
        raise ValueError(
            f"request 183 insufficient calibration scores: {len(cal_best)}"
        )
    thresholds = {
        f"CAL_P{int(level * 100)}": float(cal_best.quantile(level))
        for level in SUPPORT_LEVELS
    }

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError("request 183 requires complete request 178 shards")
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    fresh_labels = build_action_labels(fresh_positions)
    fresh_states = build_entry_states(fresh_scan, fresh_labels)
    fresh_scored = _score_models(
        fresh_states,
        enter_model,
        wait_model,
    )
    if sorted(
        fresh_scored["trading_day"].astype(str).unique()
    ) != list(FRESH_DAYS):
        raise ValueError("request 183 fresh block differs from request 178")

    enter_spearman = _safe_spearman(
        fresh_scored["enter_now_base_pct"],
        fresh_scored["predicted_enter_now_base_pct"],
    )
    wait_spearman = _safe_spearman(
        fresh_scored["wait_1m_base_pct"],
        fresh_scored["predicted_wait_1m_base_pct"],
    )

    support_results: dict[str, object] = {}
    row_parts: list[pd.DataFrame] = []
    passing: list[str] = []
    for label in ("CAL_P80", "CAL_P90", "CAL_P95"):
        evaluated, summary = evaluate_support_level(
            fresh_scored,
            threshold=thresholds[label],
            label=label,
            enter_spearman=enter_spearman,
            wait_spearman=wait_spearman,
        )
        evaluated["support_level"] = label
        row_parts.append(evaluated)
        support_results[label] = summary
        if summary["development_support_gate_pass"]:
            passing.append(label)

    candidate = passing[0] if passing else None
    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "action_head_fresh_spearman": {
            "enter_now": enter_spearman,
            "wait_1m": wait_spearman,
        },
        "calibration_best_score": {
            "rows": int(len(cal_best)),
            "mean_pct": float(cal_best.mean()),
            "median_pct": float(cal_best.median()),
            "max_pct": float(cal_best.max()),
            "thresholds": thresholds,
        },
        "support_results": support_results,
        "passing_support_levels": passing,
        "candidate_for_untouched_validation": candidate,
        "selection_rule": (
            "least selective passing support level: P80 then P90 then P95"
        ),
        "frozen_gate": {
            "min_action_value_coverage": MIN_ACTION_VALUE_COVERAGE,
            "min_executed_rate": MIN_EXECUTED_RATE,
            "max_executed_rate": MAX_EXECUTED_RATE,
            "min_positive_days": MIN_POSITIVE_DAYS,
            "executed_day_balanced_base_must_be_positive": True,
            "all_opportunity_value_must_be_positive": True,
            "matched_difference_must_be_positive": True,
            "matched_bootstrap_low_must_be_positive": True,
            "severe_loss_no_worse_than_current": True,
            "both_action_head_spearman_must_be_positive": True,
        },
        "development_gate_pass": bool(passing),
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.concat(row_parts, ignore_index=True).to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--history-scan", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
