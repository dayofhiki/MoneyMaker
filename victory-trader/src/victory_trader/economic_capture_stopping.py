"""Request 218: economic + capture recurrent stopping.

Request187 showed a positive 30-minute hindsight ceiling, Request178 learned a
fresh ranking of remaining option value, and Request189 learned strong relative
within-episode capture ranking but lost money because a losing path also has a
"best" point.

This request combines those already-frozen ideas without fitting a new model on
the opened Request178 development block.

At an executable position state:
- predicted_remaining_option_value =
    Request178 consensus excess-value score + its frozen minute baseline
- predicted_future_best_base =
    current executable BASE exit + predicted_remaining_option_value
- Request189 capture percentile estimates how high the current exit is relative
  to the episode's executable 30-minute path.

Economic HOLD requires BOTH predicted_remaining_option_value > 0 and
predicted_future_best_base > 0.
Hybrid HOLD additionally requires predicted_capture_percentile below the
Request189 calibration-selected threshold. Otherwise EXIT now.

The evaluation block is already opened and is development-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .oracle_capture_percentile_stopping import (
    BOOTSTRAP_SEED,
    MAX_HOLD_MINUTES,
    _complete_exit,
    _episode_rows,
    _first_later_exit,
    add_capture_target,
    build_trajectories,
    choose_threshold,
    matched_difference,
    predict_capture,
    summarize,
    train_capture_model,
)

REQUEST_ID = 218
MIN_START_COVERAGE = 0.85
MIN_COMPLETION_COVERAGE = 0.90
MIN_POSITIVE_DAYS = 4


def attach_economic_state(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    baseline = pd.to_numeric(
        result.get("fit_minute_baseline_pct"), errors="coerce"
    )
    excess = pd.to_numeric(
        result.get("consensus_hurdle_ev_pct"), errors="coerce"
    )
    current = pd.to_numeric(
        result.get("exit_now_base_return_pct"), errors="coerce"
    )
    result["predicted_remaining_option_value_pct"] = baseline + excess
    result["predicted_future_best_base_pct"] = (
        current + result["predicted_remaining_option_value_pct"]
    )
    return result


def _policy_decision(
    row: pd.Series,
    *,
    policy: str,
    capture_threshold: float,
) -> tuple[bool, str]:
    """Return (should_exit, reason) using only current-state information."""
    current = pd.to_numeric(
        pd.Series([row.get("exit_now_base_return_pct")]),
        errors="coerce",
    ).iloc[0]
    remaining = pd.to_numeric(
        pd.Series([row.get("predicted_remaining_option_value_pct")]),
        errors="coerce",
    ).iloc[0]
    future_best = pd.to_numeric(
        pd.Series([row.get("predicted_future_best_base_pct")]),
        errors="coerce",
    ).iloc[0]
    capture = pd.to_numeric(
        pd.Series([row.get("predicted_capture_percentile")]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(current):
        return True, "missing_current_exit"
    if (
        pd.isna(remaining)
        or not np.isfinite(float(remaining))
        or pd.isna(future_best)
        or not np.isfinite(float(future_best))
    ):
        return True, "economic_unscoreable_exit"

    economic_hold = float(remaining) > 0 and float(future_best) > 0
    if not economic_hold:
        if float(future_best) <= 0:
            return True, "no_predicted_cost_cover_exit"
        return True, "no_predicted_incremental_upside"

    if policy == "economic_only":
        return False, "economic_hold"

    if policy != "economic_capture":
        raise ValueError(f"unknown request218 policy {policy}")
    if pd.isna(capture) or not np.isfinite(float(capture)):
        return True, "capture_unscoreable_exit"
    if float(capture) >= float(capture_threshold):
        return True, "capture_percentile_exit"
    return False, "economic_capture_hold"


def build_economic_trajectories(
    scored: pd.DataFrame,
    *,
    policy: str,
    capture_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    total = int(
        scored.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    records: list[dict[str, object]] = []

    for _, group in scored.groupby(EPISODE_KEYS, sort=False):
        by_minute = _episode_rows(group)
        first = by_minute.get(1)
        if first is None:
            continue

        status = "unresolved"
        reason = "unknown"
        realized = np.nan
        exit_minute = np.nan
        hold_decisions = 0

        minute = 1
        while minute < MAX_HOLD_MINUTES:
            row = by_minute.get(minute)
            if row is None:
                later = _first_later_exit(by_minute, minute)
                if later is not None:
                    later_minute, value = later
                    status = "completed"
                    reason = "missing_reached_state_delayed_open"
                    realized = float(value)
                    exit_minute = float(later_minute)
                else:
                    reason = "missing_reached_state"
                break

            current_exit = pd.to_numeric(
                pd.Series([row.get("exit_now_base_return_pct")]),
                errors="coerce",
            ).iloc[0]
            should_exit, decision_reason = _policy_decision(
                row,
                policy=policy,
                capture_threshold=capture_threshold,
            )
            if should_exit:
                status, reason, realized, exit_minute = _complete_exit(
                    by_minute,
                    minute,
                    current_exit,
                    decision_reason,
                )
                break

            hold_decisions += 1
            if minute == MAX_HOLD_MINUTES - 1:
                next_return = pd.to_numeric(
                    pd.Series(
                        [row.get("next_minute_base_return_pct")]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(next_return):
                    status = "completed"
                    reason = "forced_30m_cap"
                    realized = float(next_return)
                    exit_minute = float(MAX_HOLD_MINUTES)
                else:
                    later = _first_later_exit(by_minute, minute)
                    if later is not None:
                        later_minute, value = later
                        status = "completed"
                        reason = "forced_cap_delayed_open"
                        realized = float(value)
                        exit_minute = float(later_minute)
                    else:
                        reason = "missing_forced_cap_exit"
                break

            next_row = by_minute.get(minute + 1)
            if next_row is None:
                next_return = pd.to_numeric(
                    pd.Series(
                        [row.get("next_minute_base_return_pct")]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(next_return):
                    status = "completed"
                    reason = "missing_state_next_minute_exit"
                    realized = float(next_return)
                    exit_minute = float(minute + 1)
                else:
                    later = _first_later_exit(by_minute, minute)
                    if later is not None:
                        later_minute, value = later
                        status = "completed"
                        reason = "missing_state_delayed_open_exit"
                        realized = float(value)
                        exit_minute = float(later_minute)
                    else:
                        reason = "missing_state_and_exit"
                break

            minute += 1

        records.append(
            {
                "trading_day": str(first["trading_day"]),
                "ticker": str(first["ticker"]).upper(),
                "hot_t": int(first["hot_t"]),
                "policy": policy,
                "status": status,
                "exit_reason": reason,
                "base_net_return_pct": (
                    float(realized) if pd.notna(realized) else np.nan
                ),
                "minutes_held": (
                    float(exit_minute)
                    if pd.notna(exit_minute)
                    else np.nan
                ),
                "hold_decisions": int(hold_decisions),
            }
        )

    trajectories = pd.DataFrame(records)
    started = int(len(trajectories))
    return trajectories, {
        "total_position_episodes": total,
        "started_episodes": started,
        "start_state_coverage": (
            float(started / total) if total else None
        ),
    }


def _positive_days(trajectories: pd.DataFrame) -> int:
    completed = trajectories.loc[
        trajectories["status"].eq("completed")
    ].copy()
    values = pd.to_numeric(
        completed["base_net_return_pct"], errors="coerce"
    )
    valid = values.notna()
    if not valid.any():
        return 0
    daily = pd.DataFrame(
        {
            "day": completed.loc[valid, "trading_day"].astype(str),
            "value": values.loc[valid].to_numpy(dtype=float),
        }
    ).groupby("day")["value"].mean()
    return int(daily.gt(0).sum())


def state_diagnostics(scored: pd.DataFrame) -> dict[str, object]:
    remaining = pd.to_numeric(
        scored["predicted_remaining_option_value_pct"],
        errors="coerce",
    )
    future_best = pd.to_numeric(
        scored["predicted_future_best_base_pct"],
        errors="coerce",
    )
    current = pd.to_numeric(
        scored["exit_now_base_return_pct"], errors="coerce"
    )
    capture = pd.to_numeric(
        scored["predicted_capture_percentile"], errors="coerce"
    )
    valid = (
        remaining.notna()
        & future_best.notna()
        & current.notna()
        & capture.notna()
    )
    actual_future_best = pd.to_numeric(
        scored.get("best_future_base_return_pct"),
        errors="coerce",
    )
    actual_remaining = pd.to_numeric(
        scored.get("remaining_option_value_pct"),
        errors="coerce",
    )
    return {
        "rows": int(len(scored)),
        "joint_evaluable_rows": int(valid.sum()),
        "joint_coverage": float(valid.mean()) if len(scored) else None,
        "predicted_remaining_positive_rate": (
            float(remaining.loc[valid].gt(0).mean())
            if int(valid.sum())
            else None
        ),
        "predicted_future_cost_cover_rate": (
            float(future_best.loc[valid].gt(0).mean())
            if int(valid.sum())
            else None
        ),
        "both_economic_hold_conditions_rate": (
            float(
                (
                    remaining.loc[valid].gt(0)
                    & future_best.loc[valid].gt(0)
                ).mean()
            )
            if int(valid.sum())
            else None
        ),
        "remaining_value_spearman": (
            float(
                actual_remaining.loc[valid].corr(
                    remaining.loc[valid], method="spearman"
                )
            )
            if int(valid.sum()) >= 20
            else None
        ),
        "future_best_spearman": (
            float(
                actual_future_best.loc[valid].corr(
                    future_best.loc[valid], method="spearman"
                )
            )
            if int(valid.sum()) >= 20
            else None
        ),
        "capture_spearman": (
            float(
                pd.to_numeric(
                    scored.loc[valid, "capture_percentile"],
                    errors="coerce",
                ).corr(capture.loc[valid], method="spearman")
            )
            if int(valid.sum()) >= 20
            else None
        ),
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    fresh_scored_path: Path,
    output_path: Path,
    trajectories_output_path: Path,
) -> int:
    fit = add_capture_target(pd.read_parquet(fit_path))
    calibration = add_capture_target(
        pd.read_parquet(calibration_path)
    )
    capture_model = train_capture_model(fit)
    calibration["predicted_capture_percentile"] = predict_capture(
        calibration, capture_model
    )
    threshold, calibration_table = choose_threshold(calibration)

    fresh = pd.read_parquet(fresh_scored_path)
    found = sorted(fresh["trading_day"].astype(str).unique())
    if found != list(FRESH_DAYS):
        raise ValueError(
            f"request218 expected {FRESH_DAYS}, found {found}"
        )
    fresh = add_capture_target(fresh)
    fresh["predicted_capture_percentile"] = predict_capture(
        fresh, capture_model
    )
    fresh = attach_economic_state(fresh)

    capture_only, capture_coverage = build_trajectories(
        fresh,
        policy="capture_percentile",
        threshold=threshold,
    )
    minute1, minute1_coverage = build_trajectories(
        fresh,
        policy="minute1_exit",
    )
    hold30, hold30_coverage = build_trajectories(
        fresh,
        policy="hold30",
    )
    economic_only, economic_coverage = build_economic_trajectories(
        fresh,
        policy="economic_only",
        capture_threshold=threshold,
    )
    hybrid, hybrid_coverage = build_economic_trajectories(
        fresh,
        policy="economic_capture",
        capture_threshold=threshold,
    )

    policies = {
        "minute1": (minute1, minute1_coverage),
        "hold30": (hold30, hold30_coverage),
        "capture_only": (capture_only, capture_coverage),
        "economic_only": (economic_only, economic_coverage),
        "economic_capture": (hybrid, hybrid_coverage),
    }
    summaries: dict[str, object] = {}
    for index, (name, (trajectory, coverage)) in enumerate(
        policies.items()
    ):
        summary = summarize(
            trajectory, seed=BOOTSTRAP_SEED + 2180 + index
        )
        summary["positive_days"] = _positive_days(trajectory)
        summaries[name] = {
            "coverage": coverage,
            "summary": summary,
        }

    diff_minute1 = matched_difference(
        hybrid,
        minute1,
        seed=BOOTSTRAP_SEED + 2190,
    )
    diff_capture = matched_difference(
        hybrid,
        capture_only,
        seed=BOOTSTRAP_SEED + 2191,
    )
    diff_economic = matched_difference(
        hybrid,
        economic_only,
        seed=BOOTSTRAP_SEED + 2192,
    )

    h = summaries["economic_capture"]["summary"]
    start = hybrid_coverage["start_state_coverage"]
    completion = h["completion_coverage"]
    day_mean = h["day_balanced_base_mean_pct"]
    low = h["day_bootstrap"]["ci_low_pct"]
    positive_days = h["positive_days"]
    severe = h["severe_loss_rate"]
    base_severe = summaries["minute1"]["summary"]["severe_loss_rate"]
    d_min = diff_minute1["day_balanced_difference_pct"]
    d_min_low = diff_minute1["bootstrap"]["ci_low_pct"]

    gate = bool(
        start is not None
        and float(start) >= MIN_START_COVERAGE
        and completion is not None
        and float(completion) >= MIN_COMPLETION_COVERAGE
        and day_mean is not None
        and float(day_mean) > 0
        and low is not None
        and float(low) > 0
        and int(positive_days) >= MIN_POSITIVE_DAYS
        and severe is not None
        and base_severe is not None
        and float(severe) <= float(base_severe)
        and d_min is not None
        and float(d_min) > 0
        and d_min_low is not None
        and float(d_min_low) > 0
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "new_model_fit_on_fresh_block": False,
        "capture_threshold_source": (
            "Request189 chronological calibration selection"
        ),
        "capture_threshold": threshold,
        "capture_calibration_table": calibration_table,
        "economic_rule": (
            "HOLD only if predicted absolute remaining option value > 0 "
            "and predicted future-best BASE return > 0"
        ),
        "hybrid_rule": (
            "economic HOLD rule AND predicted capture percentile "
            "< Request189 calibration threshold"
        ),
        "state_diagnostics": state_diagnostics(fresh),
        "policies": summaries,
        "hybrid_minus_minute1": diff_minute1,
        "hybrid_minus_capture_only": diff_capture,
        "hybrid_minus_economic_only": diff_economic,
        "request187_reference": {
            "oracle_30m_day_balanced_pct": 0.8571586344758947,
            "oracle_30m_bootstrap_low_pct": 0.34489826624737735,
            "oracle_30m_positive_rate": 0.4590909090909091,
            "positive_days": 5,
        },
        "request189_reference": {
            "capture_spearman": 0.44480542379688104,
            "candidate_day_balanced_pct": -2.0037600080494684,
        },
        "frozen_gate": {
            "min_start_coverage": MIN_START_COVERAGE,
            "min_completion_coverage": MIN_COMPLETION_COVERAGE,
            "hybrid_day_balanced_must_be_positive": True,
            "hybrid_bootstrap_low_must_be_positive": True,
            "min_positive_days": MIN_POSITIVE_DAYS,
            "severe_loss_no_worse_than_minute1": True,
            "hybrid_minus_minute1_must_be_positive": True,
            "hybrid_minus_minute1_bootstrap_low_must_be_positive": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.concat(
        [
            trajectory.assign(policy_name=name)
            for name, (trajectory, _) in policies.items()
        ],
        ignore_index=True,
    ).to_parquet(
        trajectories_output_path,
        index=False,
        compression="zstd",
    )
    fresh.to_parquet(
        output_path.with_name(output_path.stem + "-states.parquet"),
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, required=True
    )
    parser.add_argument(
        "--fresh-scored", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectories-output", type=Path, required=True
    )
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.fresh_scored,
        args.output,
        args.trajectories_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
