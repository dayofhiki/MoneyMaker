"""Request247: bootstrap WAIT from exactly one next observed eligible state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .chronological_action_value import (
    CAL_DAYS,
    KEYS,
    build_states,
    chronological_fit,
    episode_results,
    metrics,
    rollout,
)
from .direct_action_advantage import (
    DirectPolicy,
    decision_quality,
    fit_binary,
    fit_direct_policy,
    rollout_direct,
)

REQUEST_ID = 247
THRESHOLD = 0.5
MIN_ENTRIES = 20
MIN_RESOLVED = 20
MIN_WAIT_ACTIONS = 20
MIN_ACTION_PRECISION = 0.55
MAX_SEVERE_LOSS_247 = 0.35


def finite(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.notna() & np.isfinite(x)


def one_event_targets(student_targets: pd.DataFrame) -> pd.DataFrame:
    out = student_targets.copy()
    out["next_enter_value"] = np.nan
    out["next_can_enter"] = False
    out["next_state_t"] = np.nan
    for _, group in out.groupby(KEYS, sort=False):
        ids = list(group.index)
        if len(ids) < 2:
            continue
        for left, right in zip(ids[:-1], ids[1:]):
            out.at[left, "next_enter_value"] = out.at[right, "enter_value"]
            out.at[left, "next_can_enter"] = bool(out.at[right, "can_enter"]) and not bool(
                out.at[right, "terminal"]
            )
            out.at[left, "next_state_t"] = out.at[right, "state_t"]
    return out


def fit_redecision_policy(student_targets: pd.DataFrame) -> DirectPolicy:
    labeled = one_event_targets(student_targets)
    from .chronological_action_value import CAUSAL_SOURCE_FEATURES, PATH_FEATURES

    columns = tuple(
        c
        for c in dict.fromkeys([*CAUSAL_SOURCE_FEATURES, *PATH_FEATURES])
        if c in labeled and pd.to_numeric(labeled[c], errors="coerce").notna().any()
    )

    flat_valid = (
        labeled.can_enter.astype(bool)
        & ~labeled.terminal.astype(bool)
        & labeled.next_can_enter.astype(bool)
        & finite(labeled.enter_value)
        & finite(labeled.next_enter_value)
    )
    flat = labeled.loc[flat_valid].copy()
    enter_now = pd.to_numeric(flat.enter_value)
    wait_one = pd.to_numeric(flat.next_enter_value)
    enter_label = enter_now.gt(np.maximum(wait_one, 0.0))
    enter_model, enter_support = fit_binary(
        flat, enter_label, columns, seed=20261470
    )

    wait_population = flat.loc[~enter_label].copy()
    wait_label = pd.to_numeric(wait_population.next_enter_value).gt(0)
    wait_model, wait_support = fit_binary(
        wait_population, wait_label, columns, seed=20261471
    )

    hold_valid = ~labeled.terminal.astype(bool) & finite(labeled.hold_advantage)
    hold_frame = labeled.loc[hold_valid].copy()
    hold_label = pd.to_numeric(hold_frame.hold_advantage).gt(0)
    hold_model, hold_support = fit_binary(
        hold_frame, hold_label, columns, seed=20261472
    )

    support = {
        "enter_now_beats_wait_one_event_and_cash": enter_support,
        "wait_one_event_positive_given_enter_not_best": wait_support,
        "hold_better": hold_support,
        "target_population": {
            "flat_rows_with_next_eligible_state": int(len(flat)),
            "distinct_flat_episodes": int(flat[KEYS].drop_duplicates().shape[0]),
            "median_observed_gap_minutes": (
                float(
                    (
                        pd.to_numeric(flat.next_state_t)
                        - pd.to_numeric(flat.state_t)
                    ).median()
                    / 60_000
                )
                if len(flat)
                else None
            ),
        },
    }
    return DirectPolicy(
        enter_model=enter_model,
        wait_model=wait_model,
        hold_model=hold_model,
        columns=columns,
        support=support,
    )


def passes(value, threshold):
    return value is not None and np.isfinite(value) and value >= threshold


def evaluate(
    fit_positions: Path,
    calibration_positions: Path,
    output: Path,
    rows_output: Path,
) -> int:
    fit_states = build_states(pd.read_parquet(fit_positions))
    cal_states = build_states(pd.read_parquet(calibration_positions))
    if sorted(cal_states.trading_day.astype(str).unique()) != CAL_DAYS:
        raise ValueError("calibration dates changed")

    _, request243_student, student_targets = chronological_fit(fit_states)
    request246_policy = fit_direct_policy(student_targets)
    request247_policy = fit_redecision_policy(student_targets)

    scored247 = rollout_direct(cal_states, request247_policy)
    episodes247 = episode_results(scored247)
    metrics247 = metrics(episodes247, CAL_DAYS)
    quality247 = decision_quality(scored247)

    scored246 = rollout_direct(cal_states, request246_policy)
    metrics246 = metrics(episode_results(scored246), CAL_DAYS)

    scored243 = rollout(cal_states, request243_student, "full")
    metrics243 = metrics(episode_results(scored243), CAL_DAYS)

    checks = {
        "min_entries": metrics247["entries"] >= MIN_ENTRIES,
        "min_resolved_trades": metrics247["resolved_trades"] >= MIN_RESOLVED,
        "positive_resolved_trade_mean": (
            metrics247["resolved_trade_mean_pct"] is not None
            and metrics247["resolved_trade_mean_pct"] > 0
        ),
        "uses_wait": metrics247["wait_actions"] >= MIN_WAIT_ACTIONS,
        "chosen_enter_advantage_precision": passes(
            quality247["chosen_enter_advantage_positive_rate"],
            MIN_ACTION_PRECISION,
        ),
        "chosen_hold_advantage_precision": passes(
            quality247["chosen_hold_advantage_positive_rate"],
            MIN_ACTION_PRECISION,
        ),
        "severe_loss_rate": (
            metrics247["severe_loss_rate_le_minus2"] is not None
            and metrics247["severe_loss_rate_le_minus2"] <= MAX_SEVERE_LOSS_247
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "development_only": True,
        "threshold": THRESHOLD,
        "class_balancing": False,
        "training_support": request247_policy.support,
        "one_event_redecision_policy": metrics247,
        "decision_quality": quality247,
        "request246_refit_comparator": metrics246,
        "request243_refit_comparator": metrics243,
        "checks": checks,
        "development_gate_pass": bool(all(checks.values())),
        "notes": [
            "WAIT target uses exactly the next observed eligible state, never a best future state.",
            "Future event availability is used only to form historical training labels, never as a decision feature.",
            "Repeated WAIT at inference arises from receding one-step redecision.",
            "Resolved means are not portfolio returns when unresolved trades remain.",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    scored247.to_parquet(rows_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument("--calibration-positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
