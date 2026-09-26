from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.early_shape_pullback_gate import (
    build_decision_rows,
    simulate_probe_gated_policy,
)


def _states() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pullback_segment_id": "seg",
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 1,
                "pullback_age_events": 0,
                "pullback_elapsed_min": 0.0,
                "pullback_return_from_onset_pct": 0.0,
                "pullback_worst_from_onset_pct": 0.0,
                "pullback_recovery_from_worst_pct": 0.0,
                "pullback_onset_exit_pct": 1.0,
                "pullback_terminal_exit_pct": 1.8,
                "pullback_terminal_reason": "reacceleration_redecision",
                "adaptive_wait_value_pct": 0.8,
                "exit_now_base_return_pct": 1.0,
                "entry_conditioned_score": 0.5,
                "state_t": 60_000,
                "entry_to_current_close_pct": 1.0,
            },
            {
                "pullback_segment_id": "seg",
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 1,
                "pullback_age_events": 1,
                "pullback_elapsed_min": 1.0,
                "pullback_return_from_onset_pct": -0.4,
                "pullback_worst_from_onset_pct": -0.4,
                "pullback_recovery_from_worst_pct": 0.0,
                "pullback_onset_exit_pct": 1.0,
                "pullback_terminal_exit_pct": 1.8,
                "pullback_terminal_reason": "reacceleration_redecision",
                "adaptive_wait_value_pct": 1.2,
                "exit_now_base_return_pct": 0.6,
                "entry_conditioned_score": 0.5,
                "state_t": 120_000,
                "entry_to_current_close_pct": 0.6,
            },
            {
                "pullback_segment_id": "seg",
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 1,
                "pullback_age_events": 2,
                "pullback_elapsed_min": 2.0,
                "pullback_return_from_onset_pct": -0.1,
                "pullback_worst_from_onset_pct": -0.4,
                "pullback_recovery_from_worst_pct": 0.3,
                "pullback_onset_exit_pct": 1.0,
                "pullback_terminal_exit_pct": 1.8,
                "pullback_terminal_reason": "reacceleration_redecision",
                "adaptive_wait_value_pct": 0.9,
                "exit_now_base_return_pct": 0.9,
                "entry_conditioned_score": 0.5,
                "state_t": 180_000,
                "entry_to_current_close_pct": 0.9,
            },
        ]
    )


def test_decision_row_is_first_post_onset_only() -> None:
    decisions = build_decision_rows(_states())
    assert len(decisions) == 1
    row = decisions.iloc[0]
    assert row["pullback_age_events"] == 1
    assert np.isclose(row["early_shape_slope_pct_per_min"], -0.4)
    assert np.isclose(row["early_continue_value_pct"], 1.2)
    assert np.isclose(row["onset_entry_to_current_close_pct"], 1.0)
    assert np.isclose(
        row["early_delta_entry_to_current_close_pct"], -0.4
    )


def test_reject_after_probe_includes_probe_loss() -> None:
    states = _states()
    decisions = build_decision_rows(states)
    decisions["eligibility_score"] = -1.0
    rows = simulate_probe_gated_policy(
        states,
        decisions,
        threshold=0.0,
        score_column="entry_conditioned_score",
    )
    assert len(rows) == 1
    assert rows.iloc[0]["exit_reason"] == "early_shape_reject"
    assert np.isclose(rows.iloc[0]["local_uplift_pct"], -0.4)


def test_eligible_segment_hands_to_request223() -> None:
    states = _states()
    decisions = build_decision_rows(states)
    decisions["eligibility_score"] = 1.0
    rows = simulate_probe_gated_policy(
        states,
        decisions,
        threshold=0.0,
        score_column="entry_conditioned_score",
    )
    assert np.isclose(rows.iloc[0]["local_uplift_pct"], 0.8)


def test_request223_exit_after_probe_is_respected() -> None:
    states = _states()
    states.loc[
        states["pullback_age_events"].eq(2),
        "entry_conditioned_score",
    ] = -0.1
    decisions = build_decision_rows(states)
    decisions["eligibility_score"] = 1.0
    rows = simulate_probe_gated_policy(
        states,
        decisions,
        threshold=0.0,
        score_column="entry_conditioned_score",
    )
    assert rows.iloc[0]["exit_reason"] == "request223_model_exit"
    assert np.isclose(rows.iloc[0]["local_uplift_pct"], -0.1)


def test_no_decision_when_probe_reaches_terminal() -> None:
    states = _states().iloc[[0]].copy()
    decisions = build_decision_rows(states)
    assert decisions.empty
    rows = simulate_probe_gated_policy(
        states,
        decisions,
        threshold=0.0,
        score_column="entry_conditioned_score",
    )
    assert rows.iloc[0]["decision_available"] == False
    assert np.isclose(rows.iloc[0]["local_uplift_pct"], 0.8)
