from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from victory_trader.event_time_position_policy import (
    attach_event_transition_rates,
    build_event_trajectories,
    reanchor_event_positions,
)


def _entry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "entry_minute_after_hot": 2,
                "realized_base_return_pct": 0.0,
            }
        ]
    )


def test_reanchor_event_positions_preserves_silent_gap() -> None:
    rows = []
    for minute, state_t, exit_open, close in [
        (2.0, 121000, 10.0, 10.0),
        (3.0, 181000, 10.2, 10.2),
        (5.0, 301000, 10.5, 10.5),
    ]:
        rows.append(
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": minute,
                "state_t": state_t,
                "exit_reference_open": exit_open,
                "log_current_close": np.log(close),
            }
        )

    out = reanchor_event_positions(pd.DataFrame(rows), _entry())

    assert list(out["minutes_held"]) == [1.0, 3.0]
    assert list(out["event_prev_gap_minutes"]) == [1.0, 2.0]
    assert list(out["event_prev_stale_minutes"]) == [0.0, 1.0]
    assert out.iloc[0]["next_event_delta_minutes"] == pytest.approx(2.0)
    assert pd.notna(out.iloc[0]["hold_advantage_next_event_pct"])
    assert pd.isna(out.iloc[1]["hold_advantage_next_event_pct"])


def test_event_transition_rate_normalizes_by_known_previous_gap() -> None:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 1.0,
                "event_prev_gap_minutes": 1.0,
                "entry_to_current_close_pct": 1.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 3.0,
                "event_prev_gap_minutes": 2.0,
                "entry_to_current_close_pct": 5.0,
            },
        ]
    )

    out = attach_event_transition_rates(frame)
    row = out.iloc[1]
    assert row["position_return_change_1m_pct"] == pytest.approx(4.0)
    assert row[
        "position_return_change_1m_pct_per_elapsed_minute"
    ] == pytest.approx(2.0)


def test_event_trajectory_holds_across_silent_clock_gap() -> None:
    scored = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 1.0,
                "state_t": 181000,
                "entry_actual_t": 121000,
                "exit_now_base_return_pct": -0.5,
                "predicted_hold_advantage_next_event_pct": 0.4,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 3.0,
                "state_t": 301000,
                "entry_actual_t": 121000,
                "exit_now_base_return_pct": 1.2,
                "predicted_hold_advantage_next_event_pct": -0.1,
            },
        ]
    )

    trajectories, coverage = build_event_trajectories(
        scored,
        score_column="predicted_hold_advantage_next_event_pct",
    )
    row = trajectories.iloc[0]
    assert row["status"] == "completed"
    assert row["exit_reason"] == "model_exit"
    assert row["base_net_return_pct"] == pytest.approx(1.2)
    assert row["hold_decisions"] == 1
    assert row["minutes_held"] == pytest.approx(3.0)
    assert coverage["start_state_coverage"] == pytest.approx(1.0)


def test_event_trajectory_unresolved_when_hold_has_no_later_event() -> None:
    scored = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 1.0,
                "state_t": 181000,
                "entry_actual_t": 121000,
                "exit_now_base_return_pct": 0.2,
                "predicted_hold_advantage_next_event_pct": 0.5,
            }
        ]
    )

    trajectories, _ = build_event_trajectories(
        scored,
        score_column="predicted_hold_advantage_next_event_pct",
    )
    row = trajectories.iloc[0]
    assert row["status"] == "unresolved"
    assert row["exit_reason"] == "no_later_executable_event"
    assert pd.isna(row["base_net_return_pct"])


def test_event_trajectory_forces_exit_at_first_observed_cap_event() -> None:
    scored = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 29.0,
                "state_t": 1861000,
                "entry_actual_t": 121000,
                "exit_now_base_return_pct": 0.5,
                "predicted_hold_advantage_next_event_pct": 0.2,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 31.0,
                "state_t": 1981000,
                "entry_actual_t": 121000,
                "exit_now_base_return_pct": 0.7,
                "predicted_hold_advantage_next_event_pct": 0.1,
            },
        ]
    )

    trajectories, _ = build_event_trajectories(
        scored,
        score_column="predicted_hold_advantage_next_event_pct",
    )
    row = trajectories.iloc[0]
    assert row["status"] == "completed"
    assert row["exit_reason"] == "forced_cap_next_event"
    assert row["base_net_return_pct"] == pytest.approx(0.7)
    assert row["minutes_held"] == pytest.approx(31.0)
