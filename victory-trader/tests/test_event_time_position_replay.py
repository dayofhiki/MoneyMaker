from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.event_time_position_replay import (
    EVENT_FEATURES,
    build_event_trajectories,
    reanchor_event_positions,
)
from victory_trader.recurrent_wait_entry_action_value import _base_return


def _positions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 0,
                "minutes_held": 1.0,
                "state_t": 60_000,
                "exit_reference_open": 10.0,
                "log_current_close": np.log(10.0),
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 0,
                "minutes_held": 2.0,
                "state_t": 120_000,
                "exit_reference_open": 11.0,
                "log_current_close": np.log(11.0),
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 0,
                "minutes_held": 4.0,
                "state_t": 240_000,
                "exit_reference_open": 12.0,
                "log_current_close": np.log(12.0),
            },
        ]
    )


def _entries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 0,
                "entry_minute_after_hot": 1,
                "realized_base_return_pct": 1.0,
            }
        ]
    )


def test_event_reanchor_uses_next_observed_state_not_exact_next_minute() -> None:
    event = reanchor_event_positions(_positions(), _entries())
    assert list(event["event_index"]) == [1, 2]
    assert list(event["minutes_held"]) == [1.0, 3.0]
    assert list(event["elapsed_since_previous_observation_min"]) == [1.0, 2.0]

    first = event.iloc[0]
    expected_current = _base_return(10.0, 11.0)
    expected_next = _base_return(10.0, 12.0)
    assert np.isclose(first["exit_now_base_return_pct"], expected_current)
    assert np.isclose(first["next_event_base_return_pct"], expected_next)
    assert np.isclose(
        first["hold_advantage_event_pct"],
        expected_next - expected_current,
    )


def test_event_features_do_not_include_future_wait_time() -> None:
    assert "elapsed_since_previous_observation_min" in EVENT_FEATURES
    assert "next_event_t" not in EVENT_FEATURES
    assert "time_to_next_observation" not in EVENT_FEATURES


def test_event_trajectory_forces_no_action_through_silent_clock() -> None:
    scored = reanchor_event_positions(_positions(), _entries())
    scored["predicted_hold_advantage_event_pct"] = [0.25, -0.10]

    trajectories, coverage = build_event_trajectories(
        scored, _entries()
    )
    row = trajectories.iloc[0]
    assert coverage["start_state_coverage"] == 1.0
    assert row["status"] == "completed"
    assert row["exit_reason"] == "model_exit"
    assert row["hold_decisions"] == 1
    assert row["observed_decisions"] == 2
    assert np.isclose(row["forced_silence_minutes"], 1.0)
    assert np.isclose(row["minutes_held"], 3.0)


def test_event_trajectory_does_not_turn_missing_future_into_cash() -> None:
    scored = reanchor_event_positions(_positions(), _entries()).head(1)
    scored["predicted_hold_advantage_event_pct"] = [0.25]

    trajectories, _ = build_event_trajectories(scored, _entries())
    row = trajectories.iloc[0]
    assert row["status"] == "unresolved"
    assert row["exit_reason"] == "no_next_executable_observation"
    assert pd.isna(row["base_net_return_pct"])
