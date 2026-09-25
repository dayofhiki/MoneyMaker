from __future__ import annotations

import pandas as pd
import pytest

from victory_trader.economic_capture_stopping import (
    _policy_decision,
    attach_economic_state,
    build_economic_trajectories,
)


def _row(
    *,
    current: float,
    baseline: float,
    excess: float,
    capture: float,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 1.0,
                "exit_now_base_return_pct": current,
                "fit_minute_baseline_pct": baseline,
                "consensus_hurdle_ev_pct": excess,
                "predicted_capture_percentile": capture,
            }
        ]
    )
    return attach_economic_state(frame)


def test_economic_state_reconstructs_absolute_remaining_value() -> None:
    out = _row(
        current=-1.0,
        baseline=2.0,
        excess=-0.5,
        capture=0.2,
    ).iloc[0]
    assert out.predicted_remaining_option_value_pct == pytest.approx(1.5)
    assert out.predicted_future_best_base_pct == pytest.approx(0.5)


def test_hybrid_holds_only_when_economic_and_not_captured() -> None:
    row = _row(
        current=-1.0,
        baseline=2.0,
        excess=-0.5,
        capture=0.4,
    ).iloc[0]
    should_exit, reason = _policy_decision(
        row,
        policy="economic_capture",
        capture_threshold=0.7,
    )
    assert should_exit is False
    assert reason == "economic_capture_hold"


def test_hybrid_exits_losing_path_without_predicted_cost_cover() -> None:
    row = _row(
        current=-2.0,
        baseline=1.0,
        excess=0.2,
        capture=0.1,
    ).iloc[0]
    should_exit, reason = _policy_decision(
        row,
        policy="economic_capture",
        capture_threshold=0.7,
    )
    assert should_exit is True
    assert reason == "no_predicted_cost_cover_exit"


def test_hybrid_exits_when_current_state_is_already_high_capture() -> None:
    row = _row(
        current=0.5,
        baseline=1.0,
        excess=0.5,
        capture=0.8,
    ).iloc[0]
    should_exit, reason = _policy_decision(
        row,
        policy="economic_capture",
        capture_threshold=0.7,
    )
    assert should_exit is True
    assert reason == "capture_percentile_exit"


def test_hybrid_exits_when_no_incremental_upside_remains() -> None:
    row = _row(
        current=1.0,
        baseline=0.2,
        excess=-0.5,
        capture=0.1,
    ).iloc[0]
    should_exit, reason = _policy_decision(
        row,
        policy="economic_capture",
        capture_threshold=0.7,
    )
    assert should_exit is True
    assert reason == "no_predicted_incremental_upside"


def test_trajectory_holds_then_exits_on_capture() -> None:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 1.0,
                "exit_now_base_return_pct": -0.5,
                "next_minute_base_return_pct": 0.2,
                "predicted_remaining_option_value_pct": 1.0,
                "predicted_future_best_base_pct": 0.5,
                "predicted_capture_percentile": 0.3,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": 2.0,
                "exit_now_base_return_pct": 0.2,
                "next_minute_base_return_pct": 0.1,
                "predicted_remaining_option_value_pct": 0.5,
                "predicted_future_best_base_pct": 0.7,
                "predicted_capture_percentile": 0.8,
            },
        ]
    )
    trajectory, coverage = build_economic_trajectories(
        frame,
        policy="economic_capture",
        capture_threshold=0.7,
    )
    row = trajectory.iloc[0]
    assert row.status == "completed"
    assert row.exit_reason == "capture_percentile_exit"
    assert row.base_net_return_pct == pytest.approx(0.2)
    assert row.minutes_held == pytest.approx(2.0)
    assert row.hold_decisions == 1
    assert coverage["start_state_coverage"] == pytest.approx(1.0)
