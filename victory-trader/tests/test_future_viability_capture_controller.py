from __future__ import annotations

import pandas as pd

from victory_trader.future_viability_capture_controller import (
    CAPTURE_THRESHOLD,
    MAX_HOLD_MINUTES,
    MIN_POSITIVE_DAYS,
    REFERENCE_MAX_POSITIONS,
    REFERENCE_POSITION_FRACTION,
    REFERENCE_START_KRW,
    REQUEST_ID,
    _decision,
    simulate_reference_account,
)


def test_request192_contract_is_frozen():
    assert REQUEST_ID == 192
    assert CAPTURE_THRESHOLD == 0.70
    assert MAX_HOLD_MINUTES == 30
    assert REFERENCE_START_KRW == 1_000_000.0
    assert REFERENCE_MAX_POSITIONS == 5
    assert REFERENCE_POSITION_FRACTION == 0.20
    assert MIN_POSITIVE_DAYS == 4


def test_negative_state_holds_only_with_high_recovery_confidence():
    row = pd.Series(
        {
            "marked_base_return_pct": -1.0,
            "future_cost_cover_probability": 0.95,
            "predicted_best_future_base_return_pct": 2.0,
            "predicted_capture_percentile": 0.20,
        }
    )
    should_exit, reason = _decision(row, probability_gate=0.90)
    assert not should_exit
    assert reason == "negative_recovery_hold"

    row["future_cost_cover_probability"] = 0.80
    should_exit, reason = _decision(row, probability_gate=0.90)
    assert should_exit
    assert reason == "negative_recovery_failed_exit"


def test_positive_state_exits_at_capture_threshold():
    row = pd.Series(
        {
            "marked_base_return_pct": 1.0,
            "future_cost_cover_probability": 0.99,
            "predicted_best_future_base_return_pct": 3.0,
            "predicted_capture_percentile": 0.80,
        }
    )
    should_exit, reason = _decision(row, probability_gate=0.90)
    assert should_exit
    assert reason == "positive_capture_exit"


def test_reference_account_compounds_completed_trade():
    trajectories = pd.DataFrame(
        {
            "trading_day": ["2026-06-23"],
            "ticker": ["TEST"],
            "hot_t": [1_000_000],
            "status": ["completed"],
            "base_net_return_pct": [10.0],
            "minutes_held": [1.0],
        }
    )
    result = simulate_reference_account(trajectories)
    assert result["starting_balance_krw"] == 1_000_000.0
    assert result["admitted_trades"] == 1
    assert result["ending_balance_krw"] == 1_020_000.0
    assert result["total_return_pct"] == 2.0
