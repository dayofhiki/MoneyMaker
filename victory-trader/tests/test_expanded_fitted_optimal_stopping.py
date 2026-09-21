from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.expanded_fitted_optimal_stopping import (
    MAX_HOLD_MINUTES,
    MINUTE_MS,
    build_optimal_trajectories,
)


def _state(last_minute: int = 35) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "ticker": "ABC",
                "t": minute * MINUTE_MS,
                "o": 10.0 + 0.1 * minute,
                "c": 10.05 + 0.1 * minute,
            }
            for minute in range(last_minute + 1)
        ]
    )


def _anchor(probability: float = 0.8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "ticker": "ABC",
                "t": 0,
                "opportunity_probability": probability,
            }
        ]
    )


def _scored(predictions: dict[int, float]) -> pd.DataFrame:
    state = _state()
    rows = []
    for minute, prediction in predictions.items():
        row_t = minute * MINUTE_MS
        decision_t = row_t + MINUTE_MS
        price = state.loc[state["t"].eq(decision_t), "o"].iloc[0]
        rows.append(
            {
                "trading_day": "2026-01-02",
                "ticker": "ABC",
                "t": row_t,
                "minutes_held": float(minute),
                "exit_open": price,
                "predicted_stopping_advantage_pct": prediction,
            }
        )
    return pd.DataFrame(rows)


def test_fitted_policy_follows_positive_values_until_first_nonpositive():
    trajectories, decisions = build_optimal_trajectories(
        _anchor(),
        _scored({1: 0.7, 2: 0.2, 3: -0.1, 4: 5.0}),
        _state(),
        "2026-01",
    )
    trade = trajectories.iloc[0]
    assert trade["status"] == "completed"
    assert trade["exit_reason"] == "fitted_value_exit"
    assert trade["holding_minutes"] == 3
    assert trade["hold_decisions"] == 2
    assert decisions["action"].tolist() == ["HOLD", "HOLD", "EXIT"]
    assert 4 not in decisions["minutes_held"].tolist()


def test_fitted_policy_can_reach_exact_30_minute_cap():
    predictions = {
        minute: 1.0 for minute in range(1, MAX_HOLD_MINUTES)
    }
    trajectories, decisions = build_optimal_trajectories(
        _anchor(),
        _scored(predictions),
        _state(),
        "2026-01",
    )
    trade = trajectories.iloc[0]
    assert trade["status"] == "completed"
    assert trade["exit_reason"] == "forced_30m_exact_open"
    assert trade["holding_minutes"] == 30
    assert trade["hold_decisions"] == 29
    assert len(decisions) == 29


def test_missing_reached_state_stops_model_and_uses_gap_exit():
    trajectories, decisions = build_optimal_trajectories(
        _anchor(),
        _scored({1: 0.5, 3: 0.5}),
        _state(),
        "2026-01",
    )
    trade = trajectories.iloc[0]
    assert trade["status"] == "completed"
    assert trade["exit_reason"] == "gap_exit_exact_open"
    assert trade["holding_minutes"] == 2
    assert decisions.iloc[-1]["action"] == "EXIT_PENDING"
    assert decisions.iloc[-1]["reason"] == "missing_exact_decision_state"


def test_nonpositive_value_exits_immediately():
    trajectories, decisions = build_optimal_trajectories(
        _anchor(),
        _scored({1: 0.0, 2: 1.0}),
        _state(),
        "2026-01",
    )
    trade = trajectories.iloc[0]
    assert trade["holding_minutes"] == 1
    assert trade["exit_reason"] == "fitted_value_exit"
    assert decisions["action"].tolist() == ["EXIT"]


def test_missing_model_score_exits_instead_of_inventing_hold():
    rows = _scored({1: 0.5})
    rows["predicted_stopping_advantage_pct"] = np.nan
    trajectories, decisions = build_optimal_trajectories(
        _anchor(),
        rows,
        _state(),
        "2026-01",
    )
    trade = trajectories.iloc[0]
    assert trade["holding_minutes"] == 1
    assert trade["exit_reason"] == "model_score_missing_exit"
    assert decisions["action"].tolist() == ["EXIT"]


def test_gate_below_half_never_enters():
    trajectories, decisions = build_optimal_trajectories(
        _anchor(0.49),
        _scored({1: 100.0}),
        _state(),
        "2026-01",
    )
    assert trajectories.empty
    assert decisions.empty
