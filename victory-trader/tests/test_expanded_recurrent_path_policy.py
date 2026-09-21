from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.execution_costs import DEFAULT_EXECUTION_SCENARIOS
from victory_trader.expanded_recurrent_path_policy import (
    MAX_HOLD_MINUTES,
    MINUTE_MS,
    account_replay,
    build_always_hold_comparator,
    build_recurrent_trajectories,
)


def _state(last_minute: int = 35) -> pd.DataFrame:
    rows = []
    for minute in range(last_minute + 1):
        rows.append(
            {
                "trading_day": "2026-01-02",
                "ticker": "ABC",
                "t": minute * MINUTE_MS,
                "o": 10.0 + minute * 0.1,
                "c": 10.05 + minute * 0.1,
            }
        )
    return pd.DataFrame(rows)


def _anchor() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "ticker": "ABC",
                "t": 0,
                "opportunity_probability": 0.8,
            }
        ]
    )


def _scored(probabilities: dict[int, float]) -> pd.DataFrame:
    state = _state()
    rows = []
    for minute, probability in probabilities.items():
        row_t = minute * MINUTE_MS
        decision_t = row_t + MINUTE_MS
        decision = state.loc[state["t"].eq(decision_t)].iloc[0]
        rows.append(
            {
                "trading_day": "2026-01-02",
                "ticker": "ABC",
                "t": row_t,
                "minutes_held": float(minute),
                "exit_open": decision["o"],
                "hold_probability": probability,
            }
        )
    return pd.DataFrame(rows)


def test_recurrent_path_stops_after_first_classifier_exit():
    recurrent, decisions = build_recurrent_trajectories(
        _anchor(),
        _scored({1: 0.8, 2: 0.4, 3: 0.9}),
        _state(),
        "2026-01",
    )
    trade = recurrent.iloc[0]
    assert trade["status"] == "completed"
    assert trade["exit_reason"] == "classifier_exit"
    assert trade["holding_minutes"] == 2
    assert trade["hold_decisions"] == 1
    assert decisions["action"].tolist() == ["HOLD", "EXIT"]
    assert 3 not in decisions["minutes_held"].tolist()


def test_missing_reached_state_forces_first_available_open_exit():
    scored = _scored({1: 0.8, 3: 0.8})
    recurrent, decisions = build_recurrent_trajectories(
        _anchor(),
        scored,
        _state(),
        "2026-01",
    )
    trade = recurrent.iloc[0]
    assert trade["exit_reason"] == "gap_exit_exact_open"
    assert trade["holding_minutes"] == 2
    assert decisions.iloc[-1]["reason"] == "missing_exact_decision_state"
    assert decisions.iloc[-1]["action"] == "EXIT_PENDING"


def test_recurrent_path_can_hold_exactly_to_frozen_30_minute_cap():
    probabilities = {
        minute: 0.9
        for minute in range(1, MAX_HOLD_MINUTES)
    }
    recurrent, decisions = build_recurrent_trajectories(
        _anchor(),
        _scored(probabilities),
        _state(),
        "2026-01",
    )
    trade = recurrent.iloc[0]
    assert trade["status"] == "completed"
    assert trade["exit_reason"] == "forced_30m_exact_open"
    assert trade["holding_minutes"] == 30
    assert trade["hold_decisions"] == 29
    assert len(decisions) == 29


def test_always_hold_comparator_uses_same_entry_and_30_minute_cap():
    recurrent, _ = build_recurrent_trajectories(
        _anchor(),
        _scored({1: 0.4}),
        _state(),
        "2026-01",
    )
    comparator = build_always_hold_comparator(
        recurrent,
        _state(),
        "2026-01",
    )
    trade = comparator.iloc[0]
    assert trade["entry_t"] == MINUTE_MS
    assert trade["holding_minutes"] == 30
    assert trade["exit_reason"] == "always_hold_30m_exact_open"


def test_account_replay_closes_completed_path_and_applies_full_costs():
    recurrent, _ = build_recurrent_trajectories(
        _anchor(),
        _scored({1: 0.4}),
        _state(),
        "2026-01",
    )
    accounts = account_replay(
        recurrent,
        _state(),
        "recurrent_v38",
        "2026-01",
    )
    assert set(accounts["scenario"]) == {
        scenario.name for scenario in DEFAULT_EXECUTION_SCENARIOS
    }
    assert accounts["complete_accounting"].all()
    assert accounts["accepted"].eq(1).all()
    assert accounts["closed"].eq(1).all()
    assert accounts["unresolved"].eq(0).all()
    assert np.isfinite(accounts["marked_return_pct"]).all()


def test_missing_entry_is_never_invented():
    state = _state()
    state = state.loc[~state["t"].eq(MINUTE_MS)].copy()
    recurrent, decisions = build_recurrent_trajectories(
        _anchor(),
        _scored({1: 0.8}),
        state,
        "2026-01",
    )
    trade = recurrent.iloc[0]
    assert trade["status"] == "missing_entry"
    assert trade["exit_reason"] == "missing_exact_entry_open"
    assert np.isnan(trade["base_net_return_pct"])
    assert decisions.empty
