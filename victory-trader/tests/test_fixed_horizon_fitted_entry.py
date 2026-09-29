import numpy as np
import pandas as pd

from victory_trader.fixed_horizon_fitted_entry import (
    _fixed_terminal_value,
    _next_state_columns,
    _run_policy,
)


def test_fixed_terminal_does_not_move_with_delayed_entry():
    second = 1_000
    hot_t = 1_000_000
    terminal = hot_t + 10 * second
    seconds = pd.DataFrame({
        "t": [hot_t + i * second for i in range(12)],
        "o": [100.0] * 12,
        "h": [100.0] * 12,
        "l": [99.0] * 12,
        "c": [100.0, 99.0, 98.0, 101.0, 103.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 120.0],
    })
    early = _fixed_terminal_value(
        seconds,
        execution_t=hot_t + second,
        execution_price=99.0,
        terminal_t=terminal,
    )
    late = _fixed_terminal_value(
        seconds,
        execution_t=hot_t + 5 * second,
        execution_price=105.0,
        terminal_t=terminal,
    )
    assert early["complete"]
    assert late["complete"]
    # The 120 print after the shared terminal cannot benefit the late entry.
    assert np.isclose(early["max_return_pct"], (105.0 / 99.0 - 1.0) * 100.0)
    assert np.isclose(late["max_return_pct"], 0.0)


def test_next_state_target_is_one_step_not_future_best():
    episode = pd.DataFrame({
        "trading_day": ["2026-05-05"] * 4,
        "ticker": ["ABC"] * 4,
        "hot_t": [1] * 4,
        "decision_t": [1, 2, 3, 4],
        "entry_utility_fixed_pct": [2.0, 1.0, 9.0, 3.0],
    })
    result = _next_state_columns(episode)
    assert np.isclose(result.iloc[0].next_entry_utility_fixed_pct, 1.0)
    assert np.isclose(result.iloc[0].one_step_wait_advantage_pct, -1.0)
    assert np.isnan(result.iloc[-1].next_entry_utility_fixed_pct)


def test_policy_can_skip_instead_of_forcing_entry():
    scored = pd.DataFrame({
        "trading_day": ["2026-05-05", "2026-05-05"],
        "ticker": ["ABC", "ABC"],
        "hot_t": [1, 1],
        "decision_t": [1, 2],
        "execution_price": [100.0, 99.0],
        "entry_utility_fixed_pct": [-2.0, -1.0],
        "entry_mfe_fixed_pct": [1.0, 1.0],
        "entry_mae_fixed_pct": [-3.0, -2.0],
        "predicted_enter_utility_pct": [-1.0, -0.5],
        "predicted_wait_value_pct": [0.0, 0.0],
    })
    decisions = _run_policy(
        scored,
        action_margin=0.1,
        enter_floor=0.0,
    )
    assert len(decisions) == 1
    assert decisions.iloc[0].action == "SKIP"
    assert decisions.iloc[0].realized_utility_pct == 0.0
