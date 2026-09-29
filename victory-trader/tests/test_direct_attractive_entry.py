import pandas as pd

from victory_trader.direct_attractive_entry import _policy


def test_direct_policy_enters_first_state_above_threshold():
    scored = pd.DataFrame({
        "trading_day": ["2026-05-05"] * 3,
        "ticker": ["ABC"] * 3,
        "hot_t": [1] * 3,
        "decision_t": [1, 2, 3],
        "execution_price": [100.0, 98.0, 99.0],
        "entry_utility_fixed_pct": [1.0, 3.0, 2.0],
        "entry_mfe_fixed_pct": [4.0, 6.0, 5.0],
        "entry_mae_fixed_pct": [-3.0, -3.0, -3.0],
        "predicted_enter_utility_pct": [0.5, 2.0, 3.0],
    })
    decisions = _policy(scored, threshold=1.5)
    row = decisions.iloc[0]
    assert row.action == "ENTER"
    assert row.chosen_price == 98.0
    assert row.delay_s == 0.001


def test_direct_policy_skips_if_no_state_is_attractive():
    scored = pd.DataFrame({
        "trading_day": ["2026-05-05"] * 2,
        "ticker": ["ABC"] * 2,
        "hot_t": [1] * 2,
        "decision_t": [1000, 2000],
        "execution_price": [100.0, 99.0],
        "entry_utility_fixed_pct": [-1.0, -0.5],
        "entry_mfe_fixed_pct": [1.0, 1.2],
        "entry_mae_fixed_pct": [-2.0, -1.7],
        "predicted_enter_utility_pct": [0.1, 0.2],
    })
    decisions = _policy(scored, threshold=1.0)
    assert decisions.iloc[0].action == "SKIP"
    assert decisions.iloc[0].realized_utility_pct == 0.0
