import numpy as np
import pandas as pd

from victory_trader.local_turn_entry import (
    _policy,
    attach_local_turn_features,
)


def test_turn_features_are_past_only_and_detect_bounce():
    frame = pd.DataFrame({
        "trading_day": ["2026-05-05"] * 5,
        "ticker": ["ABC"] * 5,
        "hot_t": [1] * 5,
        "decision_t": [1000, 2000, 3000, 4000, 5000],
        "execution_t": [2000, 3000, 4000, 5000, 20000],
        "execution_price": [100.0, 99.0, 98.0, 99.0, 101.0],
        "last_active_return_pct": [-1.0, -1.0, -1.0, 1.0, 2.0],
        "last_close_vs_hot_pct": [0.0, -1.0, -2.0, -1.0, 1.0],
        "inter_event_gap_s": [1.0] * 5,
        "bounce_from_active_low_pct": [0.0, 0.0, 0.0, 1.0, 3.0],
        "seconds_since_active_low": [0.0, 0.0, 0.0, 1.0, 15.0],
        "current_sec_volume_burst_5": [1.0] * 5,
        "current_sec_accel_5_pct": [-1.0, -1.0, -1.0, 2.0, 2.0],
        "current_sec_signed_volume_imbalance_60": [-0.5] * 3 + [0.5, 0.7],
    })
    result = attach_local_turn_features(frame)
    assert result.iloc[2].turn_negative_streak == 3.0
    assert result.iloc[3].turn_positive_streak == 1.0
    assert result.iloc[3].turn_bounce_recent_low_5 == 1.0
    assert result.iloc[3].turn_return_accel_3 > 0
    # The 20-second future execution can label the first states, but cannot
    # change their causal turn features.
    assert np.isclose(result.iloc[0].turn_return_sum_3, -1.0)


def test_policy_supports_turn_and_strong_value_paths():
    scored = pd.DataFrame({
        "trading_day": ["2026-05-05"] * 3,
        "ticker": ["ABC"] * 3,
        "hot_t": [1] * 3,
        "decision_t": [1000, 2000, 3000],
        "execution_price": [100.0, 98.0, 99.0],
        "entry_utility_fixed_pct": [1.0, 3.0, 2.0],
        "entry_mfe_fixed_pct": [4.0, 6.0, 5.0],
        "entry_mae_fixed_pct": [-3.0, -3.0, -3.0],
        "predicted_enter_utility_pct": [1.0, 2.0, 5.0],
        "predicted_turn_15s_pct": [-1.0, 1.0, -1.0],
    })
    decisions = _policy(
        scored,
        utility_threshold=1.5,
        turn_threshold=0.5,
        override_threshold=4.0,
    )
    row = decisions.iloc[0]
    assert row.action == "ENTER"
    assert row.entry_path == "TURN_CONFIRMED"
    assert row.chosen_price == 98.0
