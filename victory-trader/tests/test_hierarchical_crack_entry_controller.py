import numpy as np
import pandas as pd

from victory_trader.hierarchical_crack_entry_controller import (
    _attach_wait_option,
    _event_features,
    _minute_crack_target,
)


def test_minute_crack_target_uses_hot_open_and_future_open_path():
    minute = 60_000
    hot_t = 1_000_000
    path = pd.DataFrame({
        "t": [
            hot_t,
            hot_t + 10 * minute,
            hot_t + 60 * minute,
            hot_t + 61 * minute,
        ],
        "o": [100.0, 108.0, 115.0, 114.0],
    })
    target = _minute_crack_target(path, hot_t)
    assert target["complete"]
    assert target["reference_price"] == 100.0
    assert np.isclose(target["mfe_pct"], 15.0)


def test_event_features_only_use_completed_active_prefix():
    second = 1_000
    hot_t = 2_000_000
    active = pd.DataFrame({
        "t": [hot_t, hot_t + second, hot_t + 3 * second, hot_t + 4 * second],
        "c": [100.0, 98.0, 101.0, 50.0],
    })
    features = _event_features(
        active,
        2,
        hot_t=hot_t,
        hot_price=100.0,
    )
    assert np.isclose(features["running_low_vs_hot_pct"], -2.0)
    assert np.isclose(features["running_high_vs_hot_pct"], 1.0)
    assert np.isclose(features["last_close_vs_hot_pct"], 1.0)
    assert np.isclose(features["inter_event_gap_s"], 2.0)


def test_wait_option_is_strict_future_best_entry_value():
    episode = pd.DataFrame({
        "decision_t": [1, 2, 3, 4],
        "entry_value_60m_pct": [4.0, 7.0, 5.0, 6.0],
    })
    result = _attach_wait_option(episode)
    values = result.wait_option_advantage_pct.tolist()
    assert np.isclose(values[0], 3.0)
    assert np.isclose(values[1], -1.0)
    assert np.isclose(values[2], 1.0)
    assert np.isnan(values[3])
