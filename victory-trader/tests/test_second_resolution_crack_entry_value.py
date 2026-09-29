import numpy as np
import pandas as pd

from victory_trader.second_resolution_crack_entry_value import (
    _execution_reference,
    _future_excursion,
    _path_features,
)


def test_second_execution_reference_and_future_excursion():
    t0 = 1_000_000
    sec = 1_000
    seconds = pd.DataFrame({
        "t": [t0, t0 + sec, t0 + 5 * sec, t0 + (60 * 60 + 1) * sec],
        "o": [100.0, 99.0, 98.0, 110.0],
        "h": [101.0, 100.0, 105.0, 112.0],
        "l": [99.0, 97.0, 97.5, 109.0],
        "c": [100.5, 98.5, 104.0, 111.0],
        "v": [10.0, 10.0, 10.0, 10.0],
        "n": [2.0, 2.0, 2.0, 2.0],
    })
    execution = _execution_reference(seconds, t0 + sec)
    assert execution is not None
    fill_t, fill_price, lag = execution
    assert fill_t == t0 + sec
    assert fill_price == 99.0
    assert lag == 0.0

    target = _future_excursion(seconds, fill_t, fill_price)
    assert target["complete"]
    assert np.isclose(target["mfe_close_pct"], (111.0 / 99.0 - 1.0) * 100.0)
    assert np.isclose(target["mfe_high_pct"], (112.0 / 99.0 - 1.0) * 100.0)
    assert np.isclose(target["mae_low_pct"], (97.0 / 99.0 - 1.0) * 100.0)


def test_path_features_use_only_completed_seconds():
    t0 = 2_000_000
    sec = 1_000
    seconds = pd.DataFrame({
        "t": [t0, t0 + sec, t0 + 2 * sec, t0 + 3 * sec],
        "c": [100.0, 98.0, 101.0, 50.0],
    })
    features = _path_features(
        seconds,
        pullback_t=t0,
        decision_t=t0 + 3 * sec,
        pullback_price=100.0,
    )
    # decision_t second (50) is not completed and must not be consumed.
    assert np.isclose(features["post_pullback_min_close_vs_pullback_pct"], -2.0)
    assert np.isclose(features["post_pullback_max_close_vs_pullback_pct"], 1.0)
    assert np.isclose(features["last_completed_close_vs_pullback_pct"], 1.0)
