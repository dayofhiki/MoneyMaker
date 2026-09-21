import numpy as np
import pandas as pd

from victory_trader.expanded_path_expected_continuation import (
    _selected_tail_calibration,
)


def test_selected_tail_calibration_applies_global_bias_and_optimism_allowance():
    calibration = pd.DataFrame(
        {
            "trading_day": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
                "2026-01-05",
            ],
            "hold_advantage_pct": [0.10, 0.20, 0.30, 0.40, 0.50],
        }
    )
    raw = np.array([0.20, 0.30, 0.40, 0.50, 0.60])
    bias, correction, days, mean_optimism, se = _selected_tail_calibration(
        calibration,
        raw,
    )
    assert np.isclose(bias, -0.10)
    assert days == 5
    assert np.isclose(mean_optimism, 0.0)
    assert np.isclose(se, 0.0)
    assert np.isclose(correction, 0.0)


def test_selected_tail_calibration_disables_hold_with_too_few_selected_days():
    calibration = pd.DataFrame(
        {
            "trading_day": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
            ],
            "hold_advantage_pct": [0.10, 0.20, 0.30, 0.40],
        }
    )
    raw = np.array([0.10, 0.20, 0.30, 0.40])
    _bias, correction, days, _mean_optimism, se = _selected_tail_calibration(
        calibration,
        raw,
    )
    assert days == 4
    assert np.isinf(correction)
    assert np.isnan(se)
