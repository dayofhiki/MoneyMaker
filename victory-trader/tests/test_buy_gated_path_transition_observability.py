from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.buy_gated_path_transition_observability import (
    PATH_TRANSITION_LAGS,
    RICH_PATH_FEATURES,
    add_exact_path_transitions,
)


def _rows():
    return pd.DataFrame(
        {
            "trading_day": ["2026-06-08"] * 3,
            "ticker": ["ABC"] * 3,
            "hot_t": [1000] * 3,
            "state_t": [61000, 121000, 241000],
            "minutes_held": [1.0, 2.0, 4.0],
            "entry_to_current_close_pct": [1.0, 2.0, 4.0],
            "running_max_return_pct": [1.2, 2.2, 4.2],
            "running_min_return_pct": [-0.5, -0.5, -0.5],
            "drawdown_from_peak_pct": [-0.2, -0.2, -0.2],
            "recovery_from_trough_pct": [1.5, 2.5, 4.5],
        }
    )


def test_request167_uses_frozen_transition_lags():
    assert PATH_TRANSITION_LAGS == (1, 2, 3, 5, 8, 13)
    assert len(RICH_PATH_FEATURES) == 10


def test_request167_exact_lag_never_compresses_gap():
    frame = add_exact_path_transitions(_rows())
    one = "path_transition_1m_path_entry_return_pct"
    two = "path_transition_2m_path_entry_return_pct"
    assert frame.loc[1, one] == 1.0
    assert np.isnan(frame.loc[2, one])
    assert frame.loc[2, two] == 2.0


def test_request167_path_state_is_causal_summary():
    frame = add_exact_path_transitions(_rows())
    assert frame.loc[2, "path_range_pct"] == 4.7
    assert frame.loc[2, "path_observed_fraction"] == 0.75
