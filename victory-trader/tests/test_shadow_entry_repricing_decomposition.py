import math

import pandas as pd

from victory_trader.attention_replay import MINUTE_MS
from victory_trader.shadow_entry_repricing_decomposition import (
    add_shadow_features,
    build_labels,
)


def test_build_labels_reconstructs_best_base():
    hot_t = 1_000_000
    state_t = hot_t + MINUTE_MS
    positions = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": hot_t,
                "state_t": state_t,
                "minutes_held": 1.0,
            }
        ]
    )
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "t": state_t + MINUTE_MS,
                "o": 100.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "t": state_t + 2 * MINUTE_MS,
                "o": 104.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "t": state_t + 3 * MINUTE_MS,
                "o": 110.0,
            },
        ]
    )
    result = build_labels(positions, scan)
    row = result.iloc[0]
    assert row["delayed_entry_reference_open"] == 100.0
    assert row["oracle_best_exit_open"] == 110.0
    assert math.isclose(
        row["remaining_best_gross_pct"]
        - row["remaining_best_drag_pct"],
        row["remaining_best_base_pct"],
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_shadow_features_are_causal_lags():
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1,
                "minutes_held": 1.0,
                "entry_to_current_close_pct": 1.0,
                "running_max_return_pct": 1.5,
                "running_min_return_pct": -0.5,
                "log_current_close": 0.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1,
                "minutes_held": 2.0,
                "entry_to_current_close_pct": 3.0,
                "running_max_return_pct": 3.5,
                "running_min_return_pct": -0.5,
                "log_current_close": 0.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1,
                "minutes_held": 3.0,
                "entry_to_current_close_pct": 6.0,
                "running_max_return_pct": 6.5,
                "running_min_return_pct": -0.5,
                "log_current_close": 0.0,
            },
        ]
    )
    result = add_shadow_features(frame)
    assert math.isnan(
        result.iloc[0]["shadow_move_1m_pct"]
    )
    assert result.iloc[1]["shadow_move_1m_pct"] == 2.0
    assert result.iloc[2]["shadow_move_1m_pct"] == 3.0
    assert result.iloc[2]["shadow_accel_1m_pct"] == 1.0
    assert (
        result.iloc[2][
            "shadow_running_range_width_pct"
        ]
        == 7.0
    )
