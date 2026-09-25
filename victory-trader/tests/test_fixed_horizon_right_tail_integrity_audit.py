import pandas as pd

from victory_trader.attention_replay import MINUTE_MS
from victory_trader.fixed_horizon_right_tail_integrity_audit import (
    add_fixed20_labels,
)


def test_fixed20_label_is_anchored_to_shadow_state():
    state_t = 2_000_000
    positions = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": state_t - 10 * MINUTE_MS,
                "state_t": state_t,
                "minutes_held": 10.0,
                "minutes_to_close": 100.0,
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
                "t": state_t + 20 * MINUTE_MS,
                "o": 106.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "t": state_t + 22 * MINUTE_MS,
                "o": 130.0,
            },
        ]
    )
    result = add_fixed20_labels(positions, scan)
    row = result.iloc[0]
    assert row["fixed20_entry_open"] == 100.0
    assert row["fixed20_exit_count"] == 1
    assert 3.0 <= row["fixed20_best_base_pct"] < 6.0
    assert bool(row["fixed20_strong_winner_3"])


def test_fixed20_excludes_states_without_full_session_window():
    positions = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1,
                "state_t": 1 + MINUTE_MS,
                "minutes_held": 1.0,
                "minutes_to_close": 19.0,
            }
        ]
    )
    scan = pd.DataFrame(
        columns=["trading_day", "ticker", "t", "o"]
    )
    result = add_fixed20_labels(positions, scan)
    assert result.empty
