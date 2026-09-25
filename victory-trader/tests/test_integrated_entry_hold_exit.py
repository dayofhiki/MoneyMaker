from __future__ import annotations

import pandas as pd
import pytest

from victory_trader.integrated_entry_hold_exit import (
    reanchor_positions,
)


def test_reanchor_positions_resets_entry_and_hold_labels() -> None:
    rows = []
    for minute, exit_open, close in [
        (1, 10.0, 10.0),
        (2, 9.5, 9.6),
        (3, 9.8, 9.9),
        (4, 10.2, 10.1),
    ]:
        rows.append(
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": float(minute),
                "state_t": 1000 + minute * 60000,
                "exit_reference_open": exit_open,
                "log_current_close": __import__("numpy").log(close),
                "entry_actual_t": 1000,
                "entry_open": 10.0,
            }
        )
    frame = pd.DataFrame(rows)
    entries = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "entry_minute_after_hot": 2,
                "realized_base_return_pct": 0.0,
            }
        ]
    )
    out = reanchor_positions(frame, entries)
    assert list(out["minutes_held"]) == [1.0, 2.0]
    assert out.iloc[0]["entry_open"] == pytest.approx(9.5)
    assert out.iloc[0]["entry_to_current_close_pct"] == pytest.approx(
        (9.9 / 9.5 - 1.0) * 100.0
    )
    assert pd.notna(out.iloc[0]["hold_advantage_1m_pct"])
