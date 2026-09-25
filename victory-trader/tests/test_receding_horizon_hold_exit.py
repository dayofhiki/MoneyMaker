from __future__ import annotations

import pandas as pd
import pytest

from victory_trader.receding_horizon_hold_exit import (
    TARGET_COLUMN,
    attach_horizon_advantage,
)


def test_attach_horizon_advantage_uses_future_executable_exit() -> None:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": minute,
                "exit_now_base_return_pct": value,
            }
            for minute, value in [
                (1.0, -1.0),
                (2.0, -0.4),
                (3.0, 0.2),
                (4.0, 1.1),
                (5.0, 0.6),
            ]
        ]
    )
    out = attach_horizon_advantage(frame, horizon_minutes=3)
    row1 = out.loc[out["minutes_held"].eq(1.0)].iloc[0]
    row2 = out.loc[out["minutes_held"].eq(2.0)].iloc[0]
    row3 = out.loc[out["minutes_held"].eq(3.0)].iloc[0]

    assert row1[TARGET_COLUMN] == pytest.approx(2.1)
    assert row2[TARGET_COLUMN] == pytest.approx(1.0)
    assert pd.isna(row3[TARGET_COLUMN])
