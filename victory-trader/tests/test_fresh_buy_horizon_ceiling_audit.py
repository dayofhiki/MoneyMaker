from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.fresh_buy_horizon_ceiling_audit import (
    HORIZONS,
    REQUEST_ID,
    build_episode_ceilings,
)


def test_request187_contract_is_frozen():
    assert REQUEST_ID == 187
    assert HORIZONS == (1, 2, 3, 5, 10, 15, 30)


def test_request187_ceiling_is_monotone_and_includes_minute30():
    rows = []
    for minute, current, next_value in (
        (1, -1.0, np.nan),
        (2, 0.5, np.nan),
        (3, -0.2, np.nan),
        (5, 1.2, np.nan),
        (10, 0.8, np.nan),
        (15, 2.0, np.nan),
        (29, 1.5, 3.0),
    ):
        rows.append(
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": minute,
                "exit_now_base_return_pct": current,
                "next_minute_base_return_pct": next_value,
            }
        )
    result = build_episode_ceilings(pd.DataFrame(rows)).iloc[0]
    assert result["oracle_1m_base_pct"] == -1.0
    assert result["oracle_2m_base_pct"] == 0.5
    assert result["oracle_3m_base_pct"] == 0.5
    assert result["oracle_5m_base_pct"] == 1.2
    assert result["oracle_15m_base_pct"] == 2.0
    assert result["oracle_30m_base_pct"] == 3.0
