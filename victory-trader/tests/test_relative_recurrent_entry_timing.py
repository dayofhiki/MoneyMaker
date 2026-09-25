from __future__ import annotations

import pandas as pd

from victory_trader.relative_recurrent_entry_timing import (
    _policy_trades,
    attach_relative_advantage,
)


def _states() -> pd.DataFrame:
    rows = []
    for minute, value in enumerate([-1.0, -0.5, 0.8, 0.2, -0.1], start=1):
        rows.append(
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_since_hot": minute,
                "enter_3m_base_pct": value,
            }
        )
    return pd.DataFrame(rows)


def test_attach_relative_advantage_is_local_one_minute_comparison() -> None:
    frame = attach_relative_advantage(_states())
    first = frame.loc[frame["minutes_since_hot"].eq(1)].iloc[0]
    assert first["enter_vs_wait1_advantage_pct"] == -0.5
    third = frame.loc[frame["minutes_since_hot"].eq(3)].iloc[0]
    assert third["enter_vs_wait1_advantage_pct"] == pytest.approx(0.6)


def test_recurrent_policy_waits_until_relative_advantage_turns_positive() -> None:
    frame = _states()
    frame["predicted_relative_advantage_pct"] = [-0.4, -0.2, 0.3, 0.1, 0.0]
    trades, counts = _policy_trades(frame, recurrent=True)
    assert len(trades) == 1
    assert int(trades.iloc[0]["entry_minute_after_hot"]) == 3
    assert counts["wait_actions"] == 2
