from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.recurrent_wait_entry_action_value import (
    EPISODE_KEYS,
    _run_policy,
    build_watch_states,
)


def _position_rows() -> pd.DataFrame:
    rows = []
    for minute in range(1, 9):
        rows.append(
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_held": float(minute),
                "log_current_close": float(np.log(1.0 + minute * 0.01)),
                "exit_reference_open": 1.0 + minute * 0.01,
                "minutes_since_open": 30.0 + minute,
                "minutes_to_close": 300.0 - minute,
            }
        )
    return pd.DataFrame(rows)


def test_build_watch_states_has_recurrent_labels() -> None:
    frame = build_watch_states(_position_rows())
    assert list(frame["minutes_since_hot"]) == [1, 2, 3, 4, 5]
    assert frame["enter_3m_base_pct"].notna().all()
    assert frame["oracle_wait_value_pct"].notna().all()
    assert frame.attrs["total_episodes"] == 1
    assert set(EPISODE_KEYS).issubset(frame.columns)


def test_recurrent_policy_can_wait_then_enter() -> None:
    frame = build_watch_states(_position_rows())
    frame["predicted_enter_value_pct"] = [-0.2, -0.1, 0.6, 0.1, -0.1]
    frame["predicted_wait_value_pct"] = [0.5, 0.4, 0.2, 0.0, -0.1]
    trades, counts = _run_policy(frame, policy="recurrent_action_value")
    assert len(trades) == 1
    assert int(trades.iloc[0]["entry_minute_after_hot"]) == 3
    assert counts["wait_actions"] == 2
    assert counts["entered"] == 1
