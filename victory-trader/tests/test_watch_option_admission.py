from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.watch_option_admission import (
    admitted_relative_policy,
    attach_episode_watch_option,
)


def _episode() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 1,
                "minutes_since_hot": minute,
                "entry_multi_event_value_pct": value,
                "enter_vs_wait_multi_event_advantage_pct": advantage,
                "predicted_enter_vs_wait_advantage_pct": predicted,
                "enter_3m_base_pct": realized,
            }
            for minute, value, advantage, predicted, realized in [
                (1, -0.5, -0.4, -0.2, -1.0),
                (2, 0.2, -0.1, -0.1, -0.5),
                (3, 0.8, 0.3, 0.1, 1.5),
                (4, 0.4, 0.2, 0.2, 0.8),
                (5, -0.2, -0.2, -0.1, -0.4),
            ]
        ]
    )


def test_episode_watch_option_is_max_multi_event_value() -> None:
    frame = attach_episode_watch_option(_episode())
    assert np.allclose(
        frame["episode_watch_option_value_pct"].to_numpy(dtype=float),
        0.8,
    )


def test_admitted_relative_policy_waits_until_advantage_nonnegative() -> None:
    frame = _episode()
    key = {("2026-05-01", "AAA", 1)}
    trades, counts = admitted_relative_policy(frame, key)
    assert len(trades) == 1
    assert trades.iloc[0]["entry_minute_after_hot"] == 3
    assert np.isclose(
        trades.iloc[0]["realized_base_return_pct"],
        1.5,
    )
    assert counts["wait_actions"] == 2


def test_rejected_episode_never_reaches_timing_policy() -> None:
    frame = _episode()
    trades, counts = admitted_relative_policy(frame, set())
    assert trades.empty
    assert counts["rejected_at_admission"] == 1
    assert counts["entered"] == 0


def test_negative_advantage_through_minute5_abstains() -> None:
    frame = _episode()
    frame["predicted_enter_vs_wait_advantage_pct"] = -0.1
    key = {("2026-05-01", "AAA", 1)}
    trades, counts = admitted_relative_policy(frame, key)
    assert trades.empty
    assert counts["abstained_after_admission"] == 1
    assert counts["wait_actions"] == 5
