from __future__ import annotations

import pandas as pd

from victory_trader.opportunity_admission_transition_timing import (
    attach_episode_opportunity,
    minute1_states,
)


def test_episode_opportunity_is_best_candidate_entry() -> None:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_since_hot": 1,
                "enter_3m_base_pct": -1.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_since_hot": 2,
                "enter_3m_base_pct": 0.7,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": 1000,
                "minutes_since_hot": 3,
                "enter_3m_base_pct": 0.2,
            },
        ]
    )
    out = attach_episode_opportunity(frame)
    assert out["episode_best_entry_3m_base_pct"].eq(0.7).all()
    first = minute1_states(out)
    assert len(first) == 1
    assert int(first.iloc[0]["minutes_since_hot"]) == 1
