from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.availability_conditioned_entry import (
    build_event_watch_states,
)


def test_event_watch_uses_observed_state_order_not_exact_minutes() -> None:
    rows = []
    hot_t = 1_000_000
    times = [
        hot_t + 40_000,
        hot_t + 95_000,
        hot_t + 155_000,
        hot_t + 250_000,
        hot_t + 390_000,
        hot_t + 450_000,
        hot_t + 510_000,
        hot_t + 570_000,
        hot_t + 630_000,
        hot_t + 690_000,
    ]
    for i, state_t in enumerate(times):
        rows.append(
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": hot_t,
                "state_t": state_t,
                "exit_reference_open": 100.0 + i,
                "log_current_close": np.log(100.0 + i),
                "attention_score": float(i),
                "attention_rank": 1.0,
            }
        )
    states = build_event_watch_states(pd.DataFrame(rows))
    assert states["watch_event_index"].tolist() == [1, 2, 3, 4, 5]
    assert np.isclose(
        states.iloc[1]["elapsed_minutes_since_hot"],
        95_000 / 60_000,
    )
    assert bool(
        states["entry_multi_event_value_pct"].notna().any()
    )


def test_availability_requires_two_evaluable_watch_states() -> None:
    hot_t = 1_000_000
    rows = []
    for i in range(6):
        rows.append(
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": hot_t,
                "state_t": hot_t + (i + 1) * 60_000,
                "exit_reference_open": 100.0 + i,
                "log_current_close": np.log(100.0 + i),
            }
        )
    states = build_event_watch_states(pd.DataFrame(rows))
    assert states["episode_continuation_available"].eq(0).all()
