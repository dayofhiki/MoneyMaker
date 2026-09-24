import math

import pandas as pd

from victory_trader.attention_replay import MINUTE_MS
from victory_trader.causal_sequence_right_tail_entry_observability import (
    add_exact_sequence_lags,
)


def test_exact_sequence_lag_requires_exact_minute():
    base_t = 1_000_000
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": base_t,
                "state_t": base_t + MINUTE_MS,
                "minutes_held": 1.0,
                "attention_score": 1.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": base_t,
                "state_t": base_t + 2 * MINUTE_MS,
                "minutes_held": 2.0,
                "attention_score": 2.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "TEST",
                "hot_t": base_t,
                "state_t": base_t + 4 * MINUTE_MS,
                "minutes_held": 4.0,
                "attention_score": 4.0,
            },
        ]
    )
    result = add_exact_sequence_lags(frame)
    row2 = result.loc[
        result["minutes_held"].eq(2.0)
    ].iloc[0]
    row4 = result.loc[
        result["minutes_held"].eq(4.0)
    ].iloc[0]

    assert row2["attention_score_lag1m"] == 1.0
    assert math.isnan(
        row4["attention_score_lag1m"]
    )
    assert row4["attention_score_lag2m"] == 2.0


def test_sequence_lags_do_not_cross_episode():
    base_t = 1_000_000
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": base_t,
                "state_t": base_t + MINUTE_MS,
                "minutes_held": 1.0,
                "attention_score": 7.0,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "BBB",
                "hot_t": base_t,
                "state_t": base_t + 2 * MINUTE_MS,
                "minutes_held": 2.0,
                "attention_score": 9.0,
            },
        ]
    )
    result = add_exact_sequence_lags(frame)
    assert result[
        "attention_score_lag1m"
    ].isna().all()
