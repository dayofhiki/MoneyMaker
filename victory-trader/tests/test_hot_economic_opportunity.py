from __future__ import annotations

import pandas as pd

from victory_trader.hot_economic_opportunity import (
    ENTRY_FEATURES,
    _attach_entry_context,
)
from victory_trader.market_calendar import regular_session_bounds


def test_entry_feature_allowlist_excludes_future_economic_labels():
    forbidden = {
        "oracle_best_base_pct",
        "oracle_best_minute",
        "oracle_base_positive",
        "return_1m_base_net_pct",
        "runner_cross_now",
        "target_next_cross",
    }
    assert forbidden.isdisjoint(ENTRY_FEATURES)


def test_attach_entry_context_adds_price_time_rank_and_transport_state():
    day = "2026-05-14"
    bounds = regular_session_bounds(pd.Timestamp(day).date())
    assert bounds is not None
    first_t = int(bounds[0].timestamp() * 1000) + 60_000

    candidates = pd.DataFrame(
        [
            {
                "trading_day": day,
                "ticker": "AAA",
                "t": first_t,
                "second_rerank_probability": 0.8,
                "transport_desired_now": True,
            },
            {
                "trading_day": day,
                "ticker": "BBB",
                "t": first_t,
                "second_rerank_probability": 0.4,
                "transport_desired_now": False,
            },
        ]
    )
    scan = pd.DataFrame(
        [
            {"trading_day": day, "ticker": "AAA", "t": first_t, "c": 2.0},
            {"trading_day": day, "ticker": "BBB", "t": first_t, "c": 4.0},
        ]
    )

    result = _attach_entry_context(candidates, scan)

    aaa = result.loc[result["ticker"].eq("AAA")].iloc[0]
    bbb = result.loc[result["ticker"].eq("BBB")].iloc[0]
    assert aaa["hot_candidate_rank"] == 1.0
    assert bbb["hot_candidate_rank"] == 2.0
    assert aaa["transport_desired_now_numeric"] == 1
    assert bbb["transport_desired_now_numeric"] == 0
    assert aaa["minutes_since_open"] == 1.0
    assert aaa["minutes_to_close"] > 300
    assert aaa["log_current_price"] < bbb["log_current_price"]
