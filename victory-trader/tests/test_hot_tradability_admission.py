from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.hot_tradability_admission import (
    annotate_hot_market_context,
    attach_persistent_tradability,
    hot_anchor_features,
)


def test_persistent_tradability_uses_best_consecutive_pair() -> None:
    watch = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 60_000,
                "minutes_since_hot": minute,
                "entry_multi_event_value_pct": value,
            }
            for minute, value in [
                (1, -0.5),
                (2, 1.0),
                (3, 0.8),
                (4, -0.2),
                (5, 2.0),
            ]
        ]
    )
    target = attach_persistent_tradability(watch)
    assert len(target) == 1
    row = target.iloc[0]
    assert np.isclose(row["episode_option_max_pct"], 2.0)
    assert np.isclose(
        row["episode_persistent_tradability_pct"],
        0.9,
    )
    assert row["persistent_pair_count"] == 4


def test_hot_context_builds_cross_sectional_ranks_and_breadth() -> None:
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "t": 60_000,
                "o": 100.0,
                "h": 102.0,
                "l": 99.0,
                "c": 101.0,
                "v": 1000.0,
                "n": 100.0,
                "attention_score": 10.0,
                "return_from_previous_close_pct": 6.0,
            },
            {
                "trading_day": "2026-05-01",
                "ticker": "BBB",
                "t": 60_000,
                "o": 100.0,
                "h": 101.0,
                "l": 98.0,
                "c": 99.0,
                "v": 500.0,
                "n": 50.0,
                "attention_score": 5.0,
                "return_from_previous_close_pct": -1.0,
            },
        ]
    )
    context = annotate_hot_market_context(scan)
    aaa = context.loc[context["ticker"].eq("AAA")].iloc[0]
    assert np.isclose(aaa["attention_rank"], 1.0)
    assert np.isclose(aaa["attention_rank_pct"], 0.5)
    assert np.isclose(aaa["market_symbol_count"], 2.0)
    assert np.isclose(aaa["market_positive_share"], 0.5)
    assert np.isclose(aaa["market_gt5_share"], 0.5)


def test_hot_anchor_features_join_only_at_hot_timestamp() -> None:
    positions = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 60_000,
                "minutes_held": 1.0,
            }
        ]
    )
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "t": 60_000,
                "o": 100.0,
                "h": 102.0,
                "l": 99.0,
                "c": 101.0,
                "v": 1000.0,
                "n": 100.0,
                "attention_score": 10.0,
                "return_from_previous_close_pct": 6.0,
            },
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "t": 120_000,
                "o": 101.0,
                "h": 105.0,
                "l": 100.0,
                "c": 104.0,
                "v": 3000.0,
                "n": 300.0,
                "attention_score": 99.0,
                "return_from_previous_close_pct": 9.0,
            },
        ]
    )
    target = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 60_000,
                "episode_option_max_pct": 2.0,
                "episode_persistent_tradability_pct": 1.0,
                "persistent_pair_count": 4,
            }
        ]
    )
    joined = hot_anchor_features(
        positions,
        scan,
        target,
    )
    assert len(joined) == 1
    assert np.isclose(
        joined.iloc[0]["attention_score"],
        10.0,
    )
    assert np.isclose(
        joined.iloc[0]["episode_persistent_tradability_pct"],
        1.0,
    )
