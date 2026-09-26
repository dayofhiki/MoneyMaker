from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.full_hot_tradability import (
    EVENT_HORIZONS,
    WATCH_MINUTES,
    attach_full_hot_tradability,
)


def _scan(
    prices: dict[int, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-06-08",
                "ticker": "AAA",
                "t": minute * 60_000,
                "o": price,
            }
            for minute, price in sorted(prices.items())
        ]
    )


def _hot() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-06-08",
                "ticker": "AAA",
                "t": 0,
            }
        ]
    )


def test_label_uses_fixed_watch_and_event_horizons() -> None:
    prices = {
        minute: 100.0 + minute
        for minute in range(1, 12)
    }
    labeled = attach_full_hot_tradability(
        _hot(),
        _scan(prices),
    )
    row = labeled.iloc[0]
    assert WATCH_MINUTES == (1, 2, 3, 4, 5)
    assert EVENT_HORIZONS == (1, 2, 3, 5)
    assert bool(row["tradability_evaluable"])
    values = [
        row[f"watch_{m}m_multi_event_value_pct"]
        for m in WATCH_MINUTES
    ]
    assert all(np.isfinite(values))
    pair_means = [
        (values[i] + values[i + 1]) / 2.0
        for i in range(len(values) - 1)
    ]
    assert np.isclose(
        row["persistent_tradability_pct"],
        max(pair_means),
    )


def test_single_good_checkpoint_is_not_enough_if_neighbor_bad() -> None:
    # Exact values are less important than the contract: persistent target is
    # a consecutive-pair mean, not max(single checkpoint).
    prices = {
        1: 100.0,
        2: 100.0,
        3: 100.0,
        4: 100.0,
        5: 100.0,
        6: 120.0,
        7: 80.0,
        8: 80.0,
        9: 80.0,
        10: 80.0,
    }
    labeled = attach_full_hot_tradability(
        _hot(),
        _scan(prices),
    )
    row = labeled.iloc[0]
    checkpoint = pd.to_numeric(
        row[
            [
                f"watch_{m}m_multi_event_value_pct"
                for m in WATCH_MINUTES
            ]
        ],
        errors="coerce",
    ).dropna()
    assert len(checkpoint) >= 2
    assert (
        float(row["persistent_tradability_pct"])
        <= float(checkpoint.max())
    )


def test_missing_exact_future_reference_stays_unresolved() -> None:
    prices = {
        1: 100.0,
        2: 101.0,
        3: 102.0,
        # minute 4 intentionally missing
        5: 104.0,
        6: 105.0,
    }
    labeled = attach_full_hot_tradability(
        _hot(),
        _scan(prices),
    )
    row = labeled.iloc[0]
    assert not bool(row["tradability_evaluable"])
    assert pd.isna(row["persistent_tradability_pct"])


def test_future_after_required_horizon_cannot_change_label() -> None:
    base_prices = {
        minute: 100.0 + minute
        for minute in range(1, 11)
    }
    changed_prices = dict(base_prices)
    changed_prices[20] = 10000.0

    a = attach_full_hot_tradability(
        _hot(),
        _scan(base_prices),
    )
    b = attach_full_hot_tradability(
        _hot(),
        _scan(changed_prices),
    )
    assert np.isclose(
        a.iloc[0]["persistent_tradability_pct"],
        b.iloc[0]["persistent_tradability_pct"],
    )
