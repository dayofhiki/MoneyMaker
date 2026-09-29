from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.event_time_full_hot_tradability import (
    EVENT_HORIZONS,
    RISK_CAP_MINUTES,
    WATCH_EVENT_INDICES,
    WATCH_WINDOW_MINUTES,
    attach_full_hot_tradability,
)


def _scan(prices: dict[int, float]) -> pd.DataFrame:
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


def test_sparse_clock_minutes_still_form_event_time_label() -> None:
    prices = {
        1: 100.0,
        3: 101.0,
        4: 102.0,
        6: 103.0,
        7: 104.0,
        8: 105.0,
        9: 106.0,
        10: 107.0,
        11: 108.0,
        12: 109.0,
    }
    labeled = attach_full_hot_tradability(_hot(), _scan(prices))
    row = labeled.iloc[0]
    assert WATCH_EVENT_INDICES == (1, 2, 3, 4, 5)
    assert WATCH_WINDOW_MINUTES == 5
    assert EVENT_HORIZONS == (1, 2, 3, 5)
    assert bool(row["tradability_evaluable"])
    assert np.isfinite(row["persistent_tradability_pct"])
    assert np.isclose(row["watch_event_1_elapsed_minutes"], 1.0)
    assert np.isclose(row["watch_event_2_elapsed_minutes"], 3.0)


def test_persistent_target_uses_consecutive_observed_watch_events() -> None:
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
        11: 80.0,
    }
    labeled = attach_full_hot_tradability(_hot(), _scan(prices))
    row = labeled.iloc[0]
    checkpoint = pd.to_numeric(
        row[
            [
                f"watch_event_{i}_multi_event_value_pct"
                for i in WATCH_EVENT_INDICES
            ]
        ],
        errors="coerce",
    ).dropna()
    assert len(checkpoint) >= 2
    assert float(row["persistent_tradability_pct"]) <= float(checkpoint.max())


def test_fifth_future_event_beyond_risk_cap_is_unresolved() -> None:
    prices = {
        1: 100.0,
        2: 100.0,
        31: 101.0,
        32: 102.0,
        33: 103.0,
        34: 104.0,
        35: 105.0,
    }
    labeled = attach_full_hot_tradability(_hot(), _scan(prices))
    row = labeled.iloc[0]
    assert RISK_CAP_MINUTES == 30
    assert not bool(row["tradability_evaluable"])
    assert pd.isna(row["persistent_tradability_pct"])


def test_future_after_risk_cap_cannot_change_label() -> None:
    base_prices = {
        1: 100.0,
        2: 101.0,
        3: 102.0,
        4: 103.0,
        5: 104.0,
        6: 105.0,
        7: 106.0,
        8: 107.0,
        9: 108.0,
        10: 109.0,
    }
    changed = dict(base_prices)
    changed[40] = 10000.0
    a = attach_full_hot_tradability(_hot(), _scan(base_prices))
    b = attach_full_hot_tradability(_hot(), _scan(changed))
    assert np.isclose(
        a.iloc[0]["persistent_tradability_pct"],
        b.iloc[0]["persistent_tradability_pct"],
    )
