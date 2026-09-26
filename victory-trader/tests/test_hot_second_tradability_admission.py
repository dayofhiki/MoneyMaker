from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.hot_second_tradability_admission import (
    HOT_AUGMENTED_FEATURES,
    HOT_SECOND_FEATURES,
    admitted_key_set,
    enrich_hot_seconds,
)
from victory_trader.hot_tradability_admission import HOT_CONTEXT_FEATURES


class FakeClient:
    def __init__(self, rows):
        self.rows = rows

    def second_bars_range(self, ticker, start, end, adjusted=False):
        return {"results": self.rows}


def _seconds(future_close: float) -> list[dict[str, float]]:
    rows = []
    price = 100.0
    for second in range(1, 62):
        if second <= 60:
            price = 100.0 + 0.01 * second
        else:
            price = future_close
        rows.append(
            {
                "t": second * 1000,
                "o": price - 0.005,
                "h": price + 0.01,
                "l": price - 0.01,
                "c": price,
                "v": 100.0 + second,
                "n": 10.0 + second,
            }
        )
    return rows


def _hot() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": 61_000,
            }
        ]
    )


def test_hot_second_features_ignore_future_second() -> None:
    a, audit_a = enrich_hot_seconds(
        _hot(), FakeClient(_seconds(50.0))
    )
    b, audit_b = enrich_hot_seconds(
        _hot(), FakeClient(_seconds(500.0))
    )
    assert audit_a["any_second_coverage"] == 1.0
    assert audit_b["any_second_coverage"] == 1.0
    for column in HOT_SECOND_FEATURES:
        av = pd.to_numeric(a[column], errors="coerce").iloc[0]
        bv = pd.to_numeric(b[column], errors="coerce").iloc[0]
        if pd.isna(av) and pd.isna(bv):
            continue
        assert np.isclose(float(av), float(bv), equal_nan=True)


def test_augmented_family_is_baseline_plus_seconds() -> None:
    assert set(HOT_CONTEXT_FEATURES).issubset(
        set(HOT_AUGMENTED_FEATURES)
    )
    assert set(HOT_SECOND_FEATURES).issubset(
        set(HOT_AUGMENTED_FEATURES)
    )
    forbidden = {
        "episode_persistent_tradability_pct",
        "entry_multi_event_value_pct",
        "enter_vs_wait_multi_event_advantage_pct",
    }
    assert forbidden.isdisjoint(HOT_AUGMENTED_FEATURES)


def test_admitted_keys_use_only_candidate_boolean() -> None:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": 1,
                "second_admitted": True,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "BBB",
                "hot_t": 2,
                "second_admitted": False,
            },
        ]
    )
    assert admitted_key_set(frame, "second") == {
        ("2026-06-23", "AAA", 1)
    }
