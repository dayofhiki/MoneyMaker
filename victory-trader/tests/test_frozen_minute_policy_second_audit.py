import numpy as np
import pandas as pd

from victory_trader import frozen_minute_policy_second_audit as r259


def test_request259_contract():
    assert r259.REQUEST_ID == 259
    assert r259.PRIMARY_LATENCY_MS == 1000
    assert r259.EXPIRY_MS == 5000
    assert r259.EVENT_HORIZONS == (1, 2, 3, 5)
    assert r259.MIN_ENTRY_COVERAGE == 0.80
    assert r259.MIN_EXIT_COVERAGE == 0.70
    assert r259.MIN_FULL_RESOLUTION == 0.60


def test_anchor_times_are_precommitted_not_best_future():
    lookup = {
        ("2026-05-11", "TEST"): [
            (60_000, 10.0),
            (120_000, 9.9),
            (180_000, 10.2),
            (240_000, 9.8),
            (300_000, 11.0),
            (360_000, 8.0),
            (420_000, 12.0),
        ]
    }
    anchors = r259.anchor_times(
        lookup,
        trading_day="2026-05-11",
        ticker="TEST",
        hot_t=0,
        entry_t=60_000,
    )
    assert anchors == (120_000, 180_000, 240_000, 360_000)


def test_reconstruct_keeps_missing_second_reference_unresolved():
    policy = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "entered": True,
                "entry_t": 60_000,
                "entry_price": 10.0,
                "value_pct": 1.0,
            }
        ]
    )
    lookup = {
        ("2026-05-11", "TEST"): [
            (60_000, 10.0),
            (120_000, 10.1),
            (180_000, 10.2),
            (240_000, 10.3),
            (300_000, 10.4),
            (360_000, 10.5),
        ]
    }
    paths = {
        ("2026-05-11", "TEST"): (
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
        )
    }
    rows = r259.reconstruct(
        policy,
        lookup,
        paths,
        latency_ms=1000,
    )
    assert rows.loc[0, "execution_status"] == "entry_reference_unavailable"
    assert pd.isna(rows.loc[0, "reconstructed_value_pct"])


def test_cash_is_known_zero():
    policy = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "entered": False,
                "entry_t": None,
                "entry_price": None,
                "value_pct": np.nan,
            }
        ]
    )
    rows = r259.reconstruct(
        policy,
        {},
        {},
        latency_ms=1000,
    )
    assert rows.loc[0, "execution_status"] == "cash"
    assert rows.loc[0, "reconstructed_value_pct"] == 0.0
