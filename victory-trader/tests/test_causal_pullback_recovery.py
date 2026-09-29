import pandas as pd
import pytest

from victory_trader.causal_pullback_recovery import (
    build_recovery_episodes,
    pick_recovery_entry,
)


def test_recovery_requires_a_state_after_activation():
    events = [
        (60_000, 10.0),
        (120_000, 10.4),
        (180_000, 10.19),
    ]
    assert pick_recovery_entry(events, 0.5) is None


def test_recovery_entry_is_first_observed_reclaim():
    events = [
        (60_000, 10.0),
        (120_000, 10.4),
        (180_000, 10.19),
        (240_000, 10.05),
        (300_000, 10.11),
        (360_000, 10.30),
    ]
    entry = pick_recovery_entry(events, 0.5)
    assert entry == (300_000, 10.11)


def test_future_path_cannot_change_an_existing_entry():
    prefix = [
        (60_000, 10.0),
        (120_000, 10.4),
        (180_000, 10.10),
        (240_000, 10.00),
        (300_000, 10.06),
    ]
    left = pick_recovery_entry(prefix, 0.5)
    right = pick_recovery_entry(
        prefix + [
            (360_000, 8.0),
            (420_000, 12.0),
        ],
        0.5,
    )
    assert left == right == (300_000, 10.06)


def test_new_low_moves_recovery_reference_down():
    events = [
        (60_000, 10.0),
        (120_000, 10.4),
        (180_000, 10.10),
        (240_000, 9.90),
        (300_000, 9.94),
        (360_000, 9.96),
    ]
    assert pick_recovery_entry(events, 0.5) == (360_000, 9.96)


def selected():
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "TEST",
            "t": 0,
        }
    ])


def scan(prices):
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "TEST",
            "t": (index + 1) * 60_000,
            "o": price,
        }
        for index, price in enumerate(prices)
    ])


def test_no_confirmed_recovery_stays_cash():
    episodes = build_recovery_episodes(
        selected(),
        scan([10.0, 10.4, 10.1, 10.0, 9.9]),
        recovery_pct=1.0,
    )
    row = episodes.iloc[0]
    assert not row.entered
    assert pd.isna(row.entry_t)


def test_episode_path_starts_strictly_after_entry():
    episodes = build_recovery_episodes(
        selected(),
        scan([10.0, 10.4, 10.0, 10.11, 10.2, 10.3]),
        recovery_pct=1.0,
    )
    row = episodes.iloc[0]
    assert row.entered
    assert row.entry_t == 240_000
    assert row.entry_price == pytest.approx(10.11)
    assert row.path[0][0] == 300_000
