from __future__ import annotations

import pandas as pd

from victory_trader.rate_limited_subscription_probe import (
    _retention_given_focus,
    select_rate_limited_active,
)


def test_rate_limited_active_moves_five_slots_toward_new_desired_set():
    rows: list[dict[str, object]] = []
    day = "2026-04-16"
    for timestamp, challenger_first in [(60_000, False), (120_000, True)]:
        for index in range(40):
            if challenger_first:
                ticker = f"C{index:02d}" if index < 20 else f"I{index - 20:02d}"
            else:
                ticker = f"I{index:02d}" if index < 20 else f"C{index - 20:02d}"
            rows.append(
                {
                    "trading_day": day,
                    "ticker": ticker,
                    "t": timestamp,
                    "market_hazard_probability": float(100 - index),
                    "focus_hazard_rank": index + 1,
                    "target_next_cross": 0,
                }
            )
    focus = pd.DataFrame(rows)

    active, desired = select_rate_limited_active(focus)

    first = set(active.loc[active["t"].eq(60_000), "ticker"])
    second = set(active.loc[active["t"].eq(120_000), "ticker"])
    desired_second = set(desired.loc[desired["t"].eq(120_000), "ticker"])

    assert len(first) == len(second) == len(desired_second) == 20
    assert len(second - first) == 5
    assert {f"C{i:02d}" for i in range(5)}.issubset(second)
    assert desired_second == {f"C{i:02d}" for i in range(20)}


def test_retention_given_focus_uses_crossing_intersection_not_ratio():
    day = "2026-04-16"
    scan = pd.DataFrame(
        [
            {
                "trading_day": day,
                "ticker": "A",
                "t": 120_000,
                "runner_cross_now": True,
            },
            {
                "trading_day": day,
                "ticker": "B",
                "t": 120_000,
                "runner_cross_now": True,
            },
            {
                "trading_day": day,
                "ticker": "C",
                "t": 120_000,
                "runner_cross_now": True,
            },
        ]
    )
    focus = pd.DataFrame(
        [
            {"trading_day": day, "ticker": "A", "t": 60_000},
            {"trading_day": day, "ticker": "B", "t": 60_000},
        ]
    )
    active = pd.DataFrame(
        [
            {"trading_day": day, "ticker": "A", "t": 60_000},
            {"trading_day": day, "ticker": "C", "t": 60_000},
        ]
    )

    assert _retention_given_focus(focus, active, scan) == 0.5
