"""Causal episode selection for the development-only account diagnostic."""

import pandas as pd
import pytest

from victory_trader.causal_second_execution import SecondBar
from victory_trader.second_portfolio_probe import (
    DEVELOPMENT_DAY,
    build_attempts,
    summarize_scenarios,
)

T = 1_800_000_000_000


def test_exact_previous_minute_close_is_required_for_sizing():
    positions = pd.DataFrame([
        {"trading_day": DEVELOPMENT_DAY, "ticker": "A", "hot_t": T},
        {"trading_day": DEVELOPMENT_DAY, "ticker": "A", "hot_t": T},
        {"trading_day": DEVELOPMENT_DAY, "ticker": "B", "hot_t": T},
    ])
    scan = pd.DataFrame([
        {"trading_day": DEVELOPMENT_DAY, "ticker": "A", "t": T - 60_000, "c": 10},
        {"trading_day": DEVELOPMENT_DAY, "ticker": "B", "t": T - 120_000, "c": 10},
    ])
    attempts, audit = build_attempts(
        positions, scan, day=DEVELOPMENT_DAY, session_close_t=T + 30_000,
    )
    assert audit == {
        "episodes": 2, "causal_decision_refs": 1, "missing_decision_refs": 1,
    }
    assert attempts[0].ticker == "A"
    assert attempts[0].decision_price == 10


def test_probe_rejects_unopened_day():
    with pytest.raises(ValueError, match="already-open"):
        build_attempts(
            pd.DataFrame(columns=["trading_day", "ticker", "hot_t"]),
            pd.DataFrame(columns=["trading_day", "ticker", "t", "c"]),
            day="2026-07-01", session_close_t=T + 30_000,
        )


def test_frozen_sensitivity_grid_has_twelve_reference_results():
    attempts, _ = build_attempts(
        pd.DataFrame([{"trading_day": DEVELOPMENT_DAY, "ticker": "A", "hot_t": T}]),
        pd.DataFrame([{
            "trading_day": DEVELOPMENT_DAY, "ticker": "A",
            "t": T - 60_000, "c": 10,
        }]),
        day=DEVELOPMENT_DAY, session_close_t=T + 30_000,
    )
    outputs = summarize_scenarios(
        attempts, {"A": [SecondBar(T, 10, 10, 10, 10)]},
    )
    assert len(outputs) == 12
    assert {(x["scenario"], x["latency_ms"]) for x in outputs} == {
        (scenario, latency) for scenario in ("light", "base", "stress")
        for latency in (0, 1_000, 2_000, 5_000)
    }
    assert all(not x["summary"]["promotion_allowed"] for x in outputs)

