from __future__ import annotations

import pandas as pd
import pytest

from victory_trader.hot_entry_economics import (
    first_hot_events,
    label_first_hot_economics,
    label_hot_event_economics,
    promotion_hot_events,
    summarize_hot_economics,
)


def test_first_hot_events_keeps_one_event_per_ticker_day():
    trace = pd.DataFrame(
        [
            {"trading_day": "2026-05-07", "ticker": "A", "t": 60_000, "state": "watch"},
            {"trading_day": "2026-05-07", "ticker": "A", "t": 120_000, "state": "hot"},
            {"trading_day": "2026-05-07", "ticker": "A", "t": 180_000, "state": "hot"},
            {"trading_day": "2026-05-07", "ticker": "B", "t": 120_000, "state": "hot"},
        ]
    )

    result = first_hot_events(trace)

    assert len(result) == 2
    assert int(result.loc[result["ticker"].eq("A"), "t"].iloc[0]) == 120_000


def test_hot_economics_uses_next_bar_open_and_costs():
    trace = pd.DataFrame(
        [
            {"trading_day": "2026-05-07", "ticker": "A", "t": 120_000, "state": "hot"},
        ]
    )
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-07",
                "ticker": "A",
                "t": 180_000,
                "o": 1.00,
                "return_from_previous_close_pct": 5.0,
            },
            {
                "trading_day": "2026-05-07",
                "ticker": "A",
                "t": 240_000,
                "o": 1.10,
                "return_from_previous_close_pct": 15.0,
            },
            {
                "trading_day": "2026-05-07",
                "ticker": "A",
                "t": 300_000,
                "o": 1.20,
                "return_from_previous_close_pct": 25.0,
            },
        ]
    )

    labeled = label_first_hot_economics(trace, scan, horizons=(1, 2))
    row = labeled.iloc[0]

    assert row["entry_price"] == 1.0
    assert row["return_1m_gross_pct"] == pytest.approx(10.0)
    assert row["return_2m_gross_pct"] == pytest.approx(20.0)
    assert row["return_1m_base_net_pct"] < 10.0
    assert row["oracle_best_minute"] == 2
    assert row["oracle_best_base_pct"] > 0


def test_hot_economics_summary_reports_fixed_and_oracle_metrics():
    labeled = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-07",
                "entry_reference_available": True,
                "return_1m_gross_pct": 2.0,
                "return_1m_base_net_pct": 1.0,
                "return_1m_stress_net_pct": -1.0,
                "oracle_best_base_pct": 3.0,
                "oracle_best_minute": 5,
            },
            {
                "trading_day": "2026-05-07",
                "entry_reference_available": True,
                "return_1m_gross_pct": -1.0,
                "return_1m_base_net_pct": -2.0,
                "return_1m_stress_net_pct": -4.0,
                "oracle_best_base_pct": 1.0,
                "oracle_best_minute": 2,
            },
        ]
    )

    summary = summarize_hot_economics(labeled, horizons=(1,))

    assert summary["first_hot_episodes"] == 2
    assert summary["fixed_horizons"]["1"]["base_positive_rate"] == 0.5
    assert summary["oracle_30m"]["base_positive_rate"] == 1.0



def test_promotion_hot_events_keeps_repromotions_as_new_episodes():
    trace = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-07",
                "ticker": "A",
                "t": 60_000,
                "state": "hot",
                "reason": "learned_hot_promote",
                "learned_hot_score": 0.8,
            },
            {
                "trading_day": "2026-05-07",
                "ticker": "A",
                "t": 120_000,
                "state": "hot",
                "reason": "learned_hot_retain",
                "learned_hot_score": 0.7,
            },
            {
                "trading_day": "2026-05-07",
                "ticker": "A",
                "t": 300_000,
                "state": "hot",
                "reason": "learned_hot_promote",
                "learned_hot_score": 0.9,
            },
        ]
    )

    result = promotion_hot_events(trace)

    assert len(result) == 2
    assert result["promotion_index"].tolist() == [1, 2]
    assert result["is_repromotion"].tolist() == [False, True]
    assert result["minutes_since_previous_promotion"].iloc[1] == pytest.approx(4.0)


def test_generic_hot_event_label_preserves_causal_scores():
    events = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-07",
                "ticker": "A",
                "t": 120_000,
                "attention_score": 0.75,
                "learned_hot_score": 0.91,
                "reason": "learned_hot_promote",
                "promotion_index": 2,
                "minutes_since_previous_promotion": 7.0,
                "is_repromotion": True,
            }
        ]
    )
    scan = pd.DataFrame(
        [
            {"trading_day": "2026-05-07", "ticker": "A", "t": 180_000, "o": 1.0},
            {"trading_day": "2026-05-07", "ticker": "A", "t": 240_000, "o": 1.05},
        ]
    )

    labeled = label_hot_event_economics(events, scan, horizons=(1,))
    row = labeled.iloc[0]

    assert row["learned_hot_score"] == pytest.approx(0.91)
    assert row["promotion_index"] == 2
    assert bool(row["is_repromotion"]) is True
