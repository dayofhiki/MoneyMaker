from __future__ import annotations

import pandas as pd

from victory_trader.hierarchical_attention_runtime import (
    EVAL_DAYS,
    HOT_BUDGET,
    _hot_readiness_metrics,
    _prior_population_capture,
    _strict_hot_metrics,
    build_learned_runtime_trace,
)


def test_eval_days_are_fresh_post_v05_sessions():
    assert EVAL_DAYS == [
        "2026-02-10",
        "2026-02-11",
        "2026-02-12",
        "2026-02-13",
        "2026-02-17",
    ]


def test_runtime_never_exceeds_hot_budget_and_prefers_incumbent_on_tie():
    focus = pd.DataFrame(
        [
            {
                "trading_day": "2026-02-10",
                "t": t,
                "ticker": f"T{i:02d}",
                "attention_score": 0.8,
                "target_next_cross": 0,
            }
            for t in [60_000, 120_000]
            for i in range(15)
        ]
    )
    candidates = focus.copy()
    candidates["second_rerank_probability"] = 0.5

    trace, audit = build_learned_runtime_trace(focus, candidates)
    hot = trace.loc[trace["state"].eq("hot")]

    assert HOT_BUDGET == 10
    assert hot.groupby("t").size().max() == 10
    first = set(hot.loc[hot["t"].eq(60_000), "ticker"])
    second = set(hot.loc[hot["t"].eq(120_000), "ticker"])
    assert first == second
    assert audit["max_hot_occupancy"] == 10
    assert audit["mean_hot_set_retention"] == 1.0


def test_strict_hot_metrics_do_not_credit_crossing_timestamp_itself():
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-02-10",
                "ticker": "AAA",
                "t": 60_000,
                "runner_cross_now": False,
            },
            {
                "trading_day": "2026-02-10",
                "ticker": "AAA",
                "t": 120_000,
                "runner_cross_now": True,
            },
        ]
    )

    same_minute_only = pd.DataFrame(
        [
            {
                "trading_day": "2026-02-10",
                "ticker": "AAA",
                "t": 120_000,
                "state": "hot",
            }
        ]
    )
    prior_hot = pd.DataFrame(
        [
            {
                "trading_day": "2026-02-10",
                "ticker": "AAA",
                "t": 60_000,
                "state": "hot",
            }
        ]
    )

    missed = _strict_hot_metrics(same_minute_only, scan)
    captured = _strict_hot_metrics(prior_hot, scan)

    assert missed["strict_hot_captures"] == 0
    assert captured["strict_hot_captures"] == 1
    assert captured["median_hot_lead_minutes"] == 1.0
    assert captured["two_minute_early_count"] == 0


def test_hot_readiness_requires_completed_minutes_before_crossing():
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-02-18",
                "ticker": "AAA",
                "t": 180_000,
                "runner_cross_now": True,
            },
            {
                "trading_day": "2026-02-18",
                "ticker": "BBB",
                "t": 180_000,
                "runner_cross_now": True,
            },
        ]
    )
    trace = pd.DataFrame(
        [
            {
                "trading_day": "2026-02-18",
                "ticker": "AAA",
                "t": 120_000,
                "state": "hot",
            },
            {
                "trading_day": "2026-02-18",
                "ticker": "AAA",
                "t": 60_000,
                "state": "hot",
            },
            {
                "trading_day": "2026-02-18",
                "ticker": "BBB",
                "t": 180_000,
                "state": "hot",
            },
        ]
    )

    metrics = _hot_readiness_metrics(trace, scan)

    assert metrics["runner_crossings"] == 2
    assert metrics["hot_within_1m_count"] == 1
    assert metrics["hot_within_2m_count"] == 1
    assert metrics["hot_sustained_last_2m_count"] == 1
    assert metrics["hot_within_1m_rate"] == 0.5


def test_prior_population_capture_decomposes_focus_and_shortlist_loss():
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-02-18",
                "ticker": ticker,
                "t": 120_000,
                "runner_cross_now": True,
            }
            for ticker in ["AAA", "BBB"]
        ]
    )
    population = pd.DataFrame(
        [
            {
                "trading_day": "2026-02-18",
                "ticker": "AAA",
                "t": 60_000,
            }
        ]
    )

    metrics = _prior_population_capture(population, scan)

    assert metrics == {
        "runner_crossings": 2,
        "prior_minute_count": 1,
        "prior_minute_rate": 0.5,
    }
