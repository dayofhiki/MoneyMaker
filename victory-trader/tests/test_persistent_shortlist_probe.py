from __future__ import annotations

import pandas as pd

from victory_trader.persistent_shortlist_probe import (
    EVAL_DAYS,
    evaluate_persistence,
    select_persistent_shortlist,
    shortlist_audit,
)


def test_persistent_shortlist_retains_incumbents_inside_top30():
    rows: list[dict[str, object]] = []
    for index in range(30):
        rows.append(
            {
                "trading_day": "2026-03-11",
                "t": 60_000,
                "ticker": f"T{index:02d}",
                "market_hazard_probability": float(30 - index),
            }
        )
        rows.append(
            {
                "trading_day": "2026-03-11",
                "t": 120_000,
                "ticker": f"T{index:02d}",
                "market_hazard_probability": float(
                    40 - index if index >= 20 else 20 - index
                ),
            }
        )
    focus = pd.DataFrame(rows)

    selected, audit = select_persistent_shortlist(focus)

    first = set(selected.loc[selected["t"].eq(60_000), "ticker"])
    second = set(selected.loc[selected["t"].eq(120_000), "ticker"])
    assert first == {f"T{index:02d}" for index in range(20)}
    assert second == first
    assert audit["mean_set_retention"] == 1.0
    assert audit["max_occupancy"] == 20


def _evaluation_frames() -> tuple[pd.DataFrame, ...]:
    scan: list[dict[str, object]] = []
    focus: list[dict[str, object]] = []
    stateless: list[dict[str, object]] = []
    persistent: list[dict[str, object]] = []
    for day in EVAL_DAYS:
        for index in range(10):
            ticker = f"T{index:02d}"
            scan.append(
                {
                    "trading_day": day,
                    "ticker": ticker,
                    "t": 180_000,
                    "runner_cross_now": True,
                }
            )
            row = {"trading_day": day, "ticker": ticker, "t": 120_000}
            focus.append(row)
            stateless.append(row)
            persistent.append(row)
    return tuple(pd.DataFrame(rows) for rows in [scan, focus, stateless, persistent])


def test_persistence_gate_requires_observation_demand_reduction():
    scan, focus, stateless, persistent = _evaluation_frames()
    stateless_audit = {
        "ticker_day_requests": 100,
        "mean_set_retention": 0.50,
        "mean_slots_replaced_per_decision": 10.0,
        "max_occupancy": 20,
    }
    persistent_audit = {
        "ticker_day_requests": 70,
        "mean_set_retention": 0.80,
        "mean_slots_replaced_per_decision": 4.0,
        "max_occupancy": 20,
    }

    passed = evaluate_persistence(
        scan,
        focus,
        stateless,
        persistent,
        stateless_audit,
        persistent_audit,
    )
    failed = evaluate_persistence(
        scan,
        focus,
        stateless,
        persistent,
        stateless_audit,
        {**persistent_audit, "ticker_day_requests": 90},
    )

    assert passed["ticker_day_request_reduction_rate"] == 0.30000000000000004
    assert passed["promotion_gate_pass"] is True
    assert failed["promotion_gate_pass"] is False


def test_shortlist_audit_resets_incumbents_between_sessions():
    rows = pd.DataFrame(
        [
            {"trading_day": day, "ticker": "AAA", "t": 60_000}
            for day in EVAL_DAYS[:2]
        ]
    )

    audit = shortlist_audit(rows)

    assert audit["mean_set_retention"] is None
    assert audit["mean_slots_replaced_per_decision"] == 1.0
    assert audit["ticker_day_requests"] == 2
