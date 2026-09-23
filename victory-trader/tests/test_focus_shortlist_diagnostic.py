from __future__ import annotations

import pandas as pd

from victory_trader.focus_shortlist_diagnostic import diagnose_selection


def test_diagnostic_identifies_hysteresis_blocking_current_top20_runner():
    rows: list[dict[str, object]] = []
    day = "2026-04-01"

    for index in range(20):
        rows.append(
            {
                "trading_day": day,
                "ticker": f"I{index:02d}",
                "t": 60_000,
                "market_hazard_probability": float(100 - index),
                "focus_hazard_rank": index + 1,
                "target_next_cross": 0,
            }
        )
    for index in range(20):
        rows.append(
            {
                "trading_day": day,
                "ticker": f"C{index:02d}",
                "t": 60_000,
                "market_hazard_probability": float(80 - index),
                "focus_hazard_rank": index + 21,
                "target_next_cross": 0,
            }
        )

    for index in range(20):
        rows.append(
            {
                "trading_day": day,
                "ticker": f"C{index:02d}",
                "t": 120_000,
                "market_hazard_probability": float(100 - index),
                "focus_hazard_rank": index + 1,
                "target_next_cross": 1 if index == 0 else 0,
            }
        )
    for index in range(20):
        rows.append(
            {
                "trading_day": day,
                "ticker": f"I{index:02d}",
                "t": 120_000,
                "market_hazard_probability": float(80 - index),
                "focus_hazard_rank": index + 21,
                "target_next_cross": 0,
            }
        )

    scored = pd.DataFrame(rows)
    scan = pd.DataFrame(
        [
            {
                "trading_day": day,
                "ticker": "C00",
                "t": 180_000,
                "runner_cross_now": True,
            }
        ]
    )

    positives, summary = diagnose_selection(scored, scan)

    positive = positives.loc[positives["ticker"].eq("C00")].iloc[0]
    assert bool(positive["in_focus"]) is True
    assert bool(positive["in_stateless_top20"]) is True
    assert bool(positive["in_hysteresis_top20"]) is False
    assert bool(positive["blocked_current_top20"]) is True
    assert summary["focus"]["prior_minute_count"] == 1
    assert summary["stateless_top20"]["prior_minute_count"] == 1
    assert summary["hysteresis_top20"]["prior_minute_count"] == 0
    assert summary["blocked_current_top20_count"] == 1
