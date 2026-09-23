from __future__ import annotations

import pandas as pd

from victory_trader.bounded_turnover_shortlist_probe import (
    evaluate,
    select_shortlist,
)


def _focus_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    day = "2026-04-09"
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
                    "target_next_cross": int(
                        timestamp == 120_000 and ticker == "C00"
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_bounded_selector_recovers_high_rank_challenger_without_full_churn():
    focus = _focus_rows()

    old = select_shortlist(focus, bounded_turnover=False)
    new = select_shortlist(focus, bounded_turnover=True)

    old_second = set(old.loc[old["t"].eq(120_000), "ticker"])
    new_second = set(new.loc[new["t"].eq(120_000), "ticker"])
    first = set(new.loc[new["t"].eq(60_000), "ticker"])

    assert "C00" not in old_second
    assert "C00" in new_second
    assert len(new_second - first) == 5


def test_selector_evaluation_can_pass_layer_gate():
    days = [
        "2026-04-09",
        "2026-04-10",
        "2026-04-13",
        "2026-04-14",
        "2026-04-15",
    ]
    scan_rows = []
    focus_rows = []
    old_rows = []
    new_rows = []
    for day in days:
        for index in range(10):
            ticker = f"T{index:02d}"
            scan_rows.append(
                {
                    "trading_day": day,
                    "ticker": ticker,
                    "t": 180_000,
                    "runner_cross_now": True,
                }
            )
            for timestamp in [60_000, 120_000]:
                focus_rows.append(
                    {
                        "trading_day": day,
                        "ticker": ticker,
                        "t": timestamp,
                    }
                )
                new_rows.append(
                    {
                        "trading_day": day,
                        "ticker": ticker,
                        "t": timestamp,
                    }
                )
                if index < 9:
                    old_rows.append(
                        {
                            "trading_day": day,
                            "ticker": ticker,
                            "t": timestamp,
                        }
                    )

    result = evaluate(
        pd.DataFrame(scan_rows),
        pd.DataFrame(focus_rows),
        pd.DataFrame(old_rows),
        pd.DataFrame(new_rows),
    )

    assert result["bounded_conditional_on_focus_rate"] == 1.0
    assert result["selector_gate_pass"] is True
