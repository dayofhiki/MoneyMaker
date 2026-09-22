from __future__ import annotations

import pandas as pd

from victory_trader.targeted_second_hot_reranker import (
    EVAL_DAYS,
    HOT_BUDGET,
    SHORTLIST_BUDGET,
    shortlist_rows,
)


def test_shortlist_keeps_only_top_twenty_per_timestamp():
    rows = []
    for t in [60_000, 120_000]:
        for index in range(30):
            rows.append(
                {
                    "trading_day": "2026-02-03",
                    "t": t,
                    "ticker": f"T{index:02d}",
                    "stage1_hazard_probability": index / 30.0,
                    "target_next_cross": int(index == 29),
                }
            )

    selected = shortlist_rows(pd.DataFrame(rows))

    assert SHORTLIST_BUDGET == 20
    assert HOT_BUDGET == 10
    assert len(selected) == 40
    counts = selected.groupby("t").size().tolist()
    assert counts == [20, 20]
    assert set(selected.loc[selected["t"].eq(60_000), "ticker"]).issuperset(
        {"T29", "T28", "T27"}
    )


def test_eval_days_are_five_fresh_sessions():
    assert EVAL_DAYS == [
        "2026-02-03",
        "2026-02-04",
        "2026-02-05",
        "2026-02-06",
        "2026-02-09",
    ]
    assert SHORTLIST_BUDGET == 2 * HOT_BUDGET
