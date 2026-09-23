from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.hot_opportunity_coverage_diagnostic import summarize


def test_coverage_diagnostic_splits_missing_causes():
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-15",
                "oracle_best_base_pct": 1.0,
                "entry_reference_available": True,
                "minutes_to_close": 20.0,
                "current_price": 2.0,
                "active_seconds_60": 60.0,
            },
            {
                "trading_day": "2026-05-15",
                "oracle_best_base_pct": np.nan,
                "entry_reference_available": False,
                "minutes_to_close": 0.5,
                "current_price": 1.0,
                "active_seconds_60": 1.0,
            },
            {
                "trading_day": "2026-05-15",
                "oracle_best_base_pct": np.nan,
                "entry_reference_available": True,
                "minutes_to_close": 8.0,
                "current_price": 0.5,
                "active_seconds_60": 4.0,
            },
        ]
    )

    result = summarize(frame)

    day = result["by_day"]["2026-05-15"]
    assert day["missing"] == 2
    assert day["missing_entry_reference"] == 1
    assert day["entry_but_no_future_exit"] == 1
    assert result["coverage_by_minutes_to_close_bucket"]["<=1"]["missing"] == 1
    assert result["coverage_by_minutes_to_close_bucket"]["(5,10]"]["missing"] == 1
