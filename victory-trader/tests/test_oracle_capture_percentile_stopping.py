from __future__ import annotations

import pandas as pd

from victory_trader.oracle_capture_percentile_stopping import (
    MIN_COMPLETION_COVERAGE,
    MIN_FRESH_SPEARMAN,
    MIN_START_COVERAGE,
    MODEL_SEED,
    REQUEST_ID,
    THRESHOLDS,
    add_capture_target,
)


def test_request189_contract_is_frozen():
    assert REQUEST_ID == 189
    assert MODEL_SEED == 20261091
    assert THRESHOLDS == (0.70, 0.80, 0.90)
    assert MIN_START_COVERAGE == 0.85
    assert MIN_COMPLETION_COVERAGE == 0.90
    assert MIN_FRESH_SPEARMAN == 0.10


def test_capture_target_ranks_current_exit_within_episode():
    frame = pd.DataFrame(
        {
            "trading_day": ["2026-05-01"] * 3,
            "ticker": ["TEST"] * 3,
            "hot_t": [1000] * 3,
            "minutes_held": [1, 2, 29],
            "exit_now_base_return_pct": [-2.0, 1.0, 0.0],
            "next_minute_base_return_pct": [None, None, 3.0],
        }
    )
    result = add_capture_target(frame).sort_values("minutes_held")
    assert result.iloc[0]["capture_percentile"] == 0.25
    assert result.iloc[1]["capture_percentile"] == 0.75
    assert result.iloc[2]["capture_percentile"] == 0.50
