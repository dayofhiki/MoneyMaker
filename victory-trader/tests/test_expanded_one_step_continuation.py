import numpy as np
import pandas as pd

from victory_trader.execution_costs import modeled_sell_fill
from victory_trader.expanded_one_step_continuation import (
    BASE_SCENARIO,
    MINUTE_MS,
    build_continuation_rows,
)


def _state(times):
    rows = []
    for minute, timestamp in enumerate(times):
        rows.append(
            {
                "trading_day": "2026-01-02",
                "ticker": "AAA",
                "t": timestamp,
                "o": 10.0 + minute,
                "trailing_return_1m_pct": float(minute),
                "hod_distance_pct": -float(minute),
                "regular_vwap_distance_pct": float(minute),
                "volume_accel_1m_vs_prior20m": 1.0,
                "bar_range_pct": 1.0,
                "bar_close_location": 0.5,
                "phase": "impulse",
            }
        )
    return pd.DataFrame(rows)


def test_exact_next_opens_define_sunk_cost_hold_advantage():
    state = _state([0, MINUTE_MS, 2 * MINUTE_MS, 3 * MINUTE_MS])
    anchors = pd.DataFrame(
        [{
            "trading_day": "2026-01-02",
            "ticker": "AAA",
            "t": 0,
            "opportunity_probability": 0.8,
            "filing_semantic_accessions_30d": 2.0,
        }]
    )
    result = build_continuation_rows(state, anchors)
    first = result.loc[result["minutes_held"].eq(1)].iloc[0]
    expected = (
        modeled_sell_fill(13.0, BASE_SCENARIO)
        / modeled_sell_fill(12.0, BASE_SCENARIO)
        - 1.0
    ) * 100.0
    assert np.isclose(first["hold_advantage_pct"], expected)
    assert first["evaluation_reason"] == "evaluated"
    assert first["opportunity_probability"] == 0.8
    assert first["filing_semantic_accessions_30d"] == 2.0
    assert first["seq_ret1_lag1"] == 0.0


def test_missing_minute_is_not_skipped_for_exit_or_hold():
    state = _state([0, MINUTE_MS, 3 * MINUTE_MS, 4 * MINUTE_MS])
    anchors = pd.DataFrame(
        [{
            "trading_day": "2026-01-02",
            "ticker": "AAA",
            "t": 0,
        }]
    )
    result = build_continuation_rows(state, anchors)
    first = result.loc[result["minutes_held"].eq(1)].iloc[0]
    assert first["evaluation_reason"] == "missing_exact_exit_open"
    assert pd.isna(first["hold_advantage_pct"])


def test_follow_up_window_excludes_entry_instant_and_thirty_minutes():
    state = _state([minute * MINUTE_MS for minute in range(33)])
    anchors = pd.DataFrame(
        [{"trading_day": "2026-01-02", "ticker": "AAA", "t": 0}]
    )
    result = build_continuation_rows(state, anchors)
    assert result["minutes_held"].min() == 1
    assert result["minutes_held"].max() == 29
    assert len(result) == 29
