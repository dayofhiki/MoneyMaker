import pandas as pd
import pytest

from victory_trader.minute_reference_alignment_audit import (
    build_reference_maps,
)
from victory_trader.rich_post_hot_state import attach_minute_features


def raw_scan(future_close=20.0):
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "TEST",
            "bar_start_t": 0,
            "t": 60_000,
            "o": 9.8,
            "h": 10.1,
            "l": 9.7,
            "c": 10.0,
            "v": 100.0,
            "n": 10.0,
            "return_from_previous_close_pct": 1.0,
            "attention_score": 0.2,
            "attention_rank": 5.0,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "TEST",
            "bar_start_t": 60_000,
            "t": 120_000,
            "o": 10.0,
            "h": 10.5,
            "l": 9.9,
            "c": 10.4,
            "v": 250.0,
            "n": 25.0,
            "return_from_previous_close_pct": 5.0,
            "attention_score": 0.7,
            "attention_rank": 2.0,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "TEST",
            "bar_start_t": 120_000,
            "t": 180_000,
            "o": 10.4,
            "h": 25.0,
            "l": 1.0,
            "c": future_close,
            "v": 99999.0,
            "n": 9999.0,
            "return_from_previous_close_pct": 100.0,
            "attention_score": 99.0,
            "attention_rank": 1.0,
        },
    ])


def state():
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "TEST",
            "state_t": 120_000,
        }
    ])


def test_raw_scan_shift_is_exactly_one_minute():
    _, _, audit = build_reference_maps(raw_scan())
    assert audit["exact_60s_shift_rate"] == pytest.approx(1.0)


def test_decision_context_ignores_execution_minute_future():
    left = attach_minute_features(state(), raw_scan(20.0))
    right = attach_minute_features(state(), raw_scan(2.0))
    columns = [
        "current_minute_body_return_pct",
        "current_minute_range_pct",
        "current_log_minute_volume",
        "current_log_minute_transactions",
        "current_attention_score",
        "current_attention_rank",
    ]
    pd.testing.assert_series_equal(
        left.iloc[0][columns],
        right.iloc[0][columns],
        check_names=False,
    )


def test_decision_context_is_the_completed_previous_minute():
    row = attach_minute_features(state(), raw_scan()).iloc[0]
    assert row.current_minute_body_return_pct == pytest.approx(4.0)
    assert row.current_attention_score == pytest.approx(0.7)
    assert row.current_attention_rank == pytest.approx(2.0)
