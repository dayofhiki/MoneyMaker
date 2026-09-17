from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from victory_trader.events import CrossingEvent
from victory_trader.exits import evaluate_barrier


ET = ZoneInfo("America/New_York")


def bars(rows):
    return pd.DataFrame(rows)


def event():
    return CrossingEvent("TEST", 0, 20.0, 10.0, 8.3333333333)


def market_ts(hour: int, minute: int) -> int:
    return int(datetime(2026, 9, 15, hour, minute, tzinfo=ET).timestamp() * 1000)


def test_take_profit_first():
    frame = bars([
        {"t": 0, "o": 10.0, "h": 10.0, "l": 10.0, "c": 10.0},
        {"t": 60_000, "o": 10.0, "h": 10.25, "l": 9.95, "c": 10.20},
    ])
    result = evaluate_barrier(event(), frame, take_profit_pct=2, stop_loss_pct=1)
    assert result.status == "take_profit"
    assert result.minutes_to_exit == 1
    assert result.exit_return_pct == 2


def test_stop_loss_first():
    frame = bars([
        {"t": 0, "o": 10.0, "h": 10.0, "l": 10.0, "c": 10.0},
        {"t": 60_000, "o": 10.0, "h": 10.05, "l": 9.85, "c": 9.90},
    ])
    result = evaluate_barrier(event(), frame, take_profit_pct=2, stop_loss_pct=1)
    assert result.status == "stop_loss"
    assert result.exit_return_pct == -1


def test_gap_through_stop_uses_observed_open():
    frame = bars([
        {"t": 0, "o": 10.0, "h": 10.0, "l": 10.0, "c": 10.0},
        {"t": 60_000, "o": 9.50, "h": 9.60, "l": 9.40, "c": 9.55},
    ])
    result = evaluate_barrier(event(), frame, take_profit_pct=2, stop_loss_pct=1)
    assert result.status == "stop_gap"
    assert result.exit_return_pct == pytest.approx(-5.0)


def test_same_bar_double_hit_is_ambiguous():
    frame = bars([
        {"t": 0, "o": 10.0, "h": 10.0, "l": 10.0, "c": 10.0},
        {"t": 60_000, "o": 10.0, "h": 10.25, "l": 9.85, "c": 10.10},
    ])
    result = evaluate_barrier(event(), frame, take_profit_pct=2, stop_loss_pct=1)
    assert result.status == "ambiguous"
    assert result.exit_return_pct is None


def test_timeout_requires_exact_clock_horizon_bar():
    frame = bars([
        {"t": 0, "o": 10.0, "h": 10.0, "l": 10.0, "c": 10.0},
        {"t": 60_000, "o": 10.0, "h": 10.10, "l": 9.95, "c": 10.05},
        {"t": 120_000, "o": 10.05, "h": 10.15, "l": 9.95, "c": 10.10},
    ])
    result = evaluate_barrier(event(), frame, take_profit_pct=2, stop_loss_pct=1, max_horizon_minutes=2)
    assert result.status == "timeout"
    assert result.minutes_to_exit == 2
    assert result.exit_return_pct == pytest.approx(1.0)


def test_missing_timeout_bar_is_not_replaced_by_earlier_close():
    frame = bars([
        {"t": 0, "o": 10.0, "h": 10.0, "l": 10.0, "c": 10.0},
        {"t": 60_000, "o": 10.0, "h": 10.05, "l": 9.95, "c": 10.0},
        {"t": 180_000, "o": 10.0, "h": 10.05, "l": 9.95, "c": 10.0},
    ])
    result = evaluate_barrier(event(), frame, take_profit_pct=2, stop_loss_pct=1, max_horizon_minutes=2)
    assert result.status == "unresolved_missing"
    assert result.exit_return_pct is None


def test_regular_session_position_is_forced_out_at_last_regular_bar():
    entry_ts = market_ts(15, 58)
    signal_ts = market_ts(15, 57)
    frame = bars([
        {"t": signal_ts, "o": 10.0, "h": 10.0, "l": 10.0, "c": 10.0},
        {"t": entry_ts, "o": 10.0, "h": 10.05, "l": 9.95, "c": 10.02},
        {"t": market_ts(15, 59), "o": 10.02, "h": 10.10, "l": 9.95, "c": 10.08},
        {"t": market_ts(16, 0), "o": 9.0, "h": 9.1, "l": 8.5, "c": 8.7},
    ])
    signal = CrossingEvent("TEST", signal_ts, 20.0, 10.0, 8.3333333333)
    result = evaluate_barrier(
        signal,
        frame,
        take_profit_pct=5,
        stop_loss_pct=3,
        max_horizon_minutes=60,
        entry_timestamp_ms=entry_ts,
        entry_price=10.0,
        regular_session_only=True,
    )
    assert result.status == "session_close"
    assert result.exit_return_pct == pytest.approx(0.8)
    assert result.minutes_to_exit == 2


def test_missing_last_regular_bar_is_unresolved_not_after_hours_fill():
    entry_ts = market_ts(15, 58)
    signal_ts = market_ts(15, 57)
    frame = bars([
        {"t": signal_ts, "o": 10.0, "h": 10.0, "l": 10.0, "c": 10.0},
        {"t": entry_ts, "o": 10.0, "h": 10.05, "l": 9.95, "c": 10.02},
        {"t": market_ts(16, 0), "o": 9.0, "h": 9.1, "l": 8.5, "c": 8.7},
    ])
    signal = CrossingEvent("TEST", signal_ts, 20.0, 10.0, 8.3333333333)
    result = evaluate_barrier(
        signal,
        frame,
        take_profit_pct=5,
        stop_loss_pct=3,
        max_horizon_minutes=60,
        entry_timestamp_ms=entry_ts,
        entry_price=10.0,
        regular_session_only=True,
    )
    assert result.status == "unresolved_session_close"
    assert result.exit_return_pct is None
