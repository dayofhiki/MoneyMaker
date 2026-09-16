import pandas as pd
import pytest

from victory_trader.events import CrossingEvent
from victory_trader.exits import evaluate_barrier


def bars(rows):
    return pd.DataFrame(rows)


def event():
    return CrossingEvent("TEST", 0, 20.0, 10.0, 8.3333333333)


def test_take_profit_first():
    frame = bars([
        {"t": 0, "h": 10.0, "l": 10.0, "c": 10.0},
        {"t": 60_000, "h": 10.25, "l": 9.95, "c": 10.20},
    ])
    result = evaluate_barrier(event(), frame, take_profit_pct=2, stop_loss_pct=1)
    assert result.status == "take_profit"
    assert result.minutes_to_exit == 1
    assert result.exit_return_pct == 2


def test_stop_loss_first():
    frame = bars([
        {"t": 0, "h": 10.0, "l": 10.0, "c": 10.0},
        {"t": 60_000, "h": 10.05, "l": 9.85, "c": 9.90},
    ])
    result = evaluate_barrier(event(), frame, take_profit_pct=2, stop_loss_pct=1)
    assert result.status == "stop_loss"
    assert result.exit_return_pct == -1


def test_same_bar_double_hit_is_ambiguous():
    frame = bars([
        {"t": 0, "h": 10.0, "l": 10.0, "c": 10.0},
        {"t": 60_000, "h": 10.25, "l": 9.85, "c": 10.10},
    ])
    result = evaluate_barrier(event(), frame, take_profit_pct=2, stop_loss_pct=1)
    assert result.status == "ambiguous"
    assert result.exit_return_pct is None


def test_timeout_uses_last_available_close():
    frame = bars([
        {"t": 0, "h": 10.0, "l": 10.0, "c": 10.0},
        {"t": 60_000, "h": 10.10, "l": 9.95, "c": 10.05},
        {"t": 120_000, "h": 10.15, "l": 9.95, "c": 10.10},
    ])
    result = evaluate_barrier(
        event(), frame, take_profit_pct=2, stop_loss_pct=1, max_horizon_minutes=2
    )
    assert result.status == "timeout"
    assert result.minutes_to_exit == 2
    assert result.exit_return_pct == pytest.approx(1.0)
