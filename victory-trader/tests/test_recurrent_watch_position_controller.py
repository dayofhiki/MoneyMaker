import math

import pandas as pd

from victory_trader.recurrent_watch_position_controller import (
    _sell_return_pct,
    _trajectory_for_entry,
    score_position_states,
)


def test_sell_return_includes_base_friction():
    value = _sell_return_pct(10.1, 10.0)
    assert value < 0.0


class DummyModel:
    columns = ("feature_x",)

    class Inner:
        def predict(self, frame):
            return frame["feature_x"].to_numpy(dtype=float)

    model = Inner()


def test_position_action_holds_only_when_future_beats_mark():
    frame = pd.DataFrame(
        {
            "feature_x": [2.0, -1.0],
            "causal_mark_base_return_pct": [1.0, 0.0],
        }
    )
    scored = score_position_states(frame, DummyModel())
    assert list(scored["position_action"]) == ["HOLD", "EXIT"]


class FakeStore:
    def __init__(self, seconds):
        self._seconds = seconds

    def seconds(self, day, ticker):
        return self._seconds.copy()

    def halts(self, day):
        return {}


def _bar(t, o, h=None, low=None, c=None):
    return {
        "t": t,
        "o": o,
        "h": o if h is None else h,
        "l": o if low is None else low,
        "c": o if c is None else c,
    }


def test_model_exit_uses_later_one_second_open():
    seconds = pd.DataFrame(
        [
            _bar(1_000, 10.0, 10.1, 9.9, 10.0),
            _bar(61_000, 10.4, 10.5, 10.3, 10.4),
            _bar(62_000, 10.35, 10.4, 10.3, 10.35),
        ]
    )
    entry = pd.Series(
        {
            "trading_day": "2026-05-11",
            "ticker": "TEST",
            "hot_t": 0,
            "state_t": 0,
            "entry_fill_t": 1_000,
            "entry_modeled_price": 10.0,
        }
    )
    decisions = pd.DataFrame(
        [
            {
                "state_t": 61_000,
                "position_action": "EXIT",
            }
        ]
    )
    row = _trajectory_for_entry(
        entry,
        decisions,
        FakeStore(seconds),
    )
    assert row["status"] == "completed"
    assert row["exit_reason"] == "model_exit"
    assert row["exit_fill_t"] == 62_000
    assert row["base_net_return_pct"] > 0.0


def test_hard_stop_overrides_later_model_hold():
    seconds = pd.DataFrame(
        [
            _bar(1_000, 10.0, 10.0, 9.9, 10.0),
            _bar(10_000, 9.8, 9.9, 9.6, 9.7),
            _bar(12_000, 9.5, 9.6, 9.4, 9.5),
            _bar(61_000, 10.2, 10.3, 10.1, 10.2),
        ]
    )
    entry = pd.Series(
        {
            "trading_day": "2026-05-11",
            "ticker": "TEST",
            "hot_t": 0,
            "state_t": 0,
            "entry_fill_t": 1_000,
            "entry_modeled_price": 10.0,
        }
    )
    decisions = pd.DataFrame(
        [
            {
                "state_t": 61_000,
                "position_action": "HOLD",
            }
        ]
    )
    row = _trajectory_for_entry(
        entry,
        decisions,
        FakeStore(seconds),
    )
    assert row["status"] == "completed"
    assert row["exit_reason"] == "hard_stop"
    assert row["exit_fill_t"] == 12_000
    assert row["base_net_return_pct"] < 0.0
