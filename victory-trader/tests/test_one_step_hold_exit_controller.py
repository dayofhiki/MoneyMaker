import numpy as np
import pandas as pd

from victory_trader import one_step_hold_exit_controller as r271


class DummyReg:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, frame):
        return self.values[: len(frame)]


class DummyCls:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict_proba(self, frame):
        p = self.values[: len(frame)]
        return np.column_stack([1.0 - p, p])


def _states():
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "entry_t": 60_000,
                "entry_price": 10.0,
                "state_t": 120_000,
                "state_price": 10.1,
                "exit_now_pct": 1.0,
                "events_since_entry": 1,
                "x": 0.0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "entry_t": 60_000,
                "entry_price": 10.0,
                "state_t": 180_000,
                "state_price": 10.4,
                "exit_now_pct": 4.0,
                "events_since_entry": 2,
                "x": 1.0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "entry_t": 60_000,
                "entry_price": 10.0,
                "state_t": 240_000,
                "state_price": 10.2,
                "exit_now_pct": 2.0,
                "events_since_entry": 3,
                "x": 2.0,
            },
        ]
    )


def test_recurrent_one_step_holds_then_exits():
    rows = r271.dynamic_policy_rows(
        _states(),
        DummyReg([0.4, -0.2, 0.0]),
        DummyCls([0.7, 0.3, 0.5]),
        ("x",),
    )
    assert rows.loc[0, "exit_t"] == 180_000
    assert rows.loc[0, "return_pct"] == 4.0


def test_terminal_always_exits():
    rows = r271.dynamic_policy_rows(
        _states().iloc[[0]].copy(),
        DummyReg([10.0]),
        DummyCls([0.99]),
        ("x",),
    )
    assert rows.loc[0, "exit_t"] == 120_000
    assert rows.loc[0, "exit_reason"] == "terminal_cap"


def test_request271_contract():
    assert r271.REQUEST_ID == 271
    assert r271.HORIZON == 1
    assert r271.HOLD_PROBABILITY_THRESHOLD == 0.50
