import numpy as np
import pandas as pd

from victory_trader import causal_hold_exit_controller as r268


class DummyReg:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, frame):
        return self.values[: len(frame)]


class DummyCls:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)

    def predict_proba(self, frame):
        p = self.probs[: len(frame)]
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
                "state_price": 10.2,
                "exit_now_pct": 2.0,
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
                "state_price": 10.1,
                "exit_now_pct": 1.0,
                "events_since_entry": 3,
                "x": 2.0,
            },
        ]
    )


def test_dynamic_policy_holds_then_exits_on_current_state():
    rows = r268.dynamic_policy_rows(
        _states(),
        DummyReg([1.0, -0.1, 3.0]),
        DummyCls([0.8, 0.4, 0.9]),
        ("x",),
    )
    assert len(rows) == 1
    assert rows.loc[0, "exit_t"] == 180_000
    assert rows.loc[0, "return_pct"] == 4.0
    assert rows.loc[0, "exit_reason"] == "model_exit"


def test_dynamic_policy_uses_terminal_if_all_states_hold():
    rows = r268.dynamic_policy_rows(
        _states(),
        DummyReg([1.0, 1.0, 1.0]),
        DummyCls([0.8, 0.8, 0.8]),
        ("x",),
    )
    assert rows.loc[0, "exit_t"] == 240_000
    assert rows.loc[0, "return_pct"] == 1.0
    assert rows.loc[0, "exit_reason"] == "terminal_cap"


def test_static_fifth_uses_terminal_fallback():
    rows = r268.static_policy_rows(_states(), "fifth")
    assert rows.loc[0, "exit_t"] == 240_000
    assert rows.loc[0, "return_pct"] == 1.0


def test_request268_contract():
    assert r268.REQUEST_ID == 268
    assert r268.HOLD_PROBABILITY_THRESHOLD == 0.50
    assert r268.MIN_EPISODE_COVERAGE == 0.80
    assert r268.MIN_GAIN_VS_BEST_STATIC == 0.10
