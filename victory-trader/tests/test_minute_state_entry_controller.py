import numpy as np
import pandas as pd

from victory_trader import minute_state_entry_controller as r258


class DummyRegressor:
    def predict(self, frame):
        return np.array([-0.2, 0.4, 0.8])[: len(frame)]


class DummyClassifier:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict_proba(self, frame):
        p = self.values[: len(frame)]
        return np.column_stack([1.0 - p, p])


class DummyModels:
    def __init__(self):
        self.value = DummyRegressor()
        self.positive = DummyClassifier([0.8, 0.4, 0.7])
        self.severe = DummyClassifier([0.1, 0.1, 0.2])
        self.columns = ("event_index",)
        self.value_offset = 0.0


def test_policy_recurrently_waits_until_all_conditions_pass():
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "state_price": 10.0,
                "enter_value_pct": -0.5,
                "drawdown_from_running_high_pct": -2.1,
                "event_index": 0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "state_price": 9.8,
                "enter_value_pct": 0.2,
                "drawdown_from_running_high_pct": -2.4,
                "event_index": 1,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 180_000,
                "state_price": 9.7,
                "enter_value_pct": 0.8,
                "drawdown_from_running_high_pct": -3.0,
                "event_index": 2,
            },
        ]
    )
    selected = pd.DataFrame(
        [{"trading_day": "2026-05-11", "ticker": "TEST", "t": 0}]
    )
    out = r258.make_policy(states, selected, DummyModels())
    assert bool(out.loc[0, "entered"])
    assert out.loc[0, "entry_t"] == 180_000
    assert out.loc[0, "value_pct"] == 0.8


def test_policy_requires_structural_pullback():
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "state_price": 10.0,
                "enter_value_pct": 1.0,
                "drawdown_from_running_high_pct": -1.0,
                "event_index": 0,
            }
        ]
    )
    selected = pd.DataFrame(
        [{"trading_day": "2026-05-11", "ticker": "TEST", "t": 0}]
    )
    out = r258.make_policy(states, selected, DummyModels())
    assert not bool(out.loc[0, "entered"])


def test_request258_contract():
    assert r258.REQUEST_ID == 258
    assert r258.MODEL_FIT_DAYS == [
        "2026-05-05",
        "2026-05-06",
        "2026-05-07",
    ]
    assert r258.MODEL_CALIBRATION_DAY == "2026-05-08"
    assert r258.PULLBACK_TRIGGER_PCT == 2.0
    assert r258.POSITIVE_PROBABILITY_THRESHOLD == 0.50
    assert r258.SEVERE_PROBABILITY_THRESHOLD == 0.50
    assert r258.MIN_GAIN_VS_FIXED_PULLBACK == 0.10
