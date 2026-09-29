import numpy as np
import pandas as pd

from victory_trader import pullback_advantage_gate as r255


class DummyClassifier:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, frame):
        p = self.probabilities[: len(frame)]
        return np.column_stack([1.0 - p, p])


def test_policy_waits_until_pullback_and_positive_advantage():
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "state_price": 10.0,
                "enter_value_pct": -0.2,
                "drawdown_from_running_high_pct": 0.0,
                "event_index": 0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "state_price": 9.8,
                "enter_value_pct": 0.1,
                "drawdown_from_running_high_pct": -2.0,
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
    out = r255.make_policy(
        states,
        selected,
        DummyClassifier([0.9, 0.4, 0.8]),
        ("event_index",),
    )
    assert bool(out.loc[0, "entered"])
    assert out.loc[0, "entry_t"] == 180_000
    assert out.loc[0, "value_pct"] == 0.8


def test_candidate_without_states_remains_cash():
    selected = pd.DataFrame(
        [{"trading_day": "2026-05-11", "ticker": "NONE", "t": 0}]
    )
    states = pd.DataFrame(
        columns=[
            "trading_day",
            "ticker",
            "hot_t",
            "drawdown_from_running_high_pct",
            "event_index",
        ]
    )
    out = r255.make_policy(
        states,
        selected,
        DummyClassifier([]),
        ("event_index",),
    )
    assert not bool(out.loc[0, "entered"])
    assert pd.isna(out.loc[0, "value_pct"])


def test_pullback_mask_is_fixed_two_percent():
    states = pd.DataFrame(
        {"drawdown_from_running_high_pct": [-1.99, -2.0, -3.0]}
    )
    assert r255.pullback_mask(states).tolist() == [False, True, True]


def test_request255_contract():
    assert r255.REQUEST_ID == 255
    assert r255.PULLBACK_TRIGGER_PCT == 2.0
    assert r255.ADVANTAGE_PROBABILITY_THRESHOLD == 0.50
    assert r255.MIN_PULLBACK_AUC == 0.55
    assert r255.MIN_GAIN_VS_FIXED_PULLBACK == 0.10
