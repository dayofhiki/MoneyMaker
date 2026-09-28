import numpy as np
import pandas as pd

from victory_trader import learned_pullback_entry as r254


class DummyModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, frame):
        return self.values[: len(frame)]


def test_learned_policy_can_wait_repeatedly_then_enter():
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "state_price": 10.0,
                "enter_value_pct": -0.5,
                "event_index": 0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "state_price": 9.8,
                "enter_value_pct": 0.2,
                "event_index": 1,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 180_000,
                "state_price": 9.7,
                "enter_value_pct": 0.8,
                "event_index": 2,
            },
        ]
    )
    out = r254.make_learned_policy(
        states,
        DummyModel([-0.2, 0.3, 0.9]),
        DummyModel([-1.0, -0.1, 0.5]),
        ("event_index",),
    )
    assert bool(out.loc[0, "entered"])
    assert out.loc[0, "entry_t"] == 180_000
    assert out.loc[0, "value_pct"] == 0.8


def test_policy_metrics_treats_abstention_as_cash_zero():
    rows = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "A",
                "hot_t": 1,
                "entered": False,
                "value_pct": np.nan,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "B",
                "hot_t": 2,
                "entered": True,
                "value_pct": 1.0,
            },
        ]
    )
    metrics = r254.policy_metrics(rows, candidate_count=2)
    assert metrics["cash_episodes"] == 1
    assert metrics["unresolved_entries"] == 0
    assert metrics["mean_pct"] == 0.5
    assert metrics["entry_rate"] == 0.5


def test_last_state_can_enter_on_positive_enter_value():
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "state_price": 10.0,
                "enter_value_pct": 0.4,
                "event_index": 0,
            }
        ]
    )
    out = r254.make_learned_policy(
        states,
        DummyModel([0.2]),
        DummyModel([-0.5]),
        ("event_index",),
    )
    assert bool(out.loc[0, "entered"])
    assert out.loc[0, "entry_t"] == 60_000


def test_request254_contract():
    assert r254.REQUEST_ID == 254
    assert r254.CANDIDATE_MODEL_DAYS == [
        "2026-04-30",
        "2026-05-01",
        "2026-05-04",
    ]
    assert r254.CONTROLLER_TRAIN_DAYS == [
        "2026-05-05",
        "2026-05-06",
        "2026-05-07",
        "2026-05-08",
    ]
    assert r254.CANDIDATE_FRACTION == 0.05
    assert r254.MIN_GAIN_VS_IMMEDIATE == 0.50
    assert r254.MIN_GAIN_VS_FIXED_PULLBACK == 0.20
