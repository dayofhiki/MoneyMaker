import numpy as np
import pandas as pd

from victory_trader import fitted_policy_hold_exit as r272


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
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "exit_now_pct": 0.0,
                "x": 0.0,
            },
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "exit_now_pct": 3.0,
                "x": 1.0,
            },
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 180_000,
                "exit_now_pct": 1.0,
                "x": 2.0,
            },
        ]
    )


def test_policy_consistent_target_follows_downstream_policy():
    out = r272.attach_policy_values_from_base(
        _states(),
        DummyReg([1.0, -1.0, 0.0]),
        DummyCls([0.8, 0.2, 0.5]),
        ("x",),
    )
    assert out.loc[0, r272.OOF_TARGET_COLUMN] == 3.0
    assert out.loc[1, r272.OOF_TARGET_COLUMN] == -2.0
    assert out.loc[2, r272.OOF_TARGET_COLUMN] == 0.0


def test_request272_contract():
    assert r272.REQUEST_ID == 272
    assert r272.HOLD_PROBABILITY_THRESHOLD == 0.50
    assert r272.TRAIN_DAYS == (
        "2026-05-05",
        "2026-05-06",
        "2026-05-07",
        "2026-05-08",
    )
    assert r272.OOF_TARGET_COLUMN == "policy_consistent_hold_advantage_pct"
