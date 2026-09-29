import numpy as np
import pandas as pd

from victory_trader import calibrated_hold_exit_controller as r269


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


def _episode(day, ticker, returns):
    rows = []
    for index, value in enumerate(returns, start=1):
        rows.append(
            {
                "trading_day": day,
                "ticker": ticker,
                "hot_t": 0,
                "entry_t": 60_000,
                "entry_price": 10.0,
                "state_t": 60_000 * (index + 1),
                "state_price": 10.0 * (1 + value / 100),
                "exit_now_pct": value,
                "events_since_entry": index,
                "x": float(index),
            }
        )
    return rows


def test_dynamic_thresholds_control_recurrent_hold():
    states = pd.DataFrame(_episode("2026-05-08", "TEST", [1.0, 3.0, -1.0]))
    rows = r269.dynamic_policy_rows(
        states,
        DummyReg([1.0, 0.2, -0.5]),
        DummyCls([0.8, 0.55, 0.2]),
        ("x",),
        probability_threshold=0.60,
        advantage_threshold=0.0,
    )
    assert rows.loc[0, "exit_t"] == 180_000
    assert rows.loc[0, "return_pct"] == 3.0


def test_request269_contract():
    assert r269.REQUEST_ID == 269
    assert r269.MODEL_FIT_DAYS == (
        "2026-05-05",
        "2026-05-06",
        "2026-05-07",
    )
    assert r269.CALIBRATION_DAY == "2026-05-08"
    assert r269.PROBABILITY_THRESHOLDS == (0.50, 0.60, 0.70, 0.80, 0.90)
    assert r269.ADVANTAGE_THRESHOLDS == (0.00, 0.25, 0.50, 1.00)
