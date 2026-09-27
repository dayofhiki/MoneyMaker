import numpy as np
import pandas as pd

from victory_trader import one_step_position_value as r247


def test_one_step_target_never_uses_later_best_state():
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "terminal": False,
                "attention_score": 1.0,
            },
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "terminal": False,
                "attention_score": 1.0,
            },
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 180_000,
                "terminal": False,
                "attention_score": 1.0,
            },
        ]
    )
    times = np.array([61_000, 121_000, 181_000], dtype=np.int64)
    opens = np.array([10.0, 9.0, 20.0])
    out = r247.one_step_targets(
        states,
        np.array([0.9, 0.9, 0.9]),
        {("2026-05-05", "TEST"): (times, opens)},
    )
    assert len(out) == 2
    assert out.iloc[0].next_state_t == 120_000
    assert out.iloc[0].hold_one_step_value < 0
    assert out.iloc[1].hold_one_step_value > 0


def test_position_controller_preserves_frozen_entry_and_reconsiders_hold():
    class Dummy:
        def predict(self, x):
            return np.array([0.0, 1.0, -1.0])[: len(x)]

    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "terminal": False,
                "attention_score": 1.0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "terminal": False,
                "attention_score": 1.0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 180_000,
                "terminal": False,
                "attention_score": 1.0,
            },
        ]
    )
    frozen = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "entered": True,
                "entry_decision_t": 60_000,
                "wait_actions": 0,
            }
        ]
    )
    out = r247.make_position_decisions(
        states,
        np.array([0.9, 0.9, 0.9]),
        frozen,
        Dummy(),
        ("attention_score",),
    )
    assert out.loc[0, "entry_decision_t"] == 60_000
    assert out.loc[0, "exit_decision_t"] == 180_000
    assert out.loc[0, "hold_actions"] == 1


def test_low_tradability_forces_hold_until_reconsideration():
    class Dummy:
        def predict(self, x):
            return np.full(len(x), -1.0)

    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "terminal": False,
                "attention_score": 1.0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "terminal": False,
                "attention_score": 1.0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 180_000,
                "terminal": False,
                "attention_score": 1.0,
            },
        ]
    )
    frozen = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "entered": True,
                "entry_decision_t": 60_000,
                "wait_actions": 0,
            }
        ]
    )
    out = r247.make_position_decisions(
        states,
        np.array([0.9, 0.2, 0.9]),
        frozen,
        Dummy(),
        ("attention_score",),
    )
    assert out.loc[0, "exit_decision_t"] == 180_000
    assert out.loc[0, "hold_actions"] == 1


def test_request247_contract_is_frozen():
    assert r247.REQUEST_ID == 247
    assert r247.ADMISSION_THRESHOLD == 0.60
    assert r247.HOLD_THRESHOLD == 0.0
    assert r247.VALUE_TRAIN_DAYS == [
        "2026-05-05",
        "2026-05-06",
        "2026-05-07",
        "2026-05-08",
    ]
    assert r247.MIN_TARGET_ROWS == 300
