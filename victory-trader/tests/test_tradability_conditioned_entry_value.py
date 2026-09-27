import numpy as np
import pandas as pd

from victory_trader import tradability_conditioned_entry_value as r246


def test_executable_return_requires_both_entry_and_exit():
    times = np.array([1_000, 61_000], dtype=np.int64)
    opens = np.array([10.0, 11.0])
    value = r246.executable_return(
        path=(times, opens),
        entry_t=0,
        exit_t=60_000,
    )
    assert value < 10.0
    assert value > 0.0
    missing = r246.executable_return(
        path=(np.array([1_000], dtype=np.int64), np.array([10.0])),
        entry_t=0,
        exit_t=60_000,
    )
    assert np.isnan(missing)


def test_wait_target_uses_next_admitted_state_not_best_future():
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "can_enter": True,
                "terminal": False,
                "held_action": "HOLD",
            },
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "can_enter": True,
                "terminal": False,
                "held_action": "HOLD",
            },
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 180_000,
                "can_enter": True,
                "terminal": False,
                "held_action": "EXIT",
            },
        ]
    )
    # First two states admitted, third is not. The first state's WAIT target must
    # come from the second admitted state, never a search over later outcomes.
    probs = np.array([0.9, 0.9, 0.1])
    times = np.array([61_000, 121_000, 181_000], dtype=np.int64)
    opens = np.array([10.0, 9.0, 11.0])
    out = r246.build_value_targets(
        frame,
        probs,
        {("2026-05-05", "TEST"): (times, opens)},
    )
    assert len(out) == 2
    assert out.iloc[0].wait_target_source == "next_admitted_enter"
    assert np.isclose(out.iloc[0].wait_value, out.iloc[1].enter_value, equal_nan=True)
    assert out.iloc[1].wait_target_source == "cash_no_later_admitted_state"
    assert out.iloc[1].wait_value == 0.0


def test_learned_decision_waits_until_predicted_enter_wins():
    class Dummy:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        def predict(self, x):
            return self.values[: len(x)]

    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "can_enter": True,
                "terminal": False,
                "held_action": "HOLD",
                "attention_score": 1.0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "can_enter": True,
                "terminal": False,
                "held_action": "HOLD",
                "attention_score": 1.0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 180_000,
                "can_enter": True,
                "terminal": False,
                "held_action": "EXIT",
                "attention_score": 1.0,
            },
        ]
    )
    models = {
        "enter_value": Dummy([-1.0, 1.0, 0.0]),
        "wait_value": Dummy([0.5, 0.0, 0.0]),
    }
    decisions = r246.make_decisions(
        frame,
        np.array([0.9, 0.9, 0.9]),
        value_models=models,
        value_model_columns=("attention_score",),
    )
    assert decisions.loc[0, "entry_decision_t"] == 120_000
    assert decisions.loc[0, "exit_decision_t"] == 180_000
    assert decisions.loc[0, "wait_actions"] == 1


def test_request246_contract_is_frozen():
    assert r246.REQUEST_ID == 246
    assert r246.ADMISSION_THRESHOLD == 0.60
    assert r246.VALUE_TRAIN_DAYS == [
        "2026-05-05",
        "2026-05-06",
        "2026-05-07",
        "2026-05-08",
    ]
    assert r246.TEST_DAYS == [
        "2026-05-11",
        "2026-05-12",
        "2026-05-13",
        "2026-05-14",
        "2026-05-15",
        "2026-05-18",
        "2026-05-19",
        "2026-05-20",
    ]
