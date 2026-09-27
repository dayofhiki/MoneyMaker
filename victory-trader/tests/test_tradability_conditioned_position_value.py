import numpy as np
import pandas as pd

from victory_trader import tradability_conditioned_position_value as r247


def test_one_step_hold_advantage_uses_next_state_only():
    times = np.array([1_000, 61_000, 121_000], dtype=np.int64)
    opens = np.array([10.0, 11.0, 5.0])
    value = r247.one_step_hold_advantage(
        (times, opens),
        current_t=0,
        next_t=60_000,
    )
    assert value > 0


def test_position_targets_do_not_search_best_future():
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 0,
                "terminal": False,
                "attention_score": 1.0,
            },
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
                "terminal": True,
                "attention_score": 1.0,
            },
        ]
    )
    probs = np.array([0.9, 0.9, 0.9])
    times = np.array([1_000, 61_000, 121_000], dtype=np.int64)
    opens = np.array([10.0, 11.0, 5.0])
    out = r247.build_position_targets(
        states,
        probs,
        {("2026-05-05", "TEST"): (times, opens)},
    )
    assert len(out) == 2
    assert out.iloc[0].next_state_t == 60_000
    assert out.iloc[0].hold_advantage > 0
    assert out.iloc[1].next_state_t == 120_000
    assert out.iloc[1].hold_advantage < 0


def test_position_policy_exits_on_first_nonpositive_prediction():
    entry_decisions = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "entered": True,
                "unresolved": False,
                "net_pct": np.nan,
                "stress_pct": np.nan,
                "entry_t": 60_000,
                "exit_t": 240_000,
                "wait_actions": 0,
                "hold_actions": 0,
                "entry_decision_t": 60_000,
                "exit_decision_t": 240_000,
                "deadline_liquidation": False,
                "policy_exit_missing": False,
            }
        ]
    )
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

    class Dummy:
        def predict(self, x):
            return np.array([1.0, 0.5, -0.2])[: len(x)]

    out = r247.apply_position_policy(
        entry_decisions,
        states,
        np.array([0.9, 0.9, 0.9]),
        Dummy(),
        ("attention_score",),
    )
    assert out.loc[0, "exit_decision_t"] == 180_000
    assert out.loc[0, "hold_actions"] == 1


def test_request247_contract_is_frozen():
    assert r247.REQUEST_ID == 247
    assert r247.POSITION_TRAIN_DAYS == [
        "2026-05-05",
        "2026-05-06",
        "2026-05-07",
        "2026-05-08",
    ]
