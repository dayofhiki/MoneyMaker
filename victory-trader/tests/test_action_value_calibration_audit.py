import numpy as np
import pandas as pd

from victory_trader import action_value_calibration_audit as a245


def frame():
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "T",
                "hot_t": 0,
                "state_t": 1,
                "flat_action": "WAIT",
                "held_action": "HOLD",
                "policy_entry_index": 2,
                "policy_exit_index": 4,
                "can_enter": True,
                "terminal": False,
                "enter_value": -1.0,
                "wait_value": 1.0,
                "hold_advantage": 0.2,
                "predicted_enter_value": 0.5,
                "predicted_wait_value": 0.1,
                "predicted_hold_advantage": 0.1,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "T",
                "hot_t": 0,
                "state_t": 2,
                "flat_action": "WAIT",
                "held_action": "HOLD",
                "policy_entry_index": 2,
                "policy_exit_index": 4,
                "can_enter": True,
                "terminal": False,
                "enter_value": 0.0,
                "wait_value": 2.0,
                "hold_advantage": 0.1,
                "predicted_enter_value": 0.4,
                "predicted_wait_value": 0.2,
                "predicted_hold_advantage": 0.2,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "T",
                "hot_t": 0,
                "state_t": 3,
                "flat_action": "ENTER",
                "held_action": "HOLD",
                "policy_entry_index": 2,
                "policy_exit_index": 4,
                "can_enter": True,
                "terminal": False,
                "enter_value": 3.0,
                "wait_value": 0.0,
                "hold_advantage": -0.2,
                "predicted_enter_value": 0.3,
                "predicted_wait_value": 0.0,
                "predicted_hold_advantage": 0.3,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "T",
                "hot_t": 0,
                "state_t": 4,
                "flat_action": "ABSTAIN",
                "held_action": "HOLD",
                "policy_entry_index": 2,
                "policy_exit_index": 4,
                "can_enter": False,
                "terminal": False,
                "enter_value": np.nan,
                "wait_value": np.nan,
                "hold_advantage": -1.0,
                "predicted_enter_value": 0.0,
                "predicted_wait_value": 0.0,
                "predicted_hold_advantage": 0.5,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "T",
                "hot_t": 0,
                "state_t": 5,
                "flat_action": "ABSTAIN",
                "held_action": "EXIT",
                "policy_entry_index": 2,
                "policy_exit_index": 4,
                "can_enter": False,
                "terminal": True,
                "enter_value": np.nan,
                "wait_value": np.nan,
                "hold_advantage": -2.0,
                "predicted_enter_value": 0.0,
                "predicted_wait_value": 0.0,
                "predicted_hold_advantage": -0.5,
            },
        ]
    )


def test_reachable_masks_stop_after_flat_decision_and_track_position():
    scored = frame()
    flat, position = a245.reachable_masks(scored)
    assert flat.tolist() == [True, True, True, False, False]
    assert position.tolist() == [False, False, False, True, True]


def test_flat_advantage_measures_best_alternative_not_raw_profit():
    scored = frame()
    flat, _ = a245.reachable_masks(scored)
    metrics = a245.flat_advantage_metrics(scored, flat)
    assert metrics["chosen_enter_rows"] == 1
    assert metrics["chosen_enter_advantage_positive_rate"] == 1.0
    assert metrics["chosen_wait_rows"] == 2
    assert metrics["chosen_wait_advantage_positive_rate"] == 1.0


def test_hold_decision_precision_uses_realized_hold_advantage():
    scored = frame()
    _, position = a245.reachable_masks(scored)
    metrics = a245.hold_decision_metrics(scored, position)
    assert metrics["chosen_hold_rows"] == 1
    assert metrics["chosen_hold_advantage_positive_rate"] == 0.0
    assert metrics["chosen_exit_rows"] == 1
    assert metrics["chosen_exit_correct_rate"] == 1.0


def test_support_summary_preserves_zero_mass():
    x = pd.DataFrame(
        {
            "trading_day": ["d"] * 4,
            "ticker": ["T"] * 4,
            "hot_t": [0] * 4,
            "wait_value": [0.0, 0.0, 0.1, np.nan],
        }
    )
    summary = a245.support_summary(x, "wait_value")
    assert summary["rows"] == 3
    assert summary["zero_rate"] == 2 / 3


def test_request245_contract_is_frozen():
    assert a245.REQUEST_ID == 245
    assert a245.MIN_SPEARMAN == 0.20
    assert a245.MIN_CHOSEN_ADVANTAGE_POSITIVE_RATE == 0.55
