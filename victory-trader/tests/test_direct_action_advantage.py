import numpy as np
import pandas as pd

from victory_trader import direct_action_advantage as d246


def states():
    return pd.DataFrame(
        {
            "can_enter": [True, True, True, False],
            "terminal": [False, False, False, True],
            "elapsed_minutes": [1.0, 2.0, 3.0, 29.0],
        }
    )


def policy(enter, wait, hold):
    return d246.DirectPolicy(
        enter_model=d246.ConstantBinary(enter),
        wait_model=d246.ConstantBinary(wait),
        hold_model=d246.ConstantBinary(hold),
        columns=("elapsed_minutes",),
        support={},
    )


def test_direct_actions_use_cascade_not_value_scale():
    flat, held, *_ = d246.direct_actions(states(), policy(0.6, 0.9, 0.6))
    assert flat.tolist()[:3] == ["ENTER", "ENTER", "ENTER"]
    assert held.tolist()[:3] == ["HOLD", "HOLD", "HOLD"]
    assert held.tolist()[3] == "EXIT"


def test_wait_only_used_when_enter_not_preferred():
    flat, _, *_ = d246.direct_actions(states(), policy(0.4, 0.7, 0.4))
    assert flat.tolist()[:3] == ["WAIT", "WAIT", "WAIT"]


def test_abstain_when_both_flat_actions_below_natural_boundary():
    flat, _, *_ = d246.direct_actions(states(), policy(0.4, 0.4, 0.4))
    assert flat.tolist()[:3] == ["ABSTAIN", "ABSTAIN", "ABSTAIN"]


def test_fit_binary_does_not_manufacture_missing_class():
    frame = pd.DataFrame(
        {
            "trading_day": ["2026-05-05"] * 3,
            "ticker": ["T"] * 3,
            "hot_t": [0] * 3,
            "state_t": [1, 2, 3],
            "elapsed_minutes": [1.0, 2.0, 3.0],
        }
    )
    model, support = d246.fit_binary(
        frame,
        pd.Series([False, False, False], index=frame.index),
        ("elapsed_minutes",),
        seed=1,
    )
    assert isinstance(model, d246.ConstantBinary)
    assert support["positive_rows"] == 0
    assert np.allclose(model.predict_proba(frame[["elapsed_minutes"]])[:, 1], 0)


def test_request246_contract_is_frozen():
    assert d246.REQUEST_ID == 246
    assert d246.THRESHOLD == 0.5
    assert d246.MIN_ACTION_PRECISION == 0.55
    assert d246.MIN_WAIT_ACTIONS == 5
