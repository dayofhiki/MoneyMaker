import numpy as np
import pandas as pd

from victory_trader import one_event_redecision as r247


def target_rows():
    return pd.DataFrame(
        {
            "trading_day": ["2026-05-05"] * 3,
            "ticker": ["T"] * 3,
            "hot_t": [0] * 3,
            "state_t": [60_000, 120_000, 180_000],
            "can_enter": [True, True, True],
            "terminal": [False, False, False],
            "enter_value": [1.0, -1.0, 10.0],
            "wait_value": [0.0, 0.0, 0.0],
            "hold_advantage": [0.1, 0.2, 0.3],
        }
    )


def test_one_event_target_uses_immediate_next_state_not_best_future():
    out = r247.one_event_targets(target_rows())
    assert out.loc[0, "next_enter_value"] == -1.0
    assert out.loc[1, "next_enter_value"] == 10.0
    assert np.isnan(out.loc[2, "next_enter_value"])


def test_next_state_must_itself_be_entry_eligible():
    x = target_rows()
    x.loc[1, "can_enter"] = False
    out = r247.one_event_targets(x)
    assert not bool(out.loc[0, "next_can_enter"])


def test_one_event_gap_is_recorded_without_becoming_feature():
    x = target_rows()
    x.loc[1, "state_t"] = 240_000
    out = r247.one_event_targets(x)
    assert out.loc[0, "next_state_t"] == 240_000


def test_request247_contract_is_frozen():
    assert r247.REQUEST_ID == 247
    assert r247.THRESHOLD == 0.5
    assert r247.MIN_WAIT_ACTIONS == 20
    assert r247.MAX_SEVERE_LOSS_247 == 0.35
