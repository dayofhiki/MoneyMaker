from __future__ import annotations

import pandas as pd

from victory_trader.causal_entry_action_value import (
    BOOTSTRAP_SAMPLES,
    ENTER_SEED,
    MAX_EXECUTED_RATE,
    MIN_ACTION_VALUE_COVERAGE,
    MIN_ENTRY_STATE_COVERAGE,
    MIN_EXECUTED_RATE,
    REQUEST_ID,
    WAIT_SEED,
    choose_action,
)


def test_request182_contract_is_frozen():
    assert REQUEST_ID == 182
    assert ENTER_SEED == 20261082
    assert WAIT_SEED == 20261083
    assert BOOTSTRAP_SAMPLES == 10_000
    assert MIN_ENTRY_STATE_COVERAGE == 0.90
    assert MIN_ACTION_VALUE_COVERAGE == 0.90
    assert MIN_EXECUTED_RATE == 0.10
    assert MAX_EXECUTED_RATE == 0.80


def test_request182_action_rule_and_ties_are_conservative():
    frame = pd.DataFrame(
        {
            "predicted_enter_now_base_pct": [1.0, 0.2, -0.1, 0.5, 0.0],
            "predicted_wait_1m_base_pct": [0.5, 0.7, -0.2, 0.5, 0.0],
        }
    )
    assert choose_action(frame).tolist() == [
        "ENTER_NOW",
        "WAIT_1M",
        "ABSTAIN",
        "WAIT_1M",
        "ABSTAIN",
    ]
