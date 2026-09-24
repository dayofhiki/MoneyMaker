from __future__ import annotations

from victory_trader.calibration_relative_entry_value import (
    MAX_EXECUTED_RATE,
    MIN_ACTION_VALUE_COVERAGE,
    MIN_EXECUTED_RATE,
    MIN_POSITIVE_DAYS,
    REQUEST_ID,
    SUPPORT_LEVELS,
)


def test_request183_contract_is_frozen():
    assert REQUEST_ID == 183
    assert SUPPORT_LEVELS == (0.80, 0.90, 0.95)
    assert MIN_ACTION_VALUE_COVERAGE == 0.90
    assert MIN_EXECUTED_RATE == 0.03
    assert MAX_EXECUTED_RATE == 0.40
    assert MIN_POSITIVE_DAYS == 3
