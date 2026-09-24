from __future__ import annotations

from victory_trader.causal_entry_executability import (
    CAL_TARGET_ENTRY_RATE,
    EXECUTION_FEATURES,
    FRESH_DAYS,
    FRESH_DAILY_ENTRY_RATE,
    FRESH_EXEC_AUC,
    FRESH_MIN_BUYS_PER_DAY,
    FRESH_POOLED_ENTRY_RATE,
)
from victory_trader.hot_economic_opportunity import ENTRY_FEATURES


def test_request165_fresh_dates_are_frozen():
    assert FRESH_DAYS == [
        "2026-06-15",
        "2026-06-16",
        "2026-06-17",
        "2026-06-18",
        "2026-06-22",
    ]


def test_request165_entry_feature_contract_is_unchanged():
    assert tuple(ENTRY_FEATURES) == EXECUTION_FEATURES


def test_request165_gates_are_frozen():
    assert CAL_TARGET_ENTRY_RATE == 0.90
    assert FRESH_POOLED_ENTRY_RATE == 0.90
    assert FRESH_DAILY_ENTRY_RATE == 0.85
    assert FRESH_EXEC_AUC == 0.55
    assert FRESH_MIN_BUYS_PER_DAY == 25
