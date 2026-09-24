from __future__ import annotations

from victory_trader.three_minute_cost_cover_entry_opportunity import (
    ENTER_SEED,
    MAX_SELECTED_RATE,
    MIN_POSITIVE_DAYS,
    MIN_SELECTED_RATE,
    MIN_SPEARMAN,
    REQUEST_ID,
    WAIT_SEED,
)


def test_request185_contract_is_frozen():
    assert REQUEST_ID == 185
    assert ENTER_SEED == 20261088
    assert WAIT_SEED == 20261089
    assert MIN_SPEARMAN == 0.10
    assert MIN_SELECTED_RATE == 0.10
    assert MAX_SELECTED_RATE == 0.70
    assert MIN_POSITIVE_DAYS == 4
