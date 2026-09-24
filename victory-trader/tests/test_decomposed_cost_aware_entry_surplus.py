from __future__ import annotations

from victory_trader.decomposed_cost_aware_entry_surplus import (
    DECOMPOSED_FEATURES,
    MAX_EXECUTED_RATE,
    MIN_ENTRY_COVERAGE,
    MIN_EXECUTED_RATE,
    MIN_POSITIVE_DAYS,
    MIN_SECOND_COVERAGE,
    REQUEST_ID,
    SEEDS,
)
from victory_trader.second_path_attention_probe import SECOND_FEATURES


def test_request184_contract_is_frozen():
    assert REQUEST_ID == 184
    assert SEEDS == {
        "enter_gross_pct": 20261084,
        "enter_drag_pct": 20261085,
        "wait_gross_pct": 20261086,
        "wait_drag_pct": 20261087,
    }
    assert MIN_ENTRY_COVERAGE == 0.90
    assert MIN_SECOND_COVERAGE == 0.90
    assert MIN_EXECUTED_RATE == 0.03
    assert MAX_EXECUTED_RATE == 0.40
    assert MIN_POSITIVE_DAYS == 3


def test_request184_adds_one_second_and_cost_proxy_features():
    for column in SECOND_FEATURES:
        assert column in DECOMPOSED_FEATURES
    assert "base_zero_move_cost_proxy_pct" in DECOMPOSED_FEATURES
