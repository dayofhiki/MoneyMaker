from __future__ import annotations

from victory_trader.buy_gated_multi_horizon_value import (
    MIN_CHOSEN_SPEARMAN,
    MIN_CHOSEN_TARGET_COVERAGE,
    MIN_GOOD_DAYS,
    MIN_SELECTED_RATE,
    REQUEST_ID,
    VALUE_HORIZONS,
)


def test_request169_contract_is_frozen():
    assert REQUEST_ID == 169
    assert VALUE_HORIZONS == (2, 5, 10)
    assert MIN_CHOSEN_TARGET_COVERAGE == 0.70
    assert MIN_CHOSEN_SPEARMAN == 0.05
    assert MIN_SELECTED_RATE == 0.10
    assert MIN_GOOD_DAYS == 4
