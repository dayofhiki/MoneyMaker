from __future__ import annotations

from victory_trader.market_regime_positive_value import (
    MIN_CONTEXT_COVERAGE,
    MIN_GLOBAL_AUC,
    MIN_GOOD_DAYS,
    MIN_SAME_MINUTE_AUC,
    MIN_SELECTED_POSITIVE_RATE,
    MIN_SELECTED_RATE,
    REQUEST_ID,
)


def test_request174_contract_is_frozen():
    assert REQUEST_ID == 174
    assert MIN_CONTEXT_COVERAGE == 0.95
    assert MIN_GLOBAL_AUC == 0.55
    assert MIN_SAME_MINUTE_AUC == 0.53
    assert MIN_SELECTED_RATE == 0.10
    assert MIN_SELECTED_POSITIVE_RATE == 0.58
    assert MIN_GOOD_DAYS == 4
