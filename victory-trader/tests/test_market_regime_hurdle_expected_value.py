from __future__ import annotations

from victory_trader.market_regime_hurdle_expected_value import (
    MIN_CONTEXT_COVERAGE,
    MIN_EV_SPEARMAN,
    MIN_GOOD_DAYS,
    MIN_SELECTED_RATE,
    MIN_SELECTED_REALIZED_MEAN_PCT,
    REQUEST_ID,
)


def test_request176_contract_is_frozen():
    assert REQUEST_ID == 176
    assert MIN_CONTEXT_COVERAGE == 0.95
    assert MIN_EV_SPEARMAN == 0.08
    assert MIN_SELECTED_RATE == 0.10
    assert MIN_SELECTED_REALIZED_MEAN_PCT == 0.20
    assert MIN_GOOD_DAYS == 4
