from __future__ import annotations

from victory_trader.trade_tape_feasibility import (
    LOOKBACK_MS,
    MIN_ANY_TRADE_COVERAGE,
    MIN_FIVE_TRADE_COVERAGE,
    MIN_RECENT_TRADE_COVERAGE,
    RECENT_MS,
    REQUEST_ID,
    SAMPLE_PER_DAY,
)


def test_request170_contract_is_frozen():
    assert REQUEST_ID == 170
    assert SAMPLE_PER_DAY == 10
    assert LOOKBACK_MS == 60_000
    assert RECENT_MS == 10_000
    assert MIN_ANY_TRADE_COVERAGE == 0.80
    assert MIN_FIVE_TRADE_COVERAGE == 0.70
    assert MIN_RECENT_TRADE_COVERAGE == 0.60
