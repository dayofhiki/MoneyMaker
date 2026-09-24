from __future__ import annotations

from victory_trader.rich_second_quantile_patience import (
    LOWER_QUANTILE,
    MIN_EVAL_BOUND_COVERAGE,
    MIN_POSITIVE_DAYS,
    MIN_SELECTED_POSITIVE_RATE,
    MIN_SELECTED_RATE,
    REQUEST_ID,
)


def test_request172_contract_is_frozen():
    assert REQUEST_ID == 172
    assert LOWER_QUANTILE == 0.20
    assert MIN_EVAL_BOUND_COVERAGE == 0.70
    assert MIN_SELECTED_RATE == 0.05
    assert MIN_SELECTED_POSITIVE_RATE == 0.55
    assert MIN_POSITIVE_DAYS == 5
