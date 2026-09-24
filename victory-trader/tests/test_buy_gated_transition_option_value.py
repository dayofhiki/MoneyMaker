from __future__ import annotations

from victory_trader.buy_gated_path_transition_observability import (
    PATH_TRANSITION_LAGS,
)
from victory_trader.buy_gated_transition_option_value import (
    MIN_POOLED_SPEARMAN_IMPROVEMENT,
    MIN_SAME_MINUTE_MEDIAN_IMPROVEMENT,
    MIN_SELECTED_RATE,
    REQUEST_ID,
)


def test_request168_contract_is_frozen():
    assert REQUEST_ID == 168
    assert PATH_TRANSITION_LAGS == (1, 2, 3, 5, 8, 13)
    assert MIN_POOLED_SPEARMAN_IMPROVEMENT == 0.02
    assert MIN_SAME_MINUTE_MEDIAN_IMPROVEMENT == 0.01
    assert MIN_SELECTED_RATE == 0.10
