from __future__ import annotations

from victory_trader.rich_second_position_value import (
    MIN_EVAL_FEATURE_COVERAGE,
    MIN_GOOD_DAYS,
    MIN_POOLED_SPEARMAN_IMPROVEMENT,
    MIN_SAME_MINUTE_MEDIAN_IMPROVEMENT,
    MIN_SELECTED_RATE,
    REQUEST_ID,
    RICH_SECOND_FEATURES,
)


def test_request171_contract_is_frozen():
    assert REQUEST_ID == 171
    assert len(RICH_SECOND_FEATURES) == 20
    assert MIN_EVAL_FEATURE_COVERAGE == 0.85
    assert MIN_POOLED_SPEARMAN_IMPROVEMENT == 0.03
    assert MIN_SAME_MINUTE_MEDIAN_IMPROVEMENT == 0.02
    assert MIN_SELECTED_RATE == 0.10
    assert MIN_GOOD_DAYS == 4


def test_request171_contains_direction_and_exhaustion_proxies():
    required = {
        "sec_signed_volume_imbalance_60",
        "sec_signed_transactions_imbalance_60",
        "sec_return_efficiency_60",
        "sec_sign_flip_rate_60",
        "sec_seconds_since_high",
        "sec_close_vs_vwap_pct",
    }
    assert required.issubset(set(RICH_SECOND_FEATURES))
