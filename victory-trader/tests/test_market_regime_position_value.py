from __future__ import annotations

from victory_trader.market_regime_position_value import (
    MARKET_REGIME_FEATURES,
    MIN_GOOD_DAYS,
    MIN_MARKET_CONTEXT_COVERAGE,
    MIN_POOLED_SPEARMAN_IMPROVEMENT,
    MIN_SAME_MINUTE_MEDIAN_IMPROVEMENT,
    MIN_SELECTED_RATE,
    REQUEST_ID,
)


def test_request173_contract_is_frozen():
    assert REQUEST_ID == 173
    assert len(MARKET_REGIME_FEATURES) == 16
    assert MIN_MARKET_CONTEXT_COVERAGE == 0.95
    assert MIN_POOLED_SPEARMAN_IMPROVEMENT == 0.02
    assert MIN_SAME_MINUTE_MEDIAN_IMPROVEMENT == 0.01
    assert MIN_SELECTED_RATE == 0.10
    assert MIN_GOOD_DAYS == 4


def test_request173_contains_breadth_and_activity_context():
    required = {
        "market_runner_5_fraction",
        "market_runner_10_fraction",
        "market_1m_positive_fraction",
        "market_attention_top10_mean",
        "market_log_total_transactions",
    }
    assert required.issubset(set(MARKET_REGIME_FEATURES))
