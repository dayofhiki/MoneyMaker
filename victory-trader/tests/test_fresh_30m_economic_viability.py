from __future__ import annotations

from victory_trader.fresh_30m_economic_viability import (
    CLASSIFIER_SEED,
    MIN_AUC,
    MIN_ENTRY_COVERAGE,
    MIN_POSITIVE_DAYS,
    MIN_POSITIVE_SPEARMAN_DAYS,
    MIN_SECOND_COVERAGE,
    MIN_SPEARMAN,
    REGRESSOR_SEED,
    REQUEST_ID,
)


def test_request190_contract_is_frozen():
    assert REQUEST_ID == 190
    assert CLASSIFIER_SEED == 20261092
    assert REGRESSOR_SEED == 20261093
    assert MIN_ENTRY_COVERAGE == 0.90
    assert MIN_SECOND_COVERAGE == 0.90
    assert MIN_AUC == 0.60
    assert MIN_SPEARMAN == 0.15
    assert MIN_POSITIVE_SPEARMAN_DAYS == 4
    assert MIN_POSITIVE_DAYS == 4
