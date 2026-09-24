from __future__ import annotations

from victory_trader.direct_recurrent_hold_exit_advantage import (
    BOOTSTRAP_SAMPLES,
    FEATURES,
    MAX_HOLD_MINUTES,
    MIN_ADVANTAGE_SPEARMAN,
    MIN_GOOD_DAYS,
    MIN_START_COVERAGE,
    MODEL_SEED,
    REQUEST_ID,
)
from victory_trader.market_regime_position_value import MARKET_REGIME_FEATURES
from victory_trader.rich_second_position_value import RICH_SECOND_FEATURES
from victory_trader.selected_hot_position_value_observability import MODEL_FEATURES


def test_request179_contract_is_frozen():
    assert REQUEST_ID == 179
    assert MODEL_SEED == 20261079
    assert BOOTSTRAP_SAMPLES == 10_000
    assert MAX_HOLD_MINUTES == 30
    assert MIN_START_COVERAGE == 0.80
    assert MIN_ADVANTAGE_SPEARMAN == 0.05
    assert MIN_GOOD_DAYS == 3


def test_request179_uses_all_current_causal_feature_families():
    for column in MODEL_FEATURES:
        assert column in FEATURES
    for column in RICH_SECOND_FEATURES:
        assert column in FEATURES
    for column in MARKET_REGIME_FEATURES:
        assert column in FEATURES
