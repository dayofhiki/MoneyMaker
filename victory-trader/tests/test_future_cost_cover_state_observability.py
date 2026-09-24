from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.future_cost_cover_state_observability import (
    CLASSIFIER_SEED,
    EARLY_MAX_MINUTE,
    FEATURES,
    MIN_AUC,
    MIN_EARLY_AUC,
    MIN_EARLY_VALUE_SPEARMAN,
    MIN_POSITIVE_SELECTED_DAYS,
    MIN_POSITIVE_SPEARMAN_DAYS,
    MIN_SELECTED_PREVALENCE_UPLIFT,
    MIN_SELECTED_VALUE_UPLIFT_PCT,
    MIN_VALUE_SPEARMAN,
    PROBABILITY_GATE_QUANTILE,
    REGRESSOR_SEED,
    REQUEST_ID,
    _day_weights,
    add_targets,
)


def test_request191_contract_is_frozen():
    assert REQUEST_ID == 191
    assert CLASSIFIER_SEED == 20261094
    assert REGRESSOR_SEED == 20261095
    assert PROBABILITY_GATE_QUANTILE == 0.75
    assert EARLY_MAX_MINUTE == 10
    assert MIN_AUC == 0.60
    assert MIN_VALUE_SPEARMAN == 0.15
    assert MIN_EARLY_AUC == 0.60
    assert MIN_EARLY_VALUE_SPEARMAN == 0.15
    assert MIN_POSITIVE_SPEARMAN_DAYS == 4
    assert MIN_SELECTED_PREVALENCE_UPLIFT == 0.10
    assert MIN_SELECTED_VALUE_UPLIFT_PCT == 0.50
    assert MIN_POSITIVE_SELECTED_DAYS == 4


def test_future_cost_cover_target_is_strictly_positive():
    frame = pd.DataFrame(
        {
            "best_future_base_return_pct": [
                -1.0,
                0.0,
                0.01,
                np.nan,
            ]
        }
    )
    result = add_targets(frame)
    target = result["future_cost_coverable"].tolist()
    assert not bool(target[0])
    assert not bool(target[1])
    assert bool(target[2])
    assert pd.isna(target[3])


def test_equal_day_weights_give_equal_total_mass_per_day():
    frame = pd.DataFrame(
        {
            "trading_day": [
                "2026-05-01",
                "2026-05-01",
                "2026-05-02",
                "2026-05-02",
                "2026-05-02",
                "2026-05-02",
            ]
        }
    )
    weights = _day_weights(frame)
    first = float(weights[:2].sum())
    second = float(weights[2:].sum())
    assert np.isclose(first, second)


def test_future_labels_and_execution_references_are_not_features():
    forbidden = {
        "best_future_base_return_pct",
        "future_cost_coverable",
        "exit_reference_open",
        "exit_now_base_return_pct",
        "next_minute_base_return_pct",
        "remaining_option_value_pct",
        "hold_advantage_1m_pct",
    }
    assert forbidden.isdisjoint(FEATURES)
