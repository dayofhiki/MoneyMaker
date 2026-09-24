from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.attention_replay import MINUTE_MS
from victory_trader.pre_entry_shadow_delayed_viability import (
    CLASSIFIER_SEED,
    FEATURES,
    MAX_FIRST_EPISODE_RATE,
    MIN_AUC,
    MIN_FIRST_EPISODE_RATE,
    MIN_FIRST_POSITIVE_RATE,
    MIN_FIRST_VALUE_UPLIFT_PCT,
    MIN_POSITIVE_FIRST_DAYS,
    MIN_POSITIVE_SPEARMAN_DAYS,
    MIN_VALUE_SPEARMAN,
    PROBABILITY_GATE_QUANTILE,
    REGRESSOR_SEED,
    REQUEST_ID,
    SHADOW_MAX_MINUTE,
    SHADOW_MIN_MINUTE,
    build_delayed_labels,
)


def test_request193_contract_is_frozen():
    assert REQUEST_ID == 193
    assert CLASSIFIER_SEED == 20261096
    assert REGRESSOR_SEED == 20261097
    assert SHADOW_MIN_MINUTE == 1
    assert SHADOW_MAX_MINUTE == 10
    assert PROBABILITY_GATE_QUANTILE == 0.75
    assert MIN_AUC == 0.60
    assert MIN_VALUE_SPEARMAN == 0.15
    assert MIN_POSITIVE_SPEARMAN_DAYS == 4
    assert MIN_FIRST_EPISODE_RATE == 0.10
    assert MAX_FIRST_EPISODE_RATE == 0.70
    assert MIN_FIRST_POSITIVE_RATE == 0.60
    assert MIN_FIRST_VALUE_UPLIFT_PCT == 0.50
    assert MIN_POSITIVE_FIRST_DAYS == 4


def test_delayed_label_reprices_from_delayed_entry():
    base = 1_000_000
    positions = pd.DataFrame(
        {
            "trading_day": ["2026-05-01", "2026-05-01"],
            "ticker": ["TEST", "TEST"],
            "hot_t": [base, base],
            "state_t": [base + MINUTE_MS, base + 2 * MINUTE_MS],
            "minutes_held": [1.0, 2.0],
        }
    )
    # _open_map maps bar timestamp t to actual open timestamp t-MINUTE_MS.
    scan = pd.DataFrame(
        {
            "trading_day": ["2026-05-01"] * 5,
            "ticker": ["TEST"] * 5,
            "t": [
                base + MINUTE_MS,
                base + 2 * MINUTE_MS,
                base + 3 * MINUTE_MS,
                base + 4 * MINUTE_MS,
                base + 5 * MINUTE_MS,
            ],
            "o": [10.0, 10.0, 10.5, 11.0, 12.0],
        }
    )
    result = build_delayed_labels(positions, scan)
    assert len(result) == 2
    assert np.isfinite(
        result["delayed_future_best_base_return_pct"]
    ).all()
    assert (
        result["delayed_future_exit_count"].to_numpy() > 0
    ).all()


def test_execution_and_oracle_fields_are_not_features():
    forbidden = {
        "exit_reference_open",
        "exit_now_base_return_pct",
        "next_minute_base_return_pct",
        "best_future_base_return_pct",
        "remaining_option_value_pct",
        "delayed_entry_reference_open",
        "delayed_future_best_base_return_pct",
        "delayed_future_cost_coverable",
    }
    assert forbidden.isdisjoint(FEATURES)
