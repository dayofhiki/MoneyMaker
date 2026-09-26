import math
from datetime import date

import pandas as pd

from victory_trader.split_history_first_passage_directional_edge import (
    add_split_features,
    coverage,
)


def test_split_features_are_strictly_prior_and_log_transformed():
    events = {
        "TEST": [
            {
                "_execution_date": date(2025, 12, 1),
                "adjustment_type": "reverse_split",
                "split_from": 20.0,
                "split_to": 1.0,
            },
            {
                "_execution_date": date(2026, 5, 12),
                "adjustment_type": "reverse_split",
                "split_from": 10.0,
                "split_to": 1.0,
            },
        ]
    }
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
            }
        ]
    )
    result = add_split_features(frame, events)
    row = result.iloc[0]
    assert row["reverse_split_count_730d"] == 1.0
    assert row["reverse_split_count_365d"] == 1.0
    assert math.isclose(
        row["split_log_latest_reverse_consolidation"],
        math.log(20.0),
    )
    assert row["split_history_strict_prior"] == 1.0
    cov = coverage(result)
    assert cov["query_success"] == 1.0
    assert cov["strict_prior"] is True
    assert cov["reverse_split_rate_730d"] == 1.0


def test_no_split_history_is_valid_zero_context():
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
            }
        ]
    )
    result = add_split_features(frame, {})
    row = result.iloc[0]
    assert row["split_any_730d"] == 0.0
    assert row["reverse_split_count_730d"] == 0.0
    assert pd.isna(
        row["split_log_latest_reverse_consolidation"]
    )
