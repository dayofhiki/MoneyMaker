import pandas as pd
import pytest

from victory_trader.shortlist_adapted_risk import (
    nested_training_stage1,
    threshold_from_shortlist_train,
)


def test_threshold_prefers_training_shortlist_support():
    probabilities = [0.1, 0.2, 0.8, 0.9, 0.95] * 7
    selected = pd.Series(
        [False, False, True, True, True] * 7
    )
    threshold = threshold_from_shortlist_train(
        probabilities,
        selected,
    )
    assert threshold >= 0.8


def test_nested_training_stage1_requires_three_outer_days():
    frame = pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 1,
        },
        {
            "trading_day": "2026-05-06",
            "ticker": "BBB",
            "hot_t": 2,
        },
    ])
    with pytest.raises(ValueError, match="three outer-training days"):
        nested_training_stage1(
            frame,
            ("x",),
            1,
        )
