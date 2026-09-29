import numpy as np
import pandas as pd
import pytest

from victory_trader.conditional_candidate_value import (
    score_conditional,
    threshold_from_training,
)


class FakeClassifier:
    def __init__(self, probability):
        self.probability = float(probability)

    def predict_proba(self, x):
        p = np.full(len(x), self.probability)
        return np.column_stack([1.0 - p, p])


class FakeRegressor:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, x):
        return np.full(len(x), self.value)


class Heads:
    pullback = FakeClassifier(0.25)
    trade_value = FakeRegressor(4.0)
    trade_positive = FakeClassifier(0.6)
    trade_severe = FakeClassifier(0.2)


def frame():
    return pd.DataFrame([
        {
            "trading_day": "2026-05-05",
            "ticker": "AAA",
            "hot_t": 1,
            "feature": 1.0,
        },
        {
            "trading_day": "2026-05-05",
            "ticker": "BBB",
            "hot_t": 2,
            "feature": 2.0,
        },
    ])


def test_expected_value_multiplies_pullback_probability():
    scored = score_conditional(
        Heads(),
        frame(),
        ("feature",),
    )
    assert scored.expected_candidate_value_pct.tolist() == pytest.approx(
        [1.0, 1.0]
    )
    assert scored.positive_probability_mass.tolist() == pytest.approx(
        [0.15, 0.15]
    )
    assert scored.severe_probability_mass.tolist() == pytest.approx(
        [0.05, 0.05]
    )


def test_threshold_uses_only_supplied_training_scores():
    train = np.arange(100, dtype=float)
    threshold = threshold_from_training(train, fraction=0.20)
    assert threshold == pytest.approx(np.quantile(train, 0.80))
