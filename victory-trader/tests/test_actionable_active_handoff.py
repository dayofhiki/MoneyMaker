from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.actionable_active_handoff import (
    add_actionable_priority,
    add_unconditional_cross_target,
)


class _FixedObservabilityModel:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probability = pd.to_numeric(
            features["attention_score"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0).to_numpy()
        return np.column_stack([1.0 - probability, probability])


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market_hazard_probability": 0.8,
                "attention_score": 0.25,
                "attention_rank": 1.0,
                "return_from_previous_close_pct": 2.0,
                "minute_body_return_pct": 0.1,
                "minute_range_pct": 0.2,
                "log_minute_volume": 3.0,
                "log_minute_transactions": 2.0,
                "minute_return_1m_pct": 0.1,
                "return_accel_1m_pct": 0.0,
                "volume_ratio_prev1": 1.0,
                "transactions_ratio_prev1": 1.0,
                "target_next_cross": 1,
            },
            {
                "market_hazard_probability": 0.8,
                "attention_score": 0.75,
                "attention_rank": 2.0,
                "return_from_previous_close_pct": 2.0,
                "minute_body_return_pct": 0.1,
                "minute_range_pct": 0.2,
                "log_minute_volume": 3.0,
                "log_minute_transactions": 2.0,
                "minute_return_1m_pct": 0.1,
                "return_accel_1m_pct": 0.0,
                "volume_ratio_prev1": 1.0,
                "transactions_ratio_prev1": 1.0,
                "target_next_cross": np.nan,
            },
        ]
    )


def test_actionable_priority_is_hazard_times_causal_observability_probability():
    result = add_actionable_priority(_rows(), _FixedObservabilityModel())

    assert result["next_minute_observability_probability"].tolist() == [0.25, 0.75]
    np.testing.assert_allclose(
        result["active_priority"].to_numpy(),
        np.array([0.2, 0.6]),
    )


def test_missing_exact_next_minute_means_no_exact_next_minute_cross_event():
    result = add_unconditional_cross_target(_rows())

    assert result["target_next_cross"].tolist() == [1, 0]
