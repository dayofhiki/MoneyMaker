from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.hot_economic_opportunity_gate import (
    FEATURES,
    _event_features,
    fit_hurdle_gate,
    score_gate,
)


def test_event_features_adds_hot_rank_and_episode_context():
    trace = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-07",
                "ticker": "A",
                "t": 1778151000000,
                "state": "hot",
                "reason": "learned_hot_promote",
                "learned_hot_score": 0.8,
                "attention_score": 0.9,
            },
            {
                "trading_day": "2026-05-07",
                "ticker": "B",
                "t": 1778151000000,
                "state": "hot",
                "reason": "learned_hot_promote",
                "learned_hot_score": 0.6,
                "attention_score": 0.8,
            },
        ]
    )

    events = _event_features(trace)

    assert len(events) == 2
    assert events.loc[events["ticker"].eq("A"), "hot_rank"].iloc[0] == 1
    assert events.loc[events["ticker"].eq("B"), "hot_rank"].iloc[0] == 2
    assert set(FEATURES).issubset(events.columns)


def _synthetic_panel(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hot = rng.uniform(0.001, 0.2, rows)
    promotion = rng.integers(1, 6, rows)
    noise = rng.normal(0, 0.8, rows)
    target = 8.0 * hot - 0.22 * (promotion - 1) + noise

    frame = pd.DataFrame(
        {
            "learned_hot_score": hot,
            "log_learned_hot_score": np.log(hot),
            "attention_score": rng.uniform(0.5, 1.0, rows),
            "hot_rank": rng.integers(1, 11, rows),
            "score_margin_to_hot_floor": rng.uniform(0, 0.1, rows),
            "promotion_index": promotion,
            "minutes_since_previous_promotion": rng.uniform(1, 30, rows),
            "is_repromotion": (promotion > 1).astype(float),
            "minutes_from_regular_open": rng.uniform(0, 360, rows),
            "oracle_best_base_pct": target,
        }
    )
    return frame


def test_hurdle_gate_scores_finite_expected_value():
    fit = _synthetic_panel(1600, 1)
    calibration = _synthetic_panel(800, 2)
    evaluation = _synthetic_panel(300, 3)

    gate = fit_hurdle_gate(fit, calibration)
    scored = score_gate(evaluation, gate)

    assert scored["predicted_opportunity_probability"].between(0, 1).all()
    assert np.isfinite(scored["predicted_oracle_ev_pct"]).all()
    assert scored["economic_gate_positive"].dtype == bool
