from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.learned_sequence_position_ranking import (
    LEARNED_SEQUENCE_FEATURES,
    SEQUENCE_LENGTH,
    attach_sequence_features,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "state_t": 60_000,
                "entry_to_current_close_pct": 1.0,
                "elapsed_since_previous_observation_min": 1.0,
                "minutes_held": 1.0,
            },
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "state_t": 120_000,
                "entry_to_current_close_pct": 2.0,
                "elapsed_since_previous_observation_min": 1.0,
                "minutes_held": 2.0,
            },
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "state_t": 300_000,
                "entry_to_current_close_pct": 5.0,
                "elapsed_since_previous_observation_min": 3.0,
                "minutes_held": 5.0,
            },
        ]
    )


def test_sequence_uses_current_and_past_only() -> None:
    base = attach_sequence_features(_frame())
    changed = _frame()
    changed.loc[2, "entry_to_current_close_pct"] = 999.0
    changed = attach_sequence_features(changed)

    columns = [
        "seq_lag0_entry_to_current_close_pct",
        "seq_lag1_entry_to_current_close_pct",
        "seq_lag1_age_min",
    ]
    for column in columns:
        a = base.loc[1, column]
        b = changed.loc[1, column]
        if pd.isna(a):
            assert pd.isna(b)
        else:
            assert np.isclose(a, b)


def test_sequence_preserves_irregular_event_time() -> None:
    enriched = attach_sequence_features(_frame())
    row = enriched.iloc[2]
    assert np.isclose(
        row["seq_lag0_entry_to_current_close_pct"], 5.0
    )
    assert np.isclose(
        row["seq_lag1_entry_to_current_close_pct"], 2.0
    )
    assert np.isclose(row["seq_lag1_age_min"], 3.0)
    assert np.isclose(row["seq_lag2_age_min"], 4.0)


def test_sequence_left_history_is_masked() -> None:
    enriched = attach_sequence_features(_frame())
    first = enriched.iloc[0]
    assert first["seq_lag0_mask"] == 1.0
    for lag in range(1, SEQUENCE_LENGTH):
        assert first[f"seq_lag{lag}_mask"] == 0.0
        assert pd.isna(first[f"seq_lag{lag}_age_min"])


def test_learned_sequence_features_exclude_future_fields() -> None:
    forbidden = {
        "next_event_t",
        "next_event_base_return_pct",
        "hold_advantage_event_pct",
        "time_to_next_observation",
        "event_exit_minute_after_hot",
    }
    assert forbidden.isdisjoint(LEARNED_SEQUENCE_FEATURES)
