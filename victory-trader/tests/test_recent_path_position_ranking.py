from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.recent_path_position_ranking import (
    CANDIDATE_FEATURES,
    attach_rank_target,
    attach_recent_path_features,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "state_t": 60_000,
                "entry_to_current_close_pct": 0.0,
                "hold_advantage_event_pct": -1.0,
            },
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "state_t": 120_000,
                "entry_to_current_close_pct": 1.0,
                "hold_advantage_event_pct": 0.0,
            },
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "state_t": 240_000,
                "entry_to_current_close_pct": 3.0,
                "hold_advantage_event_pct": 2.0,
            },
        ]
    )


def test_recent_path_features_are_causal() -> None:
    original = attach_recent_path_features(_frame())
    changed_source = _frame()
    changed_source.loc[2, "entry_to_current_close_pct"] = 99.0
    changed = attach_recent_path_features(changed_source)

    columns = [
        "path_entry_to_current_close_pct_delta1",
        "path_entry_to_current_close_pct_slope3_per_min",
        "path_entry_to_current_close_pct_std3",
        "path_entry_to_current_close_pct_range3",
    ]
    for column in columns:
        a = original.loc[1, column]
        b = changed.loc[1, column]
        if pd.isna(a):
            assert pd.isna(b)
        else:
            assert np.isclose(a, b)


def test_recent_path_slope_respects_irregular_event_time() -> None:
    enriched = attach_recent_path_features(_frame())
    row = enriched.iloc[2]
    assert np.isclose(
        row["path_entry_to_current_close_pct_delta1"],
        2.0,
    )
    assert np.isclose(
        row["path_entry_to_current_close_pct_slope3_per_min"],
        1.0,
    )


def test_rank_target_is_within_day_and_monotone() -> None:
    frame = _frame()
    second_day = frame.copy()
    second_day["trading_day"] = "2026-05-02"
    second_day["hold_advantage_event_pct"] = [100.0, 200.0, 300.0]
    ranked = attach_rank_target(
        pd.concat([frame, second_day], ignore_index=True)
    )
    day1 = ranked.loc[ranked["trading_day"].eq("2026-05-01")]
    day2 = ranked.loc[ranked["trading_day"].eq("2026-05-02")]
    assert np.allclose(
        day1["hold_advantage_day_rank"],
        [1 / 3, 2 / 3, 1.0],
    )
    assert np.allclose(
        day2["hold_advantage_day_rank"],
        [1 / 3, 2 / 3, 1.0],
    )


def test_candidate_feature_family_has_no_future_observation_fields() -> None:
    forbidden = {
        "next_event_t",
        "next_event_base_return_pct",
        "time_to_next_observation",
        "hold_advantage_event_pct",
    }
    assert forbidden.isdisjoint(CANDIDATE_FEATURES)
