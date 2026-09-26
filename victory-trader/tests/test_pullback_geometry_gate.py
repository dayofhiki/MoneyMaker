from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.pullback_geometry_gate import (
    THRESHOLD_QUANTILES,
    attach_geometry_features,
    gated_policy_rows,
)


def test_geometry_features_are_causal_algebra() -> None:
    frame = pd.DataFrame(
        {
            "position_return_change_1m_pct": [-0.6, -0.2],
            "previous_observed_return_change_pct": [0.4, 0.8],
            "position_return_accel_1m_pct": [-1.0, -1.0],
            "entry_to_current_close_pct": [2.0, 0.5],
        }
    )
    enriched = attach_geometry_features(frame)
    assert np.allclose(
        enriched["geometry_onset_drop_pct"].to_numpy(dtype=float),
        [0.6, 0.2],
    )
    assert np.allclose(
        enriched["geometry_previous_rise_pct"].to_numpy(dtype=float),
        [0.4, 0.8],
    )
    assert np.allclose(
        enriched["geometry_reversal_ratio"].to_numpy(dtype=float),
        [1.5, 0.25],
    )
    assert np.allclose(
        enriched["geometry_abs_accel_pct"].to_numpy(dtype=float),
        [1.0, 1.0],
    )
    assert np.allclose(
        enriched["geometry_entry_cushion_pct"].to_numpy(dtype=float),
        [2.0, 0.5],
    )


def test_gate_exits_ineligible_segments_at_onset() -> None:
    onset = pd.DataFrame(
        {
            "pullback_segment_id": ["a", "b"],
            "trading_day": ["2026-05-01", "2026-05-01"],
            "ticker": ["AAA", "BBB"],
            "hot_t": [1, 2],
            "segment_terminal_advantage_pct": [1.0, -1.0],
            "eligibility_score": [0.7, -0.2],
        }
    )
    adaptive = pd.DataFrame(
        {
            "pullback_segment_id": ["a", "b"],
            "trading_day": ["2026-05-01", "2026-05-01"],
            "ticker": ["AAA", "BBB"],
            "hot_t": [1, 2],
            "policy": ["request223", "request223"],
            "local_uplift_pct": [0.5, -0.8],
        }
    )
    gated = gated_policy_rows(onset, adaptive, threshold=0.0)
    values = gated.set_index("pullback_segment_id")[
        "local_uplift_pct"
    ].to_dict()
    assert np.isclose(values["a"], 0.5)
    assert np.isclose(values["b"], 0.0)


def test_threshold_grid_is_frozen_and_small() -> None:
    assert THRESHOLD_QUANTILES == (0.40, 0.50, 0.60, 0.70, 0.80)


def test_future_columns_are_not_required_for_geometry() -> None:
    base = pd.DataFrame(
        {
            "position_return_change_1m_pct": [-0.5],
            "previous_observed_return_change_pct": [0.5],
            "position_return_accel_1m_pct": [-1.0],
            "entry_to_current_close_pct": [1.0],
        }
    )
    future_changed = base.copy()
    future_changed["pullback_terminal_exit_pct"] = [999.0]
    future_changed["adaptive_wait_value_pct"] = [999.0]
    a = attach_geometry_features(base)
    b = attach_geometry_features(future_changed)
    columns = [
        "geometry_onset_drop_pct",
        "geometry_previous_rise_pct",
        "geometry_reversal_ratio",
        "geometry_abs_accel_pct",
        "geometry_entry_cushion_pct",
    ]
    assert np.allclose(
        a[columns].to_numpy(dtype=float),
        b[columns].to_numpy(dtype=float),
    )
