import numpy as np
import pandas as pd

import victory_trader.expanded_path_aware_continuation as v35
from victory_trader.execution_costs import modeled_sell_fill
from victory_trader.expanded_path_aware_continuation import (
    BASE_SCENARIO,
    MINUTE_MS,
    PATH_FEATURES,
    build_continuation_rows,
    continuation_feature_frame,
)


def _state(times):
    rows = []
    for minute, timestamp in enumerate(times):
        o = 10.0 + minute
        rows.append(
            {
                "trading_day": "2026-01-02",
                "ticker": "AAA",
                "t": timestamp,
                "o": o,
                "h": o + 1.0,
                "l": o - 0.5,
                "c": o + 0.5,
                "previous_close": 9.0,
                "trailing_return_1m_pct": float(minute),
                "hod_distance_pct": -float(minute),
                "regular_vwap_distance_pct": float(minute),
                "volume_accel_1m_vs_prior20m": 1.0,
                "bar_range_pct": 1.0,
                "bar_close_location": 0.5,
                "phase": "impulse",
            }
        )
    return pd.DataFrame(rows)


def test_exact_next_opens_and_entry_relative_path_are_causal():
    state = _state([0, MINUTE_MS, 2 * MINUTE_MS, 3 * MINUTE_MS])
    anchors = pd.DataFrame(
        [{
            "trading_day": "2026-01-02",
            "ticker": "AAA",
            "t": 0,
            "opportunity_probability": 0.8,
            "filing_semantic_accessions_30d": 2.0,
        }]
    )
    result = build_continuation_rows(state, anchors)
    first = result.loc[result["minutes_held"].eq(1)].iloc[0]

    expected_hold = (
        modeled_sell_fill(13.0, BASE_SCENARIO)
        / modeled_sell_fill(12.0, BASE_SCENARIO)
        - 1.0
    ) * 100.0
    assert np.isclose(first["hold_advantage_pct"], expected_hold)
    assert first["entry_open"] == 11.0
    assert np.isclose(first["path_entry_return_pct"], (12.0 / 11.0 - 1.0) * 100.0)
    assert np.isclose(first["path_mfe_pct"], (12.0 / 11.0 - 1.0) * 100.0)
    assert np.isclose(first["path_mae_pct"], (10.5 / 11.0 - 1.0) * 100.0)
    assert np.isclose(first["path_drawdown_from_high_pct"], 0.0)
    assert np.isclose(first["path_observed_fraction"], 1.0)
    assert first["path_minutes_since_high"] == 0.0


def test_path_features_are_model_inputs(monkeypatch):
    state = _state([0, MINUTE_MS, 2 * MINUTE_MS, 3 * MINUTE_MS])
    anchors = pd.DataFrame(
        [{"trading_day": "2026-01-02", "ticker": "AAA", "t": 0}]
    )
    rows = build_continuation_rows(state, anchors)
    monkeypatch.setattr(
        v35,
        "filing_semantic_split_supply_feature_frame",
        lambda frame: pd.DataFrame(index=frame.index),
    )
    features = continuation_feature_frame(rows)
    assert set(PATH_FEATURES).issubset(features.columns)
    evaluated = rows["evaluation_reason"].eq("evaluated")
    assert features.loc[evaluated, "path_entry_return_pct"].notna().all()


def test_missing_minute_is_not_skipped_for_exit_or_hold():
    state = _state([0, MINUTE_MS, 3 * MINUTE_MS, 4 * MINUTE_MS])
    anchors = pd.DataFrame(
        [{"trading_day": "2026-01-02", "ticker": "AAA", "t": 0}]
    )
    result = build_continuation_rows(state, anchors)
    first = result.loc[result["minutes_held"].eq(1)].iloc[0]
    assert first["evaluation_reason"] == "missing_exact_exit_open"
    assert pd.isna(first["hold_advantage_pct"])


def test_follow_up_window_excludes_entry_instant_and_thirty_minutes():
    state = _state([minute * MINUTE_MS for minute in range(33)])
    anchors = pd.DataFrame(
        [{"trading_day": "2026-01-02", "ticker": "AAA", "t": 0}]
    )
    result = build_continuation_rows(state, anchors)
    assert result["minutes_held"].min() == 1
    assert result["minutes_held"].max() == 29
    assert len(result) == 29
