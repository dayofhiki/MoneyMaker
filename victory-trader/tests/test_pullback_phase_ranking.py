from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from victory_trader.pullback_phase_ranking import (
    attach_phase_features,
    cross_sectional_leaders,
    first_crossing,
)


def test_phase_features_mark_short_turn_and_recovery() -> None:
    frame = pd.DataFrame(
        [
            {
                "watch_drawdown_from_peak_pct": -4.0,
                "watch_rebound_from_trough_pct": 1.0,
                "sec_prev5_return_pct": -1.0,
                "sec_last5_return_pct": 0.5,
                "sec_prev10_return_pct": -0.8,
                "sec_last10_return_pct": 0.2,
                "sec_prev15_return_pct": -0.5,
                "sec_last15_return_pct": 0.1,
                "sec_seconds_since_low": 4.0,
                "sec_seconds_since_high": 30.0,
                "sec_close_vs_vwap_pct": 0.2,
                "watch_close_vs_vwap_change_1m": 0.1,
                "sec_signed_volume_imbalance_60": 0.3,
                "sec_signed_transactions_imbalance_60": 0.2,
                "sec_return_efficiency_60": 0.8,
                "sec_sign_flip_rate_60": 0.1,
                "sec_volume_burst_5": 2.0,
                "sec_transactions_burst_5": 1.5,
                "watch_attention_drawdown_from_peak": -0.2,
                "watch_attention_change_1m": 0.4,
            }
        ]
    )

    out = attach_phase_features(frame).iloc[0]
    assert out["phase_pullback_depth_pct"] == pytest.approx(4.0)
    assert out["phase_rebound_pct"] == pytest.approx(1.0)
    assert out["phase_recovery_fraction"] == pytest.approx(0.25)
    assert out["phase_turn_5"] == pytest.approx(1.0)
    assert out["phase_turn_10"] == pytest.approx(1.0)
    assert out["phase_turn_15"] == pytest.approx(1.0)
    assert out["phase_rebound_after_recent_low"] == pytest.approx(1.0)
    assert out["phase_vwap_reclaim"] == pytest.approx(1.0)
    assert out["phase_flow_agreement"] == pytest.approx(1.0)
    assert out["phase_clean_turn"] > 0


def test_phase_features_do_not_invent_turn_from_missing_values() -> None:
    frame = pd.DataFrame(
        [
            {
                "watch_drawdown_from_peak_pct": np.nan,
                "watch_rebound_from_trough_pct": np.nan,
                "sec_prev5_return_pct": np.nan,
                "sec_last5_return_pct": 0.5,
            }
        ]
    )
    out = attach_phase_features(frame).iloc[0]
    assert pd.isna(out["phase_turn_5"])


def test_first_crossing_uses_earliest_threshold_state_per_episode() -> None:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": 1,
                "minutes_since_hot": 1,
                "p": 0.5,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": 1,
                "minutes_since_hot": 2,
                "p": 0.9,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": 1,
                "minutes_since_hot": 3,
                "p": 0.95,
            },
        ]
    )
    selected = first_crossing(
        frame, probability_column="p", threshold=0.8
    )
    assert len(selected) == 1
    assert selected.iloc[0]["minutes_since_hot"] == 2


def test_cross_sectional_leader_keeps_best_candidate_in_clock_bucket() -> None:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": 1,
                "minutes_since_hot": 1,
                "state_t": 120_000,
                "p": 0.80,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "BBB",
                "hot_t": 2,
                "minutes_since_hot": 1,
                "state_t": 125_000,
                "p": 0.90,
            },
            {
                "trading_day": "2026-06-23",
                "ticker": "AAA",
                "hot_t": 1,
                "minutes_since_hot": 2,
                "state_t": 180_000,
                "p": 0.95,
            },
        ]
    )
    leaders, context = cross_sectional_leaders(
        frame, probability_column="p", threshold=0.7
    )
    assert context["multi_candidate_buckets"] == 1
    assert context["max_candidates_in_bucket"] == 2
    assert list(leaders["ticker"]) == ["BBB", "AAA"]
