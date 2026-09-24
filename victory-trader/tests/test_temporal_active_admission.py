from __future__ import annotations

from victory_trader.temporal_active_admission import (
    HORIZONS,
    PRIMARY_HORIZON,
    _diagnose,
)


def _metrics(retention8: float, retention20: float, churn: float):
    return {
        "8": {
            "active_retention_given_focus": retention8,
            "transport_audit": {
                "mean_post_initial_subscription_additions_per_decision": churn,
            },
        },
        "20": {
            "active_retention_given_focus": retention20,
            "transport_audit": {
                "mean_post_initial_subscription_additions_per_decision": churn,
            },
        },
    }


def test_request155_horizons_are_frozen():
    assert HORIZONS == (1, 2, 3, 5)
    assert PRIMARY_HORIZON == 3


def test_temporal_candidate_needs_target_gain_and_nonhigher_churn():
    legacy = _metrics(0.934, 0.928, 7.8)
    primary = _metrics(0.956, 0.958, 7.2)
    result = _diagnose(legacy, primary)
    assert result["diagnosis"] == "temporal_policy_candidate"


def test_allocator_mismatch_when_full_refresh_passes_but_cap8_does_not():
    legacy = _metrics(0.934, 0.928, 7.8)
    primary = _metrics(0.944, 0.956, 7.0)
    result = _diagnose(legacy, primary)
    assert result["diagnosis"] == "temporal_ranking_candidate_allocator_mismatch"


def test_small_gain_is_insufficient():
    legacy = _metrics(0.934, 0.928, 7.8)
    primary = _metrics(0.940, 0.942, 7.0)
    result = _diagnose(legacy, primary)
    assert result["diagnosis"] == "temporal_target_insufficient"
