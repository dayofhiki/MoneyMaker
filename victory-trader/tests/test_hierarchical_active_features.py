from __future__ import annotations

from victory_trader.hierarchical_active_features import (
    BASELINE_ACTIVE_FEATURES,
    FOCUS_CAP,
    HIERARCHICAL_ACTIVE_FEATURES,
    TARGET_CAPTURE,
    TAIL_TARGET_CAPTURED,
    diagnose,
)


def test_request159_feature_contract():
    assert FOCUS_CAP == 180
    assert TARGET_CAPTURE == 0.95
    assert TAIL_TARGET_CAPTURED == 7
    assert "market_hazard_probability" not in BASELINE_ACTIVE_FEATURES
    assert "market_rank" not in BASELINE_ACTIVE_FEATURES
    assert "market_hazard_probability" in HIERARCHICAL_ACTIVE_FEATURES
    assert "market_rank" in HIERARCHICAL_ACTIVE_FEATURES


def test_diagnose_candidate():
    baseline = {
        "active_capture": {"supported_capture_rate": 0.946},
        "active_count": {"mean": 21.5},
        "rank_bucket_capture": {
            "rank_61_180": {"captured": 4},
        },
    }
    hierarchical = {
        "active_capture": {"supported_capture_rate": 0.952},
        "active_count": {"mean": 22.0},
        "rank_bucket_capture": {
            "rank_61_180": {"captured": 7},
        },
    }
    result = diagnose(baseline, hierarchical)
    assert result["diagnosis"] == "hierarchical_focus_context_candidate"


def test_diagnose_partial_tail_gain():
    baseline = {
        "active_capture": {"supported_capture_rate": 0.946},
        "active_count": {"mean": 21.5},
        "rank_bucket_capture": {
            "rank_61_180": {"captured": 4},
        },
    }
    hierarchical = {
        "active_capture": {"supported_capture_rate": 0.949},
        "active_count": {"mean": 21.0},
        "rank_bucket_capture": {
            "rank_61_180": {"captured": 6},
        },
    }
    result = diagnose(baseline, hierarchical)
    assert (
        result["diagnosis"]
        == "hierarchical_focus_context_partial_tail_gain"
    )


def test_diagnose_no_mechanism_gain():
    baseline = {
        "active_capture": {"supported_capture_rate": 0.946},
        "active_count": {"mean": 21.5},
        "rank_bucket_capture": {
            "rank_61_180": {"captured": 4},
        },
    }
    hierarchical = {
        "active_capture": {"supported_capture_rate": 0.947},
        "active_count": {"mean": 20.8},
        "rank_bucket_capture": {
            "rank_61_180": {"captured": 5},
        },
    }
    result = diagnose(baseline, hierarchical)
    assert result["diagnosis"] == "hierarchical_focus_context_no_mechanism_gain"
