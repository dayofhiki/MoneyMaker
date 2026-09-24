from __future__ import annotations

from victory_trader.active_admission_score_family import (
    _diagnose_family,
)


def test_request154_finds_structural_candidate():
    result = _diagnose_family(
        {
            "legacy": 0.928,
            "hazard_only": 0.951,
            "global_direct": 0.934,
            "focus_direct": 0.946,
        }
    )
    assert result["diagnosis"] == "structural_candidate_found"
    assert result["best_variant"] == "hazard_only"


def test_request154_calls_partial_gain_below_target():
    result = _diagnose_family(
        {
            "legacy": 0.928,
            "hazard_only": 0.941,
            "global_direct": 0.933,
            "focus_direct": 0.947,
        }
    )
    assert result["diagnosis"] == "partial_structural_gain"
    assert result["best_variant"] == "focus_direct"


def test_request154_calls_feature_bottleneck_without_one_point_gain():
    result = _diagnose_family(
        {
            "legacy": 0.928,
            "hazard_only": 0.934,
            "global_direct": 0.932,
            "focus_direct": 0.936,
        }
    )
    assert result["diagnosis"] == "feature_information_bottleneck"
    assert result["best_variant"] == "focus_direct"
