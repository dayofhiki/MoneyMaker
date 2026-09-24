from __future__ import annotations

from victory_trader.hierarchical_feature_ablation import (
    FEATURE_SETS,
    TAIL_BASELINE_CAPTURED,
    diagnose,
)


def test_request160_feature_sets_are_isolated():
    assert TAIL_BASELINE_CAPTURED == 4
    assert "market_hazard_probability" not in FEATURE_SETS["baseline"]
    assert "market_rank" not in FEATURE_SETS["baseline"]
    assert "market_hazard_probability" in FEATURE_SETS["hazard_probability_only"]
    assert "market_rank" not in FEATURE_SETS["hazard_probability_only"]
    assert "market_rank" in FEATURE_SETS["hazard_rank_only"]
    assert "market_hazard_probability" not in FEATURE_SETS["hazard_rank_only"]


def _variant(rate: float, mean: float, tail: int) -> dict[str, object]:
    return {
        "metrics": {
            "active_capture": {"supported_capture_rate": rate},
            "active_count": {"mean": mean},
            "rank_bucket_capture": {
                "rank_61_180": {"captured": tail},
            },
        }
    }


def test_diagnose_clean_candidate():
    results = {
        "baseline": _variant(0.946, 21.5, 4),
        "hazard_probability_only": _variant(0.952, 21.0, 5),
        "hazard_rank_only": _variant(0.949, 20.8, 3),
        "hazard_probability_and_rank": _variant(0.953, 21.2, 3),
    }
    out = diagnose(results)
    assert out["diagnosis"] == "hierarchical_ablation_candidate"
    assert out["best_variant"] == "hazard_probability_only"


def test_diagnose_tradeoff():
    results = {
        "baseline": _variant(0.946, 21.5, 4),
        "hazard_probability_only": _variant(0.949, 21.0, 4),
        "hazard_rank_only": _variant(0.951, 20.5, 3),
        "hazard_probability_and_rank": _variant(0.953, 21.0, 3),
    }
    out = diagnose(results)
    assert out["diagnosis"] == "overall_gain_tail_tradeoff"
