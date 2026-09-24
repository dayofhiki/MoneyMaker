from __future__ import annotations

from victory_trader.fresh_hierarchical_active_validation import (
    BASELINE_THRESHOLD,
    CANDIDATE_FEATURES,
    CANDIDATE_THRESHOLD,
    FOCUS_CAP,
    FRESH_EVAL_DAYS,
    TARGET_ACTIVE_CAPTURE,
    TARGET_FOCUS_CAPTURE,
    _diagnose,
)


def test_request161_contract_is_frozen():
    assert FOCUS_CAP == 180
    assert FRESH_EVAL_DAYS == [
        "2026-06-01",
        "2026-06-02",
        "2026-06-03",
        "2026-06-04",
        "2026-06-05",
    ]
    assert BASELINE_THRESHOLD == 0.003429286145316252
    assert CANDIDATE_THRESHOLD == 0.003093198572895277
    assert TARGET_FOCUS_CAPTURE == 0.99
    assert TARGET_ACTIVE_CAPTURE == 0.95
    assert "market_hazard_probability" in CANDIDATE_FEATURES
    assert "market_rank" not in CANDIDATE_FEATURES


def _metrics(rate: float) -> dict[str, object]:
    return {
        "active_capture": {"supported_capture_rate": rate},
    }


def test_diagnose_validated():
    focus = {
        "supported_capture_rate": 0.995,
        "supported_exact_prior_crossings": 200,
    }
    out = _diagnose(
        focus,
        _metrics(0.951),
        _metrics(0.956),
    )
    assert out["diagnosis"] == "fresh_candidate_validated"


def test_diagnose_feature_uplift_not_confirmed():
    focus = {
        "supported_capture_rate": 0.995,
        "supported_exact_prior_crossings": 200,
    }
    out = _diagnose(
        focus,
        _metrics(0.960),
        _metrics(0.955),
    )
    assert (
        out["diagnosis"]
        == "fresh_absolute_pass_feature_uplift_not_confirmed"
    )


def test_diagnose_active_failure():
    focus = {
        "supported_capture_rate": 0.995,
        "supported_exact_prior_crossings": 200,
    }
    out = _diagnose(
        focus,
        _metrics(0.946),
        _metrics(0.944),
    )
    assert out["diagnosis"] == "fresh_active_candidate_failed"
