from __future__ import annotations

from victory_trader.rate_limited_subscription_confirmation import (
    EVAL_DAYS,
    confirmation_gate,
)


def _evaluation(post_initial_mean: float, post_initial_max: int) -> dict[str, object]:
    return {
        "by_day": {
            day: {
                "focus": {"runner_crossings": 20},
            }
            for day in EVAL_DAYS
        },
        "rank40_hysteresis": {"prior_minute_rate": 0.50},
        "rate_limited_active": {"prior_minute_rate": 0.52},
        "active_retention_given_focus": 0.96,
        "nonlower_capture_days": 4,
        "rate_limited_audit": {
            "max_occupancy": 20,
            "mean_post_initial_additions_per_decision": post_initial_mean,
            "max_post_initial_additions_per_decision": post_initial_max,
        },
    }


def test_confirmation_gate_uses_post_initial_churn_definition():
    result = _evaluation(5.0, 5)
    assert confirmation_gate(result) is True


def test_confirmation_gate_rejects_post_initial_burst():
    result = _evaluation(4.9, 6)
    assert confirmation_gate(result) is False
