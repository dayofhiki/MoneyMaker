from __future__ import annotations

from victory_trader.fresh_validated_attention_economic_opportunity import (
    CANDIDATE_FEATURES,
    CANDIDATE_THRESHOLD,
    FOCUS_CAP,
    FRESH_EVAL_DAYS,
    HORIZON,
)


def test_request162_frozen_attention_contract():
    assert FOCUS_CAP == 180
    assert HORIZON == 3
    assert CANDIDATE_THRESHOLD == 0.003093198572895277
    assert FRESH_EVAL_DAYS == [
        "2026-06-08",
        "2026-06-09",
        "2026-06-10",
        "2026-06-11",
        "2026-06-12",
    ]
    assert "market_hazard_probability" in CANDIDATE_FEATURES
    assert "market_rank" not in CANDIDATE_FEATURES
