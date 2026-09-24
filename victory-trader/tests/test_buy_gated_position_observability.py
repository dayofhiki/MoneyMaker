from __future__ import annotations

from victory_trader.buy_gated_position_observability import (
    CAL_DAYS,
    EVAL_DAYS,
    FIT_DAYS,
    MODEL_FEATURES,
    PHASE_DAYS,
)
from victory_trader.extended_history_economic_opportunity import (
    EXTENDED_CAL_DAYS,
    EXTENDED_FIT_DAYS,
)
from victory_trader.fresh_validated_attention_economic_opportunity import (
    FRESH_EVAL_DAYS,
)
from victory_trader.selected_hot_position_value_observability import (
    MODEL_FEATURES as REQUEST145_MODEL_FEATURES,
)


def test_request166_dates_open_no_new_market_sessions():
    assert FIT_DAYS == list(EXTENDED_FIT_DAYS)
    assert CAL_DAYS == list(EXTENDED_CAL_DAYS)
    assert EVAL_DAYS == list(FRESH_EVAL_DAYS)
    assert set(PHASE_DAYS) == {"fit", "calibration", "evaluation"}


def test_request166_position_model_is_frozen_from_request145():
    assert MODEL_FEATURES == REQUEST145_MODEL_FEATURES


def test_request166_eval_is_june8_12_only():
    assert EVAL_DAYS == [
        "2026-06-08",
        "2026-06-09",
        "2026-06-10",
        "2026-06-11",
        "2026-06-12",
    ]
