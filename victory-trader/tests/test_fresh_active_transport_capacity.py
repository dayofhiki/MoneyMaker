from __future__ import annotations

from victory_trader.fresh_active_transport_capacity import (
    COMPARATOR_MAX_ADDITIONS,
    EVAL_DAYS,
    PRIMARY_MAX_ADDITIONS,
    _nonlower_hot_days,
)


def test_request151_fresh_block_and_resource_caps_are_frozen():
    assert EVAL_DAYS == [
        "2026-05-21",
        "2026-05-22",
        "2026-05-26",
        "2026-05-27",
        "2026-05-28",
    ]
    assert PRIMARY_MAX_ADDITIONS == 8
    assert COMPARATOR_MAX_ADDITIONS == 5


def test_nonlower_hot_days_counts_only_nonlower_primary_days():
    primary = {
        "by_day": {
            day: {"hot": {"hot_within_1m_rate": value}}
            for day, value in zip(EVAL_DAYS, [0.5, 0.4, 0.7, 0.2, 0.6], strict=True)
        }
    }
    comparator = {
        "by_day": {
            day: {"hot": {"hot_within_1m_rate": value}}
            for day, value in zip(EVAL_DAYS, [0.4, 0.4, 0.6, 0.3, 0.5], strict=True)
        }
    }

    assert _nonlower_hot_days(primary, comparator) == 4
