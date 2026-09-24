from __future__ import annotations

from datetime import date

from victory_trader.extended_history_economic_opportunity import (
    EXTENDED_CAL_DAYS,
    EXTENDED_FIT_DAYS,
    FRESH_EVAL_DAYS,
    MIN_DEVELOPMENT_LABELS,
    OPPORTUNITY_DAYS,
    _role_days,
)


def test_request163_history_contract():
    assert MIN_DEVELOPMENT_LABELS == 1500
    assert EXTENDED_FIT_DAYS == [
        "2026-04-30",
        "2026-05-01",
        "2026-05-04",
        "2026-05-05",
        "2026-05-06",
        "2026-05-07",
        "2026-05-08",
    ]
    assert EXTENDED_CAL_DAYS == [
        "2026-05-11",
        "2026-05-12",
        "2026-05-13",
        "2026-05-14",
        "2026-05-15",
        "2026-05-18",
        "2026-05-19",
        "2026-05-20",
    ]
    assert FRESH_EVAL_DAYS == [
        "2026-06-08",
        "2026-06-09",
        "2026-06-10",
        "2026-06-11",
        "2026-06-12",
    ]
    assert OPPORTUNITY_DAYS == (
        EXTENDED_FIT_DAYS + EXTENDED_CAL_DAYS + FRESH_EVAL_DAYS
    )


def test_request163_does_not_use_june1_5():
    days = _role_days(
        "opportunity",
        date.fromisoformat("2026-01-02"),
        date.fromisoformat("2026-06-12"),
    )
    assert all(
        day not in days
        for day in [
            "2026-06-01",
            "2026-06-02",
            "2026-06-03",
            "2026-06-04",
            "2026-06-05",
        ]
    )
