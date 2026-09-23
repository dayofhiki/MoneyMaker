from __future__ import annotations

import pandas as pd

from victory_trader.intraminute_observability_diagnostic import (
    crossing_observability_rows,
    summarize,
)


def test_crossing_observability_separates_exact_prior_and_blind_cases():
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-04-01",
                "ticker": "EXACT",
                "t": 60_000,
                "previous_close": 1.0,
                "return_from_previous_close_pct": 2.0,
                "runner_cross_now": False,
            },
            {
                "trading_day": "2026-04-01",
                "ticker": "EXACT",
                "t": 120_000,
                "previous_close": 1.0,
                "return_from_previous_close_pct": 11.0,
                "runner_cross_now": True,
            },
            {
                "trading_day": "2026-04-01",
                "ticker": "GAP",
                "t": 60_000,
                "previous_close": 1.0,
                "return_from_previous_close_pct": 1.0,
                "runner_cross_now": False,
            },
            {
                "trading_day": "2026-04-01",
                "ticker": "GAP",
                "t": 180_000,
                "previous_close": 1.0,
                "return_from_previous_close_pct": 12.0,
                "runner_cross_now": True,
            },
            {
                "trading_day": "2026-04-01",
                "ticker": "FIRST",
                "t": 240_000,
                "previous_close": 1.0,
                "return_from_previous_close_pct": 15.0,
                "runner_cross_now": True,
            },
        ]
    )

    crossings = crossing_observability_rows(scan).set_index("ticker")

    assert bool(crossings.loc["EXACT", "has_exact_prior_minute"]) is True
    assert bool(crossings.loc["GAP", "has_exact_prior_minute"]) is False
    assert bool(crossings.loc["FIRST", "no_prior_bar"]) is True


def test_summary_reports_intraminute_recovery_window():
    crossings = pd.DataFrame(
        [
            {
                "trading_day": "2026-04-01",
                "has_exact_prior_minute": True,
            },
            {
                "trading_day": "2026-04-01",
                "has_exact_prior_minute": False,
            },
            {
                "trading_day": "2026-04-01",
                "has_exact_prior_minute": False,
            },
        ]
    )
    second_rows = pd.DataFrame(
        [
            {
                "trading_day": "2026-04-01",
                "gap_bucket": "gap_2m",
                "second_data_present": True,
                "second_threshold_found": True,
                "first_active_second_already_crossed": False,
                "active_seconds_before_cross": 12,
                "wall_seconds_from_first_activity_to_cross": 12.0,
                "pre_cross_peak_return_pct": 9.0,
                "first_active_return_pct": 3.0,
            },
            {
                "trading_day": "2026-04-01",
                "gap_bucket": "no_prior_bar",
                "second_data_present": True,
                "second_threshold_found": True,
                "first_active_second_already_crossed": True,
                "active_seconds_before_cross": 0,
                "wall_seconds_from_first_activity_to_cross": 0.0,
                "pre_cross_peak_return_pct": None,
                "first_active_return_pct": 11.0,
            },
        ]
    )

    result = summarize(crossings, second_rows)

    assert result["exact_prior_minute_count"] == 1
    assert result["blind_crossings"] == 2
    assert result["second_threshold_found_count"] == 2
    assert result["has_pre_cross_second_count"] == 1
    assert result["pre_cross_seconds_ge_10_rate"] == 0.5
