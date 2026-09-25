from datetime import date

import pandas as pd

from victory_trader.short_interest_position_probe import (
    PUBLICATION_SCHEDULE,
    _prepare_ticker_history,
)
from victory_trader.state_short_interest_enrichment import (
    _feature_row,
)


def test_2026_april_may_publication_schedule_is_present():
    assert PUBLICATION_SCHEDULE[date(2026, 4, 15)] == date(2026, 4, 24)
    assert PUBLICATION_SCHEDULE[date(2026, 4, 30)] == date(2026, 5, 11)


def test_same_day_publication_is_not_available_intraday():
    records = _prepare_ticker_history(
        [
            {
                "settlement_date": "2026-04-15",
                "short_interest": 1000,
                "avg_daily_volume": 500,
                "days_to_cover": 2.0,
            },
            {
                "settlement_date": "2026-04-30",
                "short_interest": 2000,
                "avg_daily_volume": 800,
                "days_to_cover": 2.5,
            },
        ]
    )
    may11 = _feature_row(records, date(2026, 5, 11))
    may12 = _feature_row(records, date(2026, 5, 12))
    assert may11["short_interest_latest"] == 1000
    assert may12["short_interest_latest"] == 2000
    assert may12["short_interest_reports_available"] == 2.0


def test_publication_safe_feature_change_is_causal():
    records = _prepare_ticker_history(
        [
            {
                "settlement_date": "2026-04-15",
                "short_interest": 1000,
                "avg_daily_volume": 500,
                "days_to_cover": 2.0,
            },
            {
                "settlement_date": "2026-04-30",
                "short_interest": 1500,
                "avg_daily_volume": 600,
                "days_to_cover": 2.5,
            },
        ]
    )
    row = _feature_row(records, date(2026, 5, 12))
    assert row["short_interest_change_pct"] == 50.0
    assert row["short_interest_avg_daily_volume_change_pct"] == 20.0
    assert row["short_interest_days_to_cover_change"] == 0.5
    assert pd.notna(row["short_interest_publication_age_days"])
