from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from victory_trader.premarket_first_passage_directional_edge import (
    add_premarket_features,
)


NY = ZoneInfo("America/New_York")


class FakeStore:
    def __init__(self, seconds):
        self._seconds = seconds

    def seconds(self, day, ticker):
        return self._seconds


def _ms(day_text, hour, minute, second=0):
    day = date.fromisoformat(day_text)
    dt = datetime(
        day.year,
        day.month,
        day.day,
        hour,
        minute,
        second,
        tzinfo=NY,
    )
    return int(dt.timestamp() * 1000)


def test_premarket_features_ignore_regular_session_bars():
    day = "2026-05-11"
    seconds = pd.DataFrame(
        [
            {
                "t": _ms(day, 8, 0),
                "o": 10.0,
                "h": 10.2,
                "l": 9.9,
                "c": 10.1,
                "v": 100.0,
                "n": 10.0,
            },
            {
                "t": _ms(day, 9, 15),
                "o": 10.1,
                "h": 10.6,
                "l": 10.0,
                "c": 10.5,
                "v": 300.0,
                "n": 30.0,
            },
            {
                "t": _ms(day, 9, 30),
                "o": 99.0,
                "h": 100.0,
                "l": 98.0,
                "c": 99.5,
                "v": 9999.0,
                "n": 999.0,
            },
        ]
    )
    frame = pd.DataFrame(
        [
            {
                "trading_day": day,
                "ticker": "TEST",
                "log_current_close": 2.3513752571634776,
                "return_from_previous_close_pct": 5.0,
            }
        ]
    )
    result = add_premarket_features(frame, FakeStore(seconds))
    row = result.iloc[0]
    assert row["pm_any_activity"] == 1.0
    assert row["pm_log_active_seconds"] > 0
    assert row["pm_log_volume"] < 7.0
    assert 4.9 < row["pm_return_pct"] < 5.1
    assert row["state_vs_pm_high_pct"] < 0.0


def test_no_premarket_activity_is_explicit_not_dropped():
    day = "2026-05-11"
    seconds = pd.DataFrame(
        [
            {
                "t": _ms(day, 9, 30),
                "o": 10.0,
                "h": 10.1,
                "l": 9.9,
                "c": 10.0,
                "v": 100.0,
                "n": 10.0,
            }
        ]
    )
    frame = pd.DataFrame(
        [
            {
                "trading_day": day,
                "ticker": "TEST",
                "log_current_close": 2.302585092994046,
                "return_from_previous_close_pct": 0.0,
            }
        ]
    )
    result = add_premarket_features(frame, FakeStore(seconds))
    row = result.iloc[0]
    assert row["pm_any_activity"] == 0.0
    assert row["pm_log_active_seconds"] == 0.0
    assert pd.isna(row["pm_return_pct"])
    assert pd.isna(row["state_vs_pm_high_pct"])
