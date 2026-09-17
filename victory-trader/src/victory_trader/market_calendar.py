from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal


NEW_YORK = ZoneInfo("America/New_York")
NYSE = mcal.get_calendar("NYSE")


@lru_cache(maxsize=4096)
def regular_session_bounds(day: date) -> tuple[datetime, datetime] | None:
    """Return official regular-session open/close, including holidays and early closes."""
    schedule = NYSE.schedule(start_date=day.isoformat(), end_date=day.isoformat())
    if schedule.empty:
        return None
    row = schedule.iloc[0]
    market_open = row["market_open"].to_pydatetime().astimezone(NEW_YORK)
    market_close = row["market_close"].to_pydatetime().astimezone(NEW_YORK)
    return market_open, market_close


@lru_cache(maxsize=4096)
def is_us_equity_trading_day(day: date) -> bool:
    return regular_session_bounds(day) is not None


@lru_cache(maxsize=4096)
def previous_us_equity_trading_day(day: date) -> date:
    valid = NYSE.valid_days(
        start_date=(day - timedelta(days=14)).isoformat(),
        end_date=(day - timedelta(days=1)).isoformat(),
    )
    if len(valid) == 0:
        raise ValueError(f"No prior US equity trading day found before {day}")
    return valid[-1].date()
