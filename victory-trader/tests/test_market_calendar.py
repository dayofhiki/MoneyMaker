from datetime import date

from victory_trader.market_calendar import (
    is_us_equity_trading_day,
    previous_us_equity_trading_day,
    regular_session_bounds,
)


def test_weekend_is_not_trading_day():
    assert not is_us_equity_trading_day(date(2026, 9, 13))


def test_previous_trading_day_skips_weekend():
    assert previous_us_equity_trading_day(date(2026, 9, 14)) == date(2026, 9, 11)


def test_early_close_is_reflected_in_session_bounds():
    # U.S. equities traditionally close early on the Friday after Thanksgiving.
    bounds = regular_session_bounds(date(2025, 11, 28))
    assert bounds is not None
    market_open, market_close = bounds
    assert (market_open.hour, market_open.minute) == (9, 30)
    assert (market_close.hour, market_close.minute) == (13, 0)
