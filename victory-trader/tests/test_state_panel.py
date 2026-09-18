from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from victory_trader.market_calendar import regular_session_bounds
from victory_trader.state_panel import (
    MINUTE_MS,
    _build_ticker_states,
    _iwm_context,
)


ET = ZoneInfo("America/New_York")


def _bars(day: date, closes: list[float], ticker: str = "AAA") -> pd.DataFrame:
    start = datetime.combine(day, time(9, 30), tzinfo=ET)
    rows = []
    for i, close in enumerate(closes):
        ts = int((start.timestamp() + i * 60) * 1000)
        open_price = closes[i - 1] if i else close * 0.995
        rows.append(
            {
                "ticker": ticker,
                "t": ts,
                "o": open_price,
                "h": max(open_price, close) * 1.002,
                "l": min(open_price, close) * 0.998,
                "c": close,
                "v": 1000 + i * 10,
                "n": 100 + i,
            }
        )
    return pd.DataFrame(rows)


def test_state_panel_uses_exact_next_minute_entry_and_future_close():
    day = date(2026, 1, 2)
    bounds = regular_session_bounds(day)
    assert bounds is not None
    regular_open_ms = int(bounds[0].timestamp() * 1000)
    regular_close_ms = int(bounds[1].timestamp() * 1000)

    closes = [10.0, 10.5, 11.0, 11.2, 11.1, 11.3, 11.4, 11.5]
    bars = _bars(day, closes)
    first_cross = int(bars.iloc[2]["t"])
    iwm = _iwm_context(pd.DataFrame(), np.arange(regular_open_ms, regular_close_ms, MINUTE_MS))

    states = _build_ticker_states(
        bars,
        ticker="AAA",
        trading_day=day.isoformat(),
        first_10_timestamp_ms=first_cross,
        previous_close=10.0,
        regular_open_ms=regular_open_ms,
        regular_close_ms=regular_close_ms,
        day_crossings=[first_cross],
        iwm_context=iwm,
        cross_row=pd.Series({"rvol_cumulative_20d": 2.0}),
    )

    first = states.iloc[0]
    assert int(first["t"]) == first_cross
    assert first["entry_price"] == bars.iloc[3]["o"]
    expected_5m = (bars.iloc[7]["c"] / bars.iloc[3]["o"] - 1.0) * 100.0
    assert abs(first["buy_return_5m_pct"] - expected_5m) < 1e-9


def test_state_panel_keeps_missing_exact_entry_missing():
    day = date(2026, 1, 2)
    bounds = regular_session_bounds(day)
    assert bounds is not None
    regular_open_ms = int(bounds[0].timestamp() * 1000)
    regular_close_ms = int(bounds[1].timestamp() * 1000)

    bars = _bars(day, [10.0, 11.0, 11.2, 11.4, 11.5])
    first_cross = int(bars.iloc[1]["t"])
    # Remove the exact next-minute bar after the state.
    bars = bars.loc[bars["t"] != first_cross + MINUTE_MS].reset_index(drop=True)
    iwm = _iwm_context(pd.DataFrame(), np.arange(regular_open_ms, regular_close_ms, MINUTE_MS))

    states = _build_ticker_states(
        bars,
        ticker="AAA",
        trading_day=day.isoformat(),
        first_10_timestamp_ms=first_cross,
        previous_close=10.0,
        regular_open_ms=regular_open_ms,
        regular_close_ms=regular_close_ms,
        day_crossings=[first_cross],
        iwm_context=iwm,
        cross_row=pd.Series(dtype=object),
    )

    first = states.iloc[0]
    assert pd.isna(first["entry_price"])
    assert pd.isna(first["buy_return_5m_pct"])


def test_state_features_are_point_in_time_and_phase_is_present():
    day = date(2026, 1, 2)
    bounds = regular_session_bounds(day)
    assert bounds is not None
    regular_open_ms = int(bounds[0].timestamp() * 1000)
    regular_close_ms = int(bounds[1].timestamp() * 1000)

    closes = [10.0, 11.0, 11.4, 11.0, 10.8, 11.1, 11.45, 11.5, 11.6]
    bars = _bars(day, closes)
    first_cross = int(bars.iloc[1]["t"])
    iwm = _iwm_context(pd.DataFrame(), np.arange(regular_open_ms, regular_close_ms, MINUTE_MS))

    states = _build_ticker_states(
        bars,
        ticker="AAA",
        trading_day=day.isoformat(),
        first_10_timestamp_ms=first_cross,
        previous_close=10.0,
        regular_open_ms=regular_open_ms,
        regular_close_ms=regular_close_ms,
        day_crossings=[first_cross],
        iwm_context=iwm,
        cross_row=pd.Series({"premarket_volume": 12345}),
    )

    assert {"phase", "hod_distance_pct", "regular_vwap_distance_pct"}.issubset(states.columns)
    assert states.iloc[0]["cross_premarket_volume"] == 12345
    assert states["minutes_since_10pct_cross"].min() == 0
    assert states["runners_10pct_so_far"].min() == 1
