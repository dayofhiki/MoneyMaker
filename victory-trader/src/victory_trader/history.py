from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .features import timestamp_et
from .market_data import bars_from_massive_payload
from .massive_client import MassiveClient


def split_target_and_history(
    bars: pd.DataFrame,
    target_day: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a ranged minute response by New York trading date."""
    if bars.empty:
        return bars.copy(), bars.copy()

    trading_dates = bars["t"].map(lambda value: timestamp_et(int(value)).date())
    target = bars.loc[trading_dates == target_day].reset_index(drop=True)
    history = bars.loc[trading_dates < target_day].reset_index(drop=True)
    return target, history


def load_target_with_history(
    client: MassiveClient,
    ticker: str,
    target_day: date,
    *,
    calendar_lookback_days: int = 35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch target-day bars plus prior minute context for historical RVOL.

    Real MassiveClient instances use one ranged request. The compatibility branch
    keeps deterministic tests and lightweight mock clients working without
    pretending they contain historical context.
    """
    if hasattr(client, "minute_bars_range"):
        start = target_day - timedelta(days=calendar_lookback_days)
        payload = client.minute_bars_range(ticker, start, target_day)
        bars = bars_from_massive_payload(payload)
        return split_target_and_history(bars, target_day)

    payload = client.minute_bars(ticker, target_day)
    target = bars_from_massive_payload(payload)
    return target, pd.DataFrame(columns=target.columns)
