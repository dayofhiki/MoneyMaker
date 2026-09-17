from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from .events import CrossingEvent
from .market_calendar import regular_session_bounds


NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MINUTE_MS = 60_000
PREMARKET_START = time(4, 0)
REGULAR_START = time(9, 30)
REGULAR_END = time(16, 0)
AFTER_HOURS_END = time(20, 0)


@dataclass(frozen=True)
class EventFeatures:
    event_volume: float
    cumulative_volume: float
    volume_5m: float
    volume_15m: float
    volume_30m: float
    volume_accel_1m_vs_prior20m: float | None
    volume_accel_5m_vs_prior20m: float | None
    rvol_cumulative_20d: float | None
    rvol_5m_20d: float | None
    rvol_history_days: int
    session_vwap: float | None
    vwap_distance_pct: float | None
    regular_vwap: float | None
    regular_vwap_distance_pct: float | None
    hod_distance_pct: float | None
    trailing_return_5m_pct: float | None
    trailing_return_15m_pct: float | None
    trailing_return_30m_pct: float | None
    volatility_5m_pct: float | None
    volatility_15m_pct: float | None
    volatility_30m_pct: float | None
    premarket_return_pct: float | None
    premarket_volume: float
    minutes_from_regular_open: float | None
    is_premarket: bool
    is_regular_session: bool
    is_after_hours: bool

    def to_record(self) -> dict[str, float | bool | int | None]:
        return asdict(self)


def timestamp_et(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).astimezone(NEW_YORK)


def _minute_of_day(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def _clock_window(frame: pd.DataFrame, end_timestamp_ms: int, minutes: int) -> pd.DataFrame:
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    start = end_timestamp_ms - (minutes - 1) * MINUTE_MS
    return frame.loc[(frame["t"] >= start) & (frame["t"] <= end_timestamp_ms)]


def _return_over_clock_window(frame: pd.DataFrame, end_timestamp_ms: int, minutes: int) -> float | None:
    target = end_timestamp_ms - minutes * MINUTE_MS
    matches = frame.loc[frame["t"] == target, "c"]
    if len(matches) != 1:
        return None
    start_price = float(matches.iloc[0])
    if start_price <= 0:
        return None
    end_matches = frame.loc[frame["t"] == end_timestamp_ms, "c"]
    if len(end_matches) != 1:
        return None
    end_price = float(end_matches.iloc[0])
    return (end_price / start_price - 1.0) * 100.0


def _volatility(frame: pd.DataFrame, end_timestamp_ms: int, minutes: int) -> float | None:
    start = end_timestamp_ms - minutes * MINUTE_MS
    window = frame.loc[(frame["t"] >= start) & (frame["t"] <= end_timestamp_ms)].copy()
    if len(window) < 3:
        return None
    window = window.sort_values("t")
    prices = pd.to_numeric(window["c"], errors="coerce")
    returns = prices.pct_change()
    consecutive = pd.to_numeric(window["t"], errors="coerce").diff().eq(MINUTE_MS)
    returns = returns.loc[consecutive].dropna()
    if len(returns) < 2:
        return None
    return float(returns.std(ddof=1) * 100.0)


def _volume_acceleration(frame: pd.DataFrame, end_timestamp_ms: int, recent_minutes: int, baseline_minutes: int = 20) -> float | None:
    recent = _clock_window(frame, end_timestamp_ms, recent_minutes)
    recent_start = end_timestamp_ms - (recent_minutes - 1) * MINUTE_MS
    baseline_end = recent_start - MINUTE_MS
    baseline_start = baseline_end - (baseline_minutes - 1) * MINUTE_MS
    baseline = frame.loc[(frame["t"] >= baseline_start) & (frame["t"] <= baseline_end)]
    if recent.empty or baseline.empty:
        return None
    recent_rate = float(pd.to_numeric(recent["v"], errors="coerce").fillna(0.0).sum()) / recent_minutes
    baseline_rate = float(pd.to_numeric(baseline["v"], errors="coerce").fillna(0.0).sum()) / baseline_minutes
    if baseline_rate <= 0:
        return None
    return recent_rate / baseline_rate


def _volume_weighted_price(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    volume = pd.to_numeric(frame["v"], errors="coerce").fillna(0.0)
    if float(volume.sum()) <= 0:
        return None
    if "vw" in frame.columns:
        bar_vwap = pd.to_numeric(frame["vw"], errors="coerce")
        prices = bar_vwap.where(bar_vwap.notna(), pd.to_numeric(frame["c"], errors="coerce"))
    else:
        prices = pd.to_numeric(frame["c"], errors="coerce")
    valid = prices.notna() & volume.gt(0)
    if not valid.any():
        return None
    return float((prices[valid] * volume[valid]).sum() / volume[valid].sum())


def _historical_rvol(history_bars: pd.DataFrame | None, event_dt: datetime, current_cumulative_volume: float, current_5m_volume: float, max_days: int = 20) -> tuple[float | None, float | None, int]:
    if history_bars is None or history_bars.empty:
        return None, None, 0
    if "t" not in history_bars.columns or "v" not in history_bars.columns:
        return None, None, 0
    history = history_bars.copy()
    history["_dt_et"] = history["t"].map(lambda x: timestamp_et(int(x)))
    history = history.loc[history["_dt_et"].map(lambda x: x.date() < event_dt.date())]
    if history.empty:
        return None, None, 0
    dates = sorted(history["_dt_et"].map(lambda x: x.date()).unique())[-max_days:]
    event_minute = _minute_of_day(event_dt)
    session_start_minute = PREMARKET_START.hour * 60 + PREMARKET_START.minute
    window_start = event_minute - 4
    cumulative_samples: list[float] = []
    window_samples: list[float] = []
    for trading_date in dates:
        day_frame = history.loc[history["_dt_et"].map(lambda x: x.date() == trading_date)].copy()
        minute_index = day_frame["_dt_et"].map(_minute_of_day)
        observed = day_frame.loc[(minute_index >= session_start_minute) & (minute_index <= event_minute)]
        if observed.empty:
            continue
        cumulative_samples.append(float(pd.to_numeric(observed["v"], errors="coerce").fillna(0.0).sum()))
        observed_minutes = observed["_dt_et"].map(_minute_of_day)
        recent = observed.loc[(observed_minutes >= window_start) & (observed_minutes <= event_minute)]
        window_samples.append(float(pd.to_numeric(recent["v"], errors="coerce").fillna(0.0).sum()))
    if not cumulative_samples:
        return None, None, 0
    average_cumulative = sum(cumulative_samples) / len(cumulative_samples)
    average_5m = sum(window_samples) / len(window_samples) if window_samples else 0.0
    cumulative_rvol = current_cumulative_volume / average_cumulative if average_cumulative > 0 else None
    five_min_rvol = current_5m_volume / average_5m if average_5m > 0 else None
    return cumulative_rvol, five_min_rvol, len(cumulative_samples)


def extract_event_features(event: CrossingEvent, bars: pd.DataFrame, history_bars: pd.DataFrame | None = None) -> EventFeatures:
    required = {"t", "o", "h", "l", "c", "v"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns for features: {sorted(missing)}")

    frame = bars.sort_values("t").reset_index(drop=True).copy()
    matches = frame.index[frame["t"] == event.timestamp_ms].tolist()
    if len(matches) != 1:
        raise ValueError("event timestamp must match exactly one bar")
    observed = frame.iloc[: matches[0] + 1].copy()

    event_dt = timestamp_et(event.timestamp_ms)
    same_date_mask = observed["t"].map(lambda x: timestamp_et(int(x)).date() == event_dt.date())
    observed = observed.loc[same_date_mask].reset_index(drop=True)
    event_idx = len(observed) - 1

    bounds = regular_session_bounds(event_dt.date())
    if bounds is None:
        regular_open_dt = event_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        regular_close_dt = event_dt.replace(hour=16, minute=0, second=0, microsecond=0)
    else:
        regular_open_dt, regular_close_dt = bounds
    premarket_start_dt = event_dt.replace(hour=4, minute=0, second=0, microsecond=0)
    after_hours_end_dt = event_dt.replace(hour=20, minute=0, second=0, microsecond=0)
    is_premarket = premarket_start_dt <= event_dt < regular_open_dt
    is_regular = regular_open_dt <= event_dt < regular_close_dt
    is_after_hours = regular_close_dt <= event_dt < after_hours_end_dt

    event_volume = float(observed.iloc[event_idx]["v"])
    cumulative_volume = float(pd.to_numeric(observed["v"], errors="coerce").fillna(0.0).sum())
    volume_5m = float(pd.to_numeric(_clock_window(observed, event.timestamp_ms, 5)["v"], errors="coerce").fillna(0.0).sum())
    volume_15m = float(pd.to_numeric(_clock_window(observed, event.timestamp_ms, 15)["v"], errors="coerce").fillna(0.0).sum())
    volume_30m = float(pd.to_numeric(_clock_window(observed, event.timestamp_ms, 30)["v"], errors="coerce").fillna(0.0).sum())

    rvol_cumulative, rvol_5m, rvol_days = _historical_rvol(history_bars, event_dt, cumulative_volume, volume_5m)

    minute_index = observed["t"].map(lambda x: _minute_of_day(timestamp_et(int(x))))
    extended_start = 4 * 60
    regular_start = regular_open_dt.hour * 60 + regular_open_dt.minute
    extended_observed = observed.loc[minute_index >= extended_start]
    regular_observed = observed.loc[minute_index >= regular_start] if event_dt >= regular_open_dt else observed.iloc[0:0]
    session_vwap = _volume_weighted_price(extended_observed)
    vwap_distance = None if session_vwap is None or session_vwap <= 0 else (event.price / session_vwap - 1.0) * 100.0
    regular_vwap = _volume_weighted_price(regular_observed)
    regular_vwap_distance = None if regular_vwap is None or regular_vwap <= 0 else (event.price / regular_vwap - 1.0) * 100.0

    high_to_event = float(pd.to_numeric(observed["h"], errors="coerce").max())
    hod_distance = None if high_to_event <= 0 else (event.price / high_to_event - 1.0) * 100.0

    premarket_mask = minute_index.between(extended_start, regular_start - 1)
    premarket = observed.loc[premarket_mask]
    if not premarket.empty:
        pm_first = float(premarket.iloc[0]["o"])
        pm_last = float(premarket.iloc[-1]["c"])
        premarket_return = None if pm_first <= 0 else (pm_last / pm_first - 1.0) * 100.0
        premarket_volume = float(pd.to_numeric(premarket["v"], errors="coerce").fillna(0.0).sum())
    else:
        premarket_return = None
        premarket_volume = 0.0

    minutes_from_open = (event_dt - regular_open_dt).total_seconds() / 60.0 if event_dt >= regular_open_dt else None

    return EventFeatures(
        event_volume=event_volume,
        cumulative_volume=cumulative_volume,
        volume_5m=volume_5m,
        volume_15m=volume_15m,
        volume_30m=volume_30m,
        volume_accel_1m_vs_prior20m=_volume_acceleration(observed, event.timestamp_ms, 1, 20),
        volume_accel_5m_vs_prior20m=_volume_acceleration(observed, event.timestamp_ms, 5, 20),
        rvol_cumulative_20d=rvol_cumulative,
        rvol_5m_20d=rvol_5m,
        rvol_history_days=rvol_days,
        session_vwap=session_vwap,
        vwap_distance_pct=vwap_distance,
        regular_vwap=regular_vwap,
        regular_vwap_distance_pct=regular_vwap_distance,
        hod_distance_pct=hod_distance,
        trailing_return_5m_pct=_return_over_clock_window(observed, event.timestamp_ms, 5),
        trailing_return_15m_pct=_return_over_clock_window(observed, event.timestamp_ms, 15),
        trailing_return_30m_pct=_return_over_clock_window(observed, event.timestamp_ms, 30),
        volatility_5m_pct=_volatility(observed, event.timestamp_ms, 5),
        volatility_15m_pct=_volatility(observed, event.timestamp_ms, 15),
        volatility_30m_pct=_volatility(observed, event.timestamp_ms, 30),
        premarket_return_pct=premarket_return,
        premarket_volume=premarket_volume,
        minutes_from_regular_open=minutes_from_open,
        is_premarket=is_premarket,
        is_regular_session=is_regular,
        is_after_hours=is_after_hours,
    )
