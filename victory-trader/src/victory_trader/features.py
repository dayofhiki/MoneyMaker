from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .events import CrossingEvent


NEW_YORK = ZoneInfo("America/New_York")
PREMARKET_START = time(4, 0)
REGULAR_START = time(9, 30)
REGULAR_END = time(16, 0)


@dataclass(frozen=True)
class EventFeatures:
    event_volume: float
    cumulative_volume: float
    volume_5m: float
    volume_15m: float
    volume_30m: float
    volume_accel_1m_vs_prior20m: float | None
    volume_accel_5m_vs_prior20m: float | None
    session_vwap: float | None
    vwap_distance_pct: float | None
    hod_distance_pct: float | None
    return_5m_pct: float | None
    return_15m_pct: float | None
    return_30m_pct: float | None
    volatility_5m_pct: float | None
    volatility_15m_pct: float | None
    volatility_30m_pct: float | None
    premarket_return_pct: float | None
    premarket_volume: float
    minutes_from_regular_open: float | None
    is_premarket: bool
    is_regular_session: bool
    is_after_hours: bool

    def to_record(self) -> dict[str, float | bool | None]:
        return asdict(self)


def _timestamp_et(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=ZoneInfo("UTC")).astimezone(NEW_YORK)


def _window(frame: pd.DataFrame, end_idx: int, minutes: int) -> pd.DataFrame:
    start = max(0, end_idx - minutes + 1)
    return frame.iloc[start : end_idx + 1]


def _return_over_window(frame: pd.DataFrame, end_idx: int, minutes: int) -> float | None:
    start_idx = end_idx - minutes
    if start_idx < 0:
        return None
    start_price = float(frame.iloc[start_idx]["c"])
    if start_price <= 0:
        return None
    end_price = float(frame.iloc[end_idx]["c"])
    return (end_price / start_price - 1.0) * 100.0


def _volatility(frame: pd.DataFrame, end_idx: int, minutes: int) -> float | None:
    window = _window(frame, end_idx, minutes + 1)
    if len(window) < 3:
        return None
    returns = pd.to_numeric(window["c"], errors="coerce").pct_change().dropna()
    if len(returns) < 2:
        return None
    return float(returns.std(ddof=1) * 100.0)


def _volume_acceleration(frame: pd.DataFrame, end_idx: int, recent_minutes: int, baseline_minutes: int = 20) -> float | None:
    recent_start = max(0, end_idx - recent_minutes + 1)
    recent = frame.iloc[recent_start : end_idx + 1]
    baseline_end = recent_start
    baseline_start = max(0, baseline_end - baseline_minutes)
    baseline = frame.iloc[baseline_start:baseline_end]
    if recent.empty or baseline.empty:
        return None

    recent_rate = float(pd.to_numeric(recent["v"], errors="coerce").sum()) / len(recent)
    baseline_rate = float(pd.to_numeric(baseline["v"], errors="coerce").sum()) / len(baseline)
    if baseline_rate <= 0:
        return None
    return recent_rate / baseline_rate


def _session_vwap(frame: pd.DataFrame) -> float | None:
    volume = pd.to_numeric(frame["v"], errors="coerce").fillna(0.0)
    total_volume = float(volume.sum())
    if total_volume <= 0:
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


def extract_event_features(event: CrossingEvent, bars: pd.DataFrame) -> EventFeatures:
    required = {"t", "o", "h", "l", "c", "v"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns for features: {sorted(missing)}")

    frame = bars.sort_values("t").reset_index(drop=True).copy()
    matches = frame.index[frame["t"] == event.timestamp_ms].tolist()
    if len(matches) != 1:
        raise ValueError("event timestamp must match exactly one bar")
    event_idx = matches[0]
    observed = frame.iloc[: event_idx + 1].copy()

    event_dt = _timestamp_et(event.timestamp_ms)
    local_time = event_dt.time().replace(tzinfo=None)
    is_premarket = PREMARKET_START <= local_time < REGULAR_START
    is_regular = REGULAR_START <= local_time < REGULAR_END
    is_after_hours = local_time >= REGULAR_END

    same_date_mask = observed["t"].map(lambda x: _timestamp_et(int(x)).date() == event_dt.date())
    observed = observed.loc[same_date_mask].reset_index(drop=True)
    event_idx = len(observed) - 1

    event_volume = float(observed.iloc[event_idx]["v"])
    cumulative_volume = float(pd.to_numeric(observed["v"], errors="coerce").fillna(0.0).sum())

    session_vwap = _session_vwap(observed)
    vwap_distance = None if session_vwap is None or session_vwap <= 0 else (event.price / session_vwap - 1.0) * 100.0

    high_to_event = float(pd.to_numeric(observed["h"], errors="coerce").max())
    hod_distance = None if high_to_event <= 0 else (event.price / high_to_event - 1.0) * 100.0

    premarket_rows = []
    for idx, row in observed.iterrows():
        dt = _timestamp_et(int(row["t"]))
        lt = dt.time().replace(tzinfo=None)
        if PREMARKET_START <= lt < REGULAR_START:
            premarket_rows.append(idx)
    if premarket_rows:
        premarket = observed.iloc[premarket_rows]
        pm_first = float(premarket.iloc[0]["o"])
        pm_last = float(premarket.iloc[-1]["c"])
        premarket_return = None if pm_first <= 0 else (pm_last / pm_first - 1.0) * 100.0
        premarket_volume = float(pd.to_numeric(premarket["v"], errors="coerce").fillna(0.0).sum())
    else:
        premarket_return = None
        premarket_volume = 0.0

    regular_open = event_dt.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_from_open = (event_dt - regular_open).total_seconds() / 60.0 if local_time >= REGULAR_START else None

    return EventFeatures(
        event_volume=event_volume,
        cumulative_volume=cumulative_volume,
        volume_5m=float(pd.to_numeric(_window(observed, event_idx, 5)["v"], errors="coerce").fillna(0.0).sum()),
        volume_15m=float(pd.to_numeric(_window(observed, event_idx, 15)["v"], errors="coerce").fillna(0.0).sum()),
        volume_30m=float(pd.to_numeric(_window(observed, event_idx, 30)["v"], errors="coerce").fillna(0.0).sum()),
        volume_accel_1m_vs_prior20m=_volume_acceleration(observed, event_idx, 1, 20),
        volume_accel_5m_vs_prior20m=_volume_acceleration(observed, event_idx, 5, 20),
        session_vwap=session_vwap,
        vwap_distance_pct=vwap_distance,
        hod_distance_pct=hod_distance,
        return_5m_pct=_return_over_window(observed, event_idx, 5),
        return_15m_pct=_return_over_window(observed, event_idx, 15),
        return_30m_pct=_return_over_window(observed, event_idx, 30),
        volatility_5m_pct=_volatility(observed, event_idx, 5),
        volatility_15m_pct=_volatility(observed, event_idx, 15),
        volatility_30m_pct=_volatility(observed, event_idx, 30),
        premarket_return_pct=premarket_return,
        premarket_volume=premarket_volume,
        minutes_from_regular_open=minutes_from_open,
        is_premarket=is_premarket,
        is_regular_session=is_regular,
        is_after_hours=is_after_hours,
    )
