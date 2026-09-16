from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .events import CrossingEvent, detect_threshold_crossings
from .exits import DEFAULT_BARRIERS, evaluate_default_barriers
from .features import extract_event_features


DEFAULT_HORIZONS = (1, 2, 5, 10, 15, 30, 60)


@dataclass(frozen=True)
class EventOutcome:
    ticker: str
    timestamp_ms: int
    threshold_pct: float
    entry_price: float
    previous_close: float
    future_returns_pct: dict[int, float | None]
    mfe_pct: float | None
    mae_pct: float | None

    def to_record(self) -> dict[str, float | int | str | None]:
        record: dict[str, float | int | str | None] = {
            "ticker": self.ticker,
            "timestamp_ms": self.timestamp_ms,
            "threshold_pct": self.threshold_pct,
            "entry_price": self.entry_price,
            "previous_close": self.previous_close,
            "mfe_pct": self.mfe_pct,
            "mae_pct": self.mae_pct,
        }
        for horizon, value in sorted(self.future_returns_pct.items()):
            record[f"return_{horizon}m_pct"] = value
        return record


def _validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"t", "h", "l", "c"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")
    return bars.sort_values("t").reset_index(drop=True)


def measure_event_outcome(
    event: CrossingEvent,
    bars: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> EventOutcome:
    frame = _validate_bars(bars)
    matches = frame.index[frame["t"] == event.timestamp_ms].tolist()
    if len(matches) != 1:
        raise ValueError("event timestamp must match exactly one bar")

    event_idx = matches[0]
    requested = sorted({int(h) for h in horizons})
    if any(h <= 0 for h in requested):
        raise ValueError("horizons must be positive integers")

    future_returns: dict[int, float | None] = {}
    for horizon in requested:
        target_idx = event_idx + horizon
        if target_idx >= len(frame):
            future_returns[horizon] = None
            continue
        future_close = float(frame.iloc[target_idx]["c"])
        future_returns[horizon] = (future_close / event.price - 1.0) * 100.0

    max_horizon = max(requested, default=0)
    future_slice = frame.iloc[event_idx + 1 : event_idx + max_horizon + 1]
    if future_slice.empty:
        mfe = None
        mae = None
    else:
        mfe = (float(future_slice["h"].max()) / event.price - 1.0) * 100.0
        mae = (float(future_slice["l"].min()) / event.price - 1.0) * 100.0

    return EventOutcome(
        ticker=event.ticker,
        timestamp_ms=event.timestamp_ms,
        threshold_pct=event.threshold_pct,
        entry_price=event.price,
        previous_close=event.previous_close,
        future_returns_pct=future_returns,
        mfe_pct=mfe,
        mae_pct=mae,
    )


def run_event_study(
    ticker: str,
    bars: pd.DataFrame,
    previous_close: float,
    thresholds_pct: Iterable[float] = (10, 20, 30, 50, 75, 100),
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    history_bars: pd.DataFrame | None = None,
    barriers: Iterable[tuple[float, float]] = DEFAULT_BARRIERS,
) -> pd.DataFrame:
    """Detect crossings and emit features, future paths, and short-exit labels."""
    events = detect_threshold_crossings(
        ticker=ticker,
        bars=bars,
        previous_close=previous_close,
        thresholds_pct=thresholds_pct,
    )
    if not events:
        return pd.DataFrame()

    max_horizon = max((int(h) for h in horizons), default=60)
    records: list[dict] = []
    for event in events:
        outcome = measure_event_outcome(event, bars, horizons)
        features = extract_event_features(event, bars, history_bars=history_bars)
        barriers_record = evaluate_default_barriers(
            event,
            bars,
            barriers=barriers,
            max_horizon_minutes=max_horizon,
        )
        record = outcome.to_record()
        record.update(features.to_record())
        record.update(barriers_record)
        records.append(record)
    return pd.DataFrame(records)
