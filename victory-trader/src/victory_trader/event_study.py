from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from .events import CrossingEvent, detect_threshold_crossings
from .execution_costs import (
    DEFAULT_EXECUTION_SCENARIOS,
    add_execution_scenarios_to_record,
)
from .exits import DEFAULT_BARRIERS, evaluate_default_barriers
from .features import extract_event_features, timestamp_et
from .market_calendar import regular_session_bounds


DEFAULT_HORIZONS = (1, 2, 5, 10, 15, 30, 60)
DEFAULT_ENTRY_DELAYS = (0, 1, 2)
MINUTE_MS = 60_000


@dataclass(frozen=True)
class EventOutcome:
    ticker: str
    timestamp_ms: int
    threshold_pct: float
    signal_price: float
    previous_close: float
    entry_timestamp_ms: int | None
    entry_price: float | None
    entry_delay_minutes: int
    future_returns_pct: dict[int, float | None]
    signal_returns_pct: dict[int, float | None]
    mfe_pct: float | None
    mae_pct: float | None

    def to_record(self) -> dict[str, float | int | str | None]:
        record: dict[str, float | int | str | None] = {
            "ticker": self.ticker,
            "timestamp_ms": self.timestamp_ms,
            "threshold_pct": self.threshold_pct,
            "signal_price": self.signal_price,
            "entry_timestamp_ms": self.entry_timestamp_ms,
            "entry_price": self.entry_price,
            "entry_delay_minutes": self.entry_delay_minutes,
            "entry_model": "next_minute_open",
            "previous_close": self.previous_close,
            "mfe_pct": self.mfe_pct,
            "mae_pct": self.mae_pct,
        }
        for horizon, value in sorted(self.future_returns_pct.items()):
            record[f"return_{horizon}m_pct"] = value
        for horizon, value in sorted(self.signal_returns_pct.items()):
            record[f"signal_return_{horizon}m_pct"] = value
        return record


def _validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"t", "o", "h", "l", "c"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")
    return bars.sort_values("t").reset_index(drop=True)


def _close_at(frame: pd.DataFrame, timestamp_ms: int) -> float | None:
    values = frame.loc[frame["t"] == timestamp_ms, "c"]
    return float(values.iloc[0]) if len(values) == 1 else None


def _open_at(frame: pd.DataFrame, timestamp_ms: int) -> float | None:
    values = frame.loc[frame["t"] == timestamp_ms, "o"]
    return float(values.iloc[0]) if len(values) == 1 else None


def _is_regular_timestamp(timestamp_ms: int) -> bool:
    dt = timestamp_et(timestamp_ms)
    bounds = regular_session_bounds(dt.date())
    return bounds is not None and bounds[0] <= dt < bounds[1]


def _regular_detection_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = frame["t"].map(lambda value: _is_regular_timestamp(int(value)))
    return frame.loc[mask].reset_index(drop=True)


def _signal_close_returns(
    event: CrossingEvent,
    frame: pd.DataFrame,
    horizons: Iterable[int],
) -> dict[int, float | None]:
    result: dict[int, float | None] = {}
    for horizon in horizons:
        future_close = _close_at(frame, event.timestamp_ms + horizon * MINUTE_MS)
        result[horizon] = (
            None
            if future_close is None
            else (future_close / event.price - 1.0) * 100.0
        )
    return result


def measure_event_outcome(
    event: CrossingEvent,
    bars: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    *,
    entry_delay_minutes: int = 0,
    require_regular_entry: bool = False,
) -> EventOutcome:
    """Measure executable outcomes from a delayed next-minute-open entry.

    A close-based event is observable only after its minute closes. Delay 0 therefore
    enters at the exact next minute's open; delay 1/2 wait one/two additional clock
    minutes. Formal v0.2 regular-session research requires the entry and each measured
    executable horizon to stay inside the official regular session.
    """
    if entry_delay_minutes < 0:
        raise ValueError("entry_delay_minutes must be non-negative")

    frame = _validate_bars(bars)
    matches = frame.index[frame["t"] == event.timestamp_ms].tolist()
    if len(matches) != 1:
        raise ValueError("event timestamp must match exactly one bar")

    requested = sorted({int(h) for h in horizons})
    if any(h <= 0 for h in requested):
        raise ValueError("horizons must be positive integers")

    signal_returns = _signal_close_returns(event, frame, requested)
    entry_ts = event.timestamp_ms + (entry_delay_minutes + 1) * MINUTE_MS
    entry_price = _open_at(frame, entry_ts)
    if require_regular_entry and not _is_regular_timestamp(entry_ts):
        entry_price = None

    if entry_price is None or entry_price <= 0:
        return EventOutcome(
            ticker=event.ticker,
            timestamp_ms=event.timestamp_ms,
            threshold_pct=event.threshold_pct,
            signal_price=event.price,
            previous_close=event.previous_close,
            entry_timestamp_ms=None,
            entry_price=None,
            entry_delay_minutes=entry_delay_minutes,
            future_returns_pct={h: None for h in requested},
            signal_returns_pct=signal_returns,
            mfe_pct=None,
            mae_pct=None,
        )

    future_returns: dict[int, float | None] = {}
    for horizon in requested:
        target_bar_ts = entry_ts + (horizon - 1) * MINUTE_MS
        if require_regular_entry and not _is_regular_timestamp(target_bar_ts):
            future_returns[horizon] = None
            continue
        future_close = _close_at(frame, target_bar_ts)
        future_returns[horizon] = (
            None
            if future_close is None
            else (future_close / entry_price - 1.0) * 100.0
        )

    max_horizon = max(requested, default=0)
    window_end_bar_ts = entry_ts + max(max_horizon - 1, 0) * MINUTE_MS
    future_slice = frame.loc[(frame["t"] >= entry_ts) & (frame["t"] <= window_end_bar_ts)]
    if require_regular_entry and not future_slice.empty:
        regular_mask = future_slice["t"].map(lambda value: _is_regular_timestamp(int(value)))
        future_slice = future_slice.loc[regular_mask]
    if future_slice.empty:
        mfe = None
        mae = None
    else:
        mfe = (float(future_slice["h"].max()) / entry_price - 1.0) * 100.0
        mae = (float(future_slice["l"].min()) / entry_price - 1.0) * 100.0

    return EventOutcome(
        ticker=event.ticker,
        timestamp_ms=event.timestamp_ms,
        threshold_pct=event.threshold_pct,
        signal_price=event.price,
        previous_close=event.previous_close,
        entry_timestamp_ms=entry_ts,
        entry_price=entry_price,
        entry_delay_minutes=entry_delay_minutes,
        future_returns_pct=future_returns,
        signal_returns_pct=signal_returns,
        mfe_pct=mfe,
        mae_pct=mae,
    )


def _cost_adjusted_outcomes(
    entry_price: float | None,
    future_returns_pct: dict[int, float | None],
    barriers_record: dict,
    *,
    horizon_prefix: str = "return",
) -> dict[str, float | None]:
    adjusted: dict[str, float | None] = {}
    for horizon, gross_return in future_returns_pct.items():
        prefix = f"{horizon_prefix}_{horizon}m"
        if entry_price is None:
            for scenario in DEFAULT_EXECUTION_SCENARIOS:
                adjusted[f"{prefix}_{scenario.name}_net_return_pct"] = None
        else:
            adjusted.update(
                add_execution_scenarios_to_record(
                    entry_price,
                    gross_return,
                    DEFAULT_EXECUTION_SCENARIOS,
                    prefix=prefix,
                )
            )

    if horizon_prefix != "return":
        return adjusted

    for key, gross_return in barriers_record.items():
        if not key.endswith("_exit_return_pct"):
            continue
        prefix = key.removesuffix("_exit_return_pct")
        if entry_price is None:
            for scenario in DEFAULT_EXECUTION_SCENARIOS:
                adjusted[f"{prefix}_{scenario.name}_net_return_pct"] = None
        else:
            adjusted.update(
                add_execution_scenarios_to_record(
                    entry_price,
                    gross_return if isinstance(gross_return, (int, float)) else None,
                    DEFAULT_EXECUTION_SCENARIOS,
                    prefix=prefix,
                )
            )
    return adjusted


def _entry_unavailable_barriers(
    barriers: Iterable[tuple[float, float]],
) -> dict[str, str | int | float | None]:
    record: dict[str, str | int | float | None] = {}
    for tp, sl in barriers:
        tp_text = str(float(tp)).rstrip("0").rstrip(".").replace(".", "p")
        sl_text = str(float(sl)).rstrip("0").rstrip(".").replace(".", "p")
        key = f"tp{tp_text}_sl{sl_text}"
        record[f"{key}_status"] = "entry_unavailable"
        record[f"{key}_minutes"] = None
        record[f"{key}_exit_return_pct"] = None
    return record


def run_event_study(
    ticker: str,
    bars: pd.DataFrame,
    previous_close: float,
    thresholds_pct: Iterable[float] = (10, 20, 30, 50, 75, 100),
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    history_bars: pd.DataFrame | None = None,
    barriers: Iterable[tuple[float, float]] = DEFAULT_BARRIERS,
    entry_delays: Iterable[int] = DEFAULT_ENTRY_DELAYS,
    *,
    event_session_scope: str = "all",
    require_regular_entry: bool = False,
) -> pd.DataFrame:
    """Emit point-in-time features and executable, friction-adjusted outcomes."""
    requested_horizons = tuple(int(h) for h in horizons)
    requested_barriers = tuple((float(tp), float(sl)) for tp, sl in barriers)
    delays = tuple(sorted({int(delay) for delay in entry_delays}))
    if 0 not in delays:
        raise ValueError("entry_delays must include 0 for the primary next-minute-open model")
    if event_session_scope not in {"all", "regular"}:
        raise ValueError("event_session_scope must be 'all' or 'regular'")

    frame = _validate_bars(bars)
    detection_bars = _regular_detection_bars(frame) if event_session_scope == "regular" else frame
    events = detect_threshold_crossings(
        ticker=ticker,
        bars=detection_bars,
        previous_close=previous_close,
        thresholds_pct=thresholds_pct,
    )
    if not events:
        return pd.DataFrame()

    max_horizon = max(requested_horizons, default=60)
    records: list[dict] = []
    for event in events:
        primary = measure_event_outcome(
            event,
            frame,
            requested_horizons,
            entry_delay_minutes=0,
            require_regular_entry=require_regular_entry,
        )
        features = extract_event_features(event, frame, history_bars=history_bars)

        if primary.entry_price is None or primary.entry_timestamp_ms is None:
            barriers_record = _entry_unavailable_barriers(requested_barriers)
        else:
            barriers_record = evaluate_default_barriers(
                event,
                frame,
                barriers=requested_barriers,
                max_horizon_minutes=max_horizon,
                entry_timestamp_ms=primary.entry_timestamp_ms,
                entry_price=primary.entry_price,
                regular_session_only=require_regular_entry,
            )

        cost_record = _cost_adjusted_outcomes(
            primary.entry_price,
            primary.future_returns_pct,
            barriers_record,
        )
        record = primary.to_record()
        record["event_session_scope"] = event_session_scope
        record["regular_entry_required"] = require_regular_entry
        feature_record = features.to_record()
        overlap = set(record) & set(feature_record)
        if overlap:
            raise ValueError(f"feature/outcome column collision: {sorted(overlap)}")
        record.update(feature_record)
        record.update(barriers_record)
        record.update(cost_record)

        for delay in delays:
            if delay == 0:
                continue
            delayed = measure_event_outcome(
                event,
                frame,
                requested_horizons,
                entry_delay_minutes=delay,
                require_regular_entry=require_regular_entry,
            )
            prefix = f"delay{delay}"
            record[f"{prefix}_entry_timestamp_ms"] = delayed.entry_timestamp_ms
            record[f"{prefix}_entry_price"] = delayed.entry_price
            for horizon, value in delayed.future_returns_pct.items():
                record[f"{prefix}_return_{horizon}m_pct"] = value
            record.update(
                _cost_adjusted_outcomes(
                    delayed.entry_price,
                    delayed.future_returns_pct,
                    {},
                    horizon_prefix=f"{prefix}_return",
                )
            )

        records.append(record)
    return pd.DataFrame(records)
