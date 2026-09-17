from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .events import CrossingEvent, detect_threshold_crossings
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS, add_execution_scenarios_to_record
from .exits import DEFAULT_BARRIERS, evaluate_default_barriers
from .features import extract_event_features


DEFAULT_HORIZONS = (1, 2, 5, 10, 15, 30, 60)
MINUTE_MS = 60_000


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
    """Measure outcomes at exact clock-time horizons.

    If the exact target minute has no bar, the return is left missing rather than
    silently substituting a later bar. This matters around trading halts and other
    gaps where "five bars later" may be much more than five elapsed minutes.
    MFE/MAE use only bars whose timestamps fall inside the requested clock window.
    """
    frame = _validate_bars(bars)
    matches = frame.index[frame["t"] == event.timestamp_ms].tolist()
    if len(matches) != 1:
        raise ValueError("event timestamp must match exactly one bar")

    requested = sorted({int(h) for h in horizons})
    if any(h <= 0 for h in requested):
        raise ValueError("horizons must be positive integers")

    timestamp_to_close = {
        int(row["t"]): float(row["c"])
        for _, row in frame.iterrows()
    }
    future_returns: dict[int, float | None] = {}
    for horizon in requested:
        target_ts = event.timestamp_ms + horizon * MINUTE_MS
        future_close = timestamp_to_close.get(target_ts)
        if future_close is None:
            future_returns[horizon] = None
        else:
            future_returns[horizon] = (future_close / event.price - 1.0) * 100.0

    max_horizon = max(requested, default=0)
    window_end = event.timestamp_ms + max_horizon * MINUTE_MS
    future_slice = frame.loc[
        (frame["t"] > event.timestamp_ms) & (frame["t"] <= window_end)
    ]
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


def _cost_adjusted_outcomes(
    event: CrossingEvent,
    outcome: EventOutcome,
    barriers_record: dict,
) -> dict[str, float | None]:
    """Apply transparent friction scenarios to horizon and barrier exits."""
    adjusted: dict[str, float | None] = {}
    for horizon, gross_return in outcome.future_returns_pct.items():
        adjusted.update(
            add_execution_scenarios_to_record(
                event.price,
                gross_return,
                DEFAULT_EXECUTION_SCENARIOS,
                prefix=f"return_{horizon}m",
            )
        )

    for key, gross_return in barriers_record.items():
        if not key.endswith("_exit_return_pct"):
            continue
        prefix = key.removesuffix("_exit_return_pct")
        adjusted.update(
            add_execution_scenarios_to_record(
                event.price,
                gross_return if isinstance(gross_return, (int, float)) else None,
                DEFAULT_EXECUTION_SCENARIOS,
                prefix=prefix,
            )
        )
    return adjusted


def run_event_study(
    ticker: str,
    bars: pd.DataFrame,
    previous_close: float,
    thresholds_pct: Iterable[float] = (10, 20, 30, 50, 75, 100),
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    history_bars: pd.DataFrame | None = None,
    barriers: Iterable[tuple[float, float]] = DEFAULT_BARRIERS,
) -> pd.DataFrame:
    """Emit point-in-time features plus gross and friction-adjusted outcomes."""
    requested_horizons = tuple(int(h) for h in horizons)
    events = detect_threshold_crossings(
        ticker=ticker,
        bars=bars,
        previous_close=previous_close,
        thresholds_pct=thresholds_pct,
    )
    if not events:
        return pd.DataFrame()

    max_horizon = max(requested_horizons, default=60)
    records: list[dict] = []
    for event in events:
        outcome = measure_event_outcome(event, bars, requested_horizons)
        features = extract_event_features(event, bars, history_bars=history_bars)
        barriers_record = evaluate_default_barriers(
            event,
            bars,
            barriers=barriers,
            max_horizon_minutes=max_horizon,
        )
        cost_record = _cost_adjusted_outcomes(event, outcome, barriers_record)
        record = outcome.to_record()
        feature_record = features.to_record()
        overlap = set(record) & set(feature_record)
        if overlap:
            raise ValueError(f"feature/outcome column collision: {sorted(overlap)}")
        record.update(feature_record)
        record.update(barriers_record)
        record.update(cost_record)
        records.append(record)
    return pd.DataFrame(records)
