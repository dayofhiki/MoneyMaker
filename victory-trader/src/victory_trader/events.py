from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CrossingEvent:
    ticker: str
    timestamp_ms: int
    threshold_pct: float
    price: float
    previous_close: float


def detect_threshold_crossings(
    ticker: str,
    bars: pd.DataFrame,
    previous_close: float,
    thresholds_pct: Iterable[float] = (10, 20, 30, 50, 75, 100),
) -> list[CrossingEvent]:
    """Return the first close-price crossing for each intraday return threshold.

    Expected columns: `t` (Unix milliseconds) and `c` (close).
    This is intentionally simple. Later versions will support high/low crossing,
    session filters, halts, and richer event definitions.
    """
    if previous_close <= 0:
        raise ValueError("previous_close must be positive")
    if bars.empty:
        return []
    missing = {"t", "c"} - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    frame = bars.sort_values("t").copy()
    frame["return_pct"] = (frame["c"] / previous_close - 1.0) * 100.0

    events: list[CrossingEvent] = []
    for threshold in sorted(set(float(x) for x in thresholds_pct)):
        crossed = frame.loc[frame["return_pct"] >= threshold]
        if crossed.empty:
            continue
        row = crossed.iloc[0]
        events.append(
            CrossingEvent(
                ticker=ticker.upper(),
                timestamp_ms=int(row["t"]),
                threshold_pct=threshold,
                price=float(row["c"]),
                previous_close=float(previous_close),
            )
        )
    return events
