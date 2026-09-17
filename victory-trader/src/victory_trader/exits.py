from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from .events import CrossingEvent
from .features import timestamp_et
from .market_calendar import regular_session_bounds


DEFAULT_BARRIERS = ((2.0, 1.0), (3.0, 2.0), (5.0, 3.0), (10.0, 5.0))
MINUTE_MS = 60_000


@dataclass(frozen=True)
class BarrierOutcome:
    take_profit_pct: float
    stop_loss_pct: float
    status: str
    minutes_to_exit: int | None
    exit_return_pct: float | None

    @property
    def key(self) -> str:
        tp = str(self.take_profit_pct).rstrip("0").rstrip(".").replace(".", "p")
        sl = str(self.stop_loss_pct).rstrip("0").rstrip(".").replace(".", "p")
        return f"tp{tp}_sl{sl}"

    def to_record(self) -> dict[str, str | int | float | None]:
        prefix = self.key
        return {
            f"{prefix}_status": self.status,
            f"{prefix}_minutes": self.minutes_to_exit,
            f"{prefix}_exit_return_pct": self.exit_return_pct,
        }


def evaluate_barrier(
    event: CrossingEvent,
    bars: pd.DataFrame,
    *,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_horizon_minutes: int = 60,
    entry_timestamp_ms: int | None = None,
    entry_price: float | None = None,
    regular_session_only: bool = False,
) -> BarrierOutcome:
    """Evaluate TP/SL using elapsed clock time and conservative gap handling.

    A stop-market gap through the stop is filled at the observed bar open, not at
    the unattainable stop level. A gap through a take-profit remains conservatively
    filled at the target. If both high and low touch inside one OHLC minute, the
    ordering is unknowable and the result is ``ambiguous``.

    Formal v0.2 research is regular-session-only. If a barrier position would run
    past the close, it is forced out at the last regular bar close when observable.
    Missing that bar is reported as ``unresolved_session_close`` rather than silently
    switching to after-hours execution.
    """
    if take_profit_pct <= 0 or stop_loss_pct <= 0:
        raise ValueError("take-profit and stop-loss percentages must be positive")
    if max_horizon_minutes <= 0:
        raise ValueError("max_horizon_minutes must be positive")

    frame = bars.sort_values("t").reset_index(drop=True)
    required = {"t", "o", "h", "l", "c"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    signal_matches = frame.index[frame["t"] == event.timestamp_ms].tolist()
    if len(signal_matches) != 1:
        raise ValueError("event timestamp must match exactly one bar")

    effective_entry_ts = event.timestamp_ms + MINUTE_MS if entry_timestamp_ms is None else int(entry_timestamp_ms)
    reference_price = event.price if entry_price is None else float(entry_price)
    if reference_price <= 0:
        raise ValueError("entry_price must be positive")

    tp_price = reference_price * (1.0 + take_profit_pct / 100.0)
    sl_price = reference_price * (1.0 - stop_loss_pct / 100.0)
    requested_timeout_bar_ts = effective_entry_ts + (max_horizon_minutes - 1) * MINUTE_MS
    timeout_bar_ts = requested_timeout_bar_ts
    forced_session_close = False

    if regular_session_only:
        entry_dt = timestamp_et(effective_entry_ts)
        bounds = regular_session_bounds(entry_dt.date())
        if bounds is None or not (bounds[0] <= entry_dt < bounds[1]):
            return BarrierOutcome(
                take_profit_pct,
                stop_loss_pct,
                "entry_unavailable",
                None,
                None,
            )
        last_regular_bar_ts = int(bounds[1].timestamp() * 1000) - MINUTE_MS
        if timeout_bar_ts > last_regular_bar_ts:
            timeout_bar_ts = last_regular_bar_ts
            forced_session_close = True

    future = frame.loc[(frame["t"] >= effective_entry_ts) & (frame["t"] <= timeout_bar_ts)]

    for _, row in future.iterrows():
        row_ts = int(row["t"])
        minute_number = int((row_ts - effective_entry_ts) // MINUTE_MS) + 1
        row_open = float(row["o"])

        if row_open <= sl_price:
            gap_return = (row_open / reference_price - 1.0) * 100.0
            return BarrierOutcome(
                take_profit_pct,
                stop_loss_pct,
                "stop_gap",
                minute_number,
                gap_return,
            )
        if row_open >= tp_price:
            return BarrierOutcome(
                take_profit_pct,
                stop_loss_pct,
                "take_profit",
                minute_number,
                take_profit_pct,
            )

        hit_tp = float(row["h"]) >= tp_price
        hit_sl = float(row["l"]) <= sl_price
        if hit_tp and hit_sl:
            return BarrierOutcome(
                take_profit_pct,
                stop_loss_pct,
                "ambiguous",
                minute_number,
                None,
            )
        if hit_tp:
            return BarrierOutcome(
                take_profit_pct,
                stop_loss_pct,
                "take_profit",
                minute_number,
                take_profit_pct,
            )
        if hit_sl:
            return BarrierOutcome(
                take_profit_pct,
                stop_loss_pct,
                "stop_loss",
                minute_number,
                -stop_loss_pct,
            )

    if future.empty:
        return BarrierOutcome(take_profit_pct, stop_loss_pct, "no_future_data", None, None)

    exact_timeout = future.loc[future["t"] == timeout_bar_ts]
    if len(exact_timeout) != 1:
        status = "unresolved_session_close" if forced_session_close else "unresolved_missing"
        return BarrierOutcome(take_profit_pct, stop_loss_pct, status, None, None)

    final_close = float(exact_timeout.iloc[0]["c"])
    timeout_return = (final_close / reference_price - 1.0) * 100.0
    minutes_to_exit = int((timeout_bar_ts - effective_entry_ts) // MINUTE_MS) + 1
    return BarrierOutcome(
        take_profit_pct,
        stop_loss_pct,
        "session_close" if forced_session_close else "timeout",
        minutes_to_exit,
        timeout_return,
    )


def evaluate_default_barriers(
    event: CrossingEvent,
    bars: pd.DataFrame,
    barriers: Iterable[tuple[float, float]] = DEFAULT_BARRIERS,
    *,
    max_horizon_minutes: int = 60,
    entry_timestamp_ms: int | None = None,
    entry_price: float | None = None,
    regular_session_only: bool = False,
) -> dict[str, str | int | float | None]:
    record: dict[str, str | int | float | None] = {}
    for take_profit_pct, stop_loss_pct in barriers:
        outcome = evaluate_barrier(
            event,
            bars,
            take_profit_pct=float(take_profit_pct),
            stop_loss_pct=float(stop_loss_pct),
            max_horizon_minutes=max_horizon_minutes,
            entry_timestamp_ms=entry_timestamp_ms,
            entry_price=entry_price,
            regular_session_only=regular_session_only,
        )
        record.update(outcome.to_record())
    return record
