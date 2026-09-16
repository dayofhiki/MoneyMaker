from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .events import CrossingEvent


DEFAULT_BARRIERS = ((2.0, 1.0), (3.0, 2.0), (5.0, 3.0), (10.0, 5.0))


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
) -> BarrierOutcome:
    """Evaluate which price barrier is observed first after the event bar.

    If both barriers are touched inside the same 1-minute bar, ordering is not
    observable from OHLC data and the result is marked `ambiguous` instead of
    giving the backtest a favorable fill assumption.
    """
    if take_profit_pct <= 0 or stop_loss_pct <= 0:
        raise ValueError("take-profit and stop-loss percentages must be positive")
    if max_horizon_minutes <= 0:
        raise ValueError("max_horizon_minutes must be positive")

    frame = bars.sort_values("t").reset_index(drop=True)
    matches = frame.index[frame["t"] == event.timestamp_ms].tolist()
    if len(matches) != 1:
        raise ValueError("event timestamp must match exactly one bar")

    event_idx = matches[0]
    tp_price = event.price * (1.0 + take_profit_pct / 100.0)
    sl_price = event.price * (1.0 - stop_loss_pct / 100.0)
    future = frame.iloc[event_idx + 1 : event_idx + max_horizon_minutes + 1]

    for minute, (_, row) in enumerate(future.iterrows(), start=1):
        hit_tp = float(row["h"]) >= tp_price
        hit_sl = float(row["l"]) <= sl_price
        if hit_tp and hit_sl:
            return BarrierOutcome(
                take_profit_pct,
                stop_loss_pct,
                "ambiguous",
                minute,
                None,
            )
        if hit_tp:
            return BarrierOutcome(
                take_profit_pct,
                stop_loss_pct,
                "take_profit",
                minute,
                take_profit_pct,
            )
        if hit_sl:
            return BarrierOutcome(
                take_profit_pct,
                stop_loss_pct,
                "stop_loss",
                minute,
                -stop_loss_pct,
            )

    if future.empty:
        return BarrierOutcome(take_profit_pct, stop_loss_pct, "no_future_data", None, None)

    final_close = float(future.iloc[-1]["c"])
    timeout_return = (final_close / event.price - 1.0) * 100.0
    return BarrierOutcome(
        take_profit_pct,
        stop_loss_pct,
        "timeout",
        len(future),
        timeout_return,
    )


def evaluate_default_barriers(
    event: CrossingEvent,
    bars: pd.DataFrame,
    barriers: Iterable[tuple[float, float]] = DEFAULT_BARRIERS,
    *,
    max_horizon_minutes: int = 60,
) -> dict[str, str | int | float | None]:
    record: dict[str, str | int | float | None] = {}
    for take_profit_pct, stop_loss_pct in barriers:
        outcome = evaluate_barrier(
            event,
            bars,
            take_profit_pct=float(take_profit_pct),
            stop_loss_pct=float(stop_loss_pct),
            max_horizon_minutes=max_horizon_minutes,
        )
        record.update(outcome.to_record())
    return record
