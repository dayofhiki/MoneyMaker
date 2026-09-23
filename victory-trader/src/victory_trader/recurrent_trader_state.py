"""Causal per-ticker episode memory for the executable recurrent trader."""

from __future__ import annotations

from dataclasses import dataclass
import math

MINUTE_MS = 60_000


@dataclass
class EpisodeSnapshot:
    trading_day: str
    ticker: str
    t: int
    state: str
    promotion_index: int
    minutes_since_previous_promotion: float | None
    minutes_since_last_promotion: float | None
    minutes_continuously_hot: float
    minutes_continuously_observed: float
    hot_score_peak: float | None
    hot_score_drawdown_from_peak: float | None
    opportunity_score_peak: float | None
    opportunity_score_drawdown_from_peak: float | None
    position_open: bool


@dataclass
class _TickerMemory:
    trading_day: str
    ticker: str
    last_t: int | None = None
    observed_since_t: int | None = None
    previous_hot: bool = False
    continuous_hot_since_t: int | None = None
    promotion_index: int = 0
    last_promotion_t: int | None = None
    hot_score_peak: float | None = None
    opportunity_score_peak: float | None = None
    position_open: bool = False


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class CausalEpisodeMemory:
    """State machine memory that never needs future information.

    One instance can track many tickers. Memory resets at a trading-day change
    for a ticker, but survives HOT demotion/re-promotion within the session.
    """

    def __init__(self) -> None:
        self._memory: dict[str, _TickerMemory] = {}

    def _get(self, trading_day: str, ticker: str) -> _TickerMemory:
        key = ticker.upper()
        current = self._memory.get(key)
        if current is None or current.trading_day != trading_day:
            current = _TickerMemory(trading_day=trading_day, ticker=key)
            self._memory[key] = current
        return current

    def observe(
        self,
        *,
        trading_day: str,
        ticker: str,
        t: int,
        state: str,
        hot_score: float | None = None,
        opportunity_score: float | None = None,
        observed: bool = True,
        position_open: bool | None = None,
    ) -> EpisodeSnapshot:
        memory = self._get(str(trading_day), str(ticker).upper())
        timestamp = int(t)
        if memory.last_t is not None and timestamp < memory.last_t:
            raise ValueError(
                f"non-monotonic episode time for {memory.ticker}: "
                f"{timestamp} < {memory.last_t}"
            )

        if position_open is not None:
            memory.position_open = bool(position_open)

        if observed:
            if memory.observed_since_t is None:
                memory.observed_since_t = timestamp
        else:
            memory.observed_since_t = None

        state_text = str(state).lower()
        is_hot = state_text == "hot"
        is_position = state_text == "position"
        promotion = is_hot and not memory.previous_hot
        minutes_since_previous: float | None = None
        if promotion:
            if memory.last_promotion_t is not None:
                minutes_since_previous = (
                    timestamp - memory.last_promotion_t
                ) / MINUTE_MS
            memory.promotion_index += 1
            memory.last_promotion_t = timestamp
            memory.continuous_hot_since_t = timestamp
        elif is_hot and memory.continuous_hot_since_t is None:
            memory.continuous_hot_since_t = timestamp
        elif not is_hot and not is_position:
            memory.continuous_hot_since_t = None

        minutes_since_last = (
            (timestamp - memory.last_promotion_t) / MINUTE_MS
            if memory.last_promotion_t is not None
            else None
        )

        hot_value = _finite(hot_score)
        if hot_value is not None:
            if memory.hot_score_peak is None:
                memory.hot_score_peak = hot_value
            else:
                memory.hot_score_peak = max(memory.hot_score_peak, hot_value)

        opportunity_value = _finite(opportunity_score)
        if opportunity_value is not None:
            if memory.opportunity_score_peak is None:
                memory.opportunity_score_peak = opportunity_value
            else:
                memory.opportunity_score_peak = max(
                    memory.opportunity_score_peak,
                    opportunity_value,
                )

        hot_drawdown = (
            hot_value - memory.hot_score_peak
            if hot_value is not None and memory.hot_score_peak is not None
            else None
        )
        opportunity_drawdown = (
            opportunity_value - memory.opportunity_score_peak
            if opportunity_value is not None
            and memory.opportunity_score_peak is not None
            else None
        )

        hot_minutes = (
            (timestamp - memory.continuous_hot_since_t) / MINUTE_MS
            if (is_hot or is_position)
            and memory.continuous_hot_since_t is not None
            else 0.0
        )
        observed_minutes = (
            (timestamp - memory.observed_since_t) / MINUTE_MS
            if observed and memory.observed_since_t is not None
            else 0.0
        )

        if is_hot:
            memory.previous_hot = True
        elif not is_position:
            memory.previous_hot = False
        memory.last_t = timestamp
        return EpisodeSnapshot(
            trading_day=memory.trading_day,
            ticker=memory.ticker,
            t=timestamp,
            state=state_text,
            promotion_index=memory.promotion_index,
            minutes_since_previous_promotion=minutes_since_previous,
            minutes_since_last_promotion=(
                float(minutes_since_last)
                if minutes_since_last is not None
                else None
            ),
            minutes_continuously_hot=float(hot_minutes),
            minutes_continuously_observed=float(observed_minutes),
            hot_score_peak=memory.hot_score_peak,
            hot_score_drawdown_from_peak=hot_drawdown,
            opportunity_score_peak=memory.opportunity_score_peak,
            opportunity_score_drawdown_from_peak=opportunity_drawdown,
            position_open=memory.position_open,
        )

    def set_position(
        self,
        *,
        trading_day: str,
        ticker: str,
        position_open: bool,
    ) -> None:
        memory = self._get(str(trading_day), str(ticker).upper())
        memory.position_open = bool(position_open)

    def clear_observation(self, *, trading_day: str, ticker: str) -> None:
        memory = self._get(str(trading_day), str(ticker).upper())
        memory.observed_since_t = None

    def reset(self) -> None:
        self._memory.clear()
