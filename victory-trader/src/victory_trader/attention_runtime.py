"""Stateful market-wide attention and observation-resolution runtime.

This module intentionally does not decide how an attention score is produced.
Research models supply point-in-time scores; the runtime turns those scores and
position state into a causal, resource-bounded observation plan.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite


class AttentionState(StrEnum):
    SCAN = "scan"
    WATCH = "watch"
    HOT = "hot"
    POSITION = "position"
    DROP = "drop"


class ObservationResolution(StrEnum):
    NONE = "none"
    GROUPED_MINUTE = "grouped_minute"
    MINUTE_BARS = "minute_bars"
    SECOND_BARS = "second_bars"
    TRADES_NBBO = "trades_nbbo"


@dataclass(frozen=True)
class AttentionConfig:
    """Frozen runtime thresholds and compute budgets.

    WATCH and HOT capacities are separate: ``max_watch`` is the number of
    minute-resolution WATCH names and ``max_hot`` is the number of
    second-resolution HOT names. Open positions are safety-critical and do not
    compete for either budget.
    """

    watch_enter_score: float = 0.60
    watch_exit_score: float = 0.40
    hot_enter_score: float = 0.85
    hot_exit_score: float = 0.70
    max_watch: int = 50
    max_hot: int = 10
    drop_after_missed_batches: int = 3

    def __post_init__(self) -> None:
        thresholds = (
            self.watch_enter_score,
            self.watch_exit_score,
            self.hot_enter_score,
            self.hot_exit_score,
        )
        if not all(isfinite(value) for value in thresholds):
            raise ValueError("attention thresholds must be finite")
        if self.watch_exit_score > self.watch_enter_score:
            raise ValueError("watch exit score must not exceed enter score")
        if self.hot_exit_score > self.hot_enter_score:
            raise ValueError("hot exit score must not exceed enter score")
        if self.hot_enter_score < self.watch_enter_score:
            raise ValueError("hot enter score must be at least watch enter score")
        if min(self.max_watch, self.max_hot) < 0:
            raise ValueError("attention capacities must be non-negative")
        if self.drop_after_missed_batches < 1:
            raise ValueError("drop_after_missed_batches must be positive")


@dataclass(frozen=True)
class AttentionEvidence:
    """Point-in-time evidence emitted by the broad scanner for one ticker."""

    ticker: str
    attention_score: float
    position_open: bool = False
    position_just_closed: bool = False
    data_valid: bool = True
    eligible: bool = True

    def __post_init__(self) -> None:
        normalized = self.ticker.strip().upper()
        if not normalized:
            raise ValueError("ticker must not be blank")
        object.__setattr__(self, "ticker", normalized)
        if not isfinite(self.attention_score):
            raise ValueError("attention_score must be finite")
        if self.position_open and self.position_just_closed:
            raise ValueError(
                "position_open and position_just_closed cannot both be true"
            )


@dataclass(frozen=True)
class ObservationPlan:
    ticker: str
    state: AttentionState
    resolution: ObservationResolution
    feeds: tuple[str, ...]
    attention_score: float
    rank: int | None
    reason: str
    timestamp_ms: int


@dataclass
class _AttentionRecord:
    ticker: str
    state: AttentionState
    attention_score: float
    last_observed_ms: int
    missed_batches: int = 0
    reason: str = "new_symbol"


_OBSERVATION_REQUIREMENTS: dict[
    AttentionState, tuple[ObservationResolution, tuple[str, ...]]
] = {
    AttentionState.SCAN: (
        ObservationResolution.GROUPED_MINUTE,
        ("grouped_minute",),
    ),
    AttentionState.WATCH: (
        ObservationResolution.MINUTE_BARS,
        ("minute_bars",),
    ),
    AttentionState.HOT: (
        ObservationResolution.SECOND_BARS,
        ("second_bars", "nbbo"),
    ),
    AttentionState.POSITION: (
        ObservationResolution.TRADES_NBBO,
        ("trades", "nbbo"),
    ),
    AttentionState.DROP: (ObservationResolution.NONE, ()),
}


class AttentionRuntime:
    """Allocate finite observation capacity across a changing market.

    ``step`` represents one completed broad-market scan. Its timestamp must
    strictly increase, which makes accidental replay and look-ahead ordering
    errors explicit. Every supplied score is assumed to have been computed
    using information available at that timestamp only.
    """

    def __init__(self, config: AttentionConfig | None = None) -> None:
        self.config = config or AttentionConfig()
        self._records: dict[str, _AttentionRecord] = {}
        self._last_timestamp_ms: int | None = None

    def step(
        self,
        timestamp_ms: int,
        evidence: Iterable[AttentionEvidence],
    ) -> dict[str, ObservationPlan]:
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("attention timestamps must strictly increase")

        observed: dict[str, AttentionEvidence] = {}
        for item in evidence:
            if item.ticker in observed:
                raise ValueError(f"duplicate ticker in attention batch: {item.ticker}")
            observed[item.ticker] = item

        desired: dict[str, tuple[AttentionState, str]] = {}
        for ticker, item in observed.items():
            previous = self._records.get(ticker)
            desired[ticker] = self._desired_state(item, previous)

        ranked = sorted(
            (
                item
                for item in observed.values()
                if desired[item.ticker][0] in {AttentionState.HOT, AttentionState.WATCH}
            ),
            key=lambda item: (-item.attention_score, item.ticker),
        )
        rank_by_ticker = {item.ticker: rank for rank, item in enumerate(ranked, 1)}

        hot_candidates = [
            item for item in ranked if desired[item.ticker][0] is AttentionState.HOT
        ]
        selected_hot = {
            item.ticker for item in hot_candidates[: self.config.max_hot]
        }

        watch_candidates = [
            item
            for item in ranked
            if desired[item.ticker][0] is AttentionState.WATCH
            or (
                desired[item.ticker][0] is AttentionState.HOT
                and item.ticker not in selected_hot
            )
        ]
        # A just-closed position receives one WATCH cycle before it must win
        # ordinary score-based capacity again. This keeps exit/re-entry
        # reasoning continuous and avoids dropping its feed at the fill event.
        watch_candidates.sort(
            key=lambda item: (
                not item.position_just_closed,
                -item.attention_score,
                item.ticker,
            )
        )
        selected_watch = {
            item.ticker for item in watch_candidates[: self.config.max_watch]
        }

        for ticker, item in observed.items():
            requested_state, reason = desired[ticker]
            if requested_state is AttentionState.HOT:
                if ticker in selected_hot:
                    state = AttentionState.HOT
                elif ticker in selected_watch:
                    state = AttentionState.WATCH
                    reason = "hot_budget_demote"
                else:
                    state = AttentionState.SCAN
                    reason = "attention_budget_scan"
            elif requested_state is AttentionState.WATCH:
                if ticker in selected_watch:
                    state = AttentionState.WATCH
                else:
                    state = AttentionState.SCAN
                    reason = "watch_budget_scan"
            else:
                state = requested_state

            self._records[ticker] = _AttentionRecord(
                ticker=ticker,
                state=state,
                attention_score=item.attention_score,
                last_observed_ms=timestamp_ms,
                missed_batches=0,
                reason=reason,
            )

        for ticker, record in self._records.items():
            if ticker in observed:
                continue
            record.missed_batches += 1
            if record.state is AttentionState.POSITION:
                record.reason = "position_feed_missing"
            elif record.missed_batches >= self.config.drop_after_missed_batches:
                record.state = AttentionState.DROP
                record.reason = "observation_missing"
            else:
                record.reason = "observation_grace"

        self._last_timestamp_ms = timestamp_ms
        return {
            ticker: self._plan(record, timestamp_ms, rank_by_ticker.get(ticker))
            for ticker, record in sorted(self._records.items())
        }

    def _desired_state(
        self,
        item: AttentionEvidence,
        previous: _AttentionRecord | None,
    ) -> tuple[AttentionState, str]:
        if item.position_open:
            return AttentionState.POSITION, "position_open"
        if not item.data_valid:
            return AttentionState.DROP, "invalid_data"
        if not item.eligible:
            return AttentionState.DROP, "ineligible"
        if item.position_just_closed:
            return AttentionState.WATCH, "position_closed_reobserve"

        prior_state = previous.state if previous is not None else AttentionState.SCAN
        hot_boundary = (
            self.config.hot_exit_score
            if prior_state is AttentionState.HOT
            else self.config.hot_enter_score
        )
        if item.attention_score >= hot_boundary:
            reason = "hot_hysteresis" if prior_state is AttentionState.HOT else "hot_score"
            return AttentionState.HOT, reason

        watch_boundary = (
            self.config.watch_exit_score
            if prior_state in {AttentionState.WATCH, AttentionState.HOT}
            else self.config.watch_enter_score
        )
        if item.attention_score >= watch_boundary:
            reason = (
                "watch_hysteresis"
                if prior_state in {AttentionState.WATCH, AttentionState.HOT}
                else "watch_score"
            )
            return AttentionState.WATCH, reason
        return AttentionState.SCAN, "below_attention_boundary"

    @staticmethod
    def _plan(
        record: _AttentionRecord,
        timestamp_ms: int,
        rank: int | None,
    ) -> ObservationPlan:
        resolution, feeds = _OBSERVATION_REQUIREMENTS[record.state]
        return ObservationPlan(
            ticker=record.ticker,
            state=record.state,
            resolution=resolution,
            feeds=feeds,
            attention_score=record.attention_score,
            rank=rank,
            reason=record.reason,
            timestamp_ms=timestamp_ms,
        )

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable checkpoint of persistent runtime state."""

        return {
            "last_timestamp_ms": self._last_timestamp_ms,
            "records": [
                {
                    **asdict(record),
                    "state": record.state.value,
                }
                for _, record in sorted(self._records.items())
            ],
        }

    @classmethod
    def from_snapshot(
        cls,
        config: AttentionConfig,
        snapshot: Mapping[str, object],
    ) -> AttentionRuntime:
        """Restore a checkpoint while validating its public state values."""

        runtime = cls(config)
        last_timestamp = snapshot.get("last_timestamp_ms")
        if last_timestamp is not None and not isinstance(last_timestamp, int):
            raise ValueError("snapshot last_timestamp_ms must be an integer or null")
        runtime._last_timestamp_ms = last_timestamp

        records = snapshot.get("records", [])
        if not isinstance(records, list):
            raise TypeError("snapshot records must be a list")
        for payload in records:
            if not isinstance(payload, Mapping):
                raise TypeError("snapshot record must be an object")
            record = _AttentionRecord(
                ticker=str(payload["ticker"]),
                state=AttentionState(str(payload["state"])),
                attention_score=float(payload["attention_score"]),
                last_observed_ms=int(payload["last_observed_ms"]),
                missed_batches=int(payload.get("missed_batches", 0)),
                reason=str(payload.get("reason", "restored")),
            )
            if record.ticker in runtime._records:
                raise ValueError(f"duplicate snapshot ticker: {record.ticker}")
            runtime._records[record.ticker] = record
        return runtime
