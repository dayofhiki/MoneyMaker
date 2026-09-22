"""Transport state for a prediction-preserving high-resolution shortlist."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TransportSnapshot:
    desired: tuple[str, ...]
    active: tuple[str, ...]
    additions: tuple[str, ...]
    removals: tuple[str, ...]
    missing: tuple[str, ...]
    stale: tuple[str, ...]
    connected: bool
    reconnect_required: bool
    cumulative_additions: int
    cumulative_removals: int
    reconnects: int


@dataclass(frozen=True)
class RankedCandidate:
    ticker: str
    score: float


class RankHysteresisSelector:
    """Stateful shortlist selector that can run at any event cadence."""

    def __init__(self, *, capacity: int = 20, incumbent_rank_limit: int = 40):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if incumbent_rank_limit < capacity:
            raise ValueError("incumbent rank limit must cover capacity")
        self.capacity = capacity
        self.incumbent_rank_limit = incumbent_rank_limit
        self._incumbents: set[str] = set()

    def reset(self) -> None:
        self._incumbents.clear()

    def select(self, candidates: Iterable[RankedCandidate]) -> tuple[str, ...]:
        ranked = sorted(candidates, key=lambda item: (-item.score, item.ticker))
        seen: set[str] = set()
        unique = [
            item
            for item in ranked
            if not (item.ticker in seen or seen.add(item.ticker))
        ]
        rank = {item.ticker: index + 1 for index, item in enumerate(unique)}
        retained = {
            ticker
            for ticker in self._incumbents
            if rank.get(ticker, self.incumbent_rank_limit + 1)
            <= self.incumbent_rank_limit
        }
        ordered_retained = [item for item in unique if item.ticker in retained]
        selected = ordered_retained[: self.capacity]
        selected_names = {item.ticker for item in selected}
        for item in unique:
            if len(selected) >= self.capacity:
                break
            if item.ticker not in selected_names:
                selected.append(item)
                selected_names.add(item.ticker)
        self._incumbents = selected_names
        return tuple(item.ticker for item in selected)


@dataclass(frozen=True)
class ObservationBridgeSnapshot:
    selected: tuple[str, ...]
    transport: TransportSnapshot


class ObservationBridge:
    """Connect event-driven rank hysteresis to bounded observation transport."""

    def __init__(
        self,
        *,
        capacity: int = 20,
        incumbent_rank_limit: int = 40,
        max_staleness_ms: int = 2_000,
    ):
        self.selector = RankHysteresisSelector(
            capacity=capacity, incumbent_rank_limit=incumbent_rank_limit
        )
        self.transport = ObservationTransport(
            capacity=capacity, max_staleness_ms=max_staleness_ms
        )

    def reset_session(self) -> None:
        self.selector.reset()
        self.transport = ObservationTransport(
            capacity=self.transport.capacity,
            max_staleness_ms=self.transport.max_staleness_ms,
        )

    def update(
        self, candidates: Iterable[RankedCandidate], *, now_ms: int
    ) -> ObservationBridgeSnapshot:
        selected = self.selector.select(candidates)
        snapshot = self.transport.reconcile(selected, now_ms=now_ms)
        return ObservationBridgeSnapshot(selected=selected, transport=snapshot)


class ObservationTransport:
    """Reconcile subscriptions without changing model-selected membership."""

    def __init__(self, *, capacity: int = 20, max_staleness_ms: int = 2_000):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_staleness_ms < 0:
            raise ValueError("max staleness must be non-negative")
        self.capacity = capacity
        self.max_staleness_ms = max_staleness_ms
        self._active: set[str] = set()
        self._latest_event_ms: dict[str, int] = {}
        self._connected = True
        self._cumulative_additions = 0
        self._cumulative_removals = 0
        self._reconnects = 0

    def disconnect(self) -> None:
        self._connected = False
        self._active.clear()
        self._latest_event_ms.clear()

    def reconnect(self) -> None:
        if not self._connected:
            self._reconnects += 1
        self._connected = True

    def record_bar(
        self, ticker: str, *, event_ms: int, received_ms: int
    ) -> bool:
        """Record an active subscription bar and reject future-dated data."""

        if event_ms > received_ms:
            raise ValueError("event timestamp cannot be after receipt timestamp")
        if not self._connected or ticker not in self._active:
            return False
        previous = self._latest_event_ms.get(ticker)
        if previous is None or event_ms >= previous:
            self._latest_event_ms[ticker] = event_ms
        return True

    def reconcile(
        self, desired: list[str] | tuple[str, ...] | set[str], *, now_ms: int
    ) -> TransportSnapshot:
        desired_set = {str(ticker) for ticker in desired}
        if len(desired_set) > self.capacity:
            raise ValueError(
                f"desired subscriptions {len(desired_set)} exceed capacity "
                f"{self.capacity}"
            )

        if not self._connected:
            return self._snapshot(
                desired_set,
                additions=set(),
                removals=set(),
                now_ms=now_ms,
                reconnect_required=bool(desired_set),
            )

        removals = self._active - desired_set
        additions = desired_set - self._active
        self._active = set(desired_set)
        for ticker in removals:
            self._latest_event_ms.pop(ticker, None)
        self._cumulative_additions += len(additions)
        self._cumulative_removals += len(removals)
        return self._snapshot(
            desired_set,
            additions=additions,
            removals=removals,
            now_ms=now_ms,
            reconnect_required=False,
        )

    def _snapshot(
        self,
        desired: set[str],
        *,
        additions: set[str],
        removals: set[str],
        now_ms: int,
        reconnect_required: bool,
    ) -> TransportSnapshot:
        missing = self._active - self._latest_event_ms.keys()
        stale = {
            ticker
            for ticker, event_ms in self._latest_event_ms.items()
            if ticker in self._active and now_ms - event_ms > self.max_staleness_ms
        }
        return TransportSnapshot(
            desired=tuple(sorted(desired)),
            active=tuple(sorted(self._active)),
            additions=tuple(sorted(additions)),
            removals=tuple(sorted(removals)),
            missing=tuple(sorted(missing)),
            stale=tuple(sorted(stale)),
            connected=self._connected,
            reconnect_required=reconnect_required,
            cumulative_additions=self._cumulative_additions,
            cumulative_removals=self._cumulative_removals,
            reconnects=self._reconnects,
        )
