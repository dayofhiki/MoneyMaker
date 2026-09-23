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



class BoundedTurnoverSelector:
    """Responsive shortlist with a hard per-decision replacement budget.

    Incumbents may persist while they remain inside the broad rank band, but a
    current top-capacity challenger can replace a weaker incumbent immediately.
    At most max_replacements incumbent slots are replaced after the initial
    session fill, bounding transport churn without allowing incumbency to block
    the strongest fresh candidates indefinitely.
    """

    def __init__(
        self,
        *,
        capacity: int = 20,
        incumbent_rank_limit: int = 40,
        max_replacements: int = 5,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if incumbent_rank_limit < capacity:
            raise ValueError("incumbent rank limit must cover capacity")
        if max_replacements <= 0 or max_replacements > capacity:
            raise ValueError("max replacements must be in [1, capacity]")
        self.capacity = capacity
        self.incumbent_rank_limit = incumbent_rank_limit
        self.max_replacements = max_replacements
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
        by_ticker = {item.ticker: item for item in unique}

        if not self._incumbents:
            selected = unique[: self.capacity]
            self._incumbents = {item.ticker for item in selected}
            return tuple(item.ticker for item in selected)

        retained_names = {
            ticker
            for ticker in self._incumbents
            if rank.get(ticker, self.incumbent_rank_limit + 1)
            <= self.incumbent_rank_limit
        }
        selected_names = set(retained_names)

        # Deeply fallen or disappeared incumbents create free slots and do not
        # consume the replacement budget.
        for item in unique:
            if len(selected_names) >= self.capacity:
                break
            if item.ticker not in selected_names:
                selected_names.add(item.ticker)

        current_core = unique[: self.capacity]
        current_core_names = {item.ticker for item in current_core}
        replacements = 0
        for challenger in current_core:
            if challenger.ticker in selected_names:
                continue
            if replacements >= self.max_replacements:
                break
            replaceable = [
                by_ticker[ticker]
                for ticker in selected_names
                if ticker in by_ticker and ticker not in current_core_names
            ]
            if not replaceable:
                break
            worst = max(
                replaceable,
                key=lambda item: (rank[item.ticker], item.ticker),
            )
            selected_names.remove(worst.ticker)
            selected_names.add(challenger.ticker)
            replacements += 1

        ordered = [
            item for item in unique if item.ticker in selected_names
        ][: self.capacity]
        self._incumbents = {item.ticker for item in ordered}
        return tuple(item.ticker for item in ordered)


class RateLimitedSubscriptionSelector:
    """Track a responsive desired shortlist with bounded subscription churn.

    The model's desired set is always the current top-capacity ranking. The
    active observation set moves toward that target by at most max_additions
    new tickers per update after the initial session fill. Old subscriptions
    are temporary transport state, not model preference.
    """

    def __init__(self, *, capacity: int = 20, max_additions: int = 5):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_additions <= 0 or max_additions > capacity:
            raise ValueError("max additions must be in [1, capacity]")
        self.capacity = capacity
        self.max_additions = max_additions
        self._active: set[str] = set()
        self._desired: tuple[str, ...] = ()

    @property
    def desired(self) -> tuple[str, ...]:
        return self._desired

    @property
    def active(self) -> tuple[str, ...]:
        """Return the actual subscribed membership, including unscoreable names."""
        return tuple(sorted(self._active))

    def reset(self) -> None:
        self._active.clear()
        self._desired = ()

    def select(self, candidates: Iterable[RankedCandidate]) -> tuple[str, ...]:
        ranked = sorted(candidates, key=lambda item: (-item.score, item.ticker))
        seen: set[str] = set()
        unique = [
            item
            for item in ranked
            if not (item.ticker in seen or seen.add(item.ticker))
        ]
        desired_items = unique[: self.capacity]
        desired = tuple(item.ticker for item in desired_items)
        desired_set = set(desired)
        self._desired = desired

        if not self._active:
            self._active = set(desired)
            return desired

        rank = {item.ticker: index + 1 for index, item in enumerate(unique)}
        challengers = [ticker for ticker in desired if ticker not in self._active]
        stale = sorted(
            (ticker for ticker in self._active if ticker not in desired_set),
            key=lambda ticker: (rank.get(ticker, len(unique) + 1), ticker),
            reverse=True,
        )
        changes = min(self.max_additions, len(challengers), len(stale))
        for ticker in stale[:changes]:
            self._active.remove(ticker)
        self._active.update(challengers[:changes])

        # If the active set starts below capacity because the prior universe was
        # smaller, fill only within the same per-update addition budget.
        remaining_budget = self.max_additions - changes
        if len(self._active) < self.capacity and remaining_budget > 0:
            for ticker in desired:
                if ticker in self._active:
                    continue
                self._active.add(ticker)
                remaining_budget -= 1
                if len(self._active) >= self.capacity or remaining_budget <= 0:
                    break

        ordered_current = [
            item.ticker for item in unique if item.ticker in self._active
        ]
        current_names = set(ordered_current)
        stale_tail = sorted(self._active - current_names)
        return tuple((ordered_current + stale_tail)[: self.capacity])

class RetentionAwareSubscriptionSelector:
    """Rate-limited desired-set tracking with value-aware stale eviction.

    Admission and eviction are intentionally separate. The admission ranking
    defines the desired top-capacity set. When more challengers exist than the
    transport can subscribe in one update, stale incumbents with the lowest
    causal retention value are released first. Missing retention scores are
    treated as lowest value so vanished incumbents do not block live slots.
    """

    def __init__(self, *, capacity: int = 20, max_additions: int = 5):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_additions <= 0 or max_additions > capacity:
            raise ValueError("max additions must be in [1, capacity]")
        self.capacity = capacity
        self.max_additions = max_additions
        self._active: set[str] = set()
        self._desired: tuple[str, ...] = ()

    @property
    def desired(self) -> tuple[str, ...]:
        return self._desired

    @property
    def active(self) -> tuple[str, ...]:
        return tuple(sorted(self._active))

    def reset(self) -> None:
        self._active.clear()
        self._desired = ()

    def select(
        self,
        candidates: Iterable[RankedCandidate],
        *,
        retention_scores: dict[str, float] | None = None,
    ) -> tuple[str, ...]:
        ranked = sorted(candidates, key=lambda item: (-item.score, item.ticker))
        seen: set[str] = set()
        unique = [
            item
            for item in ranked
            if not (item.ticker in seen or seen.add(item.ticker))
        ]
        desired_items = unique[: self.capacity]
        desired = tuple(item.ticker for item in desired_items)
        desired_set = set(desired)
        self._desired = desired

        if not self._active:
            self._active = set(desired)
            return desired

        scores = retention_scores or {}
        challengers = [ticker for ticker in desired if ticker not in self._active]
        stale = [ticker for ticker in self._active if ticker not in desired_set]
        stale.sort(
            key=lambda ticker: (
                float(scores.get(ticker, float("-inf"))),
                ticker,
            )
        )

        changes = min(self.max_additions, len(challengers), len(stale))
        for ticker in stale[:changes]:
            self._active.remove(ticker)
        self._active.update(challengers[:changes])

        remaining_budget = self.max_additions - changes
        if len(self._active) < self.capacity and remaining_budget > 0:
            for ticker in desired:
                if ticker in self._active:
                    continue
                self._active.add(ticker)
                remaining_budget -= 1
                if len(self._active) >= self.capacity or remaining_budget <= 0:
                    break

        ordered_current = [
            item.ticker for item in unique if item.ticker in self._active
        ]
        current_names = set(ordered_current)
        stale_tail = sorted(
            self._active - current_names,
            key=lambda ticker: (
                -float(scores.get(ticker, float("-inf"))),
                ticker,
            ),
        )
        return tuple((ordered_current + stale_tail)[: self.capacity])


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
