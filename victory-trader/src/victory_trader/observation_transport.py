"""Transport state for a prediction-preserving high-resolution shortlist."""

from __future__ import annotations

from dataclasses import dataclass


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
