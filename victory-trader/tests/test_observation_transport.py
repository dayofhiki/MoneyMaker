from __future__ import annotations

import pytest

from victory_trader.observation_transport import (
    BoundedTurnoverSelector,
    ObservationBridge,
    ObservationTransport,
    RankHysteresisSelector,
    RankedCandidate,
)


def test_reconcile_preserves_desired_membership_and_reports_churn():
    transport = ObservationTransport(capacity=3)

    first = transport.reconcile(["AAA", "BBB"], now_ms=1_000)
    second = transport.reconcile(["BBB", "CCC"], now_ms=2_000)

    assert first.active == ("AAA", "BBB")
    assert first.additions == ("AAA", "BBB")
    assert second.active == ("BBB", "CCC")
    assert second.additions == ("CCC",)
    assert second.removals == ("AAA",)
    assert second.cumulative_additions == 3
    assert second.cumulative_removals == 1


def test_missing_and_stale_bar_telemetry_is_causal():
    transport = ObservationTransport(capacity=2, max_staleness_ms=2_000)
    transport.reconcile(["AAA", "BBB"], now_ms=1_000)

    assert transport.record_bar("AAA", event_ms=1_500, received_ms=1_600)
    snapshot = transport.reconcile(["AAA", "BBB"], now_ms=4_000)

    assert snapshot.missing == ("BBB",)
    assert snapshot.stale == ("AAA",)
    with pytest.raises(ValueError, match="after receipt"):
        transport.record_bar("AAA", event_ms=5_001, received_ms=5_000)


def test_disconnect_requires_reconnect_without_mutating_desired_set():
    transport = ObservationTransport(capacity=2)
    transport.reconcile(["AAA", "BBB"], now_ms=1_000)
    transport.disconnect()

    disconnected = transport.reconcile(["AAA", "BBB"], now_ms=2_000)
    assert disconnected.active == ()
    assert disconnected.desired == ("AAA", "BBB")
    assert disconnected.reconnect_required is True

    transport.reconnect()
    restored = transport.reconcile(["AAA", "BBB"], now_ms=3_000)
    assert restored.active == restored.desired
    assert restored.reconnects == 1


def test_transport_rejects_over_capacity_and_ignores_inactive_bars():
    transport = ObservationTransport(capacity=1)

    with pytest.raises(ValueError, match="exceed capacity"):
        transport.reconcile(["AAA", "BBB"], now_ms=1_000)
    assert not transport.record_bar("AAA", event_ms=500, received_ms=600)


def test_rank_hysteresis_selector_retains_incumbents_at_arbitrary_cadence():
    selector = RankHysteresisSelector(capacity=2, incumbent_rank_limit=4)
    first = selector.select(
        [RankedCandidate(ticker, score) for ticker, score in [
            ("AAA", 4.0), ("BBB", 3.0), ("CCC", 2.0), ("DDD", 1.0)
        ]]
    )
    second = selector.select(
        [RankedCandidate(ticker, score) for ticker, score in [
            ("CCC", 4.0), ("DDD", 3.0), ("AAA", 2.0), ("BBB", 1.0)
        ]]
    )

    assert first == ("AAA", "BBB")
    assert second == ("AAA", "BBB")


def test_observation_bridge_preserves_selection_during_fast_updates():
    bridge = ObservationBridge(capacity=2, incumbent_rank_limit=3)
    candidates = [
        RankedCandidate("AAA", 0.9),
        RankedCandidate("BBB", 0.8),
        RankedCandidate("CCC", 0.7),
    ]

    first = bridge.update(candidates, now_ms=1_000)
    second = bridge.update(list(reversed(candidates)), now_ms=1_025)

    assert first.selected == ("AAA", "BBB")
    assert second.selected == ("AAA", "BBB")
    assert second.transport.additions == ()
    assert second.transport.active == second.selected


def test_observation_bridge_disconnect_does_not_change_model_selection():
    bridge = ObservationBridge(capacity=1, incumbent_rank_limit=2)
    candidates = [RankedCandidate("AAA", 1.0), RankedCandidate("BBB", 0.5)]
    bridge.update(candidates, now_ms=1_000)
    bridge.transport.disconnect()

    snapshot = bridge.update(candidates, now_ms=1_100)

    assert snapshot.selected == ("AAA",)
    assert snapshot.transport.active == ()
    assert snapshot.transport.reconnect_required is True



def test_bounded_turnover_selector_admits_strong_challengers():
    selector = BoundedTurnoverSelector(
        capacity=4, incumbent_rank_limit=8, max_replacements=1
    )
    first = selector.select(
        [
            RankedCandidate("A", 8.0),
            RankedCandidate("B", 7.0),
            RankedCandidate("C", 6.0),
            RankedCandidate("D", 5.0),
            RankedCandidate("E", 4.0),
            RankedCandidate("F", 3.0),
            RankedCandidate("G", 2.0),
            RankedCandidate("H", 1.0),
        ]
    )
    second = selector.select(
        [
            RankedCandidate("E", 8.0),
            RankedCandidate("F", 7.0),
            RankedCandidate("G", 6.0),
            RankedCandidate("H", 5.0),
            RankedCandidate("A", 4.0),
            RankedCandidate("B", 3.0),
            RankedCandidate("C", 2.0),
            RankedCandidate("D", 1.0),
        ]
    )

    assert first == ("A", "B", "C", "D")
    assert "E" in second
    assert len(set(second) - set(first)) == 1


def test_bounded_turnover_selector_never_replaces_more_than_budget():
    selector = BoundedTurnoverSelector(
        capacity=6, incumbent_rank_limit=12, max_replacements=2
    )
    first = selector.select(
        [RankedCandidate(f"I{i}", float(12 - i)) for i in range(12)]
    )
    second = selector.select(
        [
            *[
                RankedCandidate(f"C{i}", float(20 - i))
                for i in range(6)
            ],
            *[
                RankedCandidate(f"I{i}", float(14 - i))
                for i in range(12)
            ],
        ]
    )

    assert len(first) == len(second) == 6
    assert len(set(second) - set(first)) == 2
