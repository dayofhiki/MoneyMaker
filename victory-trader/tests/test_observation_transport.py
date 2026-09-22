from __future__ import annotations

import pytest

from victory_trader.observation_transport import (
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
