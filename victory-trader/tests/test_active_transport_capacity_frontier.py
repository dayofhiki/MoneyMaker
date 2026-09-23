from __future__ import annotations

from victory_trader.active_transport_capacity_frontier import (
    CAPACITY_FRONTIER,
    _diagnose_frontier,
    _minimal_cap_meeting_target,
)


def test_request152_capacity_frontier_is_frozen():
    assert CAPACITY_FRONTIER == (5, 8, 10, 12, 16, 20)


def test_diagnosis_calls_selection_bottleneck_when_full_refresh_misses_target():
    assert (
        _diagnose_frontier(0.934, 0.947)
        == "selection_or_admission_bottleneck"
    )


def test_diagnosis_calls_transport_bottleneck_for_material_full_refresh_gain():
    assert (
        _diagnose_frontier(0.934, 0.956)
        == "residual_transport_capacity_bottleneck"
    )


def test_minimal_cap_meeting_target_returns_first_passing_cap():
    frontier = {
        str(cap): {
            "active_retention_given_focus": value,
        }
        for cap, value in zip(
            CAPACITY_FRONTIER,
            [0.92, 0.94, 0.948, 0.951, 0.953, 0.954],
            strict=True,
        )
    }

    assert _minimal_cap_meeting_target(frontier) == 12
