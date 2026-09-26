from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.adaptive_pullback_redecision import (
    build_pullback_wait_states,
)


def _episode(phases, exits, held=None):
    if held is None:
        held = list(range(1, len(phases) + 1))
    rows = []
    for i, (phase, value, minute) in enumerate(
        zip(phases, exits, held, strict=True)
    ):
        rows.append(
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 1,
                "state_t": (i + 1) * 60_000,
                "minutes_held": float(minute),
                "transition_phase": phase,
                "exit_now_base_return_pct": float(value),
            }
        )
    return pd.DataFrame(rows)


def test_segment_ends_at_first_reacceleration() -> None:
    frame = _episode(
        [
            "continuing_strength",
            "pullback_onset",
            "continuing_weakness",
            "continuing_weakness",
            "reacceleration",
            "continuing_strength",
        ],
        [1.0, 0.8, 0.4, 0.2, 1.3, 1.8],
    )
    states, support = build_pullback_wait_states(frame)
    assert support["pullback_segments_total"] == 1
    assert support["pullback_segments_resolved"] == 1
    assert states["pullback_age_events"].tolist() == [0, 1, 2]
    assert states["pullback_terminal_reason"].eq(
        "reacceleration_redecision"
    ).all()
    assert np.allclose(
        states["adaptive_wait_value_pct"].to_numpy(dtype=float),
        [0.5, 0.9, 1.1],
    )


def test_context_tracks_depth_and_recovery_causally() -> None:
    frame = _episode(
        [
            "continuing_strength",
            "pullback_onset",
            "continuing_weakness",
            "flat_or_sparse",
            "reacceleration",
        ],
        [1.0, 0.8, 0.2, 0.5, 1.1],
    )
    states, _ = build_pullback_wait_states(frame)
    assert np.allclose(
        states["pullback_return_from_onset_pct"].to_numpy(dtype=float),
        [0.0, -0.6, -0.3],
    )
    assert np.allclose(
        states["pullback_worst_from_onset_pct"].to_numpy(dtype=float),
        [0.0, -0.6, -0.6],
    )
    assert np.allclose(
        states["pullback_recovery_from_worst_pct"].to_numpy(dtype=float),
        [0.0, 0.0, 0.3],
    )


def test_future_after_reacceleration_cannot_change_wait_labels() -> None:
    frame = _episode(
        [
            "continuing_strength",
            "pullback_onset",
            "continuing_weakness",
            "reacceleration",
            "continuing_strength",
        ],
        [1.0, 0.8, 0.4, 1.1, 1.5],
    )
    base, _ = build_pullback_wait_states(frame)
    changed = frame.copy()
    changed.loc[4, "exit_now_base_return_pct"] = 99.0
    changed, _ = build_pullback_wait_states(changed)
    assert np.allclose(
        base["adaptive_wait_value_pct"].to_numpy(dtype=float),
        changed["adaptive_wait_value_pct"].to_numpy(dtype=float),
    )


def test_unresolved_pullback_is_not_converted_to_cash() -> None:
    frame = _episode(
        [
            "continuing_strength",
            "pullback_onset",
            "continuing_weakness",
            "continuing_weakness",
        ],
        [1.0, 0.8, 0.4, 0.1],
    )
    states, support = build_pullback_wait_states(frame)
    assert support["pullback_segments_total"] == 1
    assert support["pullback_segments_resolved"] == 0
    assert states.empty


def test_thirty_minute_cap_is_terminal_when_observed() -> None:
    frame = _episode(
        [
            "continuing_strength",
            "pullback_onset",
            "continuing_weakness",
            "continuing_weakness",
        ],
        [1.0, 0.8, 0.4, -0.2],
        held=[27, 28, 29, 30],
    )
    states, _ = build_pullback_wait_states(frame)
    assert states["pullback_terminal_reason"].eq(
        "forced_30m_cap"
    ).all()
    assert np.allclose(
        states["adaptive_wait_value_pct"].to_numpy(dtype=float),
        [-1.0, -0.6],
    )
