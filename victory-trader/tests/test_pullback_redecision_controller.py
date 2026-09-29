from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.pullback_redecision_controller import (
    MAX_WAIT_EVENTS,
    attach_pullback_wait_target,
)


def _episode(phases, exits):
    rows = []
    for i, (phase, value) in enumerate(zip(phases, exits, strict=True)):
        rows.append(
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 1,
                "state_t": (i + 1) * 60_000,
                "minutes_held": float(i + 1),
                "transition_phase": phase,
                "exit_now_base_return_pct": float(value),
            }
        )
    return pd.DataFrame(rows)


def test_wait_resolves_at_first_reacceleration() -> None:
    frame = _episode(
        [
            "continuing_strength",
            "pullback_onset",
            "continuing_weakness",
            "reacceleration",
            "continuing_strength",
        ],
        [1.0, 0.8, 0.5, 1.4, 1.8],
    )
    enriched = attach_pullback_wait_target(frame)
    row = enriched.iloc[1]
    assert row["pullback_resolution"] == "reacceleration_redecision"
    assert row["pullback_resolution_events"] == 2
    assert np.isclose(row["pullback_wait_advantage_pct"], 0.6)


def test_wait_times_out_at_third_event_without_turn() -> None:
    frame = _episode(
        [
            "continuing_strength",
            "pullback_onset",
            "continuing_weakness",
            "continuing_weakness",
            "continuing_weakness",
        ],
        [1.0, 0.8, 0.5, 0.2, -0.1],
    )
    enriched = attach_pullback_wait_target(frame)
    row = enriched.iloc[1]
    assert MAX_WAIT_EVENTS == 3
    assert row["pullback_resolution"] == "three_event_timeout"
    assert row["pullback_resolution_events"] == 3
    assert np.isclose(row["pullback_wait_advantage_pct"], -0.9)


def test_incomplete_timeout_horizon_is_unresolved() -> None:
    frame = _episode(
        ["continuing_strength", "pullback_onset", "continuing_weakness"],
        [1.0, 0.8, 0.4],
    )
    enriched = attach_pullback_wait_target(frame)
    assert pd.isna(enriched.iloc[1]["pullback_wait_advantage_pct"])


def test_future_after_resolution_does_not_change_target() -> None:
    frame = _episode(
        [
            "continuing_strength",
            "pullback_onset",
            "reacceleration",
            "continuing_strength",
        ],
        [1.0, 0.8, 1.1, 2.0],
    )
    base = attach_pullback_wait_target(frame)
    changed = frame.copy()
    changed.loc[3, "exit_now_base_return_pct"] = 99.0
    changed = attach_pullback_wait_target(changed)
    assert np.isclose(
        base.iloc[1]["pullback_wait_advantage_pct"],
        changed.iloc[1]["pullback_wait_advantage_pct"],
    )


def test_thirty_minute_cap_censors_resolution() -> None:
    frame = _episode(
        [
            "continuing_strength",
            "pullback_onset",
            "continuing_weakness",
            "reacceleration",
        ],
        [1.0, 0.8, 0.5, 1.4],
    )
    frame.loc[3, "minutes_held"] = 31.0
    enriched = attach_pullback_wait_target(frame)
    assert pd.isna(enriched.iloc[1]["pullback_wait_advantage_pct"])
