from __future__ import annotations

import pandas as pd

from victory_trader.focus_cap_frontier import (
    ACTIVE_THRESHOLD,
    BASELINE_FOCUS_CAP,
    FOCUS_CAPS,
    choose_knee,
    select_dynamic_active,
    select_focus,
)


def test_focus_caps_are_frozen():
    assert BASELINE_FOCUS_CAP == 60
    assert FOCUS_CAPS == (20, 30, 45, 60, 75, 90, 120, 180, None)
    assert ACTIVE_THRESHOLD == 0.0040281217293971616


def test_select_focus_respects_cap():
    scored = pd.DataFrame(
        {
            "trading_day": ["2026-05-21"] * 4,
            "t": [1, 1, 1, 1],
            "ticker": ["A", "B", "C", "D"],
            "market_hazard_probability": [0.4, 0.3, 0.2, 0.1],
        }
    )
    focus = select_focus(scored, 2)
    assert focus["ticker"].tolist() == ["A", "B"]


def test_dynamic_active_uses_frozen_threshold():
    focus = pd.DataFrame(
        {
            "cross_within_3m_probability": [
                ACTIVE_THRESHOLD - 1e-6,
                ACTIVE_THRESHOLD,
                ACTIVE_THRESHOLD + 1e-6,
            ]
        }
    )
    active = select_dynamic_active(focus)
    assert len(active) == 2


def test_choose_knee_can_require_larger_focus_cap():
    rows = [
        {
            "focus_cap": 45,
            "metrics": {
                "focus_capture": {"supported_capture_rate": 0.970},
                "active_capture": {"supported_capture_rate": 0.920},
            },
        },
        {
            "focus_cap": 60,
            "metrics": {
                "focus_capture": {"supported_capture_rate": 0.978},
                "active_capture": {"supported_capture_rate": 0.940},
            },
        },
        {
            "focus_cap": 75,
            "metrics": {
                "focus_capture": {"supported_capture_rate": 0.989},
                "active_capture": {"supported_capture_rate": 0.951},
            },
        },
        {
            "focus_cap": 90,
            "metrics": {
                "focus_capture": {"supported_capture_rate": 0.991},
                "active_capture": {"supported_capture_rate": 0.953},
            },
        },
        {
            "focus_cap": None,
            "metrics": {
                "focus_capture": {"supported_capture_rate": 0.992},
                "active_capture": {"supported_capture_rate": 0.954},
            },
        },
    ]
    knee, diagnosis = choose_knee(rows)
    assert knee is not None
    assert knee["focus_cap"] == 90
    assert diagnosis == "focus_cap_increase_supported"
