from __future__ import annotations

import pandas as pd

from victory_trader.dynamic_active_threshold import (
    FIXED_TOPK,
    THRESHOLD_QUANTILES,
    _best_dynamic,
    _decision_counts,
)


def test_threshold_quantiles_are_frozen():
    assert THRESHOLD_QUANTILES == (0.25, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95)
    assert FIXED_TOPK == 20


def test_decision_counts_include_zero_active_minutes():
    focus = pd.DataFrame(
        {
            "trading_day": ["2026-05-21"] * 4,
            "t": [1, 1, 2, 2],
            "ticker": ["A", "B", "A", "B"],
        }
    )
    selected = focus.loc[focus["t"].eq(1) & focus["ticker"].eq("A")].copy()
    counts, decisions = _decision_counts(selected, focus)
    assert decisions == 2
    assert counts.tolist() == [1.0, 0.0]


def test_dynamic_can_dominate_fixed20():
    fixed = {
        "active_retention_given_focus": 0.94,
        "mean_active_count": 20.0,
    }
    rows = [
        {
            "fit_quantile": 0.75,
            "threshold": 0.1,
            "metrics": {
                "active_retention_given_focus": 0.945,
                "mean_active_count": 15.0,
            },
        }
    ]
    best, diagnosis = _best_dynamic(rows, fixed)
    assert best is not None
    assert diagnosis == "dynamic_threshold_dominates_fixed20"


def test_dynamic_target_has_priority():
    fixed = {
        "active_retention_given_focus": 0.94,
        "mean_active_count": 20.0,
    }
    rows = [
        {
            "fit_quantile": 0.65,
            "threshold": 0.05,
            "metrics": {
                "active_retention_given_focus": 0.955,
                "mean_active_count": 18.0,
            },
        }
    ]
    _, diagnosis = _best_dynamic(rows, fixed)
    assert diagnosis == "dynamic_threshold_reaches_target_with_lower_burden"
