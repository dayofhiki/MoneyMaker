from __future__ import annotations

import pandas as pd

from victory_trader.focus180_temporal_retrain import (
    BASELINE_THRESHOLD,
    FOCUS_CAP,
    MATERIAL_GAIN,
    MAX_BURDEN_RATIO,
    TARGET_CAPTURE,
    _diagnose,
    select_focus_rows,
)


def test_request158_constants_are_frozen():
    assert FOCUS_CAP == 180
    assert BASELINE_THRESHOLD == 0.0040281217293971616
    assert TARGET_CAPTURE == 0.95
    assert MATERIAL_GAIN == 0.005
    assert MAX_BURDEN_RATIO == 1.10


def test_select_focus_rows_respects_cap_and_rank(monkeypatch):
    class DummyModel:
        def predict_proba(self, x):
            import numpy as np
            p = np.linspace(0.9, 0.1, len(x))
            return np.c_[1.0 - p, p]

    rows = pd.DataFrame(
        {
            "trading_day": ["2026-01-02"] * 4,
            "ticker": ["A", "B", "C", "D"],
            "t": [1, 1, 1, 1],
            "attention_score": [1.0, 0.9, 0.8, 0.7],
            "attention_rank": [1, 2, 3, 4],
            "return_from_previous_close_pct": [0.0] * 4,
            "minute_body_return_pct": [0.0] * 4,
            "minute_range_pct": [0.0] * 4,
            "log_minute_volume": [1.0] * 4,
            "log_minute_transactions": [1.0] * 4,
            "minute_return_1m_pct": [0.0] * 4,
            "return_accel_1m_pct": [0.0] * 4,
            "volume_ratio_prev1": [1.0] * 4,
            "transactions_ratio_prev1": [1.0] * 4,
            "target_next_cross": [0] * 4,
            "target_cross_within_3m": [0.0] * 4,
        }
    )
    focus = select_focus_rows(rows, DummyModel(), 2)
    assert focus["ticker"].tolist() == ["A", "B"]
    assert focus["market_rank"].tolist() == [1, 2]


def test_diagnose_candidate():
    baseline = {
        "active_capture": {"supported_capture_rate": 0.943},
        "active_count": {"mean": 22.6},
    }
    broad = {
        "active_capture": {"supported_capture_rate": 0.951},
        "active_count": {"mean": 23.0},
    }
    result = _diagnose(baseline, broad)
    assert result["diagnosis"] == "focus180_training_candidate"


def test_diagnose_no_material_gain():
    baseline = {
        "active_capture": {"supported_capture_rate": 0.943},
        "active_count": {"mean": 22.6},
    }
    broad = {
        "active_capture": {"supported_capture_rate": 0.946},
        "active_count": {"mean": 22.0},
    }
    result = _diagnose(baseline, broad)
    assert result["diagnosis"] == "focus180_training_no_material_gain"
