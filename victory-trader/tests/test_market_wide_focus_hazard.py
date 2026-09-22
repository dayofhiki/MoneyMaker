from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from victory_trader.market_wide_focus_hazard import (
    BASELINE_FEATURES,
    FOCUS_BUDGET,
    SHORTLIST_BUDGET,
    build_market_hazard_rows,
    fit_market_hazard,
    select_learned_focus,
)


class _ScoreEchoModel:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        score = pd.to_numeric(features["attention_score"], errors="coerce").fillna(0)
        return np.column_stack([1.0 - score, score])


def test_market_hazard_target_uses_exact_next_completed_minute():
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-02-25",
                "ticker": "AAA",
                "t": 60_000,
                "o": 1.0,
                "h": 1.05,
                "l": 0.99,
                "c": 1.04,
                "v": 100,
                "n": 10,
                "attention_score": 0.9,
                "return_from_previous_close_pct": 4.0,
                "runner_cross_now": False,
            },
            {
                "trading_day": "2026-02-25",
                "ticker": "AAA",
                "t": 120_000,
                "o": 1.04,
                "h": 1.12,
                "l": 1.03,
                "c": 1.11,
                "v": 300,
                "n": 30,
                "attention_score": 0.99,
                "return_from_previous_close_pct": 11.0,
                "runner_cross_now": True,
            },
        ]
    )

    rows = build_market_hazard_rows(scan)

    assert len(rows) == 1
    assert int(rows.iloc[0]["t"]) == 60_000
    assert int(rows.iloc[0]["target_next_cross"]) == 1


def test_market_hazard_excludes_nonconsecutive_target_and_current_runner():
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-02-25",
                "ticker": "AAA",
                "t": 60_000,
                "o": 1.0,
                "h": 1.01,
                "l": 0.99,
                "c": 1.0,
                "v": 100,
                "n": 10,
                "attention_score": 0.5,
                "return_from_previous_close_pct": 4.0,
                "runner_cross_now": False,
            },
            {
                "trading_day": "2026-02-25",
                "ticker": "AAA",
                "t": 180_000,
                "o": 1.0,
                "h": 1.12,
                "l": 1.0,
                "c": 1.11,
                "v": 300,
                "n": 30,
                "attention_score": 0.9,
                "return_from_previous_close_pct": 11.0,
                "runner_cross_now": True,
            },
        ]
    )

    assert build_market_hazard_rows(scan).empty


def test_market_hazard_fit_does_not_use_rows_after_frozen_cutoff():
    rows = pd.DataFrame(
        [
            {
                "trading_day": "2026-01-16",
                "target_next_cross": 0,
                **{feature: 0.0 for feature in BASELINE_FEATURES},
            },
            {
                "trading_day": "2026-02-25",
                "target_next_cross": 1,
                **{feature: 1.0 for feature in BASELINE_FEATURES},
            },
        ]
    )

    with pytest.raises(ValueError, match="no positives"):
        fit_market_hazard(rows)


def test_learned_focus_and_shortlist_respect_fixed_budgets():
    rows = pd.DataFrame(
        [
            {
                "trading_day": "2026-02-25",
                "t": 60_000,
                "ticker": f"T{i:03d}",
                "attention_score": i / 100,
                "attention_rank": 100 - i,
                "return_from_previous_close_pct": float(i),
                "minute_body_return_pct": 0.0,
                "minute_range_pct": 0.0,
                "log_minute_volume": 1.0,
                "log_minute_transactions": 1.0,
                "minute_return_1m_pct": 0.0,
                "return_accel_1m_pct": 0.0,
                "volume_ratio_prev1": 1.0,
                "transactions_ratio_prev1": 1.0,
                "target_next_cross": 0,
            }
            for i in range(100)
        ]
    )

    _, focus, shortlist = select_learned_focus(rows, _ScoreEchoModel())

    assert len(focus) == FOCUS_BUDGET == 60
    assert len(shortlist) == SHORTLIST_BUDGET == 20
    assert shortlist["ticker"].tolist()[0] == "T099"
    assert shortlist["ticker"].tolist()[-1] == "T080"
