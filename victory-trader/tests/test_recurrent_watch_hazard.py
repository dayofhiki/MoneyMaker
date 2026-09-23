from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.recurrent_watch_hazard import (
    EVAL_DAYS,
    MODEL_FEATURES,
    _state_history,
    fit_and_evaluate,
)


def test_state_history_counts_focus_and_watch_runs_causally():
    trace = pd.DataFrame(
        [
            {"trading_day": "2026-01-27", "ticker": "AAA", "t": 60_000, "state": "scan", "attention_score": 0.1, "attention_rank": 100},
            {"trading_day": "2026-01-27", "ticker": "AAA", "t": 120_000, "state": "watch", "attention_score": 0.7, "attention_rank": 40},
            {"trading_day": "2026-01-27", "ticker": "AAA", "t": 180_000, "state": "watch", "attention_score": 0.75, "attention_rank": 30},
            {"trading_day": "2026-01-27", "ticker": "AAA", "t": 240_000, "state": "hot", "attention_score": 0.9, "attention_rank": 5},
            {"trading_day": "2026-01-27", "ticker": "AAA", "t": 300_000, "state": "watch", "attention_score": 0.8, "attention_rank": 20},
            {"trading_day": "2026-01-27", "ticker": "AAA", "t": 360_000, "state": "scan", "attention_score": 0.2, "attention_rank": 90},
        ]
    )

    enriched = _state_history(trace).set_index("t")

    assert enriched.loc[120_000, "focus_age_minutes"] == 1.0
    assert enriched.loc[180_000, "focus_age_minutes"] == 2.0
    assert enriched.loc[240_000, "focus_age_minutes"] == 3.0
    assert enriched.loc[300_000, "focus_age_minutes"] == 4.0
    assert enriched.loc[120_000, "watch_run_age_minutes"] == 1.0
    assert enriched.loc[180_000, "watch_run_age_minutes"] == 2.0
    assert enriched.loc[300_000, "watch_run_age_minutes"] == 1.0


def test_fit_and_evaluate_uses_frozen_hazard_eval_days():
    rng = np.random.default_rng(41)
    fit_days = [
        "2026-01-20",
        "2026-01-21",
        "2026-01-22",
        "2026-01-23",
        "2026-01-26",
    ]
    days = fit_days + EVAL_DAYS
    rows = []

    for day_index, day in enumerate(days):
        for minute in range(150):
            for ticker_index in range(10):
                signal = (ticker_index + minute % 5) / 15.0
                target = int((minute + ticker_index + day_index) % 47 == 0)
                row = {
                    "trading_day": day,
                    "t": minute * 60_000,
                    "ticker": f"T{ticker_index:02d}",
                    "target_next_cross": target,
                    "attention_score": signal,
                }
                for feature in MODEL_FEATURES:
                    row[feature] = signal + rng.normal(0, 0.01)
                rows.append(row)

    modeled, summary = fit_and_evaluate(pd.DataFrame(rows))

    assert summary["eval_days"] == EVAL_DAYS
    assert summary["fit_days"] == fit_days
    assert summary["fit_rows"] == len(fit_days) * 1500
    assert summary["eval_rows"] == len(EVAL_DAYS) * 1500
    assert set(
        modeled.loc[modeled["split"].eq("eval"), "trading_day"]
    ) == set(EVAL_DAYS)
    assert modeled["hazard_probability"].between(0.0, 1.0).all()


def test_state_history_resets_same_ticker_on_new_session():
    trace = pd.DataFrame(
        [
            {
                "trading_day": "2026-01-27",
                "ticker": "AAA",
                "t": 60_000,
                "state": "watch",
            },
            {
                "trading_day": "2026-01-27",
                "ticker": "AAA",
                "t": 120_000,
                "state": "watch",
            },
            {
                "trading_day": "2026-01-28",
                "ticker": "AAA",
                "t": 60_000,
                "state": "watch",
            },
        ]
    )

    enriched = _state_history(trace).sort_values(
        ["trading_day", "t"]
    ).reset_index(drop=True)

    assert enriched.loc[0, "watch_run_age_minutes"] == 1.0
    assert enriched.loc[1, "watch_run_age_minutes"] == 2.0
    assert enriched.loc[2, "watch_run_age_minutes"] == 1.0
    assert enriched.loc[2, "focus_age_minutes"] == 1.0
