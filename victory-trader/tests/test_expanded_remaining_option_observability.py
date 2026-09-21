from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.expanded_remaining_option_observability import (
    MINUTE_MS,
    add_remaining_option_labels,
    apply_excess_target,
    fit_minute_baselines,
    oracle_entry_ceiling,
)


def _state(opens: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "ticker": "ABC",
                "t": minute * MINUTE_MS,
                "o": price,
            }
            for minute, price in sorted(opens.items())
        ]
    )


def test_oracle_entry_ceiling_uses_best_observed_open_without_compressing_gaps():
    anchors = pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "ticker": "ABC",
                "t": 0,
                "opportunity_probability": 0.9,
            }
        ]
    )
    state = _state({1: 10.0, 2: 10.2, 4: 11.0, 31: 10.5})
    oracle = oracle_entry_ceiling(anchors, state, "2026-01")
    row = oracle.iloc[0]
    assert row["status"] == "evaluated"
    assert row["observed_exit_count"] == 3
    assert row["exit_slot_count"] == 30
    assert row["oracle_best_holding_min"] == 3
    assert row["oracle_best_base_pct"] > 0


def test_remaining_option_label_counts_literal_future_slots():
    rows = pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "ticker": "ABC",
                "entry_open": 10.0,
                "exit_open": 10.1,
                "entry_reference_t": MINUTE_MS,
                "exit_reference_t": 2 * MINUTE_MS,
                "minutes_held": 1.0,
            }
        ]
    )
    # Minute 3 is absent. Minutes 4 and 5 remain literal future observations;
    # the gap is not compressed into a row-count lag.
    state = _state({1: 10.0, 2: 10.1, 4: 10.8, 5: 10.4})
    labeled = add_remaining_option_labels(
        rows,
        {"2026-01": state},
    )
    row = labeled.iloc[0]
    assert row["future_slot_count"] == 29
    assert row["future_observed_opens"] == 2
    assert np.isclose(row["future_observed_fraction"], 2 / 29)
    assert row["best_future_base_return_pct"] > row["exit_now_base_return_pct"]
    assert row["remaining_option_value_pct"] > 0


def test_remaining_option_is_missing_when_no_later_open_exists():
    rows = pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "ticker": "ABC",
                "entry_open": 10.0,
                "exit_open": 10.1,
                "entry_reference_t": MINUTE_MS,
                "exit_reference_t": 2 * MINUTE_MS,
                "minutes_held": 1.0,
            }
        ]
    )
    state = _state({1: 10.0, 2: 10.1})
    labeled = add_remaining_option_labels(rows, {"2026-01": state})
    row = labeled.iloc[0]
    assert row["future_observed_opens"] == 0
    assert np.isnan(row["best_future_base_return_pct"])
    assert np.isnan(row["remaining_option_value_pct"])


def test_fit_minute_baseline_is_reused_without_evaluation_outcomes():
    fit = pd.DataFrame(
        {
            "minutes_held": [1, 1, 2, 2],
            "remaining_option_value_pct": [1.0, 3.0, 4.0, 8.0],
        }
    )
    baselines = fit_minute_baselines(fit)
    assert baselines == {1: 2.0, 2: 6.0}

    evaluation = pd.DataFrame(
        {
            "minutes_held": [1, 2],
            "remaining_option_value_pct": [100.0, -100.0],
        }
    )
    transformed = apply_excess_target(evaluation, baselines)
    assert transformed["fit_minute_baseline_pct"].tolist() == [2.0, 6.0]
    assert transformed["excess_remaining_option_value_pct"].tolist() == [
        98.0,
        -106.0,
    ]


def test_gate_below_half_is_not_in_oracle_population():
    anchors = pd.DataFrame(
        [
            {
                "trading_day": "2026-01-02",
                "ticker": "ABC",
                "t": 0,
                "opportunity_probability": 0.49,
            }
        ]
    )
    oracle = oracle_entry_ceiling(
        anchors,
        _state({1: 10.0, 2: 20.0}),
        "2026-01",
    )
    assert oracle.empty
