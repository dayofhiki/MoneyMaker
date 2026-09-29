from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.entry_survivability_gate import (
    attach_first_stress_labels,
    gate_policy,
    selected_entry_states,
)


def test_selected_entry_states_matches_request208_minute() -> None:
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 1,
                "minutes_since_hot": 1,
                "x": 10.0,
            },
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 1,
                "minutes_since_hot": 2,
                "x": 20.0,
            },
        ]
    )
    entries = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 1,
                "entry_minute_after_hot": 2,
                "realized_base_return_pct": 1.0,
            }
        ]
    )
    selected = selected_entry_states(states, entries)
    assert len(selected) == 1
    assert selected.iloc[0]["minutes_since_hot"] == 2
    assert np.isclose(selected.iloc[0]["x"], 20.0)


def test_gate_abstains_when_survivability_prediction_nonpositive() -> None:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 1,
                "entry_minute_after_hot": 2,
                "realized_base_return_pct": -1.5,
                "predicted_first_stress_return_pct": -0.2,
                "first_stress_return_pct": -0.8,
            },
            {
                "trading_day": "2026-05-01",
                "ticker": "BBB",
                "hot_t": 2,
                "entry_minute_after_hot": 1,
                "realized_base_return_pct": 2.0,
                "predicted_first_stress_return_pct": 0.4,
                "first_stress_return_pct": 0.9,
            },
        ]
    )
    gated = gate_policy(frame).set_index("ticker")
    assert not bool(gated.loc["AAA", "admitted"])
    assert np.isclose(gated.loc["AAA", "candidate_return_pct"], 0.0)
    assert bool(gated.loc["BBB", "admitted"])
    assert np.isclose(gated.loc["BBB", "candidate_return_pct"], 2.0)


def test_zero_prediction_is_abstain() -> None:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 1,
                "entry_minute_after_hot": 1,
                "realized_base_return_pct": 1.0,
                "predicted_first_stress_return_pct": 0.0,
                "first_stress_return_pct": 0.5,
            }
        ]
    )
    gated = gate_policy(frame)
    assert not bool(gated.iloc[0]["admitted"])
    assert np.isclose(gated.iloc[0]["candidate_return_pct"], 0.0)


def test_first_stress_is_first_adverse_observed_step() -> None:
    entry_states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "entry_minute_after_hot": 1,
                "realized_base_return_pct": 0.0,
            }
        ]
    )
    entries = entry_states.loc[
        :,
        [
            "trading_day",
            "ticker",
            "hot_t",
            "entry_minute_after_hot",
            "realized_base_return_pct",
        ],
    ].copy()
    positions = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "minutes_held": 1.0,
                "state_t": 60_000,
                "exit_reference_open": 100.0,
                "log_current_close": np.log(100.0),
            },
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "minutes_held": 2.0,
                "state_t": 120_000,
                "exit_reference_open": 102.0,
                "log_current_close": np.log(102.0),
            },
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "minutes_held": 3.0,
                "state_t": 180_000,
                "exit_reference_open": 101.0,
                "log_current_close": np.log(101.0),
            },
            {
                "trading_day": "2026-05-01",
                "ticker": "AAA",
                "hot_t": 0,
                "minutes_held": 4.0,
                "state_t": 240_000,
                "exit_reference_open": 105.0,
                "log_current_close": np.log(105.0),
            },
        ]
    )
    labeled = attach_first_stress_labels(
        entry_states, positions, entries
    )
    row = labeled.iloc[0]
    assert row["first_stress_reason"] == "first_adverse_observation"
    assert np.isclose(row["first_stress_minutes"], 2.0)
    assert row["first_stress_return_pct"] < row["pre_stress_peak_return_pct"]
    assert row["first_stress_giveback_pct"] < 0
