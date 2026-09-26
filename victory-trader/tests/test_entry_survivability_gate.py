from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.entry_survivability_gate import (
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
