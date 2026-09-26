import numpy as np
import pandas as pd
import pytest

from victory_trader import second_execution_reconstruction as r244


def test_first_observed_open_never_looks_past_expiry():
    times = np.array([10_000, 12_000, 20_000], dtype=np.int64)
    opens = np.array([10.0, 11.0, 12.0])
    assert r244.first_observed_open(
        times, opens, decision_t=10_000, latency_ms=1_000, expiry_ms=5_000
    ) == (12_000, 11.0)
    assert r244.first_observed_open(
        times, opens, decision_t=13_000, latency_ms=1_000, expiry_ms=5_000
    ) == (None, None)


def test_observed_opens_rejects_duplicate_provider_second():
    with pytest.raises(ValueError, match="duplicate"):
        r244.observed_opens(
            {"results": [{"t": 1000, "o": 10}, {"t": 1000, "o": 11}]}
        )


def test_missing_policy_exit_uses_only_precommitted_deadline():
    episodes = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 1_000_000,
                "entered": True,
                "unresolved": True,
                "net_pct": np.nan,
                "stress_pct": np.nan,
                "entry_t": 1_060_000,
                "exit_t": np.nan,
                "wait_actions": 1,
                "hold_actions": 2,
            }
        ]
    )
    out = r244.execution_decisions(episodes)
    assert out.loc[0, "deadline_liquidation"]
    assert out.loc[0, "exit_decision_t"] == (
        1_000_000 + r244.EXIT_DEADLINE * r244.MINUTE_MS
    )


def test_reconstruction_does_not_choose_best_price_inside_window():
    decisions = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "hot_t": 0,
                "entered": True,
                "unresolved": True,
                "net_pct": np.nan,
                "stress_pct": np.nan,
                "entry_t": 10_000,
                "exit_t": 20_000,
                "wait_actions": 0,
                "hold_actions": 0,
                "entry_decision_t": 10_000,
                "exit_decision_t": 20_000,
                "deadline_liquidation": False,
                "policy_exit_missing": False,
            }
        ]
    )
    paths = {
        ("2026-05-11", "TEST"): (
            np.array([11_000, 12_000, 21_000, 22_000], dtype=np.int64),
            np.array([10.0, 1.0, 11.0, 100.0]),
        )
    }
    out = r244.reconstruct(decisions, paths, latency_ms=1_000)
    assert out.loc[0, "entry_fill_t"] == 11_000
    assert out.loc[0, "entry_reference"] == 10.0
    assert out.loc[0, "exit_fill_t"] == 21_000
    assert out.loc[0, "exit_reference"] == 11.0


def test_request244_contract_is_frozen():
    assert r244.REQUEST_ID == 244
    assert r244.EXPIRY_MS == 5_000
    assert r244.PRIMARY_LATENCY_MS == 1_000
    assert r244.LATENCIES_MS == (0, 1_000, 2_000, 5_000)
    assert r244.MIN_RESOLUTION_RATE == 0.80
