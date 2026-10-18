"""Check the support bound against the unchanged economic contract on fixtures."""
import math

import numpy as np
import pandas as pd
import pytest

from victory_trader import current_state_action_policy as frozen
from victory_trader.r335_target_support import build_scope, target_end_ns

HOT = 1_000_000
CAP = HOT + 3_600_000
CLOSE = CAP + 1_000_000


def bars(times, prices=None):
    if prices is None:
        prices = np.full(len(times), 10.)
    return pd.DataFrame({"t": np.asarray(times, dtype=np.int64), "o": prices, "c": prices})


def equal_results(first, second):
    for key, value in first.items():
        actual = second[key]
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(actual), key
        else:
            assert actual == value, key


@pytest.mark.parametrize("exit_offset", [0, 1000, 3000, 4000, 100_000])
def test_terminal_and_lag_boundary_preserves_every_result_field(exit_offset):
    data = bars([CAP - 1000, CAP + exit_offset, CAP + 200_000])
    end = target_end_ns(HOT, CLOSE * 1_000_000) // 1_000_000
    clipped = data.loc[data.t < end]
    full = frozen.enter_outcome(data, CAP - 1000, HOT, CLOSE)
    bounded = frozen.enter_outcome(clipped, CAP - 1000, HOT, CLOSE)
    equal_results(full, bounded)
    assert full["enter_resolved"] == (exit_offset <= 3000)


def test_cutting_at_terminal_or_at_last_accepted_start_would_lose_a_valid_exit():
    data = bars([CAP - 1000, CAP + 3000])
    assert frozen.enter_outcome(data, CAP - 1000, HOT, CLOSE)["enter_resolved"]
    for incorrect_end in (CAP, CAP + 3000):
        assert not frozen.enter_outcome(data.loc[data.t < incorrect_end], CAP - 1000, HOT, CLOSE)["enter_resolved"]
    assert target_end_ns(HOT, CLOSE * 1_000_000) == (CAP + 4000) * 1_000_000


def test_random_sparse_paths_and_stop_prices_preserve_frozen_outputs():
    rng = np.random.default_rng(33520261018)
    for _ in range(24):
        times = np.arange(HOT - 60_000, CLOSE, 1000)
        times = times[rng.random(len(times)) < .12]
        prices = np.maximum(.01, 10 + np.cumsum(rng.normal(0, .12, len(times))))
        data = bars(times, prices)
        end = target_end_ns(HOT, CLOSE * 1_000_000) // 1_000_000
        clipped = data.loc[data.t < end]
        decisions = np.r_[rng.integers(HOT, CAP, size=12) // 1000 * 1000,
                          CAP - np.array([1000, 2000, 4000, 60_000])]
        for decision in decisions:
            equal_results(frozen.enter_outcome(data, int(decision), HOT, CLOSE),
                          frozen.enter_outcome(clipped, int(decision), HOT, CLOSE))


def test_session_cap_is_never_extended():
    assert target_end_ns(HOT, (HOT + 1000) * 1_000_000) == (HOT + 1000) * 1_000_000
    # The original population includes identities whose HOT is at the close.
    # Their absence of decision clocks must not remove them from the population.
    assert target_end_ns(HOT, HOT * 1_000_000) == HOT * 1_000_000


def test_all_identities_and_both_streams_survive_without_outcome_selection():
    ledger = [{"trading_day": "2026-05-05", "ticker": ticker, "hot_t": HOT,
               "scope": "training", "stream": stream, "start_ns": (HOT - 60_000) * 1_000_000,
               "end_ns": CLOSE * 1_000_000, "economic_label": None}
              for ticker in ("A", "B") for stream in ("trades", "quotes")]
    result = build_scope(ledger)
    assert (result["identities"], result["stream_partitions"]) == (2, 4)
    assert all(x["start_ns"] == y["start_ns"] for x, y in zip(ledger, result["windows"]))
    assert not result["source_ready"] and not result["collection_enabled"]
    assert not result["full_session_census_replaced"]
    assert result["quote_initial_state_and_correction_lookback_require_separate_coverage"]
    with pytest.raises(ValueError, match="geometry"):
        build_scope(ledger[:-1])
    with pytest.raises(ValueError, match="unique"):
        build_scope(ledger + [ledger[0]])
