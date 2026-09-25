import math

import pandas as pd

from victory_trader.causal_second_risk_compatible_entry import (
    BASE_SCENARIO,
    SecondStore,
)
from victory_trader.causal_second_execution import SecondBar
from victory_trader.execution_costs import modeled_buy_fill
from victory_trader.volatility_normalized_bracket_geometry import (
    causal_r_pct,
    replay_bracket,
)


class FakeStore:
    def __init__(self, frame):
        self.frame = frame
        self.halt_status = {}

    def seconds(self, day, ticker):
        return self.frame

    def halts(self, day):
        return {}


def _seconds(rows):
    return pd.DataFrame(rows)


def test_causal_r_uses_only_prior_five_minutes_and_clips():
    decision_t = 600_000
    rows = []
    # Five completed prior minute buckets with ~1%,2%,4%,6%,10% ranges.
    for i, pct in enumerate([1.0, 2.0, 4.0, 6.0, 10.0]):
        minute_start = decision_t - (5 - i) * 60_000
        low = 100.0
        high = 100.0 * (1.0 + pct / 100.0)
        rows.append(
            {
                "t": minute_start,
                "o": 100.0,
                "h": high,
                "l": low,
                "c": 100.0,
                "v": 100.0,
                "n": 10.0,
            }
        )
    # Huge move at/after decision must not leak into R.
    rows.append(
        {
            "t": decision_t,
            "o": 100.0,
            "h": 200.0,
            "l": 50.0,
            "c": 150.0,
            "v": 1000.0,
            "n": 100.0,
        }
    )
    value = causal_r_pct(_seconds(rows), decision_t)
    # Median of [1,2,4,6,10] is 4, inside the 3-8 clip.
    assert math.isclose(value, 4.0, abs_tol=1e-12)


def test_causal_r_requires_three_active_minutes():
    decision_t = 600_000
    rows = [
        {
            "t": decision_t - 120_000,
            "o": 100.0,
            "h": 104.0,
            "l": 100.0,
            "c": 102.0,
            "v": 100.0,
            "n": 10.0,
        },
        {
            "t": decision_t - 60_000,
            "o": 100.0,
            "h": 106.0,
            "l": 100.0,
            "c": 102.0,
            "v": 100.0,
            "n": 10.0,
        },
    ]
    assert math.isnan(causal_r_pct(_seconds(rows), decision_t))


def test_bracket_trigger_fills_only_on_later_second_open():
    decision_t = 100_000
    entry_reference = 100.0
    entry_modeled = modeled_buy_fill(
        entry_reference,
        BASE_SCENARIO,
    )
    # Bar at t=101s is the entry fill. The next bar hits the +5% take,
    # but the exit must wait for the later eligible open at t=103s.
    frame = _seconds(
        [
            {
                "t": 101_000,
                "o": 100.0,
                "h": 101.0,
                "l": 99.5,
                "c": 100.5,
                "v": 100.0,
                "n": 10.0,
            },
            {
                "t": 102_000,
                "o": 100.5,
                "h": entry_modeled * 1.06,
                "l": 100.0,
                "c": 105.0,
                "v": 100.0,
                "n": 10.0,
            },
            {
                "t": 103_000,
                "o": 104.0,
                "h": 104.0,
                "l": 103.5,
                "c": 103.8,
                "v": 100.0,
                "n": 10.0,
            },
        ]
    )
    row = {
        "trading_day": "2026-05-11",
        "ticker": "TEST",
        "entry_fill_t": 101_000,
        "entry_modeled_price": entry_modeled,
    }
    result = replay_bracket(
        row,
        FakeStore(frame),
        stop_pct=3.0,
        take_pct=5.0,
    )
    assert result.status == "closed"
    assert result.reason == "take"
    assert result.exit_fill_t == 103_000


def test_same_second_stop_and_take_is_ambiguous():
    entry_reference = 100.0
    entry_modeled = modeled_buy_fill(
        entry_reference,
        BASE_SCENARIO,
    )
    frame = _seconds(
        [
            {
                "t": 101_000,
                "o": 100.0,
                "h": entry_modeled * 1.06,
                "l": entry_modeled * 0.96,
                "c": 100.0,
                "v": 100.0,
                "n": 10.0,
            }
        ]
    )
    row = {
        "trading_day": "2026-05-11",
        "ticker": "TEST",
        "entry_fill_t": 101_000,
        "entry_modeled_price": entry_modeled,
    }
    result = replay_bracket(
        row,
        FakeStore(frame),
        stop_pct=3.0,
        take_pct=5.0,
    )
    assert result.status == "ambiguous"
    assert result.reason == "stop_and_take_same_second"
