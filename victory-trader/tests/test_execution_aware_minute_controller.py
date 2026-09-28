import numpy as np
import pandas as pd

from victory_trader import execution_aware_minute_controller as r260


def test_request260_contract():
    assert r260.REQUEST_ID == 260
    assert r260.EXECUTION_THRESHOLD == 0.60
    assert r260.MIN_EXECUTION_AUC == 0.65
    assert r260.MIN_SECOND_ENTRY_COVERAGE == 0.70
    assert r260.MIN_SECOND_RESOLUTION == 0.40


def test_execution_label_requires_immediate_and_three_continuations():
    states = pd.DataFrame(
        [{
            "trading_day": "2026-05-11",
            "ticker": "TEST",
            "state_t": 60_000,
        }]
    )
    times = np.array(
        [
            61_000,
            121_000,
            181_000,
            241_000,
            500_000,
        ],
        dtype=np.int64,
    )
    opens = np.array([10.0, 10.1, 10.2, 10.3, 10.4], dtype=float)
    out = r260.label_execution_support(
        states,
        {("2026-05-11", "TEST"): (times, opens)},
    )
    assert bool(out.loc[0, "execution_immediate_available"])
    assert out.loc[0, "execution_continuation_hits"] == 3
    assert bool(out.loc[0, "execution_supported"])


def test_execution_label_rejects_missing_immediate():
    states = pd.DataFrame(
        [{
            "trading_day": "2026-05-11",
            "ticker": "TEST",
            "state_t": 60_000,
        }]
    )
    times = np.array(
        [121_000, 181_000, 241_000, 301_000, 361_000],
        dtype=np.int64,
    )
    opens = np.ones(len(times), dtype=float) * 10.0
    out = r260.label_execution_support(
        states,
        {("2026-05-11", "TEST"): (times, opens)},
    )
    assert not bool(out.loc[0, "execution_immediate_available"])
    assert not bool(out.loc[0, "execution_supported"])
