from __future__ import annotations

import pandas as pd

from victory_trader.honest_patience_override import (
    MAX_HOLD_MINUTES,
    MIN_COMPLETION_COVERAGE,
    MIN_HONEST_TARGET_ROWS,
    MIN_START_COVERAGE,
    MODEL_SEED,
    REQUEST_ID,
    split_calibration_days,
)


def test_request180_contract_is_frozen():
    assert REQUEST_ID == 180
    assert MODEL_SEED == 20261080
    assert MAX_HOLD_MINUTES == 30
    assert MIN_START_COVERAGE == 0.80
    assert MIN_COMPLETION_COVERAGE == 0.90
    assert MIN_HONEST_TARGET_ROWS == 500


def test_request180_splits_calibration_by_whole_days():
    days = [f"2026-05-{day:02d}" for day in range(1, 9)]
    frame = pd.DataFrame(
        {
            "trading_day": [day for day in days for _ in range(2)],
            "x": range(16),
        }
    )
    a, b, a_days, b_days = split_calibration_days(frame)
    assert a_days == days[:4]
    assert b_days == days[4:]
    assert sorted(a["trading_day"].unique()) == days[:4]
    assert sorted(b["trading_day"].unique()) == days[4:]
