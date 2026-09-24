from __future__ import annotations

import numpy as np
import pandas as pd

from victory_trader.economic_signal_audit import _missing_reason


def test_missing_reason_labeled():
    row = pd.Series(
        {
            "oracle_best_base_pct": 1.2,
            "entry_reference_available": True,
            "minutes_to_close": 100,
        }
    )
    assert _missing_reason(row) == "labeled"


def test_missing_reason_mid_session_missing_entry():
    row = pd.Series(
        {
            "oracle_best_base_pct": np.nan,
            "entry_reference_available": False,
            "minutes_to_close": 120,
        }
    )
    assert _missing_reason(row) == "mid_session_no_exact_next_minute_entry"


def test_missing_reason_near_close_exit():
    row = pd.Series(
        {
            "oracle_best_base_pct": np.nan,
            "entry_reference_available": True,
            "minutes_to_close": 20,
        }
    )
    assert _missing_reason(row) == "near_close_no_future_exit"
