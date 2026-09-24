from __future__ import annotations

from victory_trader.positive_value_calibration_audit import (
    HOLD_BUCKETS,
    REQUEST_ID,
)


def test_request175_is_diagnostic_only():
    assert REQUEST_ID == 175
    assert HOLD_BUCKETS == (
        ("1-3", 1, 3),
        ("4-7", 4, 7),
        ("8-15", 8, 15),
        ("16-29", 16, 29),
    )
