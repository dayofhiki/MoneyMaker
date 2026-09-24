from __future__ import annotations

import math

from victory_trader.fresh_entry_economics_audit import (
    BOOTSTRAP_SAMPLES,
    PRICE_LABELS,
    REQUEST_ID,
    _gross_return,
)


def test_request181_contract_is_frozen():
    assert REQUEST_ID == 181
    assert BOOTSTRAP_SAMPLES == 10_000
    assert PRICE_LABELS == ("<1", "1-2", "2-5", "5-10", ">=10")


def test_gross_return_is_raw_open_to_open_pct():
    assert math.isclose(_gross_return(10.0, 11.0), 10.0)
    assert math.isclose(_gross_return(10.0, 9.0), -10.0)
