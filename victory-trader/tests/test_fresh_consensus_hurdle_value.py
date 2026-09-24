from __future__ import annotations

from victory_trader.fresh_consensus_hurdle_value import (
    FRESH_DAYS,
    MIN_CONSENSUS_SPEARMAN,
    MIN_CONTEXT_COVERAGE,
    MIN_GOOD_DAYS,
    MIN_PATH_COVERAGE,
    MIN_SELECTED_RATE,
    MIN_SELECTED_REALIZED_MEAN_PCT,
    REQUEST_ID,
)


def test_request178_contract_is_frozen():
    assert REQUEST_ID == 178
    assert FRESH_DAYS == (
        "2026-06-23",
        "2026-06-24",
        "2026-06-25",
        "2026-06-26",
        "2026-06-29",
    )
    assert MIN_PATH_COVERAGE == 0.85
    assert MIN_CONTEXT_COVERAGE == 0.95
    assert MIN_CONSENSUS_SPEARMAN == 0.08
    assert MIN_SELECTED_RATE == 0.10
    assert MIN_SELECTED_REALIZED_MEAN_PCT == 0.20
    assert MIN_GOOD_DAYS == 4
