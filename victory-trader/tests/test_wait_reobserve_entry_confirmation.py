from __future__ import annotations

from victory_trader.rich_second_position_value import RICH_SECOND_FEATURES
from victory_trader.wait_reobserve_entry_confirmation import (
    MAX_SELECTED_RATE,
    MIN_POSITIVE_DAYS,
    MIN_RICH_SECOND_COVERAGE,
    MIN_SELECTED_RATE,
    MIN_SPEARMAN,
    MIN_SPEARMAN_IMPROVEMENT,
    MIN_WAIT_STATE_COVERAGE,
    MODEL_SEED,
    REQUEST185_WAIT_SPEARMAN,
    REQUEST_ID,
    WAIT_STATE_FEATURES,
)


def test_request186_contract_is_frozen():
    assert REQUEST_ID == 186
    assert MODEL_SEED == 20261090
    assert MIN_WAIT_STATE_COVERAGE == 0.85
    assert MIN_RICH_SECOND_COVERAGE == 0.90
    assert MIN_SPEARMAN == 0.20
    assert MIN_SPEARMAN_IMPROVEMENT == 0.03
    assert MIN_SELECTED_RATE == 0.05
    assert MAX_SELECTED_RATE == 0.60
    assert MIN_POSITIVE_DAYS == 4
    assert REQUEST185_WAIT_SPEARMAN == 0.18399598640288747


def test_request186_uses_rich_second_reobserved_state():
    for column in RICH_SECOND_FEATURES:
        assert column in WAIT_STATE_FEATURES
    for forbidden in (
        "entry_open",
        "entry_to_current_close_pct",
        "running_max_return_pct",
        "running_min_return_pct",
        "drawdown_from_peak_pct",
        "recovery_from_trough_pct",
        "exit_reference_open",
    ):
        assert forbidden not in WAIT_STATE_FEATURES
