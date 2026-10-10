"""Synthetic schema/order fixtures only; no real-market conclusions."""
from decimal import Decimal

import pytest

from victory_trader.r335_tick_semantics_audit import audit_rows, row_key
from victory_trader.tick_execution_readiness import format_ns

NOW = 1777990800000000001


def quote(**changes):
    return {"t": format_ns(NOW), "bp": Decimal("1.01"), "ap": Decimal("1.02"),
            "bs": 100, "as": 200, "bx": "Q", "ax": "P", "c": ["R"], "z": "C", **changes}


def trade(**changes):
    return {"t": format_ns(NOW), "p": Decimal("1.02"), "s": 10, "x": "Q",
            "i": 1, "c": ["@"], "z": "C", **changes}


def test_equal_clock_quantity_conflicts_survive_page_boundaries_and_reversal():
    rows = [(0, quote()), (1, quote(**{"as": 100}))]
    result = audit_rows(rows, "quotes")
    assert result == audit_rows(list(reversed(rows)), "quotes")
    assert result["equal_clock_groups"] == result["cross_page_equal_clock_groups"] == 1
    assert result["rows_in_equal_clock_groups"] == 2
    assert result["equal_clock_variation_groups"]["size"] == 1
    assert result["equal_clock_variation_groups"]["price"] == 0
    assert not result["source_order_is_execution_sequence"] and not result["fill_confirmed"]
    assert result["economic_label"] is None
    assert rows[0][1]["as"] == 200  # Never select/drop/overwrite a conflicting row.


def test_adjacent_nanoseconds_do_not_create_false_clock_collisions():
    result = audit_rows([(0, quote()), (0, quote(t=format_ns(NOW + 1)))], "quotes")
    assert result["equal_clock_groups"] == 0 and result["rows"] == 2
    assert result["fractional_digits_observed"] == {"9": 2}


def test_duplicate_rows_are_reported_and_retained_across_pages():
    original = quote()
    reversed_keys = dict(reversed(list(original.items())))
    result = audit_rows([(0, original), (1, reversed_keys)], "quotes")
    assert result["identical_decoded_rows"] == 1 and result["rows"] == 2
    assert result["equal_clock_variation_groups"]["size"] == 0


def test_api_zero_side_and_locked_crossed_states_are_distinct_from_schema_errors():
    result = audit_rows([(0, quote(bp=0, bs=0)), (0, quote(ap=Decimal("1.01"))),
                         (0, quote(ap=Decimal("1.00")))], "quotes")
    assert result["quote_rows_with_inactive_side"] == 1
    assert result["locked_quote_rows"] == result["crossed_quote_rows"] == 1
    assert not any(result["invalid_schema_values"].values())


@pytest.mark.parametrize("field,value", [("as", True), ("bs", -1), ("as", 2**32),
                                         ("bp", float("nan")), ("c", "R"), ("bx", None)])
def test_invalid_fields_are_counted_without_imputation(field, value):
    result = audit_rows([(0, quote(**{field: value}))], "quotes")
    assert any(result["invalid_schema_values"].values())
    assert result["rows"] == 1 and not result["source_ready"]


@pytest.mark.parametrize("status", [None, [], "future_status"])
def test_unrecognized_update_fields_remain_unknown(status):
    result = audit_rows([(0, trade(u=status))], "trades")
    assert result["trade_update_statuses"] == {"unknown": 1}


def test_current_record_updates_do_not_become_as_observed_history():
    result = audit_rows([(0, trade()), (0, trade(u="incorrect")),
                         (0, trade(u="corrected", i=2)), (1, trade(u="canceled", i=3))], "trades")
    assert result["trade_update_statuses"] == {"absent": 1, "canceled": 1, "corrected": 1, "incorrect": 1}
    assert result["rows"] == 4 and result["equal_clock_variation_groups"]["trade_update"] == 1
    assert not result["source_order_is_execution_sequence"] and result["economic_label"] is None


def test_empty_and_incomplete_samples_do_not_imply_source_completeness():
    result = audit_rows([], "quotes")
    assert result["rows"] == 0 and not result["source_ready"]


def test_precision_and_exact_decimal_fingerprints_preserve_distinct_records():
    assert row_key(quote(ap=Decimal("1.0200000000000000001"))) != row_key(quote())
    assert row_key({"bs": True}) != row_key({"bs": 1})


def test_backwards_provider_order_is_detected_without_sorting():
    rows = [(0, quote(t=format_ns(NOW + 1))), (1, quote())]
    assert audit_rows(rows, "quotes")["provider_clock_backwards"] == 1
    assert rows[0][1]["t"] == format_ns(NOW + 1)


def test_condition_mapping_reports_unknown_shapes_without_eligibility_guess():
    result = audit_rows([(0, quote(c=["R"])), (0, quote(c=["R", "Y"])),
                         (0, quote(c=[])), (0, quote(c=["R", "Y", "X"]))], "quotes")
    assert result["quote_condition_side_mapping_unresolved_rows"] == 2
    assert not result["source_ready"]


@pytest.mark.parametrize("stream", [None, "bars", "seconds"])
def test_only_explicit_tick_streams_are_audited(stream):
    with pytest.raises(ValueError):
        audit_rows([], stream)
