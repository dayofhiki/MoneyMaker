"""All prices, clocks and books in this file are synthetic code fixtures."""
from dataclasses import replace
from decimal import Decimal

import pytest

from victory_trader.tick_execution_readiness import (
    MAX_REFERENCE_LAG_NS, QuoteAuditContract, alpaca_row, format_ns, quote_at_arrival,
    rfc3339_ns,
)

NOW = 1778097480000000001
CONTRACT = QuoteAuditContract(100, 1000, Decimal("0.5"), False,
                             "registered_SIP_latency_scenario")


def quote(**updates):
    return {"available_ns": NOW, "sip_timestamp": NOW-1, "feed": "SIP_NBBO",
            "size_unit": "shares", "eligibility": "ELIGIBLE_UNDER_REGISTERED_RULE",
            "bid_price": "1.01", "ask_price": "1.02", "bid_size": "100",
            "ask_size": "200", **updates}


def audit(rows, **updates):
    return quote_at_arrival(rows, decision_ns=NOW, side="BUY",
                            requested_shares=Decimal("100"), contract=CONTRACT, **updates)


@pytest.mark.parametrize("fraction", ["", ".1", ".123456", ".123456789"])
def test_rfc3339_precision_and_offset_roundtrip(fraction):
    text = f"2026-05-06T13:18:00{fraction}Z"
    integer = rfc3339_ns(text)
    assert rfc3339_ns(format_ns(integer)) == integer
    assert integer == rfc3339_ns(f"2026-05-06T09:18:00{fraction}-04:00")


def test_adjacent_nanoseconds_do_not_collapse():
    assert rfc3339_ns(format_ns(NOW+1)) - rfc3339_ns(format_ns(NOW)) == 1


@pytest.mark.parametrize("clock", [True, NOW, float(NOW), "2026-05-06T13:18:00",
                                   "2026-05-06T13:18:00.1234567890Z",
                                   "2026-05-06T13:18:00-00:00",
                                   "2026-05-06T13:18:00+25:00", "2026-02-30T00:00:00Z"])
def test_unknown_or_lossy_clocks_are_rejected(clock):
    with pytest.raises(ValueError):
        rfc3339_ns(clock)


def test_alpaca_provider_clock_never_becomes_sip_and_lots_never_become_shares():
    raw = {"t": format_ns(NOW), "bp": "1.01", "ap": "1.02", "bs": 2, "as": 3,
           "bx": "Q", "ax": "P", "c": ["R"], "z": "C"}
    normalized = alpaca_row(raw, stream="quotes", feed="sip")
    assert normalized["provider_timestamp_ns"] == NOW
    assert normalized["sip_timestamp"] is None
    assert normalized["size_unit"] == "UNVERIFIED_HISTORICAL_SIZE_UNIT"
    assert normalized["documented_realtime_size_unit_hint"] == "round_lots"
    assert normalized["ask_size"] == 3
    assert not normalized["source_ready"]
    assert raw["t"] == format_ns(NOW) and "sip_timestamp" not in raw


def test_alpaca_trade_retains_missing_corrections_and_sequences():
    normalized = alpaca_row({"t": format_ns(NOW), "p": "1.02", "s": 1, "i": 7},
                            stream="trades", feed="sip")
    assert normalized["size_unit"] == "UNVERIFIED_HISTORICAL_SIZE_UNIT"
    assert normalized["documented_realtime_size_unit_hint"] == "shares"
    assert normalized["sequence_number"] is normalized["correction"] is None


@pytest.mark.parametrize("feed", ["iex", "boats", "otc", None])
def test_non_sip_feeds_cannot_use_this_adapter(feed):
    with pytest.raises(ValueError):
        alpaca_row({"t": format_ns(NOW)}, stream="quotes", feed=feed)


def test_buy_and_sell_use_different_displayed_sides_without_proving_fills():
    buy = audit([quote()])
    sell = quote_at_arrival([quote()], decision_ns=NOW, side="SELL",
                           requested_shares=Decimal("50"), contract=CONTRACT)
    assert buy["crossing_price"] == "1.02" and buy["displayed_share_ceiling"] == "100.0"
    assert sell["crossing_price"] == "1.01" and sell["displayed_share_ceiling"] == "50.0"
    assert buy["requested_quantity_supported"] and sell["requested_quantity_supported"]
    assert not buy["fill_confirmed"] and buy["economic_label"] is None


def test_future_quotes_do_not_change_prefix_diagnostic():
    assert audit([quote()]) == audit([quote(), quote(available_ns=NOW+101, ask_price="999")])
    assert audit([quote(available_ns=NOW+101)])["state"] == "NO_PRIOR_QUOTE"


def test_latest_bad_book_supersedes_prior_valid_book():
    latest = quote(available_ns=NOW+99, ask_price="0")
    assert audit([quote(), latest])["state"] == "INVALID_QUOTE"


def test_equal_clock_ambiguity_not_resolved_by_file_order_or_prices():
    rows = [quote(), quote(ask_price="0.01")]
    assert audit(rows) == audit(list(reversed(rows)))
    assert audit(rows)["state"] == "AMBIGUOUS_EQUAL_CLOCK_ORDER"


def test_staleness_uses_original_sip_clock_not_late_receipt():
    assert audit([quote(sip_timestamp=NOW-1000)])["state"] == "STALE_QUOTE"
    assert audit([quote(sip_timestamp=NOW-900)])["state"] == "QUOTED_SIZE_SUPPORTED"


@pytest.mark.parametrize("updates,state", [
    ({"sip_timestamp": None}, "SIP_CLOCK_UNVERIFIED"),
    ({"sip_timestamp": float(NOW)}, "SIP_CLOCK_UNVERIFIED"),
    ({"sip_timestamp": NOW+1}, "IMPOSSIBLE_AVAILABILITY_BEFORE_SIP"),
    ({"feed": "IEX"}, "NON_EQUIVALENT_FEED_OR_UNVERIFIED_UNITS"),
    ({"size_unit": "round_lots"}, "NON_EQUIVALENT_FEED_OR_UNVERIFIED_UNITS"),
    ({"eligibility": None}, "QUOTE_ELIGIBILITY_UNVERIFIED"),
    ({"ask_price": "0.99"}, "INVALID_QUOTE"),
    ({"ask_price": "1.01"}, "LOCKED_QUOTE_NOT_ADMITTED"),
    ({"ask_price": "NaN"}, "INVALID_QUOTE"),
    ({"bid_size": 0}, "INVALID_QUOTE"),
    ({"ask_size": None}, "INVALID_QUOTE"),
    ({"ask_price": 1.02}, "INVALID_QUOTE"),
    ({"ask_size": "199"}, "INSUFFICIENT_DISPLAYED_SIZE"),
])
def test_unknowns_are_explicit_not_imputed(updates, state):
    result = audit([quote(**updates)])
    assert result["state"] == state
    assert not result["fill_confirmed"] and result["economic_label"] is None


@pytest.mark.parametrize("updates", [{"latency_ns": MAX_REFERENCE_LAG_NS+1},
                                     {"latency_ns": True}, {"max_quote_age_ns": -1},
                                     {"displayed_fraction": Decimal("1.01")},
                                     {"displayed_fraction": Decimal("NaN")},
                                     {"availability_rule": "assume_zero_latency"}])
def test_contract_requires_bounded_explicit_assumptions(updates):
    with pytest.raises(ValueError):
        replace(CONTRACT, **updates).validate()


def test_unsorted_input_is_selected_by_availability_not_participant_or_file_order():
    rows = [quote(available_ns=NOW+99), quote(available_ns=NOW-100)]
    assert audit(rows) == audit(list(reversed(rows)))
    assert audit(rows)["quote_available_ns"] == NOW+99


def test_empty_input_retains_unknown():
    assert audit([])["state"] == "NO_PRIOR_QUOTE"
