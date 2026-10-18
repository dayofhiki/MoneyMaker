"""R335 offline adapters and quote diagnostics; no collection or economic labels."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

NANOSECOND = 1_000_000_000
MAX_REFERENCE_LAG_NS = 3 * NANOSECOND
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
RFC3339 = re.compile(
    r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})"
)
ALPACA_QUOTE_UNIT_CHANGE_DAY = "2025-11-03"
ALPACA_UNIT_EVIDENCE = (
    "https://docs.alpaca.markets/us/v1.1/changelog/marketdata-bid-and-ask-size-display-change"
)


def ns_integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative_integer_nanoseconds_required")
    return value


def rfc3339_ns(value: object) -> int:
    """Keep all nine digits; datetime is used only for whole calendar seconds."""
    match = RFC3339.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("exact_RFC3339_clock_required")
    day, clock, fraction, offset = match.groups()
    if offset == "-00:00":
        raise ValueError("unknown_local_offset_not_a_verified_clock")
    zone = timezone.utc
    if offset != "Z":
        hours, minutes = int(offset[1:3]), int(offset[4:6])
        if hours > 23 or minutes > 59:
            raise ValueError("invalid_UTC_offset")
        sign = 1 if offset[0] == "+" else -1
        zone = timezone(sign * timedelta(hours=hours, minutes=minutes))
    delta = datetime.fromisoformat(f"{day}T{clock}").replace(tzinfo=zone) - EPOCH
    result = (delta.days * 86400 + delta.seconds) * NANOSECOND
    result += int((fraction or "").ljust(9, "0") or "0")
    return ns_integer(result)


def format_ns(value: int) -> str:
    seconds, fraction = divmod(ns_integer(value), NANOSECOND)
    return (EPOCH + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S") + f".{fraction:09d}Z"


def alpaca_row(row: dict, *, stream: str, feed: str) -> dict:
    """Describe, never fabricate, missing Massive-style clocks or share units.

    A declared SIP request is not authentication of the response/feed. The
    historical `t` field remains a provider clock until its meaning is verified.
    CTA/UTP quote sizes changed to shares on 2025-11-03. Earlier quote lots
    require a date/symbol-specific round-lot mapping before conversion. The
    current schema documents this date rule for historical stock_quote records.
    Original response bytes and pagination evidence must be retained separately.
    """
    if not isinstance(row, dict) or stream not in ("trades", "quotes") or feed != "sip":
        raise ValueError("explicit_SIP_tick_stream_required")
    mapping = ({"price": "p", "size": "s", "exchange": "x", "id": "i"}
               if stream == "trades" else
               {"bid_price": "bp", "ask_price": "ap", "bid_size": "bs",
                "ask_size": "as", "bid_exchange": "bx", "ask_exchange": "ax"})
    provider_ns = rfc3339_ns(row.get("t"))
    seconds = provider_ns // NANOSECOND
    market_day = (EPOCH + timedelta(seconds=seconds)).astimezone(
        ZoneInfo("America/New_York")).date().isoformat()
    unit = ("shares" if market_day >= ALPACA_QUOTE_UNIT_CHANGE_DAY else "round_lots")
    result = {key: row.get(field) for key, field in mapping.items()}
    result.update(provider_timestamp_ns=provider_ns,
                  timestamp_semantics="UNVERIFIED_PROVIDER_CLOCK", declared_feed=feed,
                  sip_timestamp=None, participant_timestamp=None, trf_timestamp=None,
                  sequence_number=None, correction=None, conditions=row.get("c"),
                  tape=row.get("z"),
                  size_unit=unit if stream == "quotes" else "UNVERIFIED_HISTORICAL_SIZE_UNIT",
                  size_unit_evidence=ALPACA_UNIT_EVIDENCE if stream == "quotes" else None,
                  round_lot_shares=None,
                  trade_update_status=row.get("u") if stream == "trades" else None,
                  documented_realtime_size_unit_hint="shares" if stream == "trades" else unit,
                  source_ready=False)
    return result


def exact_decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError("decimal_text_or_exact_number_required")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError("invalid_decimal") from None
    if not result.is_finite():
        raise ValueError("finite_decimal_required")
    return result


@dataclass(frozen=True)
class QuoteAuditContract:
    """Explicit inputs for a diagnostic, not a registered fill model.

    No defaults for feed availability, latency, freshness, participation,
    eligibility or locked markets. Real-data use needs a separate preregistration.
    """
    latency_ns: int
    max_quote_age_ns: int
    displayed_fraction: Decimal
    allow_locked: bool
    availability_rule: str

    def validate(self) -> None:
        if ns_integer(self.latency_ns) > MAX_REFERENCE_LAG_NS:
            raise ValueError("original_3s_bound_exceeded")
        ns_integer(self.max_quote_age_ns)
        fraction = exact_decimal(self.displayed_fraction)
        if not 0 < fraction <= 1 or type(self.allow_locked) is not bool:
            raise ValueError("invalid_quote_audit_contract")
        if self.availability_rule not in ("measured_receipt", "registered_SIP_latency_scenario"):
            raise ValueError("explicit_availability_rule_required")


def quote_at_arrival(rows: list[dict], *, decision_ns: int, side: str,
                     requested_shares: Decimal, contract: QuoteAuditContract) -> dict:
    """Audit the last observable NBBO state, without an after-target fallback.

    `available_ns` must come from a separately frozen availability mapping, not
    a participant clock or a retrospective sorting of corrected history.
    An invalid latest quote supersedes earlier quotes: never skip it to rescue
    a stale valid book. Equal availability clocks are ambiguous without a proven
    ordering adapter. Displayed quantity is a ceiling, not an executed fill.
    """
    contract.validate()
    arrival = ns_integer(decision_ns) + contract.latency_ns
    shares = exact_decimal(requested_shares)
    if side not in ("BUY", "SELL") or shares <= 0:
        raise ValueError("positive_order_and_explicit_side_required")
    out = {"stage": "quote_observation_diagnostic_only", "arrival_ns": arrival,
           "state": "NO_PRIOR_QUOTE", "quote_available_ns": None, "quote_age_ns": None,
           "crossing_price": None, "displayed_share_ceiling": None,
           "requested_quantity_supported": False, "fill_confirmed": False,
           "economic_label": None, "availability_rule": contract.availability_rule}
    eligible = []
    for row in rows:
        available = ns_integer(row.get("available_ns"))
        if available <= arrival:
            eligible.append(row)
    if not eligible:
        return out
    latest = max(row["available_ns"] for row in eligible)
    ties = [row for row in eligible if row["available_ns"] == latest]
    out.update(quote_available_ns=latest)
    if len(ties) != 1:
        return {**out, "state": "AMBIGUOUS_EQUAL_CLOCK_ORDER"}
    row = ties[0]
    try:
        sip = ns_integer(row.get("sip_timestamp"))
    except ValueError:
        return {**out, "state": "SIP_CLOCK_UNVERIFIED"}
    if sip > latest:
        return {**out, "state": "IMPOSSIBLE_AVAILABILITY_BEFORE_SIP"}
    out["quote_age_ns"] = arrival - sip
    if row.get("feed") != "SIP_NBBO" or row.get("size_unit") != "shares":
        return {**out, "state": "NON_EQUIVALENT_FEED_OR_UNVERIFIED_UNITS"}
    # Eligibility is not guessed from a condition code from another vendor.
    if row.get("eligibility") != "ELIGIBLE_UNDER_REGISTERED_RULE":
        return {**out, "state": "QUOTE_ELIGIBILITY_UNVERIFIED"}
    if arrival - sip > contract.max_quote_age_ns:
        return {**out, "state": "STALE_QUOTE"}
    try:
        bid, ask, bid_size, ask_size = (exact_decimal(row.get(key)) for key in
                                      ("bid_price", "ask_price", "bid_size", "ask_size"))
    except ValueError:
        return {**out, "state": "INVALID_QUOTE"}
    if min(bid, ask) <= 0 or min(bid_size, ask_size) <= 0 or ask < bid:
        return {**out, "state": "INVALID_QUOTE"}
    if ask == bid and not contract.allow_locked:
        return {**out, "state": "LOCKED_QUOTE_NOT_ADMITTED"}
    price, size = (ask, ask_size) if side == "BUY" else (bid, bid_size)
    ceiling = size * exact_decimal(contract.displayed_fraction)
    supported = shares <= ceiling
    return {**out, "state": "QUOTED_SIZE_SUPPORTED" if supported else "INSUFFICIENT_DISPLAYED_SIZE",
            "crossing_price": str(price), "displayed_share_ceiling": str(ceiling),
            "requested_quantity_supported": supported}
