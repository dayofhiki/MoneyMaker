"""Offline audit of retained R335 wire pages. No collection, fills or labels.

Equal provider clocks are grouped across page boundaries for diagnostics only.
No row is sorted, dropped, scaled, assigned a SIP/receipt clock or selected as
the last market state. This audit preserves the original frozen collector.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from .offline_source_inventory import canonical, confined, sha, strict_json
from .r335_alpaca_qualification import checked_payload, load_plan, replay
from .tick_execution_readiness import alpaca_row, exact_decimal, rfc3339_ns

FIELDS = {"quotes": {"t", "bp", "ap", "bs", "as", "bx", "ax", "c", "z"},
          "trades": {"t", "p", "s", "x", "i", "c", "z"}}
TRADE_UPDATES = {"canceled", "incorrect", "corrected"}


def row_key(value):
    """Exact decoded-value comparison, without float or dictionary-order loss."""
    if isinstance(value, dict):
        return tuple((key, row_key(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(row_key(item) for item in value)
    if isinstance(value, Decimal):
        return ("Decimal", str(value))
    return (type(value).__name__, value)


def audit_rows(page_rows, stream):
    if stream not in FIELDS:
        raise ValueError("explicit_tick_stream_required")
    groups = defaultdict(list)
    missing, invalid, extra, precision, units, updates = (Counter() for _ in range(6))
    numbers = ("bp", "ap", "bs", "as") if stream == "quotes" else ("p", "s")
    sizes = ("bs", "as") if stream == "quotes" else ("s",)
    venues = ("bx", "ax") if stream == "quotes" else ("x",)
    rows, backwards, previous, locked, crossed, inactive, side_unknown = 0, 0, None, 0, 0, 0, 0
    for page, row in page_rows:
        if type(page) is not int or page < 0 or not isinstance(row, dict):
            raise ValueError("explicit_page_index_and_object_row_required")
        clock = rfc3339_ns(row.get("t"))
        backwards += int(previous is not None and clock < previous)
        previous = clock
        rows += 1
        groups[clock].append((page, row))
        match = re.search(r"\.(\d+)(?:Z|[+-]\d{2}:\d{2})$", row["t"])
        precision[str(len(match.group(1)) if match else 0)] += 1
        units[alpaca_row(row, stream=stream, feed="sip")["size_unit"]] += 1
        for field in FIELDS[stream]:
            missing[field] += int(row.get(field) is None)
        extra.update(set(row) - FIELDS[stream] - ({"u"} if stream == "trades" else set()))
        for field in numbers:
            try:
                number = exact_decimal(row.get(field))
                invalid[field] += int(number < 0 or (stream == "trades" and number == 0))
            except ValueError:
                invalid[field] += 1
        for field in sizes:
            value = row.get(field)
            invalid[field + "_uint32"] += int(type(value) is not int or not 0 <= value < 2**32)
        for field in venues:
            value = row.get(field)
            invalid[field] += int(not isinstance(value, str) or len(value) != 1)
        conditions = row.get("c")
        valid_conditions = isinstance(conditions, list) and all(isinstance(c, str) for c in conditions)
        invalid["c"] += int(not valid_conditions)
        invalid["z"] += int(row.get("z") not in ("A", "B", "C", "N", "O"))
        if stream == "trades":
            identifier = row.get("i")
            invalid["i_uint64"] += int(type(identifier) is not int or not 0 <= identifier < 2**64)
            update = row.get("u")
            known_update = isinstance(update, str) and update in TRADE_UPDATES
            updates["absent" if "u" not in row else update if known_update else "unknown"] += 1
        else:
            side_unknown += int(not valid_conditions or len(conditions) not in (1, 2))
            try:
                bid, ask, bid_size, ask_size = (exact_decimal(row.get(k)) for k in numbers)
                inactive += int(min(bid, ask, bid_size, ask_size) == 0)
                locked += int(bid > 0 and bid == ask)
                crossed += int(bid > ask > 0)
            except ValueError:
                pass
    ties = [items for items in groups.values() if len(items) > 1]
    variants = {"size": sizes, "price": ("bp", "ap") if stream == "quotes" else ("p",),
                "venue": venues, "condition": ("c",), "trade_update": ("u",)}
    return {"rows": rows, "provider_clock_backwards": backwards,
            "fractional_digits_observed": dict(sorted(precision.items())),
            "size_units": dict(sorted(units.items())),
            "missing_fields": dict(sorted(missing.items())),
            "invalid_schema_values": dict(sorted(invalid.items())),
            "additional_fields": dict(sorted(extra.items())),
            "trade_update_statuses": dict(sorted(updates.items())),
            "equal_clock_groups": len(ties), "rows_in_equal_clock_groups": sum(map(len, ties)),
            "cross_page_equal_clock_groups": sum(len({page for page, _ in items}) > 1 for items in ties),
            "identical_decoded_rows": sum(len(items) - len({row_key(r) for _, r in items}) for items in groups.values()),
            "equal_clock_variation_groups": {
                name: sum(len({tuple(row_key(row.get(k)) for k in keys) for _, row in items}) > 1 for items in ties)
                for name, keys in variants.items() if name != "trade_update" or stream == "trades"},
            "locked_quote_rows": locked, "crossed_quote_rows": crossed,
            "quote_rows_with_inactive_side": inactive,
            "quote_condition_side_mapping_unresolved_rows": side_unknown,
            "timestamp_semantics": "UNVERIFIED_PROVIDER_CLOCK",
            "source_order_is_execution_sequence": False, "source_ready": False,
            "fill_confirmed": False, "economic_label": None}


def audit_evidence(plan, root):
    verified = replay(plan, root)  # Original hashes, bounds, pages, tokens and denominator.
    manifest_bytes = confined(root, "manifest.json").read_bytes()
    manifest = strict_json(manifest_bytes)
    pairs, stream_clocks = [], defaultdict(dict)
    for pair in manifest["pairs"]:
        request = pair["request"]
        page_rows = []
        for index, page in enumerate(pair["pages"]):
            if page["observation"] is not None:
                payload = checked_payload(confined(root, page["wire_file"]).read_bytes())
                page_rows.extend((index, row) for row in payload[request["stream"]].get(request["ticker"], []))
        result = audit_rows(page_rows, request["stream"])
        pairs.append({"ticker": request["ticker"], "trading_day": request["trading_day"],
                      "stream": request["stream"], "collection_state": pair["state"], **result})
        stream_clocks[(request["trading_day"], request["ticker"])][request["stream"]] = {
            rfc3339_ns(row["t"]) for _, row in page_rows}
    return {"request_id": 335, "stage": "offline_provider_semantics_audit_only",
            "input_manifest_sha256": sha(manifest_bytes),
            "audit_implementation_sha256": sha(Path(__file__).read_bytes()),
            "adapter_implementation_sha256": sha(Path(__file__).with_name("tick_execution_readiness.py").read_bytes()),
            "original_replay_sha256": sha(canonical(verified)),
            "pairs": pairs, "cross_stream_equal_provider_clock_groups": sum(
                len(clocks.get("trades", set()) & clocks.get("quotes", set())) for clocks in stream_clocks.values()),
            "additional_market_HTTP_attempts": 0, "additional_cost_USD": 0,
            "source_ready": False, "new_economic_labels": 0,
            "full_census_collected": False, "profitability_validation_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("research/request335-free-source-plan.json.gz"))
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_evidence(load_plan(args.plan), args.evidence)
    args.output.write_bytes(canonical(result))


if __name__ == "__main__":
    main()
