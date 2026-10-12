"""Map published R335 diagnostics to every original identity; no raw replay or labels."""
from __future__ import annotations

import argparse
import gzip
from collections import Counter, defaultdict
from pathlib import Path

from .offline_source_inventory import canonical, sha, strict_json
from .r335_alpaca_qualification import PLAN_SHA, load_plan

PINS = {
    "ledger": "06151d77c57c633c67c2896dc4b5faadc3f8060a81a8f90a1871bde35d50b9dd",
    "acquisition": "341f0cc646d0b54796fe95b91af5b998f0f0eea3010ff36a25e122eefa9f6943",
    "audit": "0d284efdfccdaf7891da610ae3c6b5be414c6663a339285e021107b92d3be7be",
}
MANIFEST_SHA = "91639312f0e1c6a7845ff08c74337ec3805b3105cb24494eaaafd20b9374339b"
FLAGS = (
    "equal_provider_clock", "equal_clock_price_or_size_field_variation",
    "cross_page_equal_clock", "locked_quote", "crossed_quote", "inactive_quote_side",
    "trade_update_present", "empty_trade_or_quote_query", "required_field_or_schema_issue",
    "returned_provider_clock_reversal",
)
SCALAR_COUNTS = (
    "rows", "equal_clock_groups", "rows_in_equal_clock_groups", "cross_page_equal_clock_groups",
    "locked_quote_rows", "crossed_quote_rows", "quote_rows_with_inactive_side",
    "provider_clock_backwards",
)
MAPPING_COUNTS = ("equal_clock_variation_groups", "trade_update_statuses",
                  "missing_fields", "invalid_schema_values")


def read_pinned(path, name):
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as handle:
        data = handle.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024 or sha(data) != PINS[name]:
        raise ValueError("published_" + name + "_evidence_changed")
    return strict_json(data)


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative_integer_diagnostic_count_required")
    return value


def stream_counts(item):
    values = {name: count(item[name]) for name in SCALAR_COUNTS}
    for name in MAPPING_COUNTS:
        if not isinstance(item[name], dict):
            raise ValueError("diagnostic_count_mapping_required")
        values[name] = {key: count(value) for key, value in item[name].items()}
    groups, rows = values["equal_clock_groups"], values["rows"]
    if not 2 * groups <= values["rows_in_equal_clock_groups"] <= rows:
        raise ValueError("impossible_equal_clock_counts")
    if values["cross_page_equal_clock_groups"] > groups or any(
            value > groups for value in values["equal_clock_variation_groups"].values()):
        raise ValueError("impossible_equal_clock_counts")
    if any(values[name] > rows for name in SCALAR_COUNTS if name != "rows") or any(
            value > rows for name in MAPPING_COUNTS if name != "equal_clock_variation_groups"
            for value in values[name].values()):
        raise ValueError("diagnostic_count_exceeds_observed_rows")
    return values


def diagnose(plan, ledger, acquisition, audit):
    """Only propagate observed diagnostics; no unflagged identity is source-qualified."""
    expected = plan["identity_count"]
    descriptors = plan["full_census"]
    if len(ledger) != 2 * expected or len(descriptors) != len(acquisition["pairs"]) or (
            len(descriptors) != len(audit["pairs"])):
        raise ValueError("original_denominator_changed")
    if audit["census_replay_sha256"] != sha(canonical(acquisition)) or (
            audit["manifest_sha256"] != acquisition["manifest_sha256"]):
        raise ValueError("published_audit_acquisition_link_changed")
    for item in (acquisition, audit):
        if item["source_ready"] is not False or item["new_economic_labels"] != 0 or (
                item["profitability_validation_complete"] is not False):
            raise ValueError("technical_evidence_cannot_promote_economic_status")
        if item["original_identity_denominator"] != expected or (
                item["original_partition_stream_denominator"] != len(descriptors)):
            raise ValueError("original_denominator_changed")
    identities, linked = [], defaultdict(dict)
    for index in range(expected):
        trade, quote = ledger[2 * index:2 * index + 2]
        keys = ("scope", "trading_day", "ticker", "hot_t", "start_ns", "end_ns")
        if trade["stream"] != "trades" or quote["stream"] != "quotes" or any(
                trade[key] != quote[key] for key in keys):
            raise ValueError("original_identity_stream_pair_changed")
        identities.append({"original_identity_index": index, **{key: trade[key] for key in keys}})
    for index, (descriptor, acquired, audited) in enumerate(zip(
            descriptors, acquisition["pairs"], audit["pairs"], strict=True)):
        for item in (acquired, audited):
            if type(item["original_full_census_index"]) is not int or (
                    item["original_full_census_index"] != index) or any(
                    item[key] != descriptor[key] for key in
                    ("identity_indices", "stream", "ticker", "trading_day")):
                raise ValueError("original_partition_identity_link_changed")
            if item["source_ready"] is not False or item["economic_label"] is not None:
                raise ValueError("technical_evidence_cannot_promote_economic_status")
        if any(acquired[key] != descriptor[key] for key in ("start_ns", "end_ns")):
            raise ValueError("original_partition_window_changed")
        if acquired["collection_state"] != audited["collection_state"] or (
                acquired["rows_observed"] != audited["rows"]):
            raise ValueError("published_row_count_or_state_changed")
        if audited["fill_confirmed"] is not False or (
                audited["source_order_is_execution_sequence"] is not False):
            raise ValueError("technical_evidence_cannot_promote_economic_status")
        values = stream_counts(audited)
        stream = descriptor["stream"]
        if stream not in ("trades", "quotes"):
            raise ValueError("original_tick_stream_required")
        for identity_index in descriptor["identity_indices"]:
            if type(identity_index) is not int or not 0 <= identity_index < expected:
                raise ValueError("original_identity_index_required")
            identity = identities[identity_index]
            if stream in linked[identity_index] or any(
                    identity[key] != descriptor[key] for key in ("ticker", "trading_day")) or not (
                    descriptor["start_ns"] <= identity["start_ns"] < identity["end_ns"] <= descriptor["end_ns"]):
                raise ValueError("original_identity_window_link_changed")
            linked[identity_index][stream] = {"partition_index": index,
                "collection_state": acquired["collection_state"], **values}
    totals, scopes, days, combinations = Counter(), defaultdict(Counter), defaultdict(Counter), Counter()
    for identity in identities:
        streams = linked[identity["original_identity_index"]]
        if set(streams) != {"trades", "quotes"}:
            raise ValueError("every_original_identity_requires_both_streams")
        trade, quote = streams["trades"], streams["quotes"]
        observed = {
            "equal_provider_clock": any(s["equal_clock_groups"] for s in streams.values()),
            "equal_clock_price_or_size_field_variation": any(
                s["equal_clock_variation_groups"].get(key, 0) for s in streams.values()
                for key in ("price", "size")),
            "cross_page_equal_clock": any(s["cross_page_equal_clock_groups"] for s in streams.values()),
            "locked_quote": quote["locked_quote_rows"] > 0,
            "crossed_quote": quote["crossed_quote_rows"] > 0,
            "inactive_quote_side": quote["quote_rows_with_inactive_side"] > 0,
            "trade_update_present": any(value for key, value in trade["trade_update_statuses"].items()
                                        if key != "absent"),
            "empty_trade_or_quote_query": any(s["rows"] == 0 for s in streams.values()),
            "required_field_or_schema_issue": any(value for s in streams.values()
                for name in ("missing_fields", "invalid_schema_values") for value in s[name].values()),
            "returned_provider_clock_reversal": any(s["provider_clock_backwards"] for s in streams.values()),
        }
        flags = [key for key in FLAGS if observed[key]]
        identity.update(stream_diagnostics=streams, observed_flags=flags, source_ready=False,
                        economic_label=None, execution_resolution="NOT_EVALUATED")
        for group in (totals, scopes[identity["scope"]], days[identity["trading_day"]]):
            group["identities"] += 1
            group["with_observed_flags"] += int(bool(flags))
            group["without_these_observed_flags_source_still_unqualified"] += int(not flags)
            group.update(flags)
        combinations["|".join(flags) or "NO_LISTED_DIAGNOSTIC_OBSERVED"] += 1
    def counts(group):
        return {key: group[key] for key in ("identities", "with_observed_flags",
                    "without_these_observed_flags_source_still_unqualified", *FLAGS)}
    return {"request_id": 335, "stage": "published_diagnostics_original_identity_bottleneck_analysis",
        "evidence_level": "hash_pinned_published_diagnostics_not_current_raw_replay",
        "source_plan_sha256": PLAN_SHA, "input_uncompressed_sha256": PINS,
        "original_identity_denominator": expected, "original_partition_stream_denominator": len(descriptors),
        "row_diagnostics_scope": "whole_original_partition_not_decision_instant_or_unique_market_events",
        "equal_clock_variation": "exact_decoded_field_fingerprints_including_type_and_Decimal_text",
        "unobserved_flags_do_not_prove_no_arrival_or_correction_ambiguity": True,
        "totals": counts(totals), "scope_totals": {key: counts(value) for key, value in sorted(scopes.items())},
        "day_totals": {key: counts(value) for key, value in sorted(days.items())},
        "observed_flag_combinations": dict(sorted(combinations.items())), "identities": identities,
        "raw_replay_performed_this_analysis": False, "source_ready": False,
        "scientifically_qualified_identities": 0, "new_economic_labels": 0,
        "execution_resolution": "NOT_EVALUATED", "profitability_validation_complete": False,
        "additional_market_HTTP_attempts": 0, "additional_cost_USD": 0,
        "june": "HOLD", "july_august": "SEALED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, default=Path("research"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.research / "request335-free-source-plan.json.gz")
    ledger = read_pinned(args.research / "request335-input-ledger.json.gz", "ledger")
    acquisition = read_pinned(args.research / "request335-census-result-20261011.json.gz", "acquisition")
    audit = read_pinned(args.research / "request335-census-audit-20261011.json.gz", "audit")
    if acquisition["manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("registered_manifest_changed")
    result = diagnose(plan, ledger, acquisition, audit)
    if result["scope_totals"]["training"]["identities"] != 407 or (
            result["scope_totals"]["evaluation"]["identities"] != 828):
        raise ValueError("original_training_development_denominators_changed")
    data = canonical(result)
    args.output.write_bytes(gzip.compress(data, mtime=0) if args.output.suffix == ".gz" else data)
    print(sha(data))


if __name__ == "__main__":
    main()
