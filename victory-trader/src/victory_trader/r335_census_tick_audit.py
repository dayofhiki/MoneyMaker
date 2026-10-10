"""Offline all-original-window census audit; counts never qualify execution."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .offline_source_inventory import canonical, confined, sha, strict_json
from .r335_alpaca_qualification import checked_payload, load_plan
from .r335_census_acquisition import REUSED, TERMINAL, replay
from .r335_tick_semantics_audit import audit_rows
from .tick_execution_readiness import rfc3339_ns


def audit_evidence(plan, root, progress=None):
    verified = replay(plan, root)
    manifest = strict_json(confined(root, "manifest.json").read_bytes())
    pilot = strict_json(confined(root / "pilot", "manifest.json").read_bytes())
    pairs, totals, days = [], defaultdict(Counter), defaultdict(Counter)
    cross_stream, both_terminal, prior = 0, 0, None
    for index, pair in enumerate(manifest["pairs"]):
        request = pair["request"]
        raw_root, raw_pair = (root / "pilot", pilot["pairs"][pair["pilot_index"]]) if (
            pair["pilot_index"] is not None) else (root, pair)
        clocks = set()
        def rows():
            for page_index, page in enumerate(raw_pair["pages"]):
                if page["observation"] is not None:
                    decoded = checked_payload(confined(raw_root, page["wire_file"]).read_bytes())
                    for row in decoded[request["stream"]].get(request["ticker"], []):
                        clocks.add(rfc3339_ns(row["t"]))
                        yield page_index, row
        audited = audit_rows(rows(), request["stream"])
        if audited["rows"] != verified["pairs"][index]["rows_observed"]:
            raise ValueError("audit_and_authenticated_replay_rows_disagree")
        item = {"original_full_census_index": index, "identity_indices": request["identity_indices"],
                "trading_day": request["trading_day"], "ticker": request["ticker"],
                "stream": request["stream"], "collection_state": pair["state"], **audited}
        pairs.append(item)
        total, day = totals[request["stream"]], days[request["trading_day"]]
        for name in ("rows", "provider_clock_backwards", "equal_clock_groups",
                     "rows_in_equal_clock_groups", "cross_page_equal_clock_groups",
                     "identical_decoded_rows", "locked_quote_rows", "crossed_quote_rows",
                     "quote_rows_with_inactive_side", "quote_condition_side_mapping_unresolved_rows"):
            total[name] += audited[name]
        for name in ("missing_fields", "invalid_schema_values", "additional_fields",
                     "size_units", "trade_update_statuses", "fractional_digits_observed",
                     "equal_clock_variation_groups"):
            for key, count in audited[name].items():
                total[name + ":" + key] += count
        day["partitions"] += 1
        day[request["stream"] + "_partition_row_observations"] += audited["rows"]
        day["api_terminal_partitions"] += int(pair["state"] in {REUSED, TERMINAL})
        key = (tuple(request["identity_indices"]), request["trading_day"], request["ticker"],
               request["start_ns"], request["end_ns"])
        if index % 2 == 0:
            if request["stream"] != "trades":
                raise ValueError("original_trade_quote_pair_order_required")
            prior = (key, clocks, pair["state"] in {REUSED, TERMINAL})
        else:
            if request["stream"] != "quotes" or prior[0] != key:
                raise ValueError("exact_original_paired_window_required")
            cross_stream += len(prior[1] & clocks)
            complete = prior[2] and pair["state"] in {REUSED, TERMINAL}
            both_terminal += int(complete)
            day["both_streams_api_terminal_identities"] += int(complete)
            prior = None
        if progress is not None and (index + 1) % 50 == 0:
            progress(index + 1, len(manifest["pairs"]))
    return {"request_id": 335, "stage": "offline_original_census_provider_semantics_audit_only",
            "manifest_sha256": verified["manifest_sha256"],
            "census_replay_sha256": sha(canonical(verified)),
            "audit_implementation_sha256": sha(Path(__file__).read_bytes()),
            "semantics_implementation_sha256": sha(Path(__file__).with_name("r335_tick_semantics_audit.py").read_bytes()),
            "adapter_implementation_sha256": sha(Path(__file__).with_name("tick_execution_readiness.py").read_bytes()),
            "original_identity_denominator": 1235, "original_partition_stream_denominator": 2470,
            "both_streams_api_terminal_identities": both_terminal,
            "scientifically_qualified_identities": 0, "scientifically_qualified_partitions": 0,
            "cross_stream_equal_provider_clock_groups": cross_stream,
            "row_count_unit": "partition_observations_including_overlapping_original_windows_not_unique_market_events",
            "stream_totals": {k: dict(sorted(v.items())) for k, v in sorted(totals.items())},
            "day_totals": {k: dict(sorted(v.items())) for k, v in sorted(days.items())},
            "pairs": pairs, "additional_market_HTTP_attempts": 0, "additional_cost_USD": 0,
            "source_ready": False, "new_economic_labels": 0, "profitability_validation_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("research/request335-free-source-plan.json.gz"))
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_evidence(load_plan(args.plan), args.evidence,
                            progress=lambda done, total: print(f"audited_partitions={done}/{total}", flush=True))
    args.output.write_bytes(canonical(result))


if __name__ == "__main__":
    main()
