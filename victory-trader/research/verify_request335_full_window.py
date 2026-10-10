"""Offline aggregate audit for the registered full-window pilot; no network."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from victory_trader.offline_source_inventory import canonical, confined, sha, strict_json
from victory_trader.r335_alpaca_qualification import checked_payload, load_plan, replay as minute_replay
from victory_trader.r335_full_window_qualification import replay
from victory_trader.r335_tick_semantics_audit import audit_rows, row_key
from victory_trader.tick_execution_readiness import rfc3339_ns


def audit(plan, root):
    verified = replay(plan, root)
    manifest_bytes = confined(root, "manifest.json").read_bytes()
    manifest = strict_json(manifest_bytes)
    pairs, clocks = [], defaultdict(dict)
    for pair in manifest["pairs"]:
        request = pair["request"]
        page_rows = []
        for index, page in enumerate(pair["pages"]):
            if page["observation"] is not None:
                payload = checked_payload(confined(root, page["wire_file"]).read_bytes())
                page_rows.extend((index, row) for row in payload[request["stream"]].get(request["ticker"], []))
        pairs.append({"ticker": request["ticker"], "trading_day": request["trading_day"],
                      "stream": request["stream"], "collection_state": pair["state"],
                      **audit_rows(page_rows, request["stream"])})
        clocks[(request["trading_day"], request["ticker"])][request["stream"]] = {
            rfc3339_ns(row["t"]) for _, row in page_rows}
    return {"request_id": 335, "stage": "offline_full_window_provider_semantics_audit_only",
            "input_manifest_sha256": sha(manifest_bytes), "replay_sha256": sha(canonical(verified)),
            "wrapper_sha256": sha(Path(__file__).read_bytes()),
            "audit_implementation_sha256": sha(Path(audit_rows.__code__.co_filename).read_bytes()),
            "adapter_implementation_sha256": sha(Path(rfc3339_ns.__code__.co_filename).read_bytes()),
            "pairs": pairs, "cross_stream_equal_provider_clock_groups": sum(
                len(c.get("trades", set()) & c.get("quotes", set())) for c in clocks.values()),
            "additional_market_HTTP_attempts": 0, "additional_cost_USD": 0,
            "source_ready": False, "new_economic_labels": 0,
            "full_census_collected": False, "profitability_validation_complete": False}


def compare_prefixes(plan, original, fresh):
    minute_replay(plan, original)
    replay(plan, fresh)
    old_bytes = confined(original, "manifest.json").read_bytes()
    if sha(old_bytes) != "44b5f124d1ad2533dc42a800d9032c64fbffae3d7b7c5f1938db0b88e8a20507":
        raise ValueError("registered_original_sample_manifest_required")
    new_bytes = confined(fresh, "manifest.json").read_bytes()
    pairs = []
    for old, new in zip(strict_json(old_bytes)["pairs"], strict_json(new_bytes)["pairs"]):
        request = old["request"]
        def rows(pair, root):
            return [row for p in pair["pages"] if p["observation"] is not None
                    for row in checked_payload(confined(root, p["wire_file"]).read_bytes())[request["stream"]].get(request["ticker"], [])]
        a, all_b = rows(old, original), rows(new, fresh)
        b = [r for r in all_b if rfc3339_ns(r["t"]) < request["end_ns"]]
        state = "NOT_COMPARABLE_NO_SUCCESSFUL_FRESH_PAGE"
        if all_b:
            matched = len(b) >= len(a) and all(row_key(x) == row_key(y) for x, y in zip(a, b))
            state = "MATCHED_ORIGINAL_ORDERED_PREFIX" if matched else "DIFFERENT_ORIGINAL_ORDERED_PREFIX"
        changed = Counter()
        for x, y in zip(a, b):
            changed.update(k for k in set(x) | set(y) if row_key(x.get(k)) != row_key(y.get(k)))
        pairs.append({"trading_day": request["trading_day"], "ticker": request["ticker"],
                      "stream": request["stream"], "original_rows": len(a),
                      "fresh_rows_within_old_window": len(b), "comparison_state": state,
                      "changed_fields_at_original_ordinals": dict(changed),
                      "original_terminal": old["state"].startswith("TERMINAL"),
                      "fresh_terminal": new["state"].startswith("TERMINAL")})
    return {"request_id": 335, "stage": "offline_cross_query_prefix_comparison_only",
            "original_manifest_sha256": sha(old_bytes), "fresh_manifest_sha256": sha(new_bytes),
            "wrapper_sha256": sha(Path(__file__).read_bytes()), "pairs": pairs,
            "comparison_does_not_establish_correction_delivery_chronology": True,
            "source_ready": False, "new_economic_labels": 0, "additional_market_HTTP_attempts": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("research/request335-free-source-plan.json.gz"))
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-evidence", type=Path)
    parser.add_argument("--prefix-output", type=Path)
    args = parser.parse_args()
    result = canonical(audit(load_plan(args.plan), args.evidence))
    with args.output.open("xb") as handle:
        handle.write(result)
    print(sha(result))
    if args.original_evidence is not None:
        if args.prefix_output is None:
            parser.error("prefix output required with original evidence")
        prefix = canonical(compare_prefixes(load_plan(args.plan), args.original_evidence, args.evidence))
        with args.prefix_output.open("xb") as handle:
            handle.write(prefix)
        print(sha(prefix))


if __name__ == "__main__":
    main()
