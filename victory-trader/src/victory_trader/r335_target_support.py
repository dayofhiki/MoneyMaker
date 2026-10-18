"""Offline acquisition-scope analysis for the unchanged R330/R331 target.

This does not collect data, qualify a source, build features or compute targets.
The bound is conditional on a qualified source's filter clock matching the
reference clock. It is not a replacement for the full-session R332/R335 audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

ORIGINAL_LEDGER_SHA256 = "315a445c080b7639a0b5fcd5db60bca2824bcc5343d81204fd398b185dd1c795"
NS_PER_MS = 1_000_000
EPISODE_MS, MAX_LAG_MS, BAR_MS = 3_600_000, 3_000, 1_000
KEYS = ("trading_day", "ticker", "hot_t")


def target_end_ns(hot_ms: int, original_end_ns: int) -> int:
    """Include the entire second starting exactly at terminal + 3s.

enter_outcome accepts an exit open at submission+MAX_LAG_MS, even after
HOT+60m. Every submission is <=min(HOT+60m, closing). The final accepted
second therefore ends at HOT+60m+3s+1s (exclusive), bounded by the original
session. Truncating at HOT+60m or +3s would be incorrect.
    """
    if type(hot_ms) is not int or type(original_end_ns) is not int:
        raise ValueError("exact_integer_clock_required")
    if hot_ms < 0 or original_end_ns <= 0:
        raise ValueError("nonnegative_HOT_and_positive_session_end_required")
    return min(original_end_ns, (hot_ms + EPISODE_MS + MAX_LAG_MS + BAR_MS) * NS_PER_MS)


def build_scope(ledger: list[dict]) -> dict:
    """Retain both streams and every identity; inspect no price or payoff."""
    groups = defaultdict(dict)
    windows, by_day = [], defaultdict(lambda: Counter())
    for row in ledger:
        stream = row["stream"]
        key = tuple(row[k] for k in KEYS)
        if stream not in ("trades", "quotes") or stream in groups[key]:
            raise ValueError("unique_trade_quote_pair_required")
        start, end = row["start_ns"], row["end_ns"]
        if type(start) is not int or not 0 <= start < end or start > row["hot_t"] * NS_PER_MS:
            raise ValueError("original_pre_HOT_start_required")
        bounded_end = target_end_ns(row["hot_t"], end)
        groups[key][stream] = (start, end, row["scope"])
        item = {k: row[k] for k in (*KEYS, "scope", "stream", "start_ns")}
        item.update(end_ns=bounded_end, original_end_ns=end)
        windows.append(item)
        counts = by_day[row["trading_day"]]
        counts["stream_partitions"] += 1
        counts["original_interval_ns"] += end - start
        counts["target_interval_ns"] += bounded_end - start
    if not windows:
        raise ValueError("nonempty_original_population_required")
    for pair in groups.values():
        if set(pair) != {"trades", "quotes"} or pair["trades"] != pair["quotes"]:
            raise ValueError("identical_original_trade_quote_geometry_required")
    original = sum(x["original_interval_ns"] for x in by_day.values())
    bounded = sum(x["target_interval_ns"] for x in by_day.values())
    return {
        "request_id": 335, "stage": "offline_fixed_target_support_analysis",
        "identities": len(groups), "stream_partitions": len(windows),
        "scope_identities": dict(Counter(pair["trades"][2] for pair in groups.values())),
        "HOT_at_or_after_session_end_identities": sum(
            key[2] * NS_PER_MS >= pair["trades"][1] for key, pair in groups.items()
        ),
        "original_interval_ns": original, "target_interval_ns": bounded,
        "interval_reduction_pct": 100 * (original - bounded) / original,
        "per_date": dict(sorted(by_day.items())), "windows": windows,
        "applicability": "fixed_R330_R331_immediate_target_reference_support_only",
        "full_session_census_replaced": False, "feature_rebuild_authorized": False,
        "query_filter_clock_must_match_reference_clock": True,
        "quote_initial_state_and_correction_lookback_require_separate_coverage": True,
        "actual_row_byte_request_or_runtime_reduction_established": False,
        "collection_enabled": False, "source_ready": False,
        "new_market_requests": 0, "new_economic_labels": 0, "new_fits": 0,
        "additional_cost_USD": 0, "original_resolution_gate_pct": 90,
        "prior_primary_resolution_pct": 34.8352,
        "june": "HOLD", "july_august": "SEALED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.ledger.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_LEDGER_SHA256:
        raise ValueError("original_published_ledger_digest_changed")
    result = build_scope(json.loads(gzip.decompress(raw)))
    if (result["identities"], result["stream_partitions"], result["scope_identities"]) != (
        1235, 2470, {"training": 407, "evaluation": 828}
    ):
        raise ValueError("original_census_changed")
    result["input_ledger_sha256"] = ORIGINAL_LEDGER_SHA256
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with args.output.open("xb") as handle:
        handle.write(gzip.compress(payload, mtime=0))
    print(json.dumps({k: v for k, v in result.items() if k not in ("windows", "per_date")}, sort_keys=True))


if __name__ == "__main__":
    main()
