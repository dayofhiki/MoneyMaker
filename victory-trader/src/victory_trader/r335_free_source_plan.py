"""Build fixed-census source request descriptors offline; no network or secrets."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .offline_source_inventory import canonical, fixed_windows, sha, strict_json
from .tick_execution_readiness import format_ns


def request_descriptor(day: str, ticker: str, stream: str, start: int, end: int,
                       limit: int) -> dict:
    if stream not in ("trades", "quotes") or not start < end:
        raise ValueError("explicit_tick_stream_and_half_open_window_required")
    return {"trading_day": day, "ticker": ticker, "stream": stream,
            "host": "data.alpaca.markets", "path": f"/v2/stocks/{stream}",
            "start_ns": start, "end_ns": end,
            "params": {"symbols": ticker, "start": format_ns(start),
                       "end": format_ns(end-1), "feed": "sip", "asof": "-",
                       "currency": "USD", "sort": "asc", "limit": limit}}


def build_plan(windows: list[dict], original_plan: dict) -> dict:
    """Union only overlapping same-day/ticker windows; preserve every identity.

    The actual CLI authenticates the original census and plan before calling.
    Pagination/byte caps are stop limits, not evidence of stream completeness.
    None of these descriptors is authorization to collect or bill an API.
    """
    groups = defaultdict(list)
    for index, row in enumerate(windows):
        groups[(row["trading_day"], row["ticker"])].append((index, row))
    full = []
    for (day, ticker), identities in sorted(groups.items()):
        # Original windows all end at the same session close. Reject a changed
        # geometry rather than downloading irrelevant disjoint intervals.
        ends = {row["end_ns"] for _, row in identities}
        if len(ends) != 1:
            raise ValueError("original_same_close_geometry_required")
        start, end = min(row["start_ns"] for _, row in identities), ends.pop()
        for stream in ("trades", "quotes"):
            descriptor = request_descriptor(day, ticker, stream, start, end, 10000)
            descriptor["identity_indices"] = [index for index, _ in identities]
            full.append(descriptor)
    expected = []
    for day in sorted({row["trading_day"] for row in windows}):
        row = min((w for w in windows if w["trading_day"] == day),
                  key=lambda w: (w["trading_day"], w["ticker"], w["hot_t"]))
        expected.append({"trading_day": day, "ticker": row["ticker"], "hot_t": row["hot_t"],
                         "scope": row["scope"], "start_ms": row["start_ns"]//1000000,
                         "end_ms": min(row["start_ns"]+60000000000, row["end_ns"])//1000000})
    if expected != original_plan["qualification_windows"]:
        raise ValueError("original_qualification_selection_changed")
    qualification = [request_descriptor(row["trading_day"], row["ticker"], stream,
                                        row["start_ms"]*1000000, row["end_ms"]*1000000, 100)
                     for row in expected for stream in ("trades", "quotes")]
    return {"request_id": 335, "stage": "offline_zero_cost_source_planning_only",
            "original_windows_sha256": sha(canonical(windows)), "identity_count": len(windows),
            "qualification": qualification, "full_census": full,
            "full_identity_streams": len(windows)*2, "full_unique_partition_streams": len(full),
            "limits_for_proposed_qualification": {"http_attempts": 48, "pages_per_pair": 2,
                                                  "bytes_per_page": 2097152,
                                                  "total_response_bytes": 33554432,
                                                  "minimum_interval_seconds": 12.5,
                                                  "timeout_seconds": 30, "retries": 0},
            "collection_enabled": False, "additional_cost_authorized_USD": 0,
            "market_http_attempts": 0, "source_completeness_verified": False,
            "new_economic_labels": 0, "profitability_validation_complete": False,
            "requires_before_collection": ["authorized_Basic_account_and_license",
                                            "independently_registered_collection_contract",
                                            "timestamp_NBBO_units_corrections_ordering_specification"],
            "full_collection_budget": "UNREGISTERED_DO_NOT_COLLECT"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "pins", "plan", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    windows = fixed_windows(args.previous, args.pins, args.plan)
    result = build_plan(windows, strict_json(args.plan.read_bytes()))
    with args.output.open("xb") as handle:
        handle.write(canonical(result))
    print(sha(canonical(result)))


if __name__ == "__main__":
    main()
