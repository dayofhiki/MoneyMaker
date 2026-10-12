"""Reproduce the all-identity day-shard proposal offline; never collect data."""
from __future__ import annotations

import argparse
import gzip
from collections import Counter
from pathlib import Path

from victory_trader.offline_source_inventory import canonical, sha
from victory_trader.r335_alpaca_qualification import load_plan


def proposal(plan):
    shards = []
    for day in sorted({r["trading_day"] for r in plan["full_census"]}):
        indexed = [(i, r) for i, r in enumerate(plan["full_census"]) if r["trading_day"] == day]
        shards.append({
            "trading_day": day, "original_full_census_indices": [i for i, _ in indexed],
            "original_descriptors_sha256": sha(canonical([r for _, r in indexed])),
            "identity_count": len({j for _, r in indexed for j in r["identity_indices"]}),
            "partition_stream_count": len(indexed),
            "trades_partitions": sum(r["stream"] == "trades" for _, r in indexed),
            "quotes_partitions": sum(r["stream"] == "quotes" for _, r in indexed),
            "initial_state": "NOT_COLLECTED_BY_THIS_PLAN"})
    indices = [i for s in shards for i in s["original_full_census_indices"]]
    if (Counter(indices) != Counter(range(2470)) or
            sum(s["identity_count"] for s in shards) != 1235 or
            sum(s["identity_count"] for s in shards if s["trading_day"] <= "2026-05-08") != 407 or
            sum(s["identity_count"] for s in shards if s["trading_day"] >= "2026-05-11") != 828):
        raise ValueError("original_shard_identity_denominator_changed")
    return {"request_id": 335,
            "stage": "offline_all_identity_day_sharding_proposal_not_collection_registration",
            "source_plan_sha256": sha(canonical(plan)), "original_identity_denominator": 1235,
            "original_partition_stream_denominator": 2470, "training_identity_count": 407,
            "reused_development_identity_count": 828,
            "shard_order": "original_trading_day_ascending_no_outcome_or_density_selection",
            "shards": shards, "every_original_partition_in_exactly_one_shard": True,
            "market_http_attempts": 0, "additional_cost_authorized_USD": 0,
            "new_economic_labels": 0, "collection_enabled": False, "raw_reuse_enabled": False,
            "source_ready": False, "profitability_validation_complete": False,
            "budget_required_before_collection": [
                "per_shard_and_global_http_page_byte_wall_disk_limits",
                "failure_page_cap_and_no_automatic_retry_resume_policy",
                "private_raw_retention_and_authenticated_restore",
                "clock_condition_correction_and_ordering_qualification_before_economic_use"],
            "sealed_dates_opened": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("research/request335-free-source-plan.json.gz"))
    parser.add_argument("--registered", type=Path,
                        default=Path("research/request335-full-census-shards-20261011.json.gz"))
    args = parser.parse_args()
    result = canonical(proposal(load_plan(args.plan)))
    compressed = args.registered.read_bytes()
    if gzip.decompress(compressed) != result or gzip.compress(result, mtime=0) != compressed:
        raise ValueError("registered_all_identity_shard_plan_changed")
    print(canonical({"plan_sha256": sha(result), "gzip_sha256": sha(compressed),
                     "all_2470_indices_exactly_once": True, "market_http_attempts": 0,
                     "collection_enabled": False}).decode(), end="")


if __name__ == "__main__":
    main()
