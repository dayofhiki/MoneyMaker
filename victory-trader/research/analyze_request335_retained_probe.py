"""Inspect retained first-date fields/integrity without creating economic labels."""
import argparse
import collections
import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    reference_bytes = args.reference.read_bytes()
    reference = json.loads(gzip.decompress(reference_bytes))
    day = min(p["trading_day"] for p in reference["pairs"])
    expected = {p["original_full_census_index"]: p for p in reference["pairs"]
                if p["trading_day"] == day}
    counts = collections.Counter()
    fields = collections.defaultdict(collections.Counter)
    rows, pages = collections.Counter(), collections.Counter()
    attempts, raw_summaries, names = {}, {}, set()
    # Stream the gzip once. Random extractfile lookups would repeatedly decompress
    # the archive as reads alternate between the attempts and pages directories.
    with tarfile.open(args.archive, mode="r|gz") as tar:
        for member in tar:
            if member.name in names or not member.isfile():
                raise ValueError("regular_unique_archive_members_required")
            names.add(member.name)
            if member.name.startswith("attempts/"):
                attempt = json.load(tar.extractfile(member))
                if attempt["http_status"] != 200 or not attempt["wire_complete"]:
                    raise ValueError("complete_raw_response_required")
                if attempt["wire_file"] in attempts:
                    raise ValueError("one_attempt_per_retained_response_required")
                attempts[attempt["wire_file"]] = attempt
                continue
            match = re.fullmatch(r"pages/(\d+)-\d+\.wire", member.name)
            if not match:
                continue
            body = tar.extractfile(member).read()
            index = int(match.group(1))
            stream = expected[index]["stream"]
            items = json.loads(body)[stream]
            observations = [r for v in items.values() for r in v] if isinstance(items, dict) else items
            raw_summaries[member.name] = {
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body), "rows": len(observations),
            }
            for row in observations:
                fields[stream].update(row.keys())
            rows[index] += len(observations)
            pages[index] += 1
            counts[stream + "_rows"] += len(observations)
            counts["wire_bytes"] += len(body)
            counts["wire_files_verified"] += 1
    if set(attempts) != set(raw_summaries):
        raise ValueError("raw_attempt_membership_differs")
    for name, attempt in attempts.items():
        raw = raw_summaries[name]
        if raw["sha256"] != attempt["wire_sha256"] or raw["bytes"] != attempt["body_bytes"]:
            raise ValueError("raw_digest_mismatch")
        if raw["rows"] != attempt["observation"]["rows"]:
            raise ValueError("internal_row_count_mismatch")
    if set(rows) != set(expected):
        raise ValueError("original_partition_membership_changed")
    for index, partition in expected.items():
        if (rows[index] != partition["rows_observed"] or
                pages[index] != partition["pages_observed"]):
            raise ValueError("published_partition_counts_differ")
    all_fields = set().union(*(set(f) for f in fields.values()))
    result = {
        "request_id": 335, "stage": "retained_first_date_internal_integrity_and_field_audit",
        "sample_day": day, "selection": "first_original_date_not_selected_by_outcome",
        "reference_ledger_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "archive_bytes": args.archive.stat().st_size,
        "observed_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "external_original_archive_digest_pin_available": False,
        "archive_members": len(names), "archive_scan_passes": 1,
        "original_stream_partitions": len(expected),
        "original_identities": len(expected) // 2,
        "all_raw_body_digests_equal_internal_attempt_records": True,
        "all_partition_page_row_counts_equal_published_recovery_ledger": True,
        "counts": dict(counts),
        "field_presence": {s: dict(sorted(v.items())) for s, v in fields.items()},
        "named_historical_SIP_publication_clock_field_found": "sip_timestamp" in all_fields,
        "named_sequence_field_found": bool(all_fields & {"sequence_number", "sequence", "seq"}),
        "named_receive_clock_field_found": bool(all_fields & {"ts_recv", "receive_timestamp", "receipt_timestamp"}),
        "original_receipt_correction_event_history_established": False,
        "full_original_census_replayed": False, "source_ready": False,
        "new_economic_labels": 0, "new_fits": 0, "new_market_requests": 0,
        "additional_cost_USD": 0,
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"sample_day": day, "counts": dict(counts), "source_ready": False}))


if __name__ == "__main__":
    main()
