"""R335 preparation: explicit offline exports; never infer source completeness."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

STREAMS = ("trades", "quotes")
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_RECORD_BYTES = 1024 * 1024
KEYS = ("trading_day", "ticker", "hot_t")
ORIGINAL_PINS_SHA256 = "6e058f72bf12dfc08c8c9419e2ceb1a36fd66c441435bf5540615b14c65a6eb6"
ORIGINAL_PLAN_SHA256 = "3f56e490a8a5e79e4871107dd8889f3266f39ba4e212baf0aa44f898ba2aa907"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def strict_json(data):
    def bad_constant(value):
        raise ValueError("nonfinite_JSON")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_JSON_key")
            result[key] = value
        return result
    return json.loads(data, parse_constant=bad_constant, object_pairs_hook=unique_object)


def integer(value, csv_input=False):
    if csv_input and isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return int(value)
    if not csv_input and type(value) is int:
        return value
    raise ValueError("integer_nanosecond_clock_required")


def confined(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("explicit_relative_file_required")
    p = Path(relative)
    if p.is_absolute() or any(x in ("..", ".") or x.startswith(".") for x in p.parts):
        raise ValueError("unsafe_export_path")
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink_export_path")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("export_path_outside_root")
    return current


def read_hashed(root, path, digest, budget):
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("explicit_SHA256_required")
    p = confined(root, path)
    if not p.is_file():
        raise ValueError("regular_export_file_required")
    size = p.stat().st_size
    if size > MAX_FILE_BYTES or budget[0] + size > MAX_TOTAL_BYTES:
        raise ValueError("offline_byte_budget_exceeded")
    with p.open("rb") as handle:
        data = handle.read(MAX_FILE_BYTES + 1)
    budget[0] += len(data)
    if len(data) > MAX_FILE_BYTES or budget[0] > MAX_TOTAL_BYTES:
        raise ValueError("offline_byte_budget_exceeded")
    if sha(data) != digest:
        raise ValueError("source_digest_mismatch")
    return data


def inspect_export(data, entry):
    fmt = entry.get("format")
    if entry.get("timestamp_unit") != "nanosecond" or fmt not in ("rest_jsonl", "provider_csv"):
        raise ValueError("explicit_supported_schema_required")
    start, end = integer(entry["start_ns"]), integer(entry["end_ns"])
    if start >= end:
        raise ValueError("invalid_export_bounds")
    if any(len(line) > MAX_RECORD_BYTES for line in data.splitlines()):
        raise ValueError("offline_record_budget_exceeded")
    if fmt == "rest_jsonl":
        rows = (strict_json(line) for line in data.splitlines() if line.strip())
    else:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields) or "sip_timestamp" not in fields:
            raise ValueError("invalid_CSV_header")
        rows = reader
    count, previous, first, last, repeated, backwards = 0, None, None, None, 0, 0
    seen = set()
    optional = ["participant_timestamp", "trf_timestamp", "conditions", "sequence_number"]
    optional += ["price", "size", "exchange", "correction", "id"] if entry["stream"] == "trades" else ["bid_price", "ask_price", "bid_size", "ask_size", "bid_exchange", "ask_exchange"]
    missing = Counter({field: 0 for field in optional})
    for row in rows:
        if not isinstance(row, dict) or None in row:
            raise ValueError("invalid_record_schema")
        if row.get("ticker", entry["ticker"]) != entry["ticker"]:
            raise ValueError("mixed_ticker_export")
        event = integer(row.get("sip_timestamp"), fmt == "provider_csv")
        if not start <= event < end:
            raise ValueError("record_outside_declared_half_open_window")
        for field in ("participant_timestamp", "trf_timestamp"):
            if row.get(field) not in (None, ""):
                integer(row[field], fmt == "provider_csv")
        for field in optional:
            missing[field] += int(row.get(field) in (None, ""))
        repeated += int(event in seen)
        backwards += int(previous is not None and event < previous)
        seen.add(event)
        previous = event
        first = event if first is None else min(first, event)
        last = event if last is None else max(last, event)
        count += 1
    return {"rows": count, "first_SIP_ns": first, "last_SIP_ns": last,
            "duplicate_SIP_timestamps": repeated, "out_of_order_pairs": backwards,
            "missing_fields": dict(missing), "empty_export_observed": count == 0}


def fixed_windows(previous, pins_path, plan_path):
    """Use the original authenticated identity ledgers, not a replacement sample."""
    pins_bytes, plan_bytes = pins_path.read_bytes(), plan_path.read_bytes()
    if sha(pins_bytes) != ORIGINAL_PINS_SHA256 or sha(plan_bytes) != ORIGINAL_PLAN_SHA256:
        raise ValueError("original_input_contract_changed")
    pins, plan = strict_json(pins_bytes), strict_json(plan_bytes)
    for name, digest in pins.items():
        if sha((previous / name).read_bytes()) != digest:
            raise ValueError("original_source_changed")
    if (previous / "official332/source-sha.txt").read_text().strip() != plan["source_artifact_sha"]:
        raise ValueError("original_source_SHA_changed")
    import pandas as pd
    from .action_target_support_census import EVAL_DAYS, TRAIN_DAYS
    from .pending_exit_integrity import session_limits

    windows = []
    for scope, days, expected in (("training", TRAIN_DAYS, 407), ("evaluation", EVAL_DAYS, 828)):
        frame = pd.read_parquet(previous / f"official332/request332-{scope}-source-ledger.parquet", columns=list(KEYS))
        if len(frame) != expected or set(frame.trading_day) != set(days) or frame.duplicated(list(KEYS)).any():
            raise ValueError("original_full_census_required")
        for r in frame.itertuples():
            opening, closing = session_limits(r.trading_day)
            windows.append({"trading_day": r.trading_day, "ticker": r.ticker, "hot_t": int(r.hot_t),
                            "scope": scope, "start_ns": max(opening, int(r.hot_t)-60000)*1000000,
                            "end_ns": closing*1000000})
    return sorted(windows, key=lambda r: tuple(r[k] for k in KEYS))


def inventory(windows, catalog, root):
    """Hash/schema checks do not authenticate licensing or page-chain completeness."""
    if not isinstance(catalog, list):
        raise ValueError("explicit_catalog_list_required")
    allowed = {(w["trading_day"], w["ticker"]) for w in windows}
    if len({tuple(w[k] for k in KEYS) for w in windows}) != len(windows):
        raise ValueError("duplicate_plan_identity")
    entries, observations, budget = {}, {}, [0]
    for entry in catalog:
        if not isinstance(entry, dict):
            raise ValueError("invalid_catalog_entry")
        key = (entry.get("trading_day"), entry.get("ticker"), entry.get("stream"))
        if key[:2] not in allowed or key[2] not in STREAMS or key in entries:
            raise ValueError("unexpected_or_duplicate_source_partition")
        entries[key] = entry
        try:
            if not isinstance(entry.get("source_ref"), str) or not entry["source_ref"].strip():
                raise ValueError("source_provenance_required")
            related = [w for w in windows if (w["trading_day"], w["ticker"]) == key[:2]]
            # Limit declared exports to these same regular-session May windows.
            if integer(entry["start_ns"]) < min(w["start_ns"] for w in related) or integer(entry["end_ns"]) > max(w["end_ns"] for w in related):
                raise ValueError("export_bounds_outside_fixed_scope")
            read_hashed(root, entry["provenance_file"], entry["provenance_sha256"], budget)
            data = read_hashed(root, entry["file"], entry["sha256"], budget)
            observation = inspect_export(data, entry)
            observation.update(state="SOURCE_COMPLETENESS_UNVERIFIED", error_type=None,
                               source_sha256=sha(data), provenance_sha256=entry["provenance_sha256"])
        except (ValueError, KeyError, TypeError, OSError, UnicodeError) as error:
            # Never echo untrusted paths, source references, rows or exception text.
            observation = {"state": "INVALID_OR_UNAVAILABLE_EXPORT", "error_type": type(error).__name__}
        observations[key] = observation
    ledger = []
    for w in windows:
        for stream in STREAMS:
            key = (w["trading_day"], w["ticker"], stream)
            item = {**w, "stream": stream, "state": "NOT_SUPPLIED", "error_type": None}
            if key in observations:
                item.update(observations[key])
                if item["state"] == "SOURCE_COMPLETENESS_UNVERIFIED" and not (entries[key]["start_ns"] <= w["start_ns"] and entries[key]["end_ns"] >= w["end_ns"]):
                    item["state"] = "PARTIAL_DECLARED_WINDOW"
            ledger.append(item)
    counts = Counter(x["state"] for x in ledger)
    report = {"request_id": 335, "stage": "offline_input_preparation_only",
              "identities": len(windows), "declared_identity_streams": len(ledger),
              "identity_counts_by_scope": dict(sorted(Counter(w["scope"] for w in windows).items())),
              "primary_May6_8_identities": sum(w["trading_day"] in ("2026-05-06", "2026-05-07", "2026-05-08") for w in windows),
              "fixed_windows_sha256": sha(canonical(windows)),
              "state_counts": dict(sorted(counts.items())), "source_partitions_supplied": len(catalog),
              "source_bytes_read": budget[0], "market_http_attempts": 0,
              "independent_complete_source_pairs_verified": 0, "new_economic_labels": 0,
              "model_fits": 0, "promotion_eligible": False,
              "decision": "AUTHORIZED_SOURCE_AND_COMPLETENESS_EVIDENCE_REQUIRED",
              "ledger_sha256": sha(canonical(ledger))}
    return report, ledger


def write_once(report, ledger, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "request335-input-ledger.json").write_bytes(canonical(ledger))
    (output / "request335.json").write_bytes(canonical(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = fixed_windows(args.previous, args.pins, args.plan)
    catalog = strict_json(args.catalog.read_bytes())
    report, ledger = inventory(windows, catalog, args.export_root)
    write_once(report, ledger, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
