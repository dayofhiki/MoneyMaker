"""Registered complete-chain reuse after audited storage failure; no cursor resumption."""
from __future__ import annotations

import gzip
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .offline_source_inventory import canonical, confined, sha, strict_json
from .r335_alpaca_qualification import PLAN_SHA, checked_payload, inspect_page, save_private
from .r335_census_acquisition import (
    BUDGETS, DAY_CAP, ERRORS, LIMITS, PAGE_CAP, TERMINAL, Transport,
    checkpoint, descriptors, validate_schedule, verify_rights,
)

from . import r335_recovery_census as prefix_source

STORAGE_FAILURE = "PRIVATE_CHECKPOINT_FAILED"
REUSED = "REUSED_VERIFIED_TEN_DATE_SOURCE_UNQUALIFIED"
PREFIX_SHA = "667878ad607dc203e6df5b51abe25aafd89822d92f0a9c5c9044884afd233eca"
PREFIX_REPLAY_SHA = "7867dbfe6a7fd0c009f13e5fe75aad6d4e9e0147673e29af27749b642f2eaa8e"
CONTRACT = {
    "request_id": 335, "stage": "registered_final_original_census_with_atomic_private_archives",
    "prefix_manifest_sha256": PREFIX_SHA,
    "prefix_replay_sha256": PREFIX_REPLAY_SHA,
    "reuse_verification": "previous_frozen_full_raw_replay_cache_plus_current_raw_manifest_page_attempt_and_source_hash_verification",
    "prefix_source_commit": "0c6eb1488615ea12fda3ba54f38f850644fa6adf",
    "source_plan_sha256": PLAN_SHA, "limits": LIMITS,
    "selection": "all_original_2470_descriptors_407_training_828_reused_development",
    "pilot_reuse": "2052_complete_original_chains_raw_verified_no_failed_or_partial_chain_reuse",
    "archive_publication": "canonical_headers_gzip_close_fsync_temporary_then_atomic_rename_and_directory_fsync",
    "archive_readiness": "read_only_wait_for_expected_bytes_and_hash_before_any_upload",
    "archive_rebuild": "if_needed_rebuild_same_canonical_archive_in_foreground_and_require_exact_expected_hash",
    "prior_storage_failure": "preserved_not_overwritten_or_automatically_resumed",
    "known_storage_corruption_repair": "one_version_guarded_replacement_from_held_master_then_reverify_no_unknown_write_retry",
    "scheduling": "day_ascending_round_robin_original_indices_waves_at_most_four",
    "checkpoint_policy": "each_completed_date_save_retrieve_hash_verify_before_next_date",
    "checkpoint_failure_policy": "stop_new_market_requests_no_auto_resume",
    "HTTP_policy": "first_error_stop_new_starts_drain_inflight_no_retry_redirect_fallback",
    "raw_retention": "user_private_backup_no_public_market_rows_tokens_credentials",
    "preflight_archive_sha256": "648c0456b73cee0c1224d167ffa4dff63b8d079586965d0350d7e6bf8cdfe0de",
    "feed": "sip", "asof": "-", "additional_cost_authorized_USD": 0,
    "clock": "provider_RFC3339_not_aliased_to_SIP_or_strategy_receipt",
    "original_resolution_gate_percent": 90, "source_ready": False,
    "economic_labels_enabled": False, "profitability_validation_complete": False,
    "june": "HOLD", "july_august": "SEALED",
}


def source_hashes():
    return {name: sha(Path(__file__).with_name(name).read_bytes()) for name in (
        "r335_recovery_census.py", "r335_fresh_census.py", "r335_atomic_checkpoint.py", "r335_census_acquisition.py", "r335_alpaca_qualification.py",
        "r335_full_window_qualification.py", "r335_private_checkpoint.py", "offline_source_inventory.py",
        "tick_execution_readiness.py")}


def rights(data):
    verify_rights(data)
    if strict_json(data).get("user_private_persistent_backup_authorized") is not True:
        raise ValueError("private_durable_backup_scope_required")


def storage_proof(value, day):
    required = {"day", "archive_sha256", "archive_bytes", "private_files",
                "saved", "retrieved_archive_hash_equal"}
    if (not isinstance(value, dict) or set(value) != required or value["day"] != day or
            value["saved"] is not True or value["retrieved_archive_hash_equal"] is not True or
            not isinstance(value["archive_sha256"], str) or len(value["archive_sha256"]) != 64 or
            any(c not in "0123456789abcdef" for c in value["archive_sha256"]) or
            any(type(value[k]) is not int or value[k] <= 0 for k in ("archive_bytes", "private_files"))):
        raise ValueError("verified_private_checkpoint_required")
    return value


def verified_prefix(plan, root):
    raw = confined(root, "manifest.json").read_bytes()
    if sha(raw) != PREFIX_SHA:
        raise ValueError("registered_prefix_manifest_changed")
    with gzip.open(confined(root, "prefix-replay.json.gz"), "rb") as handle:
        cached = handle.read(2 * 1024 * 1024 + 1)
    if len(cached) > 2 * 1024 * 1024 or sha(cached) != PREFIX_REPLAY_SHA:
        raise ValueError("registered_prefix_raw_replay_cache_changed")
    parent, result = strict_json(raw), strict_json(cached)
    fixed = descriptors(plan)
    prefix_source.verified_prefix(plan, root / "prefix")
    if (parent["implementation_sha256"] != sha(Path(prefix_source.__file__).read_bytes()) or
            parent["dependency_sha256"] != prefix_source.source_hashes() or
            confined(root, "contract.json").read_bytes() != canonical(prefix_source.CONTRACT)):
        raise ValueError("registered_prefix_source_changed")
    attestation = confined(root, "attestation.json").read_bytes()
    prefix_source.rights(attestation)
    if sha(attestation) != parent["attestation_sha256"] or len(parent["pairs"]) != len(fixed):
        raise ValueError("prefix_attestation_or_denominator_changed")
    for pair, request in zip(parent["pairs"], fixed, strict=True):
        if pair["request"] != request:
            raise ValueError("prefix_original_descriptor_changed")
        for page in pair["pages"]:
            wire = confined(root, page["wire_file"]).read_bytes()
            if len(wire) != page["body_bytes"] or sha(wire) != page["wire_sha256"]:
                raise ValueError("wire_evidence_changed")
            if confined(root, f"attempts/{page['attempt']:05d}.json").read_bytes() != canonical(page):
                raise ValueError("prefix_attempt_checkpoint_changed")
    if (result["manifest_sha256"] != PREFIX_SHA or result["stop_reason"] != STORAGE_FAILURE or
            result["partition_states"].get(TERMINAL, 0) + result["partition_states"].get(prefix_source.REUSED, 0) != 2052):
        raise ValueError("registered_2052_complete_prefix_raw_required")
    return parent, result


def collect(plan, prefix, root, *, credentials, attestation, persist, prefix_proofs, get=None, clock=None,
            sleep=None, disk=None):
    fixed = descriptors(plan)
    parent, parent_report = verified_prefix(plan, prefix)
    parent_days = [d["day"] for d in parent["days"]]
    if [storage_proof(p, p["day"])["day"] for p in prefix_proofs] != parent_days:
        raise ValueError("all_restored_prefix_date_proofs_required")
    rights(attestation)
    if (root.exists() or not callable(persist) or len(credentials) != 2 or
            any(not isinstance(v, str) or not v or "\r" in v or "\n" in v for v in credentials)):
        raise ValueError("fresh_private_directory_credentials_and_checkpoint_callback_required")
    root.mkdir(parents=True, mode=0o700)
    for path in sorted(prefix.rglob("*")):
        if path.is_symlink():
            raise ValueError("private_prefix_symlink_not_allowed")
        if path.is_file():
            save_private(root, "prefix/" + path.relative_to(prefix).as_posix(), path.read_bytes())
    verified_prefix(plan, root / "prefix")
    save_private(root, "contract.json", canonical(CONTRACT))
    save_private(root, "attestation.json", attestation)
    complete = {i for i, p in enumerate(parent["pairs"]) if p["state"] in {TERMINAL, prefix_source.REUSED}}
    pairs = [{"request": r, "pages": [], "state": REUSED if i in complete else "ACTIVE",
              "pilot_index": i if i in complete else None} for i, r in enumerate(fixed)]
    m = {"request_id": 335, "stage": CONTRACT["stage"], "source_plan_sha256": PLAN_SHA,
         "implementation_sha256": sha(Path(__file__).read_bytes()), "dependency_sha256": source_hashes(),
         "contract_sha256": sha(canonical(CONTRACT)), "attestation_sha256": sha(attestation),
         "pairs": pairs, "waves": [], "days": [], "private_checkpoints": [],
         "started_utc": datetime.now(timezone.utc).isoformat(), "http_attempts": 0,
         "new_response_bytes": 0, "reused_response_bytes": parent_report["new_response_bytes"] + parent_report["reused_response_bytes"],
         "prefix_manifest_sha256": PREFIX_SHA, "prefix_private_checkpoint_proofs": prefix_proofs, "stop_reason": None,
         "source_ready": False, "new_economic_labels": 0, "profitability_validation_complete": False}
    transport = Transport(root, credentials, get, clock, sleep, disk)
    tokens, seen = [None] * len(pairs), [set() for _ in pairs]
    checkpoint(root, m)
    try:
        with ThreadPoolExecutor(max_workers=LIMITS["parallel_requests"]) as pool:
            for day in sorted({r["trading_day"] for r in fixed}):
                start, round_index = transport.clock() - transport.started, 0
                day_record = {"day": day, "start_elapsed_seconds": start}
                m["days"].append(day_record)
                if day in parent_days:
                    day_record["end_elapsed_seconds"] = transport.clock() - transport.started
                    m["private_checkpoints"].append(next(p for p in prefix_proofs if p["day"] == day))
                    checkpoint(root, m)
                    continue
                indices = [i for i, r in enumerate(fixed) if r["trading_day"] == day]
                while not transport.stopped.is_set():
                    active = [i for i in indices if pairs[i]["state"] == "ACTIVE"]
                    if not active:
                        break
                    for offset in range(0, len(active), LIMITS["parallel_requests"]):
                        if transport.stopped.is_set():
                            break
                        batch, wave = active[offset:offset + LIMITS["parallel_requests"]], len(m["waves"]) + 1
                        m["waves"].append({"number": wave, "day": day, "day_start_elapsed_seconds": start,
                                           "round": round_index, "offset": offset, "indices": batch})
                        futures = [pool.submit(transport.fetch, fixed[i], tokens[i],
                            f"{i:04d}-{len(pairs[i]['pages']):03d}", start, wave) for i in batch]
                        for i, future in zip(batch, futures, strict=True):
                            page, token, state = future.result()
                            if page is not None:
                                pairs[i]["pages"].append(page)
                            if state == "RECEIVED":
                                if token is None:
                                    pairs[i]["state"] = TERMINAL
                                elif token in seen[i]:
                                    pairs[i]["state"] = "REPEATED_PAGINATION_TOKEN"
                                    transport.stop("REPEATED_PAGINATION_TOKEN")
                                elif len(pairs[i]["pages"]) >= LIMITS["pages_per_partition"]:
                                    pairs[i]["state"] = PAGE_CAP
                                else:
                                    seen[i].add(token)
                                    tokens[i] = token
                            elif page is not None or state == DAY_CAP:
                                pairs[i]["state"] = state
                        if transport.clock() - transport.started - start >= LIMITS["day_wall_seconds"]:
                            for i in indices:
                                if pairs[i]["state"] == "ACTIVE":
                                    pairs[i]["state"] = DAY_CAP
                        m["waves"][-1]["end_elapsed_seconds"] = transport.clock() - transport.started
                        m.update(http_attempts=transport.attempts, new_response_bytes=transport.bytes,
                                 stop_reason=transport.reason, stop_elapsed_seconds=transport.stop_elapsed,
                                 elapsed_seconds=transport.clock() - transport.started)
                        checkpoint(root, m)
                        if any(pairs[i]["state"] == DAY_CAP for i in indices):
                            break
                    round_index += 1
                day_record["end_elapsed_seconds"] = transport.clock() - transport.started
                checkpoint(root, m)
                if transport.stopped.is_set():
                    break
                try:
                    proof = storage_proof(persist(root, day, m), day)
                    m["private_checkpoints"].append(proof)
                except Exception:
                    # Storage errors can contain signed URLs; retain the failure class only.
                    transport.stop(STORAGE_FAILURE)
                    break
    finally:
        transport.close()
    for pair in pairs:
        if pair["state"] == "ACTIVE":
            pair["state"] = "INCOMPLETE_AFTER_" + transport.reason
    m.update(finished_utc=datetime.now(timezone.utc).isoformat(), http_attempts=transport.attempts,
             new_response_bytes=transport.bytes, stop_reason=transport.reason,
             stop_elapsed_seconds=transport.stop_elapsed, elapsed_seconds=transport.clock() - transport.started)
    checkpoint(root, m)
    save_private(root, "manifest.json", canonical(m))
    return m


def replay(plan, root):
    fixed = descriptors(plan)
    parent, parent_report = verified_prefix(plan, root / "prefix")
    raw = confined(root, "manifest.json").read_bytes()
    m = strict_json(raw)
    if (m["source_plan_sha256"] != PLAN_SHA or m["stage"] != CONTRACT["stage"] or
            m["implementation_sha256"] != sha(Path(__file__).read_bytes()) or
            m["dependency_sha256"] != source_hashes() or
            m["contract_sha256"] != sha(canonical(CONTRACT)) or
            confined(root, "contract.json").read_bytes() != canonical(CONTRACT)):
        raise ValueError("registered_fresh_source_or_contract_changed")
    attestation = confined(root, "attestation.json").read_bytes()
    rights(attestation)
    if sha(attestation) != m["attestation_sha256"] or len(m["pairs"]) != len(fixed):
        raise ValueError("attestation_or_original_denominator_changed")
    if (m["source_ready"] is not False or m["new_economic_labels"] != 0 or
            m["profitability_validation_complete"] is not False or
            m["reused_response_bytes"] != parent_report["new_response_bytes"] + parent_report["reused_response_bytes"] or
            m["prefix_manifest_sha256"] != PREFIX_SHA):
        raise ValueError("fresh_evidence_cannot_claim_reuse_or_economic_qualification")
    attempts, output, total = [], [], 0
    for index, (pair, request) in enumerate(zip(m["pairs"], fixed, strict=True)):
        if pair["request"] != request or (
                len(pair["pages"]) > LIMITS["pages_per_partition"]):
            raise ValueError("original_descriptor_required")
        if pair["pilot_index"] is not None:
            i = pair["pilot_index"]
            if (type(i) is not int or i != index or pair["state"] != REUSED or pair["pages"] or
                    parent["pairs"][i]["state"] not in {TERMINAL, prefix_source.REUSED} or parent["pairs"][i]["request"] != request):
                raise ValueError("complete_original_prefix_chain_required")
            output.append({"original_full_census_index": index, "identity_indices": request["identity_indices"],
                "trading_day": request["trading_day"], "ticker": request["ticker"], "stream": request["stream"],
                "start_ns": request["start_ns"], "end_ns": request["end_ns"], "collection_state": REUSED,
                "pages_observed": len(parent["pairs"][i]["pages"]),
                "rows_observed": parent_report["pairs"][i]["rows_observed"],
                "source_ready": False, "economic_label": None})
            continue
        prior, terminal, failed, seen, rows = None, False, None, set(), 0
        for page_index, page in enumerate(pair["pages"]):
            if terminal or failed or page["request_token_sha256"] != prior:
                raise ValueError("broken_or_post_terminal_page_chain")
            count = page["body_bytes"]
            if type(count) is not int or not 0 <= count <= LIMITS["bytes_per_page"]:
                raise ValueError("invalid_retained_byte_count")
            attempts.append((page["attempt"], index, page))
            total += count
            if page["wire_file"] is None:
                if not page["wire_evidence_withheld"] or page["transport_state"] != "SENSITIVE_BODY_WITHHELD":
                    raise ValueError("invalid_withheld_evidence")
                failed = page["transport_state"]
                continue
            if page["wire_file"] != f"pages/{index:04d}-{page_index:03d}.wire":
                raise ValueError("registered_wire_path_changed")
            wire = confined(root, page["wire_file"]).read_bytes()
            if len(wire) != count or sha(wire) != page["wire_sha256"]:
                raise ValueError("wire_evidence_changed")
            if page["transport_state"] != "RECEIVED":
                if page["observation"] is not None or page["next_token_sha256"] is not None:
                    raise ValueError("failed_response_cannot_support_observation")
                failed = page["transport_state"]
                continue
            if page["http_status"] != 200 or not page["wire_complete"]:
                raise ValueError("unsuccessful_page_cannot_support_observation")
            observed, token = inspect_page(checked_payload(wire), request)
            digest = sha(token.encode()) if token else None
            if observed != page["observation"] or digest != page["next_token_sha256"]:
                raise ValueError("decoded_observation_or_token_changed")
            rows += observed["rows"]
            if digest is not None and digest in seen:
                failed = "REPEATED_PAGINATION_TOKEN"
            if digest is not None:
                seen.add(digest)
            terminal, prior = token is None, digest
        if ((pair["state"] == TERMINAL) != terminal or
                (failed and pair["state"] != failed) or
                (pair["state"] in ERRORS and pair["state"] != failed) or
                (pair["state"] == PAGE_CAP and len(pair["pages"]) != LIMITS["pages_per_partition"])):
            raise ValueError("terminal_error_or_cap_state_changed")
        allowed = {TERMINAL, PAGE_CAP, DAY_CAP} | ERRORS
        if m["stop_reason"]:
            allowed.add("INCOMPLETE_AFTER_" + m["stop_reason"])
        if pair["state"] not in allowed:
            raise ValueError("unregistered_partition_state")
        output.append({"original_full_census_index": index, "identity_indices": request["identity_indices"],
            "trading_day": request["trading_day"], "ticker": request["ticker"], "stream": request["stream"],
            "start_ns": request["start_ns"], "end_ns": request["end_ns"], "collection_state": pair["state"],
            "pages_observed": len(pair["pages"]), "rows_observed": rows,
            "source_ready": False, "economic_label": None})
    attempts.sort(key=lambda x: x[0])
    last = None
    for ordinal, (number, index, page) in enumerate(attempts, 1):
        if confined(root, f"attempts/{number:05d}.json").read_bytes() != canonical(page):
            raise ValueError("attempt_checkpoint_changed")
        start, end = page["start_elapsed_seconds"], page["end_elapsed_seconds"]
        if number != ordinal or not 0 <= start <= end <= m["elapsed_seconds"] or start >= LIMITS["wall_seconds"]:
            raise ValueError("attempt_or_transport_time_changed")
        if last is not None and start - last < LIMITS["minimum_interval_seconds"] - 1e-6:
            raise ValueError("global_request_start_rate_violated")
        last = start
        wave = m["waves"][page["wave"] - 1]
        if index not in wave["indices"] or start - wave["day_start_elapsed_seconds"] >= LIMITS["day_wall_seconds"]:
            raise ValueError("registered_day_or_wave_changed")
        if m["stop_elapsed_seconds"] is not None and start > m["stop_elapsed_seconds"]:
            raise ValueError("new_request_started_after_stop")
    schedule = strict_json(canonical(m))
    for pair in schedule["pairs"]:
        if pair["pilot_index"] is not None:
            pair["state"] = "REUSED_API_TERMINAL_SOURCE_UNQUALIFIED"
    validate_schedule(schedule, fixed, attempts)
    points = sorted([(p["start_elapsed_seconds"], 1) for _, _, p in attempts
                     if p["end_elapsed_seconds"] > p["start_elapsed_seconds"]] +
                    [(p["end_elapsed_seconds"], -1) for _, _, p in attempts
                     if p["end_elapsed_seconds"] > p["start_elapsed_seconds"]])
    active, maximum = 0, 0
    for _, delta in points:
        active += delta
        maximum = max(maximum, active)
    if (maximum > LIMITS["parallel_requests"] or len(attempts) != m["http_attempts"] or
            total != m["new_response_bytes"] or total > LIMITS["total_new_response_bytes"] or
            len(attempts) > LIMITS["http_attempts"] or m["stop_reason"] not in ({None, STORAGE_FAILURE} | ERRORS | BUDGETS)):
        raise ValueError("aggregate_budget_or_stop_record_changed")
    days = [d["day"] for d in m["days"]]
    stored_days = [storage_proof(p, p["day"])["day"] for p in m["private_checkpoints"]]
    expected_days = days if m["stop_reason"] is None else days[:-1]
    if stored_days != expected_days:
        raise ValueError("checkpoint_before_next_day_required")
    if sum(p["collection_state"] == REUSED for p in output) != 2052:
        raise ValueError("all_registered_complete_prefix_chains_required")
    prefix_days = [d["day"] for d in parent["days"]]
    proofs = m["prefix_private_checkpoint_proofs"]
    if [storage_proof(p, p["day"])["day"] for p in proofs] != prefix_days or (
            m["private_checkpoints"][:len(proofs)] != proofs):
        raise ValueError("restored_prefix_checkpoint_proofs_changed")
    return {"request_id": 335, "stage": "offline_final_original_census_atomic_checkpoint_replay",
        "manifest_sha256": sha(raw), "implementation_sha256": m["implementation_sha256"],
        "contract_sha256": m["contract_sha256"], "http_attempts": len(attempts),
        "new_response_bytes": total, "reused_response_bytes": parent_report["new_response_bytes"] + parent_report["reused_response_bytes"],
        "prefix_manifest_sha256": PREFIX_SHA, "prior_HTTP_attempts": parent_report["http_attempts"] + parent_report["prior_HTTP_attempts"],
        "prior_stop_reason": parent_report["stop_reason"], "pairs": output,
        "original_identity_denominator": 1235, "original_partition_stream_denominator": 2470,
        "partition_states": dict(Counter(p["collection_state"] for p in output)),
        "verified_private_date_checkpoints": len(stored_days), "max_observed_fetch_worker_occupancy": maximum,
        "elapsed_seconds": m["elapsed_seconds"], "started_utc": m["started_utc"], "finished_utc": m["finished_utc"],
        "stop_reason": m["stop_reason"], "source_ready": False, "new_economic_labels": 0,
        "profitability_validation_complete": False, "additional_cost_USD": 0}
