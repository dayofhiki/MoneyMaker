"""Fixed original-census, private free SIP acquisition; no economic use or resume."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

from .offline_source_inventory import canonical, confined, sha, strict_json
from .r335_alpaca_qualification import (
    HOST, PLAN_SHA, checked_payload, inspect_page, load_plan, save_private, sensitive_payload,
)
from .r335_full_window_qualification import replay as pilot_replay

PILOT_SHA = "a36544d5d2d09004967d3705b4bf07d10ea8e8dd1bc7f29a15ddd44e1cd3a0c8"
LIMITS = {"http_attempts": 16384, "pages_per_partition": 128, "bytes_per_page": 2097152,
          "total_new_response_bytes": 8589934592, "minimum_interval_seconds": 1,
          "timeout_seconds": 30, "wall_seconds": 14400, "day_wall_seconds": 1200,
          "minimum_free_disk_bytes": 2147483648, "parallel_requests": 4, "retries": 0}
CONTRACT = {"request_id": 335, "stage": "original_2470_partition_free_census_acquisition_only",
            "source_plan_sha256": PLAN_SHA, "pilot_manifest_sha256": PILOT_SHA, "limits": LIMITS,
            "selection": "all_original_2470_descriptors_407_training_828_reused_development",
            "scheduling": "day_ascending_round_robin_original_indices_waves_at_most_four",
            "reuse": "copy_and_verify_all_24_pinned_pilot_chains_not_source_qualification",
            "HTTP_policy": "first_error_stops_new_starts_drain_inflight_no_retry_redirect_fallback_resume",
            "cap_policy": "page_and_day_wall_caps_explicit_incomplete_continue_other_fixed_entries",
            "feed": "sip", "asof": "-", "additional_cost_authorized_USD": 0,
            "clock": "provider_RFC3339_not_aliased_to_SIP_or_strategy_receipt",
            "raw_storage": "private_local_0700_directories_0600_files_no_redistribution",
            "source_ready": False, "economic_labels_enabled": False,
            "profitability_validation_complete": False, "original_resolution_gate_percent": 90,
            "june": "HOLD", "july_august": "SEALED"}
TERMINAL = "API_TERMINAL_SOURCE_UNQUALIFIED"
REUSED = "REUSED_API_TERMINAL_SOURCE_UNQUALIFIED"
PAGE_CAP = "PAGE_CAP_INCOMPLETE"
DAY_CAP = "DAY_WALL_CAP_INCOMPLETE"
ERRORS = {"TRANSPORT_ERROR", "INVALID_JSON", "HTTP_DENIED_OR_ERROR", "INVALID_MARKET_SCHEMA",
          "SENSITIVE_BODY_WITHHELD", "PARTIAL_BODY_BYTE_CAP", "PARTIAL_BODY_WALL_CAP",
          "REPEATED_PAGINATION_TOKEN"}
BUDGETS = {"HTTP_BUDGET_EXHAUSTED", "BYTE_RESERVATION_BUDGET_EXHAUSTED",
           "WALL_BUDGET_EXHAUSTED", "DISK_BUDGET_EXHAUSTED"}


def descriptors(plan):
    if sha(canonical(plan)) != PLAN_SHA or len(plan["full_census"]) != 2470:
        raise ValueError("pinned_full_census_required")
    result = copy.deepcopy(plan["full_census"])
    if {i for r in result for i in r["identity_indices"]} != set(range(1235)):
        raise ValueError("all_original_identity_indices_required")
    return result


def verify_rights(data):
    if len(data) > 16384:
        raise ValueError("small_census_attestation_required")
    value = strict_json(data)
    required = ("account_is_free_Basic", "no_incremental_charge", "no_redistribution",
                "historical_SIP_research_and_local_retention_permitted",
                "registered_original_2470_private_technical_census_authorized")
    if (not isinstance(value, dict) or any(value.get(k) is not True for k in required) or
            not isinstance(value.get("rights_evidence_reference"), str) or
            not value["rights_evidence_reference"].strip()):
        raise ValueError("explicit_census_scope_and_rights_required")


def verified_pilot(plan, root):
    if sha(confined(root, "manifest.json").read_bytes()) != PILOT_SHA:
        raise ValueError("registered_pilot_raw_manifest_required")
    result = pilot_replay(plan, root)
    if len(result["pairs"]) != 24 or any(not p["terminal_page_observed"] for p in result["pairs"]):
        raise ValueError("all_pinned_pilot_chains_must_be_terminal")
    return strict_json(confined(root, "manifest.json").read_bytes()), result


def checkpoint(root, manifest):
    path = root / "progress.tmp"
    with path.open("wb") as handle:
        os.chmod(path, 0o600)
        handle.write(canonical(manifest))
    path.replace(root / "progress.json")


def source_hashes():
    return {name: sha(Path(__file__).with_name(name).read_bytes()) for name in (
        "r335_alpaca_qualification.py", "r335_full_window_qualification.py",
        "tick_execution_readiness.py", "offline_source_inventory.py")}


class Transport:
    """A single locked request-start/byte budget shared by all four workers."""
    def __init__(self, root, credentials, get=None, clock=None, sleep=None, disk=None):
        self.root, self.credentials = root, credentials
        self.get = get
        self.clock, self.sleep = clock or time.monotonic, sleep or time.sleep
        self.disk = disk or (lambda: shutil.disk_usage(root).free)
        self.started, self.last = self.clock(), None
        self.attempts, self.bytes, self.reserved = 0, 0, 0
        self.lock, self.local, self.stopped = threading.Lock(), threading.local(), threading.Event()
        self.reason, self.stop_elapsed, self.sessions = None, None, []

    def stop(self, reason):
        self.stopped.set()  # Visible even if another worker holds the rate lock during sleep.
        with self.lock:
            if self.reason is None:
                self.reason, self.stop_elapsed = reason, self.clock() - self.started

    def request_get(self, *args, **kwargs):
        if self.get is not None:
            return self.get(*args, **kwargs)
        if not hasattr(self.local, "session"):
            session = requests.Session()
            session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
            self.local.session = session
            with self.lock:
                self.sessions.append(session)
        return self.local.session.get(*args, **kwargs)

    def fetch(self, request, token, name, day_start, wave):
        reason = None
        with self.lock:
            if self.last is not None:
                self.sleep(max(0., LIMITS["minimum_interval_seconds"] - (self.clock() - self.last)))
            elapsed = self.clock() - self.started
            if self.stopped.is_set():
                return None, None, "NOT_STARTED_AFTER_STOP"
            if elapsed - day_start >= LIMITS["day_wall_seconds"]:
                return None, None, DAY_CAP
            if self.attempts >= LIMITS["http_attempts"]:
                reason = "HTTP_BUDGET_EXHAUSTED"
            elif self.bytes + self.reserved + LIMITS["bytes_per_page"] > LIMITS["total_new_response_bytes"]:
                reason = "BYTE_RESERVATION_BUDGET_EXHAUSTED"
            elif elapsed >= LIMITS["wall_seconds"]:
                reason = "WALL_BUDGET_EXHAUSTED"
            elif self.disk() < LIMITS["minimum_free_disk_bytes"] + self.reserved + LIMITS["bytes_per_page"]:
                reason = "DISK_BUDGET_EXHAUSTED"
            if reason is None:
                self.last = self.clock()
                self.attempts += 1
                self.reserved += LIMITS["bytes_per_page"]
                number, start = self.attempts, self.last - self.started
        if reason:
            self.stop(reason)
            return None, None, reason
        meta = {"attempt": number, "wave": wave, "start_elapsed_seconds": start,
                "request_utc": datetime.now(timezone.utc).isoformat(), "http_status": None,
                "wire_complete": False, "wire_file": None, "observation": None,
                "request_token_sha256": sha(token.encode()) if token else None,
                "next_token_sha256": None}
        response, body, state = None, bytearray(), "RECEIVED"
        try:
            response = self.request_get("https://" + HOST + request["path"],
                params={**request["params"], **({"page_token": token} if token else {})},
                headers={"APCA-API-KEY-ID": self.credentials[0], "APCA-API-SECRET-KEY": self.credentials[1]},
                timeout=LIMITS["timeout_seconds"], stream=True, allow_redirects=False)
            meta["http_status"] = int(response.status_code)
            for chunk in response.iter_content(chunk_size=4096):
                if self.clock() - self.started >= LIMITS["wall_seconds"]:
                    state = "PARTIAL_BODY_WALL_CAP"
                    break
                if len(body) + len(chunk) > LIMITS["bytes_per_page"]:
                    body.extend(chunk[:LIMITS["bytes_per_page"] - len(body)])
                    state = "PARTIAL_BODY_BYTE_CAP"
                    break
                body.extend(chunk)
            else:
                meta["wire_complete"] = True
        except requests.RequestException:
            state = "TRANSPORT_ERROR"  # Never persist exception text, URL or authentication.
        finally:
            if response is not None:
                response.close()
        wire, payload, token_out = bytes(body), None, None
        try:
            payload = checked_payload(wire)
        except (ValueError, UnicodeError, RecursionError):
            if state == "RECEIVED":
                state = "INVALID_JSON"
        unsafe = any(s.encode() in wire for s in self.credentials) or (
            payload is not None and sensitive_payload(payload, self.credentials))
        meta["wire_evidence_withheld"] = unsafe
        if unsafe:
            state, payload = "SENSITIVE_BODY_WITHHELD", None
        if state == "RECEIVED":
            if meta["http_status"] != 200:
                state = "HTTP_DENIED_OR_ERROR"
            else:
                try:
                    meta["observation"], token_out = inspect_page(payload, request)
                    meta["next_token_sha256"] = sha(token_out.encode()) if token_out else None
                except (ValueError, TypeError):
                    state = "INVALID_MARKET_SCHEMA"
        if state != "RECEIVED":
            self.stop(state)
        if not unsafe:
            meta["wire_file"] = "pages/" + name + ".wire"
            save_private(self.root, meta["wire_file"], wire)
        with self.lock:
            self.bytes += len(wire)
            self.reserved -= LIMITS["bytes_per_page"]
        meta.update(body_bytes=len(wire), wire_sha256=sha(wire), transport_state=state,
                    end_elapsed_seconds=self.clock() - self.started,
                    response_utc=datetime.now(timezone.utc).isoformat())
        save_private(self.root, f"attempts/{number:05d}.json", canonical(meta))
        return meta, token_out, state

    def close(self):
        for session in self.sessions:
            session.close()


def collect(plan, pilot, root, *, credentials, attestation, get=None, clock=None, sleep=None, disk=None):
    fixed = descriptors(plan)
    parent, parent_report = verified_pilot(plan, pilot)
    verify_rights(attestation)
    if root.exists() or len(credentials) != 2 or any(
            not isinstance(v, str) or not v or "\r" in v or "\n" in v for v in credentials):
        raise ValueError("fresh_directory_and_private_credentials_required")
    root.mkdir(parents=True, mode=0o700)
    for p in sorted(pilot.rglob("*")):
        if p.is_symlink():
            raise ValueError("pilot_symlink_not_allowed")
        if p.is_file():
            save_private(root, "pilot/" + p.relative_to(pilot).as_posix(), p.read_bytes())
    verified_pilot(plan, root / "pilot")
    save_private(root, "contract.json", canonical(CONTRACT))
    save_private(root, "attestation.json", attestation)
    lookup = {canonical(p["request"]): i for i, p in enumerate(parent["pairs"])}
    pairs = [{"request": r, "pages": [], "state": REUSED if canonical(r) in lookup else "ACTIVE",
              "pilot_index": lookup.get(canonical(r))} for r in fixed]
    if sum(p["state"] == REUSED for p in pairs) != 24:
        raise ValueError("exact_24_original_pilot_descriptors_required")
    manifest = {"request_id": 335, "stage": CONTRACT["stage"], "source_plan_sha256": PLAN_SHA,
                "contract_sha256": sha(canonical(CONTRACT)), "pilot_manifest_sha256": PILOT_SHA,
                "implementation_sha256": sha(Path(__file__).read_bytes()),
                "attestation_sha256": sha(attestation), "dependency_sha256": source_hashes(),
                "pairs": pairs, "waves": [], "days": [],
                "started_utc": datetime.now(timezone.utc).isoformat(), "http_attempts": 0,
                "new_response_bytes": 0, "reused_response_bytes": parent_report["retained_response_bytes"],
                "stop_reason": None, "source_ready": False, "new_economic_labels": 0,
                "profitability_validation_complete": False}
    transport = Transport(root, credentials, get, clock, sleep, disk)
    tokens, seen = [None] * len(pairs), [set() for _ in pairs]
    checkpoint(root, manifest)
    try:
        with ThreadPoolExecutor(max_workers=LIMITS["parallel_requests"]) as pool:
            for day in sorted({p["request"]["trading_day"] for p in pairs}):
                day_start, round_index = transport.clock() - transport.started, 0
                day_record = {"day": day, "start_elapsed_seconds": day_start}
                manifest["days"].append(day_record)
                day_indices = [i for i, p in enumerate(pairs) if p["request"]["trading_day"] == day]
                while not transport.stopped.is_set():
                    active = [i for i in day_indices if pairs[i]["state"] == "ACTIVE"]
                    if not active:
                        break
                    for offset in range(0, len(active), LIMITS["parallel_requests"]):
                        if transport.stopped.is_set():
                            break
                        batch = active[offset:offset + LIMITS["parallel_requests"]]
                        wave = len(manifest["waves"]) + 1
                        manifest["waves"].append({"number": wave, "day": day, "day_start_elapsed_seconds": day_start,
                                                  "round": round_index, "offset": offset, "indices": batch})
                        futures = [pool.submit(transport.fetch, pairs[i]["request"], tokens[i],
                                               f"{i:04d}-{len(pairs[i]['pages']):03d}", day_start, wave) for i in batch]
                        for i, future in zip(batch, futures):
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
                            elif page is not None:
                                pairs[i]["state"] = state
                            elif state == DAY_CAP:
                                pairs[i]["state"] = DAY_CAP
                        if transport.clock() - transport.started - day_start >= LIMITS["day_wall_seconds"]:
                            for i in day_indices:
                                if pairs[i]["state"] == "ACTIVE":
                                    pairs[i]["state"] = DAY_CAP
                        manifest["waves"][-1]["end_elapsed_seconds"] = transport.clock() - transport.started
                        manifest.update(http_attempts=transport.attempts, new_response_bytes=transport.bytes,
                                        stop_reason=transport.reason, stop_elapsed_seconds=transport.stop_elapsed,
                                        elapsed_seconds=transport.clock() - transport.started)
                        checkpoint(root, manifest)
                        if any(pairs[i]["state"] == DAY_CAP for i in day_indices):
                            break
                    round_index += 1
                day_record["end_elapsed_seconds"] = transport.clock() - transport.started
                if transport.stopped.is_set():
                    break
    finally:
        transport.close()
    for pair in pairs:
        if pair["state"] == "ACTIVE":
            pair["state"] = "INCOMPLETE_AFTER_" + transport.reason
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), http_attempts=transport.attempts,
                    new_response_bytes=transport.bytes, stop_reason=transport.reason,
                    stop_elapsed_seconds=transport.stop_elapsed, elapsed_seconds=transport.clock() - transport.started)
    checkpoint(root, manifest)
    save_private(root, "manifest.json", canonical(manifest))
    return manifest


def replay(plan, root):
    fixed = descriptors(plan)
    m = strict_json(confined(root, "manifest.json").read_bytes())
    if (m["source_plan_sha256"] != PLAN_SHA or m["contract_sha256"] != sha(canonical(CONTRACT)) or
            confined(root, "contract.json").read_bytes() != canonical(CONTRACT) or
            m["implementation_sha256"] != sha(Path(__file__).read_bytes()) or
            m["pilot_manifest_sha256"] != PILOT_SHA or m["dependency_sha256"] != source_hashes()):
        raise ValueError("registered_source_or_contract_changed")
    attestation = confined(root, "attestation.json").read_bytes()
    if sha(attestation) != m["attestation_sha256"]:
        raise ValueError("attestation_evidence_changed")
    verify_rights(attestation)
    parent, parent_report = verified_pilot(plan, root / "pilot")
    if (m["source_ready"] is not False or m["profitability_validation_complete"] is not False or
            m["new_economic_labels"] != 0 or len(m["pairs"]) != len(fixed)):
        raise ValueError("all_denominators_and_unknown_economic_status_required")
    attempts, total, reused, output = [], 0, set(), []
    for index, (pair, request) in enumerate(zip(m["pairs"], fixed)):
        if pair["request"] != request or len(pair["pages"]) > LIMITS["pages_per_partition"]:
            raise ValueError("original_descriptor_changed")
        rows, previous, backward, ties, prior, terminal, failed = 0, None, 0, 0, None, False, None
        if pair["pilot_index"] is not None:
            i = pair["pilot_index"]
            if (type(i) is not int or i in reused or not 0 <= i < 24 or pair["pages"] or
                    pair["state"] != REUSED or parent["pairs"][i]["request"] != request):
                raise ValueError("invalid_or_duplicate_pilot_reuse")
            reused.add(i)
            rows = parent_report["pairs"][i]["rows_observed"]
        else:
            seen = set()
            for pindex, page in enumerate(pair["pages"]):
                if terminal or failed or page["request_token_sha256"] != prior:
                    raise ValueError("broken_or_post_terminal_page_chain")
                count = page["body_bytes"]
                if type(count) is not int or not 0 <= count <= LIMITS["bytes_per_page"]:
                    raise ValueError("invalid_retained_byte_count")
                total += count
                attempts.append((page["attempt"], index, page))
                if page["wire_file"] is None:
                    if (not page["wire_evidence_withheld"] or page["observation"] is not None or
                            page["transport_state"] != "SENSITIVE_BODY_WITHHELD"):
                        raise ValueError("withheld_body_cannot_support_observation")
                    failed = page["transport_state"]
                    continue
                if page["wire_file"] != f"pages/{index:04d}-{pindex:03d}.wire":
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
                first, last = observed["first_provider_ns"], observed["last_provider_ns"]
                if first is not None and previous is not None:
                    backward += int(first < previous)
                    ties += int(first == previous)
                if last is not None:
                    previous = last
                rows += observed["rows"]
                if digest is not None and digest in seen:
                    failed = "REPEATED_PAGINATION_TOKEN"
                if digest is not None:
                    seen.add(digest)
                terminal, prior = token is None, digest
            if ((pair["state"] == TERMINAL) != terminal or
                    (pair["state"] in ERRORS and pair["state"] != failed) or
                    (failed and pair["state"] != failed) or
                    (pair["state"] == PAGE_CAP and len(pair["pages"]) != LIMITS["pages_per_partition"])):
                raise ValueError("terminal_error_or_cap_state_changed")
            allowed = {TERMINAL, PAGE_CAP, DAY_CAP} | ERRORS
            if m["stop_reason"]:
                allowed.add("INCOMPLETE_AFTER_" + m["stop_reason"])
            if pair["state"] not in allowed:
                raise ValueError("unregistered_partition_state")
        output.append({"original_full_census_index": index, "identity_indices": request["identity_indices"],
                       "trading_day": request["trading_day"], "ticker": request["ticker"], "stream": request["stream"],
                       "start_ns": request["start_ns"], "end_ns": request["end_ns"],
                       "collection_state": pair["state"], "new_pages": len(pair["pages"]), "rows_observed": rows,
                       "pilot_index": pair["pilot_index"], "cross_page_backwards": backward,
                       "cross_page_equal_clock": ties, "source_ready": False, "economic_label": None})
    if reused != set(range(24)) or m["reused_response_bytes"] != parent_report["retained_response_bytes"]:
        raise ValueError("all_24_pinned_reuse_provenances_required")
    attempts.sort(key=lambda p: p[0])
    last = None
    for ordinal, (number, index, page) in enumerate(attempts, 1):
        if confined(root, f"attempts/{number:05d}.json").read_bytes() != canonical(page):
            raise ValueError("attempt_checkpoint_changed")
        start, end = page["start_elapsed_seconds"], page["end_elapsed_seconds"]
        if number != ordinal or not 0 <= start <= end or start >= LIMITS["wall_seconds"]:
            raise ValueError("attempt_or_transport_time_changed")
        if last is not None and start - last < LIMITS["minimum_interval_seconds"] - 1e-6:
            raise ValueError("global_request_start_rate_violated")
        last = start
        wave = m["waves"][page["wave"] - 1]
        if (wave["number"] != page["wave"] or index not in wave["indices"] or
                wave["day"] != fixed[index]["trading_day"] or
                start - wave["day_start_elapsed_seconds"] >= LIMITS["day_wall_seconds"]):
            raise ValueError("registered_day_or_wave_changed")
        if m["stop_elapsed_seconds"] is not None and start > m["stop_elapsed_seconds"]:
            raise ValueError("new_request_started_after_error_or_global_stop")
    validate_schedule(m, fixed, attempts)
    points = sorted([(p["start_elapsed_seconds"], 1) for _, _, p in attempts
                     if p["end_elapsed_seconds"] > p["start_elapsed_seconds"]] +
                    [(p["end_elapsed_seconds"], -1) for _, _, p in attempts
                     if p["end_elapsed_seconds"] > p["start_elapsed_seconds"]])
    active, maximum = 0, 0
    for _, delta in points:
        active += delta
        maximum = max(maximum, active)
    if maximum > LIMITS["parallel_requests"]:
        raise ValueError("parallel_request_cap_violated")
    if (len(attempts) != m["http_attempts"] or total != m["new_response_bytes"] or
            total > LIMITS["total_new_response_bytes"] or len(attempts) > LIMITS["http_attempts"] or
            m["stop_reason"] not in ({None} | ERRORS | BUDGETS)):
        raise ValueError("aggregate_budget_or_stop_record_changed")
    return {"request_id": 335, "stage": "offline_census_acquisition_replay_only",
            "manifest_sha256": sha(canonical(m)), "implementation_sha256": m["implementation_sha256"],
            "contract_sha256": m["contract_sha256"], "pilot_manifest_sha256": PILOT_SHA,
            "http_attempts": len(attempts), "new_response_bytes": total,
            "reused_response_bytes": m["reused_response_bytes"], "original_identity_denominator": 1235,
            "original_partition_stream_denominator": 2470, "pairs": output,
            "partition_states": dict(Counter(p["collection_state"] for p in output)),
            "max_observed_parallel_requests": maximum, "stop_reason": m["stop_reason"],
            "started_utc": m["started_utc"], "finished_utc": m["finished_utc"],
            "elapsed_seconds": m["elapsed_seconds"], "additional_cost_USD": 0,
            "source_ready": False, "new_economic_labels": 0, "profitability_validation_complete": False}


def validate_schedule(m, fixed, attempts):
    """Reconstruct every wave and final state, including entries with no HTTP page."""
    states = [REUSED if p["pilot_index"] is not None else "ACTIVE" for p in m["pairs"]]
    pages = {(p["wave"], i): p for _, i, p in attempts}
    if len(pages) != len(attempts):
        raise ValueError("duplicate_partition_in_wave")
    counts, tokens = Counter(), {i: set() for i in range(len(fixed))}
    days = sorted({r["trading_day"] for r in fixed})
    if [d["day"] for d in m["days"]] != days[:len(m["days"])]:
        raise ValueError("fixed_day_sequence_changed")
    if m["stop_reason"] is None and len(m["days"]) != len(days):
        raise ValueError("uncensored_day_omitted")
    cursor, previous_end = 0, 0.
    for day in m["days"]:
        start, end = day["start_elapsed_seconds"], day["end_elapsed_seconds"]
        if not previous_end <= start <= end <= m["elapsed_seconds"]:
            raise ValueError("day_wall_evidence_changed")
        previous_end = end
        indices = [i for i, r in enumerate(fixed) if r["trading_day"] == day["day"]]
        round_index, wave_end = 0, start
        while cursor < len(m["waves"]) and m["waves"][cursor]["day"] == day["day"]:
            active = [i for i in indices if states[i] == "ACTIVE"]
            if not active:
                raise ValueError("wave_after_day_completion")
            for offset in range(0, len(active), LIMITS["parallel_requests"]):
                if cursor >= len(m["waves"]) or m["waves"][cursor]["day"] != day["day"]:
                    break
                wave = m["waves"][cursor]
                batch = active[offset:offset + LIMITS["parallel_requests"]]
                if (wave["number"] != cursor + 1 or wave["indices"] != batch or
                        wave["round"] != round_index or wave["offset"] != offset or
                        wave["day_start_elapsed_seconds"] != start or
                        not wave_end <= wave["end_elapsed_seconds"] <= end):
                    raise ValueError("fixed_round_robin_schedule_changed")
                wave_end = wave["end_elapsed_seconds"]
                for i in batch:
                    page = pages.get((cursor + 1, i))
                    if page is None:
                        continue
                    if page["end_elapsed_seconds"] > wave_end:
                        raise ValueError("wave_finished_before_inflight_response")
                    counts[i] += 1
                    state, token = page["transport_state"], page["next_token_sha256"]
                    if state != "RECEIVED":
                        states[i] = state
                    elif token is None:
                        states[i] = TERMINAL
                    elif token in tokens[i]:
                        states[i] = "REPEATED_PAGINATION_TOKEN"
                    elif counts[i] == LIMITS["pages_per_partition"]:
                        states[i] = PAGE_CAP
                    else:
                        tokens[i].add(token)
                cursor += 1
                if wave_end - start >= LIMITS["day_wall_seconds"]:
                    for i in indices:
                        if states[i] == "ACTIVE":
                            states[i] = DAY_CAP
                    break
            round_index += 1
        if any(states[i] == "ACTIVE" for i in indices) and m["stop_reason"] is None:
            raise ValueError("uncensored_active_partition_omitted")
    if cursor != len(m["waves"]):
        raise ValueError("wave_outside_recorded_days")
    expected = ["INCOMPLETE_AFTER_" + m["stop_reason"] if s == "ACTIVE" and m["stop_reason"] else s
                for s in states]
    if expected != [p["state"] for p in m["pairs"]]:
        raise ValueError("partition_state_without_acquisition_evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("contract", "collect", "replay"))
    parser.add_argument("--plan", type=Path, default=Path("research/request335-free-source-plan.json.gz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--attestation", type=Path)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.mode == "contract":
        descriptors(plan)
        args.output.write_bytes(canonical(CONTRACT))
        result = {"http_attempts": 0, "source_ready": False}
    elif args.mode == "replay":
        result = replay(plan, args.output)
    else:
        if args.pilot is None or args.credentials is None or args.attestation is None:
            parser.error("private pinned pilot, credentials and census attestation required")
        if args.credentials.stat().st_mode & 0o077 or args.credentials.stat().st_size > 16384:
            parser.error("small private credentials required")
        credentials = strict_json(args.credentials.read_bytes())
        result = collect(plan, args.pilot, args.output,
                         credentials=(credentials.get("APCA_API_KEY_ID"), credentials.get("APCA_API_SECRET_KEY")),
                         attestation=args.attestation.read_bytes())
    print(json.dumps({"http_attempts": result["http_attempts"], "source_ready": False,
                      "new_economic_labels": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
