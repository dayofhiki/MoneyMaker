"""Fixed May anchors, full windows: private free acquisition feasibility only.

No order placement, market-clock inference, labels, fits, paid fallback or resume.
Transport elapsed seconds are resource measurements, never market timestamps.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

from .offline_source_inventory import canonical, confined, sha, strict_json
from .r335_alpaca_qualification import (
    HOST, PLAN_SHA, checked_payload, inspect_page, load_plan, save_private,
    sensitive_payload, verify_attestation,
)

LIMITS = {"http_attempts": 256, "pages_per_pair": 32, "bytes_per_page": 2097152,
          "total_retained_response_bytes": 268435456, "minimum_interval_seconds": 1,
          "timeout_seconds": 30, "wall_seconds": 1800, "minimum_free_disk_bytes": 536870912,
          "retries": 0}
CONTRACT = {"request_id": 335, "stage": "fixed_12_anchors_full_window_technical_only",
            "source_plan_sha256": PLAN_SHA, "limits": LIMITS,
            "selection": "same_original_12_lexical_anchors_all_24_full_census_descriptors",
            "scheduling": "round_robin_one_page_per_active_pair_in_original_order",
            "feed": "sip", "symbol_mapping": "disabled_asof_minus",
            "clock": "provider_RFC3339_preserved_not_aliased_to_SIP_or_receipt",
            "HTTP_policy": "any_error_stop_all_no_retry_redirect_fallback_or_resume",
            "page_cap_policy": "retain_pair_in_denominator_and_continue_other_pairs",
            "raw_evidence": "private_local_0700_directory_0600_files_no_redistribution",
            "additional_cost_authorized_USD": 0, "full_2470_collection_enabled": False,
            "source_ready": False, "economic_labels_enabled": False,
            "profitability_validation_complete": False,
            "unverified_requirements": ["provider_clock_SIP_or_receipt_semantics",
                                        "SIP_NBBO_condition_eligibility",
                                        "trade_historical_size_units",
                                        "correction_cancel_delivery_chronology",
                                        "equal_clock_sequence_order",
                                        "original_symbol_OTC_coverage",
                                        "vendor_loss_or_omission",
                                        "causal_execution_and_account_contract"]}
TERMINAL = "TERMINAL_PAGE_OBSERVED_SOURCE_UNQUALIFIED"


def selected_requests(plan):
    if sha(canonical(plan)) != PLAN_SHA:
        raise ValueError("pinned_source_plan_required")
    lookup = {(r["trading_day"], r["ticker"], r["stream"]): r for r in plan["full_census"]}
    result = [copy.deepcopy(lookup[(q["trading_day"], q["ticker"], q["stream"])]) for q in plan["qualification"]]
    if len(result) != 24 or len(lookup) != 2470:
        raise ValueError("original_denominators_required")
    return result


def checkpoint(root, manifest):
    """Crash evidence only; a checkpoint never authorizes automatic resumption."""
    temp = root / "progress.tmp"
    with temp.open("wb") as handle:
        os.chmod(temp, 0o600)
        handle.write(canonical(manifest))
    temp.replace(root / "progress.json")


class Transport:
    def __init__(self, root, credentials, get=None, clock=None, sleep=None, disk=None):
        self.root, self.credentials = root, credentials
        self.get, self.clock, self.sleep = get or requests.get, clock or time.monotonic, sleep or time.sleep
        self.disk = disk or (lambda: shutil.disk_usage(root).free)
        self.started, self.last = self.clock(), None
        self.attempts, self.bytes = 0, 0

    def budget_stop(self):
        if self.attempts >= LIMITS["http_attempts"]:
            return "HTTP_BUDGET_EXHAUSTED"
        if self.bytes >= LIMITS["total_retained_response_bytes"]:
            return "BYTE_BUDGET_EXHAUSTED"
        if self.clock() - self.started >= LIMITS["wall_seconds"]:
            return "WALL_BUDGET_EXHAUSTED"
        if self.disk() < LIMITS["minimum_free_disk_bytes"] + LIMITS["bytes_per_page"]:
            return "DISK_BUDGET_EXHAUSTED"
        return None

    def fetch(self, request, token, name):
        if self.last is not None:
            self.sleep(max(0., LIMITS["minimum_interval_seconds"] - (self.clock() - self.last)))
        stop = self.budget_stop()
        if stop:
            return None, None, stop
        self.last = self.clock()
        self.attempts += 1
        meta = {"attempt": self.attempts, "start_elapsed_seconds": self.last - self.started,
                "request_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": None, "wire_file": None, "wire_complete": False,
                "request_token_sha256": sha(token.encode()) if token else None,
                "next_token_sha256": None, "observation": None}
        response, body, state = None, bytearray(), "RECEIVED"
        try:
            response = self.get("https://" + HOST + request["path"],
                                params={**request["params"], **({"page_token": token} if token else {})},
                                headers={"APCA-API-KEY-ID": self.credentials[0],
                                         "APCA-API-SECRET-KEY": self.credentials[1]},
                                timeout=LIMITS["timeout_seconds"], stream=True, allow_redirects=False)
            meta["http_status"] = int(response.status_code)
            bound = min(LIMITS["bytes_per_page"], LIMITS["total_retained_response_bytes"] - self.bytes)
            for chunk in response.iter_content(chunk_size=4096):
                if self.clock() - self.started >= LIMITS["wall_seconds"]:
                    state = "PARTIAL_BODY_WALL_BUDGET"
                    break
                if len(body) + len(chunk) > bound:
                    body.extend(chunk[:bound - len(body)])
                    state = "PARTIAL_BODY_BYTE_BUDGET"
                    break
                body.extend(chunk)
            else:
                meta["wire_complete"] = True
        except requests.RequestException:
            state = "TRANSPORT_ERROR"  # Exception text can contain authentication.
        finally:
            if response is not None:
                response.close()
        wire = bytes(body)
        self.bytes += len(wire)
        meta.update(body_bytes=len(wire), wire_sha256=sha(wire),
                    end_elapsed_seconds=self.clock() - self.started,
                    response_utc=datetime.now(timezone.utc).isoformat())
        payload = None
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
        else:
            meta["wire_file"] = "pages/" + name + ".wire"
            save_private(self.root, meta["wire_file"], wire)
        token_out = None
        if state == "RECEIVED":
            if meta["http_status"] != 200:
                state = "HTTP_DENIED_OR_ERROR"
            else:
                try:
                    meta["observation"], token_out = inspect_page(payload, request)
                    meta["next_token_sha256"] = sha(token_out.encode()) if token_out else None
                except (ValueError, TypeError):
                    state = "INVALID_MARKET_SCHEMA"
        meta["transport_state"] = state
        return meta, token_out, state


def collect(plan, root, *, credentials, attestation, get=None, clock=None, sleep=None, disk=None):
    requests_fixed = selected_requests(plan)
    if root.exists():
        raise ValueError("fresh_private_directory_required")
    if len(attestation) > 16384:
        raise ValueError("attestation_byte_limit")
    verify_attestation(attestation)
    if len(credentials) != 2 or any(not isinstance(s, str) or not s or "\r" in s or "\n" in s for s in credentials):
        raise ValueError("approved_private_credentials_required")
    root.mkdir(parents=True, mode=0o700)
    save_private(root, "contract.json", canonical(CONTRACT))
    save_private(root, "attestation.json", attestation)
    manifest = {"request_id": 335, "stage": CONTRACT["stage"], "source_plan_sha256": PLAN_SHA,
                "contract_sha256": sha(canonical(CONTRACT)),
                "implementation_file_sha256": sha(Path(__file__).read_bytes()),
                "attestation_sha256": sha(attestation),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "pairs": [{"request": r, "pages": [], "state": "ACTIVE"} for r in requests_fixed],
                "http_attempts": 0, "retained_response_bytes": 0, "stop_reason": None,
                "source_ready": False, "new_economic_labels": 0,
                "full_census_collected": False, "profitability_validation_complete": False}
    transport = Transport(root, credentials, get, clock, sleep, disk)
    tokens, seen, stop = [None] * 24, [set() for _ in range(24)], None
    checkpoint(root, manifest)
    while any(p["state"] == "ACTIVE" for p in manifest["pairs"]) and stop is None:
        for index, pair in enumerate(manifest["pairs"]):
            if pair["state"] != "ACTIVE":
                continue
            page, token, state = transport.fetch(pair["request"], tokens[index],
                                                  f"{index:02d}-{len(pair['pages']):02d}")
            if page is not None:
                pair["pages"].append(page)
            if state != "RECEIVED":
                pair["state"], stop = state, state
            elif token is None:
                pair["state"] = TERMINAL
            elif token in seen[index]:
                pair["state"], stop = "REPEATED_PAGINATION_TOKEN", "REPEATED_PAGINATION_TOKEN"
            elif len(pair["pages"]) >= LIMITS["pages_per_pair"]:
                pair["state"] = "PAGE_CAP_INCOMPLETE"
            else:
                seen[index].add(token)
                tokens[index] = token
            manifest.update(http_attempts=transport.attempts, retained_response_bytes=transport.bytes,
                            stop_reason=stop, elapsed_seconds=transport.clock() - transport.started)
            checkpoint(root, manifest)
            if stop:
                break
    for pair in manifest["pairs"]:
        if pair["state"] == "ACTIVE":
            pair["state"] = "INCOMPLETE_AFTER_" + stop
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint(root, manifest)
    save_private(root, "manifest.json", canonical(manifest))
    return manifest


def replay(plan, root):
    requests_fixed = selected_requests(plan)
    manifest = strict_json(confined(root, "manifest.json").read_bytes())
    if (manifest["source_plan_sha256"] != PLAN_SHA or
            manifest["contract_sha256"] != sha(canonical(CONTRACT)) or
            confined(root, "contract.json").read_bytes() != canonical(CONTRACT) or
            manifest["implementation_file_sha256"] != sha(Path(__file__).read_bytes())):
        raise ValueError("registered_source_contract_or_implementation_changed")
    data = confined(root, "attestation.json").read_bytes()
    if len(data) > 16384 or sha(data) != manifest["attestation_sha256"]:
        raise ValueError("attestation_evidence_changed")
    verify_attestation(data)
    if any(manifest.get(f) is not False for f in
           ("source_ready", "full_census_collected", "profitability_validation_complete")) or manifest.get("new_economic_labels") != 0:
        raise ValueError("technical_acquisition_cannot_claim_economic_success")
    pairs = manifest["pairs"]
    if len(pairs) != 24:
        raise ValueError("original_pair_denominator_changed")
    stop = manifest["stop_reason"]
    allowed_stops = {"HTTP_BUDGET_EXHAUSTED", "BYTE_BUDGET_EXHAUSTED", "WALL_BUDGET_EXHAUSTED",
                     "DISK_BUDGET_EXHAUSTED", "PARTIAL_BODY_WALL_BUDGET", "PARTIAL_BODY_BYTE_BUDGET",
                     "TRANSPORT_ERROR", "INVALID_JSON", "SENSITIVE_BODY_WITHHELD",
                     "HTTP_DENIED_OR_ERROR", "INVALID_MARKET_SCHEMA", "REPEATED_PAGINATION_TOKEN"}
    if stop is not None and stop not in allowed_stops:
        raise ValueError("unregistered_stop_reason")
    attempts, output, total = [], [], 0
    expected_order = []
    for round_index in range(LIMITS["pages_per_pair"]):
        expected_order.extend((i, round_index) for i, p in enumerate(pairs) if len(p["pages"]) > round_index)
    for index, (pair, request) in enumerate(zip(pairs, requests_fixed)):
        if pair["request"] != request or len(pair["pages"]) > LIMITS["pages_per_pair"]:
            raise ValueError("registered_pair_changed")
        prior, previous_last, tokens = None, None, set()
        rows, backwards, equal, terminal = 0, 0, 0, False
        within_backwards, within_equal, duplicates, missing, invalid = 0, 0, 0, Counter(), Counter()
        failure = None
        for page_index, page in enumerate(pair["pages"]):
            if terminal or failure or page["request_token_sha256"] != prior:
                raise ValueError("broken_or_post_terminal_page_chain")
            attempts.append((page["attempt"], index, page_index, page))
            count = page["body_bytes"]
            if type(count) is not int or not 0 <= count <= LIMITS["bytes_per_page"]:
                raise ValueError("invalid_wire_byte_count")
            total += count
            state = page["transport_state"]
            if page["wire_file"] is None:
                if not page["wire_evidence_withheld"] or page["observation"] is not None or state != "SENSITIVE_BODY_WITHHELD":
                    raise ValueError("missing_or_withheld_wire_cannot_support_observations")
                failure = state
                continue
            if page["wire_file"] != f"pages/{index:02d}-{page_index:02d}.wire":
                raise ValueError("registered_wire_path_changed")
            wire = confined(root, page["wire_file"]).read_bytes()
            if len(wire) != count or sha(wire) != page["wire_sha256"]:
                raise ValueError("wire_size_or_digest_changed")
            if state != "RECEIVED":
                if page["observation"] is not None or page["next_token_sha256"] is not None:
                    raise ValueError("failed_transport_cannot_support_observations")
                failure = state
                continue
            if page["http_status"] != 200 or not page["wire_complete"]:
                raise ValueError("unsuccessful_page_cannot_support_observations")
            observed, token = inspect_page(checked_payload(wire), request)
            digest = sha(token.encode()) if token else None
            if observed != page["observation"] or digest != page["next_token_sha256"]:
                raise ValueError("observation_or_token_changed")
            first, last = observed["first_provider_ns"], observed["last_provider_ns"]
            if first is not None and previous_last is not None:
                backwards += int(first < previous_last)
                equal += int(first == previous_last)
            if last is not None:
                previous_last = last
            rows += observed["rows"]
            within_backwards += observed["within_page_backwards"]
            within_equal += observed["within_page_equal_clock"]
            duplicates += observed["within_page_duplicate_rows"]
            missing.update(observed["missing_fields"])
            invalid.update(observed["invalid_positive_numeric_fields"])
            if token is not None and digest in tokens:
                failure = "REPEATED_PAGINATION_TOKEN"
            if token is not None:
                tokens.add(digest)
            terminal, prior = token is None, digest
        if (pair["state"] == TERMINAL) != terminal or (failure is not None and pair["state"] != failure):
            raise ValueError("terminal_or_failure_state_changed")
        if pair["state"] == "PAGE_CAP_INCOMPLETE" and (terminal or failure or len(pair["pages"]) != LIMITS["pages_per_pair"]):
            raise ValueError("page_cap_state_changed")
        allowed_states = {TERMINAL, "PAGE_CAP_INCOMPLETE"}
        if stop is not None:
            allowed_states.update((stop, "INCOMPLETE_AFTER_" + stop))
        if pair["state"] not in allowed_states or (failure is not None and failure != stop):
            raise ValueError("unregistered_pair_state")
        output.append({"trading_day": request["trading_day"], "ticker": request["ticker"],
                       "stream": request["stream"], "identity_indices": request["identity_indices"],
                       "start_ns": request["start_ns"], "end_ns": request["end_ns"],
                       "pages": len(pair["pages"]), "collection_state": pair["state"],
                       "rows_observed": rows, "terminal_page_observed": terminal,
                       "cross_page_backwards": backwards, "cross_page_equal_clock": equal,
                       "within_page_backwards": within_backwards, "within_page_equal_clock": within_equal,
                       "within_page_duplicate_rows": duplicates, "missing_fields": dict(missing),
                       "nonpositive_or_invalid_numeric_fields": dict(invalid), "source_ready": False})
    attempts.sort(key=lambda a: a[0])
    if [(i, p) for _, i, p, _ in attempts] != expected_order:
        raise ValueError("round_robin_schedule_changed")
    previous_start = None
    for ordinal, (number, _, _, page) in enumerate(attempts, 1):
        start, end = page["start_elapsed_seconds"], page["end_elapsed_seconds"]
        if number != ordinal or not 0 <= start <= end or start >= LIMITS["wall_seconds"]:
            raise ValueError("attempt_or_transport_clock_changed")
        if previous_start is not None and start - previous_start < LIMITS["minimum_interval_seconds"] - 1e-6:
            raise ValueError("request_rate_contract_violated")
        previous_start = start
    if (len(attempts) != manifest["http_attempts"] or total != manifest["retained_response_bytes"] or
            len(attempts) > LIMITS["http_attempts"] or total > LIMITS["total_retained_response_bytes"]):
        raise ValueError("aggregate_budget_evidence_changed")
    return {"request_id": 335, "stage": "offline_full_window_technical_replay_only", "pairs": output,
            "manifest_sha256": sha(canonical(manifest)), "contract_sha256": sha(canonical(CONTRACT)),
            "http_attempts": len(attempts), "retained_response_bytes": total,
            "started_utc": manifest["started_utc"], "finished_utc": manifest["finished_utc"],
            "elapsed_seconds": manifest["elapsed_seconds"], "stop_reason": manifest["stop_reason"],
            "original_identity_denominator": 1235, "original_partition_stream_denominator": 2470,
            "technical_partitions_attempted": sum(bool(p["pages"]) for p in pairs),
            "terminal_technical_partitions": sum(p["terminal_page_observed"] for p in output),
            "original_partitions_not_requested": 2470 - sum(bool(p["pages"]) for p in pairs),
            "source_ready": False, "full_census_collected": False, "new_economic_labels": 0,
            "profitability_validation_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("contract", "collect", "replay"))
    parser.add_argument("--plan", type=Path, default=Path("research/request335-free-source-plan.json.gz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--attestation", type=Path)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.mode == "contract":
        selected_requests(plan)
        args.output.write_bytes(canonical(CONTRACT))
        result = {"http_attempts": 0, "stage": "offline_contract_only"}
    elif args.mode == "replay":
        result = replay(plan, args.output)
    else:
        if args.credentials is None or args.attestation is None:
            parser.error("approved private credential and attestation files required")
        if args.credentials.stat().st_mode & 0o077 or args.credentials.stat().st_size > 16384:
            parser.error("small private credential file required")
        credentials = strict_json(args.credentials.read_bytes())
        result = collect(plan, args.output,
                         credentials=(credentials.get("APCA_API_KEY_ID"), credentials.get("APCA_API_SECRET_KEY")),
                         attestation=args.attestation.read_bytes())
    print(json.dumps({"stage": result["stage"], "http_attempts": result["http_attempts"],
                      "source_ready": False, "new_economic_labels": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
