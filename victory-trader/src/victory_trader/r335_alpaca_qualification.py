"""R335 bounded SIP access evidence. No trades, labels, fits, or paid fallback.

CLI production input is pinned. Tests inject synthetic transports; they never
establish source equivalence. Raw evidence stays private on the local filesystem.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import requests

from .offline_source_inventory import canonical, confined, sha, strict_json
from .r335_free_source_plan import request_descriptor
from .tick_execution_readiness import exact_decimal, rfc3339_ns

PLAN_SHA = "0ad2603a2a62ef49a975f21c5951e55fa6708fbe0d062bf5df5c832bb88c235e"
HOST = "data.alpaca.markets"
LIMITS = {"http_attempts": 48, "pages_per_pair": 2, "bytes_per_page": 2097152,
          "total_response_bytes": 33554432, "minimum_interval_seconds": 12.5,
          "timeout_seconds": 30, "retries": 0}
UNVERIFIED = ["historical_timestamp_semantics", "official_NBBO_equivalence",
              "historical_size_units_and_round_lots", "correction_cancel_chronology",
              "equal_clock_sequence_order", "original_symbol_OTC_coverage",
              "full_window_completeness", "causal_execution_and_account_contract"]
CONTRACT = {"request_id": 335, "stage": "bounded_free_access_qualification_only",
            "source_plan_sha256": PLAN_SHA, "limits": LIMITS,
            "selection": "original_12_lexical_anchors_first_60_seconds_trades_quotes",
            "feed": "sip", "symbol_mapping": "disabled_asof_minus",
            "clock": "provider_RFC3339_preserved_not_aliased_to_SIP",
            "unverified_requirements": UNVERIFIED,
            "HTTP_policy": "401_429_stop_all_403_stop_stream_no_retries_no_redirects",
            "full_collection_enabled": False, "additional_cost_authorized_USD": 0,
            "economic_labels_enabled": False, "raw_evidence": "private_local_0600_only"}
SENSITIVE = {"authorization", "api_key", "apikey", "secret", "password", "access_token",
             "apca-api-key-id", "apca-api-secret-key"}


def load_plan(path: Path) -> dict:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("source_plan_byte_limit")
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as handle:
        data = handle.read(2 * 1024 * 1024 + 1)
    if sha(data) != PLAN_SHA:
        raise ValueError("registered_source_plan_changed")
    plan = strict_json(data)
    if plan["limits_for_proposed_qualification"] != LIMITS or len(plan["qualification"]) != 24:
        raise ValueError("registered_qualification_changed")
    return plan


def save_private(root: Path, name: str, data: bytes) -> None:
    path = confined(root, name)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as handle:
        os.chmod(path, 0o600)
        handle.write(data)


def checked_payload(data: bytes) -> dict:
    # Decimal keeps prices exact during inspection; only raw bytes are persisted.
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_JSON_key")
            result[key] = value
        return result
    def bad(value):
        raise ValueError("nonfinite_JSON")
    result = json.loads(data, parse_float=Decimal, parse_constant=bad, object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError("object_response_required")
    return result


def sensitive_payload(value, secrets: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        return any(k.lower() in SENSITIVE or sensitive_payload(k, secrets) or
                   sensitive_payload(v, secrets) for k, v in value.items())
    if isinstance(value, list):
        return any(sensitive_payload(v, secrets) for v in value)
    return isinstance(value, str) and any(s and s in value for s in secrets)


def inspect_page(payload: dict, request: dict) -> tuple[dict, str | None]:
    stream, ticker = request["stream"], request["ticker"]
    if "next_page_token" not in payload:
        raise ValueError("missing_terminal_marker")
    token = payload["next_page_token"]
    if token is not None and (not isinstance(token, str) or not token or len(token) > 4096):
        raise ValueError("invalid_pagination_token")
    rows_by_symbol = payload.get(stream)
    if not isinstance(rows_by_symbol, dict) or set(rows_by_symbol) - {ticker}:
        raise ValueError("unexpected_symbol_or_stream_schema")
    rows = rows_by_symbol.get(ticker, [])
    if not isinstance(rows, list) or len(rows) > request["params"]["limit"]:
        raise ValueError("invalid_row_count")
    fields = (["p", "s", "x", "i", "c", "z"] if stream == "trades" else
              ["bp", "ap", "bs", "as", "bx", "ax", "c", "z"])
    times, missing, seen, duplicates, precision = [], Counter(), set(), 0, Counter()
    invalid_numbers = Counter()
    numeric_fields = ["p", "s"] if stream == "trades" else ["bp", "ap", "bs", "as"]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("object_row_required")
        timestamp = rfc3339_ns(row.get("t"))
        if not request["start_ns"] <= timestamp < request["end_ns"]:
            raise ValueError("row_outside_original_window")
        match = re.search(r"\.(\d+)(?:Z|[+-]\d{2}:\d{2})$", row["t"])
        precision[str(len(match.group(1)) if match else 0)] += 1
        for field in fields:
            missing[field] += int(row.get(field) is None)
        for field in numeric_fields:
            if row.get(field) is not None:
                try:
                    invalid_numbers[field] += int(exact_decimal(row[field]) <= 0)
                except ValueError:
                    invalid_numbers[field] += 1
        fingerprint = repr(sorted(row.items()))
        duplicates += int(fingerprint in seen)
        seen.add(fingerprint)
        times.append(timestamp)
    return {"rows": len(rows), "first_provider_ns": times[0] if times else None,
            "last_provider_ns": times[-1] if times else None,
            "within_page_backwards": sum(b < a for a, b in zip(times, times[1:])),
            "within_page_equal_clock": sum(a == b for a, b in zip(times, times[1:])),
            "within_page_duplicate_rows": duplicates, "missing_fields": dict(missing),
            "invalid_positive_numeric_fields": dict(invalid_numbers),
            "fractional_digits_observed": dict(precision),
            "timestamp_semantics": "UNVERIFIED_PROVIDER_CLOCK",
            "source_ready": False}, token


def verify_attestation(data: bytes) -> dict:
    value = strict_json(data)
    required = ("account_is_free_Basic", "no_incremental_charge",
                "historical_SIP_research_and_local_retention_permitted",
                "no_redistribution", "only_registered_technical_sample")
    if not isinstance(value, dict) or any(value.get(key) is not True for key in required):
        raise ValueError("explicit_free_account_license_attestation_required")
    if not isinstance(value.get("rights_evidence_reference"), str) or not value["rights_evidence_reference"].strip():
        raise ValueError("license_evidence_reference_required")
    # The manifest carries its hash, never private account/license details.
    return value


class Transport:
    def __init__(self, root, credentials, get=None, clock=None, sleep=None):
        self.root, self.credentials = root, credentials
        self.get = get or requests.get
        self.clock, self.sleep = clock or time.monotonic, sleep or time.sleep
        self.attempts, self.bytes, self.last_start = 0, 0, None

    def fetch(self, request, token, name):
        params = {**request["params"], **({"page_token": token} if token else {})}
        meta = {"attempt": None, "http_status": None, "wire_file": None,
                "wire_sha256": None, "wire_complete": False, "body_bytes": 0,
                "request_token_sha256": sha(token.encode()) if token else None,
                "next_token_sha256": None, "observation": None}
        if self.attempts >= LIMITS["http_attempts"] or self.bytes >= LIMITS["total_response_bytes"]:
            return meta, None, "BUDGET_EXHAUSTED"
        if self.last_start is not None:
            self.sleep(max(0., LIMITS["minimum_interval_seconds"] - (self.clock()-self.last_start)))
        self.last_start = self.clock()
        self.attempts += 1
        meta.update(attempt=self.attempts, request_utc=datetime.now(timezone.utc).isoformat())
        response, body = None, bytearray()
        state = "RECEIVED"
        try:
            response = self.get("https://"+HOST+request["path"], params=params,
                                headers={"APCA-API-KEY-ID": self.credentials[0],
                                         "APCA-API-SECRET-KEY": self.credentials[1]},
                                timeout=LIMITS["timeout_seconds"], stream=True, allow_redirects=False)
            meta["http_status"] = int(response.status_code)
            bound = min(LIMITS["bytes_per_page"], LIMITS["total_response_bytes"]-self.bytes)
            for chunk in response.iter_content(chunk_size=4096):
                if len(body)+len(chunk) > bound:
                    body.extend(chunk[:bound-len(body)])
                    state = "PARTIAL_BODY_BUDGET"
                    break
                body.extend(chunk)
            else:
                meta["wire_complete"] = True
        except requests.RequestException:
            # Exception text may contain keys/URLs; never serialize it.
            state = "TRANSPORT_ERROR"
        finally:
            if response is not None:
                response.close()
        wire = bytes(body)
        self.bytes += len(wire)
        meta.update(body_bytes=len(wire), wire_sha256=sha(wire),
                    response_utc=datetime.now(timezone.utc).isoformat())
        payload = None
        try:
            payload = checked_payload(wire)
        except (ValueError, UnicodeError, RecursionError):
            if state == "RECEIVED":
                state = "INVALID_JSON"
        unsafe = any(s.encode() in wire for s in self.credentials) or (
            payload is not None and sensitive_payload(payload, self.credentials))
        if not unsafe:
            meta["wire_file"] = "pages/"+name+".wire"
            save_private(self.root, meta["wire_file"], wire)
        else:
            state, payload = "SENSITIVE_BODY_WITHHELD", None
        meta["wire_evidence_withheld"] = unsafe
        if state == "RECEIVED" and meta["http_status"] == 200:
            try:
                meta["observation"], next_token = inspect_page(payload, request)
                meta["next_token_sha256"] = sha(next_token.encode()) if next_token else None
                return meta, next_token, state
            except (ValueError, TypeError):
                state = "INVALID_MARKET_SCHEMA"
        return meta, None, state


def collect(plan, root, *, credentials=(), attestation=None, get=None, clock=None, sleep=None):
    if root.exists():
        raise ValueError("fresh_private_output_directory_required")
    attested = attestation is not None
    if attested:
        verify_attestation(attestation)
    if credentials and (len(credentials) != 2 or not all(credentials) or not attested):
        raise ValueError("credentials_require_explicit_free_license_attestation")
    if any(not isinstance(s, str) or "\r" in s or "\n" in s for s in credentials):
        raise ValueError("invalid_header_credential")
    # Disallow arbitrary endpoint/parameter changes, even for injected callers.
    for req in plan["qualification"]:
        expected = request_descriptor(req["trading_day"], req["ticker"], req["stream"],
                                      req["start_ns"], req["end_ns"], 100)
        if req != expected or not req["trading_day"].startswith("2026-05-") or req["end_ns"]-req["start_ns"] > 60000000000:
            raise ValueError("registered_May_technical_descriptor_required")
    root.mkdir(parents=True, mode=0o700)
    save_private(root, "contract.json", canonical(CONTRACT))
    if attested:
        save_private(root, "attestation.json", attestation)
    transport = Transport(root, credentials, get, clock, sleep)
    manifest = {"request_id": 335, "stage": CONTRACT["stage"],
                "source_plan_sha256": sha(canonical(plan)), "contract_sha256": sha(canonical(CONTRACT)),
                "implementation_file_sha256": sha(Path(__file__).read_bytes()),
                "attestation_sha256": sha(attestation) if attested else None,
                "credential_configured": bool(credentials), "pairs": []}
    stop, denied = None, set()
    for index, request in enumerate(plan["qualification"]):
        pair = {"request": request, "pages": [], "state": None}
        manifest["pairs"].append(pair)
        stream = request["stream"]
        if not credentials or stop or stream in denied:
            pair["state"] = ("NOT_REQUESTED_MISSING_CREDENTIAL" if not credentials else
                             "SKIPPED_AFTER_"+stop if stop else "SKIPPED_AFTER_STREAM_DENIAL")
            continue
        token, seen = None, set()
        for page_number in range(LIMITS["pages_per_pair"]):
            meta, next_token, state = transport.fetch(request, token, f"{index:02d}-{page_number:02d}")
            pair["pages"].append(meta)
            status = meta["http_status"]
            if status in (401, 429):
                pair["state"], stop = f"DENIED_{status}", f"HTTP_{status}"
                break
            if status == 403:
                pair["state"] = "DENIED_403"
                denied.add(stream)
                break
            if state != "RECEIVED" or status != 200:
                pair["state"] = state if state != "RECEIVED" else f"HTTP_{status}"
                if state in ("BUDGET_EXHAUSTED", "PARTIAL_BODY_BUDGET", "SENSITIVE_BODY_WITHHELD"):
                    stop = state
                break
            if next_token is None:
                pair["state"] = "TERMINAL_PAGE_OBSERVED_SOURCE_UNQUALIFIED"
                break
            if next_token in seen:
                pair["state"] = "REPEATED_PAGINATION_TOKEN"
                break
            seen.add(next_token)
            token = next_token
        else:
            pair["state"] = "PAGE_CAP_INCOMPLETE"
    manifest.update(http_attempts=transport.attempts, retained_response_bytes=transport.bytes,
                    source_ready=False, new_economic_labels=0, profitability_validation_complete=False,
                    full_census_collected=False, unverified_requirements=UNVERIFIED)
    save_private(root, "manifest.json", canonical(manifest))
    return manifest


def replay(plan, root):
    manifest_path = confined(root, "manifest.json")
    if manifest_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("manifest_byte_limit")
    manifest = strict_json(manifest_path.read_bytes())
    if manifest["source_plan_sha256"] != sha(canonical(plan)) or manifest["contract_sha256"] != sha(canonical(CONTRACT)):
        raise ValueError("registered_plan_or_contract_mismatch")
    if confined(root, "contract.json").read_bytes() != canonical(CONTRACT):
        raise ValueError("contract_evidence_changed")
    if manifest.get("implementation_file_sha256") != sha(Path(__file__).read_bytes()):
        raise ValueError("replay_requires_original_implementation")
    if manifest.get("attestation_sha256") is not None:
        path = confined(root, "attestation.json")
        if path.stat().st_size > 16384:
            raise ValueError("attestation_byte_limit")
        data = path.read_bytes()
        if sha(data) != manifest["attestation_sha256"]:
            raise ValueError("attestation_evidence_changed")
        verify_attestation(data)
    if manifest.get("credential_configured") is not False and manifest.get("attestation_sha256") is None:
        raise ValueError("credentialed_run_requires_attestation_evidence")
    pairs = manifest["pairs"]
    if any(manifest.get(field) is not False for field in
           ("source_ready", "profitability_validation_complete", "full_census_collected")) or manifest.get("new_economic_labels") != 0:
        raise ValueError("qualification_cannot_claim_economic_success")
    if len(pairs) != len(plan["qualification"]):
        raise ValueError("original_pair_denominator_changed")
    attempts, total, output = 0, 0, []
    for pair, request in zip(pairs, plan["qualification"]):
        if pair["request"] != request or len(pair["pages"]) > LIMITS["pages_per_pair"]:
            raise ValueError("registered_pair_changed")
        prior_token, previous_last, rows = None, None, 0
        backwards, equal, terminal = 0, 0, False
        for page in pair["pages"]:
            if page["request_token_sha256"] != prior_token or terminal:
                raise ValueError("broken_or_post_terminal_page_chain")
            if page["attempt"] is None:
                if pair["state"] != "BUDGET_EXHAUSTED":
                    raise ValueError("unattempted_page_claim")
                continue
            attempts += 1
            if page["attempt"] != attempts:
                raise ValueError("HTTP_attempt_order_changed")
            count = page["body_bytes"]
            if type(count) is not int or not 0 <= count <= LIMITS["bytes_per_page"]:
                raise ValueError("invalid_wire_byte_count")
            total += count
            if page["wire_file"] is None:
                if not page.get("wire_evidence_withheld"):
                    raise ValueError("missing_wire_evidence")
                if page["observation"] is not None:
                    raise ValueError("withheld_wire_cannot_support_observation")
                continue
            path = confined(root, page["wire_file"])
            if not path.is_file() or path.stat().st_size != count:
                raise ValueError("wire_file_size_changed")
            wire = path.read_bytes()
            if sha(wire) != page["wire_sha256"]:
                raise ValueError("wire_digest_changed")
            if page["observation"] is None:
                continue
            if page["http_status"] != 200 or not page["wire_complete"]:
                raise ValueError("unsuccessful_page_cannot_support_observation")
            observed, token = inspect_page(checked_payload(wire), request)
            digest = sha(token.encode()) if token else None
            if observed != page["observation"] or digest != page["next_token_sha256"]:
                raise ValueError("page_observation_or_token_changed")
            first, last = observed["first_provider_ns"], observed["last_provider_ns"]
            if first is not None and previous_last is not None:
                backwards += int(first < previous_last)
                equal += int(first == previous_last)
            if last is not None:
                previous_last = last
            rows += observed["rows"]
            terminal, prior_token = token is None, digest
        if (pair["state"] == "TERMINAL_PAGE_OBSERVED_SOURCE_UNQUALIFIED") != terminal:
            raise ValueError("terminal_state_changed")
        if not pair["pages"] and pair["state"] == "PAGE_CAP_INCOMPLETE":
            raise ValueError("empty_pair_cannot_be_page_capped")
        output.append({"ticker": request["ticker"], "trading_day": request["trading_day"],
                       "stream": request["stream"], "collection_state": pair["state"],
                       "rows_observed": rows, "terminal_page_observed": terminal,
                       "cross_page_backwards": backwards, "cross_page_equal_clock": equal,
                       "source_ready": False})
    if attempts != manifest["http_attempts"] or total != manifest["retained_response_bytes"] or attempts > LIMITS["http_attempts"] or total > LIMITS["total_response_bytes"]:
        raise ValueError("aggregate_budget_evidence_changed")
    return {"request_id": 335, "stage": "offline_evidence_replay_only", "pairs": output,
            "http_attempts": attempts, "retained_response_bytes": total,
            "source_ready": False, "new_economic_labels": 0,
            "profitability_validation_complete": False, "unverified_requirements": UNVERIFIED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "collect", "replay"))
    parser.add_argument("--plan", type=Path, default=Path("research/request335-free-source-plan.json.gz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attestation", type=Path)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.mode == "replay":
        result = replay(plan, args.output)
    elif args.mode == "preflight":
        result = collect(plan, args.output)
    else:
        if args.attestation is None or args.attestation.stat().st_size > 16384:
            parser.error("small explicit attestation file required")
        attestation = args.attestation.read_bytes()
        verify_attestation(attestation)
        credentials = (os.environ.get("APCA_API_KEY_ID", ""), os.environ.get("APCA_API_SECRET_KEY", ""))
        if not all(credentials):
            parser.error("APCA_API_KEY_ID and APCA_API_SECRET_KEY required; values never printed")
        result = collect(plan, args.output, credentials=credentials, attestation=attestation)
    # Only summary counts reach stdout. Do not expose error bodies or raw ticks.
    print(json.dumps({"stage": result["stage"], "http_attempts": result["http_attempts"],
                      "pairs": len(result["pairs"]), "source_ready": False,
                      "new_economic_labels": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
