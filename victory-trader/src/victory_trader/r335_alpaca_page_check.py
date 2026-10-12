"""Separately registered re-query of the sole capped R335 technical pair.

The old acquisition remains immutable and incomplete. A fresh larger-page
chain is compared with its retained prefix; no old token is resumed or spliced.
No full-window/census, execution label, paid endpoint or automatic retry.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .offline_source_inventory import canonical, confined, sha, strict_json
from .r335_alpaca_qualification import (
    Transport, checked_payload, inspect_page, load_plan, replay as parent_replay,
    save_private, verify_attestation,
)
from .r335_free_source_plan import request_descriptor
from .r335_tick_semantics_audit import audit_rows, row_key

PARENT_SHA = "44b5f124d1ad2533dc42a800d9032c64fbffae3d7b7c5f1938db0b88e8a20507"
COLLECTOR_SHA = "b1e3bb1dd6909c8e83f92adaa2070f60dbe2dcbb5ef1569a4af7aad0daec18d1"
REQUEST = request_descriptor("2026-05-05", "AIRS", "quotes",
                             1777990800000000000, 1777990860000000000, 10000)
CONTRACT = {"request_id": 335, "stage": "separate_capped_pair_technical_requery_only",
            "parent_manifest_sha256": PARENT_SHA, "request": REQUEST,
            "limits": {"http_attempts": 2, "pages_per_pair": 2, "bytes_per_page": 2097152,
                       "total_response_bytes": 4194304, "minimum_interval_seconds": 12.5,
                       "timeout_seconds": 30, "retries": 0},
            "selection": "sole_PAGE_CAP_INCOMPLETE_pair_from_fixed_original_sample",
            "comparison": "all_original_retained_rows_vs_fresh_chain_prefix_in_original_order",
            "HTTP_policy": "any_error_stop_no_retries_no_redirects_no_fallback",
            "old_acquisition_mutation_permitted": False, "full_collection_enabled": False,
            "additional_cost_authorized_USD": 0, "economic_labels_enabled": False,
            "raw_evidence": "private_local_0600_only"}


def source_prefix(plan, parent):
    if sha(confined(parent, "manifest.json").read_bytes()) != PARENT_SHA:
        raise ValueError("pinned_parent_manifest_required")
    parent_replay(plan, parent)
    manifest = strict_json(confined(parent, "manifest.json").read_bytes())
    capped = [pair for pair in manifest["pairs"] if pair["state"] == "PAGE_CAP_INCOMPLETE"]
    original_request = {**REQUEST, "params": {**REQUEST["params"], "limit": 100}}
    if len(capped) != 1 or capped[0]["request"] != original_request:
        raise ValueError("sole_original_capped_pair_required")
    rows = []
    for page in capped[0]["pages"]:
        payload = checked_payload(confined(parent, page["wire_file"]).read_bytes())
        rows.extend(payload["quotes"].get("AIRS", []))
    return rows


def collect(plan, parent, root, *, credentials, get=None, clock=None, sleep=None):
    source_prefix(plan, parent)
    attestation = confined(parent, "attestation.json").read_bytes()
    verify_attestation(attestation)
    if root.exists():
        raise ValueError("fresh_private_output_directory_required")
    if len(credentials) != 2 or any(not isinstance(s, str) or not s or "\r" in s or "\n" in s for s in credentials):
        raise ValueError("explicit_valid_private_credentials_required")
    collector = Path(__file__).with_name("r335_alpaca_qualification.py")
    if sha(collector.read_bytes()) != COLLECTOR_SHA:
        raise ValueError("original_frozen_transport_required")
    root.mkdir(parents=True, mode=0o700)
    save_private(root, "contract.json", canonical(CONTRACT))
    save_private(root, "attestation.json", attestation)
    transport = Transport(root, credentials, get, clock, sleep)
    pages, token, seen, state = [], None, set(), "PAGE_CAP_INCOMPLETE"
    for index in range(CONTRACT["limits"]["pages_per_pair"]):
        page, following, received = transport.fetch(REQUEST, token, f"page-check-{index}")
        pages.append(page)
        if received != "RECEIVED" or page["http_status"] != 200:
            state = received if received != "RECEIVED" else "HTTP_DENIED_OR_ERROR"
            break
        if following is None:
            state = "TERMINAL_PAGE_OBSERVED_SOURCE_UNQUALIFIED"
            break
        if following in seen:
            state = "REPEATED_PAGINATION_TOKEN"
            break
        seen.add(following)
        token = following
    manifest = {"contract_sha256": sha(canonical(CONTRACT)),
                "implementation_file_sha256": sha(Path(__file__).read_bytes()),
                "attestation_sha256": sha(attestation), "request": REQUEST, "pages": pages,
                "state": state, "http_attempts": transport.attempts,
                "retained_response_bytes": transport.bytes,
                "source_ready": False, "new_economic_labels": 0,
                "full_census_collected": False, "profitability_validation_complete": False}
    save_private(root, "manifest.json", canonical(manifest))
    return manifest


def replay_check(plan, parent, root):
    original = source_prefix(plan, parent)
    manifest_bytes = confined(root, "manifest.json").read_bytes()
    manifest = strict_json(manifest_bytes)
    if (manifest["contract_sha256"] != sha(canonical(CONTRACT)) or
            confined(root, "contract.json").read_bytes() != canonical(CONTRACT) or
            manifest["request"] != REQUEST or
            manifest["implementation_file_sha256"] != sha(Path(__file__).read_bytes())):
        raise ValueError("page_check_contract_or_implementation_changed")
    attestation = confined(root, "attestation.json").read_bytes()
    if sha(attestation) != manifest["attestation_sha256"]:
        raise ValueError("attestation_changed")
    verify_attestation(attestation)
    if any(manifest.get(k) is not False for k in ("source_ready", "full_census_collected", "profitability_validation_complete")) or manifest.get("new_economic_labels") != 0:
        raise ValueError("technical_check_cannot_claim_economic_success")
    pages = manifest["pages"]
    if not 1 <= len(pages) <= 2 or manifest["http_attempts"] != len(pages):
        raise ValueError("page_check_attempt_budget_changed")
    total, prior_token, terminal, rows = 0, None, False, []
    for index, page in enumerate(pages):
        if terminal or page["attempt"] != index + 1 or page["request_token_sha256"] != prior_token:
            raise ValueError("broken_or_post_terminal_page_chain")
        count = page["body_bytes"]
        if type(count) is not int or not 0 <= count <= CONTRACT["limits"]["bytes_per_page"]:
            raise ValueError("invalid_wire_byte_count")
        total += count
        if page["wire_file"] is None:
            if not page.get("wire_evidence_withheld") or page["observation"] is not None:
                raise ValueError("missing_or_withheld_evidence_claim")
            continue
        wire = confined(root, page["wire_file"]).read_bytes()
        if len(wire) != count or sha(wire) != page["wire_sha256"]:
            raise ValueError("wire_evidence_changed")
        if page["observation"] is None:
            if index + 1 != len(pages):
                raise ValueError("continued_after_error")
            continue
        if page["http_status"] != 200 or not page["wire_complete"]:
            raise ValueError("invalid_observed_page")
        payload = checked_payload(wire)
        observed, token = inspect_page(payload, REQUEST)
        digest = sha(token.encode()) if token else None
        if observed != page["observation"] or digest != page["next_token_sha256"]:
            raise ValueError("page_observation_changed")
        rows.extend((index, row) for row in payload["quotes"].get("AIRS", []))
        prior_token, terminal = digest, token is None
    if total != manifest["retained_response_bytes"] or total > CONTRACT["limits"]["total_response_bytes"]:
        raise ValueError("total_byte_budget_changed")
    if terminal != (manifest["state"] == "TERMINAL_PAGE_OBSERVED_SOURCE_UNQUALIFIED"):
        raise ValueError("terminal_state_changed")
    prefix = [row for _, row in rows[:len(original)]]
    match = len(prefix) == len(original) and all(row_key(a) == row_key(b) for a, b in zip(original, prefix))
    return {"request_id": 335, "stage": "separate_capped_pair_technical_requery_replay",
            "parent_manifest_sha256": PARENT_SHA, "fresh_manifest_sha256": sha(manifest_bytes),
            "http_attempts": manifest["http_attempts"], "retained_response_bytes": total,
            "collection_state": manifest["state"], "original_prefix_rows": len(original),
            "fresh_rows": len(rows), "original_prefix_matches_fresh_chain": match,
            "original_acquisition_state": "PAGE_CAP_INCOMPLETE",
            "terminal_page_observed": terminal, "audit": audit_rows(rows, "quotes"),
            "additional_cost_USD": 0, "source_ready": False, "new_economic_labels": 0,
            "full_census_collected": False, "profitability_validation_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("contract", "collect", "replay"))
    parser.add_argument("--plan", type=Path, default=Path("research/request335-free-source-plan.json.gz"))
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "contract":
        args.output.write_bytes(canonical(CONTRACT))
        return
    plan = load_plan(args.plan)
    if args.parent is None or args.evidence is None:
        parser.error("explicit original and fresh private evidence paths required")
    if args.mode == "collect":
        credentials = tuple(os.environ.get(name, "") for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"))
        collect(plan, args.parent, args.evidence, credentials=credentials)
    args.output.write_bytes(canonical(replay_check(plan, args.parent, args.evidence)))


if __name__ == "__main__":
    main()
