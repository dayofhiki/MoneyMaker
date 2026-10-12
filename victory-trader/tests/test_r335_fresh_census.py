"""Independent-census checkpoint ordering and replay; synthetic transports only."""
import copy

import pytest

from victory_trader import r335_fresh_census as fresh
from victory_trader.offline_source_inventory import canonical, strict_json
from test_r335_census_acquisition import Clock, PLAN, CREDS, Response, payload

RIGHTS = canonical({"account_is_free_Basic": True, "no_incremental_charge": True,
    "no_redistribution": True, "historical_SIP_research_and_local_retention_permitted": True,
    "registered_original_2470_private_technical_census_authorized": True,
    "user_private_persistent_backup_authorized": True, "rights_evidence_reference": "SYNTHETIC"})


def proof(day):
    return {"day": day, "archive_sha256": "0" * 64, "archive_bytes": 1, "private_files": 1,
            "saved": True, "retrieved_archive_hash_equal": True}


def acquire(tmp_path, monkeypatch, storage=None, response=None):
    monkeypatch.setattr(fresh, "checkpoint", lambda *args: None)
    calls, stored, clock = [], [], Clock()
    requests = {(r["ticker"], r["params"]["start"], r["path"]): r for r in PLAN["full_census"]}
    def get(url, **kwargs):
        req = requests[(kwargs["params"]["symbols"], kwargs["params"]["start"],
                        url.removeprefix("https://data.alpaca.markets"))]
        earlier = sorted({r["trading_day"] for r in PLAN["full_census"] if r["trading_day"] < req["trading_day"]})
        assert stored == earlier, "next date started before every prior private checkpoint"
        calls.append(req)
        return response(req, kwargs) if response else Response(payload(req))
    def persist(root, day, manifest):
        result = storage(root, day, manifest) if storage else proof(day)
        if result.get("retrieved_archive_hash_equal"):
            stored.append(day)
        return result
    root = tmp_path / "fresh"
    m = fresh.collect(PLAN, root, credentials=CREDS, attestation=RIGHTS, persist=persist,
                      get=get, clock=clock, sleep=clock.sleep, disk=lambda: 10**12)
    return root, m, calls, stored


def test_complete_original_population_without_any_prior_pilot_and_checkpoints_before_next_day(tmp_path, monkeypatch):
    root, m, calls, stored = acquire(tmp_path, monkeypatch)
    r = fresh.replay(PLAN, root)
    assert len(calls) == 2470 and len(r["pairs"]) == 2470
    assert r["partition_states"] == {fresh.TERMINAL: 2470}
    assert len(stored) == r["verified_private_date_checkpoints"] == 12
    assert all(p["pilot_index"] is None for p in m["pairs"])
    assert r["reused_response_bytes"] == 0 and r["new_economic_labels"] == 0
    assert all(not p["source_ready"] and p["economic_label"] is None for p in r["pairs"])
    assert canonical(fresh.replay(PLAN, root)) == canonical(r)


@pytest.mark.parametrize("mode", ["exception", "unsaved", "unretrieved", "bad_hash"])
def test_private_storage_failure_stops_before_next_date_without_losing_denominator(tmp_path, monkeypatch, mode):
    def storage(root, day, m):
        if mode == "exception":
            raise RuntimeError("private signed URL must never be copied into manifest")
        p = proof(day)
        p[{"unsaved": "saved", "unretrieved": "retrieved_archive_hash_equal", "bad_hash": "archive_sha256"}[mode]] = False
        return p
    root, m, calls, _ = acquire(tmp_path, monkeypatch, storage)
    r = fresh.replay(PLAN, root)
    assert len(calls) == 204 and {c["trading_day"] for c in calls} == {"2026-05-05"}
    assert r["stop_reason"] == fresh.STORAGE_FAILURE and len(r["pairs"]) == 2470
    assert r["verified_private_date_checkpoints"] == 0
    assert b"signed URL" not in canonical(m)


@pytest.mark.parametrize("status", [401, 403, 429, 500, 302])
def test_transport_error_uses_frozen_stop_no_retry_policy(tmp_path, monkeypatch, status):
    root, _, calls, stored = acquire(tmp_path, monkeypatch,
                                   response=lambda r, k: Response(payload(r), status))
    r = fresh.replay(PLAN, root)
    assert 1 <= len(calls) <= 4 and stored == []
    assert r["stop_reason"] == "HTTP_DENIED_OR_ERROR" and len(r["pairs"]) == 2470


def test_original_scope_without_private_backup_authorization_rejected(tmp_path, monkeypatch):
    rights = strict_json(RIGHTS)
    rights.pop("user_private_persistent_backup_authorized")
    with pytest.raises(ValueError, match="private_durable_backup_scope_required"):
        fresh.collect(PLAN, tmp_path / "fresh", credentials=CREDS, attestation=canonical(rights), persist=lambda *a: {})


def test_forged_checkpoint_or_missing_page_cannot_support_replay(tmp_path, monkeypatch):
    monkeypatch.setitem(fresh.LIMITS, "http_attempts", 1)
    root, m, _, _ = acquire(tmp_path, monkeypatch)
    changed = copy.deepcopy(m)
    changed["private_checkpoints"] = [proof("2026-05-05")]
    (root / "manifest.json").write_bytes(canonical(changed))
    with pytest.raises(ValueError, match="checkpoint_before_next_day_required"):
        fresh.replay(PLAN, root)
    (root / "manifest.json").write_bytes(canonical(m))
    page = next(p["pages"][0] for p in m["pairs"] if p["pages"])
    (root / page["wire_file"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="wire_evidence_changed"):
        fresh.replay(PLAN, root)
