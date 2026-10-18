"""Complete original-chain reuse; no partial cursor resume or outcome selection."""
import copy
import gzip

import pytest

from victory_trader import r335_recovery_census as recovery
from victory_trader import r335_fresh_census as fresh
from victory_trader.offline_source_inventory import canonical, sha
from test_r335_fresh_census import acquire, proof, RIGHTS
from test_r335_census_acquisition import PLAN, CREDS, Clock, Response, payload


@pytest.fixture
def prefix(tmp_path, monkeypatch):
    def storage(root, day, m):
        p = proof(day)
        if day == "2026-05-08":
            p["retrieved_archive_hash_equal"] = False
        return p
    root, m, _, _ = acquire(tmp_path / "parent", monkeypatch, storage)
    replay = fresh.replay(PLAN, root)
    cache = canonical(replay)
    (root / "prefix-replay.json.gz").write_bytes(gzip.compress(cache, mtime=0))
    digest = sha((root / "manifest.json").read_bytes())
    monkeypatch.setattr(recovery, "PREFIX_SHA", digest)
    monkeypatch.setitem(recovery.CONTRACT, "prefix_manifest_sha256", digest)
    monkeypatch.setattr(recovery, "PREFIX_REPLAY_SHA", sha(cache))
    monkeypatch.setitem(recovery.CONTRACT, "prefix_replay_sha256", sha(cache))
    monkeypatch.setattr(recovery, "checkpoint", lambda *a: None)
    return root, [proof(d["day"]) for d in m["days"]]


def collect(prefix, root, status=200):
    prior, proofs = prefix
    calls, clock = [], Clock()
    requests = {(r["ticker"], r["params"]["start"], r["path"]): r for r in PLAN["full_census"]}
    def get(url, **kwargs):
        req = requests[(kwargs["params"]["symbols"], kwargs["params"]["start"],
                        url.removeprefix("https://data.alpaca.markets"))]
        calls.append(req)
        return Response(payload(req), status)
    m = recovery.collect(PLAN, prior, root, credentials=CREDS, attestation=RIGHTS,
            persist=lambda raw, day, manifest: proof(day), prefix_proofs=proofs,
            get=get, clock=clock, sleep=clock.sleep, disk=lambda: 10**12)
    return m, calls


def test_restored_complete_814_reused_remaining_1656_new_without_querying_parent_dates(prefix, tmp_path):
    root = tmp_path / "child"
    m, calls = collect(prefix, root)
    result = recovery.replay(PLAN, root)
    assert len(calls) == 1656
    assert all(r["trading_day"] >= "2026-05-11" for r in calls)
    assert result["partition_states"] == {recovery.REUSED: 814, fresh.TERMINAL: 1656}
    assert result["verified_private_date_checkpoints"] == 12
    assert result["prior_stop_reason"] == fresh.STORAGE_FAILURE
    assert len(result["pairs"]) == 2470 and result["original_identity_denominator"] == 1235
    assert all(p["economic_label"] is None and not p["source_ready"] for p in result["pairs"])
    assert sum(len(p["pages"]) for p in m["pairs"] if p["pilot_index"] is not None) == 0
    assert canonical(recovery.replay(PLAN, root)) == canonical(result)


def test_new_HTTP_failure_preserves_parent_and_all_unrequested_original_entries(prefix, tmp_path):
    root = tmp_path / "child"
    _, calls = collect(prefix, root, 403)
    result = recovery.replay(PLAN, root)
    assert 1 <= len(calls) <= 4 and result["partition_states"][recovery.REUSED] == 814
    assert result["stop_reason"] == "HTTP_DENIED_OR_ERROR" and len(result["pairs"]) == 2470
    assert result["verified_private_date_checkpoints"] == 4


def test_changed_prefix_wire_cannot_be_reused(prefix, tmp_path):
    root, _ = prefix
    wire = next((root / "pages").iterdir())
    wire.write_bytes(b"{}")
    with pytest.raises(ValueError, match="wire_evidence_changed"):
        collect(prefix, tmp_path / "child")


def test_incomplete_parent_chain_cannot_be_declared_reused(prefix, tmp_path):
    root = tmp_path / "child"
    m, _ = collect(prefix, root)
    changed = copy.deepcopy(m)
    index = next(i for i, p in enumerate(changed["pairs"]) if p["pilot_index"] is None)
    changed["pairs"][index].update(pilot_index=index, state=recovery.REUSED, pages=[])
    (root / "manifest.json").write_bytes(canonical(changed))
    with pytest.raises(ValueError, match="complete_original_prefix_chain_required"):
        recovery.replay(PLAN, root)
