import gzip

import pytest

from victory_trader import r335_final_census as final
from victory_trader import r335_recovery_census as recovery
from victory_trader.offline_source_inventory import canonical, sha
from test_r335_recovery_census import prefix as parent_fixture
from test_r335_fresh_census import proof, RIGHTS
from test_r335_census_acquisition import PLAN, CREDS, Clock, Response, payload


@pytest.fixture
def prefix(tmp_path, monkeypatch):
    return parent_fixture.__wrapped__(tmp_path, monkeypatch)


def fixture_transport():
    calls, clock = [], Clock()
    requests = {(r["ticker"], r["params"]["start"], r["path"]): r for r in PLAN["full_census"]}
    def get(url, **kwargs):
        req = requests[(kwargs["params"]["symbols"], kwargs["params"]["start"],
                        url.removeprefix("https://data.alpaca.markets"))]
        calls.append(req)
        return Response(payload(req))
    return calls, clock, get


@pytest.fixture
def ten_dates(prefix, tmp_path, monkeypatch):
    root, proofs = prefix
    child = tmp_path / "ten"
    _, clock, get = fixture_transport()
    def persist(raw, day, m):
        p = proof(day)
        if day == "2026-05-18":
            p["retrieved_archive_hash_equal"] = False
        return p
    m = recovery.collect(PLAN, root, child, credentials=CREDS, attestation=RIGHTS,
             prefix_proofs=proofs, persist=persist, get=get, clock=clock,
             sleep=clock.sleep, disk=lambda: 10**12)
    r = recovery.replay(PLAN, child)
    cache = canonical(r)
    (child / "prefix-replay.json.gz").write_bytes(gzip.compress(cache, mtime=0))
    digest = sha((child / "manifest.json").read_bytes())
    monkeypatch.setattr(final, "PREFIX_SHA", digest)
    monkeypatch.setattr(final, "PREFIX_REPLAY_SHA", sha(cache))
    monkeypatch.setitem(final.CONTRACT, "prefix_manifest_sha256", digest)
    monkeypatch.setitem(final.CONTRACT, "prefix_replay_sha256", sha(cache))
    monkeypatch.setattr(final, "checkpoint", lambda *a: None)
    return child, [proof(d["day"]) for d in m["days"]]


def test_nested_2052_complete_chains_reused_only_last_418_queried(ten_dates, tmp_path):
    parent, proofs = ten_dates
    calls, clock, get = fixture_transport()
    root = tmp_path / "final"
    final.collect(PLAN, parent, root, credentials=CREDS, attestation=RIGHTS,
        prefix_proofs=proofs, persist=lambda raw, day, m: proof(day), get=get,
        clock=clock, sleep=clock.sleep, disk=lambda: 10**12)
    r = final.replay(PLAN, root)
    assert len(calls) == 418 and all(x["trading_day"] >= "2026-05-19" for x in calls)
    assert r["partition_states"] == {final.REUSED: 2052, final.TERMINAL: 418}
    assert r["verified_private_date_checkpoints"] == 12 and len(r["pairs"]) == 2470
    assert r["prior_stop_reason"] == final.STORAGE_FAILURE
    assert r["new_economic_labels"] == 0 and not r["source_ready"]


def test_missing_nested_raw_file_cannot_be_replaced_by_replay_cache(ten_dates):
    parent, _ = ten_dates
    wire = next((parent / "prefix" / "pages").iterdir())
    wire.unlink()
    with pytest.raises(FileNotFoundError):
        final.verified_prefix(PLAN, parent)
