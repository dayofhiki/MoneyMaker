"""Synthetic original-census transport and adversarial provenance checks."""
import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from victory_trader import r335_census_acquisition as census
from victory_trader.offline_source_inventory import canonical, sha
from test_r335_full_window_qualification import (
    PLAN, CREDS, Response, payload, acquire as pilot_acquire, full,
)

RIGHTS = canonical({"account_is_free_Basic": True, "no_incremental_charge": True,
    "historical_SIP_research_and_local_retention_permitted": True, "no_redistribution": True,
    "registered_original_2470_private_technical_census_authorized": True,
    "rights_evidence_reference": "SYNTHETIC_ONLY"})


class Clock:
    def __init__(self):
        self.value, self.lock = 0., threading.Lock()

    def __call__(self):
        with self.lock:
            return self.value

    def sleep(self, duration):
        with self.lock:
            self.value += duration


@pytest.fixture
def setup(tmp_path, monkeypatch):
    pilot, _, _, _ = pilot_acquire(tmp_path / "fixture")
    pin = sha((pilot / "manifest.json").read_bytes())
    monkeypatch.setattr(census, "PILOT_SHA", pin)
    monkeypatch.setitem(census.CONTRACT, "pilot_manifest_sha256", pin)
    monkeypatch.setattr(census, "checkpoint", lambda *args: None)
    return pilot, tmp_path / "census"


def acquire(setup, callback=None, **options):
    pilot, root = setup
    calls, clock = [], Clock()
    lookup = {(r["ticker"], r["params"]["start"], r["path"]): r for r in PLAN["full_census"]}
    def get(url, **kwargs):
        calls.append((url, kwargs))
        req = lookup[(kwargs["params"]["symbols"], kwargs["params"]["start"],
                      url.removeprefix("https://data.alpaca.markets"))]
        return callback(req, kwargs, clock) if callback else Response(payload(req))
    manifest = census.collect(PLAN, pilot, root, credentials=CREDS, attestation=RIGHTS,
        get=get, clock=clock, sleep=clock.sleep, disk=options.pop("disk", lambda: 10**12), **options)
    return root, manifest, calls, clock


def test_all_original_partitions_terminal_with_exact_authenticated_pilot_reuse(setup):
    root, manifest, calls, _ = acquire(setup)
    report = census.replay(PLAN, root)
    assert len(calls) == 2446 and len(report["pairs"]) == 2470
    assert report["partition_states"] == {census.TERMINAL: 2446, census.REUSED: 24}
    assert report["original_identity_denominator"] == 1235
    assert not report["source_ready"] and report["new_economic_labels"] == 0
    assert all(p["economic_label"] is None for p in report["pairs"])
    assert canonical(census.replay(PLAN, root)) == canonical(report)
    assert len(manifest["days"]) == 12
    assert all(c[1]["params"]["feed"] == "sip" and c[1]["params"]["asof"] == "-"
               and not c[1]["allow_redirects"] for c in calls)
    assert root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in root.rglob("*") if p.is_file())


@pytest.mark.parametrize("status", [401, 403, 429, 500, 302])
def test_error_stops_new_starts_and_retains_all_original_entries(setup, status):
    root, _, calls, _ = acquire(setup, lambda req, kw, clock: Response(payload(req), status))
    report = census.replay(PLAN, root)
    assert 1 <= len(calls) <= 4 and len(report["pairs"]) == 2470
    assert report["stop_reason"] == "HTTP_DENIED_OR_ERROR"
    assert report["partition_states"][census.REUSED] == 24


@pytest.mark.parametrize("limit,value,reason", [
    ("http_attempts", 1, "HTTP_BUDGET_EXHAUSTED"),
    ("wall_seconds", 1, "WALL_BUDGET_EXHAUSTED"),
    ("total_new_response_bytes", 0, "BYTE_RESERVATION_BUDGET_EXHAUSTED"),
])
def test_global_budgets_censor_without_changing_population(setup, monkeypatch, limit, value, reason):
    monkeypatch.setitem(census.LIMITS, limit, value)
    root, _, _, _ = acquire(setup)
    report = census.replay(PLAN, root)
    assert report["stop_reason"] == reason and len(report["pairs"]) == 2470


def test_disk_floor_blocks_all_network_starts(setup):
    root, _, calls, _ = acquire(setup, disk=lambda: 0)
    report = census.replay(PLAN, root)
    assert calls == [] and report["stop_reason"] == "DISK_BUDGET_EXHAUSTED"


def test_day_cap_continues_all_registered_days_without_false_terminals(setup, monkeypatch):
    monkeypatch.setitem(census.LIMITS, "day_wall_seconds", 2)
    root, manifest, calls, _ = acquire(setup, lambda req, kw, clock: Response(payload(req, "next")))
    report = census.replay(PLAN, root)
    assert len(manifest["days"]) == 12 and 12 <= len(calls) <= 24
    assert report["partition_states"] == {census.DAY_CAP: 2446, census.REUSED: 24}
    assert report["stop_reason"] is None


def test_page_cap_follows_short_pages_and_continues_other_descriptors(setup, monkeypatch):
    monkeypatch.setitem(census.LIMITS, "pages_per_partition", 2)
    first = next(r for r in PLAN["full_census"] if r not in full.selected_requests(PLAN))
    def callback(req, kw, clock):
        token = ("two" if "page_token" in kw["params"] else "one") if req == first else None
        return Response(payload(req, token))
    root, _, _, _ = acquire(setup, callback)
    report = census.replay(PLAN, root)
    assert report["stop_reason"] is None and len(report["pairs"]) == 2470
    assert report["partition_states"][census.PAGE_CAP] == 1
    assert report["partition_states"][census.TERMINAL] == 2445


def test_repeated_token_is_failure_and_does_not_resume(setup, monkeypatch):
    monkeypatch.setitem(census.LIMITS, "day_wall_seconds", 1200)
    root, _, _, _ = acquire(setup, lambda req, kw, clock: Response(payload(req, "same")))
    report = census.replay(PLAN, root)
    assert report["stop_reason"] == "REPEATED_PAGINATION_TOKEN"
    assert report["partition_states"].get(census.TERMINAL, 0) == 0


@pytest.mark.parametrize("body,reason", [
    (b'{"x":NaN}', "INVALID_JSON"),
    (b'{"next_page_token":null}', "INVALID_MARKET_SCHEMA"),
    (canonical({"secret": CREDS[1]}), "SENSITIVE_BODY_WITHHELD"),
])
def test_invalid_or_sensitive_body_cannot_support_observations(setup, body, reason):
    root, _, _, _ = acquire(setup, lambda req, kw, clock: Response(body))
    report = census.replay(PLAN, root)
    assert report["stop_reason"] == reason
    assert all(p["rows_observed"] == 0 for p in report["pairs"] if p["pilot_index"] is None)
    assert all(CREDS[1].encode() not in p.read_bytes() for p in root.rglob("*") if p.is_file())


def test_oversized_body_is_retained_as_partial_without_market_decoding(setup, monkeypatch):
    monkeypatch.setitem(census.LIMITS, "bytes_per_page", 32)
    root, _, _, _ = acquire(setup, lambda req, kw, clock: Response(b"x" * 33))
    report = census.replay(PLAN, root)
    assert report["stop_reason"] == "PARTIAL_BODY_BYTE_CAP"


def test_transport_exception_never_persists_private_text(setup):
    def failure(*args):
        raise requests.ConnectionError(CREDS[1])
    root, _, _, _ = acquire(setup, failure)
    assert census.replay(PLAN, root)["stop_reason"] == "TRANSPORT_ERROR"
    assert all(CREDS[1].encode() not in p.read_bytes() for p in root.rglob("*") if p.is_file())


@pytest.mark.parametrize("change", ["wire", "attempt", "wave", "day", "reuse", "fake_error", "fake_cap", "economic", "denominator", "contract"])
def test_adversarial_evidence_and_false_completion_rejected(setup, change):
    root, manifest, _, _ = acquire(setup, disk=lambda: 0)
    fresh = next(p for p in manifest["pairs"] if p["pilot_index"] is None)
    imported = next(p for p in manifest["pairs"] if p["pilot_index"] is not None)
    if change == "wire":
        p = next((root / "pilot/pages").glob("*"))
        p.write_bytes(b"{}")
    elif change == "attempt":
        manifest["http_attempts"] = 1
    elif change == "wave":
        manifest["waves"][0]["indices"].reverse()
    elif change == "day":
        manifest["days"][0]["start_elapsed_seconds"] = -1
    elif change == "reuse":
        imported["pilot_index"] = 99
    elif change == "fake_error":
        fresh["state"] = "TRANSPORT_ERROR"
    elif change == "fake_cap":
        fresh["state"] = census.DAY_CAP
    elif change == "economic":
        manifest["source_ready"] = True
    elif change == "denominator":
        manifest["pairs"].pop()
    else:
        (root / "contract.json").write_bytes(b"{}")
    (root / "manifest.json").write_bytes(canonical(manifest))
    with pytest.raises((ValueError, IndexError)):
        census.replay(PLAN, root)


def test_sample_only_rights_do_not_authorize_census(setup):
    pilot, root = setup
    from test_r335_full_window_qualification import ATTESTATION
    with pytest.raises(ValueError, match="explicit_census_scope"):
        census.collect(PLAN, pilot, root, credentials=CREDS, attestation=ATTESTATION,
                       get=lambda *args, **kw: pytest.fail("no HTTP allowed"))
    assert not root.exists()


def test_changed_population_or_pilot_rejected_before_network(setup):
    pilot, root = setup
    changed = copy.deepcopy(PLAN)
    changed["full_census"].pop()
    with pytest.raises(ValueError, match="pinned_full_census"):
        census.collect(changed, pilot, root, credentials=CREDS, attestation=RIGHTS)
    (pilot / "manifest.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="registered_pilot"):
        census.collect(PLAN, pilot, root, credentials=CREDS, attestation=RIGHTS)


def test_four_inflight_requests_share_rate_and_reservation_budget(tmp_path):
    root = tmp_path / "transport"
    root.mkdir(mode=0o700)
    clock, barrier = Clock(), threading.Barrier(4)
    req = PLAN["full_census"][0]
    def get(*args, **kwargs):
        barrier.wait(timeout=10)
        clock.sleep(.1)
        return Response(payload(req))
    transport = census.Transport(root, CREDS, get, clock, clock.sleep, lambda: 10**12)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(transport.fetch, req, None, f"{i:04d}-000", 0., 1) for i in range(4)]
        pages = [f.result()[0] for f in futures]
    assert sorted(p["start_elapsed_seconds"] for p in pages) == [0., 1., 2., 3.]
    assert transport.attempts == 4 and transport.reserved == 0
    assert transport.bytes == 4 * len(payload(req)) and transport.reason is None


def test_wall_crossing_during_body_is_partial_and_not_terminal(setup, monkeypatch):
    monkeypatch.setitem(census.LIMITS, "wall_seconds", 2)
    def get(req, kw, clock):
        clock.sleep(2)
        return Response(payload(req))
    root, _, _, _ = acquire(setup, get)
    report = census.replay(PLAN, root)
    assert report["stop_reason"] == "PARTIAL_BODY_WALL_CAP"
    assert report["partition_states"].get(census.TERMINAL, 0) == 0
