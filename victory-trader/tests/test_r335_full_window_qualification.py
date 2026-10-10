"""Synthetic transport fixtures only; no entitlement or profitability claims."""
import copy
from pathlib import Path

import pytest
import requests

from victory_trader import r335_full_window_qualification as full
from victory_trader.offline_source_inventory import canonical
from victory_trader.r335_alpaca_qualification import load_plan
from victory_trader.tick_execution_readiness import format_ns

PLAN = load_plan(Path(__file__).parents[1] / "research/request335-free-source-plan.json.gz")
CREDS = ("synthetic-private-key", "synthetic-private-secret")
ATTESTATION = canonical({"account_is_free_Basic": True, "no_incremental_charge": True,
                         "historical_SIP_research_and_local_retention_permitted": True,
                         "no_redistribution": True, "only_registered_technical_sample": True,
                         "rights_evidence_reference": "SYNTHETIC_NOT_ACTUAL_ENTITLEMENT"})


class Clock:
    def __init__(self):
        self.value, self.sleeps = 0., []

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.value += duration


class Response:
    def __init__(self, body, status=200):
        self.status_code, self.body = status, body

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        pass


def payload(req, token=None, delta=0, rows=1):
    timestamp = format_ns(req["start_ns"] + delta)
    row = ({"t": timestamp, "p": 2, "s": 1, "x": "Q", "i": 1, "c": ["@"], "z": "C"}
           if req["stream"] == "trades" else
           {"t": timestamp, "bp": 1, "ap": 2, "bs": 100, "as": 200,
            "bx": "Q", "ax": "P", "c": ["R"], "z": "C"})
    return canonical({req["stream"]: {req["ticker"]: [row] * rows}, "next_page_token": token})


def acquire(tmp_path, callback=None, **options):
    root, calls, clock = tmp_path / "private", [], Clock()
    lookup = {(r["ticker"], r["params"]["start"], r["path"]): r for r in full.selected_requests(PLAN)}
    def get(url, **kwargs):
        calls.append((url, kwargs))
        req = lookup[(kwargs["params"]["symbols"], kwargs["params"]["start"], url.removeprefix("https://data.alpaca.markets"))]
        return callback(req, kwargs, clock) if callback else Response(payload(req))
    manifest = full.collect(PLAN, root, credentials=CREDS, attestation=ATTESTATION,
                            get=get, clock=clock, sleep=clock.sleep,
                            disk=options.pop("disk", lambda: 10**10), **options)
    return root, manifest, calls, clock


def test_full_original_windows_and_all_denominators_are_preserved(tmp_path):
    selected = full.selected_requests(PLAN)
    assert len(selected) == 24 and len(PLAN["full_census"]) == 2470
    assert all(r in PLAN["full_census"] for r in selected)
    assert selected[0]["end_ns"] - selected[0]["start_ns"] == 20400 * 10**9
    root, _, calls, clock = acquire(tmp_path)
    result = full.replay(PLAN, root)
    assert len(calls) == 24 and result["terminal_technical_partitions"] == 24
    assert result["original_partitions_not_requested"] == 2446
    assert result["original_identity_denominator"] == 1235
    assert not result["source_ready"] and result["new_economic_labels"] == 0
    assert clock.sleeps == [1.] * 23
    for url, kwargs in calls:
        assert url in ("https://data.alpaca.markets/v2/stocks/trades", "https://data.alpaca.markets/v2/stocks/quotes")
        assert kwargs["params"]["feed"] == "sip" and kwargs["params"]["asof"] == "-"
        assert kwargs["params"]["limit"] == 10000 and not kwargs["allow_redirects"]
    assert root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in root.rglob("*") if p.is_file())
    assert canonical(full.replay(PLAN, root)) == canonical(result)


def test_round_robin_follows_sub_limit_tokens_and_preserves_equal_clock_rows(tmp_path):
    def callback(req, kwargs, clock):
        second = "page_token" in kwargs["params"]
        return Response(payload(req, token=None if second else "next", rows=2 if second else 1))
    root, _, calls, _ = acquire(tmp_path, callback)
    result = full.replay(PLAN, root)
    assert len(calls) == 48
    assert all("page_token" not in c[1]["params"] for c in calls[:24])
    assert all(c[1]["params"]["page_token"] == "next" for c in calls[24:])
    assert all(p["rows_observed"] == 3 and p["cross_page_equal_clock"] == 1 and
               p["within_page_duplicate_rows"] == 1 for p in result["pairs"])


def test_pair_page_cap_does_not_drop_other_pairs(tmp_path, monkeypatch):
    monkeypatch.setitem(full.LIMITS, "pages_per_pair", 2)
    first = full.selected_requests(PLAN)[0]
    def callback(req, kwargs, clock):
        cap = req == first
        token = ("two" if "page_token" in kwargs["params"] else "one") if cap else None
        return Response(payload(req, token))
    root, _, calls, _ = acquire(tmp_path, callback)
    result = full.replay(PLAN, root)
    assert len(calls) == 25 and result["terminal_technical_partitions"] == 23
    assert result["pairs"][0]["collection_state"] == "PAGE_CAP_INCOMPLETE"
    assert result["stop_reason"] is None


def test_repeated_token_is_failure_without_retry_or_dropping_denominator(tmp_path):
    root, _, calls, _ = acquire(tmp_path, lambda req, kw, clock: Response(payload(req, "same")))
    result = full.replay(PLAN, root)
    assert len(calls) == 25 and len(result["pairs"]) == 24
    assert result["stop_reason"] == "REPEATED_PAGINATION_TOKEN"
    assert result["terminal_technical_partitions"] == 0


@pytest.mark.parametrize("status", [401, 403, 429, 302, 500])
def test_http_error_stops_all_without_retry_paid_or_alternate_feed(tmp_path, status):
    root, _, calls, _ = acquire(tmp_path, lambda req, kw, clock: Response(payload(req), status))
    result = full.replay(PLAN, root)
    assert len(calls) == 1 and result["stop_reason"] == "HTTP_DENIED_OR_ERROR"
    assert result["original_partitions_not_requested"] == 2469
    assert result["terminal_technical_partitions"] == 0


@pytest.mark.parametrize("budget,value,reason", [
    ("http_attempts", 1, "HTTP_BUDGET_EXHAUSTED"),
    ("wall_seconds", 1, "WALL_BUDGET_EXHAUSTED"),
    ("total_retained_response_bytes", 1, "PARTIAL_BODY_BYTE_BUDGET"),
])
def test_global_budget_retains_all_incomplete_entries(tmp_path, monkeypatch, budget, value, reason):
    monkeypatch.setitem(full.LIMITS, budget, value)
    root, _, calls, _ = acquire(tmp_path)
    result = full.replay(PLAN, root)
    assert len(calls) == 1 and len(result["pairs"]) == 24 and result["stop_reason"] == reason


def test_disk_floor_blocks_before_first_http_request(tmp_path):
    root, _, calls, _ = acquire(tmp_path, disk=lambda: 0)
    result = full.replay(PLAN, root)
    assert not calls and result["stop_reason"] == "DISK_BUDGET_EXHAUSTED"
    assert result["original_partitions_not_requested"] == 2470


@pytest.mark.parametrize("body,reason", [
    (b'{"x":NaN}', "INVALID_JSON"),
    (b'{"x":1,"x":2}', "INVALID_JSON"),
    (b'{"next_page_token":null}', "INVALID_MARKET_SCHEMA"),
    (canonical({"secret": CREDS[1]}), "SENSITIVE_BODY_WITHHELD"),
])
def test_invalid_and_sensitive_responses_never_support_market_observations(tmp_path, body, reason):
    root, _, calls, _ = acquire(tmp_path, lambda req, kw, clock: Response(body))
    result = full.replay(PLAN, root)
    assert len(calls) == 1 and result["stop_reason"] == reason
    assert result["pairs"][0]["rows_observed"] == 0
    if reason == "SENSITIVE_BODY_WITHHELD":
        assert not list((root / "pages").glob("*")) if (root / "pages").exists() else True


def test_transport_exception_text_is_not_persisted(tmp_path):
    def callback(req, kwargs, clock):
        raise requests.ConnectionError(CREDS[1])
    root, _, calls, _ = acquire(tmp_path, callback)
    assert len(calls) == 1 and full.replay(PLAN, root)["stop_reason"] == "TRANSPORT_ERROR"
    assert all(CREDS[1].encode() not in p.read_bytes() for p in root.rglob("*") if p.is_file())


@pytest.mark.parametrize("change", ["wire", "observation", "token", "terminal", "schedule", "rate", "economic", "denominator", "request"])
def test_mutated_evidence_is_rejected(tmp_path, change):
    root, manifest, _, _ = acquire(tmp_path)
    page = manifest["pairs"][0]["pages"][0]
    if change == "wire":
        (root / page["wire_file"]).write_bytes(b"{}")
    elif change == "observation":
        page["observation"]["rows"] += 1
    elif change == "token":
        page["request_token_sha256"] = "fake"
    elif change == "terminal":
        manifest["pairs"][0]["state"] = "PAGE_CAP_INCOMPLETE"
    elif change == "schedule":
        page["attempt"] = 2
    elif change == "rate":
        manifest["pairs"][1]["pages"][0]["start_elapsed_seconds"] = 0
    elif change == "economic":
        manifest["source_ready"] = True
    elif change == "denominator":
        manifest["pairs"].pop()
    else:
        manifest["pairs"][0]["request"]["params"]["feed"] = "iex"
    (root / "manifest.json").write_bytes(canonical(manifest))
    with pytest.raises(ValueError):
        full.replay(PLAN, root)


@pytest.mark.parametrize("credentials", [(), ("", "x"), ("x\ny", "z"), (None, "x")])
def test_invalid_credentials_never_reach_network(tmp_path, credentials):
    with pytest.raises(ValueError):
        full.collect(PLAN, tmp_path / "private", credentials=credentials, attestation=ATTESTATION,
                     get=lambda *a, **kw: pytest.fail("must not request"))
    assert not (tmp_path / "private").exists()


def test_sealed_or_arbitrary_descriptors_and_reuse_are_blocked(tmp_path):
    changed = copy.deepcopy(PLAN)
    changed["qualification"][0]["trading_day"] = "2026-07-01"
    with pytest.raises(ValueError, match="pinned_source_plan"):
        full.selected_requests(changed)
    root, _, _, _ = acquire(tmp_path)
    with pytest.raises(ValueError, match="fresh_private_directory"):
        full.collect(PLAN, root, credentials=CREDS, attestation=ATTESTATION)


def test_half_open_end_and_exact_nanoseconds_are_enforced(tmp_path):
    root, _, calls, _ = acquire(tmp_path, lambda req, kw, clock: Response(payload(req, delta=req["end_ns"] - req["start_ns"])))
    assert len(calls) == 1 and full.replay(PLAN, root)["stop_reason"] == "INVALID_MARKET_SCHEMA"
