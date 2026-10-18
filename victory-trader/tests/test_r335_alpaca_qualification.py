"""Synthetic HTTP/clock fixtures only; no real quotes or profitability claims."""
from copy import deepcopy
from pathlib import Path

import pytest
import requests

from victory_trader.offline_source_inventory import canonical, sha, strict_json
from victory_trader.r335_alpaca_qualification import (
    CONTRACT, LIMITS, collect, inspect_page, load_plan, replay, verify_attestation,
)
from victory_trader.r335_free_source_plan import request_descriptor
from victory_trader.tick_execution_readiness import format_ns

START = 1778097480000000000
CREDS = ("synthetic-key-123456", "synthetic-secret-654321")
ATTESTATION = canonical({"account_is_free_Basic": True, "no_incremental_charge": True,
                         "historical_SIP_research_and_local_retention_permitted": True,
                         "no_redistribution": True, "only_registered_technical_sample": True,
                         "rights_evidence_reference": "SYNTHETIC_TEST_NOT_REAL_ENTITLEMENT"})


def plan(pairs=2):
    reqs = [request_descriptor("2026-05-06", "ABVC", stream, START, START+60000000000, 100)
            for stream in ("trades", "quotes")]
    return {"qualification": reqs[:pairs]}


def payload(stream="trades", times=(START,), token=None):
    return {stream: {"ABVC": [{"t": format_ns(t), "p": 1.1234567890123456789,
                               "s": 10, "x": "SYNTHETIC", "c": [], "z": "C"} for t in times]},
            "next_page_token": token}


class Response:
    def __init__(self, body, status=200, fail=False):
        self.body = body if isinstance(body, bytes) else canonical(body)
        self.status_code, self.fail, self.closed = status, fail, False

    def iter_content(self, chunk_size):
        yield self.body
        if self.fail:
            raise requests.ConnectionError("contains synthetic-secret-654321; must not serialize")

    def close(self):
        self.closed = True


class Fake:
    def __init__(self, responses):
        self.responses, self.calls, self.sleeps = iter(responses), [], []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)

    def collect(self, tmp_path, source=None):
        root = tmp_path / "evidence"
        result = collect(source or plan(), root, credentials=CREDS, attestation=ATTESTATION,
                         get=self.get, clock=lambda: 0, sleep=self.sleeps.append)
        return root, result


def write_manifest(root, manifest):
    (root / "manifest.json").write_bytes(canonical(manifest))


def test_preflight_has_original_denominator_no_credentials_and_no_network(tmp_path):
    source = load_plan(Path("research/request335-free-source-plan.json.gz"))
    def forbidden(*args, **kwargs):
        pytest.fail("preflight must never call a market API")
    result = collect(source, tmp_path / "evidence", get=forbidden)
    assert len(result["pairs"]) == 24 and result["http_attempts"] == 0
    assert {p["state"] for p in result["pairs"]} == {"NOT_REQUESTED_MISSING_CREDENTIAL"}
    assert replay(source, tmp_path / "evidence")["new_economic_labels"] == 0


def test_pinned_plan_changed_is_rejected(tmp_path):
    path = tmp_path / "plan.json"
    path.write_bytes(canonical(plan()))
    with pytest.raises(ValueError, match="source_plan_changed"):
        load_plan(path)


@pytest.mark.parametrize("field", ["account_is_free_Basic", "no_incremental_charge",
                                    "historical_SIP_research_and_local_retention_permitted",
                                    "no_redistribution", "only_registered_technical_sample"])
def test_license_or_cost_gate_cannot_be_truthy_string(field):
    value = strict_json(ATTESTATION)
    value[field] = "true"
    with pytest.raises(ValueError, match="attestation_required"):
        verify_attestation(canonical(value))


def test_missing_attestation_blocks_before_any_request_or_output(tmp_path):
    with pytest.raises(ValueError, match="attestation"):
        collect(plan(), tmp_path / "evidence", credentials=CREDS)
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize("mutation", ["feed", "host", "path", "end", "limit", "date"])
def test_no_non_sip_arbitrary_url_full_download_or_sealed_date(tmp_path, mutation):
    source = deepcopy(plan())
    req = source["qualification"][0]
    if mutation == "feed":
        req["params"]["feed"] = "iex"
    elif mutation == "host":
        req["host"] = "api.alpaca.markets"
    elif mutation == "path":
        req["path"] = "/v2/orders"
    elif mutation == "end":
        req["end_ns"] += 60000000000
    elif mutation == "limit":
        req["params"]["limit"] = 10000
    else:
        req["trading_day"] = "2026-07-01"
    with pytest.raises(ValueError, match="descriptor_required"):
        collect(source, tmp_path / "evidence")


def test_sub_limit_page_still_follows_token_and_preserves_raw_exact_decimals(tmp_path):
    first = Response(b'{"trades":{"ABVC":[{"t":"2026-05-06T20:38:00.000000001Z","p":1.1234567890123456789}]},"next_page_token":"opaque-page"}')
    # Use the fixture's actual timestamp, not a synthetic approximate datetime.
    source = plan(1)
    source["qualification"][0] = request_descriptor("2026-05-06", "ABVC", "trades", START, START+60000000000, 100)
    first.body = first.body.replace(b"2026-05-06T20:38:00.000000001Z", format_ns(START+1).encode())
    fake = Fake([first, Response(payload(times=(START+2,)))])
    root, result = fake.collect(tmp_path, source)
    assert len(fake.calls) == 2 and fake.calls[1][1]["params"]["page_token"] == "opaque-page"
    assert fake.sleeps == [12.5]
    assert first.closed and (root / "pages/00-00.wire").read_bytes() == first.body
    assert b"1.1234567890123456789" in (root / "pages/00-00.wire").read_bytes()
    assert result["pairs"][0]["state"] == "TERMINAL_PAGE_OBSERVED_SOURCE_UNQUALIFIED"
    out = replay(source, root)
    assert out["pairs"][0]["rows_observed"] == 2 and not out["source_ready"]
    url, options = fake.calls[0]
    assert url == "https://data.alpaca.markets/v2/stocks/trades"
    assert options["params"]["feed"] == "sip" and options["params"]["asof"] == "-"
    assert not options["allow_redirects"]
    assert (root / "manifest.json").stat().st_mode & 0o777 == 0o600
    assert root.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("status,attempts", [(401, 1), (429, 1), (403, 2), (302, 2)])
def test_denial_rate_limit_and_redirect_have_no_retry_fallback(tmp_path, status, attempts):
    fake = Fake([Response({"message": "SYNTHETIC"}, status), Response(payload("quotes"))])
    root, result = fake.collect(tmp_path)
    assert result["http_attempts"] == attempts and len(result["pairs"]) == 2
    assert not replay(plan(), root)["source_ready"]


def test_403_stops_only_denied_stream_and_keeps_other_day_denominator(tmp_path):
    source = plan()
    source["qualification"] *= 2
    fake = Fake([Response({}, 403), Response(payload("quotes")), Response(payload("quotes"))])
    root, result = fake.collect(tmp_path, source)
    assert result["http_attempts"] == 3
    assert result["pairs"][2]["state"] == "SKIPPED_AFTER_STREAM_DENIAL"
    assert len(replay(source, root)["pairs"]) == 4


@pytest.mark.parametrize("tokens,state", [(("p1", "p2"), "PAGE_CAP_INCOMPLETE"),
                                           (("p1", "p1"), "REPEATED_PAGINATION_TOKEN")])
def test_page_caps_and_loops_never_claim_complete(tmp_path, tokens, state):
    fake = Fake([Response(payload(token=t)) for t in tokens])
    root, result = fake.collect(tmp_path, plan(1))
    assert result["pairs"][0]["state"] == state
    assert not replay(plan(1), root)["pairs"][0]["terminal_page_observed"]


def test_byte_cap_retains_prefix_digest_and_does_not_claim_full_body(tmp_path, monkeypatch):
    monkeypatch.setitem(LIMITS, "bytes_per_page", 10)
    fake = Fake([Response(payload())])
    root, result = fake.collect(tmp_path)
    page = result["pairs"][0]["pages"][0]
    assert page["body_bytes"] == 10 and not page["wire_complete"]
    assert page["observation"] is None
    assert result["pairs"][1]["state"] == "SKIPPED_AFTER_PARTIAL_BODY_BUDGET"
    assert replay(plan(), root)["retained_response_bytes"] == 10


def test_transport_exception_retains_received_prefix_without_exception_text(tmp_path):
    fake = Fake([Response(payload(), fail=True)])
    root, result = fake.collect(tmp_path, plan(1))
    assert result["pairs"][0]["state"] == "TRANSPORT_ERROR"
    assert not result["pairs"][0]["pages"][0]["wire_complete"]
    assert CREDS[1].encode() not in (root / "manifest.json").read_bytes()
    assert replay(plan(1), root)["http_attempts"] == 1


@pytest.mark.parametrize("body", [b'{"next_page_token":null,"api_key":"other"}',
                                   b'{"message":"synthetic-secret-654321"}'])
def test_sensitive_body_is_withheld_and_stops_all(tmp_path, body):
    fake = Fake([Response(body)])
    root, result = fake.collect(tmp_path)
    assert result["pairs"][0]["state"] == "SENSITIVE_BODY_WITHHELD"
    assert not list((root / "pages").glob("*")) if (root / "pages").exists() else True
    assert replay(plan(), root)["http_attempts"] == 1
    for path in root.iterdir():
        assert CREDS[0].encode() not in path.read_bytes() and CREDS[1].encode() not in path.read_bytes()


@pytest.mark.parametrize("body", [b'{"trades":{},"next_page_token":null,"next_page_token":"x"}',
                                   b'{"trades":{},"next_page_token":null,"v":NaN}',
                                   b'not JSON'])
def test_malformed_wire_retained_but_not_market_evidence(tmp_path, body):
    fake = Fake([Response(body)])
    root, result = fake.collect(tmp_path, plan(1))
    assert result["pairs"][0]["state"] == "INVALID_JSON"
    assert replay(plan(1), root)["pairs"][0]["rows_observed"] == 0


@pytest.mark.parametrize("change", ["outside", "mixed_symbol", "missing_marker", "float_time"])
def test_invalid_market_schema_never_becomes_observation(change):
    value = payload()
    if change == "outside":
        value["trades"]["ABVC"][0]["t"] = format_ns(START+60000000000)
    elif change == "mixed_symbol":
        value["trades"]["OTHER"] = []
    elif change == "missing_marker":
        del value["next_page_token"]
    else:
        value["trades"]["ABVC"][0]["t"] = float(START)
    with pytest.raises(ValueError):
        inspect_page(value, plan()["qualification"][0])


def test_order_anomalies_and_duplicates_are_counted_not_sorted_or_deduplicated(tmp_path):
    fake = Fake([Response(payload(times=(START+5, START+5, START+4), token="p1")),
                 Response(payload(times=(START+3,)))])
    root, result = fake.collect(tmp_path, plan(1))
    obs = result["pairs"][0]["pages"][0]["observation"]
    assert obs["within_page_backwards"] == 1 and obs["within_page_equal_clock"] == 1
    assert obs["within_page_duplicate_rows"] == 1
    out = replay(plan(1), root)
    assert out["pairs"][0]["cross_page_backwards"] == 1
    assert out["pairs"][0]["rows_observed"] == 4
    assert obs["timestamp_semantics"] == "UNVERIFIED_PROVIDER_CLOCK"


@pytest.mark.parametrize("change", ["raw", "request", "token", "terminal", "denominator", "counts", "observation", "path", "contract"])
def test_offline_replay_rejects_changed_evidence(tmp_path, change):
    fake = Fake([Response(payload())])
    root, result = fake.collect(tmp_path, plan(1))
    page = result["pairs"][0]["pages"][0]
    if change == "raw":
        (root / page["wire_file"]).write_bytes(b"bad")
    elif change == "request":
        result["pairs"][0]["request"]["params"]["feed"] = "iex"
    elif change == "token":
        page["request_token_sha256"] = "0"*64
    elif change == "terminal":
        result["pairs"][0]["state"] = "PAGE_CAP_INCOMPLETE"
    elif change == "denominator":
        result["pairs"] = []
    elif change == "counts":
        result["retained_response_bytes"] += 1
    elif change == "observation":
        page["observation"]["rows"] += 1
    elif change == "path":
        page["wire_file"] = "../outside"
    else:
        (root / "contract.json").write_bytes(canonical({**CONTRACT, "economic_labels_enabled": True}))
    write_manifest(root, result)
    with pytest.raises(ValueError):
        replay(plan(1), root)


def test_second_run_cannot_overwrite_evidence(tmp_path):
    source = plan()
    collect(source, tmp_path / "evidence")
    with pytest.raises(ValueError, match="fresh_private"):
        collect(source, tmp_path / "evidence")


def test_attestation_private_reference_is_hashed_not_published(tmp_path):
    fake = Fake([Response(payload())])
    root, result = fake.collect(tmp_path, plan(1))
    assert result["attestation_sha256"] == sha(ATTESTATION)
    assert b"SYNTHETIC_TEST_NOT_REAL_ENTITLEMENT" not in (root / "manifest.json").read_bytes()


def test_total_byte_budget_never_reads_past_registered_remaining_bytes(tmp_path, monkeypatch):
    body = canonical(payload())
    monkeypatch.setitem(LIMITS, "total_response_bytes", len(body)+5)
    fake = Fake([Response(body), Response(payload("quotes"))])
    root, result = fake.collect(tmp_path)
    assert result["retained_response_bytes"] == len(body)+5
    assert result["pairs"][1]["state"] == "PARTIAL_BODY_BUDGET"
    assert replay(plan(), root)["retained_response_bytes"] == len(body)+5


def test_attempt_cap_is_observed_not_counted_as_an_actual_request(tmp_path, monkeypatch):
    monkeypatch.setitem(LIMITS, "http_attempts", 1)
    fake = Fake([Response(payload())])
    root, result = fake.collect(tmp_path)
    assert len(fake.calls) == result["http_attempts"] == 1
    assert result["pairs"][1]["state"] == "BUDGET_EXHAUSTED"
    assert replay(plan(), root)["http_attempts"] == 1


def test_invalid_prices_sizes_stay_in_raw_evidence_and_are_counted(tmp_path):
    value = payload()
    value["trades"]["ABVC"][0].update(p=-1, s=True)
    fake = Fake([Response(value)])
    root, result = fake.collect(tmp_path, plan(1))
    obs = result["pairs"][0]["pages"][0]["observation"]
    assert obs["rows"] == 1 and obs["invalid_positive_numeric_fields"] == {"p": 1, "s": 1}
    assert not replay(plan(1), root)["source_ready"]


def test_preflight_evidence_reproduces_byte_for_byte(tmp_path):
    first = collect(plan(), tmp_path / "one")
    second = collect(plan(), tmp_path / "two")
    assert canonical(first) == canonical(second)
    assert canonical(replay(plan(), tmp_path / "one")) == canonical(replay(plan(), tmp_path / "two"))


def test_claiming_economic_success_is_rejected(tmp_path):
    collect(plan(), tmp_path / "evidence")
    root = tmp_path / "evidence"
    manifest = strict_json((root / "manifest.json").read_bytes())
    manifest["source_ready"] = True
    write_manifest(root, manifest)
    with pytest.raises(ValueError, match="economic_success"):
        replay(plan(), root)


def test_license_attestation_bytes_cannot_change_after_collection(tmp_path):
    fake = Fake([Response(payload())])
    root, _ = fake.collect(tmp_path, plan(1))
    (root / "attestation.json").write_bytes(ATTESTATION+b" ")
    with pytest.raises(ValueError, match="attestation_evidence_changed"):
        replay(plan(1), root)
