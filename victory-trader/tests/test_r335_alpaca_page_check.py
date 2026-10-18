"""Synthetic re-query fixtures. No account, entitlement or market claims."""
from pathlib import Path

import pytest

from victory_trader import r335_alpaca_page_check as check
from victory_trader.offline_source_inventory import canonical, strict_json
from victory_trader.tick_execution_readiness import format_ns

START = check.REQUEST["start_ns"]
CREDS = ("synthetic-key-123456", "synthetic-secret-654321")
ATTESTATION = canonical({"account_is_free_Basic": True, "no_incremental_charge": True,
                         "historical_SIP_research_and_local_retention_permitted": True,
                         "no_redistribution": True, "only_registered_technical_sample": True,
                         "rights_evidence_reference": "SYNTHETIC_TEST_NOT_REAL_ENTITLEMENT"})


def quote(delta=0, **changes):
    return {"t": format_ns(START + delta), "bp": 1, "ap": 2, "bs": 100, "as": 200,
            "bx": "Q", "ax": "P", "c": ["R"], "z": "C", **changes}


class Response:
    def __init__(self, rows=(), token=None, status=200, body=None):
        self.status_code = status
        self.body = body if body is not None else canonical({"quotes": {"AIRS": list(rows)}, "next_page_token": token})

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        pass


@pytest.fixture
def fixture_parent(tmp_path, monkeypatch):
    parent = tmp_path / "synthetic-parent"
    parent.mkdir()
    (parent / "attestation.json").write_bytes(ATTESTATION)
    # Re-query tests isolate the new transport/chain; parent pinning is tested separately.
    monkeypatch.setattr(check, "source_prefix", lambda plan, root: [quote(), quote(1)])
    return parent


def acquire(tmp_path, parent, responses):
    root, calls, sleeps = tmp_path / "fresh", [], []
    responses = iter(responses)
    def get(url, **options):
        calls.append((url, options))
        return next(responses)
    manifest = check.collect({}, parent, root, credentials=CREDS, get=get,
                             clock=lambda: 0, sleep=sleeps.append)
    return root, manifest, calls, sleeps


def test_fresh_large_page_chain_matches_prefix_without_rewriting_original(tmp_path, fixture_parent):
    original = (fixture_parent / "attestation.json").read_bytes()
    root, _, calls, _ = acquire(tmp_path, fixture_parent, [Response([quote(), quote(1), quote(2)])])
    result = check.replay_check({}, fixture_parent, root)
    assert len(calls) == 1 and calls[0][0] == "https://data.alpaca.markets/v2/stocks/quotes"
    assert calls[0][1]["params"]["limit"] == 10000
    assert calls[0][1]["params"]["feed"] == "sip" and not calls[0][1]["allow_redirects"]
    assert "page_token" not in calls[0][1]["params"]
    assert result["original_prefix_matches_fresh_chain"] and result["fresh_rows"] == 3
    assert result["original_acquisition_state"] == "PAGE_CAP_INCOMPLETE"
    assert not result["source_ready"] and result["new_economic_labels"] == 0
    assert (fixture_parent / "attestation.json").read_bytes() == original
    assert root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in root.rglob("*") if p.is_file())


def test_sub_limit_page_with_token_is_followed_and_ties_are_not_dropped(tmp_path, fixture_parent):
    root, _, calls, sleeps = acquire(tmp_path, fixture_parent,
                                    [Response([quote()], "next"), Response([quote(1), quote(1, **{"as": 100})])])
    result = check.replay_check({}, fixture_parent, root)
    assert len(calls) == 2 and calls[1][1]["params"]["page_token"] == "next"
    assert sleeps == [12.5]
    assert result["fresh_rows"] == 3 and result["audit"]["equal_clock_variation_groups"]["size"] == 1


def test_changed_prefix_is_recorded_without_splicing_or_economic_success(tmp_path, fixture_parent):
    root, _, _, _ = acquire(tmp_path, fixture_parent, [Response([quote(**{"as": 300}), quote(1)])])
    result = check.replay_check({}, fixture_parent, root)
    assert not result["original_prefix_matches_fresh_chain"]
    assert result["terminal_page_observed"] and not result["source_ready"]


def test_two_pages_with_continuation_remain_incomplete_without_a_third_call(tmp_path, fixture_parent):
    root, _, calls, _ = acquire(tmp_path, fixture_parent,
                               [Response([quote()], "one"), Response([quote(1)], "two")])
    result = check.replay_check({}, fixture_parent, root)
    assert len(calls) == 2 and result["collection_state"] == "PAGE_CAP_INCOMPLETE"
    assert not result["terminal_page_observed"]


@pytest.mark.parametrize("status", [401, 403, 429, 302, 500])
def test_any_http_error_stops_without_retry_fallback(tmp_path, fixture_parent, status):
    root, _, calls, _ = acquire(tmp_path, fixture_parent, [Response(status=status)])
    result = check.replay_check({}, fixture_parent, root)
    assert len(calls) == 1 and not result["terminal_page_observed"]
    assert result["collection_state"] == "HTTP_DENIED_OR_ERROR"


def test_repeated_pagination_token_is_terminal_failure(tmp_path, fixture_parent):
    root, _, calls, _ = acquire(tmp_path, fixture_parent,
                               [Response([quote()], "same"), Response([quote(1)], "same")])
    assert len(calls) == 2
    assert check.replay_check({}, fixture_parent, root)["collection_state"] == "REPEATED_PAGINATION_TOKEN"


@pytest.mark.parametrize("credentials", [(), ("", "a"), ("a\nb", "c"), (None, "c")])
def test_invalid_credentials_block_before_output_and_network(tmp_path, fixture_parent, credentials):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid credentials must never reach network")
    with pytest.raises(ValueError):
        check.collect({}, fixture_parent, tmp_path / "fresh", credentials=credentials, get=forbidden)
    assert not (tmp_path / "fresh").exists()


def test_original_manifest_pin_blocks_arbitrary_or_sealed_source(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "manifest.json").write_bytes(canonical({"trading_day": "2026-07-01"}))
    with pytest.raises(ValueError, match="pinned_parent_manifest"):
        check.source_prefix({}, parent)


@pytest.mark.parametrize("change", ["wire", "observation", "token", "terminal", "economic_claim"])
def test_evidence_mutations_fail_replay(tmp_path, fixture_parent, change):
    root, manifest, _, _ = acquire(tmp_path, fixture_parent, [Response([quote(), quote(1)])])
    if change == "wire":
        (root / manifest["pages"][0]["wire_file"]).write_bytes(b"{}")
    elif change == "observation":
        manifest["pages"][0]["observation"]["rows"] += 1
    elif change == "token":
        manifest["pages"][0]["request_token_sha256"] = "fake"
    elif change == "terminal":
        manifest["state"] = "PAGE_CAP_INCOMPLETE"
    else:
        manifest["source_ready"] = True
    (root / "manifest.json").write_bytes(canonical(manifest))
    with pytest.raises(ValueError):
        check.replay_check({}, fixture_parent, root)


def test_sensitive_response_is_withheld_and_cannot_support_market_observations(tmp_path, fixture_parent):
    root, _, calls, _ = acquire(tmp_path, fixture_parent,
                               [Response(body=canonical({"secret": CREDS[1]}))])
    result = check.replay_check({}, fixture_parent, root)
    assert len(calls) == 1 and result["collection_state"] == "SENSITIVE_BODY_WITHHELD"
    assert result["fresh_rows"] == 0 and not list((root / "pages").glob("*.wire"))
    assert CREDS[1].encode() not in (root / "manifest.json").read_bytes()


def test_committed_contract_equals_code_and_never_enables_full_collection():
    assert strict_json(Path("research/request335-page-check-contract.json").read_bytes()) == check.CONTRACT
    assert not check.CONTRACT["full_collection_enabled"] and check.CONTRACT["additional_cost_authorized_USD"] == 0
