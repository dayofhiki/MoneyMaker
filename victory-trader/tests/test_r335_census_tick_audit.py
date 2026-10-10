"""Synthetic census audit linkage and unknownness; no actual entitlement."""
import pytest

from victory_trader import r335_census_acquisition as census
from victory_trader.r335_census_tick_audit import audit_evidence
from victory_trader.offline_source_inventory import canonical
from test_r335_census_acquisition import setup as setup_fixture, acquire, PLAN


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    return setup_fixture.__wrapped__(tmp_path, monkeypatch)


def test_partial_acquisition_audits_all_partitions_without_replacing_unknowns(evidence, monkeypatch):
    monkeypatch.setitem(census.LIMITS, "http_attempts", 1)
    root, _, calls, _ = acquire(evidence)
    result = audit_evidence(PLAN, root)
    assert len(calls) == 1 and len(result["pairs"]) == 2470
    assert result["both_streams_api_terminal_identities"] == 12
    assert result["cross_stream_equal_provider_clock_groups"] == 12
    assert sum(v["rows"] for v in result["stream_totals"].values()) == 25
    assert result["stream_totals"]["trades"]["size_units:UNVERIFIED_HISTORICAL_SIZE_UNIT"] == 13
    assert result["stream_totals"]["quotes"]["size_units:shares"] == 12
    assert result["scientifically_qualified_identities"] == 0 and not result["source_ready"]
    assert result["new_economic_labels"] == 0 and result["additional_market_HTTP_attempts"] == 0
    assert all(p["economic_label"] is None for p in result["pairs"])
    assert canonical(audit_evidence(PLAN, root)) == canonical(result)


def test_altered_private_evidence_is_not_audited_as_valid(evidence, monkeypatch):
    monkeypatch.setitem(census.LIMITS, "http_attempts", 1)
    root, manifest, _, _ = acquire(evidence)
    page = next(p["pages"][0] for p in manifest["pairs"] if p["pages"])
    (root / page["wire_file"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="wire_evidence_changed"):
        audit_evidence(PLAN, root)
