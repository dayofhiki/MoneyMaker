"""Published-evidence integrity, identity attribution and non-promotion regressions."""
import copy
from pathlib import Path

import pytest

from victory_trader.offline_source_inventory import canonical, sha
from victory_trader.r335_alpaca_qualification import load_plan
from victory_trader.r335_census_bottlenecks import diagnose, read_pinned


@pytest.fixture
def evidence():
    root = Path(__file__).parents[1] / "research"
    return (load_plan(root / "request335-free-source-plan.json.gz"),
            read_pinned(root / "request335-input-ledger.json.gz", "ledger"),
            read_pinned(root / "request335-census-result-20261011.json.gz", "acquisition"),
            read_pinned(root / "request335-census-audit-20261011.json.gz", "audit"))


def test_all_original_identities_and_scope_denominators_preserved_without_labels(evidence):
    result = diagnose(*evidence)
    assert len(result["identities"]) == 1235
    assert result["scope_totals"]["training"]["identities"] == 407
    assert result["scope_totals"]["evaluation"]["identities"] == 828
    assert sum(x["identities"] for x in result["day_totals"].values()) == 1235
    assert sum(result["observed_flag_combinations"].values()) == 1235
    assert not result["raw_replay_performed_this_analysis"] and not result["source_ready"]
    assert result["scientifically_qualified_identities"] == 0
    assert result["execution_resolution"] == "NOT_EVALUATED"
    assert result["new_economic_labels"] == 0 and result["additional_market_HTTP_attempts"] == 0
    assert all(not i["source_ready"] and i["economic_label"] is None for i in result["identities"])
    assert canonical(diagnose(*evidence)) == canonical(result)


def test_diagnostics_are_attributed_to_original_identity_not_ticker_only(evidence):
    plan, ledger, acquisition, audit = copy.deepcopy(evidence)
    # Repeated AIRS ticker on different dates must retain different diagnostics.
    indices = [n for n, item in enumerate(audit["pairs"]) if item["ticker"] == "AIRS"
               and item["stream"] == "trades"]
    assert len(indices) > 1
    for n in indices:
        audit["pairs"][n]["trade_update_statuses"] = {"absent": audit["pairs"][n]["rows"]}
    selected = indices[0]
    audit["pairs"][selected]["trade_update_statuses"]["canceled"] = 1
    audit["pairs"][selected]["trade_update_statuses"]["absent"] -= 1
    result = diagnose(plan, ledger, acquisition, audit)
    flagged = [i["original_identity_index"] for i in result["identities"]
               if i["ticker"] == "AIRS" and "trade_update_present" in i["observed_flags"]]
    assert flagged == audit["pairs"][selected]["identity_indices"]


def test_no_listed_diagnostic_still_has_unknown_execution_and_source(evidence):
    plan, ledger, acquisition, audit = copy.deepcopy(evidence)
    for item in audit["pairs"][:2]:
        for name in ("equal_clock_groups", "rows_in_equal_clock_groups", "cross_page_equal_clock_groups",
                     "locked_quote_rows", "crossed_quote_rows", "quote_rows_with_inactive_side",
                     "provider_clock_backwards"):
            item[name] = 0
        item["equal_clock_variation_groups"] = {key: 0 for key in item["equal_clock_variation_groups"]}
        item["trade_update_statuses"] = {"absent": item["rows"]} if item["stream"] == "trades" else {}
    identity = diagnose(plan, ledger, acquisition, audit)["identities"][0]
    assert identity["observed_flags"] == []
    assert not identity["source_ready"] and identity["economic_label"] is None
    assert identity["execution_resolution"] == "NOT_EVALUATED"


@pytest.mark.parametrize("name", ["ledger", "acquisition", "audit"])
def test_changed_public_evidence_is_not_authenticated(tmp_path, name):
    path = tmp_path / "changed.json"
    path.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="published_.*_evidence_changed"):
        read_pinned(path, name)


@pytest.mark.parametrize("field,value", [("rows", -1), ("equal_clock_groups", True),
                                       ("crossed_quote_rows", "1")])
def test_invalid_diagnostic_counts_rejected(evidence, field, value):
    plan, ledger, acquisition, audit = copy.deepcopy(evidence)
    audit["pairs"][0][field] = value
    if field == "rows":
        acquisition["pairs"][0]["rows_observed"] = value
        audit["census_replay_sha256"] = sha(canonical(acquisition))
    with pytest.raises(ValueError, match="nonnegative_integer_diagnostic_count_required"):
        diagnose(plan, ledger, acquisition, audit)


@pytest.mark.parametrize("change,match", [
    ("drop_identity", "original_denominator_changed"),
    ("switch_stream", "original_identity_stream_pair_changed"),
    ("move_index", "original_partition_identity_link_changed"),
    ("alter_window", "original_partition_window_changed"),
    ("change_row_count", "published_row_count_or_state_changed"),
    ("add_label", "technical_evidence_cannot_promote_economic_status"),
    ("promote_source", "technical_evidence_cannot_promote_economic_status"),
    ("impossible_groups", "impossible_equal_clock_counts"),
])
def test_changed_links_or_false_economic_status_rejected(evidence, change, match):
    plan, ledger, acquisition, audit = copy.deepcopy(evidence)
    if change == "drop_identity":
        ledger.pop()
    elif change == "switch_stream":
        ledger[1]["stream"] = "trades"
    elif change == "move_index":
        audit["pairs"][0]["identity_indices"] = [1]
    elif change == "alter_window":
        acquisition["pairs"][0]["start_ns"] += 1
        audit["census_replay_sha256"] = sha(canonical(acquisition))
    elif change == "change_row_count":
        audit["pairs"][0]["rows"] += 1
    elif change == "add_label":
        audit["pairs"][0]["economic_label"] = 0
    elif change == "promote_source":
        audit["pairs"][0]["source_ready"] = True
    elif change == "impossible_groups":
        audit["pairs"][0]["equal_clock_groups"] = audit["pairs"][0]["rows"]
    with pytest.raises(ValueError, match=match):
        diagnose(plan, ledger, acquisition, audit)
