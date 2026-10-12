"""Synthetic plan fixtures verify geometry, never market coverage or profit."""
from copy import deepcopy

import pytest

from victory_trader.r335_free_source_plan import build_plan
from victory_trader.tick_execution_readiness import rfc3339_ns

START = 1778097480000000000


def inputs():
    windows = [{"trading_day": "2026-05-06", "ticker": "ABVC", "hot_t": START//1000000+60000,
                "scope": "training", "start_ns": START, "end_ns": START+600000000000}]
    original = {"qualification_windows": [{"trading_day": "2026-05-06", "ticker": "ABVC",
                                           "hot_t": START//1000000+60000, "scope": "training",
                                           "start_ms": START//1000000,
                                           "end_ms": START//1000000+60000}]}
    return windows, original


def test_exact_half_open_end_and_explicit_sip_asof_parameters():
    windows, original = inputs()
    plan = build_plan(windows, original)
    for request in plan["qualification"]:
        assert rfc3339_ns(request["params"]["end"]) == request["end_ns"]-1
        assert request["params"]["feed"] == "sip"
        assert request["params"]["asof"] == "-"
    assert plan["market_http_attempts"] == 0
    assert not plan["collection_enabled"] and not plan["profitability_validation_complete"]


def test_overlapping_windows_share_transport_preserving_all_identities():
    windows, original = inputs()
    windows.append({**windows[0], "hot_t": windows[0]["hot_t"]+10000, "start_ns": START+10000000000})
    plan = build_plan(windows, original)
    assert plan["full_identity_streams"] == 4 and plan["full_unique_partition_streams"] == 2
    assert all(request["identity_indices"] == [0, 1] for request in plan["full_census"])
    assert all(request["start_ns"] == START for request in plan["full_census"])


def test_sample_selection_is_fixed_not_driven_by_future_prices():
    windows, original = inputs()
    plan = build_plan(windows, original)
    windows[0]["future_return"] = 999
    assert plan["qualification"] == build_plan(windows, original)["qualification"]


def test_changed_sample_or_session_close_fails():
    windows, original = inputs()
    altered = deepcopy(original)
    altered["qualification_windows"][0]["ticker"] = "OTHER"
    with pytest.raises(ValueError, match="selection_changed"):
        build_plan(windows, altered)
    windows.append({**windows[0], "end_ns": START+500000000000})
    with pytest.raises(ValueError, match="geometry"):
        build_plan(windows, original)


def test_full_census_does_not_become_60_second_sample():
    windows, original = inputs()
    plan = build_plan(windows, original)
    assert plan["qualification"][0]["end_ns"] == START+60000000000
    assert plan["full_census"][0]["end_ns"] == START+600000000000
    assert plan["full_collection_budget"] == "UNREGISTERED_DO_NOT_COLLECT"
