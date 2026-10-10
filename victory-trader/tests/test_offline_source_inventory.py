import json
from pathlib import Path
from unittest.mock import patch

import pytest

from victory_trader import offline_source_inventory as m


START = 1778097480000000000
END = START + 60000000000


def window(**updates):
    return {"trading_day": "2026-05-06", "ticker": "ABVC", "hot_t": START//1000000+60000,
            "scope": "training", "start_ns": START, "end_ns": END, **updates}


def entry(root, rows=None, **updates):
    data = b"" if rows is None else b"".join(m.canonical(x) for x in rows)
    (root / "ticks.jsonl").write_bytes(data)
    (root / "provenance.json").write_bytes(b'{"stage":"synthetic_fixture_only"}\n')
    return {"trading_day": "2026-05-06", "ticker": "ABVC", "stream": "trades",
            "file": "ticks.jsonl", "sha256": m.sha(data), "format": "rest_jsonl",
            "timestamp_unit": "nanosecond", "start_ns": START, "end_ns": END,
            "source_ref": "synthetic-only", "provenance_file": "provenance.json",
            "provenance_sha256": m.sha((root / "provenance.json").read_bytes()), **updates}


def test_empty_inventory_retains_every_identity_and_stream(tmp_path):
    windows = [window(), window(hot_t=START//1000000+120000)]
    report, ledger = m.inventory(windows, [], tmp_path)
    assert len(ledger) == 4
    assert report["state_counts"] == {"NOT_SUPPLIED": 4}
    assert report["independent_complete_source_pairs_verified"] == 0
    assert report["market_http_attempts"] == 0


def test_int_nanoseconds_preserved_duplicates_and_order_retained(tmp_path):
    times = [START+2, START+1, START+1]
    e = entry(tmp_path, [{"sip_timestamp": t, "price": 1.2, "correction": 2} for t in times])
    report, ledger = m.inventory([window()], [e], tmp_path)
    row = ledger[0]
    assert row["first_SIP_ns"] == START+1
    assert row["last_SIP_ns"] == START+2
    assert row["rows"] == 3
    assert row["duplicate_SIP_timestamps"] == row["out_of_order_pairs"] == 1
    assert row["state"] == "SOURCE_COMPLETENESS_UNVERIFIED"
    assert report["independent_complete_source_pairs_verified"] == 0


@pytest.mark.parametrize("clock", [float(START), str(START), True, None])
def test_json_clock_coercion_forbidden(tmp_path, clock):
    e = entry(tmp_path, [{"sip_timestamp": clock}])
    assert m.inventory([window()], [e], tmp_path)[1][0]["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"


@pytest.mark.parametrize("clock", [START-1, END])
def test_half_open_window_enforced(tmp_path, clock):
    e = entry(tmp_path, [{"sip_timestamp": clock}])
    assert m.inventory([window()], [e], tmp_path)[1][0]["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"


def test_empty_export_is_not_empty_market(tmp_path):
    e = entry(tmp_path, complete=True)
    row = m.inventory([window()], [e], tmp_path)[1][0]
    assert row["rows"] == 0 and row["empty_export_observed"]
    assert row["state"] == "SOURCE_COMPLETENESS_UNVERIFIED"
    assert row["first_SIP_ns"] is None


def test_changed_source_or_provenance_is_invalid(tmp_path):
    e = entry(tmp_path, [{"sip_timestamp": START}])
    for name in ("ticks.jsonl", "provenance.json"):
        data = (tmp_path / name).read_bytes()
        (tmp_path / name).write_bytes(data+b" ")
        assert m.inventory([window()], [e], tmp_path)[1][0]["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"
        (tmp_path / name).write_bytes(data)


def test_partial_export_window_cannot_claim_full_support(tmp_path):
    e = entry(tmp_path, [{"sip_timestamp": START}], end_ns=END-1)
    assert m.inventory([window()], [e], tmp_path)[1][0]["state"] == "PARTIAL_DECLARED_WINDOW"


def test_missing_export_and_error_text_not_exposed(tmp_path):
    e = entry(tmp_path, file="private-secret-file")
    report, ledger = m.inventory([window()], [e], tmp_path)
    assert ledger[0]["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"
    assert "private-secret" not in json.dumps([report, ledger])


@pytest.mark.parametrize("file", ["../secret", "/etc/passwd", ".env", "sub/.env"])
def test_unsafe_paths_are_not_opened(tmp_path, file):
    e = entry(tmp_path, file=file)
    row = m.inventory([window()], [e], tmp_path)[1][0]
    assert row["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"


def test_symlink_file_and_directory_rejected(tmp_path):
    e = entry(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path / "ticks.jsonl")
    (tmp_path / "dirlink").symlink_to(tmp_path, target_is_directory=True)
    for name in ("link", "dirlink/ticks.jsonl"):
        assert m.inventory([window()], [{**e, "file": name}], tmp_path)[1][0]["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"


@pytest.mark.parametrize("changes", [{"trading_day": "2026-07-01"}, {"ticker": "OTHER"}, {"stream": "minutes"}])
def test_extra_scope_rejected_before_file_access(tmp_path, changes):
    e = entry(tmp_path, **changes)
    with pytest.raises(ValueError, match="unexpected"):
        m.inventory([window()], [e], tmp_path)


def test_duplicate_partition_or_identity_rejected(tmp_path):
    e = entry(tmp_path)
    with pytest.raises(ValueError):
        m.inventory([window()], [e, e], tmp_path)
    with pytest.raises(ValueError):
        m.inventory([window(), window()], [], tmp_path)


def test_quote_field_missingness_stays_explicit(tmp_path):
    e = entry(tmp_path, [{"sip_timestamp": START, "bid_price": 1.0}], stream="quotes")
    row = m.inventory([window()], [e], tmp_path)[1][1]
    assert row["missing_fields"]["ask_price"] == 1
    assert row["missing_fields"]["bid_price"] == 0
    assert row["missing_fields"]["participant_timestamp"] == 1


def test_csv_nanoseconds_exact_and_scientific_notation_rejected(tmp_path):
    e = entry(tmp_path, format="provider_csv")
    data = f"ticker,sip_timestamp,price\nABVC,{START+1},1.1\n".encode()
    (tmp_path / "ticks.jsonl").write_bytes(data)
    e["sha256"] = m.sha(data)
    assert m.inventory([window()], [e], tmp_path)[1][0]["first_SIP_ns"] == START+1
    data = b"ticker,sip_timestamp\nABVC,1.778097480e18\n"
    (tmp_path / "ticks.jsonl").write_bytes(data)
    e["sha256"] = m.sha(data)
    assert m.inventory([window()], [e], tmp_path)[1][0]["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"


def test_mixed_ticker_file_rejected(tmp_path):
    e = entry(tmp_path, [{"ticker": "OTHER", "sip_timestamp": START}])
    assert m.inventory([window()], [e], tmp_path)[1][0]["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"


def test_csv_parser_failure_retains_full_unknown_ledger(tmp_path):
    import csv
    e = entry(tmp_path, format="provider_csv")
    data = f"sip_timestamp,conditions\n{START},".encode() + b"x"*33 + b"\n"
    (tmp_path/"ticks.jsonl").write_bytes(data)
    e["sha256"] = m.sha(data)
    original_limit = csv.field_size_limit(32)
    try:
        report, ledger = m.inventory([window()], [e], tmp_path)
    finally:
        csv.field_size_limit(original_limit)
    assert len(ledger) == 2
    assert ledger[0]["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"
    assert ledger[0]["error_type"] == "Error"
    assert report["independent_complete_source_pairs_verified"] == 0


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_and_duplicate_json_keys_rejected(constant):
    with pytest.raises(ValueError):
        m.strict_json('{"x":'+constant+'}')
    with pytest.raises(ValueError):
        m.strict_json('{"x":1,"x":2}')


def test_no_network_or_secret_lookup_and_immutable_exact_output(tmp_path):
    with patch("requests.get", side_effect=AssertionError("no_network")), patch.dict("os.environ", {}, clear=True):
        report, ledger = m.inventory([window()], [], tmp_path)
    m.write_once(report, ledger, tmp_path/"one")
    m.write_once(report, ledger, tmp_path/"two")
    for name in ("request335.json", "request335-input-ledger.json"):
        assert (tmp_path/"one"/name).read_bytes() == (tmp_path/"two"/name).read_bytes()
    with pytest.raises(FileExistsError):
        m.write_once(report, ledger, tmp_path/"one")


def test_byte_and_record_budgets_no_partial_success(tmp_path, monkeypatch):
    e = entry(tmp_path, [{"sip_timestamp": START}])
    for name in ("MAX_FILE_BYTES", "MAX_TOTAL_BYTES", "MAX_RECORD_BYTES"):
        with monkeypatch.context() as p:
            p.setattr(m, name, 1)
            assert m.inventory([window()], [e], tmp_path)[1][0]["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"


@pytest.mark.parametrize("changes", [{"format": "parquet"}, {"timestamp_unit": "millisecond"}, {"source_ref": ""}, {"start_ns": START-1}, {"end_ns": END+1}])
def test_unsupported_contracts_rejected(tmp_path, changes):
    e = entry(tmp_path, **changes)
    assert m.inventory([window()], [e], tmp_path)[1][0]["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"


def test_directory_not_read_as_source(tmp_path):
    e = entry(tmp_path, file="folder")
    (tmp_path/"folder").mkdir()
    assert m.inventory([window()], [e], tmp_path)[1][0]["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"


def test_invalid_optional_clock_does_not_disappear(tmp_path):
    e = entry(tmp_path, [{"sip_timestamp": START, "participant_timestamp": float(START)}])
    assert m.inventory([window()], [e], tmp_path)[1][0]["state"] == "INVALID_OR_UNAVAILABLE_EXPORT"


def test_original_source_hash_guard_stops_before_parquet_read(tmp_path):
    research = Path(__file__).parents[1]/"research"
    pins = research/"request333-inputs.json"
    plan = research/"request333-plan.json"
    destination = tmp_path/"official332"
    destination.mkdir()
    (destination/"source-sha.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="original_source_changed"):
        m.fixed_windows(tmp_path, pins, plan)


def test_caller_cannot_replace_registered_pins_and_plan(tmp_path):
    pins, plan = tmp_path/"pins.json", tmp_path/"plan.json"
    pins.write_text("{}")
    plan.write_text("{}")
    with pytest.raises(ValueError, match="original_input_contract_changed"):
        m.fixed_windows(tmp_path, pins, plan)
