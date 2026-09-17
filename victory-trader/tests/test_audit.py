import pandas as pd

from victory_trader.audit import audit_event_dataset
from victory_trader.model_schema import MODEL_FEATURE_COLUMNS


def valid_frame():
    row = {
        "trading_day": "2026-09-15",
        "ticker": "AAA",
        "timestamp_ms": 1,
        "threshold_pct": 20.0,
        "signal_price": 2.0,
        "entry_price": 2.01,
        "entry_timestamp_ms": 60_001,
        "entry_model": "next_minute_open",
        "entry_delay_minutes": 0,
        "previous_close": 1.5,
        "dataset_schema_version": "0.2",
        "source_prices_adjusted": False,
        "discovery_high_return_threshold_pct": 10.0,
        "debug_candidate_limit": None,
        "security_type": "CS",
        "primary_exchange": "XNAS",
        "is_premarket": False,
        "is_regular_session": True,
        "is_after_hours": False,
    }
    for column in MODEL_FEATURE_COLUMNS:
        if column in row:
            continue
        row[column] = 1.0
    for horizon in (1, 2, 5, 10, 15, 30, 60):
        row[f"return_{horizon}m_pct"] = 1.0
        for scenario in ("light", "base", "stress"):
            row[f"return_{horizon}m_{scenario}_net_return_pct"] = 0.5
    for delay in (1, 2):
        row[f"delay{delay}_entry_price"] = 2.02
        row[f"delay{delay}_return_5m_base_net_return_pct"] = 0.3
    return pd.DataFrame([row])


def test_valid_full_universe_dataset_passes():
    result = audit_event_dataset(valid_frame())
    assert result.ok, result.render()
    assert "STATUS: PASS" in result.render()


def test_debug_sample_fails_full_universe_audit():
    frame = valid_frame()
    frame["debug_candidate_limit"] = 5
    result = audit_event_dataset(frame)
    assert not result.ok
    assert any("debug candidate limit" in issue for issue in result.issues)


def test_gross_net_missingness_mismatch_fails():
    frame = valid_frame()
    frame.loc[0, "return_5m_base_net_return_pct"] = None
    result = audit_event_dataset(frame)
    assert not result.ok
    assert any("gross/net missingness mismatch at 5m base" in issue for issue in result.issues)


def test_duplicate_event_key_fails():
    frame = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
    result = audit_event_dataset(frame)
    assert not result.ok
    assert any("duplicate event keys" in issue for issue in result.issues)


def test_future_selected_discovery_threshold_fails():
    frame = valid_frame()
    frame["discovery_high_return_threshold_pct"] = 30.0
    result = audit_event_dataset(frame)
    assert not result.ok
    assert any("future selection bias" in issue for issue in result.issues)


def test_legacy_completed_day_or_fundamental_columns_fail():
    frame = valid_frame()
    frame["day_high"] = 99.0
    frame["market_cap"] = 10_000_000
    result = audit_event_dataset(frame)
    assert not result.ok
    assert any("legacy columns" in issue for issue in result.issues)


def test_adjusted_source_prices_fail():
    frame = valid_frame()
    frame["source_prices_adjusted"] = True
    result = audit_event_dataset(frame)
    assert not result.ok
    assert any("split-adjusted" in issue for issue in result.issues)
