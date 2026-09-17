import pandas as pd

from victory_trader.audit import audit_event_dataset


def valid_frame():
    row = {
        "trading_day": "2026-09-15",
        "ticker": "AAA",
        "timestamp_ms": 1,
        "threshold_pct": 20.0,
        "entry_price": 2.0,
        "previous_close": 1.5,
        "debug_candidate_limit": None,
        "trailing_return_5m_pct": 1.0,
        "trailing_return_15m_pct": 2.0,
        "trailing_return_30m_pct": 3.0,
    }
    for horizon in (1, 2, 5, 10, 15, 30, 60):
        row[f"return_{horizon}m_pct"] = 1.0
        for scenario in ("light", "base", "stress"):
            row[f"return_{horizon}m_{scenario}_net_return_pct"] = 0.5
    return pd.DataFrame([row])


def test_valid_full_universe_dataset_passes():
    result = audit_event_dataset(valid_frame())
    assert result.ok
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
