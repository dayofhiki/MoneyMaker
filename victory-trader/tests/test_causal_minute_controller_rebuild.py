import pandas as pd

from victory_trader import causal_minute_controller_rebuild as r263


def test_causal_execution_scan_uses_original_bar_start():
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "bar_start_t": 60_000,
                "t": 120_000,
                "o": 11.0,
            }
        ]
    )
    out = r263.causal_execution_scan(scan)
    assert out.loc[0, "t"] == 60_000
    assert out.loc[0, "o"] == 11.0


def test_request263_contract():
    assert r263.REQUEST_ID == 263
    assert r263.MINUTE_MS == 60_000
