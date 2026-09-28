import pandas as pd

from victory_trader import minute_reference_alignment_audit as r262


def test_reference_maps_distinguish_availability_from_bar_start():
    scan = pd.DataFrame(
        [{
            "trading_day": "2026-05-11",
            "ticker": "TEST",
            "bar_start_t": 60_000,
            "t": 120_000,
            "o": 10.0,
        }]
    )
    legacy, causal, audit = r262.build_reference_maps(scan)
    assert legacy[("2026-05-11", "TEST", 120_000)] == 10.0
    assert causal[("2026-05-11", "TEST", 60_000)] == 10.0
    assert audit["exact_60s_shift_rate"] == 1.0
    assert audit["lag_ms_median"] == 60_000.0


def test_causal_reference_is_next_bar_open_at_decision_time():
    scan = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "bar_start_t": 0,
                "t": 60_000,
                "o": 10.0,
            },
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "bar_start_t": 60_000,
                "t": 120_000,
                "o": 11.0,
            },
        ]
    )
    legacy, causal, _ = r262.build_reference_maps(scan)
    assert legacy[("2026-05-11", "TEST", 60_000)] == 10.0
    assert causal[("2026-05-11", "TEST", 60_000)] == 11.0


def test_request262_contract():
    assert r262.REQUEST_ID == 262
    assert r262.MINUTE_MS == 60_000
    assert r262.MIN_COVERAGE == 0.70
    assert r262.MIN_PAIRED_DELTA == -0.50
