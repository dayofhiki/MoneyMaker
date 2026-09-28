import numpy as np
import pandas as pd

from victory_trader import multistep_wait_option as r256


def test_wait_option_is_fixed_mean_not_future_max():
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "enter_value_pct": 1.0,
            },
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "enter_value_pct": 10.0,
            },
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 180_000,
                "enter_value_pct": -10.0,
            },
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 240_000,
                "enter_value_pct": 0.0,
            },
        ]
    )
    out = r256.attach_multistep_target(states)
    assert out.loc[0, "wait_option_value_pct"] == 0.0
    assert out.loc[0, "enter_minus_wait_option_pct"] == 1.0


def test_window_end_contributes_cash_zero():
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "enter_value_pct": 3.0,
            },
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "enter_value_pct": 3.0,
            },
        ]
    )
    out = r256.attach_multistep_target(states)
    assert out.loc[0, "wait_option_value_pct"] == 1.0
    assert out.loc[1, "wait_option_value_pct"] == 0.0


def test_existing_unresolved_future_state_keeps_target_unresolved():
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "enter_value_pct": 1.0,
            },
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "enter_value_pct": np.nan,
            },
        ]
    )
    out = r256.attach_multistep_target(states)
    assert pd.isna(out.loc[0, "wait_option_value_pct"])


def test_request256_contract():
    assert r256.REQUEST_ID == 256
    assert r256.WAIT_OPTION_HORIZON == 3
    assert r256.MIN_PULLBACK_AUC == 0.55
    assert r256.MIN_GAIN_VS_FIXED_PULLBACK == 0.10
