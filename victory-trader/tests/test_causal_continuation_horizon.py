import pandas as pd

from victory_trader import causal_continuation_horizon as r270


def _states():
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-05-08",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "exit_now_pct": 0.0,
            },
            {
                "trading_day": "2026-05-08",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "exit_now_pct": 1.0,
            },
            {
                "trading_day": "2026-05-08",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 180_000,
                "exit_now_pct": -1.0,
            },
        ]
    )


def test_continuation_targets_use_fixed_future_state_and_terminal_fallback():
    out = r270.attach_continuation_targets(_states())
    assert out.loc[0, "continuation_1_advantage_pct"] == 1.0
    assert out.loc[0, "continuation_2_advantage_pct"] == -1.0
    assert out.loc[0, "continuation_5_advantage_pct"] == -1.0
    assert out.loc[2, "continuation_1_advantage_pct"] == 0.0


def test_choose_horizon_uses_only_calibration_quality():
    reports = [
        {
            "horizon_states": 1,
            "auc": 0.60,
            "spearman": 0.04,
        },
        {
            "horizon_states": 2,
            "auc": 0.64,
            "spearman": 0.06,
        },
        {
            "horizon_states": 3,
            "auc": 0.64,
            "spearman": 0.08,
        },
        {
            "horizon_states": 5,
            "auc": 0.54,
            "spearman": 0.20,
        },
    ]
    assert r270.choose_horizon(reports) == 3


def test_request270_contract():
    assert r270.REQUEST_ID == 270
    assert r270.HORIZONS == (1, 2, 3, 5)
    assert r270.MODEL_FIT_DAYS == (
        "2026-05-05",
        "2026-05-06",
        "2026-05-07",
    )
    assert r270.CALIBRATION_DAY == "2026-05-08"
