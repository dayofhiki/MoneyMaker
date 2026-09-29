import numpy as np
import pandas as pd

from victory_trader import rich_post_hot_state as r257


def test_dynamic_deltas_anchor_to_first_observed_state():
    states = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 60_000,
                "current_minute_body_return_pct": 1.0,
                "current_sec_last5_return_pct": 0.2,
            },
            {
                "trading_day": "2026-05-05",
                "ticker": "TEST",
                "hot_t": 0,
                "state_t": 120_000,
                "current_minute_body_return_pct": -1.0,
                "current_sec_last5_return_pct": 0.5,
            },
        ]
    )
    out = r257.attach_dynamic_deltas(states)
    assert out.loc[0, "delta_current_minute_body_return_pct"] == 0.0
    assert out.loc[1, "delta_current_minute_body_return_pct"] == -2.0
    assert np.isclose(
        out.loc[1, "delta_current_sec_last5_return_pct"],
        0.3,
    )


def test_pullback_rows_uses_fixed_two_percent_and_finite_value():
    states = pd.DataFrame(
        {
            "drawdown_from_running_high_pct": [-1.9, -2.0, -3.0],
            "enter_value_pct": [1.0, np.nan, 0.5],
        }
    )
    out = r257.pullback_rows(states)
    assert len(out) == 1
    assert out.iloc[0].drawdown_from_running_high_pct == -3.0


def test_safe_auc_handles_single_class():
    assert r257.safe_auc(
        np.array([True, True]),
        np.array([0.2, 0.8]),
    ) is None


def test_request257_contract():
    assert r257.REQUEST_ID == 257
    assert r257.TRAIN_CANDIDATE_FRACTION == 0.20
    assert r257.TEST_CANDIDATE_FRACTION == 0.05
    assert r257.PULLBACK_TRIGGER_PCT == 2.0
    assert r257.MIN_RICH_SECOND_COVERAGE == 0.80
    assert r257.MIN_VALUE_SPEARMAN_GAIN == 0.05
    assert r257.MIN_POSITIVE_AUC_GAIN == 0.03
