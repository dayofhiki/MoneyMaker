import pandas as pd

from victory_trader.state_market_breadth import (
    enrich_market_breadth,
)


def test_market_breadth_is_leave_one_out_and_point_in_time():
    frame = pd.DataFrame(
        {
            "trading_day": ["2026-01-02"] * 3,
            "t": [1000, 1000, 1000],
            "ticker": ["AAA", "BBB", "CCC"],
            "above_regular_vwap": [True, False, True],
            "new_hod": [False, True, False],
            "reclaim_prior_5m_high_after_pullback": [False, False, True],
            "trailing_return_5m_pct": [1.0, -2.0, 3.0],
            "hod_distance_pct": [-1.0, -6.0, -2.0],
            "return_from_previous_close_pct": [12.0, 20.0, 15.0],
            "volume_accel_1m_vs_prior20m": [1.0, 2.0, 3.0],
        }
    )
    out = enrich_market_breadth(frame)

    aaa = out.loc[out["ticker"].eq("AAA")].iloc[0]
    assert aaa["market_runner_count"] == 3.0
    assert aaa["market_other_runner_count"] == 2.0
    assert abs(aaa["market_above_vwap_frac_other"] - 0.5) < 1e-12
    assert abs(aaa["market_new_hod_frac_other"] - 0.5) < 1e-12
    assert abs(aaa["market_trailing5_positive_frac_other"] - 0.5) < 1e-12
    assert abs(aaa["market_failure_frac_other"] - 0.5) < 1e-12
    assert abs(aaa["market_mean_trailing5_pct_other"] - 0.5) < 1e-12
    assert abs(aaa["market_mean_hod_distance_pct_other"] - (-4.0)) < 1e-12


def test_single_runner_has_no_other_runner_statistics():
    frame = pd.DataFrame(
        {
            "trading_day": ["2026-01-02"],
            "t": [1000],
            "ticker": ["AAA"],
            "above_regular_vwap": [True],
            "new_hod": [False],
            "reclaim_prior_5m_high_after_pullback": [False],
            "trailing_return_5m_pct": [1.0],
            "hod_distance_pct": [-1.0],
            "return_from_previous_close_pct": [12.0],
            "volume_accel_1m_vs_prior20m": [1.0],
        }
    )
    out = enrich_market_breadth(frame)
    assert out.iloc[0]["market_other_runner_count"] == 0.0
    assert pd.isna(out.iloc[0]["market_above_vwap_frac_other"])
