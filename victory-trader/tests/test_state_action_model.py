import numpy as np
import pandas as pd

from victory_trader.state_action_model import (
    prepare_features,
    run_leave_one_month_out,
    summarize,
)


def _month(label: str, shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for episode in range(80):
        ticker = f"T{episode:03d}"
        for minute in range(8):
            good = 1.0 if ((episode + minute) % 7 == 0) else 0.0
            current = 5.0 + 0.02 * minute + 0.01 * episode
            future = 2.8 * good - 0.2 + shift
            row = {
                "trading_day": f"{label}-{1 + episode % 8:02d}",
                "ticker": ticker,
                "t": minute,
                "c": current,
                "entry_price": current * 1.001,
                "minutes_since_10pct_cross": minute,
                "minutes_from_regular_open": 30 + minute,
                "return_from_previous_close_pct": 10 + minute,
                "running_hod_return_pct": 12 + minute * 0.1,
                "hod_distance_pct": -2.0 + 2.0 * good,
                "rebound_from_running_low_pct": 1.0 + good,
                "minutes_since_hod": 3 - good,
                "new_hod": bool(good),
                "trailing_return_1m_pct": 0.1 + good,
                "trailing_return_3m_pct": 0.2 + 2 * good,
                "trailing_return_5m_pct": 0.3 + good,
                "trailing_return_10m_pct": 0.4 + good,
                "trailing_return_15m_pct": 0.5 + good,
                "volatility_5m_pct": 1.0 - 0.4 * good,
                "volatility_15m_pct": 1.2 - 0.5 * good,
                "bar_range_pct": 1.0 - 0.5 * good,
                "bar_body_pct": 0.2 + good,
                "bar_close_location": 0.5 + 0.3 * good,
                "range_contraction_3m_vs_15m": 0.8 - 0.3 * good,
                "regular_vwap_distance_pct": -0.2 + good,
                "above_regular_vwap": bool(good),
                "recent_deepest_pullback_15m_pct": -2.0,
                "reclaim_prior_5m_high_after_pullback": bool(good),
                "volume_accel_1m_vs_prior20m": 1.0 + good,
                "volume_accel_5m_vs_prior20m": 1.0 + 0.5 * good,
                "volume_3m_vs_postcross_peak": 0.5 + 0.5 * good,
                "transactions_1m": 100 + 100 * good,
                "transactions_5m": 500 + 400 * good,
                "dollar_volume_5m": 100_000 + 500_000 * good,
                "active_minute_fraction_15m": 0.8 + 0.1 * good,
                "runners_10pct_so_far": 5,
                "runners_10pct_last_30m": 2,
                "iwm_return_since_open_pct": 0.1,
                "iwm_return_15m_pct": 0.05,
                "iwm_volatility_15m_pct": 0.2,
                "cross_rvol_cumulative_20d": 2.0,
                "cross_rvol_5m_20d": 2.0,
                "cross_premarket_return_pct": 3.0,
                "cross_premarket_volume": 100_000,
                "cross_premarket_high_return_pct": 5.0,
                "cross_premarket_high_distance_pct": -1.0,
                "cross_regular_open_gap_pct": 2.0,
            }
            for horizon in (5, 10, 15):
                gross = future + horizon * 0.001
                row[f"buy_return_{horizon}m_pct"] = gross
                row[f"buy_return_{horizon}m_base_net_return_pct"] = gross - 0.7
                row[f"buy_return_{horizon}m_stress_net_return_pct"] = gross - 1.8
            rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_features_is_numeric_and_same_width():
    frame = _month("2026-01")
    x = prepare_features(frame)
    assert len(x) == len(frame)
    assert x.shape[1] > 30
    assert all(np.issubdtype(dtype, np.number) for dtype in x.dtypes)


def test_lomo_state_model_finds_profitable_top_states_on_synthetic_signal():
    frames = {
        "jan": _month("2026-01", 0.00),
        "feb": _month("2026-02", 0.05),
        "mar": _month("2026-03", -0.05),
    }
    details = run_leave_one_month_out(frames)
    assert set(details["month"]) == {"jan", "feb", "mar"}
    assert set(details["horizon_min"]) == {5, 10, 15}
    assert set(details["top_fraction"]) == {0.01, 0.025, 0.05, 0.10}

    summary = summarize(details)
    assert not summary.empty
    assert summary.iloc[0]["base_positive_months"] >= 2
