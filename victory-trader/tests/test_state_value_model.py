import numpy as np
import pandas as pd

from victory_trader.state_value_model import (
    _eligible,
    _simulate_non_overlapping_trades,
    feature_frame,
    train_fold_model,
)


def _training_frame() -> pd.DataFrame:
    rows = []
    for day in range(1, 21):
        for episode in range(20):
            for minute in range(0, 30, 3):
                quality = (episode % 10) / 10.0
                gross = 2.5 * quality - 0.8
                row = {
                    "trading_day": f"2026-01-{day:02d}",
                    "ticker": f"T{episode:03d}",
                    "t": day * 10_000_000 + episode * 100_000 + minute * 60_000,
                    "c": 5.0 + quality,
                    "entry_price": 5.0 + quality,
                    "active_minute_fraction_15m": 1.0,
                    "minutes_since_10pct_cross": float(minute),
                    "return_from_previous_close_pct": 10.0 + quality * 10.0,
                    "minutes_from_regular_open": 30.0 + minute,
                    "regular_vwap_distance_pct": quality,
                    "running_hod_return_pct": 12.0 + quality * 5.0,
                    "hod_distance_pct": -2.0 + quality,
                    "rebound_from_running_low_pct": quality * 3.0,
                    "minutes_since_hod": 2.0,
                    "trailing_return_1m_pct": quality,
                    "trailing_return_3m_pct": quality * 1.5,
                    "trailing_return_5m_pct": quality * 2.0,
                    "trailing_return_10m_pct": quality * 2.2,
                    "trailing_return_15m_pct": quality * 2.4,
                    "volatility_5m_pct": 1.0 - quality * 0.3,
                    "volatility_15m_pct": 1.2 - quality * 0.3,
                    "bar_range_pct": 1.0 - quality * 0.2,
                    "bar_body_pct": quality * 0.2,
                    "bar_close_location": 0.5 + quality * 0.4,
                    "range_contraction_3m_vs_15m": 0.8 - quality * 0.2,
                    "volume_accel_1m_vs_prior20m": 1.0 + quality,
                    "volume_accel_5m_vs_prior20m": 1.0 + quality,
                    "volume_3m_vs_postcross_peak": 0.6,
                    "runners_10pct_so_far": 4.0,
                    "runners_10pct_last_30m": 2.0,
                    "iwm_return_since_open_pct": 0.2,
                    "iwm_return_15m_pct": 0.1,
                    "iwm_volatility_15m_pct": 0.2,
                    "above_regular_vwap": True,
                    "new_hod": quality > 0.8,
                    "reclaim_prior_5m_high_after_pullback": quality > 0.7,
                    "volume_1m": 1000.0,
                    "volume_3m": 3000.0,
                    "volume_5m": 5000.0,
                    "transactions_1m": 100.0,
                    "transactions_5m": 500.0,
                    "dollar_volume_5m": 25000.0,
                    "buy_return_5m_pct": gross,
                    "buy_return_5m_base_net_return_pct": gross - 0.8,
                    "buy_return_5m_stress_net_return_pct": gross - 2.0,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def test_feature_frame_is_numeric_and_keeps_rows():
    frame = _training_frame().head(20)
    features = feature_frame(frame)
    assert len(features) == len(frame)
    assert features.shape[1] > 30
    assert all(np.issubdtype(dtype, np.number) for dtype in features.dtypes)


def test_train_fold_model_produces_calibration_cutoffs():
    frame = _training_frame()
    fold = train_fold_model(frame, 5)
    assert set(fold.calibration_cutoffs) == {0.95, 0.98, 0.99}
    assert len(fold.feature_columns) > 30


def test_non_overlapping_policy_allows_reentry_after_exit():
    frame = pd.DataFrame(
        {
            "trading_day": ["2026-01-02"] * 5,
            "ticker": ["AAA"] * 5,
            "t": [0, 60_000, 120_000, 300_000, 360_000],
            "entry_price": [5.0] * 5,
            "predicted_base_net_pct": [2.0] * 5,
            "buy_return_5m_base_net_return_pct": [1.0] * 5,
        }
    )
    trades = _simulate_non_overlapping_trades(frame, horizon=5, cutoff=1.0)
    assert list(trades["t"]) == [0, 300_000]


def test_scoring_eligibility_does_not_require_future_entry_or_label():
    frame = _training_frame().head(3).copy()
    frame["entry_price"] = np.nan
    frame["buy_return_5m_pct"] = np.nan
    mask = _eligible(frame, 5)
    assert mask.all()


def test_calibration_cutoffs_do_not_depend_on_future_entry_price():
    frame = _training_frame()
    first = train_fold_model(frame, 5)

    changed = frame.copy()
    changed["entry_price"] = changed["entry_price"] * 10.0
    second = train_fold_model(changed, 5)

    assert first.calibration_cutoffs == second.calibration_cutoffs
