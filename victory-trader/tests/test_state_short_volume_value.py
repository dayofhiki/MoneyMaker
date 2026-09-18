from datetime import date

import numpy as np
import pandas as pd

from victory_trader.state_short_volume_enrichment import (
    SHORT_VOLUME_FEATURES,
    _feature_row,
)
from victory_trader.state_short_volume_value import (
    _select_first_trades,
    short_volume_action_feature_frame,
)


def _record(day: str, ratio: float, short: float, total: float):
    return {
        "_record_day": pd.Timestamp(day).date(),
        "short_volume_ratio": ratio,
        "short_volume": short,
        "total_volume": total,
    }


def test_short_volume_features_use_only_records_before_event_day():
    records = [
        _record("2026-01-02", 10.0, 100.0, 1000.0),
        _record("2026-01-05", 20.0, 200.0, 1100.0),
        _record("2026-01-06", 30.0, 300.0, 1200.0),
        _record("2026-01-07", 40.0, 400.0, 1300.0),
        _record("2026-01-08", 50.0, 500.0, 1400.0),
        _record("2026-01-09", 99.0, 999.0, 9999.0),
    ]
    features = _feature_row(records, date(2026, 1, 9))

    assert features["short_ratio_latest_prior"] == 50.0
    assert features["short_ratio_mean5_prior"] == 30.0
    assert features["short_ratio_latest_minus_mean5"] == 20.0
    assert features["short_volume_latest_age_days"] == 1.0


def test_twenty_record_features_require_full_twenty_records():
    records = [
        _record(
            f"2026-01-{day:02d}",
            float(day),
            float(day * 10),
            float(day * 100),
        )
        for day in range(1, 11)
    ]
    features = _feature_row(records, date(2026, 1, 20))

    assert np.isnan(features["short_ratio_mean20_prior"])
    assert np.isnan(features["short_ratio_mean5_minus_mean20"])
    assert np.isfinite(features["short_ratio_mean5_prior"])


def test_augmented_feature_frame_contains_all_fixed_short_volume_features():
    frame = pd.DataFrame(
        {
            "return_from_previous_close_pct": [10.0],
            "minutes_since_10pct_cross": [1.0],
            "minutes_from_regular_open": [10.0],
            "active_minute_fraction_15m": [1.0],
            "c": [5.0],
            "previous_close": [4.0],
            **{feature: [1.0] for feature in SHORT_VOLUME_FEATURES},
        }
    )
    features = short_volume_action_feature_frame(frame)
    for column in SHORT_VOLUME_FEATURES:
        assert column in features.columns
        assert features.iloc[0][column] == 1.0


def _row(
    minute: int,
    *,
    short_ev: float,
    horizon: int = 15,
    entry_price: float = 5.0,
):
    row = {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": minute * 60_000,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": entry_price,
        "predicted_best_base_ev_pct": 0.0,
        "predicted_best_horizon_min": horizon,
        "predicted_best_short_volume_ev_pct": short_ev,
        "predicted_best_short_volume_horizon_min": horizon,
    }
    for current in (5, 10, 15):
        row[f"buy_return_{current}m_base_net_return_pct"] = current / 10
        row[f"buy_return_{current}m_pct"] = current / 10 + 1.0
        row[f"buy_return_{current}m_stress_net_return_pct"] = current / 10 - 1.0
    return row


def test_short_volume_policy_enters_first_half_percent_signal():
    frame = pd.DataFrame(
        [
            _row(0, short_ev=0.49),
            _row(1, short_ev=0.50, horizon=10),
            _row(2, short_ev=2.00, horizon=15),
        ]
    )
    trades = _select_first_trades(frame, policy="short_volume_ev_cap1")
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000
    assert int(trades.iloc[0]["action_horizon_min"]) == 10


def test_first_short_volume_signal_is_consumed_when_unfillable():
    frame = pd.DataFrame(
        [
            _row(0, short_ev=0.75, entry_price=np.nan),
            _row(1, short_ev=1.50),
        ]
    )
    trades = _select_first_trades(frame, policy="short_volume_ev_cap1")
    assert trades.empty
