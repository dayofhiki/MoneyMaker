import numpy as np
import pandas as pd

from victory_trader.state_robust_median_value import (
    _median_calibration_offset,
    select_first_trades,
    success_check,
)


def test_median_calibration_uses_only_additive_median_residual():
    pred = np.array([1.0, 2.0, 3.0] * 40)
    actual = pred + 0.75
    assert np.isclose(_median_calibration_offset(pred, actual), 0.75)


def test_median_calibration_returns_zero_when_too_sparse():
    pred = np.array([1.0] * 99)
    actual = np.array([2.0] * 99)
    assert _median_calibration_offset(pred, actual) == 0.0


def _trade_row(minute, *, median_score, horizon=15, entry_price=5.0):
    row = {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": minute * 60_000,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": entry_price,
        "predicted_best_base_ev_pct": 0.0,
        "predicted_best_horizon_min": horizon,
        "predicted_best_median_pct": median_score,
        "predicted_best_median_horizon_min": horizon,
    }
    for current in (5, 10, 15):
        row[f"buy_return_{current}m_base_net_return_pct"] = current / 10
        row[f"buy_return_{current}m_pct"] = current / 10 + 1.0
        row[f"buy_return_{current}m_stress_net_return_pct"] = current / 10 - 1.0
    return row


def test_median_policy_uses_first_half_percent_signal():
    frame = pd.DataFrame(
        [
            _trade_row(0, median_score=0.49),
            _trade_row(1, median_score=0.50, horizon=10),
            _trade_row(2, median_score=5.00, horizon=15),
        ]
    )
    trades = select_first_trades(frame, policy="median_multi_source_cap1")
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000
    assert int(trades.iloc[0]["action_horizon_min"]) == 10


def test_median_first_signal_consumed_if_unfillable():
    frame = pd.DataFrame(
        [
            _trade_row(0, median_score=1.0, entry_price=np.nan),
            _trade_row(1, median_score=2.0),
        ]
    )
    trades = select_first_trades(frame, policy="median_multi_source_cap1")
    assert trades.empty


def test_median_success_check_requires_all_preregistered_rules():
    rows = []
    for month in ("2026-01", "2026-02", "2026-03"):
        rows.extend(
            [
                {
                    "month": month,
                    "policy": "earliest_ev_cap1",
                    "trades": 20,
                    "base_mean_pct": 0.5,
                    "day_balanced_base_mean_pct": 0.5,
                    "base_p05_pct": -10.0,
                    "stress_mean_pct": -1.0,
                },
                {
                    "month": month,
                    "policy": "median_multi_source_cap1",
                    "trades": 20,
                    "base_mean_pct": 1.0,
                    "day_balanced_base_mean_pct": 1.0,
                    "base_p05_pct": -9.0,
                    "stress_mean_pct": 0.0,
                },
            ]
        )
    details = pd.DataFrame(rows)
    coverage = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02", "2026-03"],
            "short_volume_latest_coverage": [1.0, 1.0, 1.0],
            "short_interest_latest_coverage": [1.0, 1.0, 1.0],
            "eight_k_query_complete_coverage": [1.0, 1.0, 1.0],
        }
    )

    assert success_check(
        details,
        coverage,
        {"ci_low_pct": 0.1},
    )["all_pass"] is True

    details.loc[
        (details["month"].eq("2026-03"))
        & details["policy"].eq("median_multi_source_cap1"),
        "base_mean_pct",
    ] = -0.1
    assert success_check(
        details,
        coverage,
        {"ci_low_pct": 0.1},
    )["all_pass"] is False
