import numpy as np
import pandas as pd

from victory_trader.state_action_value import (
    _fit_affine_calibration,
    _non_overlapping_action_trades,
    action_feature_frame,
)


def test_affine_calibration_recovers_simple_linear_relation():
    prediction = np.array([0.0, 1.0, 2.0, 3.0])
    actual = 0.5 + 0.8 * prediction
    intercept, slope = _fit_affine_calibration(prediction, actual)
    assert abs(intercept - 0.5) < 1e-9
    assert abs(slope - 0.8) < 1e-9


def test_action_features_include_current_absolute_price():
    frame = pd.DataFrame(
        {
            "c": [2.0, 10.0],
            "previous_close": [1.5, 8.0],
        }
    )
    out = action_feature_frame(frame)
    assert "log_current_price" in out.columns
    assert "log_previous_close" in out.columns
    assert out.loc[1, "log_current_price"] > out.loc[0, "log_current_price"]


def test_non_overlapping_actions_respect_selected_horizon():
    rows = []
    for minute in range(20):
        rows.append(
            {
                "trading_day": "2026-01-02",
                "ticker": "AAA",
                "t": minute * 60_000,
                "entry_price": 5.0,
                "predicted_best_base_ev_pct": 1.0,
                "predicted_best_horizon_min": 10 if minute == 0 else 5,
                "buy_return_5m_base_net_return_pct": 0.5,
                "buy_return_10m_base_net_return_pct": 0.8,
                "buy_return_15m_base_net_return_pct": 1.0,
                "buy_return_5m_pct": 1.5,
                "buy_return_10m_pct": 1.8,
                "buy_return_15m_pct": 2.0,
                "buy_return_5m_stress_net_return_pct": -1.0,
                "buy_return_10m_stress_net_return_pct": -0.8,
                "buy_return_15m_stress_net_return_pct": -0.6,
            }
        )
    trades = _non_overlapping_action_trades(
        pd.DataFrame(rows),
        ev_gate=0.5,
    )
    assert int(trades.iloc[0]["action_horizon_min"]) == 10
    assert int(trades.iloc[1]["t"]) >= 10 * 60_000
