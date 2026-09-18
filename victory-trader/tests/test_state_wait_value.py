import numpy as np
import pandas as pd

from victory_trader.state_wait_value import (
    _select_first_trades,
    wait_advantage_target,
)


def _row(
    minute: int,
    *,
    ticker: str = "AAA",
    day: str = "2026-01-02",
    ev: float = 0.8,
    wait_pred: float = -0.1,
    entry_price: float = 5.0,
    ret5: float = 0.5,
    ret10: float = 1.0,
    ret15: float = 1.5,
) -> dict[str, object]:
    return {
        "trading_day": day,
        "ticker": ticker,
        "t": minute * 60_000,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": entry_price,
        "predicted_best_base_ev_pct": ev,
        "predicted_best_horizon_min": 15,
        "predicted_wait_advantage_pct": wait_pred,
        "buy_return_5m_base_net_return_pct": ret5,
        "buy_return_10m_base_net_return_pct": ret10,
        "buy_return_15m_base_net_return_pct": ret15,
        "buy_return_5m_pct": ret5 + 1.0,
        "buy_return_10m_pct": ret10 + 1.0,
        "buy_return_15m_pct": ret15 + 1.0,
        "buy_return_5m_stress_net_return_pct": ret5 - 1.0,
        "buy_return_10m_stress_net_return_pct": ret10 - 1.0,
        "buy_return_15m_stress_net_return_pct": ret15 - 1.0,
    }


def test_wait_target_uses_strictly_future_state_inside_same_episode():
    frame = pd.DataFrame(
        [
            _row(0, ret15=1.0),
            _row(5, ret15=4.0),
            _row(10, ret15=2.0),
        ]
    )
    target = wait_advantage_target(frame)
    assert target.iloc[0] == 3.0
    assert target.iloc[1] == -2.0
    assert np.isnan(target.iloc[2])


def test_wait_target_honors_window_and_does_not_cross_ticker():
    frame = pd.DataFrame(
        [
            _row(0, ticker="AAA", ret15=1.0),
            _row(20, ticker="AAA", ret15=10.0),
            _row(5, ticker="BBB", ret15=20.0),
        ]
    )
    target = wait_advantage_target(frame, wait_window_minutes=15)
    assert np.isnan(target.iloc[0])
    assert np.isnan(target.iloc[1])
    assert np.isnan(target.iloc[2])


def test_baseline_takes_first_ev_qualified_state():
    frame = pd.DataFrame(
        [
            _row(0, wait_pred=2.0),
            _row(1, wait_pred=-1.0),
        ]
    )
    trades = _select_first_trades(frame, policy="earliest_ev_cap1")
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 0


def test_wait_policy_waits_until_continuation_value_is_not_positive():
    frame = pd.DataFrame(
        [
            _row(0, wait_pred=1.0),
            _row(1, wait_pred=0.2),
            _row(2, wait_pred=0.0),
            _row(3, wait_pred=-0.5),
        ]
    )
    trades = _select_first_trades(frame, policy="wait_value_cap1")
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 2 * 60_000


def test_wait_policy_still_requires_direct_ev_gate():
    frame = pd.DataFrame(
        [
            _row(0, ev=0.4, wait_pred=-1.0),
            _row(1, ev=0.8, wait_pred=-0.1),
        ]
    )
    trades = _select_first_trades(frame, policy="wait_value_cap1")
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000


def test_first_unfillable_accepted_signal_does_not_fall_through():
    frame = pd.DataFrame(
        [
            _row(0, wait_pred=-0.1, entry_price=np.nan),
            _row(1, wait_pred=-0.2),
        ]
    )
    trades = _select_first_trades(frame, policy="wait_value_cap1")
    assert trades.empty
