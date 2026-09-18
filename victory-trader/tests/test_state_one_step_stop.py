import numpy as np
import pandas as pd

from victory_trader.state_one_step_stop import (
    _select_first_trades,
    one_step_delta_target,
)


def _row(
    minute: int,
    *,
    ticker: str = "AAA",
    day: str = "2026-01-02",
    best_horizon: int = 15,
    ev: float = 0.8,
    selected_delta: float = -0.1,
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
        "predicted_best_horizon_min": best_horizon,
        "predicted_selected_one_step_delta_pct": selected_delta,
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


def test_one_step_target_uses_exact_next_minute():
    frame = pd.DataFrame(
        [
            _row(0, ret10=1.0),
            _row(1, ret10=4.0),
            _row(2, ret10=2.0),
        ]
    )
    target = one_step_delta_target(frame, 10)
    assert target.iloc[0] == 3.0
    assert target.iloc[1] == -2.0
    assert np.isnan(target.iloc[2])


def test_one_step_target_rejects_gap_and_episode_crossing():
    frame = pd.DataFrame(
        [
            _row(0, ticker="AAA", ret10=1.0),
            _row(2, ticker="AAA", ret10=8.0),
            _row(1, ticker="BBB", ret10=20.0),
        ]
    )
    target = one_step_delta_target(frame, 10)
    assert np.isnan(target.iloc[0])
    assert np.isnan(target.iloc[1])
    assert np.isnan(target.iloc[2])


def test_baseline_enters_first_ev_qualified_state():
    frame = pd.DataFrame(
        [
            _row(0, selected_delta=1.0),
            _row(1, selected_delta=-1.0),
        ]
    )
    trades = _select_first_trades(frame, policy="earliest_ev_cap1")
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 0


def test_one_step_policy_waits_while_next_minute_is_expected_better():
    frame = pd.DataFrame(
        [
            _row(0, selected_delta=0.4),
            _row(1, selected_delta=0.1),
            _row(2, selected_delta=0.0),
            _row(3, selected_delta=-0.2),
        ]
    )
    trades = _select_first_trades(frame, policy="one_step_stop_cap1")
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 2 * 60_000


def test_one_step_policy_still_requires_ev_gate():
    frame = pd.DataFrame(
        [
            _row(0, ev=0.4, selected_delta=-1.0),
            _row(1, ev=0.8, selected_delta=-0.1),
        ]
    )
    trades = _select_first_trades(frame, policy="one_step_stop_cap1")
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000


def test_selected_horizon_controls_realized_trade_return():
    frame = pd.DataFrame(
        [
            _row(
                0,
                best_horizon=10,
                selected_delta=-0.1,
                ret5=0.5,
                ret10=3.0,
                ret15=-2.0,
            )
        ]
    )
    trades = _select_first_trades(frame, policy="one_step_stop_cap1")
    assert int(trades.iloc[0]["action_horizon_min"]) == 10
    assert trades.iloc[0]["realized_base_net_return_pct"] == 3.0


def test_first_unfillable_accepted_signal_does_not_fall_through():
    frame = pd.DataFrame(
        [
            _row(0, selected_delta=-0.1, entry_price=np.nan),
            _row(1, selected_delta=-0.2),
        ]
    )
    trades = _select_first_trades(frame, policy="one_step_stop_cap1")
    assert trades.empty
