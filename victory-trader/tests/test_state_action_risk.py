import pandas as pd

from victory_trader.state_action_risk import (
    PolicyVariant,
    _select_trades,
)


def _row(minute: int, *, risk: float = 0.10) -> dict[str, object]:
    return {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": minute * 60_000,
        "entry_price": 5.0,
        "predicted_best_base_ev_pct": 0.75,
        "predicted_best_severe_loss_prob": risk,
        "predicted_best_horizon_min": 5,
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


def test_cap_one_policy_allows_only_first_executable_entry():
    frame = pd.DataFrame([_row(minute) for minute in range(20)])
    policy = PolicyVariant("cap1", 1, None)
    trades = _select_trades(frame, policy=policy)
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 0


def test_risk_gate_skips_unsafe_state_and_uses_next_safe_state():
    frame = pd.DataFrame(
        [_row(0, risk=0.50), _row(1, risk=0.10), _row(8, risk=0.05)]
    )
    policy = PolicyVariant("cap1_risk20", 1, 0.20)
    trades = _select_trades(frame, policy=policy)
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000


def test_repeat_policy_still_respects_selected_horizon():
    frame = pd.DataFrame([_row(minute) for minute in range(12)])
    policy = PolicyVariant("repeat", None, None)
    trades = _select_trades(frame, policy=policy)
    assert list(trades["t"].astype(int)) == [0, 5 * 60_000, 10 * 60_000]
