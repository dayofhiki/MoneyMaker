import numpy as np
import pandas as pd

from victory_trader.state_rank_turn import (
    _select_rank_turn_trades,
)


def _row(
    minute: int,
    *,
    ev: float = 0.8,
    score: float = 0.5,
    horizon: int = 15,
    entry_price: float = 5.0,
) -> dict[str, object]:
    return {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": minute * 60_000,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": entry_price,
        "rank_action_predicted_base_ev_pct": ev,
        "predicted_rank_score": score,
        "rank_action_horizon_min": horizon,
        "buy_return_5m_base_net_return_pct": 0.5,
        "buy_return_10m_base_net_return_pct": 1.0,
        "buy_return_15m_base_net_return_pct": 1.5,
        "buy_return_5m_pct": 1.5,
        "buy_return_10m_pct": 2.0,
        "buy_return_15m_pct": 2.5,
        "buy_return_5m_stress_net_return_pct": -0.5,
        "buy_return_10m_stress_net_return_pct": 0.0,
        "buy_return_15m_stress_net_return_pct": 0.5,
    }


def test_rank_turn_waits_while_score_rises_then_enters_on_first_nonincrease():
    frame = pd.DataFrame(
        [
            _row(0, score=0.4),
            _row(1, score=0.5),
            _row(2, score=0.6),
            _row(3, score=0.55),
            _row(4, score=0.7),
        ]
    )
    trades = _select_rank_turn_trades(frame)
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 3 * 60_000


def test_equal_score_counts_as_turn():
    frame = pd.DataFrame([_row(0, score=0.5), _row(1, score=0.5)])
    trades = _select_rank_turn_trades(frame)
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000


def test_ev_break_resets_tracker():
    frame = pd.DataFrame(
        [
            _row(0, score=0.6),
            _row(1, ev=0.4, score=0.2),
            _row(2, score=0.5),
            _row(3, score=0.4),
        ]
    )
    trades = _select_rank_turn_trades(frame)
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 3 * 60_000


def test_time_gap_resets_tracker():
    frame = pd.DataFrame(
        [
            _row(0, score=0.6),
            _row(2, score=0.4),
            _row(3, score=0.3),
        ]
    )
    trades = _select_rank_turn_trades(frame)
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 3 * 60_000


def test_current_rank_selected_horizon_is_executed():
    frame = pd.DataFrame(
        [
            _row(0, score=0.5, horizon=15),
            _row(1, score=0.4, horizon=10),
        ]
    )
    trades = _select_rank_turn_trades(frame)
    assert int(trades.iloc[0]["action_horizon_min"]) == 10
    assert trades.iloc[0]["realized_base_net_return_pct"] == 1.0


def test_first_turn_signal_consumed_when_unfillable():
    frame = pd.DataFrame(
        [
            _row(0, score=0.5),
            _row(1, score=0.4, entry_price=np.nan),
            _row(2, score=0.3),
        ]
    )
    trades = _select_rank_turn_trades(frame)
    assert trades.empty
