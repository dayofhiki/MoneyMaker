import numpy as np
import pandas as pd

from victory_trader.state_entry_ranker import (
    _select_first_trades,
    episode_percentile_target,
    evaluation_folds,
)


def _row(
    minute: int,
    *,
    best_horizon: int = 5,
    rank_5m: float = 0.4,
    rank_10m: float = 0.3,
    ev_5m: float = 0.8,
    ev_10m: float = 0.7,
    entry_price: float = 5.0,
) -> dict[str, object]:
    return {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": minute * 60_000,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": entry_price,
        "predicted_best_base_ev_pct": max(ev_5m, ev_10m),
        "predicted_best_horizon_min": best_horizon,
        "predicted_base_ev_5m_pct": ev_5m,
        "predicted_base_ev_10m_pct": ev_10m,
        "predicted_base_ev_15m_pct": 0.1,
        "predicted_episode_rank_5m": rank_5m,
        "predicted_episode_rank_10m": rank_10m,
        "predicted_episode_rank_15m": 0.2,
        "rank_action_horizon_min": 5 if rank_5m >= rank_10m else 10,
        "predicted_rank_score": max(rank_5m, rank_10m),
        "rank_action_predicted_base_ev_pct": (
            ev_5m if rank_5m >= rank_10m else ev_10m
        ),
        "buy_return_5m_base_net_return_pct": 0.5,
        "buy_return_10m_base_net_return_pct": 1.5,
        "buy_return_15m_base_net_return_pct": 2.5,
        "buy_return_5m_pct": 1.5,
        "buy_return_10m_pct": 2.5,
        "buy_return_15m_pct": 3.5,
        "buy_return_5m_stress_net_return_pct": -1.0,
        "buy_return_10m_stress_net_return_pct": 0.0,
        "buy_return_15m_stress_net_return_pct": 1.0,
    }


def test_episode_percentile_target_is_computed_inside_each_episode():
    frame = pd.DataFrame(
        {
            "trading_day": ["2026-01-02"] * 16,
            "ticker": ["AAA"] * 8 + ["BBB"] * 8,
            "buy_return_5m_base_net_return_pct": list(range(8))
            + list(range(80, 0, -10)),
        }
    )
    ranks = episode_percentile_target(frame, 5)
    assert ranks.iloc[0] == 1 / 8
    assert ranks.iloc[7] == 1.0
    assert ranks.iloc[8] == 1.0
    assert ranks.iloc[15] == 1 / 8


def test_baseline_takes_first_ev_qualified_state():
    frame = pd.DataFrame([_row(0), _row(1, rank_5m=0.9)])
    trades = _select_first_trades(
        frame,
        policy="earliest_ev_cap1",
        rank_gate=0.75,
    )
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 0
    assert int(trades.iloc[0]["action_horizon_min"]) == 5


def test_rank_policy_waits_causally_for_first_rank_qualified_state():
    frame = pd.DataFrame(
        [_row(0, rank_5m=0.4), _row(1, rank_5m=0.8), _row(2, rank_5m=0.9)]
    )
    trades = _select_first_trades(
        frame,
        policy="rank_top25_cap1",
        rank_gate=0.75,
    )
    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000


def test_rank_policy_can_choose_non_max_ev_horizon():
    frame = pd.DataFrame(
        [
            _row(
                0,
                best_horizon=5,
                rank_5m=0.80,
                rank_10m=0.95,
                ev_5m=1.0,
                ev_10m=0.6,
            )
        ]
    )
    trades = _select_first_trades(
        frame,
        policy="rank_top25_cap1",
        rank_gate=0.75,
    )
    assert int(trades.iloc[0]["action_horizon_min"]) == 10
    assert trades.iloc[0]["realized_base_net_return_pct"] == 1.5


def test_first_unfillable_signal_does_not_fall_through_to_future_state():
    frame = pd.DataFrame(
        [
            _row(0, rank_5m=0.9, entry_price=np.nan),
            _row(1, rank_5m=0.95),
        ]
    )
    trades = _select_first_trades(
        frame,
        policy="rank_top25_cap1",
        rank_gate=0.75,
    )
    assert trades.empty


def test_forward_folds_use_strictly_prior_months():
    monthly = {
        month: pd.DataFrame({"trading_day": [month + "-15"]})
        for month in (
            "2025-09",
            "2025-10",
            "2025-11",
            "2025-12",
            "2026-01",
            "2026-02",
            "2026-03",
        )
    }
    folds = list(
        evaluation_folds(
            monthly,
            "forward",
            evaluation_min_month="2026-01",
            evaluation_max_month="2026-03",
        )
    )
    assert [month for month, _, _ in folds] == [
        "2026-01",
        "2026-02",
        "2026-03",
    ]
    assert folds[0][1] == ["2025-09", "2025-10", "2025-11", "2025-12"]
    assert folds[1][1][-1] == "2026-01"
    assert folds[2][1][-1] == "2026-02"
