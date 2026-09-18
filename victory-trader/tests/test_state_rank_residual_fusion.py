import numpy as np
import pandas as pd

from victory_trader.state_rank_residual_fusion import (
    _fit_rank_residual_corrections,
    _select_fused_trades,
)


def _crossfit_frame(ranks: list[float], realized: list[float]) -> pd.DataFrame:
    data: dict[str, object] = {}
    for horizon in (5, 10, 15):
        data[f"predicted_base_ev_{horizon}m_pct"] = [0.0] * len(ranks)
        data[f"predicted_episode_rank_{horizon}m"] = ranks
        data[f"buy_return_{horizon}m_base_net_return_pct"] = realized
    return pd.DataFrame(data)


def test_rank_residual_correction_is_monotone():
    left = _crossfit_frame(
        [0.1, 0.2, 0.8, 0.9],
        [-2.0, -1.0, 1.0, 2.0],
    )
    right = _crossfit_frame(
        [0.15, 0.25, 0.75, 0.85],
        [-1.5, -0.5, 0.5, 1.5],
    )

    corrections, diagnostics = _fit_rank_residual_corrections(
        [("2026-01", left), ("2026-02", right)],
        holdout_month="2026-03",
    )

    assert set(corrections) == {5, 10, 15}
    for correction in corrections.values():
        predicted = correction.model.predict(np.asarray([0.2, 0.8]))
        assert predicted[0] <= predicted[1]

    assert diagnostics["residual_rank_spearman"].gt(0).all()


def _row(
    minute: int,
    *,
    fused_ev: float,
    horizon: int = 15,
    entry_price: float = 5.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "trading_day": "2026-01-02",
        "ticker": "AAA",
        "t": minute * 60_000,
        "minutes_since_10pct_cross": float(minute),
        "entry_price": entry_price,
        "predicted_best_fused_ev_pct": fused_ev,
        "predicted_best_fused_horizon_min": horizon,
    }
    for current in (5, 10, 15):
        row[f"predicted_base_ev_{current}m_pct"] = 0.4 + current / 100
        row[f"predicted_episode_rank_{current}m"] = 0.3 + current / 100
        row[f"buy_return_{current}m_base_net_return_pct"] = current / 10
        row[f"buy_return_{current}m_pct"] = current / 10 + 1.0
        row[f"buy_return_{current}m_stress_net_return_pct"] = current / 10 - 1.0
    return row


def test_fused_policy_enters_first_gate_crossing():
    frame = pd.DataFrame(
        [
            _row(0, fused_ev=0.4),
            _row(1, fused_ev=0.7, horizon=10),
            _row(2, fused_ev=1.2, horizon=15),
        ]
    )
    trades = _select_fused_trades(frame)

    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000
    assert int(trades.iloc[0]["action_horizon_min"]) == 10
    assert trades.iloc[0]["realized_base_net_return_pct"] == 1.0
    assert trades.iloc[0]["selected_predicted_base_ev_pct"] == 0.7


def test_first_fused_signal_is_consumed_when_unfillable():
    frame = pd.DataFrame(
        [
            _row(0, fused_ev=0.7, entry_price=np.nan),
            _row(1, fused_ev=1.0),
        ]
    )
    trades = _select_fused_trades(frame)
    assert trades.empty


def test_gate_is_fixed_at_half_percent():
    frame = pd.DataFrame(
        [
            _row(0, fused_ev=0.499999),
            _row(1, fused_ev=0.5),
        ]
    )
    trades = _select_fused_trades(frame)

    assert len(trades) == 1
    assert int(trades.iloc[0]["t"]) == 60_000
