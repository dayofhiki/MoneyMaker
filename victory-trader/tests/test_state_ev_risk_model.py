import numpy as np
import pandas as pd

from victory_trader.state_ev_risk_model import (
    _non_overlapping,
    score_states,
    train_fold,
)


def _frame(month: int) -> pd.DataFrame:
    rows = []
    for day in range(1, 21):
        for episode in range(30):
            quality = (episode % 10) / 9.0
            for minute in range(0, 30, 3):
                good = quality > 0.65
                gross10 = 3.0 * quality - 1.0
                mae15 = -6.0 if (episode + minute) % 13 == 0 else -1.5
                rows.append(
                    {
                        "trading_day": f"2026-{month:02d}-{day:02d}",
                        "ticker": f"T{episode:03d}",
                        "t": day * 100_000_000 + episode * 1_000_000 + minute * 60_000,
                        "c": 5.0 + quality,
                        "entry_price": 5.01 + quality,
                        "active_minute_fraction_15m": 1.0,
                        "minutes_since_10pct_cross": float(minute),
                        "return_from_previous_close_pct": 10.0 + quality * 10.0,
                        "regular_vwap_distance_pct": quality,
                        "hod_distance_pct": -2.0 + quality,
                        "trailing_return_3m_pct": quality,
                        "bar_range_pct": 1.0 - 0.4 * quality,
                        "above_regular_vwap": bool(good),
                        "new_hod": bool(quality > 0.8),
                        "reclaim_prior_5m_high_after_pullback": bool(good),
                        "volume_1m": 1000.0 + quality * 1000.0,
                        "volume_3m": 3000.0 + quality * 2000.0,
                        "volume_5m": 5000.0 + quality * 3000.0,
                        "transactions_1m": 100.0 + quality * 100.0,
                        "transactions_5m": 500.0 + quality * 300.0,
                        "dollar_volume_5m": 25_000.0 + quality * 50_000.0,
                        "buy_return_10m_pct": gross10,
                        "buy_return_10m_base_net_return_pct": gross10 - 0.8,
                        "buy_return_10m_stress_net_return_pct": gross10 - 2.0,
                        "buy_return_15m_pct": gross10 + 0.1,
                        "buy_mae_15m_pct": mae15,
                    }
                )
    return pd.DataFrame(rows)


def test_ev_risk_fold_scores_without_future_entry_dependency():
    train = pd.concat([_frame(1), _frame(2)], ignore_index=True)
    fold = train_fold(train)

    holdout = _frame(3).head(200).copy()
    holdout["entry_price"] = np.nan
    scored = score_states(holdout, fold)

    assert len(scored) == len(holdout)
    assert scored["predicted_mean_gross_10m_pct"].notna().all()
    assert scored["predicted_mean_base_10m_pct"].notna().all()
    assert scored["predicted_severe_mae_prob"].between(0, 1).all()


def test_positive_ev_gate_can_create_non_overlapping_trades():
    train = pd.concat([_frame(1), _frame(2)], ignore_index=True)
    fold = train_fold(train)
    scored = score_states(_frame(3), fold)

    trades = _non_overlapping(scored, ev_gate=0.0, risk_gate=False)
    assert not trades.empty
    assert (trades["predicted_mean_base_10m_pct"] >= 0.0).all()
