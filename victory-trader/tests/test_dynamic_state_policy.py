import pandas as pd

from victory_trader.dynamic_state_policy import (
    metrics,
    simulate_one_position,
)


def _scored_frame() -> pd.DataFrame:
    rows = []
    # Two ticker-day episodes across one day. AAA is selected first,
    # exits when 5m prediction turns non-positive, then BBB can enter later.
    for t in (0, 60_000, 120_000, 180_000, 240_000):
        for ticker in ("AAA", "BBB"):
            minute = t // 60_000
            entry_price = 10.0 + 0.1 * minute + (0.5 if ticker == "BBB" else 0.0)
            score = -10.0
            pred5 = 0.5
            if ticker == "AAA" and minute == 0:
                score = 2.0
            if ticker == "AAA" and minute >= 2:
                pred5 = -0.1
            if ticker == "BBB" and minute == 3:
                score = 2.5

            rows.append(
                {
                    "trading_day": "2026-01-02",
                    "ticker": ticker,
                    "t": int(t),
                    "entry_price": entry_price,
                    "c": entry_price,
                    "predicted_base_10m_pct": score,
                    "predicted_gross_5m_pct": pred5,
                    "hod_distance_pct": -1.0,
                    "above_regular_vwap": True,
                }
            )
    return pd.DataFrame(rows)


def test_dynamic_policy_can_exit_and_reenter():
    trades, equity = simulate_one_position(
        _scored_frame(),
        entry_cutoff=1.0,
    )
    assert len(trades) >= 1
    assert trades.iloc[0]["ticker"] == "AAA"
    assert trades.iloc[0]["exit_reason"] == "predicted_5m_nonpositive"
    assert not equity.empty


def test_metrics_reports_compounded_month_result():
    trades, equity = simulate_one_position(
        _scored_frame(),
        entry_cutoff=1.0,
    )
    result = metrics(
        trades,
        equity,
        month="2026-01",
        entry_cutoff=1.0,
    )
    assert result["month"] == "2026-01"
    assert result["trades"] == len(trades)
    assert "base_month_return_pct" in result
    assert "base_max_drawdown_pct" in result
