import pandas as pd

from victory_trader.candidate_v2_exit_research import (
    run_exit_research,
    summarize_exits,
)


def _month(month: int) -> pd.DataFrame:
    rows = []
    for day in range(1, 7):
        for i in range(100):
            good = float(i)
            entry = 4.0 + i * 0.05
            row = {
                "trading_day": f"2026-{month:02d}-{day:02d}",
                "threshold_pct": 10.0 if i < 70 else 20.0,
                "ticker": f"T{i:03d}",
                "entry_price": entry,
                "volatility_15m_pct": good,
                "signal_bar_range_pct": good + 0.1,
                "prior_15m_low_rebound_pct": good + 0.2,
                "dollar_volume_5m": 2_000_000.0 + (i % 9) * 10_000,
                "transactions_5m": 2_000.0 + (i % 7) * 10,
                "active_minute_fraction_15m": 0.9 + (i % 4) * 0.02,
            }
            for key, win_return, loss_return in (
                ("tp2_sl1", 2.0, -1.0),
                ("tp3_sl2", 3.0, -2.0),
                ("tp5_sl3", 5.0, -3.0),
                ("tp10_sl5", 10.0, -5.0),
            ):
                is_good = i < 20
                row[f"{key}_status"] = "take_profit" if is_good else "stop_loss"
                gross = win_return if is_good else loss_return
                row[f"{key}_exit_return_pct"] = gross
            rows.append(row)
    return pd.DataFrame(rows)


def test_exit_research_scores_all_months_and_barriers():
    frames = {"jan": _month(1), "feb": _month(2), "mar": _month(3)}
    details = run_exit_research(frames)
    assert set(details["month"]) == {"jan", "feb", "mar"}
    assert set(details["barrier"]) == {
        "tp2_sl1",
        "tp3_sl2",
        "tp5_sl3",
        "tp10_sl5",
    }


def test_exit_summary_prefers_profitable_barrier_on_synthetic_path():
    frames = {"jan": _month(1), "feb": _month(2), "mar": _month(3)}
    details = run_exit_research(frames)
    summary = summarize_exits(details)
    assert not summary.empty
    assert summary.iloc[0]["gross_positive_months"] >= 1
