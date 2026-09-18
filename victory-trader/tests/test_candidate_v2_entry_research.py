import pandas as pd

from victory_trader.candidate_v2_entry_research import (
    run_entry_timing_research,
    summarize_entry_timing,
)


def _month(month: int) -> pd.DataFrame:
    rows = []
    for day in range(1, 7):
        for i in range(100):
            good = float(i)
            row = {
                "trading_day": f"2026-{month:02d}-{day:02d}",
                "threshold_pct": 10.0 if i < 70 else 20.0,
                "ticker": f"T{i:03d}",
                "entry_price": 4.0 + i * 0.05,
                "delay1_entry_price": 4.1 + i * 0.05,
                "delay2_entry_price": 4.2 + i * 0.05,
                "volatility_15m_pct": good,
                "signal_bar_range_pct": good + 0.1,
                "prior_15m_low_rebound_pct": good + 0.2,
                "dollar_volume_5m": 2_000_000.0 + (i % 9) * 10_000,
                "transactions_5m": 2_000.0 + (i % 7) * 10,
                "active_minute_fraction_15m": 0.9 + (i % 4) * 0.02,
            }
            for delay, prefix in ((0, ""), (1, "delay1_"), (2, "delay2_")):
                for horizon in (5, 10, 15):
                    gross = 1.5 - good * 0.02 + delay * 0.25 + horizon * 0.001
                    row[f"{prefix}return_{horizon}m_pct"] = gross
                    row[f"{prefix}return_{horizon}m_base_net_return_pct"] = gross - 0.8
                    row[f"{prefix}return_{horizon}m_stress_net_return_pct"] = gross - 2.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_entry_timing_research_scores_all_months_delays_and_horizons():
    frames = {"jan": _month(1), "feb": _month(2), "mar": _month(3)}
    details = run_entry_timing_research(frames)
    assert set(details["month"]) == {"jan", "feb", "mar"}
    assert set(details["delay_min"]) == {0, 1, 2}
    assert set(details["horizon_min"]) == {5, 10, 15}


def test_entry_timing_summary_prefers_better_delayed_entry_on_synthetic_data():
    frames = {"jan": _month(1), "feb": _month(2), "mar": _month(3)}
    details = run_entry_timing_research(frames)
    summary = summarize_entry_timing(details)
    assert not summary.empty
    assert summary.iloc[0]["delay_min"] == 2
