import pandas as pd

from victory_trader.candidate_v2_research import (
    modeled_base_zero_return_cost_pct,
    run_leave_one_month_out,
    summarize_configs,
)


def _month(month: int) -> pd.DataFrame:
    rows = []
    for day in range(1, 9):
        for i in range(100):
            good = float(i)
            price = 1.0 + i * 0.08
            row = {
                "trading_day": f"2026-{month:02d}-{day:02d}",
                "threshold_pct": 10.0 if i < 70 else 20.0,
                "ticker": f"T{i:03d}",
                "entry_price": price,
                "volatility_15m_pct": good,
                "signal_bar_range_pct": good + 0.1,
                "prior_15m_low_rebound_pct": good + 0.2,
                "dollar_volume_5m": 2_000_000.0 + (i % 9) * 10_000,
                "transactions_5m": 2_000.0 + (i % 7) * 10,
                "active_minute_fraction_15m": 0.8 + (i % 5) * 0.02,
            }
            for horizon in (5, 10, 15):
                gross = 3.0 - good * 0.035 + horizon * 0.002
                row[f"return_{horizon}m_pct"] = gross
                row[f"return_{horizon}m_base_net_return_pct"] = gross - 1.0
                row[f"return_{horizon}m_stress_net_return_pct"] = gross - 2.5
            rows.append(row)
    return pd.DataFrame(rows)


def test_base_zero_return_cost_falls_with_price():
    assert modeled_base_zero_return_cost_pct(1.0) > modeled_base_zero_return_cost_pct(5.0)


def test_leave_one_month_out_scores_every_month_and_grid():
    frames = {"jan": _month(1), "feb": _month(2), "mar": _month(3)}
    details = run_leave_one_month_out(frames)
    assert set(details["month"]) == {"jan", "feb", "mar"}
    assert set(details["horizon_min"]) == {5, 10, 15}
    assert details["config"].nunique() == 18


def test_stricter_alpha_cut_selects_fewer_rows():
    frames = {"jan": _month(1), "feb": _month(2), "mar": _month(3)}
    details = run_leave_one_month_out(frames)
    sample = details.loc[
        details["month"].eq("jan")
        & details["horizon_min"].eq(10)
        & details["alpha_family"].eq("core3")
        & details["base_cost_ceiling_pct"].isna()
    ]
    top20 = sample.loc[sample["selection_quantile"].eq(0.80), "selected_n"].iloc[0]
    top5 = sample.loc[sample["selection_quantile"].eq(0.95), "selected_n"].iloc[0]
    assert top5 < top20


def test_summary_has_robust_winner_on_synthetic_signal():
    frames = {"jan": _month(1), "feb": _month(2), "mar": _month(3)}
    details = run_leave_one_month_out(frames)
    summary = summarize_configs(details)
    assert summary["robust"].any()
    winner = summary.iloc[0]
    assert winner["base_lift_positive_months"] == 3
    assert winner["gross_positive_months"] >= 2
