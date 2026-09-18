import pandas as pd

from victory_trader.state_discovery import (
    evaluate_month,
    first_occurrences,
    state_masks,
    summarize_stability,
)


def _panel(month: str) -> pd.DataFrame:
    rows = []
    for episode in range(40):
        ticker = f"T{episode:03d}"
        for minute in range(6):
            hod_distance = -0.2 * minute
            above_vwap = True
            volume_ratio = 1.0
            range_ratio = 0.9
            reclaim = False
            new_hod = minute == 0
            trailing3 = 1.0 if minute == 0 else -0.2

            if minute == 3:
                hod_distance = -2.0
                volume_ratio = 0.5
            if minute == 4:
                hod_distance = -0.5
                reclaim = True
                trailing3 = 0.5
            if minute == 5 and episode % 2 == 0:
                hod_distance = -6.0
                above_vwap = False

            row = {
                "trading_day": f"{month}-{1 + episode % 8:02d}",
                "ticker": ticker,
                "t": minute,
                "minutes_since_10pct_cross": minute,
                "hod_distance_pct": hod_distance,
                "above_regular_vwap": above_vwap,
                "volume_3m_vs_postcross_peak": volume_ratio,
                "range_contraction_3m_vs_15m": range_ratio,
                "trailing_return_3m_pct": trailing3,
                "reclaim_prior_5m_high_after_pullback": reclaim,
                "new_hod": new_hod,
                "buy_mfe_15m_pct": 3.0,
                "buy_mae_15m_pct": -1.0,
            }
            for horizon in (5, 10, 15):
                gross = 2.0 if reclaim else 0.2
                base = 1.0 if reclaim else -0.8
                row[f"buy_return_{horizon}m_pct"] = gross
                row[f"buy_return_{horizon}m_base_net_return_pct"] = base
                row[f"buy_return_{horizon}m_stress_net_return_pct"] = base - 2.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_first_occurrence_only_one_signal_per_episode():
    frame = _panel("2026-01")
    reclaim = state_masks(frame)["reclaim_after_pullback"]
    selected = first_occurrences(frame, reclaim)
    assert len(selected) == 40
    assert selected[["trading_day", "ticker"]].drop_duplicates().shape[0] == 40


def test_evaluate_month_finds_positive_reclaim_state():
    details = evaluate_month(_panel("2026-01"), "2026-01")
    row = details.loc[
        details["state"].eq("reclaim_after_pullback")
        & details["horizon_min"].eq(15)
    ].iloc[0]
    assert row["signal_n"] == 40
    assert row["base_mean_pct"] == 1.0


def test_stability_counts_positive_months():
    details = pd.concat(
        [
            evaluate_month(_panel("2026-01"), "2026-01"),
            evaluate_month(_panel("2026-02"), "2026-02"),
            evaluate_month(_panel("2026-03"), "2026-03"),
        ],
        ignore_index=True,
    )
    stability = summarize_stability(details)
    row = stability.loc[
        stability["state"].eq("reclaim_after_pullback")
        & stability["horizon_min"].eq(15)
    ].iloc[0]
    assert row["months_tested"] == 3
    assert row["base_positive_months"] == 3
