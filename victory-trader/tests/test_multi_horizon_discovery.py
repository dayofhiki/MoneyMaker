import pandas as pd

from victory_trader.multi_horizon_discovery import (
    collect_multi_horizon_holdouts,
    render_multi_horizon_report,
    summarize_cross_horizon_stability,
)


def _frame() -> pd.DataFrame:
    rows = []
    horizons = (1, 2, 5, 10, 15, 30, 60)
    for day in range(1, 15):
        for i in range(30):
            feature = float(i + 1)
            row = {
                "trading_day": f"2026-03-{day:02d}",
                "threshold_pct": 10.0,
                "previous_close": feature,
                "entry_price": feature + 0.1,
                "timestamp_ms": 1_772_450_000_000 + day * 86_400_000 + i * 60_000,
                "entry_timestamp_ms": 1_772_450_060_000 + day * 86_400_000 + i * 60_000,
            }
            for h in horizons:
                resolved = i % 7 != 0
                gross = (feature * 0.02 + h * 0.001) if resolved else float("nan")
                row[f"return_{h}m_pct"] = gross
                row[f"return_{h}m_base_net_return_pct"] = (
                    gross - 0.5 if resolved else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_multi_horizon_collects_all_target_families_and_horizons():
    frame = _frame()
    holdouts = collect_multi_horizon_holdouts(frame)
    assert set(holdouts["target_kind"]) == {"gross", "base_net", "availability"}
    assert set(holdouts["horizon_min"]) == {1, 2, 5, 10, 15, 30, 60}
    assert "global_train_q_value_bh" in holdouts.columns


def test_cross_horizon_stability_requires_repeated_holdout_support():
    holdouts = collect_multi_horizon_holdouts(_frame())
    gross = summarize_cross_horizon_stability(holdouts, target_kind="gross")
    row = gross.loc[gross["feature"] == "previous_close"].iloc[0]
    assert row["horizons_tested"] == 7
    assert row["direction_consistency"] == 1.0
    assert row["positive_holdout_lift_horizons"] == 7
    assert row["robust_horizons"] >= 1


def test_multi_horizon_report_contains_alpha_net_execution_and_coverage():
    report, holdouts, stability = render_multi_horizon_report(_frame())
    assert "GROSS alpha cross-horizon stability" in report
    assert "BASE-NET cross-horizon stability" in report
    assert "EXECUTABILITY / availability cross-horizon stability" in report
    assert "Horizon coverage / exact-bar availability" in report
    assert not holdouts.empty
    assert set(stability["target_kind"]) == {"gross", "base_net", "availability"}
