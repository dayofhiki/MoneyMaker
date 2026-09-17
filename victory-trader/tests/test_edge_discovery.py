import pandas as pd

from victory_trader.edge_discovery import (
    summarize_chronological_feature_holdout,
    summarize_univariate_discovery,
)


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for day in range(1, 13):
        for threshold in (10.0, 20.0):
            for rank in range(10):
                feature = float(rank)
                day_effect = (day % 3 - 1) * 0.2
                threshold_effect = -0.5 if threshold == 20.0 else 0.0
                target = (feature - 4.5) * 0.5 + day_effect + threshold_effect
                rows.append(
                    {
                        "trading_day": f"2026-03-{day:02d}",
                        "threshold_pct": threshold,
                        "signal_feature": feature,
                        "return_5m_base_net_return_pct": target,
                    }
                )
    return pd.DataFrame(rows)


def test_univariate_discovery_controls_day_and_threshold():
    result = summarize_univariate_discovery(
        synthetic_frame(),
        features=("signal_feature",),
        min_n=20,
        min_days=4,
    )

    row = result.iloc[0]
    assert row["direction"] == "higher"
    assert row["day_threshold_spearman"] > 0.95
    assert row["q_value_bh"] < 0.001
    assert row["favorable_adjusted_lift_pct"] > 0
    assert row["raw_favorable_minus_adverse_pct"] > 0


def test_chronological_holdout_learns_cut_only_from_early_dates():
    result = summarize_chronological_feature_holdout(
        synthetic_frame(),
        features=("signal_feature",),
        train_fraction=0.70,
        min_train_n=20,
        min_test_n=5,
    )

    row = result.iloc[0]
    assert row["train_direction"] == "higher"
    assert row["train_day_threshold_spearman"] > 0.95
    assert row["test_selected_n"] > 0
    assert row["test_selected_days"] >= 2
    assert row["test_adjusted_lift_pct"] > 0
    assert row["enough_sample"] == 1
