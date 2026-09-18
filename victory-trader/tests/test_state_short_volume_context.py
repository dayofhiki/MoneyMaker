import numpy as np
import pandas as pd

from victory_trader.state_short_volume_context import (
    enrich_relative_short_volume_context,
    success_check,
)
from victory_trader.state_short_volume_enrichment import SHORT_VOLUME_FEATURES


def _base_row(day, t, ticker, latest, mean5, short_vs5, total_vs5):
    row = {
        "trading_day": day,
        "t": t,
        "ticker": ticker,
        "c": 5.0,
        "active_minute_fraction_15m": 1.0,
        "short_ratio_latest_prior": latest,
        "short_ratio_mean5_prior": mean5,
        "short_ratio_mean20_prior": mean5,
        "short_ratio_latest_minus_mean5": latest - mean5,
        "short_ratio_mean5_minus_mean20": 0.0,
        "short_volume_latest_vs_mean5": short_vs5,
        "finra_total_volume_latest_vs_mean5": total_vs5,
        "short_volume_latest_age_days": 1.0,
    }
    assert set(SHORT_VOLUME_FEATURES).issubset(row)
    return row


def test_context_is_exact_timestamp_cross_section_only():
    frame = pd.DataFrame(
        [
            _base_row("2026-01-02", 100, "AAA", 20.0, 30.0, 1.0, 1.0),
            _base_row("2026-01-02", 100, "BBB", 40.0, 50.0, 2.0, 3.0),
            _base_row("2026-01-02", 200, "CCC", 90.0, 90.0, 9.0, 9.0),
        ]
    )
    enriched = enrich_relative_short_volume_context(frame)

    aaa = enriched.loc[enriched["ticker"].eq("AAA")].iloc[0]
    bbb = enriched.loc[enriched["ticker"].eq("BBB")].iloc[0]
    ccc = enriched.loc[enriched["ticker"].eq("CCC")].iloc[0]

    assert aaa["runner_other_mean_short_ratio_latest_prior"] == 40.0
    assert aaa["runner_short_ratio_latest_minus_other_mean"] == -20.0
    assert aaa["runner_short_ratio_latest_percentile"] == 0.5
    assert bbb["runner_other_mean_short_ratio_latest_prior"] == 20.0
    assert bbb["runner_short_ratio_latest_percentile"] == 1.0

    assert ccc["runner_context_count"] == 1.0
    assert np.isnan(ccc["runner_other_mean_short_ratio_latest_prior"])
    assert np.isnan(ccc["runner_short_ratio_latest_percentile"])


def test_context_handles_missing_peer_value_without_using_future_row():
    frame = pd.DataFrame(
        [
            _base_row("2026-01-02", 100, "AAA", 20.0, 30.0, 1.0, 1.0),
            _base_row("2026-01-02", 100, "BBB", np.nan, 50.0, 2.0, 3.0),
            _base_row("2026-01-02", 200, "BBB", 40.0, 50.0, 2.0, 3.0),
        ]
    )
    enriched = enrich_relative_short_volume_context(frame)
    aaa = enriched.loc[
        enriched["ticker"].eq("AAA") & enriched["t"].eq(100)
    ].iloc[0]

    assert np.isnan(aaa["runner_other_mean_short_ratio_latest_prior"])
    assert np.isnan(aaa["runner_short_ratio_latest_percentile"])


def test_context_probe_success_rules():
    coverage = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02", "2026-03"],
            "runner_short_ratio_latest_percentile_coverage": [0.7, 0.8, 0.9],
            "runner_short_ratio_latest_minus_other_mean_coverage": [0.7, 0.8, 0.9],
            "median_runner_context_count": [2.0, 3.0, 2.0],
        }
    )
    checks = success_check(coverage)
    assert checks["all_pass"] is True

    coverage.loc[1, "runner_short_ratio_latest_percentile_coverage"] = 0.59
    checks = success_check(coverage)
    assert checks["all_pass"] is False
