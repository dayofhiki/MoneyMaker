import math

import pandas as pd

from victory_trader.multiday_first_passage_directional_edge import (
    HistoryRecord,
    add_multiday_features,
    coverage,
)


class FakeHistory:
    def record(self, day, ticker):
        return HistoryRecord(
            query_success=True,
            split_query_success=True,
            latest_prior_split_day=None,
            max_history_day="2026-05-08",
            sessions=20,
            values={
                "hist_log_avg_volume_5d": math.log1p(1000.0),
                "hist_log_avg_volume_20d": math.log1p(800.0),
                "hist_volume_ratio_5d_20d": 1.25,
                "hist_log_avg_dollar_volume_5d": math.log1p(10_000.0),
                "hist_log_avg_dollar_volume_20d": math.log1p(8_000.0),
                "hist_realized_vol_5d_pct": 5.0,
                "hist_realized_vol_20d_pct": 4.0,
                "hist_avg_intraday_range_5d_pct": 7.0,
                "hist_avg_intraday_range_20d_pct": 6.0,
                "hist_runner_days_20d": 2.0,
                "hist_big_down_days_20d": 1.0,
                "hist_prior_close_vs_20d_high_pct": -10.0,
                "hist_prior_close_vs_20d_low_pct": 50.0,
                "hist_days_since_last_runner": 3.0,
                "_hist_high20": 12.0,
                "_hist_low20": 6.0,
            },
        )


def test_multiday_features_use_causal_state_against_prior_range():
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "log_current_close": math.log(10.0),
            }
        ]
    )
    result = add_multiday_features(frame, FakeHistory())
    row = result.iloc[0]
    assert math.isclose(
        row["state_vs_20d_high_pct"],
        (10.0 / 12.0 - 1.0) * 100.0,
    )
    assert math.isclose(
        row["state_vs_20d_low_pct"],
        (10.0 / 6.0 - 1.0) * 100.0,
    )
    assert row["hist_runner_days_20d"] == 2.0
    cov = coverage(result)
    assert cov["request_success"] == 1.0
    assert cov["support_ge_5_sessions"] == 1.0
    assert cov["support_ge_20_sessions"] == 1.0
    assert cov["strict_prior_max_history_day"] is True


def test_short_history_is_retained_as_missing_context():
    class ShortHistory:
        def record(self, day, ticker):
            return HistoryRecord(
                query_success=True,
                split_query_success=True,
                latest_prior_split_day=None,
                max_history_day="2026-05-08",
                sessions=3,
                values={},
            )

    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-05-11",
                "ticker": "TEST",
                "log_current_close": math.log(10.0),
            }
        ]
    )
    result = add_multiday_features(frame, ShortHistory())
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["hist_runner_days_20d"])
    cov = coverage(result)
    assert cov["support_ge_5_sessions"] == 0.0
    assert cov["strict_prior_max_history_day"] is True
