from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from victory_trader.path_enrichment import ENRICHMENT_FEATURE_COLUMNS, enrich_path_features


ET = ZoneInfo("America/New_York")


class FakeStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def read(self, dataset_prefix, day: date, *, tickers=None):
        result = self.frame.copy()
        if tickers is not None:
            result = result.loc[result["ticker"].isin(tickers)]
        return result.reset_index(drop=True)


def ns(day: int, hour: int, minute: int) -> int:
    return int(datetime(2026, 3, day, hour, minute, tzinfo=ET).timestamp() * 1_000_000_000)


def test_enrichment_keeps_existing_results_and_adds_point_in_time_features():
    event_10_ms = ns(2, 9, 40) // 1_000_000
    event_20_ms = ns(2, 9, 45) // 1_000_000
    events = pd.DataFrame(
        {
            "trading_day": ["2026-03-02", "2026-03-02"],
            "ticker": ["TEST", "TEST"],
            "timestamp_ms": [event_10_ms, event_20_ms],
            "threshold_pct": [10.0, 20.0],
            "signal_price": [11.0, 12.0],
            "previous_close": [10.0, 10.0],
            "existing_result": [123.0, 456.0],
        }
    )

    test_rows = [
        {
            "ticker": "TEST",
            "volume": 500.0,
            "open": 10.2,
            "close": 10.4,
            "high": 10.5,
            "low": 10.1,
            "window_start": ns(2, 9, 0),
            "transactions": 5.0,
        }
    ]
    closes = [10.0 + 0.1 * i for i in range(15)] + [12.0]
    for i, close in enumerate(closes):
        test_rows.append(
            {
                "ticker": "TEST",
                "volume": 100.0,
                "open": close - 0.05,
                "close": close,
                "high": close + 0.10,
                "low": close - 0.10,
                "window_start": ns(2, 9, 30 + i),
                "transactions": 10.0,
            }
        )

    iwm_rows = []
    for i in range(16):
        close = 100.0 + 0.1 * i
        iwm_rows.append(
            {
                "ticker": "IWM",
                "volume": 10_000.0,
                "open": 100.0 if i == 0 else close - 0.05,
                "close": close,
                "high": close + 0.05,
                "low": close - 0.05,
                "window_start": ns(2, 9, 30 + i),
                "transactions": 100.0,
            }
        )

    result = enrich_path_features(events, FakeStore(pd.DataFrame(test_rows + iwm_rows)))

    assert result["existing_result"].tolist() == [123.0, 456.0]
    assert set(ENRICHMENT_FEATURE_COLUMNS).issubset(result.columns)
    assert result.loc[0, "minutes_since_10pct_cross"] == pytest.approx(0.0)
    assert pd.isna(result.loc[0, "minutes_since_prior_threshold_cross"])
    assert result.loc[1, "minutes_since_10pct_cross"] == pytest.approx(5.0)
    assert result.loc[1, "minutes_since_prior_threshold_cross"] == pytest.approx(5.0)
    assert result.loc[1, "threshold_overshoot_pct"] == pytest.approx(0.0)
    assert result.loc[1, "prior_15m_high_distance_pct"] == pytest.approx(
        (12.0 / 11.5 - 1.0) * 100.0
    )
    assert result.loc[1, "prior_15m_low_rebound_pct"] == pytest.approx(
        (12.0 / 9.9 - 1.0) * 100.0
    )
    assert result.loc[1, "trend_efficiency_15m"] == pytest.approx(1.0)
    assert result.loc[1, "max_drawdown_since_10pct_pct"] == pytest.approx(0.0)
    assert result.loc[1, "signal_bar_range_pct"] == pytest.approx((0.2 / 12.0) * 100.0)
    assert result.loc[1, "signal_bar_body_pct"] == pytest.approx((12.0 / 11.95 - 1.0) * 100.0)
    assert result.loc[1, "signal_close_location"] == pytest.approx(0.5)
    assert result.loc[1, "transactions_5m"] == pytest.approx(50.0)
    expected_dollar_volume = 100.0 * sum([11.1, 11.2, 11.3, 11.4, 12.0])
    assert result.loc[1, "dollar_volume_5m"] == pytest.approx(expected_dollar_volume)
    assert result.loc[1, "avg_trade_size_5m"] == pytest.approx(10.0)
    assert result.loc[1, "active_minute_fraction_15m"] == pytest.approx(1.0)
    assert result.loc[1, "regular_open_gap_pct"] == pytest.approx(-0.5)
    assert result.loc[1, "premarket_high_return_pct"] == pytest.approx(5.0)
    assert result.loc[1, "premarket_high_distance_pct"] == pytest.approx(
        (12.0 / 10.5 - 1.0) * 100.0
    )
    assert result.loc[1, "runners_10pct_so_far"] == pytest.approx(1.0)
    assert result.loc[1, "runners_10pct_last_30m"] == pytest.approx(1.0)
    assert result.loc[1, "iwm_return_since_open_pct"] == pytest.approx(1.5)
    assert result.loc[1, "iwm_return_15m_pct"] == pytest.approx(1.5)
    assert result.loc[1, "iwm_volatility_15m_pct"] > 0
