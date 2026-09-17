from datetime import date

import pandas as pd
import pytest

from victory_trader.path_enrichment import enrich_path_features


class FakeStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def read(self, dataset_prefix, day: date, *, tickers=None):
        result = self.frame.copy()
        if tickers is not None:
            result = result.loc[result["ticker"].isin(tickers)]
        return result.reset_index(drop=True)


def test_enrichment_keeps_existing_columns_and_adds_path_features():
    minute_ns = 60_000_000_000
    base_ns = 1_700_000_000_000_000_000
    base_ms = base_ns // 1_000_000
    events = pd.DataFrame(
        {
            "trading_day": ["2026-03-02", "2026-03-02"],
            "ticker": ["TEST", "TEST"],
            "timestamp_ms": [base_ms + 10 * 60_000, base_ms + 14 * 60_000],
            "threshold_pct": [10.0, 20.0],
            "signal_price": [11.0, 12.0],
            "existing_result": [123.0, 456.0],
        }
    )
    bars = pd.DataFrame(
        {
            "ticker": ["TEST"] * 15,
            "volume": [100.0] * 15,
            "open": [10.0] * 15,
            "close": [10.0] * 15,
            "high": [10.0 + i * 0.1 for i in range(15)],
            "low": [9.0 + i * 0.05 for i in range(15)],
            "window_start": [base_ns + i * minute_ns for i in range(15)],
            "transactions": [1.0] * 15,
        }
    )

    result = enrich_path_features(events, FakeStore(bars))

    assert result["existing_result"].tolist() == [123.0, 456.0]
    assert result.loc[0, "minutes_since_10pct_cross"] == pytest.approx(0.0)
    assert pd.isna(result.loc[0, "minutes_since_prior_threshold_cross"])
    assert result.loc[1, "minutes_since_10pct_cross"] == pytest.approx(4.0)
    assert result.loc[1, "minutes_since_prior_threshold_cross"] == pytest.approx(4.0)
    assert result.loc[1, "prior_15m_high_distance_pct"] == pytest.approx(
        (12.0 / 11.3 - 1.0) * 100.0
    )
    assert result.loc[1, "prior_15m_low_rebound_pct"] == pytest.approx(
        (12.0 / 9.0 - 1.0) * 100.0
    )
