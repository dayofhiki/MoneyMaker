from datetime import date, datetime, timezone

import pandas as pd

from victory_trader.flatfile_dataset import build_flatfile_market_event_dataset
from victory_trader.market_dataset import DayBuildStats


class FakeRestClient:
    def splits_on(self, day):
        return {"results": []}

    def reference_tickers(self, day, *, market, security_type, active):
        assert market == "stocks"
        assert security_type == "CS"
        assert active is True
        return [
            {
                "ticker": "TEST",
                "name": "Test Co",
                "type": "CS",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "XNAS",
                "active": True,
            }
        ]


class FakeStore:
    def grouped_daily_payload(self, day):
        if day == date(2026, 3, 31):
            return {"results": [{"T": "TEST", "c": 1.0, "h": 1.0, "v": 1000}]}
        if day == date(2026, 4, 1):
            return {"results": [{"T": "TEST", "c": 1.18, "h": 1.25, "v": 100000}]}
        raise AssertionError(f"unexpected grouped day {day}")

    def minute_aggregates(self, day, *, tickers=None):
        if day != date(2026, 4, 1):
            return pd.DataFrame(columns=["ticker", "t", "o", "h", "l", "c", "v", "n"])
        assert tickers == {"TEST"}
        start = datetime(2026, 4, 1, 13, 30, tzinfo=timezone.utc)
        rows = []
        prices = [1.05, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16]
        for i, close in enumerate(prices):
            ts_ms = int(start.timestamp() * 1000) + i * 60_000
            rows.append(
                {
                    "ticker": "TEST",
                    "t": ts_ms,
                    "o": close - 0.005,
                    "h": close + 0.01,
                    "l": close - 0.01,
                    "c": close,
                    "v": 10000 + i * 1000,
                    "n": 100,
                }
            )
        return pd.DataFrame(rows)


def test_flatfile_builder_preserves_point_in_time_event_semantics():
    stats = DayBuildStats()
    frame = build_flatfile_market_event_dataset(
        FakeRestClient(),
        FakeStore(),
        date(2026, 4, 1),
        thresholds_pct=(10,),
        horizons=(1, 2, 5),
        historical_context_days=0,
        build_stats=stats,
    )

    assert not frame.empty
    row = frame.iloc[0]
    assert row["ticker"] == "TEST"
    assert row["threshold_pct"] == 10.0
    assert row["entry_model"] == "next_minute_open"
    assert row["market_data_backend"] == "massive_flatfiles"
    assert row["dataset_schema_version"] == "0.3"
    assert row["bar_vwap_source"] == "volume_weighted_bar_close_proxy"
    assert not bool(row["source_prices_adjusted"])
    assert stats.discovered_candidates == 1
    assert stats.event_tickers == 1
