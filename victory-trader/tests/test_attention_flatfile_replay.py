from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from victory_trader.attention_flatfile_replay import (
    build_flatfile_scan_day,
    run_flatfile_attention_replay,
)
from victory_trader.attention_runtime import AttentionConfig
from victory_trader.flatfiles import FlatFileStoreStats

ET = ZoneInfo("America/New_York")


class FakeStore:
    def __init__(self) -> None:
        self.day_calls: list[date] = []
        self.minute_calls: list[date] = []
        self.stats = FlatFileStoreStats()

    def day_aggregates(self, day: date) -> pd.DataFrame:
        self.day_calls.append(day)
        return pd.DataFrame(
            {
                "ticker": ["AAA", "ETF", "SPLT", "PRICEY"],
                "close": [10.0, 10.0, 10.0, 25.0],
            }
        )

    def minute_aggregates(self, day: date) -> pd.DataFrame:
        self.minute_calls.append(day)
        start = datetime.combine(day, time(9, 30), tzinfo=ET)
        rows = []
        closes = {
            "AAA": [10.0, 10.6, 11.1],
            "ETF": [10.0, 11.0, 12.0],
            "SPLT": [10.0, 11.0, 12.0],
            "PRICEY": [25.0, 26.0, 27.0],
        }
        for minute in range(3):
            timestamp = int((start.timestamp() + minute * 60) * 1000)
            for ticker, values in closes.items():
                rows.append(
                    {
                        "ticker": ticker,
                        "t": timestamp,
                        "o": values[minute],
                        "h": values[minute],
                        "l": values[minute],
                        "c": values[minute],
                        "v": 1_000,
                        "n": 100,
                    }
                )
        # Extended-hours print must not enter the regular-session scan.
        rows.append(
            {
                "ticker": "AAA",
                "t": int(datetime.combine(day, time(8, 0), tzinfo=ET).timestamp() * 1000),
                "o": 20.0,
                "h": 20.0,
                "l": 20.0,
                "c": 20.0,
                "v": 1_000,
                "n": 100,
            }
        )
        return pd.DataFrame(rows)


class FakeRestClient:
    def reference_tickers(self, day, **kwargs):
        del day, kwargs
        return [
            {
                "ticker": "AAA",
                "type": "CS",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "XNAS",
            },
            {
                "ticker": "ETF",
                "type": "ETF",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "XNAS",
            },
            {
                "ticker": "SPLT",
                "type": "CS",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "XNYS",
            },
            {
                "ticker": "PRICEY",
                "type": "CS",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "XNAS",
            },
        ]

    def splits_on(self, day):
        del day
        return {"results": [{"ticker": "SPLT"}]}


def _config() -> AttentionConfig:
    return AttentionConfig(
        watch_enter_score=0.50,
        watch_exit_score=0.30,
        hot_enter_score=0.80,
        hot_exit_score=0.60,
        max_watch=2,
        max_hot=1,
        drop_after_missed_batches=2,
    )


def test_flatfile_adapter_uses_prior_day_and_point_in_time_universe():
    store = FakeStore()
    day = date(2026, 1, 2)

    scan, summary = build_flatfile_scan_day(store, FakeRestClient(), day)

    assert store.day_calls == [date(2025, 12, 31)]
    assert store.minute_calls == [day]
    assert set(scan["ticker"]) == {"AAA"}
    assert len(scan) == 3
    assert scan["runner_cross_now"].sum() == 1
    assert summary["same_day_split_exclusions"] == 1
    assert summary["scan_symbols"] == 1


def test_flatfile_range_runs_replay_and_reports_provenance():
    store = FakeStore()
    day = date(2026, 1, 2)

    scan, trace, summary = run_flatfile_attention_replay(
        store,
        FakeRestClient(),
        day,
        day,
        config=_config(),
    )

    assert not scan.empty
    assert not trace.empty
    assert summary["start"] == "2026-01-02"
    assert summary["end"] == "2026-01-02"
    assert summary["attention_config"]["max_hot"] == 1
    assert summary["replay"]["runner_episodes"] == 1
    assert summary["days"][0]["previous_trading_day"] == "2025-12-31"
