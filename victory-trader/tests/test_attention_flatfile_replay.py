from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from victory_trader.attention_flatfile_replay import (
    build_flatfile_scan_day,
    run_flatfile_attention_replay,
)
from victory_trader.attention_runtime import AttentionConfig
from victory_trader.flatfiles import FlatFileStoreStats

ET = ZoneInfo("America/New_York")


class FakeStore:
    def __init__(self, previous: pd.DataFrame | None = None) -> None:
        self.day_calls: list[date] = []
        self.minute_calls: list[date] = []
        self.stats = FlatFileStoreStats()
        self.previous = previous

    def day_aggregates(self, day: date) -> pd.DataFrame:
        self.day_calls.append(day)
        if self.previous is not None:
            return self.previous.copy()
        return pd.DataFrame({
            "ticker": ["AAA", "ETF", "SPLT", "PRICEY"],
            "close": [10.0, 10.0, 10.0, 25.0],
        })

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

    def daily_bars(self, ticker, start, end, *, adjusted=False):
        del ticker, start, end, adjusted
        raise AssertionError("daily REST fallback was not expected")


class ResolvingRestClient(FakeRestClient):
    def __init__(self, close: float) -> None:
        self.close = close

    def daily_bars(self, ticker, start, end, *, adjusted=False):
        assert ticker == "AAA"
        assert start == end == date(2025, 12, 31)
        assert adjusted is False
        return {"results": [{"c": self.close}]}


class DuplicateMinuteStore(FakeStore):
    def minute_aggregates(self, day: date) -> pd.DataFrame:
        frame = super().minute_aggregates(day)
        duplicate = frame.loc[
            frame["ticker"].eq("AAA")
            & frame["t"].eq(frame.loc[frame["ticker"].eq("AAA"), "t"].iloc[0])
        ].copy()
        duplicate[["o", "h", "l", "c"]] = 99.0
        return pd.concat([frame, duplicate], ignore_index=True)


class MinuteResolvingRestClient(FakeRestClient):
    def minute_bars(self, ticker, day, *, adjusted=False):
        assert ticker == "AAA"
        assert day == date(2026, 1, 2)
        assert adjusted is False
        start = datetime.combine(day, time(9, 30), tzinfo=ET)
        return {
            "results": [
                {
                    "t": int((start.timestamp() + minute * 60) * 1000),
                    "o": close,
                    "h": close,
                    "l": close,
                    "c": close,
                    "v": 1_000,
                    "n": 100,
                }
                for minute, close in enumerate([10.0, 10.6, 11.1])
            ]
        }


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


def test_flatfile_adapter_collapses_only_identical_prior_close_duplicates():
    previous = pd.DataFrame({
        "ticker": ["AAA", " aaa ", "PRICEY"],
        "close": [10.0, 10.0, 25.0],
    })

    scan, summary = build_flatfile_scan_day(
        FakeStore(previous),
        FakeRestClient(),
        date(2026, 1, 2),
    )

    assert set(scan["ticker"]) == {"AAA"}
    assert summary["duplicate_prior_rows_collapsed"] == 1


def test_flatfile_adapter_resolves_conflict_with_exact_date_rest_bar():
    previous = pd.DataFrame({
        "ticker": ["AAA", "AAA", "PRICEY"],
        "close": [10.0, 10.01, 25.0],
    })

    scan, summary = build_flatfile_scan_day(
        FakeStore(previous),
        ResolvingRestClient(10.01),
        date(2026, 1, 2),
    )

    assert set(scan["previous_close"]) == {10.01}
    assert summary["duplicate_prior_rows_collapsed"] == 1
    assert summary["conflicting_prior_tickers_resolved"] == 1


def test_flatfile_adapter_rejects_conflict_without_unique_rest_match():
    previous = pd.DataFrame({
        "ticker": ["AAA", "AAA", "PRICEY"],
        "close": [10.0, 10.01, 25.0],
    })

    with pytest.raises(ValueError, match="could not resolve conflicting"):
        build_flatfile_scan_day(
            FakeStore(previous),
            ResolvingRestClient(12.0),
            date(2026, 1, 2),
        )


def test_flatfile_adapter_resolves_conflicting_minute_series_with_rest():
    scan, summary = build_flatfile_scan_day(
        DuplicateMinuteStore(),
        MinuteResolvingRestClient(),
        date(2026, 1, 2),
    )

    assert len(scan) == 3
    assert scan["c"].tolist() == [10.0, 10.6, 11.1]
    assert summary["duplicate_minute_rows_collapsed"] == 1
    assert summary["conflicting_minute_tickers_resolved"] == 1
