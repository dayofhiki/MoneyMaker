from datetime import date

from victory_trader.flatfiles import FlatFileObject
from victory_trader.high_resolution_access_probe import (
    QUOTES_FLATFILE_PREFIX,
    TRADES_FLATFILE_PREFIX,
    run_probe,
)


class FakeRestClient:
    def second_bars_range(self, ticker, start, end, *, adjusted=False):
        assert ticker == "AAPL"
        assert start == end == date(2026, 1, 2)
        assert adjusted is False
        return {"results": [{"t": 1}, {"t": 2}]}

    def trades(
        self,
        ticker,
        *,
        timestamp_gte=None,
        timestamp_lte=None,
        limit=50_000,
        order="asc",
    ):
        assert ticker == "AAPL"
        assert timestamp_gte is not None
        assert timestamp_lte is not None
        assert timestamp_lte > timestamp_gte
        assert limit == 50_000
        assert order == "asc"
        return [{"sip_timestamp": timestamp_gte + 1}]


class FakeFlatFileClient:
    def head(self, dataset_prefix, day):
        assert day == date(2026, 1, 2)
        assert dataset_prefix in {
            TRADES_FLATFILE_PREFIX,
            QUOTES_FLATFILE_PREFIX,
        }
        return FlatFileObject(
            dataset=dataset_prefix,
            day=day,
            key=f"{dataset_prefix}/2026/01/2026-01-02.csv.gz",
            size_bytes=123,
        )


def test_high_resolution_probe_reports_available_sources():
    result = run_probe(
        FakeRestClient(),
        FakeFlatFileClient(),
        ticker="aapl",
        day=date(2026, 1, 2),
    )

    assert result["second_bars_rest"]["status"] == "AVAILABLE"
    assert result["second_bars_rest"]["rows"] == 2
    assert result["trades_rest_10s"]["status"] == "AVAILABLE"
    assert result["trades_rest_10s"]["rows"] == 1
    assert result["trades_flatfile"]["status"] == "AVAILABLE"
    assert result["quotes_flatfile"]["status"] == "AVAILABLE"
