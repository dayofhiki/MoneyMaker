from datetime import date

import pytest

from victory_trader.market_dataset import build_market_event_dataset, select_candidates


def grouped(rows):
    return {"results": rows}


def test_select_candidates_filters_price_and_move():
    previous = grouped(
        [
            {"T": "AAA", "c": 2.00},
            {"T": "BBB", "c": 25.00},
            {"T": "CCC", "c": 4.00},
        ]
    )
    target = grouped(
        [
            {"T": "AAA", "h": 2.60, "c": 2.40, "v": 1_000_000, "vw": 2.30},
            {"T": "BBB", "h": 40.00, "c": 35.00, "v": 1_000_000, "vw": 34.00},
            {"T": "CCC", "h": 4.50, "c": 4.20, "v": 1_000_000, "vw": 4.20},
        ]
    )

    result = select_candidates(previous, target, min_high_return_pct=20.0)

    assert [item.ticker for item in result] == ["AAA"]
    assert result[0].high_return_pct == pytest.approx(30.0)
    assert result[0].day_dollar_volume == pytest.approx(2_300_000.0)


class FakeClient:
    def __init__(self, split=False, security_type="CS", exchange="XNAS"):
        self.split = split
        self.security_type = security_type
        self.exchange = exchange
        self.market = {
            "2026-09-14": grouped([{"T": "AAA", "c": 2.00}]),
            "2026-09-15": grouped(
                [{"T": "AAA", "h": 3.20, "c": 2.80, "v": 1_000_000, "vw": 2.60}]
            ),
        }

    def _get(self, path, params=None):
        day = path.rsplit("/", 1)[-1]
        return self.market.get(day, grouped([]))

    def splits_on(self, day):
        rows = [{"ticker": "AAA", "execution_date": day.isoformat()}] if self.split else []
        return {"results": rows}

    def ticker_details(self, ticker, day=None):
        return {
            "results": {
                "ticker": ticker,
                "name": "AAA Corp",
                "type": self.security_type,
                "market": "stocks",
                "locale": "us",
                "primary_exchange": self.exchange,
                "active": True,
                "market_cap": 50_000_000,
                "share_class_shares_outstanding": 20_000_000,
            }
        }

    def minute_bars(self, ticker, day):
        assert ticker == "AAA"
        return {
            "results": [
                {"t": 0, "o": 2.00, "h": 2.05, "l": 1.95, "c": 2.00, "v": 1000},
                {"t": 60_000, "o": 2.00, "h": 2.45, "l": 2.00, "c": 2.40, "v": 5000},
                {"t": 120_000, "o": 2.40, "h": 2.65, "l": 2.35, "c": 2.60, "v": 6000},
                {"t": 180_000, "o": 2.60, "h": 3.05, "l": 2.55, "c": 3.00, "v": 8000},
                {"t": 240_000, "o": 3.00, "h": 3.20, "l": 2.90, "c": 3.10, "v": 9000},
            ]
        }


def test_build_market_event_dataset_end_to_end_without_network():
    result = build_market_event_dataset(
        FakeClient(),
        date(2026, 9, 15),
        thresholds_pct=(20, 30, 50),
        horizons=(1, 2),
        request_interval_seconds=0,
    )

    assert not result.empty
    assert list(result["threshold_pct"]) == [20.0, 30.0, 50.0]
    assert set(result["ticker"]) == {"AAA"}
    assert set(result["security_type"]) == {"CS"}
    assert set(result["primary_exchange"]) == {"XNAS"}
    assert set(result["market_cap"]) == {50_000_000.0}
    assert "return_1m_pct" in result.columns


def test_split_day_is_excluded():
    result = build_market_event_dataset(
        FakeClient(split=True),
        date(2026, 9, 15),
        request_interval_seconds=0,
    )
    assert result.empty


def test_non_common_stock_is_excluded():
    result = build_market_event_dataset(
        FakeClient(security_type="ETF"),
        date(2026, 9, 15),
        request_interval_seconds=0,
    )
    assert result.empty
