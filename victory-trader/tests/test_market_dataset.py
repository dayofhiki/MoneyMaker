from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from victory_trader.market_dataset import (
    Candidate,
    build_market_event_dataset,
    limit_candidates_for_debug,
    select_candidates,
)


ET = ZoneInfo("America/New_York")


def ts(hour: int, minute: int) -> int:
    return int(datetime(2026, 9, 15, hour, minute, tzinfo=ET).timestamp() * 1000)


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


def test_completed_day_liquidity_filter_is_forbidden():
    with pytest.raises(ValueError, match="completed-day dollar-volume"):
        select_candidates(
            grouped([{"T": "AAA", "c": 2.0}]),
            grouped([{"T": "AAA", "h": 2.5, "c": 2.4, "v": 1000}]),
            min_day_dollar_volume=1.0,
        )


def _candidate(ticker: str, high_return_pct: float) -> Candidate:
    return Candidate(
        ticker=ticker,
        previous_close=2.0,
        day_high=2.0 * (1.0 + high_return_pct / 100.0),
        day_close=2.1,
        day_volume=1_000_000,
        day_vwap=2.05,
        high_return_pct=high_return_pct,
        day_dollar_volume=2_050_000,
    )


def test_debug_candidate_limit_does_not_depend_on_eventual_return_rank():
    trading_day = date(2026, 9, 15)
    first = [_candidate("AAA", 25.0), _candidate("BBB", 500.0), _candidate("CCC", 40.0), _candidate("DDD", 200.0)]
    second = [_candidate("AAA", 900.0), _candidate("BBB", 21.0), _candidate("CCC", 700.0), _candidate("DDD", 22.0)]

    chosen_first = limit_candidates_for_debug(first, day=trading_day, max_candidates=2)
    chosen_second = limit_candidates_for_debug(second, day=trading_day, max_candidates=2)

    assert [item.ticker for item in chosen_first] == [item.ticker for item in chosen_second]
    assert len(chosen_first) == 2


def test_debug_candidate_limit_requires_positive_count():
    with pytest.raises(ValueError):
        limit_candidates_for_debug([_candidate("AAA", 25.0)], day=date(2026, 9, 15), max_candidates=0)


class FakeClient:
    def __init__(self, split=False, security_type="CS", exchange="XNAS"):
        self.split = split
        self.security_type = security_type
        self.exchange = exchange
        self.market = {
            "2026-09-14": grouped([{"T": "AAA", "c": 2.00}]),
            "2026-09-15": grouped([{"T": "AAA", "h": 3.20, "c": 2.80, "v": 1_000_000, "vw": 2.60}]),
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
                {"t": ts(9, 30), "o": 2.00, "h": 2.05, "l": 1.95, "c": 2.00, "v": 1000},
                {"t": ts(9, 31), "o": 2.00, "h": 2.45, "l": 2.00, "c": 2.40, "v": 5000},
                {"t": ts(9, 32), "o": 2.40, "h": 2.65, "l": 2.35, "c": 2.60, "v": 6000},
                {"t": ts(9, 33), "o": 2.60, "h": 3.05, "l": 2.55, "c": 3.00, "v": 8000},
                {"t": ts(9, 34), "o": 3.00, "h": 3.20, "l": 2.90, "c": 3.10, "v": 9000},
                {"t": ts(9, 35), "o": 3.10, "h": 3.15, "l": 3.00, "c": 3.05, "v": 7000},
            ]
        }


def test_build_market_event_dataset_end_to_end_without_network():
    result = build_market_event_dataset(
        FakeClient(),
        date(2026, 9, 15),
        thresholds_pct=(20, 30, 50),
        min_high_return_pct=20,
        horizons=(1, 2),
        request_interval_seconds=0,
    )

    assert not result.empty
    assert list(result["threshold_pct"]) == [20.0, 30.0, 50.0]
    assert set(result["ticker"]) == {"AAA"}
    assert set(result["security_type"]) == {"CS"}
    assert set(result["primary_exchange"]) == {"XNAS"}
    assert "market_cap" not in result.columns
    assert "day_high" not in result.columns
    assert set(result["source_prices_adjusted"]) == {False}
    assert set(result["discovery_high_return_threshold_pct"]) == {20.0}
    assert set(result["event_session_scope"]) == {"regular"}
    assert set(result["regular_entry_required"]) == {True}
    assert result["debug_candidate_limit"].isna().all()
    assert "return_1m_pct" in result.columns
    assert "delay1_return_1m_pct" in result.columns


def test_discovery_threshold_cannot_exceed_lowest_studied_event():
    with pytest.raises(ValueError, match="future-selected"):
        build_market_event_dataset(
            FakeClient(),
            date(2026, 9, 15),
            thresholds_pct=(10, 20),
            min_high_return_pct=20,
            request_interval_seconds=0,
        )


def test_split_day_is_excluded():
    result = build_market_event_dataset(FakeClient(split=True), date(2026, 9, 15), request_interval_seconds=0)
    assert result.empty


def test_non_common_stock_is_excluded():
    result = build_market_event_dataset(FakeClient(security_type="ETF"), date(2026, 9, 15), request_interval_seconds=0)
    assert result.empty
