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
    def __init__(self):
        self.market = {
            "2026-09-14": grouped([{"T": "AAA", "c": 2.00}]),
            "2026-09-15": grouped(
                [{"T": "AAA", "h": 3.20, "c": 2.80, "v": 1_000_000, "vw": 2.60}]
            ),
        }

    def _get(self, path, params=None):
        day = path.rsplit("/", 1)[-1]
        return self.market.get(day, grouped([]))

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
    assert set(result["trading_day"]) == {"2026-09-15"}
    assert set(result["previous_trading_day"]) == {"2026-09-14"}
    assert "return_1m_pct" in result.columns
    assert "day_high_return_pct" in result.columns
