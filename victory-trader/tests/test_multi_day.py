from datetime import date

import pandas as pd

import victory_trader.multi_day as multi_day


def test_daterange_is_inclusive():
    assert list(multi_day.daterange(date(2026, 9, 14), date(2026, 9, 16))) == [
        date(2026, 9, 14),
        date(2026, 9, 15),
        date(2026, 9, 16),
    ]


def test_multi_day_accumulates_and_skips(monkeypatch):
    def fake_builder(client, day, **kwargs):
        if day == date(2026, 9, 14):
            return pd.DataFrame([
                {"trading_day": "2026-09-14", "ticker": "AAA", "timestamp_ms": 1, "threshold_pct": 20.0}
            ])
        if day == date(2026, 9, 15):
            raise ValueError("holiday-like missing data")
        return pd.DataFrame([
            {"trading_day": "2026-09-16", "ticker": "BBB", "timestamp_ms": 2, "threshold_pct": 30.0}
        ])

    monkeypatch.setattr(multi_day, "build_market_event_dataset", fake_builder)
    result, skipped = multi_day.build_multi_day_dataset(
        object(), date(2026, 9, 14), date(2026, 9, 16)
    )

    assert list(result["trading_day"]) == ["2026-09-14", "2026-09-16"]
    assert list(result["ticker"]) == ["AAA", "BBB"]
    assert skipped and skipped[0][0] == date(2026, 9, 15)
