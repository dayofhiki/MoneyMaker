from datetime import date

import pandas as pd
import pytest

import victory_trader.multi_day as multi_day


def test_daterange_is_inclusive():
    assert list(multi_day.daterange(date(2026, 9, 14), date(2026, 9, 16))) == [
        date(2026, 9, 14),
        date(2026, 9, 15),
        date(2026, 9, 16),
    ]


def test_multi_day_skips_only_expected_closed_market_days(monkeypatch):
    def fake_builder(client, day, **kwargs):
        if day == date(2026, 9, 14):
            return pd.DataFrame([
                {"trading_day": "2026-09-14", "ticker": "AAA", "timestamp_ms": 1, "threshold_pct": 20.0}
            ])
        if day == date(2026, 9, 15):
            raise ValueError("No grouped market data returned for 2026-09-15")
        return pd.DataFrame([
            {"trading_day": "2026-09-16", "ticker": "BBB", "timestamp_ms": 2, "threshold_pct": 30.0}
        ])

    monkeypatch.setattr(multi_day, "build_market_event_dataset", fake_builder)
    result, skipped = multi_day.build_multi_day_dataset(
        object(), date(2026, 9, 14), date(2026, 9, 16)
    )

    assert list(result["trading_day"]) == ["2026-09-14", "2026-09-16"]
    assert list(result["ticker"]) == ["AAA", "BBB"]
    assert skipped == [
        (date(2026, 9, 15), "closed_market: No grouped market data returned for 2026-09-15")
    ]


def test_multi_day_fails_fast_on_unexpected_api_or_code_error(monkeypatch):
    def fake_builder(client, day, **kwargs):
        raise RuntimeError("429 rate limit or parser bug")

    monkeypatch.setattr(multi_day, "build_market_event_dataset", fake_builder)

    with pytest.raises(RuntimeError, match="429 rate limit or parser bug"):
        multi_day.build_multi_day_dataset(
            object(), date(2026, 9, 14), date(2026, 9, 14)
        )


def test_multi_day_can_continue_unexpected_errors_only_when_explicitly_requested(monkeypatch):
    def fake_builder(client, day, **kwargs):
        raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(multi_day, "build_market_event_dataset", fake_builder)
    result, skipped = multi_day.build_multi_day_dataset(
        object(),
        date(2026, 9, 14),
        date(2026, 9, 14),
        continue_on_error=True,
    )

    assert result.empty
    assert skipped == [(date(2026, 9, 14), "ERROR RuntimeError: diagnostic failure")]
