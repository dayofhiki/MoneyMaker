from datetime import date

import pandas as pd
import pytest

from victory_trader.short_volume_supply_probe import (
    _event_metrics,
    _safe_prior_short_volume,
    success_check,
)


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def short_volume(self, ticker, *, date_gte, date_lte, limit=50_000):
        self.calls.append(
            {
                "ticker": ticker,
                "date_gte": date_gte,
                "date_lte": date_lte,
                "limit": limit,
            }
        )
        return list(self.rows)


def test_probe_requests_only_days_before_event():
    client = FakeClient(
        [
            {
                "date": "2026-01-08",
                "short_volume_ratio": 25.0,
                "short_volume": 100,
                "total_volume": 400,
            }
        ]
    )
    rows = _safe_prior_short_volume(
        client,
        "AAA",
        date(2026, 1, 9),
    )
    assert len(rows) == 1
    assert client.calls[0]["date_lte"] == date(2026, 1, 8)
    assert client.calls[0]["date_gte"] == date(2025, 12, 10)


def test_probe_rejects_event_day_or_future_record():
    client = FakeClient(
        [
            {
                "date": "2026-01-09",
                "short_volume_ratio": 25.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="point-in-time violation"):
        _safe_prior_short_volume(client, "AAA", date(2026, 1, 9))


def test_event_metrics_uses_latest_five_available_ratios():
    records = []
    for day, ratio in [
        ("2026-01-02", 10.0),
        ("2026-01-05", 20.0),
        ("2026-01-06", 30.0),
        ("2026-01-07", 40.0),
        ("2026-01-08", 50.0),
    ]:
        records.append(
            {
                "_record_day": pd.Timestamp(day).date(),
                "short_volume_ratio": ratio,
                "short_volume": 100,
                "total_volume": 400,
            }
        )

    result = _event_metrics(
        month="2026-01",
        trading_day="2026-01-09",
        ticker="AAA",
        threshold_pct=10.0,
        records=records,
    )
    assert result["has_prior_within_1d"] is True
    assert result["has_five_records"] is True
    assert result["latest_short_volume_ratio"] == 50.0
    assert result["latest5_short_volume_ratio_mean"] == 30.0
    assert result["latest_minus_latest5_mean"] == 20.0


def test_success_check_requires_all_three_coverage_rules():
    rows = []
    for month in ("2026-01", "2026-02", "2026-03"):
        for _ in range(10):
            rows.append(
                {
                    "month": month,
                    "has_prior_within_7d": True,
                    "has_five_records": True,
                }
            )
    frame = pd.DataFrame(rows)
    checks = success_check(frame)
    assert checks["all_pass"] is True

    frame.loc[frame["month"].eq("2026-02"), "has_prior_within_7d"] = False
    checks = success_check(frame)
    assert checks["every_month_recent_7d_at_least_80pct"] is False
    assert checks["all_pass"] is False
