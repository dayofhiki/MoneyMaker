from datetime import date

import pandas as pd
import pytest

from victory_trader.sec_8k_supply_probe import (
    _event_metrics,
    _safe_prior_disclosures,
    success_check,
)


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def eight_k_disclosures(
        self,
        ticker,
        *,
        filing_date_gte,
        filing_date_lte,
        limit=1000,
    ):
        self.calls.append(
            {
                "ticker": ticker,
                "filing_date_gte": filing_date_gte,
                "filing_date_lte": filing_date_lte,
            }
        )
        return list(self.rows)


def test_probe_requests_only_filings_before_event_day():
    client = FakeClient(
        [
            {
                "filing_date": "2026-01-08",
                "accession_number": "a",
                "primary_category": "financial_results",
                "secondary_category": "earnings",
                "tertiary_category": "quarterly_results",
            }
        ]
    )
    rows = _safe_prior_disclosures(
        client,
        "AAA",
        date(2026, 1, 9),
    )
    assert len(rows) == 1
    assert client.calls[0]["filing_date_lte"] == date(2026, 1, 8)
    assert client.calls[0]["filing_date_gte"] == date(2025, 12, 10)


def test_probe_rejects_same_day_filing():
    client = FakeClient(
        [
            {
                "filing_date": "2026-01-09",
                "accession_number": "a",
            }
        ]
    )
    with pytest.raises(ValueError, match="point-in-time violation"):
        _safe_prior_disclosures(client, "AAA", date(2026, 1, 9))


def test_event_metrics_deduplicates_filings_but_counts_disclosures():
    disclosures = [
        {
            "_filing_day": pd.Timestamp("2026-01-05").date(),
            "accession_number": "a",
            "primary_category": "financial_results",
            "secondary_category": "earnings",
            "tertiary_category": "quarterly_results",
        },
        {
            "_filing_day": pd.Timestamp("2026-01-05").date(),
            "accession_number": "a",
            "primary_category": "corporate_governance",
            "secondary_category": "leadership",
            "tertiary_category": "executive_change",
        },
    ]
    result = _event_metrics(
        month="2026-01",
        trading_day="2026-01-09",
        ticker="AAA",
        threshold_pct=10.0,
        disclosures=disclosures,
    )
    assert result["unique_filings_30d"] == 1
    assert result["disclosures_30d"] == 2
    assert result["has_filing_within_7d"] is True
    assert result["distinct_primary_categories"] == 2


def test_success_check_requires_month_coverage_and_categories():
    events = pd.DataFrame(
        [
            {
                "month": month,
                "has_filing_within_30d": covered,
            }
            for month in ("2026-01", "2026-02", "2026-03")
            for covered in ([True] * 2 + [False] * 8)
        ]
    )
    disclosures = pd.DataFrame(
        [{"category_complete": True} for _ in range(10)]
    )
    checks = success_check(events, disclosures)
    assert checks["all_pass"] is True

    events.loc[events["month"].eq("2026-02"), "has_filing_within_30d"] = False
    checks = success_check(events, disclosures)
    assert checks["every_month_30d_coverage_at_least_10pct"] is False
    assert checks["all_pass"] is False
