from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from victory_trader.expanded_share_supply_enrichment import (
    coverage_row,
    enrich_anchors,
)


@dataclass
class FakeClient:
    calls: list[tuple[str, date]]

    def ticker_details(self, ticker: str, day: date):
        self.calls.append((ticker, day))
        return {
            "results": {
                "weighted_shares_outstanding": 10_000_000,
                "share_class_shares_outstanding": 8_000_000,
                "market_cap": 50_000_000,
            }
        }


def _row(ticker: str, *, dollar_volume=200_000, tx=100, active=1.0):
    return {
        "trading_day": "2026-01-05",
        "ticker": ticker,
        "t": 1,
        "c": 6.0,
        "previous_close": 5.0,
        "entry_price": 6.0,
        "active_minute_fraction_15m": active,
        "dollar_volume_5m": dollar_volume,
        "transactions_5m": tx,
        "volume_5m": 40_000,
    }


def test_enrichment_queries_only_execution_feasible_anchor():
    frame = pd.DataFrame(
        [
            _row("AAA"),
            _row("BBB", dollar_volume=50_000),
        ]
    )
    client = FakeClient(calls=[])

    enriched = enrich_anchors(frame, client)

    assert client.calls == [("AAA", date(2026, 1, 4))]
    aaa = enriched.loc[enriched["ticker"].eq("AAA")].iloc[0]
    bbb = enriched.loc[enriched["ticker"].eq("BBB")].iloc[0]
    assert aaa["supply_query_success"] == 1.0
    assert aaa["supply_weighted_shares_outstanding_prior"] == 10_000_000
    assert aaa["supply_query_date"] == "2026-01-04"
    assert bbb["supply_query_success"] == 0.0
    assert np.isnan(bbb["supply_weighted_shares_outstanding_prior"])


def test_coverage_is_measured_only_inside_gated_universe():
    frame = pd.DataFrame(
        [
            {
                **_row("AAA"),
                "supply_query_date": "2026-01-04",
                "supply_query_success": 1.0,
                "supply_weighted_shares_outstanding_prior": 10_000_000,
                "supply_share_class_shares_outstanding_prior": 8_000_000,
                "supply_provider_market_cap_prior": 50_000_000,
            },
            {
                **_row("BBB", dollar_volume=50_000),
                "supply_query_date": "",
                "supply_query_success": 0.0,
                "supply_weighted_shares_outstanding_prior": np.nan,
                "supply_share_class_shares_outstanding_prior": np.nan,
                "supply_provider_market_cap_prior": np.nan,
            },
        ]
    )

    summary = coverage_row(frame, "2026-01")

    assert summary["feasible_anchors"] == 1
    assert summary["request_success"] == 1.0
    assert summary["weighted_shares_coverage"] == 1.0
    assert summary["strict_prior_query_dates"] is True
