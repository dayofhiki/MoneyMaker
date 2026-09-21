from datetime import date

import numpy as np
import pandas as pd

from victory_trader.expanded_split_history_enrichment import (
    coverage_row,
    enrich_frame,
)
from victory_trader.massive_client import MassiveClient


def _frame():
    return pd.DataFrame(
        [
            {
                "trading_day": "2026-01-20",
                "ticker": "AAA",
                "dollar_volume_5m": 200000.0,
                "transactions_5m": 100.0,
                "active_minute_fraction_15m": 1.0,
            },
            {
                "trading_day": "2026-01-20",
                "ticker": "BBB",
                "dollar_volume_5m": 200000.0,
                "transactions_5m": 100.0,
                "active_minute_fraction_15m": 1.0,
            },
        ]
    )


def test_enrich_frame_uses_only_strict_prior_window():
    events = {
        "AAA": [
            {
                "_execution_date": date(2025, 12, 1),
                "adjustment_type": "reverse_split",
                "split_from": 10,
                "split_to": 1,
            },
            {
                "_execution_date": date(2026, 1, 20),
                "adjustment_type": "reverse_split",
                "split_from": 20,
                "split_to": 1,
            },
            {
                "_execution_date": date(2023, 1, 1),
                "adjustment_type": "reverse_split",
                "split_from": 100,
                "split_to": 1,
            },
        ],
        "BBB": [],
    }

    enriched = enrich_frame(_frame(), events)
    aaa = enriched.loc[enriched["ticker"].eq("AAA")].iloc[0]
    bbb = enriched.loc[enriched["ticker"].eq("BBB")].iloc[0]

    assert aaa["reverse_split_count_730d"] == 1.0
    assert aaa["reverse_split_count_365d"] == 1.0
    assert aaa["split_latest_reverse_consolidation"] == 10.0
    assert aaa["split_max_reverse_consolidation"] == 10.0
    assert aaa["split_history_strict_prior"] == 1.0

    assert bbb["split_any_730d"] == 0.0
    assert bbb["reverse_split_count_730d"] == 0.0
    assert np.isnan(bbb["split_days_since_latest_reverse"])

    coverage = coverage_row(enriched, "2026-01")
    assert coverage["query_success"] == 1.0
    assert coverage["strict_prior"] is True
    assert coverage["reverse_split_anchors_730d"] == 1


def test_massive_splits_market_omits_ticker_and_paginates(monkeypatch):
    calls = []
    payloads = [
        {
            "results": [
                {
                    "ticker": "AAA",
                    "execution_date": "2025-01-02",
                }
            ],
            "next_url": "https://api.massive.com/stocks/v1/splits?cursor=next",
        },
        {
            "results": [
                {
                    "ticker": "BBB",
                    "execution_date": "2025-01-03",
                }
            ],
        },
    ]

    class Response:
        status_code = 200
        headers = {}

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, *, params, headers, timeout):
        calls.append((url, params))
        return Response(payloads.pop(0))

    monkeypatch.setattr(
        "victory_trader.massive_client.requests.get",
        fake_get,
    )
    client = MassiveClient("secret")
    rows = client.splits_market(
        execution_date_gte=date(2024, 1, 1),
        execution_date_lte=date(2025, 12, 31),
    )

    assert len(rows) == 2
    assert calls[0][0].endswith("/stocks/v1/splits")
    assert "ticker" not in calls[0][1]
    assert calls[0][1]["execution_date.gte"] == "2024-01-01"
    assert calls[0][1]["execution_date.lte"] == "2025-12-31"
    assert calls[1][1] == {"cursor": "next"}
