from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from victory_trader.supply_admission_transition_timing import (
    enrich_supply,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, date]] = []

    def ticker_details(
        self, ticker: str, query_day: date
    ) -> dict[str, object]:
        self.calls.append((ticker, query_day))
        return {
            "results": {
                "weighted_shares_outstanding": 1_000_000,
                "share_class_shares_outstanding": 800_000,
                "market_cap": 12_000_000,
            }
        }


def test_supply_enrichment_is_strictly_prior_and_causal() -> None:
    frame = pd.DataFrame(
        [
            {
                "trading_day": "2026-06-29",
                "ticker": "TEST",
                "hot_t": 1000,
                "current_price": 12.0,
                "return_from_previous_close_pct": 20.0,
                "log_minute_volume": np.log1p(100_000),
            }
        ]
    )
    client = FakeClient()
    out = enrich_supply(frame, client)

    assert client.calls == [
        ("TEST", date(2026, 6, 28))
    ]
    assert out.iloc[0]["supply_query_date"] == "2026-06-28"
    assert out.iloc[0][
        "supply_weighted_share_turnover_1m"
    ] == pytest.approx(0.1)
    assert out.iloc[0][
        "supply_share_class_to_weighted_ratio"
    ] == pytest.approx(0.8)
    assert out.iloc[0][
        "supply_log_implied_market_cap_prior"
    ] == pytest.approx(np.log(10_000_000))
