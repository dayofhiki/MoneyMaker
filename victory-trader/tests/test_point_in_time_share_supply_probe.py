from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from victory_trader.point_in_time_share_supply_probe import (
    deterministic_sample,
    probe_rows,
    success_check,
)


@dataclass
class FakeClient:
    calls: list[tuple[str, date]]

    def ticker_details(self, ticker: str, day: date):
        self.calls.append((ticker, day))
        return {
            "results": {
                "ticker": ticker,
                "share_class_shares_outstanding": 10_000_000,
                "weighted_shares_outstanding": 12_000_000,
                "market_cap": 60_000_000,
            }
        }


def _trade(month: str, day: str, ticker: str, t: int):
    return {
        "month": month,
        "trading_day": day,
        "ticker": ticker,
        "t": t,
        "policy": "feasible_earliest_15m_cap1",
        "previous_close": 5.0,
        "volume_5m": 100_000,
        "dollar_volume_5m": 500_000,
        "realized_base_net_return_pct": 999.0,
    }


def test_sampling_ignores_outcome_and_deduplicates_ticker_day():
    rows = [
        _trade("2026-01", "2026-01-02", "AAA", 2),
        _trade("2026-01", "2026-01-02", "AAA", 1),
        _trade("2026-01", "2026-01-03", "BBB", 1),
    ]
    rows.append(
        {
            **_trade("2026-01", "2026-01-04", "CCC", 1),
            "policy": "feasible_hurdle_ev_15m_cap1",
        }
    )
    frame = pd.DataFrame(rows)

    sample = deterministic_sample(frame)

    assert sample[["trading_day", "ticker"]].values.tolist() == [
        ["2026-01-02", "AAA"],
        ["2026-01-03", "BBB"],
    ]
    assert sample.iloc[0]["t"] == 1


def test_probe_queries_strictly_prior_day_and_computes_supply_ratios():
    frame = pd.DataFrame(
        [_trade("2026-01", "2026-01-05", "AAA", 1)]
    )
    client = FakeClient(calls=[])

    rows = probe_rows(frame, client)

    assert client.calls == [("AAA", date(2026, 1, 4))]
    item = rows.iloc[0]
    assert item["query_date"] == "2026-01-04"
    assert item["implied_market_cap_prior"] == 60_000_000
    assert item["volume_5m_share_turnover"] == 0.01
    assert item["dollar_volume_5m_market_cap_turnover"] == (
        500_000 / 60_000_000
    )


def test_success_check_requires_all_months_and_strict_prior_dates():
    rows = pd.DataFrame(
        [
            {
                "month": month,
                "trading_day": f"{month}-10",
                "query_date": f"{month}-09",
            }
            for month in ("2026-01", "2026-02", "2026-03")
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "month": month,
                "request_success": 1.0,
                "weighted_shares_coverage": 0.9,
                "share_class_coverage": 0.9,
                "implied_market_cap_coverage": 0.9,
            }
            for month in ("2026-01", "2026-02", "2026-03")
        ]
    )

    checks = success_check(rows, summary)

    assert checks["all_pass"] is True
