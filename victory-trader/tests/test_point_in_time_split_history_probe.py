from datetime import date

import numpy as np
import pandas as pd

from victory_trader.massive_client import MassiveClient
from victory_trader.point_in_time_split_history_probe import (
    coverage_summary,
    deterministic_sample,
    probe_rows,
    success_check,
)


class FakeClient:
    def __init__(self, records_by_ticker):
        self.records_by_ticker = records_by_ticker
        self.calls = []
        self.stats = type(
            "Stats",
            (),
            {"to_dict": lambda self: {}},
        )()

    def splits(
        self,
        ticker,
        *,
        execution_date_gte=None,
        execution_date_lte=None,
        limit=1000,
    ):
        self.calls.append(
            (
                ticker,
                execution_date_gte,
                execution_date_lte,
                limit,
            )
        )
        return list(self.records_by_ticker.get(ticker, []))


def _attempts():
    rows = []
    for month, day_prefix in (
        ("2026-01", "2026-01"),
        ("2026-02", "2026-02"),
        ("2026-03", "2026-03"),
    ):
        for index in range(30):
            rows.append(
                {
                    "month": month,
                    "policy": "feasible_earliest_15m_cap1",
                    "trading_day": f"{day_prefix}-{(index % 20) + 1:02d}",
                    "ticker": f"T{index:02d}",
                    "decision_t": index,
                }
            )
    return pd.DataFrame(rows)


def test_deterministic_sample_returns_25_per_month():
    sample = deterministic_sample(_attempts())
    counts = sample.groupby("month").size().to_dict()
    assert counts == {
        "2026-01": 25,
        "2026-02": 25,
        "2026-03": 25,
    }


def test_probe_uses_strict_prior_bounds_and_counts_reverse_splits():
    sample = pd.DataFrame(
        [
            {
                "month": "2026-01",
                "trading_day": "2026-01-20",
                "ticker": "AAA",
            }
        ]
    )
    client = FakeClient(
        {
            "AAA": [
                {
                    "ticker": "AAA",
                    "execution_date": "2025-12-01",
                    "split_from": 10,
                    "split_to": 1,
                    "adjustment_type": "reverse_split",
                },
                {
                    "ticker": "AAA",
                    "execution_date": "2025-06-01",
                    "split_from": 1,
                    "split_to": 2,
                    "adjustment_type": "forward_split",
                },
            ]
        }
    )

    rows = probe_rows(sample, client)
    row = rows.iloc[0]

    assert client.calls[0][1] == date(2024, 1, 21)
    assert client.calls[0][2] == date(2026, 1, 19)
    assert row["reverse_split_count_730d"] == 1
    assert row["forward_split_count_730d"] == 1
    assert row["strict_prior_records"] == 1.0
    assert np.isclose(row["latest_reverse_ratio"], 0.1)
    assert np.isclose(
        row["latest_reverse_consolidation_factor"],
        10.0,
    )


def test_success_check_requires_reverse_support():
    rows = pd.DataFrame(
        [
            {
                "month": month,
                "request_success": 1.0,
                "strict_prior_records": 1.0,
                "reverse_split_count_730d": 1 if index < reverse else 0,
                "split_count_730d": 1 if index < reverse else 0,
            }
            for month, reverse in (
                ("2026-01", 4),
                ("2026-02", 3),
                ("2026-03", 3),
            )
            for index in range(25)
        ]
    )
    summary = coverage_summary(rows)
    assert success_check(rows, summary)["all_pass"] is True

    rows.loc[
        rows["month"].eq("2026-03"),
        "reverse_split_count_730d",
    ] = 0
    summary = coverage_summary(rows)
    assert success_check(rows, summary)["all_pass"] is False


def test_massive_splits_uses_bounds_and_paginates(monkeypatch):
    calls = []
    payloads = [
        {
            "results": [
                {
                    "ticker": "AAA",
                    "execution_date": "2025-01-02",
                    "adjustment_type": "reverse_split",
                }
            ],
            "next_url": "https://api.massive.com/stocks/v1/splits?cursor=next",
        },
        {
            "results": [
                {
                    "ticker": "AAA",
                    "execution_date": "2025-06-01",
                    "adjustment_type": "forward_split",
                }
            ]
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
    rows = client.splits(
        "aaa",
        execution_date_gte=date(2024, 1, 1),
        execution_date_lte=date(2025, 12, 31),
    )

    assert len(rows) == 2
    assert calls[0][0].endswith("/stocks/v1/splits")
    assert calls[0][1]["ticker"] == "AAA"
    assert calls[0][1]["execution_date.gte"] == "2024-01-01"
    assert calls[0][1]["execution_date.lte"] == "2025-12-31"
    assert calls[0][1]["sort"] == "execution_date.asc"
    assert calls[1][1] == {"cursor": "next"}
