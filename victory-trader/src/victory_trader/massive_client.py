from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import requests


@dataclass
class MassiveClient:
    api_key: str
    base_url: str = "https://api.massive.com"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = dict(params or {})
        query["apiKey"] = self.api_key
        response = requests.get(f"{self.base_url}{path}", params=query, timeout=30)
        response.raise_for_status()
        return response.json()

    def previous_close(self, ticker: str) -> dict[str, Any]:
        return self._get(f"/v2/aggs/ticker/{ticker.upper()}/prev", {"adjusted": "true"})

    def minute_bars(self, ticker: str, day: date) -> dict[str, Any]:
        return self.minute_bars_range(ticker, day, day)

    def minute_bars_range(self, ticker: str, start: date, end: date) -> dict[str, Any]:
        return self._get(
            f"/v2/aggs/ticker/{ticker.upper()}/range/1/minute/{start.isoformat()}/{end.isoformat()}",
            {"adjusted": "true", "sort": "asc", "limit": 50000},
        )

    def daily_bars(self, ticker: str, start: date, end: date) -> dict[str, Any]:
        return self._get(
            f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start.isoformat()}/{end.isoformat()}",
            {"adjusted": "true", "sort": "asc", "limit": 5000},
        )

    def ticker_details(self, ticker: str, day: date | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if day is not None:
            params["date"] = day.isoformat()
        return self._get(f"/v3/reference/tickers/{ticker.upper()}", params)

    def splits_on(self, day: date) -> dict[str, Any]:
        return self._get(
            "/stocks/v1/splits",
            {
                "execution_date": day.isoformat(),
                "limit": 5000,
                "sort": "execution_date.asc",
            },
        )

    def historical_previous_close(self, ticker: str, day: date) -> float:
        payload = self.daily_bars(ticker, day - timedelta(days=14), day - timedelta(days=1))
        results = payload.get("results") or []
        if not results:
            raise ValueError(f"No prior daily bar found for {ticker.upper()} before {day}")
        return float(results[-1]["c"])
