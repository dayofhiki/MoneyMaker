from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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
        day_str = day.isoformat()
        return self._get(
            f"/v2/aggs/ticker/{ticker.upper()}/range/1/minute/{day_str}/{day_str}",
            {"adjusted": "true", "sort": "asc", "limit": 50000},
        )
