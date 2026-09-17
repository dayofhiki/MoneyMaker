from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests


@dataclass
class MassiveClient:
    api_key: str
    base_url: str = "https://api.massive.com"
    cache_dir: Path | None = None

    def _cache_path(self, path: str, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        canonical = json.dumps(
            {"path": path, "params": params},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, cache_path: Path | None) -> dict[str, Any] | None:
        if cache_path is None or not cache_path.exists():
            return None
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, cache_path: Path | None, payload: dict[str, Any]) -> None:
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(cache_path)

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """GET JSON from Massive without ever putting the API key in the URL.

        Historical research responses can be persisted under ``cache_dir``.
        Cache keys contain only the request path and public query parameters,
        never credentials. Only successful JSON responses are cached.
        """
        query = dict(params or {})
        cache_path = self._cache_path(path, query) if use_cache else None
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        response = requests.get(
            f"{self.base_url}{path}",
            params=query,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        self._write_cache(cache_path, payload)
        return payload

    def previous_close(self, ticker: str) -> dict[str, Any]:
        # This endpoint means "latest previous close", so do not persist it.
        return self._get(
            f"/v2/aggs/ticker/{ticker.upper()}/prev",
            {"adjusted": "true"},
            use_cache=False,
        )

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
