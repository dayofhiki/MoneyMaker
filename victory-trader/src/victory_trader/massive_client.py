from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

import requests


@dataclass
class MassiveClientStats:
    cache_hits: int = 0
    network_requests: int = 0
    retries: int = 0
    cache_writes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "network_requests": self.network_requests,
            "retries": self.retries,
            "cache_writes": self.cache_writes,
        }


@dataclass
class MassiveClient:
    api_key: str
    base_url: str = "https://api.massive.com"
    cache_dir: Path | None = None
    request_interval_seconds: float = 0.0
    max_retries: int = 4
    retry_backoff_seconds: float = 1.0
    stats: MassiveClientStats = field(default_factory=MassiveClientStats, init=False)
    _last_network_request_at: float | None = field(default=None, init=False, repr=False)

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
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        self.stats.cache_hits += 1
        return payload

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
        self.stats.cache_writes += 1

    def _throttle_network(self) -> None:
        if self.request_interval_seconds <= 0 or self._last_network_request_at is None:
            return
        elapsed = time.monotonic() - self._last_network_request_at
        remaining = self.request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _retry_after_seconds(response: requests.Response, fallback: float) -> float:
        raw = getattr(response, "headers", {}).get("Retry-After")
        if raw is None:
            return fallback
        try:
            return max(float(raw), 0.0)
        except (TypeError, ValueError):
            return fallback

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """GET JSON from Massive with secret-safe auth, caching, and bounded retries.

        Rate limiting is applied only when a real network request is required.
        Cache hits therefore return immediately. 401/403/other non-retryable 4xx
        failures remain fatal; 429, 5xx, and transient request failures receive
        bounded retry/backoff.
        """
        query = dict(params or {})
        query.pop("apiKey", None)
        cache_path = self._cache_path(path, query) if use_cache else None
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            self._throttle_network()
            self._last_network_request_at = time.monotonic()
            self.stats.network_requests += 1
            try:
                response = requests.get(
                    url,
                    params=query,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30,
                )
            except (requests.Timeout, requests.ConnectionError):
                if attempt >= self.max_retries:
                    raise
                self.stats.retries += 1
                time.sleep(self.retry_backoff_seconds * (2**attempt))
                continue

            status = int(response.status_code)
            if status == 429 or 500 <= status < 600:
                if attempt >= self.max_retries:
                    response.raise_for_status()
                self.stats.retries += 1
                fallback = self.retry_backoff_seconds * (2**attempt)
                time.sleep(self._retry_after_seconds(response, fallback))
                continue

            response.raise_for_status()
            payload = response.json()
            self._write_cache(cache_path, payload)
            return payload

        raise RuntimeError("unreachable Massive request state")

    def _get_next_url(self, next_url: str) -> dict[str, Any]:
        """Follow a Massive pagination URL without ever forwarding credentials."""
        parsed = urlparse(next_url)
        expected_host = urlparse(self.base_url).netloc
        if parsed.netloc and parsed.netloc != expected_host:
            raise ValueError(f"unexpected Massive pagination host: {parsed.netloc}")
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        params.pop("apiKey", None)
        return self._get(parsed.path, params)

    def previous_close(self, ticker: str) -> dict[str, Any]:
        # This endpoint means "latest previous close", so do not persist it.
        return self._get(
            f"/v2/aggs/ticker/{ticker.upper()}/prev",
            {"adjusted": "true"},
            use_cache=False,
        )

    def minute_bars(self, ticker: str, day: date, *, adjusted: bool = False) -> dict[str, Any]:
        return self.minute_bars_range(ticker, day, day, adjusted=adjusted)

    def minute_bars_range(
        self,
        ticker: str,
        start: date,
        end: date,
        *,
        adjusted: bool = False,
    ) -> dict[str, Any]:
        return self._get(
            f"/v2/aggs/ticker/{ticker.upper()}/range/1/minute/{start.isoformat()}/{end.isoformat()}",
            {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000},
        )

    def daily_bars(
        self,
        ticker: str,
        start: date,
        end: date,
        *,
        adjusted: bool = False,
    ) -> dict[str, Any]:
        return self._get(
            f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start.isoformat()}/{end.isoformat()}",
            {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 5000},
        )

    def ticker_details(self, ticker: str, day: date | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if day is not None:
            params["date"] = day.isoformat()
        return self._get(f"/v3/reference/tickers/{ticker.upper()}", params)

    def quotes(
        self,
        ticker: str,
        *,
        timestamp_gte: int | str | None = None,
        timestamp_lte: int | str | None = None,
        limit: int = 50_000,
        order: str = "asc",
    ) -> list[dict[str, Any]]:
        """Fetch historical NBBO quotes for a stock ticker."""
        params: dict[str, Any] = {
            "limit": int(limit),
            "sort": "timestamp",
            "order": order,
        }
        if timestamp_gte is not None:
            params["timestamp.gte"] = timestamp_gte
        if timestamp_lte is not None:
            params["timestamp.lte"] = timestamp_lte

        page = self._get(f"/v3/quotes/{ticker.upper()}", params)
        rows: list[dict[str, Any]] = list(page.get("results") or [])
        next_url = page.get("next_url")
        while next_url:
            page = self._get_next_url(str(next_url))
            rows.extend(page.get("results") or [])
            next_url = page.get("next_url")
        return rows

    def news(
        self,
        ticker: str,
        *,
        published_gte: str | None = None,
        published_lte: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch ticker-tagged news, optionally bounded by publication time."""
        params: dict[str, Any] = {
            "ticker": ticker.upper(),
            "limit": int(limit),
            "sort": "published_utc",
            "order": "asc",
        }
        if published_gte is not None:
            params["published_utc.gte"] = published_gte
        if published_lte is not None:
            params["published_utc.lte"] = published_lte

        page = self._get("/v2/reference/news", params)
        rows: list[dict[str, Any]] = list(page.get("results") or [])
        next_url = page.get("next_url")
        while next_url:
            page = self._get_next_url(str(next_url))
            rows.extend(page.get("results") or [])
            next_url = page.get("next_url")
        return rows

    def short_volume(
        self,
        ticker: str,
        *,
        date_gte: date | str | None = None,
        date_lte: date | str | None = None,
        limit: int = 50_000,
    ) -> list[dict[str, Any]]:
        """Fetch historical FINRA daily short-volume records for a stock."""
        params: dict[str, Any] = {
            "ticker": ticker.upper(),
            "limit": int(limit),
            "sort": "date.asc",
        }
        if date_gte is not None:
            params["date.gte"] = (
                date_gte.isoformat() if isinstance(date_gte, date) else str(date_gte)
            )
        if date_lte is not None:
            params["date.lte"] = (
                date_lte.isoformat() if isinstance(date_lte, date) else str(date_lte)
            )

        page = self._get("/stocks/v1/short-volume", params)
        rows: list[dict[str, Any]] = list(page.get("results") or [])
        next_url = page.get("next_url")
        while next_url:
            page = self._get_next_url(str(next_url))
            rows.extend(page.get("results") or [])
            next_url = page.get("next_url")
        return rows

    def short_volume_on(self, day: date | str) -> list[dict[str, Any]]:
        """Fetch the whole-market FINRA short-volume file for one trade date."""
        day_text = day.isoformat() if isinstance(day, date) else str(day)
        page = self._get(
            "/stocks/v1/short-volume",
            {
                "date": day_text,
                "limit": 50_000,
                "sort": "ticker.asc",
            },
        )
        rows: list[dict[str, Any]] = list(page.get("results") or [])
        next_url = page.get("next_url")
        while next_url:
            page = self._get_next_url(str(next_url))
            rows.extend(page.get("results") or [])
            next_url = page.get("next_url")
        return rows

    def eight_k_disclosures(
        self,
        ticker: str,
        *,
        filing_date_gte: date | str | None = None,
        filing_date_lte: date | str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch classified SEC 8-K disclosures for a ticker."""
        params: dict[str, Any] = {
            "tickers": ticker.upper(),
            "limit": int(limit),
            "sort": "filing_date.asc",
        }
        if filing_date_gte is not None:
            params["filing_date.gte"] = (
                filing_date_gte.isoformat()
                if isinstance(filing_date_gte, date)
                else str(filing_date_gte)
            )
        if filing_date_lte is not None:
            params["filing_date.lte"] = (
                filing_date_lte.isoformat()
                if isinstance(filing_date_lte, date)
                else str(filing_date_lte)
            )
        page = self._get("/stocks/filings/8-K/vX/disclosures", params)
        rows: list[dict[str, Any]] = list(page.get("results") or [])
        next_url = page.get("next_url")
        while next_url:
            page = self._get_next_url(str(next_url))
            rows.extend(page.get("results") or [])
            next_url = page.get("next_url")
        return rows

    def eight_k_disclosures_market(
        self,
        *,
        filing_date_gte: date | str | None = None,
        filing_date_lte: date | str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch classified SEC 8-K disclosures across the market."""
        params: dict[str, Any] = {
            "limit": int(limit),
            "sort": "filing_date.asc",
        }
        if filing_date_gte is not None:
            params["filing_date.gte"] = (
                filing_date_gte.isoformat()
                if isinstance(filing_date_gte, date)
                else str(filing_date_gte)
            )
        if filing_date_lte is not None:
            params["filing_date.lte"] = (
                filing_date_lte.isoformat()
                if isinstance(filing_date_lte, date)
                else str(filing_date_lte)
            )

        page = self._get("/stocks/filings/8-K/vX/disclosures", params)
        rows: list[dict[str, Any]] = list(page.get("results") or [])
        next_url = page.get("next_url")
        while next_url:
            page = self._get_next_url(str(next_url))
            rows.extend(page.get("results") or [])
            next_url = page.get("next_url")
        return rows

    def reference_tickers(
        self,
        day: date,
        *,
        market: str = "stocks",
        security_type: str = "CS",
        active: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch the complete point-in-time ticker list using max-size pages.

        One paginated daily universe request replaces one ticker-details request per
        candidate. This preserves point-in-time taxonomy while dramatically reducing
        network calls on rate-limited plans.
        """
        page = self._get(
            "/v3/reference/tickers",
            {
                "date": day.isoformat(),
                "market": market,
                "type": security_type,
                "active": str(active).lower(),
                "limit": 1000,
                "sort": "ticker",
                "order": "asc",
            },
        )
        rows: list[dict[str, Any]] = list(page.get("results") or [])
        next_url = page.get("next_url")
        while next_url:
            page = self._get_next_url(str(next_url))
            rows.extend(page.get("results") or [])
            next_url = page.get("next_url")
        return rows

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
        payload = self.daily_bars(
            ticker,
            day - timedelta(days=14),
            day - timedelta(days=1),
            adjusted=False,
        )
        results = payload.get("results") or []
        if not results:
            raise ValueError(f"No prior daily bar found for {ticker.upper()} before {day}")
        return float(results[-1]["c"])
