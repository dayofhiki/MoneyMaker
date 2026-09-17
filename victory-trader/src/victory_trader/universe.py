from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import requests

from .massive_client import MassiveClient


ALLOWED_PRIMARY_EXCHANGES = {"XNAS", "XNYS", "XASE"}


@dataclass(frozen=True)
class SecurityMetadata:
    ticker: str
    name: str | None
    security_type: str | None
    market: str | None
    locale: str | None
    primary_exchange: str | None
    active: bool | None
    market_cap: float | None
    shares_outstanding: float | None

    @property
    def is_research_common_stock(self) -> bool:
        return (
            self.security_type == "CS"
            and self.market == "stocks"
            and self.locale == "us"
            and self.primary_exchange in ALLOWED_PRIMARY_EXCHANGES
        )


def metadata_from_row(row: dict) -> SecurityMetadata:
    ticker = str(row.get("ticker") or "").upper()
    if not ticker:
        raise ValueError("ticker metadata row has no ticker")

    def optional_float(key: str) -> float | None:
        raw = row.get(key)
        return None if raw is None else float(raw)

    active_raw = row.get("active")
    active = bool(active_raw) if active_raw is not None else None
    return SecurityMetadata(
        ticker=ticker,
        name=row.get("name"),
        security_type=row.get("type"),
        market=row.get("market"),
        locale=row.get("locale"),
        primary_exchange=row.get("primary_exchange"),
        active=active,
        market_cap=optional_float("market_cap"),
        shares_outstanding=optional_float("share_class_shares_outstanding"),
    )


def metadata_from_payload(payload: dict) -> SecurityMetadata:
    return metadata_from_row(payload.get("results") or {})


def _missing_metadata(ticker: str) -> SecurityMetadata:
    """Sentinel metadata that safely fails the research-universe predicate."""
    return SecurityMetadata(
        ticker=ticker.upper(),
        name=None,
        security_type=None,
        market=None,
        locale=None,
        primary_exchange=None,
        active=None,
        market_cap=None,
        shares_outstanding=None,
    )


def fetch_security_metadata(client: MassiveClient, ticker: str, day: date) -> SecurityMetadata:
    """Fetch point-in-time metadata; treat a 404 as an ineligible candidate."""
    try:
        return metadata_from_payload(client.ticker_details(ticker, day))
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return _missing_metadata(ticker)
        raise


def fetch_research_universe_metadata(
    client: MassiveClient,
    day: date,
) -> dict[str, SecurityMetadata]:
    """Fetch point-in-time common-stock taxonomy in bulk for one trading day.

    Massive's all-tickers endpoint can return up to 1000 symbols per page. Using it
    once per day avoids a separate rate-limited ticker-details request for every
    momentum candidate while preserving historical type/exchange filtering.
    """
    rows = client.reference_tickers(day, market="stocks", security_type="CS", active=True)
    metadata: dict[str, SecurityMetadata] = {}
    for row in rows:
        try:
            item = metadata_from_row(row)
        except (TypeError, ValueError):
            continue
        metadata[item.ticker] = item
    return metadata


def split_tickers_from_payload(payload: dict) -> set[str]:
    return {
        str(row.get("ticker") or "").upper()
        for row in (payload.get("results") or [])
        if row.get("ticker")
    }
