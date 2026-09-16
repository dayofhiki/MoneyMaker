from __future__ import annotations

from dataclasses import dataclass
from datetime import date

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


def metadata_from_payload(payload: dict) -> SecurityMetadata:
    row = payload.get("results") or {}
    ticker = str(row.get("ticker") or "").upper()
    if not ticker:
        raise ValueError("ticker details payload has no ticker")

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


def fetch_security_metadata(client: MassiveClient, ticker: str, day: date) -> SecurityMetadata:
    return metadata_from_payload(client.ticker_details(ticker, day))


def split_tickers_from_payload(payload: dict) -> set[str]:
    return {
        str(row.get("ticker") or "").upper()
        for row in (payload.get("results") or [])
        if row.get("ticker")
    }
