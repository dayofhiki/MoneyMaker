"""Read-only probe for historical high-resolution market-data access."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import requests
from botocore.exceptions import ClientError

from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient
from .market_calendar import regular_session_bounds
from .massive_client import MassiveClient

TRADES_FLATFILE_PREFIX = "us_stocks_sip/trades_v1"
QUOTES_FLATFILE_PREFIX = "us_stocks_sip/quotes_v1"


def _http_status(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return int(status) if status is not None else None


def _s3_status(exc: ClientError) -> str:
    error = exc.response.get("Error", {})
    meta = exc.response.get("ResponseMetadata", {})
    code = error.get("Code")
    status = meta.get("HTTPStatusCode")
    return str(status if status is not None else code or "unknown")


def run_probe(
    rest_client: MassiveClient,
    flatfile_client: MassiveFlatFilesClient,
    *,
    ticker: str,
    day: date,
) -> dict[str, object]:
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        raise ValueError("ticker must not be blank")
    result: dict[str, object] = {
        "ticker": normalized_ticker,
        "day": day.isoformat(),
    }

    try:
        payload = rest_client.second_bars_range(
            normalized_ticker, day, day, adjusted=False
        )
        rows = list(payload.get("results") or [])
        result["second_bars_rest"] = {
            "status": "AVAILABLE",
            "rows": len(rows),
            "first_timestamp": rows[0].get("t") if rows else None,
            "last_timestamp": rows[-1].get("t") if rows else None,
        }
    except requests.HTTPError as exc:
        result["second_bars_rest"] = {
            "status": "UNAVAILABLE",
            "http_status": _http_status(exc),
        }

    bounds = regular_session_bounds(day)
    if bounds is None:
        raise ValueError(f"{day} is not a regular US equity trading day")
    start_ns = int(bounds[0].timestamp() * 1_000_000_000)
    end_ns = start_ns + 10_000_000_000
    try:
        rows = rest_client.trades(
            normalized_ticker,
            timestamp_gte=start_ns,
            timestamp_lte=end_ns,
            limit=50_000,
            order="asc",
        )
        result["trades_rest_10s"] = {
            "status": "AVAILABLE",
            "rows": len(rows),
            "first_sip_timestamp": (
                rows[0].get("sip_timestamp") if rows else None
            ),
            "last_sip_timestamp": (
                rows[-1].get("sip_timestamp") if rows else None
            ),
        }
    except requests.HTTPError as exc:
        result["trades_rest_10s"] = {
            "status": "UNAVAILABLE",
            "http_status": _http_status(exc),
        }

    for name, prefix in (
        ("trades_flatfile", TRADES_FLATFILE_PREFIX),
        ("quotes_flatfile", QUOTES_FLATFILE_PREFIX),
    ):
        try:
            obj = flatfile_client.head(prefix, day)
            result[name] = {
                "status": "AVAILABLE",
                "size_bytes": obj.size_bytes,
                "key": obj.key,
            }
        except FileNotFoundError:
            result[name] = {"status": "MISSING"}
        except ClientError as exc:
            result[name] = {
                "status": "UNAVAILABLE",
                "error_code": _s3_status(exc),
            }

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe existing historical second/trade/quote access read-only."
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--day", type=date.fromisoformat, default=date(2026, 1, 2))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    rest_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=None,
        request_interval_seconds=0.0,
    )
    flatfile_client = MassiveFlatFilesClient(access_key, secret_key)
    result = run_probe(
        rest_client,
        flatfile_client,
        ticker=args.ticker,
        day=args.day,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
