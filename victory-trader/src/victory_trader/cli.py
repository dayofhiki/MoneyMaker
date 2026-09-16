from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .config import load_settings
from .event_study import run_event_study
from .market_data import bars_from_massive_payload
from .market_dataset import build_market_event_dataset, save_dataset
from .massive_client import MassiveClient


def check_api() -> int:
    settings = load_settings()
    client = MassiveClient(settings.massive_api_key)
    payload = client.previous_close("AAPL")
    print(json.dumps(payload, indent=2))
    return 0


def event_study(ticker: str, day: date) -> int:
    settings = load_settings()
    client = MassiveClient(settings.massive_api_key)
    payload = client.minute_bars(ticker, day)
    bars = bars_from_massive_payload(payload)
    previous_close = client.historical_previous_close(ticker, day)

    result = run_event_study(
        ticker=ticker,
        bars=bars,
        previous_close=previous_close,
    )

    if result.empty:
        print("No threshold crossing events found.")
        return 0

    print(f"Previous close: {previous_close:.4f}")
    print(result.to_string(index=False))
    return 0


def market_dataset(
    day: date,
    output: Path | None,
    min_price: float,
    max_price: float,
    min_high_return: float,
    min_dollar_volume: float,
    max_candidates: int | None,
    request_interval: float,
) -> int:
    settings = load_settings()
    client = MassiveClient(settings.massive_api_key)
    result = build_market_event_dataset(
        client,
        day,
        min_price=min_price,
        max_price=max_price,
        min_high_return_pct=min_high_return,
        min_day_dollar_volume=min_dollar_volume,
        max_candidates=max_candidates,
        request_interval_seconds=request_interval,
    )

    if result.empty:
        print("No qualifying momentum events found.")
        return 0

    output_path = output or Path("data/events") / f"market_events_{day.isoformat()}.parquet"
    save_dataset(result, output_path)
    print(
        f"Saved {len(result)} event rows across "
        f"{result['ticker'].nunique()} tickers to {output_path}"
    )
    print(result.head(20).to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="victory-trader")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-api", help="Check Massive API connectivity using AAPL previous close")

    study = sub.add_parser(
        "event-study",
        help="Run the v0.1 threshold event study for one ticker and trading day",
    )
    study.add_argument("ticker", help="U.S. equity ticker, e.g. AAPL")
    study.add_argument("day", type=date.fromisoformat, help="Trading day in YYYY-MM-DD format")

    dataset = sub.add_parser(
        "market-dataset",
        help="Scan the market for low-priced momentum events and save a research dataset",
    )
    dataset.add_argument("day", type=date.fromisoformat, help="Trading day in YYYY-MM-DD format")
    dataset.add_argument("--output", type=Path, default=None)
    dataset.add_argument("--min-price", type=float, default=0.50)
    dataset.add_argument("--max-price", type=float, default=20.0)
    dataset.add_argument("--min-high-return", type=float, default=20.0)
    dataset.add_argument("--min-dollar-volume", type=float, default=0.0)
    dataset.add_argument("--max-candidates", type=int, default=None)
    dataset.add_argument(
        "--request-interval",
        type=float,
        default=12.5,
        help="Seconds between Massive requests; 12.5 is conservative for the free plan.",
    )

    args = parser.parse_args()

    if args.command == "check-api":
        return check_api()
    if args.command == "event-study":
        return event_study(args.ticker, args.day)
    if args.command == "market-dataset":
        return market_dataset(
            day=args.day,
            output=args.output,
            min_price=args.min_price,
            max_price=args.max_price,
            min_high_return=args.min_high_return,
            min_dollar_volume=args.min_dollar_volume,
            max_candidates=args.max_candidates,
            request_interval=args.request_interval,
        )
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
