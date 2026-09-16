from __future__ import annotations

import argparse
import json
from datetime import date

from .config import load_settings
from .event_study import run_event_study
from .market_data import bars_from_massive_payload
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

    args = parser.parse_args()

    if args.command == "check-api":
        return check_api()
    if args.command == "event-study":
        return event_study(args.ticker, args.day)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
