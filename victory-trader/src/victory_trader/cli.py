from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .analytics import load_event_dataset, summarize_by_threshold, summarize_excursions, summarize_rvol
from .config import load_settings
from .event_study import run_event_study
from .history import load_target_with_history
from .market_dataset import build_market_event_dataset, save_dataset
from .massive_client import MassiveClient
from .multi_day import build_multi_day_dataset


def check_api() -> int:
    settings = load_settings()
    client = MassiveClient(settings.massive_api_key)
    payload = client.previous_close("AAPL")
    print(json.dumps(payload, indent=2))
    return 0


def event_study(ticker: str, day: date) -> int:
    settings = load_settings()
    client = MassiveClient(settings.massive_api_key)
    bars, history_bars = load_target_with_history(client, ticker, day)
    previous_close = client.historical_previous_close(ticker, day)

    result = run_event_study(
        ticker=ticker,
        bars=bars,
        previous_close=previous_close,
        history_bars=history_bars,
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
    print(f"Saved {len(result)} event rows across {result['ticker'].nunique()} tickers to {output_path}")
    print(result.head(20).to_string(index=False))
    return 0


def multi_day_dataset(
    start: date,
    end: date,
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
    result, skipped = build_multi_day_dataset(
        client,
        start,
        end,
        min_price=min_price,
        max_price=max_price,
        min_high_return_pct=min_high_return,
        min_day_dollar_volume=min_dollar_volume,
        max_candidates=max_candidates,
        request_interval_seconds=request_interval,
    )

    if not result.empty:
        output_path = output or Path("data/events") / (
            f"market_events_{start.isoformat()}_{end.isoformat()}.parquet"
        )
        save_dataset(result, output_path)
        print(
            f"Saved {len(result)} event rows across {result['ticker'].nunique()} tickers "
            f"and {result['trading_day'].nunique()} trading days to {output_path}"
        )
    else:
        print("No qualifying momentum events found in the requested range.")

    if skipped:
        print(f"Skipped {len(skipped)} calendar days.")
        for skipped_day, reason in skipped[:10]:
            print(f"  {skipped_day}: {reason}")
    return 0


def analyze_dataset(path: Path, horizon: int) -> int:
    frame = load_event_dataset(path)
    if frame.empty:
        print("Dataset is empty.")
        return 0

    print("\n=== Continuation by threshold ===")
    print(summarize_by_threshold(frame).to_string(index=False))
    print("\n=== MFE / MAE by threshold ===")
    print(summarize_excursions(frame).to_string(index=False))

    if "rvol_cumulative_20d" in frame.columns:
        print(f"\n=== RVOL buckets at +{horizon}m ===")
        rvol = summarize_rvol(frame, horizon_min=horizon)
        print(rvol.to_string(index=False) if not rvol.empty else "No rows with historical RVOL yet.")
    return 0


def _add_dataset_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-price", type=float, default=0.50)
    parser.add_argument("--max-price", type=float, default=20.0)
    parser.add_argument("--min-high-return", type=float, default=20.0)
    parser.add_argument("--min-dollar-volume", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=12.5,
        help="Seconds between Massive requests; 12.5 is conservative for the free plan.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="victory-trader")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-api", help="Check Massive API connectivity using AAPL previous close")

    study = sub.add_parser("event-study", help="Run the threshold event study for one ticker/day")
    study.add_argument("ticker", help="U.S. equity ticker, e.g. AAPL")
    study.add_argument("day", type=date.fromisoformat, help="Trading day in YYYY-MM-DD format")

    dataset = sub.add_parser("market-dataset", help="Build one trading day's market-wide dataset")
    dataset.add_argument("day", type=date.fromisoformat)
    _add_dataset_options(dataset)

    multi = sub.add_parser("multi-day-dataset", help="Build a market event dataset across a date range")
    multi.add_argument("start", type=date.fromisoformat)
    multi.add_argument("end", type=date.fromisoformat)
    _add_dataset_options(multi)

    analyze = sub.add_parser("analyze-dataset", help="Summarize continuation, excursions, and RVOL")
    analyze.add_argument("path", type=Path)
    analyze.add_argument("--horizon", type=int, default=5, help="Horizon used for RVOL bucket analysis")

    args = parser.parse_args()
    if args.command == "check-api":
        return check_api()
    if args.command == "event-study":
        return event_study(args.ticker, args.day)
    if args.command == "market-dataset":
        return market_dataset(
            args.day, args.output, args.min_price, args.max_price,
            args.min_high_return, args.min_dollar_volume, args.max_candidates,
            args.request_interval,
        )
    if args.command == "multi-day-dataset":
        return multi_day_dataset(
            args.start, args.end, args.output, args.min_price, args.max_price,
            args.min_high_return, args.min_dollar_volume, args.max_candidates,
            args.request_interval,
        )
    if args.command == "analyze-dataset":
        return analyze_dataset(args.path, args.horizon)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
