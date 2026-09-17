from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .analytics import (
    load_event_dataset,
    summarize_barriers,
    summarize_by_threshold,
    summarize_cluster_uncertainty,
    summarize_concentration,
    summarize_cost_scenarios,
    summarize_excursions,
    summarize_latency,
    summarize_missingness,
    summarize_monthly_stability,
    summarize_price_buckets,
    summarize_rvol,
    summarize_sessions,
    summarize_tail_risk,
    summarize_time_buckets,
)
from .audit import audit_event_dataset
from .config import load_settings, require_flatfile_credentials
from .event_study import run_event_study
from .exits import summarize_unresolved_exposure
from .flatfiles import MassiveFlatFilesClient
from .history import load_target_with_history
from .market_dataset import build_market_event_dataset, save_dataset
from .massive_client import MassiveClient
from .multi_day import build_multi_day_dataset


CACHE_DIR = Path("data/cache/massive")


def _client(request_interval: float = 0.0) -> MassiveClient:
    settings = load_settings()
    return MassiveClient(
        settings.massive_api_key,
        cache_dir=CACHE_DIR,
        request_interval_seconds=request_interval,
    )


def check_api() -> int:
    client = _client()
    payload = client.previous_close("AAPL")
    print(json.dumps(payload, indent=2))
    return 0


def check_flatfiles(day: date) -> int:
    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    client = MassiveFlatFilesClient(access_key, secret_key)
    objects = client.check_stock_aggregate_access(day)
    print("Massive Flat Files access: OK")
    for obj in objects:
        size_text = "unknown" if obj.size_bytes is None else str(obj.size_bytes)
        print(f"  {obj.key} size_bytes={size_text}")
    return 0


def event_study(ticker: str, day: date) -> int:
    client = _client()
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
    annotate_halts: bool,
) -> int:
    client = _client(request_interval)
    result = build_market_event_dataset(
        client,
        day,
        min_price=min_price,
        max_price=max_price,
        min_high_return_pct=min_high_return,
        min_day_dollar_volume=min_dollar_volume,
        max_candidates=max_candidates,
        annotate_halts=annotate_halts,
    )
    if result.empty:
        print("No qualifying momentum events found.")
        return 0
    output_path = output or Path("data/events") / f"market_events_{day.isoformat()}.parquet"
    save_dataset(result, output_path)
    print(f"Saved {len(result)} event rows across {result['ticker'].nunique()} tickers to {output_path}")
    print(f"Massive stats: {client.stats.to_dict()}")
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
    annotate_halts: bool,
    checkpoint_dir: Path,
    manifest: Path,
) -> int:
    client = _client(request_interval)
    result, skipped = build_multi_day_dataset(
        client,
        start,
        end,
        min_price=min_price,
        max_price=max_price,
        min_high_return_pct=min_high_return,
        min_day_dollar_volume=min_dollar_volume,
        max_candidates=max_candidates,
        annotate_halts=annotate_halts,
        checkpoint_dir=checkpoint_dir,
        manifest_path=manifest,
    )

    if not result.empty:
        output_path = output or Path("data/events") / f"market_events_{start.isoformat()}_{end.isoformat()}.parquet"
        save_dataset(result, output_path)
        print(
            f"Saved {len(result)} event rows across {result['ticker'].nunique()} tickers "
            f"and {result['trading_day'].nunique()} trading days to {output_path}"
        )
    else:
        print("No qualifying momentum events found in the requested range.")

    if skipped:
        print(f"Skipped {len(skipped)} closed/error calendar days.")
        for skipped_day, reason in skipped[:10]:
            print(f"  {skipped_day}: {reason}")
    print(f"Manifest: {manifest}")
    print(f"Massive stats: {client.stats.to_dict()}")
    return 0


def audit_dataset(path: Path, allow_debug_sample: bool) -> int:
    frame = load_event_dataset(path)
    result = audit_event_dataset(frame, require_full_universe=not allow_debug_sample)
    print(result.render())
    return 0 if result.ok else 1


def _print_section(title: str, frame) -> None:
    if frame is not None and not frame.empty:
        print(f"\n=== {title} ===")
        print(frame.to_string(index=False))


def analyze_dataset(path: Path, horizon: int) -> int:
    frame = load_event_dataset(path)
    if frame.empty:
        print("Dataset is empty.")
        return 0

    _print_section("Executable continuation by threshold", summarize_by_threshold(frame))
    _print_section(
        f"Gross vs execution-friction scenarios at +{horizon}m",
        summarize_cost_scenarios(frame, horizon_min=horizon),
    )
    _print_section(
        "Cluster-aware uncertainty (primary base endpoint)",
        summarize_cluster_uncertainty(frame, horizon_min=horizon, scenario="base"),
    )
    _print_section(
        "Tail-risk diagnostics (primary base endpoint)",
        summarize_tail_risk(frame, horizon_min=horizon, scenario="base"),
    )
    _print_section(
        "Unresolved entered-exposure sensitivity",
        summarize_unresolved_exposure(frame, horizon_min=horizon, scenario="base"),
    )
    _print_section(
        "Ticker-day concentration diagnostics",
        summarize_concentration(frame, horizon_min=horizon, scenario="base"),
    )
    _print_section(
        "Monthly stability",
        summarize_monthly_stability(frame, horizon_min=horizon, scenario="base"),
    )
    _print_section(
        "Prior-close price buckets",
        summarize_price_buckets(frame, horizon_min=horizon, scenario="base"),
    )
    _print_section(
        "Intraday time buckets",
        summarize_time_buckets(frame, horizon_min=horizon, scenario="base"),
    )
    _print_section(
        "Entry-latency sensitivity",
        summarize_latency(frame, horizon_min=horizon, scenario="base"),
    )
    _print_section(
        "Session labels",
        summarize_sessions(frame, horizon_min=horizon, scenario="base"),
    )
    _print_section(
        "Entry/outcome availability diagnostics",
        summarize_missingness(frame, horizon_min=horizon),
    )
    _print_section("MFE / MAE by threshold", summarize_excursions(frame))
    _print_section("Short TP / SL first-hit experiments", summarize_barriers(frame))

    if "rvol_cumulative_20d" in frame.columns:
        rvol = summarize_rvol(frame, horizon_min=horizon)
        if rvol.empty:
            print(f"\n=== RVOL buckets at +{horizon}m ===")
            print("No rows with historical RVOL yet.")
        else:
            _print_section(f"RVOL buckets at +{horizon}m", rvol)
    return 0


def _add_dataset_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-price", type=float, default=0.50)
    parser.add_argument("--max-price", type=float, default=20.0)
    parser.add_argument("--min-high-return", type=float, default=10.0)
    parser.add_argument(
        "--min-dollar-volume",
        type=float,
        default=0.0,
        help="Reserved compatibility option. Values >0 are rejected because completed-day liquidity leaks future data.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Debug-only deterministic subsample. Leave unset for unbiased full research.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=12.5,
        help="Minimum seconds between real Massive network requests. Cache hits return immediately.",
    )
    parser.add_argument(
        "--annotate-halts",
        action="store_true",
        help="Attach official Nasdaq Trader halt records when the public feed is available.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="moneymaker")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-api", help="Check Massive API connectivity using AAPL previous close")

    flatfiles = sub.add_parser(
        "check-flatfiles",
        help="Verify Massive S3 Flat Files credentials without downloading data",
    )
    flatfiles.add_argument("--day", type=date.fromisoformat, default=date(2026, 4, 1))

    study = sub.add_parser("event-study", help="Run the threshold event study for one ticker/day")
    study.add_argument("ticker", help="U.S. equity ticker, e.g. AAPL")
    study.add_argument("day", type=date.fromisoformat, help="Trading day in YYYY-MM-DD format")

    dataset = sub.add_parser("market-dataset", help="Build one trading day's market-wide dataset")
    dataset.add_argument("day", type=date.fromisoformat)
    _add_dataset_options(dataset)

    multi = sub.add_parser("multi-day-dataset", help="Build/resume a market event dataset across a date range")
    multi.add_argument("start", type=date.fromisoformat)
    multi.add_argument("end", type=date.fromisoformat)
    _add_dataset_options(multi)
    multi.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints"))
    multi.add_argument("--manifest", type=Path, default=Path("artifacts/run-manifest.json"))

    audit = sub.add_parser("audit-dataset", help="Fail if a research dataset violates integrity invariants")
    audit.add_argument("path", type=Path)
    audit.add_argument("--allow-debug-sample", action="store_true")

    analyze = sub.add_parser("analyze-dataset", help="Summarize executable edge and robustness diagnostics")
    analyze.add_argument("path", type=Path)
    analyze.add_argument("--horizon", type=int, default=5)

    args = parser.parse_args()
    if args.command == "check-api":
        return check_api()
    if args.command == "check-flatfiles":
        return check_flatfiles(args.day)
    if args.command == "event-study":
        return event_study(args.ticker, args.day)
    if args.command == "market-dataset":
        return market_dataset(
            args.day,
            args.output,
            args.min_price,
            args.max_price,
            args.min_high_return,
            args.min_dollar_volume,
            args.max_candidates,
            args.request_interval,
            args.annotate_halts,
        )
    if args.command == "multi-day-dataset":
        return multi_day_dataset(
            args.start,
            args.end,
            args.output,
            args.min_price,
            args.max_price,
            args.min_high_return,
            args.min_dollar_volume,
            args.max_candidates,
            args.request_interval,
            args.annotate_halts,
            args.checkpoint_dir,
            args.manifest,
        )
    if args.command == "audit-dataset":
        return audit_dataset(args.path, args.allow_debug_sample)
    if args.command == "analyze-dataset":
        return analyze_dataset(args.path, args.horizon)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
