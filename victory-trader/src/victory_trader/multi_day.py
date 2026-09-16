from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import load_settings
from .market_dataset import build_market_event_dataset, save_dataset
from .massive_client import MassiveClient


def daterange(start: date, end: date):
    if end < start:
        raise ValueError("end must be on or after start")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_multi_day_dataset(
    client: MassiveClient,
    start: date,
    end: date,
    *,
    continue_on_error: bool = True,
    **day_kwargs,
) -> tuple[pd.DataFrame, list[tuple[date, str]]]:
    """Accumulate daily event datasets across a calendar range.

    Weekends/holidays naturally produce no grouped data and are recorded as
    skipped errors when continue_on_error=True. This keeps the first version
    independent of a separate exchange-calendar dependency.
    """
    frames: list[pd.DataFrame] = []
    skipped: list[tuple[date, str]] = []

    for day in daterange(start, end):
        try:
            frame = build_market_event_dataset(client, day, **day_kwargs)
        except Exception as exc:
            if not continue_on_error:
                raise
            skipped.append((day, f"{type(exc).__name__}: {exc}"))
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(), skipped

    result = pd.concat(frames, ignore_index=True).sort_values(
        ["trading_day", "ticker", "timestamp_ms", "threshold_pct"]
    ).reset_index(drop=True)
    return result, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.multi_day",
        description="Build a momentum event dataset across a date range.",
    )
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-price", type=float, default=0.50)
    parser.add_argument("--max-price", type=float, default=20.0)
    parser.add_argument("--min-high-return", type=float, default=20.0)
    parser.add_argument("--min-dollar-volume", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--request-interval", type=float, default=12.5)
    args = parser.parse_args()

    settings = load_settings()
    client = MassiveClient(settings.massive_api_key)
    frame, skipped = build_multi_day_dataset(
        client,
        args.start,
        args.end,
        min_price=args.min_price,
        max_price=args.max_price,
        min_high_return_pct=args.min_high_return,
        min_day_dollar_volume=args.min_dollar_volume,
        max_candidates=args.max_candidates,
        request_interval_seconds=args.request_interval,
    )

    output = args.output or Path("data/events") / (
        f"market_events_{args.start.isoformat()}_{args.end.isoformat()}.parquet"
    )
    if not frame.empty:
        save_dataset(frame, output)
        print(
            f"Saved {len(frame)} event rows across {frame['ticker'].nunique()} tickers "
            f"and {frame['trading_day'].nunique()} trading days to {output}"
        )
    else:
        print("No qualifying momentum events found in the requested range.")

    if skipped:
        print(f"Skipped {len(skipped)} calendar days (weekends, holidays, or API/data errors).")
        for day, reason in skipped[:10]:
            print(f"  {day}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
