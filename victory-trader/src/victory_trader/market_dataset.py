from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import load_settings
from .event_study import DEFAULT_HORIZONS, run_event_study
from .market_data import bars_from_massive_payload
from .massive_client import MassiveClient


@dataclass(frozen=True)
class Candidate:
    ticker: str
    previous_close: float
    day_high: float
    day_close: float
    day_volume: float
    day_vwap: float | None
    high_return_pct: float
    day_dollar_volume: float


def _market_rows(payload: dict) -> list[dict]:
    return list(payload.get("results") or [])


def _ticker_map(payload: dict) -> dict[str, dict]:
    rows = _market_rows(payload)
    return {
        str(row.get("T", "")).upper(): row
        for row in rows
        if row.get("T")
    }


def select_candidates(
    previous_payload: dict,
    target_payload: dict,
    *,
    min_price: float = 0.50,
    max_price: float = 20.0,
    min_high_return_pct: float = 20.0,
    min_day_dollar_volume: float = 0.0,
) -> list[Candidate]:
    """Select low-priced symbols that actually produced a large intraday move.

    Price eligibility is based on the prior trading day's adjusted close. This
    prevents a stock that started above our research universe from entering only
    because it crashed into the range during the target session.
    """
    if min_price <= 0 or max_price <= min_price:
        raise ValueError("price bounds must satisfy 0 < min_price < max_price")

    previous = _ticker_map(previous_payload)
    target = _ticker_map(target_payload)
    candidates: list[Candidate] = []

    for ticker, today in target.items():
        prior = previous.get(ticker)
        if prior is None:
            continue

        try:
            previous_close = float(prior["c"])
            day_high = float(today["h"])
            day_close = float(today["c"])
            day_volume = float(today.get("v", 0.0) or 0.0)
        except (KeyError, TypeError, ValueError):
            continue

        if not (min_price <= previous_close <= max_price):
            continue
        if previous_close <= 0:
            continue

        high_return_pct = (day_high / previous_close - 1.0) * 100.0
        if high_return_pct < min_high_return_pct:
            continue

        raw_vwap = today.get("vw")
        day_vwap = float(raw_vwap) if raw_vwap is not None else None
        reference_price = day_vwap if day_vwap is not None else day_close
        day_dollar_volume = reference_price * day_volume
        if day_dollar_volume < min_day_dollar_volume:
            continue

        candidates.append(
            Candidate(
                ticker=ticker,
                previous_close=previous_close,
                day_high=day_high,
                day_close=day_close,
                day_volume=day_volume,
                day_vwap=day_vwap,
                high_return_pct=high_return_pct,
                day_dollar_volume=day_dollar_volume,
            )
        )

    return sorted(candidates, key=lambda item: item.high_return_pct, reverse=True)


def grouped_daily(client: MassiveClient, day: date) -> dict:
    """Fetch Massive's market-wide daily aggregate without OTC symbols."""
    return client._get(  # centralized auth/error handling lives on MassiveClient
        f"/v2/aggs/grouped/locale/us/market/stocks/{day.isoformat()}",
        {"adjusted": "true", "include_otc": "false"},
    )


def previous_market_payload(
    client: MassiveClient,
    day: date,
    *,
    max_calendar_lookback: int = 10,
    request_interval_seconds: float = 0.0,
) -> tuple[date, dict]:
    """Find the latest prior date for which Massive returns grouped daily data."""
    for offset in range(1, max_calendar_lookback + 1):
        candidate_day = day - timedelta(days=offset)
        payload = grouped_daily(client, candidate_day)
        if _market_rows(payload):
            return candidate_day, payload
        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)
    raise ValueError(f"No prior market summary found within {max_calendar_lookback} days before {day}")


def build_market_event_dataset(
    client: MassiveClient,
    day: date,
    *,
    min_price: float = 0.50,
    max_price: float = 20.0,
    min_high_return_pct: float = 20.0,
    min_day_dollar_volume: float = 0.0,
    thresholds_pct: Iterable[float] = (10, 20, 30, 50, 75, 100),
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    max_candidates: int | None = None,
    request_interval_seconds: float = 12.5,
) -> pd.DataFrame:
    """Build one trading day's market-wide event-study dataset.

    Two cheap grouped-daily requests identify which low-priced stocks could have
    crossed our momentum thresholds. Only those candidates receive expensive
    1-minute requests. This is intentionally compatible with Massive's free-plan
    workflow, although a multi-year study should later use Flat Files.
    """
    target_payload = grouped_daily(client, day)
    if not _market_rows(target_payload):
        raise ValueError(f"No grouped market data returned for {day}")

    if request_interval_seconds > 0:
        time.sleep(request_interval_seconds)
    previous_day, prior_payload = previous_market_payload(
        client,
        day,
        request_interval_seconds=request_interval_seconds,
    )

    candidates = select_candidates(
        prior_payload,
        target_payload,
        min_price=min_price,
        max_price=max_price,
        min_high_return_pct=min_high_return_pct,
        min_day_dollar_volume=min_day_dollar_volume,
    )
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    frames: list[pd.DataFrame] = []
    for candidate in candidates:
        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)

        bars = bars_from_massive_payload(client.minute_bars(candidate.ticker, day))
        if bars.empty:
            continue

        study = run_event_study(
            ticker=candidate.ticker,
            bars=bars,
            previous_close=candidate.previous_close,
            thresholds_pct=thresholds_pct,
            horizons=horizons,
        )
        if study.empty:
            continue

        study.insert(0, "trading_day", day.isoformat())
        study.insert(1, "previous_trading_day", previous_day.isoformat())
        study["day_high"] = candidate.day_high
        study["day_close"] = candidate.day_close
        study["day_volume"] = candidate.day_volume
        study["day_vwap"] = candidate.day_vwap
        study["day_dollar_volume"] = candidate.day_dollar_volume
        study["day_high_return_pct"] = candidate.high_return_pct
        frames.append(study)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["ticker", "timestamp_ms", "threshold_pct"]
    ).reset_index(drop=True)


def save_dataset(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(output, index=False)
    elif suffix == ".csv":
        frame.to_csv(output, index=False)
    else:
        raise ValueError("output must end in .parquet or .csv")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.market_dataset",
        description="Build a market-wide low-priced momentum event dataset for one U.S. trading day.",
    )
    parser.add_argument("day", type=date.fromisoformat, help="Trading day in YYYY-MM-DD format")
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
    args = parser.parse_args()

    settings = load_settings()
    client = MassiveClient(settings.massive_api_key)
    frame = build_market_event_dataset(
        client,
        args.day,
        min_price=args.min_price,
        max_price=args.max_price,
        min_high_return_pct=args.min_high_return,
        min_day_dollar_volume=args.min_dollar_volume,
        max_candidates=args.max_candidates,
        request_interval_seconds=args.request_interval,
    )

    if frame.empty:
        print("No qualifying momentum events found.")
        return 0

    output = args.output or Path("data/events") / f"market_events_{args.day.isoformat()}.parquet"
    save_dataset(frame, output)
    print(f"Saved {len(frame)} event rows across {frame['ticker'].nunique()} tickers to {output}")
    print(frame.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
