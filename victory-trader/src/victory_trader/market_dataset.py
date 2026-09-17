from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import load_settings
from .event_study import DEFAULT_HORIZONS, run_event_study
from .halts import annotate_frame_with_halts, fetch_nasdaq_halts
from .history import load_target_with_history
from .massive_client import MassiveClient
from .universe import fetch_security_metadata, split_tickers_from_payload


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


def _sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def select_candidates(
    previous_payload: dict,
    target_payload: dict,
    *,
    min_price: float = 0.50,
    max_price: float = 20.0,
    min_high_return_pct: float = 20.0,
    min_day_dollar_volume: float = 0.0,
) -> list[Candidate]:
    """Find historical days that contained a qualifying momentum event.

    The completed daily high is used only as an efficient historical discovery
    filter. The returned order is ticker order, never outcome rank, so callers
    cannot accidentally select the day's biggest eventual winners merely by
    taking the first N rows.
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

    return sorted(candidates, key=lambda item: item.ticker)


def limit_candidates_for_debug(
    candidates: list[Candidate],
    *,
    day: date,
    max_candidates: int | None,
) -> list[Candidate]:
    """Deterministically subsample candidates without looking at their outcome.

    Production research should leave ``max_candidates`` unset and process the
    full qualifying universe. This helper exists only for smoke/debug runs. The
    ordering is a stable hash of trading day + ticker and does not use day high,
    close, volume, or any forward return.
    """
    if max_candidates is None:
        return candidates
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive when provided")

    def rank(candidate: Candidate) -> str:
        token = f"{day.isoformat()}:{candidate.ticker}".encode("utf-8")
        return hashlib.sha256(token).hexdigest()

    return sorted(candidates, key=rank)[:max_candidates]


def grouped_daily(client: MassiveClient, day: date) -> dict:
    return client._get(
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
    for offset in range(1, max_calendar_lookback + 1):
        candidate_day = day - timedelta(days=offset)
        payload = grouped_daily(client, candidate_day)
        if _market_rows(payload):
            return candidate_day, payload
        _sleep(request_interval_seconds)
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
    enforce_common_stock: bool = True,
    exclude_split_days: bool = True,
    historical_context_days: int = 35,
    annotate_halts: bool = False,
) -> pd.DataFrame:
    """Build one day's point-in-time momentum research dataset.

    Completed daily summaries are used only to discover historical event days.
    Event features themselves are point-in-time. ``max_candidates`` is a debug
    subsample only; full research should leave it unset. Halt annotations are
    optional because Nasdaq's public feed has independent availability/coverage.
    """
    target_payload = grouped_daily(client, day)
    if not _market_rows(target_payload):
        raise ValueError(f"No grouped market data returned for {day}")

    _sleep(request_interval_seconds)
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
    discovered_candidate_count = len(candidates)
    candidates = limit_candidates_for_debug(
        candidates,
        day=day,
        max_candidates=max_candidates,
    )

    split_tickers: set[str] = set()
    if exclude_split_days and candidates:
        _sleep(request_interval_seconds)
        split_tickers = split_tickers_from_payload(client.splits_on(day))

    frames: list[pd.DataFrame] = []
    for candidate in candidates:
        if candidate.ticker in split_tickers:
            continue

        metadata = None
        if enforce_common_stock:
            _sleep(request_interval_seconds)
            metadata = fetch_security_metadata(client, candidate.ticker, day)
            if not metadata.is_research_common_stock:
                continue

        _sleep(request_interval_seconds)
        bars, history_bars = load_target_with_history(
            client,
            candidate.ticker,
            day,
            calendar_lookback_days=historical_context_days,
        )
        if bars.empty:
            continue

        study = run_event_study(
            ticker=candidate.ticker,
            bars=bars,
            previous_close=candidate.previous_close,
            thresholds_pct=thresholds_pct,
            horizons=horizons,
            history_bars=history_bars,
        )
        if study.empty:
            continue

        study.insert(0, "trading_day", day.isoformat())
        study.insert(1, "previous_trading_day", previous_day.isoformat())
        study["discovered_candidate_count"] = discovered_candidate_count
        study["debug_candidate_limit"] = max_candidates
        study["day_high"] = candidate.day_high
        study["day_close"] = candidate.day_close
        study["day_volume"] = candidate.day_volume
        study["day_vwap"] = candidate.day_vwap
        study["day_dollar_volume"] = candidate.day_dollar_volume
        study["day_high_return_pct"] = candidate.high_return_pct
        study["split_day"] = False
        if metadata is not None:
            study["security_type"] = metadata.security_type
            study["primary_exchange"] = metadata.primary_exchange
            study["market_cap"] = metadata.market_cap
            study["shares_outstanding"] = metadata.shares_outstanding
        frames.append(study)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True).sort_values(
        ["ticker", "timestamp_ms", "threshold_pct"]
    ).reset_index(drop=True)

    if annotate_halts:
        try:
            halt_records = fetch_nasdaq_halts(day)
        except Exception as exc:
            result["halt_data_available"] = False
            result["halt_annotation_error"] = f"{type(exc).__name__}: {exc}"
        else:
            result = annotate_frame_with_halts(result, halt_records)
            result["halt_data_available"] = True
            result["halt_annotation_error"] = None
    return result


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
    parser.add_argument("--request-interval", type=float, default=12.5)
    parser.add_argument("--history-days", type=int, default=35)
    parser.add_argument("--annotate-halts", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    client = MassiveClient(settings.massive_api_key, cache_dir=Path("data/cache/massive"))
    frame = build_market_event_dataset(
        client,
        args.day,
        min_price=args.min_price,
        max_price=args.max_price,
        min_high_return_pct=args.min_high_return,
        min_day_dollar_volume=args.min_dollar_volume,
        max_candidates=args.max_candidates,
        request_interval_seconds=args.request_interval,
        historical_context_days=args.history_days,
        annotate_halts=args.annotate_halts,
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
