from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .config import load_settings
from .event_study import DEFAULT_HORIZONS, run_event_study
from .halts import annotate_frame_with_halts, fetch_nasdaq_halts
from .history import load_target_with_history
from .market_calendar import previous_us_equity_trading_day
from .massive_client import MassiveClient
from .universe import fetch_security_metadata, split_tickers_from_payload


DATASET_SCHEMA_VERSION = "0.2"


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


@dataclass
class DayBuildStats:
    trading_day: str = ""
    discovered_candidates: int = 0
    debug_selected_candidates: int = 0
    split_excluded: int = 0
    metadata_excluded: int = 0
    no_bars: int = 0
    no_events: int = 0
    event_tickers: int = 0
    event_rows: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return vars(self).copy()


def _market_rows(payload: dict) -> list[dict]:
    return list(payload.get("results") or [])


def _ticker_map(payload: dict) -> dict[str, dict]:
    return {
        str(row.get("T", "")).upper(): row
        for row in _market_rows(payload)
        if row.get("T")
    }


def select_candidates(
    previous_payload: dict,
    target_payload: dict,
    *,
    min_price: float = 0.50,
    max_price: float = 20.0,
    min_high_return_pct: float = 10.0,
    min_day_dollar_volume: float = 0.0,
) -> list[Candidate]:
    """Find historical dates/tickers that could contain the lowest studied event.

    Completed daily high is a discovery accelerator only. It may never rank or
    truncate production candidates. Completed-day dollar-volume filtering is
    intentionally forbidden because it is future information at entry time.
    """
    if min_price <= 0 or max_price <= min_price:
        raise ValueError("price bounds must satisfy 0 < min_price < max_price")
    if min_day_dollar_volume > 0:
        raise ValueError(
            "completed-day dollar-volume filtering is forbidden in point-in-time research; "
            "use event-time cumulative/trailing liquidity features instead"
        )

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
    if max_candidates is None:
        return candidates
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive when provided")

    def rank(candidate: Candidate) -> str:
        token = f"{day.isoformat()}:{candidate.ticker}".encode()
        return hashlib.sha256(token).hexdigest()

    return sorted(candidates, key=rank)[:max_candidates]


def grouped_daily(client: MassiveClient, day: date) -> dict:
    # Nominal, as-traded prices are required for the historical $0.50-$20 universe.
    return client._get(
        f"/v2/aggs/grouped/locale/us/market/stocks/{day.isoformat()}",
        {"adjusted": "false", "include_otc": "false"},
    )


def previous_market_payload(
    client: MassiveClient,
    day: date,
    *,
    max_calendar_lookback: int = 10,
    request_interval_seconds: float = 0.0,
) -> tuple[date, dict]:
    """Load the official prior U.S. equity trading day's grouped summary once."""
    # Kept in the signature for backwards-compatible callers/tests. Calendar
    # lookup replaces provider probing across weekends and holidays.
    del max_calendar_lookback, request_interval_seconds
    previous_day = previous_us_equity_trading_day(day)
    payload = grouped_daily(client, previous_day)
    if not _market_rows(payload):
        raise ValueError(
            f"No grouped market data returned for official previous trading day {previous_day}"
        )
    return previous_day, payload


def build_market_event_dataset(
    client: MassiveClient,
    day: date,
    *,
    min_price: float = 0.50,
    max_price: float = 20.0,
    min_high_return_pct: float = 10.0,
    min_day_dollar_volume: float = 0.0,
    thresholds_pct: Iterable[float] = (10, 20, 30, 50, 75, 100),
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    max_candidates: int | None = None,
    request_interval_seconds: float = 12.5,
    enforce_common_stock: bool = True,
    exclude_split_days: bool = True,
    historical_context_days: int = 35,
    annotate_halts: bool = False,
    build_stats: DayBuildStats | None = None,
) -> pd.DataFrame:
    """Build one day's point-in-time regular-session momentum research dataset."""
    del request_interval_seconds
    thresholds = tuple(sorted({float(value) for value in thresholds_pct}))
    if not thresholds:
        raise ValueError("at least one event threshold is required")
    if min_high_return_pct > min(thresholds):
        raise ValueError(
            "discovery high-return threshold cannot exceed the lowest studied event threshold; "
            "that would create future-selected samples"
        )

    stats = build_stats if build_stats is not None else DayBuildStats()
    stats.trading_day = day.isoformat()

    target_payload = grouped_daily(client, day)
    if not _market_rows(target_payload):
        raise ValueError(f"No grouped market data returned for {day}")

    previous_day, prior_payload = previous_market_payload(client, day)
    candidates = select_candidates(
        prior_payload,
        target_payload,
        min_price=min_price,
        max_price=max_price,
        min_high_return_pct=min_high_return_pct,
        min_day_dollar_volume=min_day_dollar_volume,
    )
    stats.discovered_candidates = len(candidates)
    candidates = limit_candidates_for_debug(candidates, day=day, max_candidates=max_candidates)
    stats.debug_selected_candidates = len(candidates)

    split_tickers: set[str] = set()
    if exclude_split_days and candidates:
        split_tickers = split_tickers_from_payload(client.splits_on(day))

    frames: list[pd.DataFrame] = []
    for candidate in candidates:
        if candidate.ticker in split_tickers:
            stats.split_excluded += 1
            continue

        metadata = None
        if enforce_common_stock:
            metadata = fetch_security_metadata(client, candidate.ticker, day)
            if not metadata.is_research_common_stock:
                stats.metadata_excluded += 1
                continue

        bars, history_bars = load_target_with_history(
            client,
            candidate.ticker,
            day,
            calendar_lookback_days=historical_context_days,
        )
        if bars.empty:
            stats.no_bars += 1
            continue

        study = run_event_study(
            ticker=candidate.ticker,
            bars=bars,
            previous_close=candidate.previous_close,
            thresholds_pct=thresholds,
            horizons=horizons,
            history_bars=history_bars,
            event_session_scope="regular",
            require_regular_entry=True,
        )
        if study.empty:
            stats.no_events += 1
            continue

        study.insert(0, "trading_day", day.isoformat())
        study.insert(1, "previous_trading_day", previous_day.isoformat())
        study["dataset_schema_version"] = DATASET_SCHEMA_VERSION
        study["source_prices_adjusted"] = False
        study["discovery_high_return_threshold_pct"] = float(min_high_return_pct)
        study["discovered_candidate_count"] = stats.discovered_candidates
        study["debug_candidate_limit"] = max_candidates
        study["split_day"] = False
        if metadata is not None:
            # Identity/taxonomy only. Provider fundamentals such as historical market
            # cap/shares are deliberately excluded until publication-time semantics
            # are guaranteed for model use.
            study["security_type"] = metadata.security_type
            study["primary_exchange"] = metadata.primary_exchange
        frames.append(study)
        stats.event_tickers += 1
        stats.event_rows += len(study)

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
    parser.add_argument("--min-high-return", type=float, default=10.0)
    parser.add_argument("--min-dollar-volume", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--request-interval", type=float, default=12.5)
    parser.add_argument("--history-days", type=int, default=35)
    parser.add_argument("--annotate-halts", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=args.request_interval,
    )
    frame = build_market_event_dataset(
        client,
        args.day,
        min_price=args.min_price,
        max_price=args.max_price,
        min_high_return_pct=args.min_high_return,
        min_day_dollar_volume=args.min_dollar_volume,
        max_candidates=args.max_candidates,
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
