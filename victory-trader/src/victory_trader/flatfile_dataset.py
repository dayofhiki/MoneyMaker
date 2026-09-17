from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

import pandas as pd

from .event_study import DEFAULT_HORIZONS, run_event_study
from .flatfiles import (
    STOCKS_DAY_PREFIX,
    STOCKS_MINUTE_PREFIX,
    MassiveFlatFileStore,
)
from .halts import annotate_frame_with_halts, fetch_nasdaq_halts
from .market_calendar import is_us_equity_trading_day, previous_us_equity_trading_day
from .market_dataset import (
    DayBuildStats,
    limit_candidates_for_debug,
    select_candidates,
)
from .massive_client import MassiveClient
from .universe import fetch_research_universe_metadata, split_tickers_from_payload


FLATFILE_DATASET_SCHEMA_VERSION = "0.3"


def _history_days(target_day: date, calendar_lookback_days: int) -> list[date]:
    start = target_day - timedelta(days=calendar_lookback_days)
    days: list[date] = []
    current = start
    while current < target_day:
        if is_us_equity_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def _group_by_ticker(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if frame.empty:
        return {}
    return {
        str(ticker).upper(): group.drop(columns=["ticker"]).sort_values("t").reset_index(drop=True)
        for ticker, group in frame.groupby("ticker", sort=False)
    }


def build_flatfile_market_event_dataset(
    rest_client: MassiveClient,
    store: MassiveFlatFileStore,
    day: date,
    *,
    min_price: float = 0.50,
    max_price: float = 20.0,
    min_high_return_pct: float = 10.0,
    min_day_dollar_volume: float = 0.0,
    thresholds_pct: Iterable[float] = (10, 20, 30, 50, 75, 100),
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    max_candidates: int | None = None,
    enforce_common_stock: bool = True,
    exclude_split_days: bool = True,
    historical_context_days: int = 35,
    annotate_halts: bool = False,
    build_stats: DayBuildStats | None = None,
) -> pd.DataFrame:
    """Build one formal research day from market-wide Massive Flat Files.

    Day aggregates perform only coarse historical candidate discovery. One market-wide
    minute file is then queried locally for every candidate, while historical minute
    files are reused from the local Parquet cache for point-in-time RVOL features.
    """
    thresholds = tuple(sorted({float(value) for value in thresholds_pct}))
    if not thresholds:
        raise ValueError("at least one event threshold is required")
    if min_high_return_pct > min(thresholds):
        raise ValueError(
            "discovery high-return threshold cannot exceed the lowest studied event threshold"
        )

    stats = build_stats if build_stats is not None else DayBuildStats()
    stats.trading_day = day.isoformat()

    previous_day = previous_us_equity_trading_day(day)
    target_payload = store.grouped_daily_payload(day)
    previous_payload = store.grouped_daily_payload(previous_day)
    candidates = select_candidates(
        previous_payload,
        target_payload,
        min_price=min_price,
        max_price=max_price,
        min_high_return_pct=min_high_return_pct,
        min_day_dollar_volume=min_day_dollar_volume,
    )
    stats.discovered_candidates = len(candidates)
    candidates = limit_candidates_for_debug(
        candidates,
        day=day,
        max_candidates=max_candidates,
    )
    stats.debug_selected_candidates = len(candidates)

    split_tickers: set[str] = set()
    if exclude_split_days and candidates:
        split_tickers = split_tickers_from_payload(rest_client.splits_on(day))

    metadata_by_ticker = None
    if enforce_common_stock and candidates:
        metadata_by_ticker = fetch_research_universe_metadata(rest_client, day)

    eligible = []
    for candidate in candidates:
        if candidate.ticker in split_tickers:
            stats.split_excluded += 1
            continue
        metadata = None
        if enforce_common_stock:
            metadata = metadata_by_ticker.get(candidate.ticker) if metadata_by_ticker else None
            if metadata is None or not metadata.is_research_common_stock:
                stats.metadata_excluded += 1
                continue
        eligible.append((candidate, metadata))

    if not eligible:
        return pd.DataFrame()

    ticker_set = {candidate.ticker for candidate, _ in eligible}
    target_frame = store.minute_aggregates(day, tickers=ticker_set)
    target_by_ticker = _group_by_ticker(target_frame)

    history_frames: list[pd.DataFrame] = []
    for history_day in _history_days(day, historical_context_days):
        history = store.minute_aggregates(history_day, tickers=ticker_set)
        if not history.empty:
            history_frames.append(history)
    history_all = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame(columns=target_frame.columns)
    )
    history_by_ticker = _group_by_ticker(history_all)

    frames: list[pd.DataFrame] = []
    for candidate, metadata in eligible:
        bars = target_by_ticker.get(candidate.ticker)
        if bars is None or bars.empty:
            stats.no_bars += 1
            continue
        history_bars = history_by_ticker.get(candidate.ticker)
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
        study["dataset_schema_version"] = FLATFILE_DATASET_SCHEMA_VERSION
        study["source_prices_adjusted"] = False
        study["market_data_backend"] = "massive_flatfiles"
        study["flatfile_day_dataset"] = STOCKS_DAY_PREFIX
        study["flatfile_minute_dataset"] = STOCKS_MINUTE_PREFIX
        study["bar_vwap_source"] = "volume_weighted_bar_close_proxy"
        study["discovery_high_return_threshold_pct"] = float(min_high_return_pct)
        study["discovered_candidate_count"] = stats.discovered_candidates
        study["debug_candidate_limit"] = max_candidates
        study["split_day"] = False
        if metadata is not None:
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
