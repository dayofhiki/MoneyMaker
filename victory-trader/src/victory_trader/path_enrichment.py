from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import load_settings, require_flatfile_credentials
from .features import MINUTE_MS, STANDARD_THRESHOLDS
from .flatfiles import MassiveFlatFileStore, MassiveFlatFilesClient, STOCKS_MINUTE_PREFIX
from .market_calendar import regular_session_bounds


FLATFILE_CACHE_DIR = Path("data/cache/massive-flatfiles")
ET = ZoneInfo("America/New_York")

# These are pre-specified, point-in-time discovery features. They deliberately
# stay outside MODEL_FEATURE_COLUMNS until they survive out-of-sample research.
ENRICHMENT_FEATURE_COLUMNS = (
    "minutes_since_10pct_cross",
    "minutes_since_prior_threshold_cross",
    "prior_15m_high_distance_pct",
    "prior_15m_low_rebound_pct",
    "max_drawdown_since_10pct_pct",
    "signal_bar_range_pct",
    "signal_bar_body_pct",
    "signal_close_location",
    "transactions_5m",
    "dollar_volume_5m",
    "avg_trade_size_5m",
    "active_minute_fraction_15m",
    "regular_open_gap_pct",
    "premarket_high_return_pct",
    "runners_10pct_so_far",
    "runners_10pct_last_30m",
    "iwm_return_since_open_pct",
)


def _add_crossing_timing_and_runner_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "minutes_since_10pct_cross",
        "minutes_since_prior_threshold_cross",
        "runners_10pct_so_far",
        "runners_10pct_last_30m",
    ):
        result[column] = pd.NA

    first_10_by_day: dict[str, list[int]] = {}
    threshold_10 = pd.to_numeric(result["threshold_pct"], errors="coerce").eq(10.0)
    for trading_day, rows in result.loc[threshold_10].groupby("trading_day", sort=False):
        first_10_by_day[str(trading_day)] = sorted(
            pd.to_numeric(rows["timestamp_ms"], errors="coerce").dropna().astype("int64").tolist()
        )

    for (trading_day, _), group in result.groupby(["trading_day", "ticker"], sort=False):
        timestamps = {
            float(row.threshold_pct): int(row.timestamp_ms)
            for row in group[["threshold_pct", "timestamp_ms"]].itertuples(index=False)
        }
        first_10 = timestamps.get(10.0)
        day_crossings = first_10_by_day.get(str(trading_day), [])
        for idx in group.index:
            threshold = float(result.at[idx, "threshold_pct"])
            event_ts = int(result.at[idx, "timestamp_ms"])
            if first_10 is not None:
                result.at[idx, "minutes_since_10pct_cross"] = (
                    event_ts - first_10
                ) / MINUTE_MS
            lowers = [value for value in STANDARD_THRESHOLDS if value < threshold]
            if lowers:
                prior_ts = timestamps.get(max(lowers))
                if prior_ts is not None:
                    result.at[idx, "minutes_since_prior_threshold_cross"] = (
                        event_ts - prior_ts
                    ) / MINUTE_MS
            if day_crossings:
                right = bisect_right(day_crossings, event_ts)
                left = bisect_left(day_crossings, event_ts - 30 * MINUTE_MS)
                result.at[idx, "runners_10pct_so_far"] = right
                result.at[idx, "runners_10pct_last_30m"] = right - left

    for column in (
        "minutes_since_10pct_cross",
        "minutes_since_prior_threshold_cross",
        "runners_10pct_so_far",
        "runners_10pct_last_30m",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _normalize_minute_bars(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["ticker", "t", "o", "h", "l", "c", "v", "n"])
    bars = raw.rename(
        columns={
            "volume": "v",
            "open": "o",
            "close": "c",
            "high": "h",
            "low": "l",
            "transactions": "n",
        }
    ).copy()
    bars["t"] = (bars["window_start"].astype("int64") // 1_000_000).astype("int64")
    for column in ("o", "h", "l", "c", "v", "n"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars.loc[:, ["ticker", "t", "o", "h", "l", "c", "v", "n"]]


def _session_ms(day: date) -> tuple[int, int, int]:
    bounds = regular_session_bounds(day)
    if bounds is None:
        regular_open = datetime.combine(day, time(9, 30), tzinfo=ET)
        regular_close = datetime.combine(day, time(16, 0), tzinfo=ET)
    else:
        regular_open, regular_close = bounds
    premarket_start = datetime.combine(day, time(4, 0), tzinfo=ET)
    return (
        int(premarket_start.timestamp() * 1000),
        int(regular_open.timestamp() * 1000),
        int(regular_close.timestamp() * 1000),
    )


def _market_return_since_open(
    bars: pd.DataFrame,
    event_timestamp_ms: int,
    regular_open_ms: int,
) -> float | None:
    regular = bars.loc[(bars["t"] >= regular_open_ms) & (bars["t"] <= event_timestamp_ms)]
    if regular.empty:
        return None
    first_open = float(regular.iloc[0]["o"])
    event_bar = regular.loc[regular["t"] == event_timestamp_ms]
    if first_open <= 0 or event_bar.empty:
        return None
    last_close = float(event_bar.iloc[-1]["c"])
    return (last_close / first_open - 1.0) * 100.0


def _event_bar_features(
    bars: pd.DataFrame,
    *,
    event_timestamp_ms: int,
    event_price: float,
    previous_close: float,
    first_10_timestamp_ms: int | None,
    premarket_start_ms: int,
    regular_open_ms: int,
) -> dict[str, float | None]:
    event_bar = bars.loc[bars["t"] == event_timestamp_ms]
    if event_bar.empty:
        return {column: None for column in ENRICHMENT_FEATURE_COLUMNS if column not in {
            "minutes_since_10pct_cross",
            "minutes_since_prior_threshold_cross",
            "runners_10pct_so_far",
            "runners_10pct_last_30m",
            "iwm_return_since_open_pct",
        }}

    row = event_bar.iloc[-1]
    o = float(row["o"])
    h = float(row["h"])
    l = float(row["l"])
    c = float(row["c"])
    bar_range = max(h - l, 0.0)
    signal_bar_range_pct = (bar_range / c) * 100.0 if c > 0 else None
    signal_bar_body_pct = ((c / o) - 1.0) * 100.0 if o > 0 else None
    signal_close_location = (c - l) / bar_range if bar_range > 0 else 0.5

    prior_start = event_timestamp_ms - 15 * MINUTE_MS
    prior = bars.loc[(bars["t"] >= prior_start) & (bars["t"] < event_timestamp_ms)]
    if prior.empty:
        prior_high_distance = None
        prior_low_rebound = None
    else:
        prior_high = float(prior["h"].max())
        prior_low = float(prior["l"].min())
        prior_high_distance = (
            (event_price / prior_high - 1.0) * 100.0 if prior_high > 0 else None
        )
        prior_low_rebound = (
            (event_price / prior_low - 1.0) * 100.0 if prior_low > 0 else None
        )

    window_5m = bars.loc[
        (bars["t"] >= event_timestamp_ms - 4 * MINUTE_MS)
        & (bars["t"] <= event_timestamp_ms)
    ]
    transactions_5m = float(window_5m["n"].fillna(0.0).sum()) if not window_5m.empty else None
    dollar_volume_5m = (
        float((window_5m["v"].fillna(0.0) * window_5m["c"].fillna(0.0)).sum())
        if not window_5m.empty
        else None
    )
    volume_5m = float(window_5m["v"].fillna(0.0).sum()) if not window_5m.empty else 0.0
    avg_trade_size_5m = (
        volume_5m / transactions_5m
        if transactions_5m is not None and transactions_5m > 0
        else None
    )

    window_15m = bars.loc[
        (bars["t"] >= event_timestamp_ms - 14 * MINUTE_MS)
        & (bars["t"] <= event_timestamp_ms)
    ]
    active_minute_fraction_15m = min(window_15m["t"].nunique() / 15.0, 1.0)

    regular_to_event = bars.loc[
        (bars["t"] >= regular_open_ms) & (bars["t"] <= event_timestamp_ms)
    ]
    regular_open_gap_pct = None
    if not regular_to_event.empty and previous_close > 0:
        first_regular_open = float(regular_to_event.iloc[0]["o"])
        regular_open_gap_pct = (first_regular_open / previous_close - 1.0) * 100.0

    premarket = bars.loc[
        (bars["t"] >= premarket_start_ms) & (bars["t"] < regular_open_ms)
    ]
    premarket_high_return_pct = None
    if not premarket.empty and previous_close > 0:
        premarket_high = float(premarket["h"].max())
        premarket_high_return_pct = (premarket_high / previous_close - 1.0) * 100.0

    max_drawdown_since_10pct_pct = None
    if first_10_timestamp_ms is not None:
        since_10 = bars.loc[
            (bars["t"] >= first_10_timestamp_ms) & (bars["t"] <= event_timestamp_ms)
        ].copy()
        if not since_10.empty:
            closes = since_10["c"].astype(float)
            running_peak = closes.cummax()
            drawdowns = closes / running_peak - 1.0
            max_drawdown_since_10pct_pct = float(drawdowns.min() * 100.0)

    return {
        "prior_15m_high_distance_pct": prior_high_distance,
        "prior_15m_low_rebound_pct": prior_low_rebound,
        "max_drawdown_since_10pct_pct": max_drawdown_since_10pct_pct,
        "signal_bar_range_pct": signal_bar_range_pct,
        "signal_bar_body_pct": signal_bar_body_pct,
        "signal_close_location": signal_close_location,
        "transactions_5m": transactions_5m,
        "dollar_volume_5m": dollar_volume_5m,
        "avg_trade_size_5m": avg_trade_size_5m,
        "active_minute_fraction_15m": active_minute_fraction_15m,
        "regular_open_gap_pct": regular_open_gap_pct,
        "premarket_high_return_pct": premarket_high_return_pct,
    }


def enrich_path_features(
    frame: pd.DataFrame,
    store: MassiveFlatFileStore,
) -> pd.DataFrame:
    required = {
        "trading_day",
        "ticker",
        "timestamp_ms",
        "threshold_pct",
        "signal_price",
        "previous_close",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing enrichment columns: {sorted(missing)}")

    result = _add_crossing_timing_and_runner_context(frame)
    bar_feature_columns = [
        column
        for column in ENRICHMENT_FEATURE_COLUMNS
        if column not in {
            "minutes_since_10pct_cross",
            "minutes_since_prior_threshold_cross",
            "runners_10pct_so_far",
            "runners_10pct_last_30m",
        }
    ]
    for column in bar_feature_columns:
        result[column] = pd.NA

    for trading_day, day_rows in result.groupby("trading_day", sort=True):
        day = date.fromisoformat(str(trading_day))
        tickers = set(day_rows["ticker"].astype(str).str.upper()) | {"IWM"}
        raw = store.read(STOCKS_MINUTE_PREFIX, day, tickers=tickers)
        bars = _normalize_minute_bars(raw)
        if bars.empty:
            continue
        premarket_start_ms, regular_open_ms, _ = _session_ms(day)

        iwm_bars = bars.loc[bars["ticker"] == "IWM"].sort_values("t")
        first_10_by_ticker = {
            str(ticker).upper(): int(group.loc[pd.to_numeric(group["threshold_pct"], errors="coerce").eq(10.0), "timestamp_ms"].iloc[0])
            for ticker, group in day_rows.groupby("ticker", sort=False)
            if pd.to_numeric(group["threshold_pct"], errors="coerce").eq(10.0).any()
        }

        for ticker, ticker_rows in day_rows.groupby("ticker", sort=False):
            symbol = str(ticker).upper()
            ticker_bars = bars.loc[bars["ticker"] == symbol].sort_values("t")
            if ticker_bars.empty:
                continue
            first_10 = first_10_by_ticker.get(symbol)
            for idx in ticker_rows.index:
                event_ts = int(result.at[idx, "timestamp_ms"])
                values = _event_bar_features(
                    ticker_bars,
                    event_timestamp_ms=event_ts,
                    event_price=float(result.at[idx, "signal_price"]),
                    previous_close=float(result.at[idx, "previous_close"]),
                    first_10_timestamp_ms=first_10,
                    premarket_start_ms=premarket_start_ms,
                    regular_open_ms=regular_open_ms,
                )
                for column, value in values.items():
                    result.at[idx, column] = value
                result.at[idx, "iwm_return_since_open_pct"] = _market_return_since_open(
                    iwm_bars,
                    event_ts,
                    regular_open_ms,
                )

    for column in ENRICHMENT_FEATURE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.path_enrichment",
        description="Append pre-specified point-in-time discovery features to an existing research dataset.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key),
        FLATFILE_CACHE_DIR,
    )
    result = enrich_path_features(frame, store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(
        f"Enriched {len(result)} existing event rows with "
        f"{len(ENRICHMENT_FEATURE_COLUMNS)} point-in-time discovery features; "
        f"Flat File stats: {store.stats.to_dict()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
