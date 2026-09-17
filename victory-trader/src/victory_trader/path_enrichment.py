from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from .config import load_settings, require_flatfile_credentials
from .features import MINUTE_MS, STANDARD_THRESHOLDS
from .flatfiles import MassiveFlatFileStore, MassiveFlatFilesClient, STOCKS_MINUTE_PREFIX


FLATFILE_CACHE_DIR = Path("data/cache/massive-flatfiles")
PATH_FEATURE_COLUMNS = (
    "minutes_since_10pct_cross",
    "minutes_since_prior_threshold_cross",
    "prior_15m_high_distance_pct",
    "prior_15m_low_rebound_pct",
)


def _add_crossing_timing_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["minutes_since_10pct_cross"] = pd.NA
    result["minutes_since_prior_threshold_cross"] = pd.NA

    for (_, _), group in result.groupby(["trading_day", "ticker"], sort=False):
        timestamps = {
            float(row.threshold_pct): int(row.timestamp_ms)
            for row in group[["threshold_pct", "timestamp_ms"]].itertuples(index=False)
        }
        first_10 = timestamps.get(10.0)
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

    for column in (
        "minutes_since_10pct_cross",
        "minutes_since_prior_threshold_cross",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _path_values_for_event(
    bars: pd.DataFrame,
    event_timestamp_ms: int,
    event_price: float,
) -> tuple[float | None, float | None]:
    start = event_timestamp_ms - 15 * MINUTE_MS
    prior = bars.loc[(bars["t"] >= start) & (bars["t"] < event_timestamp_ms)]
    if prior.empty:
        return None, None
    prior_high = float(pd.to_numeric(prior["h"], errors="coerce").max())
    prior_low = float(pd.to_numeric(prior["l"], errors="coerce").min())
    high_distance = (
        (event_price / prior_high - 1.0) * 100.0 if prior_high > 0 else None
    )
    low_rebound = (
        (event_price / prior_low - 1.0) * 100.0 if prior_low > 0 else None
    )
    return high_distance, low_rebound


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
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing path enrichment columns: {sorted(missing)}")

    result = _add_crossing_timing_features(frame)
    result["prior_15m_high_distance_pct"] = pd.NA
    result["prior_15m_low_rebound_pct"] = pd.NA

    for trading_day, day_rows in result.groupby("trading_day", sort=True):
        day = date.fromisoformat(str(trading_day))
        tickers = set(day_rows["ticker"].astype(str).str.upper())
        bars = store.read(STOCKS_MINUTE_PREFIX, day, tickers=tickers)
        if bars.empty:
            continue
        bars = bars.rename(
            columns={
                "volume": "v",
                "open": "o",
                "close": "c",
                "high": "h",
                "low": "l",
            }
        ).copy()
        bars["t"] = (bars["window_start"].astype("int64") // 1_000_000).astype("int64")
        for ticker, ticker_rows in day_rows.groupby("ticker", sort=False):
            ticker_bars = bars.loc[bars["ticker"] == str(ticker).upper(), ["t", "h", "l"]]
            if ticker_bars.empty:
                continue
            for idx in ticker_rows.index:
                high_distance, low_rebound = _path_values_for_event(
                    ticker_bars,
                    int(result.at[idx, "timestamp_ms"]),
                    float(result.at[idx, "signal_price"]),
                )
                result.at[idx, "prior_15m_high_distance_pct"] = high_distance
                result.at[idx, "prior_15m_low_rebound_pct"] = low_rebound

    for column in ("prior_15m_high_distance_pct", "prior_15m_low_rebound_pct"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.path_enrichment",
        description="Append v0.4 point-in-time path features to an existing research dataset.",
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
        f"Enriched {len(result)} existing event rows with path features; "
        f"Flat File stats: {store.stats.to_dict()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
