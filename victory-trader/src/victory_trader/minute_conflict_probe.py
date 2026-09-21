"""Read-only diagnostic for conflicting historical minute aggregate rows."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, MINUTE_VALUE_COLUMNS
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .market_calendar import regular_session_bounds
from .massive_client import MassiveClient


def run_probe(
    store: MassiveFlatFileStore,
    rest_client: MassiveClient,
    *,
    ticker: str,
    day: date,
) -> dict[str, object]:
    symbol = ticker.strip().upper()
    minutes = store.minute_aggregates(day, tickers={symbol}).copy()
    minutes["ticker"] = minutes["ticker"].astype(str).str.strip().str.upper()
    minutes = minutes.loc[minutes["ticker"].eq(symbol)].copy()

    bounds = regular_session_bounds(day)
    if bounds is None:
        raise ValueError(f"{day} is not a US equity trading day")
    open_ms = int(bounds[0].timestamp() * 1000)
    close_ms = int(bounds[1].timestamp() * 1000)

    minutes["t"] = pd.to_numeric(minutes["t"], errors="coerce")
    minutes = minutes.loc[
        minutes["t"].between(open_ms, close_ms - 1, inclusive="both")
    ].copy()
    minutes["t"] = minutes["t"].astype("int64")

    duplicate = minutes.duplicated(["ticker", "t"], keep=False)
    duplicate_rows = minutes.loc[duplicate].copy()

    payload = rest_client.minute_bars(symbol, day, adjusted=False)
    rest = pd.DataFrame(payload.get("results") or [])
    if not rest.empty:
        rest["t"] = pd.to_numeric(rest["t"], errors="coerce")
        rest = rest.loc[
            rest["t"].between(open_ms, close_ms - 1, inclusive="both")
        ].copy()
        rest["t"] = rest["t"].astype("int64")
    rest_by_time = rest.set_index("t", drop=False) if not rest.empty else rest

    conflicts: list[dict[str, object]] = []
    for timestamp, group in duplicate_rows.groupby("t", sort=True):
        distinct = group.loc[:, list(MINUTE_VALUE_COLUMNS)].nunique(dropna=False)
        if not distinct.gt(1).any():
            continue
        reference = None
        if not rest.empty and int(timestamp) in rest_by_time.index:
            row = rest_by_time.loc[int(timestamp)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            reference = {
                column: (
                    None
                    if column not in row or pd.isna(row[column])
                    else float(row[column])
                )
                for column in (*MINUTE_VALUE_COLUMNS, "n")
            }
        candidates = []
        for _, row in group.iterrows():
            candidates.append(
                {
                    column: (
                        None
                        if column not in row or pd.isna(row[column])
                        else float(row[column])
                    )
                    for column in (*MINUTE_VALUE_COLUMNS, "n")
                }
            )
        conflicts.append(
            {
                "t": int(timestamp),
                "candidates": candidates,
                "rest": reference,
            }
        )

    return {
        "ticker": symbol,
        "day": day.isoformat(),
        "regular_rows": int(len(minutes)),
        "duplicate_rows": int(duplicate.sum()),
        "conflicting_timestamps": len(conflicts),
        "conflicts": conflicts,
        "flatfile_stats": store.stats.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key),
        FLATFILE_CACHE_DIR,
    )
    rest_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=0.0,
    )
    result = run_probe(
        store, rest_client, ticker=args.ticker, day=args.day
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
