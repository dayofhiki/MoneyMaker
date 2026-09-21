"""Massive Flat Files adapter for chronological market attention replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from .attention_replay import (
    build_market_scan_frame,
    replay_attention,
    summarize_attention_replay,
)
from .attention_runtime import AttentionConfig
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .market_calendar import (
    is_us_equity_trading_day,
    previous_us_equity_trading_day,
    regular_session_bounds,
)
from .massive_client import MassiveClient
from .multi_day import daterange
from .universe import (
    fetch_research_universe_metadata,
    split_tickers_from_payload,
)

FLATFILE_CACHE_DIR = Path("data/cache/massive-flatfiles")
REST_CACHE_DIR = Path("data/cache/massive")


def build_flatfile_scan_day(
    store: MassiveFlatFileStore,
    rest_client: MassiveClient,
    day: date,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Build one causal broad-scan day without completed-day selection."""

    bounds = regular_session_bounds(day)
    if bounds is None:
        return pd.DataFrame(), {
            "trading_day": day.isoformat(),
            "kind": "closed_market",
        }
    open_ms = int(bounds[0].timestamp() * 1000)
    close_ms = int(bounds[1].timestamp() * 1000)

    previous_day = previous_us_equity_trading_day(day)
    previous = store.day_aggregates(previous_day)
    minutes = store.minute_aggregates(day)
    metadata = fetch_research_universe_metadata(rest_client, day)
    split_tickers = split_tickers_from_payload(rest_client.splits_on(day))
    eligible = {
        ticker
        for ticker, item in metadata.items()
        if item.is_research_common_stock and ticker not in split_tickers
    }

    prior = previous.loc[
        previous["ticker"].astype(str).str.upper().isin(eligible),
        ["ticker", "close"],
    ].rename(columns={"close": "previous_close"})
    prior["trading_day"] = day.isoformat()
    prior["eligible"] = True

    minutes = minutes.loc[
        minutes["ticker"].astype(str).str.upper().isin(eligible)
        & pd.to_numeric(minutes["t"], errors="coerce").between(
            open_ms,
            close_ms - 1,
            inclusive="both",
        )
    ].copy()
    minutes["trading_day"] = day.isoformat()

    scan = build_market_scan_frame(minutes, prior)
    return scan, {
        "trading_day": day.isoformat(),
        "kind": "scan",
        "previous_trading_day": previous_day.isoformat(),
        "point_in_time_common_stocks": len(metadata),
        "eligible_after_exchange_type_split": len(eligible),
        "same_day_split_exclusions": len(split_tickers),
        "regular_minute_input_rows": len(minutes),
        "scan_rows": len(scan),
        "scan_symbols": int(scan["ticker"].nunique()) if not scan.empty else 0,
        "runner_crossings": (
            int(scan["runner_cross_now"].sum()) if not scan.empty else 0
        ),
    }


def run_flatfile_attention_replay(
    store: MassiveFlatFileStore,
    rest_client: MassiveClient,
    start: date,
    end: date,
    *,
    config: AttentionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    scans: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    days: list[dict[str, int | str]] = []
    runtime_config = config or AttentionConfig()

    for day in daterange(start, end):
        if not is_us_equity_trading_day(day):
            days.append(
                {"trading_day": day.isoformat(), "kind": "closed_market"}
            )
            continue
        scan, day_summary = build_flatfile_scan_day(store, rest_client, day)
        days.append(day_summary)
        if scan.empty:
            continue
        scans.append(scan)
        traces.append(replay_attention(scan, runtime_config))

    scan_frame = pd.concat(scans, ignore_index=True) if scans else pd.DataFrame()
    trace_frame = (
        pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    )
    replay_summary = (
        summarize_attention_replay(trace_frame, scan_frame)
        if not trace_frame.empty
        else {"trace_rows": 0, "runner_episodes": 0}
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "attention_config": asdict(runtime_config),
        "score_definition": (
            "same-timestamp cross-sectional percentile of nominal return "
            "from prior close"
        ),
        "days": days,
        "replay": replay_summary,
        "flatfile_stats": store.stats.to_dict(),
    }
    return scan_frame, trace_frame, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.attention_flatfile_replay",
        description="Replay hierarchical attention on market-wide Massive minute files.",
    )
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--scan-output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key),
        FLATFILE_CACHE_DIR,
    )
    rest_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=REST_CACHE_DIR,
        request_interval_seconds=0.0,
    )
    scan, trace, summary = run_flatfile_attention_replay(
        store,
        rest_client,
        args.start,
        args.end,
    )
    for path in (args.scan_output, args.trace_output, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    scan.to_parquet(args.scan_output, index=False, compression="zstd")
    trace.to_parquet(args.trace_output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
