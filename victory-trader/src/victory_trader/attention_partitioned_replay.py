"""Day-partitioned Massive Flat Files attention replay.

This module preserves the causal replay semantics of attention_flatfile_replay
while bounding memory to one trading session at a time. Large development
windows are written immediately as per-day Parquet partitions instead of
concatenating all scan and trace rows in memory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .attention_flatfile_replay import (
    FLATFILE_CACHE_DIR,
    REST_CACHE_DIR,
    build_flatfile_scan_day,
)
from .attention_replay import MINUTE_MS, replay_attention, summarize_attention_replay
from .attention_runtime import AttentionConfig, AttentionState
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .market_calendar import is_us_equity_trading_day
from .massive_client import MassiveClient
from .multi_day import daterange

WATCH_OR_BETTER = {
    AttentionState.WATCH.value,
    AttentionState.HOT.value,
    AttentionState.POSITION.value,
}
HOT_OR_POSITION = {
    AttentionState.HOT.value,
    AttentionState.POSITION.value,
}
ADDITIVE_REPLAY_FIELDS = (
    "trace_rows",
    "sessions",
    "runner_episodes",
    "runner_watch_or_better_by_cross",
    "runner_hot_or_position_by_cross",
    "state_changes",
    "grouped_minute_requests",
    "minute_bar_requests",
    "second_bar_requests",
    "trade_nbbo_requests",
    "dropped_rows",
)


def _runner_leads(
    trace: pd.DataFrame,
    scan_frame: pd.DataFrame,
    states: set[str],
) -> list[float]:
    """Return exact per-runner lead minutes for one session."""

    if "runner_cross_now" not in scan_frame.columns:
        return []
    crossings = scan_frame.loc[
        scan_frame["runner_cross_now"].fillna(False).astype(bool),
        ["trading_day", "ticker", "t"],
    ]
    leads: list[float] = []
    for row in crossings.itertuples(index=False):
        eligible = trace.loc[
            trace["trading_day"].eq(str(row.trading_day))
            & trace["ticker"].eq(str(row.ticker).upper())
            & trace["t"].le(int(row.t))
            & trace["state"].isin(states)
        ]
        if eligible.empty:
            continue
        first_t = int(eligible["t"].min())
        leads.append((int(row.t) - first_t) / MINUTE_MS)
    return leads


def _aggregate_replay(
    day_replays: list[dict[str, int | float | None]],
    unique_symbols: set[str],
    watch_leads: list[float],
    hot_leads: list[float],
) -> dict[str, int | float | None]:
    if not day_replays:
        return {"trace_rows": 0, "runner_episodes": 0}

    combined: dict[str, int | float | None] = {}
    for field in ADDITIVE_REPLAY_FIELDS:
        combined[field] = int(
            sum(int(summary.get(field, 0) or 0) for summary in day_replays)
        )

    crossings = int(combined["runner_episodes"] or 0)
    watch_capture = int(combined["runner_watch_or_better_by_cross"] or 0)
    hot_capture = int(combined["runner_hot_or_position_by_cross"] or 0)
    combined["unique_symbols"] = len(unique_symbols)
    combined["runner_watch_or_better_capture_rate"] = (
        watch_capture / crossings if crossings else None
    )
    combined["runner_hot_or_position_capture_rate"] = (
        hot_capture / crossings if crossings else None
    )
    combined["median_watch_lead_minutes"] = (
        float(np.median(watch_leads)) if watch_leads else None
    )
    combined["median_hot_lead_minutes"] = (
        float(np.median(hot_leads)) if hot_leads else None
    )
    combined["max_watch_occupancy"] = max(
        int(summary.get("max_watch_occupancy", 0) or 0)
        for summary in day_replays
    )
    combined["max_hot_occupancy"] = max(
        int(summary.get("max_hot_occupancy", 0) or 0)
        for summary in day_replays
    )
    return combined


def run_partitioned_flatfile_attention_replay(
    store: MassiveFlatFileStore,
    rest_client: MassiveClient,
    start: date,
    end: date,
    *,
    scan_dir: Path,
    trace_dir: Path,
    config: AttentionConfig | None = None,
) -> dict[str, object]:
    """Replay a date range with memory bounded to one trading day."""

    runtime_config = config or AttentionConfig()
    days: list[dict[str, int | str]] = []
    day_replays: list[dict[str, int | float | None]] = []
    session_replays: list[dict[str, object]] = []
    unique_symbols: set[str] = set()
    watch_leads: list[float] = []
    hot_leads: list[float] = []
    partitions: list[dict[str, str | int]] = []

    scan_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

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

        trace = replay_attention(scan, runtime_config)
        day_key = day.isoformat()
        scan_path = scan_dir / f"trading_day={day_key}" / "part-000.parquet"
        trace_path = trace_dir / f"trading_day={day_key}" / "part-000.parquet"
        scan_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        scan.to_parquet(scan_path, index=False, compression="zstd")
        trace.to_parquet(trace_path, index=False, compression="zstd")

        day_replay = summarize_attention_replay(trace, scan)
        day_replays.append(day_replay)
        session_replays.append({"trading_day": day_key, **day_replay})
        unique_symbols.update(
            trace["ticker"].astype(str).str.strip().str.upper().unique().tolist()
        )
        watch_leads.extend(_runner_leads(trace, scan, WATCH_OR_BETTER))
        hot_leads.extend(_runner_leads(trace, scan, HOT_OR_POSITION))
        partitions.append(
            {
                "trading_day": day_key,
                "scan_rows": len(scan),
                "trace_rows": len(trace),
                "scan_path": str(scan_path),
                "trace_path": str(trace_path),
            }
        )

    replay_summary = _aggregate_replay(
        day_replays,
        unique_symbols,
        watch_leads,
        hot_leads,
    )
    return {
        "schema_version": 2,
        "output_mode": "day_partitioned",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "attention_config": asdict(runtime_config),
        "score_definition": (
            "same-timestamp cross-sectional percentile of nominal return "
            "from prior close"
        ),
        "days": days,
        "session_replays": session_replays,
        "partitions": partitions,
        "replay": replay_summary,
        "flatfile_stats": store.stats.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.attention_partitioned_replay",
        description=(
            "Replay hierarchical attention on market-wide Massive minute files "
            "and write one Parquet partition per trading day."
        ),
    )
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
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
    summary = run_partitioned_flatfile_attention_replay(
        store,
        rest_client,
        args.start,
        args.end,
        scan_dir=args.scan_dir,
        trace_dir=args.trace_dir,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
