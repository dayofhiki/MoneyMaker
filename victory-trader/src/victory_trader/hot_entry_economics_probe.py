"""Request-conditional first-HOT economics probe.

This replays the already-opened request-138 dates, then labels only economic
outcomes that were not used by the attention promotion gate. It opens no later
calendar block.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, _prepare_minute_bars
from .config import load_settings, require_flatfile_credentials
from .explicit_transport_integrated_hierarchy import EVAL_DAYS
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .attention_replay import MINUTE_MS
from .hot_entry_economics import label_first_hot_economics, summarize_hot_economics
from .market_calendar import regular_session_bounds
from .massive_client import MassiveClient

REQUEST_ID = 139


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    attention_trace_path: Path,
    attention_summary_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    trace = pd.read_parquet(attention_trace_path)
    attention_summary = json.loads(
        attention_summary_path.read_text(encoding="utf-8")
    )
    if not bool(attention_summary["evaluation"]["promotion_gate_pass"]):
        raise ValueError(
            "request 139 is conditional on request-138 attention promotion"
        )
    if list(attention_summary.get("eval_days") or []) != EVAL_DAYS:
        raise ValueError(
            "request-138 artifact evaluation dates do not match frozen request 139 dates"
        )

    price_frames: list[pd.DataFrame] = []
    hot = trace.loc[trace["state"].astype(str).eq("hot")].copy()
    for day_text in EVAL_DAYS:
        day = date.fromisoformat(day_text)
        tickers = set(
            hot.loc[
                hot["trading_day"].astype(str).eq(day_text),
                "ticker",
            ].astype(str)
        )
        if not tickers:
            continue
        bounds = regular_session_bounds(day)
        if bounds is None:
            raise ValueError(f"expected trading session for {day_text}")
        open_ms = int(bounds[0].timestamp() * 1000)
        close_ms = int(bounds[1].timestamp() * 1000)
        raw_minutes = store.minute_aggregates(day)
        minutes, _, _ = _prepare_minute_bars(
            raw_minutes,
            tickers,
            day,
            scan_client,
            open_ms,
            close_ms,
        )
        prices = minutes.loc[:, ["trading_day", "ticker", "t", "o"]].copy()
        prices["t"] = pd.to_numeric(prices["t"], errors="raise").astype("int64")
        prices["t"] += MINUTE_MS
        price_frames.append(prices)
    eval_prices = pd.concat(price_frames, ignore_index=True)

    labeled = label_first_hot_economics(trace, eval_prices)
    economics = summarize_hot_economics(labeled)
    summary = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "diagnostic_only": True,
        "reused_attention_eval_days": EVAL_DAYS,
        "opens_new_dates": False,
        "attention_promotion_gate_pass": True,
        "attention_evaluation": attention_summary["evaluation"],
        "economics": economics,
    }
    return labeled, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose first-HOT entry economics on request-138 dates."
    )
    parser.add_argument("--attention-trace", type=Path, required=True)
    parser.add_argument("--attention-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key), FLATFILE_CACHE_DIR
    )
    scan_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=0.0,
    )
    output, summary = run_probe(
        store,
        scan_client,
        args.attention_trace,
        args.attention_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
