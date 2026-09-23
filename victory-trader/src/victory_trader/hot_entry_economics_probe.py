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

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .config import load_settings, require_flatfile_credentials
from .explicit_transport_integrated_hierarchy import EVAL_DAYS, run_probe as run_attention
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hot_entry_economics import label_first_hot_economics, summarize_hot_economics
from .massive_client import MassiveClient
from .second_path_attention_probe import SECOND_CACHE_DIR

REQUEST_ID = 139


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    second_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    trace, attention_summary = run_attention(
        store,
        scan_client,
        second_client,
        start,
        end,
    )
    if not bool(attention_summary["evaluation"]["promotion_gate_pass"]):
        raise ValueError(
            "request 139 is conditional on request-138 attention promotion"
        )

    scans: list[pd.DataFrame] = []
    for day_text in EVAL_DAYS:
        scan, _ = build_flatfile_scan_day(
            store,
            scan_client,
            date.fromisoformat(day_text),
        )
        scans.append(scan)
    eval_scan = pd.concat(scans, ignore_index=True)

    labeled = label_first_hot_economics(trace, eval_scan)
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
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
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
    second_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR,
        request_interval_seconds=0.0,
    )
    output, summary = run_probe(
        store,
        scan_client,
        second_client,
        args.start,
        args.end,
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
