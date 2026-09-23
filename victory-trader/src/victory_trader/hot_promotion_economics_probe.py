"""Request 140: economic diagnostics for every HOT promotion episode."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .config import load_settings, require_flatfile_credentials
from .explicit_transport_integrated_hierarchy import EVAL_DAYS
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hot_entry_economics import (
    label_hot_event_economics,
    promotion_hot_events,
    summarize_hot_economics,
)
from .massive_client import MassiveClient

REQUEST_ID = 140


def _oracle_group(group: pd.DataFrame) -> dict[str, object]:
    oracle = pd.to_numeric(group["oracle_best_base_pct"], errors="coerce")
    valid = oracle.notna()
    return {
        "episodes": int(len(group)),
        "evaluable": int(valid.sum()),
        "coverage": float(valid.mean()) if len(group) else None,
        "oracle_base_mean_pct": (
            float(oracle.loc[valid].mean()) if valid.any() else None
        ),
        "oracle_base_median_pct": (
            float(oracle.loc[valid].median()) if valid.any() else None
        ),
        "oracle_base_positive_rate": (
            float(oracle.loc[valid].gt(0).mean()) if valid.any() else None
        ),
    }


def _score_diagnostics(labeled: pd.DataFrame) -> dict[str, object]:
    oracle = pd.to_numeric(labeled["oracle_best_base_pct"], errors="coerce")
    score = pd.to_numeric(labeled.get("learned_hot_score"), errors="coerce")
    valid = oracle.notna() & score.notna()
    if valid.sum() < 20:
        return {
            "rows": int(valid.sum()),
            "spearman_oracle": None,
            "quintiles": {},
        }

    frame = labeled.loc[valid].copy()
    frame["_oracle"] = oracle.loc[valid].astype(float)
    frame["_score"] = score.loc[valid].astype(float)
    spearman = float(frame["_score"].corr(frame["_oracle"], method="spearman"))

    rank = frame["_score"].rank(method="first")
    frame["_quintile"] = pd.qcut(rank, 5, labels=False) + 1
    quintiles: dict[str, object] = {}
    for quintile, group in frame.groupby("_quintile", sort=True):
        quintiles[str(int(quintile))] = {
            "episodes": int(len(group)),
            "score_mean": float(group["_score"].mean()),
            "oracle_base_mean_pct": float(group["_oracle"].mean()),
            "oracle_base_median_pct": float(group["_oracle"].median()),
            "oracle_base_positive_rate": float(group["_oracle"].gt(0).mean()),
        }

    return {
        "rows": int(len(frame)),
        "spearman_oracle": spearman,
        "quintiles": quintiles,
    }


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
        raise ValueError("request 140 requires promoted request-138 attention")
    if list(attention_summary.get("eval_days") or []) != EVAL_DAYS:
        raise ValueError("request-138 dates do not match request-140 frozen dates")

    scans: list[pd.DataFrame] = []
    for day_text in EVAL_DAYS:
        scan, _ = build_flatfile_scan_day(
            store,
            scan_client,
            date.fromisoformat(day_text),
        )
        scans.append(scan)
    eval_scan = pd.concat(scans, ignore_index=True)

    events = promotion_hot_events(trace)
    labeled = label_hot_event_economics(events, eval_scan)
    overall = summarize_hot_economics(labeled)
    overall["promotion_episodes"] = overall.pop("first_hot_episodes")

    first = labeled.loc[
        ~labeled["is_repromotion"].fillna(False).astype(bool)
    ].copy()
    repeat = labeled.loc[
        labeled["is_repromotion"].fillna(False).astype(bool)
    ].copy()

    promotion_index = pd.to_numeric(
        labeled["promotion_index"], errors="coerce"
    )
    index_groups: dict[str, object] = {}
    buckets = pd.Series(
        np.where(
            promotion_index.eq(1),
            "1",
            np.where(promotion_index.eq(2), "2", "3+"),
        ),
        index=labeled.index,
    )
    for name, group in labeled.groupby(buckets, sort=True):
        index_groups[str(name)] = _oracle_group(group)

    spacing = pd.to_numeric(
        repeat["minutes_since_previous_promotion"], errors="coerce"
    ).dropna()

    summary = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "diagnostic_only": True,
        "opens_new_dates": False,
        "reused_attention_eval_days": EVAL_DAYS,
        "overall": overall,
        "first_promotions": _oracle_group(first),
        "repromotions": _oracle_group(repeat),
        "promotion_index_groups": index_groups,
        "repromotion_spacing_minutes": {
            "count": int(len(spacing)),
            "median": float(spacing.median()) if len(spacing) else None,
            "p25": float(spacing.quantile(0.25)) if len(spacing) else None,
            "p75": float(spacing.quantile(0.75)) if len(spacing) else None,
        },
        "hot_score_economic_alignment": _score_diagnostics(labeled),
    }
    return labeled, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose every HOT promotion as a separate economic episode."
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
