"""Request 138: confirm integrated HOT handoff with explicit transport state."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import active_transport_integrated_hierarchy as base
from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hierarchical_attention_runtime import (
    STAGE2_FIT_END,
    STAGE2_FIT_START,
    add_stage2_scores,
    build_learned_runtime_trace,
    fit_stage2,
)
from .integrated_learned_focus_hierarchy import (
    add_learned_focus_state,
    learned_shortlist,
    observation_transport_audit,
)
from .market_calendar import is_us_equity_trading_day
from .market_wide_focus_hazard import (
    FIT_END,
    HAZARD_COLUMNS,
    build_market_hazard_rows,
    fit_market_hazard,
    select_learned_focus,
)
from .massive_client import MassiveClient
from .multi_day import daterange
from .second_path_attention_probe import SECOND_CACHE_DIR
from .targeted_second_hot_reranker import add_second_features

EVAL_DAYS = [
    "2026-05-07",
    "2026-05-08",
    "2026-05-11",
    "2026-05-12",
    "2026-05-13",
]


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    second_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fit_rows: list[pd.DataFrame] = []
    day_summaries: list[dict[str, object]] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text > FIT_END or not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, scan_client, day)
        market_rows = build_market_hazard_rows(scan)
        fit_rows.append(market_rows.loc[:, HAZARD_COLUMNS].copy())
        day_summaries.append(
            {
                "trading_day": day_text,
                "phase": "focus_fit",
                "scan_rows": int(len(scan)),
                "hazard_rows": int(len(market_rows)),
                "runner_crossings": int(scan["runner_cross_now"].sum()),
                "point_in_time_common_stocks": scan_summary.get(
                    "point_in_time_common_stocks"
                ),
            }
        )

    if not fit_rows:
        raise ValueError("focus fit period produced no rows")
    market_model = fit_market_hazard(pd.concat(fit_rows, ignore_index=True))

    baseline_candidates: list[pd.DataFrame] = []
    active_candidates: list[pd.DataFrame] = []
    eval_subscription_traces: list[pd.DataFrame] = []
    eval_focus_rows: list[pd.DataFrame] = []
    eval_scored_rows: list[pd.DataFrame] = []
    eval_scans: list[pd.DataFrame] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        in_stage2_fit = STAGE2_FIT_START <= day_text <= STAGE2_FIT_END
        in_eval = day_text in EVAL_DAYS
        if (not in_stage2_fit and not in_eval) or not is_us_equity_trading_day(day):
            continue

        scan, scan_summary = build_flatfile_scan_day(store, scan_client, day)
        market_rows = build_market_hazard_rows(scan).loc[:, HAZARD_COLUMNS]
        scored, learned_focus, _ = select_learned_focus(
            market_rows, market_model
        )

        baseline_focus = add_learned_focus_state(learned_focus)
        rank40_candidates = learned_shortlist(baseline_focus)
        active_rows, subscription_trace = (
            base.active_observation_rows_with_state(scored)
        )

        baseline_candidates.append(rank40_candidates)
        active_candidates.append(active_rows)
        if in_eval:
            eval_scans.append(scan)
            eval_focus_rows.append(learned_focus)
            eval_scored_rows.append(scored)
            eval_subscription_traces.append(subscription_trace)

        day_summaries.append(
            {
                "trading_day": day_text,
                "phase": "stage2_fit" if in_stage2_fit else "evaluation",
                "scan_rows": int(len(scan)),
                "scored_market_rows": int(len(scored)),
                "focus_rows": int(len(learned_focus)),
                "rank40_candidate_rows": int(len(rank40_candidates)),
                "active_candidate_rows": int(len(active_rows)),
                "subscription_trace_rows": int(len(subscription_trace)),
                "runner_crossings": int(scan["runner_cross_now"].sum()),
                "point_in_time_common_stocks": scan_summary.get(
                    "point_in_time_common_stocks"
                ),
            }
        )

    if (
        not eval_scans
        or not baseline_candidates
        or not active_candidates
        or not eval_subscription_traces
    ):
        raise ValueError("stage-2 or evaluation period produced no rows")

    eval_scan = pd.concat(eval_scans, ignore_index=True)
    found_days = sorted(eval_scan["trading_day"].astype(str).unique())
    if found_days != EVAL_DAYS:
        raise ValueError(f"expected eval days {EVAL_DAYS}, found {found_days}")

    all_baseline_candidates, baseline_second_audit = add_second_features(
        pd.concat(baseline_candidates, ignore_index=True), second_client
    )
    all_active_candidates, active_second_audit = add_second_features(
        pd.concat(active_candidates, ignore_index=True), second_client
    )

    baseline_minute, baseline_second = fit_stage2(all_baseline_candidates)
    active_minute, active_second = fit_stage2(all_active_candidates)
    scored_baseline = add_stage2_scores(
        all_baseline_candidates, baseline_minute, baseline_second
    )
    scored_active = add_stage2_scores(
        all_active_candidates, active_minute, active_second
    )

    eval_baseline_candidates = scored_baseline.loc[
        scored_baseline["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()
    eval_active_candidates = scored_active.loc[
        scored_active["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()
    eval_focus = pd.concat(eval_focus_rows, ignore_index=True)
    eval_scored_market = pd.concat(eval_scored_rows, ignore_index=True)

    baseline_focus_state = add_learned_focus_state(eval_focus)
    baseline_trace, baseline_runtime_audit = build_learned_runtime_trace(
        baseline_focus_state, eval_baseline_candidates
    )
    active_trace, active_runtime_audit = build_learned_runtime_trace(
        eval_active_candidates, eval_active_candidates
    )

    subscription_trace = pd.concat(
        eval_subscription_traces, ignore_index=True
    )
    explicit_transport = base.explicit_subscription_audit(
        subscription_trace
    )
    reconstructed_transport = observation_transport_audit(
        eval_active_candidates
    )
    active_second_coverage = (
        float(
            pd.to_numeric(
                eval_active_candidates.get("active_seconds_60"),
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .mean()
        )
        if len(eval_active_candidates)
        else None
    )

    previous_days = base.EVAL_DAYS
    base.EVAL_DAYS = EVAL_DAYS
    try:
        evaluation = base.evaluate(
            eval_scan,
            eval_scored_market,
            eval_focus,
            eval_baseline_candidates,
            eval_active_candidates,
            baseline_trace,
            active_trace,
            active_second_coverage,
            active_runtime_audit,
            explicit_transport,
        )
    finally:
        base.EVAL_DAYS = previous_days

    evaluation["row_reconstructed_transport_audit"] = reconstructed_transport
    evaluation["explicit_subscription_transport_audit"] = explicit_transport

    output = active_trace.merge(
        eval_scan.loc[:, ["trading_day", "ticker", "t", "runner_cross_now"]],
        on=["trading_day", "ticker", "t"],
        how="left",
    )
    summary: dict[str, object] = {
        "schema_version": 2,
        "request_id": 138,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "focus_fit_end": FIT_END,
        "stage2_fit_start": STAGE2_FIT_START,
        "stage2_fit_end": STAGE2_FIT_END,
        "eval_days": EVAL_DAYS,
        "focus_budget": base.FOCUS_BUDGET,
        "active_subscription_budget": base.SHORTLIST_BUDGET,
        "hot_budget": base.HOT_BUDGET,
        "max_post_initial_additions": base.MAX_ADDITIONS,
        "days": day_summaries,
        "baseline_second_audit": baseline_second_audit,
        "active_second_audit": active_second_audit,
        "baseline_runtime_audit": baseline_runtime_audit,
        "evaluation": evaluation,
        "flatfile_stats": store.stats.to_dict(),
        "second_client_stats": second_client.stats.to_dict(),
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Confirm integrated HOT handoff with explicit transport state."
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
        store, scan_client, second_client, args.start, args.end
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
