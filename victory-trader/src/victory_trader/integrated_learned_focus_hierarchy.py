"""Integrate learned market-wide focus admission with the frozen HOT hierarchy."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import MINUTE_MS
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hierarchical_attention_runtime import (
    HOT_BUDGET,
    MODEL_KWARGS,
    STAGE2_FIT_END,
    STAGE2_FIT_START,
    _hot_readiness_metrics,
    _prior_population_capture,
    add_stage1_scores,
    add_stage2_scores,
    build_focus_rows,
    build_learned_runtime_trace,
    fit_stage1,
    fit_stage2,
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
from .second_path_attention_probe import BASELINE_FEATURES, SECOND_CACHE_DIR
from .targeted_second_hot_reranker import (
    MINUTE_RERANK_FEATURES,
    SECOND_FEATURES,
    SECOND_RERANK_FEATURES,
    add_second_features,
    shortlist_rows,
)

EVAL_DAYS = [
    "2026-03-04",
    "2026-03-05",
    "2026-03-06",
    "2026-03-09",
    "2026-03-10",
]
FOCUS_BUDGET = 60
SHORTLIST_BUDGET = 20


def add_learned_focus_state(focus: pd.DataFrame) -> pd.DataFrame:
    """Add causal episode ages expected by the frozen downstream reranker."""

    work = focus.sort_values(
        ["trading_day", "ticker", "t"], kind="stable"
    ).copy()
    groups = work.groupby(["trading_day", "ticker"], sort=False)
    previous_t = groups["t"].shift(1)
    episode_start = (
        pd.to_numeric(work["t"], errors="coerce")
        - pd.to_numeric(previous_t, errors="coerce")
    ).ne(MINUTE_MS)
    episode = episode_start.groupby(
        [work["trading_day"], work["ticker"]], sort=False
    ).cumsum()
    age = work.groupby(
        [work["trading_day"], work["ticker"], episode], sort=False
    ).cumcount() + 1
    work["focus_age_minutes"] = age.astype(int)
    work["watch_run_age_minutes"] = age.astype(int)
    work["stage1_hazard_probability"] = work[
        "market_hazard_probability"
    ]
    return work


def learned_shortlist(focus: pd.DataFrame) -> pd.DataFrame:
    ranked = focus.sort_values(
        ["trading_day", "t", "market_hazard_probability", "ticker"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    return (
        ranked.groupby(["trading_day", "t"], sort=False)
        .head(SHORTLIST_BUDGET)
        .copy()
    )


def evaluate_integration(
    eval_scan: pd.DataFrame,
    frozen_trace: pd.DataFrame,
    integrated_trace: pd.DataFrame,
    integrated_focus: pd.DataFrame,
    integrated_candidates: pd.DataFrame,
    second_coverage: float | None,
    runtime_audit: dict[str, object],
) -> dict[str, object]:
    """Evaluate whether learned focus improves the complete HOT handoff."""

    frozen = _hot_readiness_metrics(frozen_trace, eval_scan)
    integrated = _hot_readiness_metrics(integrated_trace, eval_scan)
    focus = _prior_population_capture(integrated_focus, eval_scan)
    shortlist = _prior_population_capture(integrated_candidates, eval_scan)

    focus_rate = focus["prior_minute_rate"]
    shortlist_rate = shortlist["prior_minute_rate"]
    hot_rate = integrated["hot_within_1m_rate"]
    conditional_shortlist = (
        float(shortlist_rate / focus_rate)
        if focus_rate not in (None, 0) and shortlist_rate is not None
        else None
    )
    conditional_hot = (
        float(hot_rate / shortlist_rate)
        if shortlist_rate not in (None, 0) and hot_rate is not None
        else None
    )

    by_day: dict[str, object] = {}
    nonlower_days = 0
    support_ok = True
    for day in EVAL_DAYS:
        scan_day = eval_scan.loc[eval_scan["trading_day"].astype(str).eq(day)]
        frozen_day = frozen_trace.loc[
            frozen_trace["trading_day"].astype(str).eq(day)
        ]
        integrated_day = integrated_trace.loc[
            integrated_trace["trading_day"].astype(str).eq(day)
        ]
        baseline_metrics = _hot_readiness_metrics(frozen_day, scan_day)
        integrated_metrics = _hot_readiness_metrics(integrated_day, scan_day)
        baseline_rate = baseline_metrics["hot_within_1m_rate"]
        learned_rate = integrated_metrics["hot_within_1m_rate"]
        if int(baseline_metrics["runner_crossings"]) < 10:
            support_ok = False
        if (
            baseline_rate is not None
            and learned_rate is not None
            and learned_rate >= baseline_rate
        ):
            nonlower_days += 1
        by_day[day] = {
            "frozen_hierarchy": baseline_metrics,
            "integrated_hierarchy": integrated_metrics,
        }

    frozen_rate = frozen["hot_within_1m_rate"]
    gate = bool(
        support_ok
        and frozen_rate is not None
        and hot_rate is not None
        and hot_rate > frozen_rate
        and hot_rate >= 0.30
        and nonlower_days >= 4
        and focus_rate is not None
        and focus_rate >= 0.50
        and conditional_shortlist is not None
        and conditional_shortlist >= 0.95
        and conditional_hot is not None
        and conditional_hot >= 0.90
        and second_coverage is not None
        and second_coverage >= 0.99
        and int(runtime_audit["max_hot_occupancy"]) <= HOT_BUDGET
    )
    return {
        "eval_days": EVAL_DAYS,
        "frozen_hierarchy": frozen,
        "integrated_hierarchy": integrated,
        "integrated_focus_prior_minute_capture": focus,
        "integrated_shortlist_prior_minute_capture": shortlist,
        "shortlist_conditional_on_focus_rate": conditional_shortlist,
        "hot_conditional_on_shortlist_rate": conditional_hot,
        "nonlower_immediate_capture_days": nonlower_days,
        "second_data_row_coverage": second_coverage,
        "runtime_audit": runtime_audit,
        "by_day": by_day,
        "promotion_gate_pass": gate,
    }


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    second_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    market_fit_rows: list[pd.DataFrame] = []
    frozen_fit_focus: list[pd.DataFrame] = []
    day_summaries: list[dict[str, object]] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text > FIT_END or not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, scan_client, day)
        market_rows = build_market_hazard_rows(scan)
        focus, _ = build_focus_rows(scan)
        market_fit_rows.append(market_rows.loc[:, HAZARD_COLUMNS].copy())
        frozen_fit_focus.append(focus)
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

    if not market_fit_rows or not frozen_fit_focus:
        raise ValueError("focus fit period produced no rows")
    market_model = fit_market_hazard(pd.concat(market_fit_rows, ignore_index=True))
    frozen_stage1 = fit_stage1(pd.concat(frozen_fit_focus, ignore_index=True))

    integrated_eval_focus: list[pd.DataFrame] = []
    integrated_candidates: list[pd.DataFrame] = []
    frozen_eval_focus: list[pd.DataFrame] = []
    frozen_candidates: list[pd.DataFrame] = []
    eval_scans: list[pd.DataFrame] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        in_stage2_fit = STAGE2_FIT_START <= day_text <= STAGE2_FIT_END
        in_eval = day_text in EVAL_DAYS
        if (not in_stage2_fit and not in_eval) or not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, scan_client, day)
        market_rows = build_market_hazard_rows(scan).loc[:, HAZARD_COLUMNS]
        _, learned_focus, _ = select_learned_focus(
            market_rows, market_model
        )
        learned_focus = add_learned_focus_state(learned_focus)
        learned_top20 = learned_shortlist(learned_focus)
        frozen_focus, _ = build_focus_rows(scan)
        scored_frozen_focus = add_stage1_scores(frozen_focus, frozen_stage1)
        frozen_shortlist = shortlist_rows(
            scored_frozen_focus, budget=SHORTLIST_BUDGET
        )

        integrated_candidates.append(learned_top20)
        frozen_candidates.append(frozen_shortlist)
        if in_eval:
            eval_scans.append(scan)
            integrated_eval_focus.append(learned_focus)
            frozen_eval_focus.append(scored_frozen_focus)
        day_summaries.append(
            {
                "trading_day": day_text,
                "phase": "stage2_fit" if in_stage2_fit else "evaluation",
                "scan_rows": int(len(scan)),
                "integrated_focus_rows": int(len(learned_focus)),
                "integrated_candidate_rows": int(len(learned_top20)),
                "frozen_focus_rows": int(len(scored_frozen_focus)),
                "frozen_candidate_rows": int(len(frozen_shortlist)),
                "runner_crossings": int(scan["runner_cross_now"].sum()),
                "point_in_time_common_stocks": scan_summary.get(
                    "point_in_time_common_stocks"
                ),
            }
        )

    if not eval_scans or not integrated_candidates or not frozen_candidates:
        raise ValueError("stage-2 or evaluation period produced no rows")
    eval_scan = pd.concat(eval_scans, ignore_index=True)
    found_days = sorted(eval_scan["trading_day"].astype(str).unique())
    if found_days != EVAL_DAYS:
        raise ValueError(f"expected eval days {EVAL_DAYS}, found {found_days}")

    all_integrated_candidates, integrated_second_audit = add_second_features(
        pd.concat(integrated_candidates, ignore_index=True), second_client
    )
    all_frozen_candidates, frozen_second_audit = add_second_features(
        pd.concat(frozen_candidates, ignore_index=True), second_client
    )
    integrated_minute, integrated_second = fit_stage2(all_integrated_candidates)
    frozen_minute, frozen_second = fit_stage2(all_frozen_candidates)
    scored_integrated = add_stage2_scores(
        all_integrated_candidates, integrated_minute, integrated_second
    )
    scored_frozen = add_stage2_scores(
        all_frozen_candidates, frozen_minute, frozen_second
    )

    eval_integrated_candidates = scored_integrated.loc[
        scored_integrated["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()
    eval_frozen_candidates = scored_frozen.loc[
        scored_frozen["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()
    eval_integrated_focus = pd.concat(integrated_eval_focus, ignore_index=True)
    eval_frozen_focus = pd.concat(frozen_eval_focus, ignore_index=True)
    integrated_trace, integrated_runtime_audit = build_learned_runtime_trace(
        eval_integrated_focus, eval_integrated_candidates
    )
    frozen_trace, frozen_runtime_audit = build_learned_runtime_trace(
        eval_frozen_focus, eval_frozen_candidates
    )
    second_coverage = (
        float(
            pd.to_numeric(
                eval_integrated_candidates.get("active_seconds_60"),
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .mean()
        )
        if len(eval_integrated_candidates)
        else None
    )
    evaluation = evaluate_integration(
        eval_scan,
        frozen_trace,
        integrated_trace,
        eval_integrated_focus,
        eval_integrated_candidates,
        second_coverage,
        integrated_runtime_audit,
    )
    output = integrated_trace.merge(
        eval_scan.loc[:, ["trading_day", "ticker", "t", "runner_cross_now"]],
        on=["trading_day", "ticker", "t"],
        how="left",
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "focus_fit_end": FIT_END,
        "stage2_fit_start": STAGE2_FIT_START,
        "stage2_fit_end": STAGE2_FIT_END,
        "eval_days": EVAL_DAYS,
        "focus_budget": FOCUS_BUDGET,
        "shortlist_budget": SHORTLIST_BUDGET,
        "hot_budget": HOT_BUDGET,
        "focus_features": BASELINE_FEATURES,
        "minute_rerank_features": MINUTE_RERANK_FEATURES,
        "second_rerank_features": SECOND_RERANK_FEATURES,
        "second_features": SECOND_FEATURES,
        "model": MODEL_KWARGS,
        "days": day_summaries,
        "integrated_candidate_rows": int(len(scored_integrated)),
        "frozen_candidate_rows": int(len(scored_frozen)),
        "integrated_second_data_audit": integrated_second_audit,
        "frozen_second_data_audit": frozen_second_audit,
        "integrated_runtime_audit": integrated_runtime_audit,
        "frozen_runtime_audit": frozen_runtime_audit,
        "second_client_stats": second_client.stats.to_dict(),
        "flatfile_stats": store.stats.to_dict(),
        "evaluation": evaluation,
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Integrate learned focus admission with the HOT hierarchy."
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
