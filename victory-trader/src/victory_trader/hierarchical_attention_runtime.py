"""Chronological learned hierarchical attention runtime replay."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import MINUTE_MS, replay_attention
from .attention_runtime import AttentionConfig
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .market_calendar import is_us_equity_trading_day
from .massive_client import MassiveClient
from .multi_day import daterange
from .recurrent_watch_hazard import (
    MODEL_FEATURES,
    MODEL_KWARGS,
    _state_history,
)
from .second_path_attention_probe import (
    BASELINE_FEATURES,
    SECOND_CACHE_DIR,
    SECOND_FEATURES,
    _annotate_scan,
)
from .targeted_second_hot_reranker import (
    MINUTE_RERANK_FEATURES,
    SECOND_RERANK_FEATURES,
    add_second_features,
    add_stage1_scores,
    shortlist_rows,
)

STAGE1_FIT_END = "2026-01-16"
STAGE2_FIT_START = "2026-01-20"
STAGE2_FIT_END = "2026-02-09"
EVAL_DAYS = [
    "2026-02-10",
    "2026-02-11",
    "2026-02-12",
    "2026-02-13",
    "2026-02-17",
]
SHORTLIST_BUDGET = 20
HOT_BUDGET = 10

FOCUS_CONFIG = AttentionConfig(
    watch_enter_score=0.60,
    watch_exit_score=0.40,
    hot_enter_score=1.01,
    hot_exit_score=1.01,
    max_watch=60,
    max_hot=0,
    drop_after_missed_batches=3,
)


def build_focus_rows(
    scan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a causal 60-name focus pool and recurrent next-step targets."""

    trace = replay_attention(scan, FOCUS_CONFIG)
    state = _state_history(trace)

    annotated = _annotate_scan(scan).sort_values(
        ["ticker", "t"], kind="stable"
    )
    next_t = annotated.groupby("ticker", sort=False)["t"].shift(-1)
    next_cross = (
        annotated.groupby("ticker", sort=False)["runner_cross_now"]
        .shift(-1)
        .astype("boolean")
    )
    annotated["has_exact_next_minute"] = (
        pd.to_numeric(next_t, errors="coerce")
        - pd.to_numeric(annotated["t"], errors="coerce")
    ).eq(MINUTE_MS)
    annotated["target_next_cross"] = (
        next_cross.fillna(False).astype(bool)
        & annotated["has_exact_next_minute"]
    ).astype(int)

    focus = state.loc[state["state"].astype(str).eq("watch")].copy()
    merged = focus.merge(
        annotated,
        on=["trading_day", "ticker", "t"],
        how="inner",
        suffixes=("", "_scan"),
        validate="one_to_one",
    )
    merged = merged.loc[
        ~merged["already_runner"].fillna(True)
        & merged["has_exact_next_minute"].fillna(False)
    ].copy()

    required = set(BASELINE_FEATURES) | {
        "focus_age_minutes",
        "watch_run_age_minutes",
        "target_next_cross",
    }
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"focus rows missing columns: {sorted(missing)}")
    return merged, trace


def fit_stage1(focus_rows: pd.DataFrame) -> HistGradientBoostingClassifier:
    fit = focus_rows.loc[
        focus_rows["trading_day"].astype(str).le(STAGE1_FIT_END)
    ].copy()
    if fit.empty or int(fit["target_next_cross"].sum()) == 0:
        raise ValueError("stage-1 focus fit is empty or has no positives")
    model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    model.fit(
        fit[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit["target_next_cross"].astype(int),
    )
    return model


def fit_stage2(
    candidates: pd.DataFrame,
) -> tuple[HistGradientBoostingClassifier, HistGradientBoostingClassifier]:
    day = candidates["trading_day"].astype(str)
    fit = candidates.loc[
        day.ge(STAGE2_FIT_START) & day.le(STAGE2_FIT_END)
    ].copy()
    if fit.empty or int(fit["target_next_cross"].sum()) == 0:
        raise ValueError("stage-2 runtime fit is empty or has no positives")

    minute_model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    second_model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    minute_model.fit(
        fit[MINUTE_RERANK_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit["target_next_cross"].astype(int),
    )
    second_model.fit(
        fit[SECOND_RERANK_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit["target_next_cross"].astype(int),
    )
    return minute_model, second_model


def add_stage2_scores(
    candidates: pd.DataFrame,
    minute_model: HistGradientBoostingClassifier,
    second_model: HistGradientBoostingClassifier,
) -> pd.DataFrame:
    result = candidates.copy()
    result["minute_rerank_probability"] = minute_model.predict_proba(
        result[MINUTE_RERANK_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    result["second_rerank_probability"] = second_model.predict_proba(
        result[SECOND_RERANK_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    return result


def build_learned_runtime_trace(
    eval_focus: pd.DataFrame,
    scored_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Allocate HOT-10 causally while storing persistent membership."""

    candidate_lookup = {
        (str(day), int(t)): group.copy()
        for (day, t), group in scored_candidates.groupby(
            ["trading_day", "t"], sort=False
        )
    }

    rows: list[dict[str, object]] = []
    promotions = 0
    demotions = 0
    replacements: list[int] = []
    retentions: list[float] = []

    for day, day_frame in eval_focus.groupby("trading_day", sort=True):
        previous_hot: set[str] = set()
        for timestamp, focus_batch in day_frame.groupby("t", sort=True):
            key = (str(day), int(timestamp))
            candidates = candidate_lookup.get(key)
            if candidates is None:
                candidates = focus_batch.iloc[0:0].copy()
                candidates["second_rerank_probability"] = pd.Series(
                    dtype=float
                )

            ranked = candidates.copy()
            if not ranked.empty:
                ranked["_incumbent"] = ranked["ticker"].astype(str).isin(
                    previous_hot
                )
                ranked = ranked.sort_values(
                    ["second_rerank_probability", "_incumbent", "ticker"],
                    ascending=[False, False, True],
                    kind="stable",
                )
                selected_hot = set(
                    ranked.head(HOT_BUDGET)["ticker"].astype(str)
                )
            else:
                selected_hot = set()

            promoted = selected_hot - previous_hot
            demoted = previous_hot - selected_hot
            promotions += len(promoted)
            demotions += len(demoted)
            replacements.append(len(promoted))
            if previous_hot:
                retentions.append(
                    len(previous_hot & selected_hot) / len(previous_hot)
                )

            score_map = (
                ranked.set_index("ticker")["second_rerank_probability"].to_dict()
                if not ranked.empty
                else {}
            )
            for item in focus_batch.itertuples(index=False):
                ticker = str(item.ticker)
                state = "hot" if ticker in selected_hot else "watch"
                reason = (
                    "learned_hot_retain"
                    if ticker in selected_hot and ticker in previous_hot
                    else "learned_hot_promote"
                    if ticker in selected_hot
                    else "learned_watch"
                )
                rows.append(
                    {
                        "trading_day": str(day),
                        "t": int(timestamp),
                        "ticker": ticker,
                        "state": state,
                        "attention_score": float(item.attention_score),
                        "learned_hot_score": score_map.get(ticker),
                        "reason": reason,
                    }
                )

            previous_hot = selected_hot

    trace = pd.DataFrame(rows)
    occupancy = (
        trace.loc[trace["state"].eq("hot")]
        .groupby(["trading_day", "t"], sort=False)
        .size()
    )
    return trace, {
        "hot_promotions": promotions,
        "hot_demotions": demotions,
        "mean_hot_set_retention": (
            float(np.mean(retentions)) if retentions else None
        ),
        "mean_hot_slots_replaced_per_decision": (
            float(np.mean(replacements)) if replacements else None
        ),
        "max_hot_occupancy": int(occupancy.max()) if len(occupancy) else 0,
        "mean_hot_occupancy": float(occupancy.mean()) if len(occupancy) else 0.0,
    }


def _strict_hot_metrics(
    trace: pd.DataFrame,
    scan: pd.DataFrame,
) -> dict[str, object]:
    crossings = scan.loc[
        scan["runner_cross_now"].fillna(False).astype(bool),
        ["trading_day", "ticker", "t"],
    ].copy()
    captures = 0
    leads: list[float] = []
    two_minute_early = 0

    hot = trace.loc[trace["state"].astype(str).eq("hot")].copy()
    for row in crossings.itertuples(index=False):
        prior = hot.loc[
            hot["trading_day"].astype(str).eq(str(row.trading_day))
            & hot["ticker"].astype(str).eq(str(row.ticker).upper())
            & hot["t"].lt(int(row.t))
        ]
        if prior.empty:
            continue
        captures += 1
        first_t = int(prior["t"].min())
        lead = (int(row.t) - first_t) / MINUTE_MS
        leads.append(lead)
        if lead >= 2.0:
            two_minute_early += 1

    total = int(len(crossings))
    return {
        "runner_crossings": total,
        "strict_hot_captures": captures,
        "strict_hot_capture_rate": (
            float(captures / total) if total else None
        ),
        "median_hot_lead_minutes": (
            float(np.median(leads)) if leads else None
        ),
        "mean_hot_lead_minutes": (
            float(np.mean(leads)) if leads else None
        ),
        "two_minute_early_count": two_minute_early,
        "two_minute_early_rate": (
            float(two_minute_early / total) if total else None
        ),
    }


def evaluate_runtime(
    eval_scan: pd.DataFrame,
    baseline_trace: pd.DataFrame,
    learned_trace: pd.DataFrame,
    eval_focus: pd.DataFrame,
    eval_candidates: pd.DataFrame,
    second_coverage: float | None,
    runtime_audit: dict[str, object],
) -> dict[str, object]:
    baseline = _strict_hot_metrics(baseline_trace, eval_scan)
    learned = _strict_hot_metrics(learned_trace, eval_scan)

    all_positive = int(eval_focus["target_next_cross"].sum())
    candidate_positive = int(eval_candidates["target_next_cross"].sum())
    shortlist_coverage = (
        float(candidate_positive / all_positive) if all_positive else None
    )

    by_day: dict[str, object] = {}
    nonlower_days = 0
    support_ok = True
    for day in EVAL_DAYS:
        scan_day = eval_scan.loc[
            eval_scan["trading_day"].astype(str).eq(day)
        ]
        baseline_day = baseline_trace.loc[
            baseline_trace["trading_day"].astype(str).eq(day)
        ]
        learned_day = learned_trace.loc[
            learned_trace["trading_day"].astype(str).eq(day)
        ]
        b = _strict_hot_metrics(baseline_day, scan_day)
        learned_metrics = _strict_hot_metrics(learned_day, scan_day)
        if (
            b["strict_hot_capture_rate"] is not None
            and learned_metrics["strict_hot_capture_rate"] is not None
            and learned_metrics["strict_hot_capture_rate"] >= b["strict_hot_capture_rate"]
        ):
            nonlower_days += 1
        if int(b["runner_crossings"]) < 10:
            support_ok = False
        by_day[day] = {"baseline": b, "learned": learned_metrics}

    b_capture = baseline["strict_hot_capture_rate"]
    l_capture = learned["strict_hot_capture_rate"]
    b_median = baseline["median_hot_lead_minutes"]
    l_median = learned["median_hot_lead_minutes"]
    b_early = baseline["two_minute_early_rate"]
    l_early = learned["two_minute_early_rate"]

    gate = bool(
        support_ok
        and b_capture is not None
        and l_capture is not None
        and l_capture > b_capture
        and nonlower_days >= 4
        and b_median is not None
        and l_median is not None
        and l_median >= b_median
        and b_early is not None
        and l_early is not None
        and l_early > b_early
        and shortlist_coverage is not None
        and shortlist_coverage >= 0.95
        and second_coverage is not None
        and second_coverage >= 0.99
        and int(runtime_audit["max_hot_occupancy"]) <= HOT_BUDGET
    )
    return {
        "eval_days": EVAL_DAYS,
        "baseline": baseline,
        "learned": learned,
        "nonlower_capture_days": nonlower_days,
        "stage1_top20_next_step_positive_coverage": shortlist_coverage,
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
    scans: list[pd.DataFrame] = []
    focus_rows: list[pd.DataFrame] = []
    baseline_traces: list[pd.DataFrame] = []
    day_summaries: list[dict[str, object]] = []

    for day in daterange(start, end):
        if not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, scan_client, day)
        focus, _ = build_focus_rows(scan)
        baseline = replay_attention(scan, AttentionConfig())

        scans.append(scan)
        focus_rows.append(focus)
        baseline_traces.append(baseline)
        day_summaries.append(
            {
                "trading_day": day.isoformat(),
                "scan_rows": int(len(scan)),
                "focus_rows": int(len(focus)),
                "focus_tickers": int(focus["ticker"].nunique())
                if len(focus)
                else 0,
                "runner_crossings": int(
                    scan["runner_cross_now"].fillna(False).astype(bool).sum()
                ),
                "point_in_time_common_stocks": scan_summary.get(
                    "point_in_time_common_stocks"
                ),
            }
        )

    all_scan = pd.concat(scans, ignore_index=True)
    all_focus = pd.concat(focus_rows, ignore_index=True)
    all_baseline = pd.concat(baseline_traces, ignore_index=True)

    stage1 = fit_stage1(all_focus)
    scored_focus = add_stage1_scores(all_focus, stage1)

    day_text = scored_focus["trading_day"].astype(str)
    rerank_period = day_text.ge(STAGE2_FIT_START) & (
        day_text.le(STAGE2_FIT_END) | day_text.isin(EVAL_DAYS)
    )
    candidates = shortlist_rows(
        scored_focus.loc[rerank_period].copy(),
        budget=SHORTLIST_BUDGET,
    )
    candidates, second_audit = add_second_features(
        candidates, second_client
    )
    minute_model, second_model = fit_stage2(candidates)
    scored_candidates = add_stage2_scores(
        candidates, minute_model, second_model
    )

    eval_scan = all_scan.loc[
        all_scan["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()
    eval_focus = scored_focus.loc[
        scored_focus["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()
    eval_candidates = scored_candidates.loc[
        scored_candidates["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()
    eval_baseline = all_baseline.loc[
        all_baseline["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()

    found_days = sorted(eval_scan["trading_day"].astype(str).unique())
    if found_days != EVAL_DAYS:
        raise ValueError(f"expected eval days {EVAL_DAYS}, found {found_days}")

    learned_trace, runtime_audit = build_learned_runtime_trace(
        eval_focus, eval_candidates
    )
    second_coverage = (
        float(
            pd.to_numeric(
                eval_candidates.get("active_seconds_60"), errors="coerce"
            )
            .fillna(0)
            .gt(0)
            .mean()
        )
        if len(eval_candidates)
        else None
    )
    evaluation = evaluate_runtime(
        eval_scan,
        eval_baseline,
        learned_trace,
        eval_focus,
        eval_candidates,
        second_coverage,
        runtime_audit,
    )

    summary: dict[str, object] = {
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "stage1_fit_end": STAGE1_FIT_END,
        "stage2_fit_start": STAGE2_FIT_START,
        "stage2_fit_end": STAGE2_FIT_END,
        "eval_days": EVAL_DAYS,
        "focus_budget": 60,
        "shortlist_budget": SHORTLIST_BUDGET,
        "hot_budget": HOT_BUDGET,
        "stage1_features": MODEL_FEATURES,
        "minute_rerank_features": MINUTE_RERANK_FEATURES,
        "second_features": SECOND_FEATURES,
        "model": MODEL_KWARGS,
        "days": day_summaries,
        "focus_rows": int(len(scored_focus)),
        "candidate_rows": int(len(scored_candidates)),
        "second_data_audit": second_audit,
        "second_client_stats": second_client.stats.to_dict(),
        "flatfile_stats": store.stats.to_dict(),
        "evaluation": evaluation,
    }
    output = learned_trace.merge(
        eval_scan.loc[
            :, ["trading_day", "ticker", "t", "runner_cross_now"]
        ],
        on=["trading_day", "ticker", "t"],
        how="left",
    )
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the learned hierarchical attention policy chronologically."
    )
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key),
        FLATFILE_CACHE_DIR,
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
    trace, summary = run_probe(
        store, scan_client, second_client, args.start, args.end
    )
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    trace.to_parquet(args.trace_output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
