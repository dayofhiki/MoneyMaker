"""Request 151: fresh confirmation of the Active transport churn ceiling."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import active_transport_integrated_hierarchy as transport
from .actionable_active_handoff import (
    add_actionable_priority,
    add_unconditional_cross_target,
    fit_observability_model,
    mark_focus_eligibility,
)
from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .balanced_retention_active_allocator import (
    add_balanced_retention_value,
    add_transport_survival_probability,
    balanced_active_rows_with_state,
    build_transport_survival_rows,
    classify_subscription_state,
    fit_transport_survival_model,
)
from .causal_inference_population_handoff import _population_summary
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hierarchical_attention_runtime import (
    HOT_BUDGET,
    STAGE2_FIT_END,
    STAGE2_FIT_START,
    _hot_readiness_metrics,
    _prior_population_capture,
    add_stage2_scores,
    build_learned_runtime_trace,
    fit_stage2,
)
from .market_calendar import is_us_equity_trading_day
from .market_wide_focus_hazard import (
    FIT_END,
    HAZARD_COLUMNS,
    build_market_hazard_frame,
    build_market_hazard_rows,
    fit_market_hazard,
    select_learned_focus,
)
from .massive_client import MassiveClient
from .multi_day import daterange
from .second_path_attention_probe import SECOND_CACHE_DIR
from .targeted_second_hot_reranker import add_second_features

REQUEST_ID = 151
EVAL_DAYS = [
    "2026-05-21",
    "2026-05-22",
    "2026-05-26",
    "2026-05-27",
    "2026-05-28",
]
PRIMARY_MAX_ADDITIONS = 8
COMPARATOR_MAX_ADDITIONS = 5
MIN_SCOREABLE_OCCUPANCY = 19.60
MAX_SPARSE_WASTE_RATE = 0.02


def _evaluate_policy(
    eval_scan: pd.DataFrame,
    eval_scored_market: pd.DataFrame,
    eval_focus: pd.DataFrame,
    eval_active: pd.DataFrame,
    eval_trace: pd.DataFrame,
    subscription_trace: pd.DataFrame,
    runtime_audit: dict[str, object],
    second_coverage: float | None,
    population_by_day: dict[str, object],
    *,
    max_additions: int,
) -> dict[str, object]:
    hot = _hot_readiness_metrics(eval_trace, eval_scan)
    focus = _prior_population_capture(eval_focus, eval_scan)
    active = _prior_population_capture(eval_active, eval_scan)
    observable = transport._observable_prior_metrics(
        eval_scored_market, eval_focus, eval_scan
    )
    active_retention = transport._retention_given(
        eval_focus, eval_active, eval_scan
    )
    active_rate = active["prior_minute_rate"]
    hot_rate = hot["hot_within_1m_rate"]
    hot_given_active = (
        float(hot_rate / active_rate)
        if active_rate not in (None, 0) and hot_rate is not None
        else None
    )
    audit = transport.explicit_subscription_audit(subscription_trace)

    by_day: dict[str, object] = {}
    for day in EVAL_DAYS:
        scan_day = eval_scan.loc[eval_scan["trading_day"].astype(str).eq(day)]
        scored_day = eval_scored_market.loc[
            eval_scored_market["trading_day"].astype(str).eq(day)
        ]
        focus_day = eval_focus.loc[
            eval_focus["trading_day"].astype(str).eq(day)
        ]
        active_day = eval_active.loc[
            eval_active["trading_day"].astype(str).eq(day)
        ]
        trace_day = eval_trace.loc[
            eval_trace["trading_day"].astype(str).eq(day)
        ]
        day_active = _prior_population_capture(active_day, scan_day)
        day_hot = _hot_readiness_metrics(trace_day, scan_day)
        day_active_rate = day_active["prior_minute_rate"]
        day_hot_rate = day_hot["hot_within_1m_rate"]
        by_day[day] = {
            "population": population_by_day[day],
            "observable_focus": transport._observable_prior_metrics(
                scored_day, focus_day, scan_day
            ),
            "active_retention_given_focus": transport._retention_given(
                focus_day, active_day, scan_day
            ),
            "hot": day_hot,
            "hot_conditional_on_active_rate": (
                float(day_hot_rate / day_active_rate)
                if day_active_rate not in (None, 0)
                and day_hot_rate is not None
                else None
            ),
        }

    post_mean = audit["mean_post_initial_subscription_additions_per_decision"]
    post_max = audit["max_post_initial_subscription_additions_per_decision"]
    focus_observable = observable["focus_capture_given_observable_rate"]

    standard_gate = bool(
        focus_observable is not None
        and float(focus_observable) >= 0.95
        and active_retention is not None
        and float(active_retention) >= 0.95
        and hot_given_active is not None
        and float(hot_given_active) >= 0.90
        and hot_rate is not None
        and float(hot_rate) >= 0.30
        and second_coverage is not None
        and float(second_coverage) >= 0.99
        and int(audit["max_concurrent_subscriptions"]) <= 20
        and int(runtime_audit["max_hot_occupancy"]) <= HOT_BUDGET
        and post_mean is not None
        and float(post_mean) <= float(max_additions)
        and int(post_max) <= max_additions
        and int(audit["selection_transport_mismatch_rows"]) == 0
    )
    return {
        "eval_days": EVAL_DAYS,
        "observable_focus": observable,
        "focus_raw_prior_minute_capture": focus,
        "active_prior_minute_capture": active,
        "active_retention_given_focus": active_retention,
        "hot": hot,
        "hot_conditional_on_active_rate": hot_given_active,
        "second_data_row_coverage": second_coverage,
        "runtime_audit": runtime_audit,
        "transport_audit": audit,
        "by_day": by_day,
        "standard_gate_pass": standard_gate,
    }


def _nonlower_hot_days(
    primary: dict[str, object],
    comparator: dict[str, object],
) -> int:
    count = 0
    for day in EVAL_DAYS:
        p = primary["by_day"][day]["hot"]["hot_within_1m_rate"]
        c = comparator["by_day"][day]["hot"]["hot_within_1m_rate"]
        if p is not None and c is not None and float(p) >= float(c):
            count += 1
    return count


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    second_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fit_labelable: list[pd.DataFrame] = []
    fit_causal: list[pd.DataFrame] = []
    fit_survival: list[pd.DataFrame] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text > FIT_END or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        fit_labelable.append(
            build_market_hazard_rows(scan).loc[:, HAZARD_COLUMNS].copy()
        )
        fit_causal.append(build_market_hazard_frame(scan))
        fit_survival.append(build_transport_survival_rows(scan))

    if not fit_labelable:
        raise ValueError("request 151 fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_labelable, ignore_index=True))
    observability_model = fit_observability_model(
        pd.concat(fit_causal, ignore_index=True)
    )
    survival_model = fit_transport_survival_model(
        pd.concat(fit_survival, ignore_index=True)
    )

    primary_candidate_parts: list[pd.DataFrame] = []
    comparator_eval_parts: list[pd.DataFrame] = []
    eval_scans: list[pd.DataFrame] = []
    eval_focus_parts: list[pd.DataFrame] = []
    eval_scored_parts: list[pd.DataFrame] = []
    primary_trace_parts: list[pd.DataFrame] = []
    comparator_trace_parts: list[pd.DataFrame] = []
    population_by_day: dict[str, object] = {}

    for day in daterange(start, end):
        day_text = day.isoformat()
        in_fit = STAGE2_FIT_START <= day_text <= STAGE2_FIT_END
        in_eval = day_text in EVAL_DAYS
        if (not in_fit and not in_eval) or not is_us_equity_trading_day(day):
            continue

        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        causal = build_market_hazard_frame(scan).copy()
        scored, focus, _ = select_learned_focus(
            causal.loc[:, HAZARD_COLUMNS].copy(),
            market_model,
        )
        scored = add_actionable_priority(scored, observability_model)
        scored = mark_focus_eligibility(scored, focus)
        scored = add_transport_survival_probability(scored, survival_model)
        scored = add_balanced_retention_value(scored)

        primary, primary_trace = balanced_active_rows_with_state(
            scored,
            max_additions=PRIMARY_MAX_ADDITIONS,
        )
        primary = add_unconditional_cross_target(primary)
        primary_candidate_parts.append(primary)

        if in_eval:
            comparator, comparator_trace = balanced_active_rows_with_state(
                scored,
                max_additions=COMPARATOR_MAX_ADDITIONS,
            )
            comparator = add_unconditional_cross_target(comparator)
            comparator_eval_parts.append(comparator)
            eval_scans.append(scan)
            eval_focus_parts.append(focus)
            eval_scored_parts.append(scored)
            primary_trace_parts.append(primary_trace)
            comparator_trace_parts.append(comparator_trace)
            population_by_day[day_text] = _population_summary(causal)

    eval_scan = pd.concat(eval_scans, ignore_index=True)
    found = sorted(eval_scan["trading_day"].astype(str).unique())
    if found != EVAL_DAYS:
        raise ValueError(f"expected request 151 eval days {EVAL_DAYS}, found {found}")

    primary_with_seconds, second_audit = add_second_features(
        pd.concat(primary_candidate_parts, ignore_index=True),
        second_client,
    )
    minute_model, second_model = fit_stage2(primary_with_seconds)
    primary_scored = add_stage2_scores(
        primary_with_seconds, minute_model, second_model
    )
    eval_primary = primary_scored.loc[
        primary_scored["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()

    comparator_with_seconds, comparator_second_audit = add_second_features(
        pd.concat(comparator_eval_parts, ignore_index=True),
        second_client,
    )
    eval_comparator = add_stage2_scores(
        comparator_with_seconds, minute_model, second_model
    )

    eval_focus = pd.concat(eval_focus_parts, ignore_index=True)
    eval_scored = pd.concat(eval_scored_parts, ignore_index=True)
    primary_subscription_trace = pd.concat(primary_trace_parts, ignore_index=True)
    comparator_subscription_trace = pd.concat(
        comparator_trace_parts, ignore_index=True
    )

    primary_runtime, primary_runtime_audit = build_learned_runtime_trace(
        eval_primary, eval_primary
    )
    comparator_runtime, comparator_runtime_audit = build_learned_runtime_trace(
        eval_comparator, eval_comparator
    )
    primary_second_coverage = float(
        pd.to_numeric(eval_primary["active_seconds_60"], errors="coerce")
        .fillna(0)
        .gt(0)
        .mean()
    )
    comparator_second_coverage = float(
        pd.to_numeric(eval_comparator["active_seconds_60"], errors="coerce")
        .fillna(0)
        .gt(0)
        .mean()
    )

    primary_eval = _evaluate_policy(
        eval_scan,
        eval_scored,
        eval_focus,
        eval_primary,
        primary_runtime,
        primary_subscription_trace,
        primary_runtime_audit,
        primary_second_coverage,
        population_by_day,
        max_additions=PRIMARY_MAX_ADDITIONS,
    )
    comparator_eval = _evaluate_policy(
        eval_scan,
        eval_scored,
        eval_focus,
        eval_comparator,
        comparator_runtime,
        comparator_subscription_trace,
        comparator_runtime_audit,
        comparator_second_coverage,
        population_by_day,
        max_additions=COMPARATOR_MAX_ADDITIONS,
    )

    primary_state = classify_subscription_state(
        primary_subscription_trace, eval_scan
    )
    comparator_state = classify_subscription_state(
        comparator_subscription_trace, eval_scan
    )

    p_ret = primary_eval["active_retention_given_focus"]
    c_ret = comparator_eval["active_retention_given_focus"]
    p_hot = primary_eval["hot"]
    c_hot = comparator_eval["hot"]
    p_occ = primary_eval["transport_audit"]["mean_scoreable_active_occupancy"]
    c_occ = comparator_eval["transport_audit"]["mean_scoreable_active_occupancy"]
    nonlower_days = _nonlower_hot_days(primary_eval, comparator_eval)

    promotion = bool(
        primary_eval["standard_gate_pass"]
        and p_ret is not None
        and c_ret is not None
        and float(p_ret) >= 0.95
        and float(p_ret) > float(c_ret)
        and float(primary_state["sparse_waste_rate"]) <= MAX_SPARSE_WASTE_RATE
        and float(primary_state["sparse_waste_rate"])
        <= float(comparator_state["sparse_waste_rate"])
        and p_occ is not None
        and c_occ is not None
        and float(p_occ) >= MIN_SCOREABLE_OCCUPANCY
        and float(p_occ) >= float(c_occ)
        and p_hot["hot_within_1m_rate"] is not None
        and c_hot["hot_within_1m_rate"] is not None
        and float(p_hot["hot_within_1m_rate"])
        >= float(c_hot["hot_within_1m_rate"])
        and p_hot["hot_within_2m_rate"] is not None
        and c_hot["hot_within_2m_rate"] is not None
        and float(p_hot["hot_within_2m_rate"])
        >= float(c_hot["hot_within_2m_rate"])
        and p_hot["hot_within_5m_rate"] is not None
        and c_hot["hot_within_5m_rate"] is not None
        and float(p_hot["hot_within_5m_rate"])
        >= float(c_hot["hot_within_5m_rate"])
        and nonlower_days >= 4
    )

    summary: dict[str, object] = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": True,
        "eval_days": EVAL_DAYS,
        "primary_policy": {
            "retention_value": (
                "active_priority * transport_survival_probability"
            ),
            "max_additions": PRIMARY_MAX_ADDITIONS,
            "actual_additions_are_demand_responsive": True,
        },
        "comparator_policy": {
            "retention_value": (
                "active_priority * transport_survival_probability"
            ),
            "max_additions": COMPARATOR_MAX_ADDITIONS,
        },
        "primary": primary_eval,
        "comparator": comparator_eval,
        "primary_state_reason_audit": primary_state,
        "comparator_state_reason_audit": comparator_state,
        "nonlower_hot_1m_days": nonlower_days,
        "fresh_gate": {
            "min_active_retention": 0.95,
            "min_scoreable_occupancy": MIN_SCOREABLE_OCCUPANCY,
            "max_sparse_waste_rate": MAX_SPARSE_WASTE_RATE,
            "require_pooled_hot_nonlower": True,
            "require_hot_1m_nonlower_days": 4,
        },
        "second_audit": second_audit,
        "comparator_second_audit": comparator_second_audit,
        "promotion_gate_pass": promotion,
        "flatfile_stats": store.stats.to_dict(),
        "second_client_stats": second_client.stats.to_dict(),
    }
    return primary_runtime, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, required=True)
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
    output, summary = run_probe(
        store, scan_client, second_client, args.start, args.end
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
