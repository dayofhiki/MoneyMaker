"""Request 146: revalidate attention/HOT on a fully causal inference population."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import active_transport_integrated_hierarchy as transport
from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
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

REQUEST_ID = 146
EVAL_DAYS = [
    "2026-05-07",
    "2026-05-08",
    "2026-05-11",
    "2026-05-12",
    "2026-05-13",
]


def _population_summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
    rows = int(len(frame))
    labelable = int(
        pd.to_numeric(frame["target_next_cross"], errors="coerce").notna().sum()
    )
    removed = rows - labelable
    return {
        "causal_inference_rows": rows,
        "exact_next_minute_labelable_rows": labelable,
        "historically_future_filtered_rows": removed,
        "historical_filter_fraction": float(removed / rows) if rows else None,
    }


def _evaluate(
    eval_scan: pd.DataFrame,
    eval_scored_market: pd.DataFrame,
    eval_focus: pd.DataFrame,
    eval_active: pd.DataFrame,
    eval_trace: pd.DataFrame,
    subscription_trace: pd.DataFrame,
    runtime_audit: dict[str, object],
    second_coverage: float | None,
    population_by_day: dict[str, object],
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
    transport_audit = transport.explicit_subscription_audit(subscription_trace)

    by_day: dict[str, object] = {}
    for day in EVAL_DAYS:
        scan_day = eval_scan.loc[
            eval_scan["trading_day"].astype(str).eq(day)
        ]
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
            "focus": _prior_population_capture(focus_day, scan_day),
            "active": day_active,
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

    focus_observable = observable["focus_capture_given_observable_rate"]
    post_mean = transport_audit[
        "mean_post_initial_subscription_additions_per_decision"
    ]
    post_max = transport_audit[
        "max_post_initial_subscription_additions_per_decision"
    ]
    gate = bool(
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
        and int(transport_audit["max_concurrent_subscriptions"]) <= 20
        and int(runtime_audit["max_hot_occupancy"]) <= HOT_BUDGET
        and post_mean is not None
        and float(post_mean) <= 5.0
        and int(post_max) <= 5
        and int(transport_audit["selection_transport_mismatch_rows"]) == 0
    )
    return {
        "eval_days": EVAL_DAYS,
        "focus_raw_prior_minute_capture": focus,
        "observable_focus": observable,
        "active_prior_minute_capture": active,
        "active_retention_given_focus": active_retention,
        "hot": hot,
        "hot_conditional_on_active_rate": hot_given_active,
        "second_data_row_coverage": second_coverage,
        "runtime_audit": runtime_audit,
        "transport_audit": transport_audit,
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
    fit_rows: list[pd.DataFrame] = []
    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text > FIT_END or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        fit_rows.append(
            build_market_hazard_rows(scan).loc[:, HAZARD_COLUMNS].copy()
        )
    if not fit_rows:
        raise ValueError("request 146 market-hazard fit rows are empty")
    market_model = fit_market_hazard(pd.concat(fit_rows, ignore_index=True))

    candidate_parts: list[pd.DataFrame] = []
    eval_scans: list[pd.DataFrame] = []
    eval_focus_parts: list[pd.DataFrame] = []
    eval_scored_parts: list[pd.DataFrame] = []
    eval_subscription_parts: list[pd.DataFrame] = []
    population_by_day: dict[str, object] = {}

    for day in daterange(start, end):
        day_text = day.isoformat()
        in_stage2_fit = STAGE2_FIT_START <= day_text <= STAGE2_FIT_END
        in_eval = day_text in EVAL_DAYS
        if (not in_stage2_fit and not in_eval) or not is_us_equity_trading_day(day):
            continue

        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        causal_market = build_market_hazard_frame(scan).loc[
            :, HAZARD_COLUMNS
        ].copy()
        scored, focus, _ = select_learned_focus(causal_market, market_model)
        active, subscription_trace = (
            transport.active_observation_rows_with_state(scored)
        )
        candidate_parts.append(active)

        if in_eval:
            eval_scans.append(scan)
            eval_focus_parts.append(focus)
            eval_scored_parts.append(scored)
            eval_subscription_parts.append(subscription_trace)
            population_by_day[day_text] = _population_summary(causal_market)

    if (
        not candidate_parts
        or not eval_scans
        or not eval_focus_parts
        or not eval_scored_parts
        or not eval_subscription_parts
    ):
        raise ValueError("request 146 produced incomplete fit/evaluation state")

    candidates, second_audit = add_second_features(
        pd.concat(candidate_parts, ignore_index=True),
        second_client,
    )
    minute_model, second_model = fit_stage2(candidates)
    scored_candidates = add_stage2_scores(
        candidates, minute_model, second_model
    )
    eval_active = scored_candidates.loc[
        scored_candidates["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()

    eval_scan = pd.concat(eval_scans, ignore_index=True)
    eval_focus = pd.concat(eval_focus_parts, ignore_index=True)
    eval_scored = pd.concat(eval_scored_parts, ignore_index=True)
    subscription_trace = pd.concat(
        eval_subscription_parts, ignore_index=True
    )

    found = sorted(eval_scan["trading_day"].astype(str).unique())
    if found != EVAL_DAYS:
        raise ValueError(f"expected request 146 eval days {EVAL_DAYS}, found {found}")

    trace, runtime_audit = build_learned_runtime_trace(
        eval_active,
        eval_active,
    )
    second_coverage = (
        float(
            pd.to_numeric(
                eval_active.get("active_seconds_60"), errors="coerce"
            )
            .fillna(0)
            .gt(0)
            .mean()
        )
        if len(eval_active)
        else None
    )

    evaluation = _evaluate(
        eval_scan,
        eval_scored,
        eval_focus,
        eval_active,
        trace,
        subscription_trace,
        runtime_audit,
        second_coverage,
        population_by_day,
    )

    eval_population = pd.concat(
        [
            build_market_hazard_frame(scan).loc[:, HAZARD_COLUMNS]
            for scan in eval_scans
        ],
        ignore_index=True,
    )
    summary = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "market_fit_end": FIT_END,
        "stage2_fit_start": STAGE2_FIT_START,
        "stage2_fit_end": STAGE2_FIT_END,
        "eval_days": EVAL_DAYS,
        "population": _population_summary(eval_population),
        "population_by_day": population_by_day,
        "second_audit": second_audit,
        "evaluation": evaluation,
        "flatfile_stats": store.stats.to_dict(),
        "second_client_stats": second_client.stats.to_dict(),
    }
    return trace, summary


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
