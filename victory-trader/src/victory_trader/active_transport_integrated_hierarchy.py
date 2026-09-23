"""Integrate confirmed rate-limited observation transport with 1-second HOT-10.

Request 137 compares the prior rank-40 integrated hierarchy against the
promoted desired/active transport on a fresh block. Raw crossing recall remains
reported, while focus quality is gated conditional on a causal exact-prior row
existing, following request 134's observability diagnostic.
"""

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
    STAGE2_FIT_END,
    STAGE2_FIT_START,
    _hot_readiness_metrics,
    _prior_population_capture,
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
from .observation_transport import (
    RateLimitedSubscriptionSelector,
    RankedCandidate,
)
from .second_path_attention_probe import BASELINE_FEATURES, SECOND_CACHE_DIR
from .targeted_second_hot_reranker import add_second_features

EVAL_DAYS = [
    "2026-04-30",
    "2026-05-01",
    "2026-05-04",
    "2026-05-05",
    "2026-05-06",
]
FOCUS_BUDGET = 60
SHORTLIST_BUDGET = 20
MAX_ADDITIONS = 5


def active_observation_rows(scored_market: pd.DataFrame) -> pd.DataFrame:
    """Build active high-resolution rows from broad scored market evidence.

    Desired membership is the instantaneous top-20 hazard ranking. Active
    membership follows it at <=5 new names per post-initial update. Current
    feature rows come from the broad scored market, so a temporarily retained
    subscription remains usable even after leaving Focus-60.
    """

    selected: list[pd.DataFrame] = []
    for _, day_rows in scored_market.groupby("trading_day", sort=True):
        selector = RateLimitedSubscriptionSelector(
            capacity=SHORTLIST_BUDGET,
            max_additions=MAX_ADDITIONS,
        )
        previous_actual: set[str] = set()
        previous_t: dict[str, int] = {}
        active_age: dict[str, int] = {}

        for timestamp, group in day_rows.groupby("t", sort=True):
            ranked = group.sort_values(
                ["market_hazard_probability", "ticker"],
                ascending=[False, True],
                kind="stable",
            )
            candidates = [
                RankedCandidate(
                    ticker=str(row.ticker),
                    score=float(row.market_hazard_probability),
                )
                for row in ranked.itertuples(index=False)
            ]
            requested_active = set(selector.select(candidates))
            desired = set(selector.desired)

            current = ranked.loc[
                ranked["ticker"].astype(str).isin(requested_active)
            ].copy()
            actual = set(current["ticker"].astype(str))
            additions = actual - previous_actual

            ages: list[int] = []
            for ticker in current["ticker"].astype(str):
                last_t = previous_t.get(ticker)
                if last_t is not None and int(timestamp) - last_t == MINUTE_MS:
                    age = active_age.get(ticker, 0) + 1
                else:
                    age = 1
                active_age[ticker] = age
                previous_t[ticker] = int(timestamp)
                ages.append(age)

            current["focus_age_minutes"] = ages
            current["watch_run_age_minutes"] = ages
            current["stage1_hazard_probability"] = current[
                "market_hazard_probability"
            ]
            current["transport_active"] = True
            current["transport_desired_now"] = current[
                "ticker"
            ].astype(str).isin(desired)
            current["transport_addition"] = current[
                "ticker"
            ].astype(str).isin(additions)
            selected.append(current)
            previous_actual = actual

    if not selected:
        return scored_market.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True)


def _crossing_keys(frame: pd.DataFrame, scan: pd.DataFrame) -> set[tuple[str, str, int]]:
    row_keys = {
        (str(row.trading_day), str(row.ticker).upper(), int(row.t))
        for row in frame.itertuples(index=False)
    }
    captured: set[tuple[str, str, int]] = set()
    crossings = scan.loc[
        scan["runner_cross_now"].fillna(False).astype(bool),
        ["trading_day", "ticker", "t"],
    ]
    for row in crossings.itertuples(index=False):
        prior = (
            str(row.trading_day),
            str(row.ticker).upper(),
            int(row.t) - MINUTE_MS,
        )
        if prior in row_keys:
            captured.add(
                (str(row.trading_day), str(row.ticker).upper(), int(row.t))
            )
    return captured


def _retention_given(
    parent: pd.DataFrame,
    child: pd.DataFrame,
    scan: pd.DataFrame,
) -> float | None:
    parent_keys = _crossing_keys(parent, scan)
    if not parent_keys:
        return None
    child_keys = _crossing_keys(child, scan)
    return float(len(parent_keys & child_keys) / len(parent_keys))


def _observable_prior_metrics(
    scored_market: pd.DataFrame,
    focus: pd.DataFrame,
    scan: pd.DataFrame,
) -> dict[str, object]:
    observable = _crossing_keys(scored_market, scan)
    focus_keys = _crossing_keys(focus, scan)
    total = int(
        scan["runner_cross_now"].fillna(False).astype(bool).sum()
    )
    observable_count = len(observable)
    captured = len(observable & focus_keys)
    return {
        "runner_crossings": total,
        "observable_exact_prior_crossings": observable_count,
        "observable_exact_prior_rate": (
            float(observable_count / total) if total else None
        ),
        "focus_captured_observable": captured,
        "focus_capture_given_observable_rate": (
            float(captured / observable_count) if observable_count else None
        ),
    }


def evaluate(
    eval_scan: pd.DataFrame,
    eval_scored_market: pd.DataFrame,
    eval_focus: pd.DataFrame,
    baseline_candidates: pd.DataFrame,
    active_candidates: pd.DataFrame,
    baseline_trace: pd.DataFrame,
    active_trace: pd.DataFrame,
    active_second_coverage: float | None,
    active_runtime_audit: dict[str, object],
    active_transport_audit: dict[str, object],
) -> dict[str, object]:
    baseline_hot = _hot_readiness_metrics(baseline_trace, eval_scan)
    active_hot = _hot_readiness_metrics(active_trace, eval_scan)
    focus = _prior_population_capture(eval_focus, eval_scan)
    baseline_shortlist = _prior_population_capture(
        baseline_candidates, eval_scan
    )
    active_shortlist = _prior_population_capture(active_candidates, eval_scan)
    observable = _observable_prior_metrics(
        eval_scored_market, eval_focus, eval_scan
    )
    active_retention = _retention_given(
        eval_focus, active_candidates, eval_scan
    )
    baseline_retention = _retention_given(
        eval_focus, baseline_candidates, eval_scan
    )

    active_shortlist_rate = active_shortlist["prior_minute_rate"]
    active_hot_rate = active_hot["hot_within_1m_rate"]
    hot_given_active = (
        float(active_hot_rate / active_shortlist_rate)
        if active_shortlist_rate not in (None, 0)
        and active_hot_rate is not None
        else None
    )

    by_day: dict[str, object] = {}
    nonlower_days = 0
    support_ok = True
    for day in EVAL_DAYS:
        scan_day = eval_scan.loc[eval_scan["trading_day"].astype(str).eq(day)]
        scored_day = eval_scored_market.loc[
            eval_scored_market["trading_day"].astype(str).eq(day)
        ]
        focus_day = eval_focus.loc[
            eval_focus["trading_day"].astype(str).eq(day)
        ]
        baseline_candidates_day = baseline_candidates.loc[
            baseline_candidates["trading_day"].astype(str).eq(day)
        ]
        active_candidates_day = active_candidates.loc[
            active_candidates["trading_day"].astype(str).eq(day)
        ]
        baseline_trace_day = baseline_trace.loc[
            baseline_trace["trading_day"].astype(str).eq(day)
        ]
        active_trace_day = active_trace.loc[
            active_trace["trading_day"].astype(str).eq(day)
        ]
        b = _hot_readiness_metrics(baseline_trace_day, scan_day)
        a = _hot_readiness_metrics(active_trace_day, scan_day)
        if int(a["runner_crossings"]) < 10:
            support_ok = False
        if (
            b["hot_within_1m_rate"] is not None
            and a["hot_within_1m_rate"] is not None
            and a["hot_within_1m_rate"] >= b["hot_within_1m_rate"]
        ):
            nonlower_days += 1
        by_day[day] = {
            "rank40_integrated": b,
            "active_integrated": a,
            "observable_focus": _observable_prior_metrics(
                scored_day, focus_day, scan_day
            ),
            "active_retention_given_focus": _retention_given(
                focus_day, active_candidates_day, scan_day
            ),
            "rank40_retention_given_focus": _retention_given(
                focus_day, baseline_candidates_day, scan_day
            ),
        }

    baseline_one = baseline_hot["hot_within_1m_rate"]
    active_one = active_hot["hot_within_1m_rate"]
    baseline_two = baseline_hot["hot_within_2m_rate"]
    active_two = active_hot["hot_within_2m_rate"]
    focus_observable_rate = observable["focus_capture_given_observable_rate"]
    post_mean = active_transport_audit[
        "mean_post_initial_subscription_additions_per_decision"
    ]
    post_max = active_transport_audit[
        "max_post_initial_subscription_additions_per_decision"
    ]

    gate = bool(
        support_ok
        and baseline_one is not None
        and active_one is not None
        and active_one > baseline_one
        and active_one >= 0.30
        and nonlower_days >= 4
        and baseline_two is not None
        and active_two is not None
        and active_two >= baseline_two
        and focus_observable_rate is not None
        and float(focus_observable_rate) >= 0.95
        and active_retention is not None
        and active_retention >= 0.95
        and hot_given_active is not None
        and hot_given_active >= 0.90
        and active_second_coverage is not None
        and active_second_coverage >= 0.99
        and int(active_runtime_audit["max_hot_occupancy"]) <= HOT_BUDGET
        and int(active_transport_audit["max_concurrent_subscriptions"])
        <= SHORTLIST_BUDGET
        and int(active_transport_audit["selection_transport_mismatch_rows"]) == 0
        and post_mean is not None
        and float(post_mean) <= MAX_ADDITIONS
        and int(post_max) <= MAX_ADDITIONS
    )
    return {
        "eval_days": EVAL_DAYS,
        "rank40_integrated_hot": baseline_hot,
        "active_integrated_hot": active_hot,
        "focus_raw_prior_minute_capture": focus,
        "observable_focus": observable,
        "rank40_prior_minute_capture": baseline_shortlist,
        "active_prior_minute_capture": active_shortlist,
        "rank40_retention_given_focus": baseline_retention,
        "active_retention_given_focus": active_retention,
        "hot_conditional_on_active_rate": hot_given_active,
        "nonlower_immediate_capture_days": nonlower_days,
        "active_second_data_row_coverage": active_second_coverage,
        "active_runtime_audit": active_runtime_audit,
        "active_transport_audit": active_transport_audit,
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
        active_rows = active_observation_rows(scored)

        baseline_candidates.append(rank40_candidates)
        active_candidates.append(active_rows)
        if in_eval:
            eval_scans.append(scan)
            eval_focus_rows.append(learned_focus)
            eval_scored_rows.append(scored)
        day_summaries.append(
            {
                "trading_day": day_text,
                "phase": "stage2_fit" if in_stage2_fit else "evaluation",
                "scan_rows": int(len(scan)),
                "scored_market_rows": int(len(scored)),
                "focus_rows": int(len(learned_focus)),
                "rank40_candidate_rows": int(len(rank40_candidates)),
                "active_candidate_rows": int(len(active_rows)),
                "runner_crossings": int(scan["runner_cross_now"].sum()),
                "point_in_time_common_stocks": scan_summary.get(
                    "point_in_time_common_stocks"
                ),
            }
        )

    if not eval_scans or not baseline_candidates or not active_candidates:
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
    active_transport = observation_transport_audit(eval_active_candidates)
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

    evaluation = evaluate(
        eval_scan,
        eval_scored_market,
        eval_focus,
        eval_baseline_candidates,
        eval_active_candidates,
        baseline_trace,
        active_trace,
        active_second_coverage,
        active_runtime_audit,
        active_transport,
    )

    output = active_trace.merge(
        eval_scan.loc[:, ["trading_day", "ticker", "t", "runner_cross_now"]],
        on=["trading_day", "ticker", "t"],
        how="left",
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "request_id": 137,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "focus_fit_end": FIT_END,
        "stage2_fit_start": STAGE2_FIT_START,
        "stage2_fit_end": STAGE2_FIT_END,
        "eval_days": EVAL_DAYS,
        "focus_budget": FOCUS_BUDGET,
        "active_subscription_budget": SHORTLIST_BUDGET,
        "hot_budget": HOT_BUDGET,
        "max_post_initial_additions": MAX_ADDITIONS,
        "focus_features": BASELINE_FEATURES,
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
        description="Integrate promoted active transport with one-second HOT-10."
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
