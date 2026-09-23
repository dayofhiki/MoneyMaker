"""Request 150: balanced Active retention value."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import active_transport_integrated_hierarchy as transport
from .actionable_active_handoff import (
    add_actionable_priority,
    add_unconditional_cross_target,
    fit_observability_model,
    mark_focus_eligibility,
)
from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import MINUTE_MS
from .causal_inference_population_handoff import (
    EVAL_DAYS,
    _evaluate as evaluate_causal_handoff,
    _population_summary,
)
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hierarchical_attention_runtime import (
    MODEL_KWARGS,
    STAGE2_FIT_END,
    STAGE2_FIT_START,
    add_stage2_scores,
    build_learned_runtime_trace,
    fit_stage2,
)
from .market_calendar import is_us_equity_trading_day, regular_session_bounds
from .market_wide_focus_hazard import (
    BASELINE_FEATURES,
    FIT_END,
    HAZARD_COLUMNS,
    build_market_hazard_frame,
    build_market_hazard_rows,
    fit_market_hazard,
    select_learned_focus,
)
from .massive_client import MassiveClient
from .multi_day import daterange
from .observation_transport import RankedCandidate, RetentionAwareSubscriptionSelector
from .second_path_attention_probe import SECOND_CACHE_DIR, _annotate_scan
from .targeted_second_hot_reranker import add_second_features

REQUEST_ID = 150
SURVIVAL_HORIZON_MINUTES = 3
PRIMARY_MAX_ADDITIONS = 5
SENSITIVITY_ADDITIONS = (6, 8, 20)

REQUEST147_HOT_1M = 0.5025693730729702
REQUEST147_HOT_2M = 0.591983556012333
REQUEST147_HOT_5M = 0.6557040082219938
REQUEST149_SPARSE_WASTE = 0.018487179487179487
REQUEST149_SCOREABLE_OCCUPANCY = 19.603589743589744


def build_transport_survival_rows(scan: pd.DataFrame) -> pd.DataFrame:
    """Label whether a current pre-runner state stays useful for three minutes.

    Success means either graduating into runner state before a sparse gap, or
    remaining continuously observable through the full horizon. A sparse gap
    before either outcome is failure. Rows without enough session time remain
    unlabeled unless graduation occurs first.
    """

    base = build_market_hazard_frame(scan).copy()
    raw = _annotate_scan(scan).loc[
        :, ["trading_day", "ticker", "t", "already_runner"]
    ].copy()
    raw["trading_day"] = raw["trading_day"].astype(str)
    raw["ticker"] = raw["ticker"].astype(str).str.upper()
    base["trading_day"] = base["trading_day"].astype(str)
    base["ticker"] = base["ticker"].astype(str).str.upper()

    for minute in range(1, SURVIVAL_HORIZON_MINUTES + 1):
        future = raw.copy()
        future["t"] = (
            pd.to_numeric(future["t"], errors="raise").astype("int64")
            - minute * MINUTE_MS
        )
        future = future.rename(
            columns={"already_runner": f"future_{minute}_runner"}
        )
        future[f"future_{minute}_exists"] = True
        base = base.merge(
            future[
                [
                    "trading_day",
                    "ticker",
                    "t",
                    f"future_{minute}_exists",
                    f"future_{minute}_runner",
                ]
            ],
            on=["trading_day", "ticker", "t"],
            how="left",
            validate="one_to_one",
        )
        base[f"future_{minute}_exists"] = (
            base[f"future_{minute}_exists"].fillna(False).astype(bool)
        )
        base[f"future_{minute}_runner"] = (
            base[f"future_{minute}_runner"].fillna(False).astype(bool)
        )

    close_by_day: dict[str, int] = {}
    for day_text in base["trading_day"].astype(str).unique():
        bounds = regular_session_bounds(date.fromisoformat(day_text))
        if bounds is None:
            raise ValueError(f"missing session bounds for {day_text}")
        close_by_day[day_text] = int(bounds[1].timestamp() * 1000)

    t = pd.to_numeric(base["t"], errors="raise").astype("int64")
    close_ms = base["trading_day"].map(close_by_day).astype("int64")
    full_horizon = (
        t + SURVIVAL_HORIZON_MINUTES * MINUTE_MS
    ).le(close_ms)

    alive = pd.Series(True, index=base.index)
    success = pd.Series(False, index=base.index)
    sparse_failure = pd.Series(False, index=base.index)
    for minute in range(1, SURVIVAL_HORIZON_MINUTES + 1):
        exists = base[f"future_{minute}_exists"]
        runner = base[f"future_{minute}_runner"]
        success |= alive & exists & runner
        sparse_failure |= alive & ~exists
        alive &= exists & ~runner

    success |= alive
    target = pd.Series(np.nan, index=base.index, dtype=float)
    target.loc[success] = 1.0
    target.loc[full_horizon & sparse_failure & ~success] = 0.0
    base["transport_survival_3m_target"] = target
    return base


def fit_transport_survival_model(
    rows: pd.DataFrame,
) -> HistGradientBoostingClassifier:
    fit = rows.loc[rows["trading_day"].astype(str).le(FIT_END)].copy()
    target = pd.to_numeric(
        fit["transport_survival_3m_target"], errors="coerce"
    )
    fit = fit.loc[target.notna()].copy()
    y = pd.to_numeric(
        fit["transport_survival_3m_target"], errors="raise"
    ).astype(int)
    if fit.empty or y.nunique() < 2:
        raise ValueError("transport survival fit needs both classes")

    model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    model.fit(
        fit[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan),
        y,
    )
    return model


def add_transport_survival_probability(
    scored: pd.DataFrame,
    model: HistGradientBoostingClassifier,
) -> pd.DataFrame:
    result = scored.copy()
    result["transport_survival_probability"] = model.predict_proba(
        result[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    return result


def _request149_retention_aware_active_rows_with_state(
    scored_market: pd.DataFrame,
    *,
    max_additions: int = PRIMARY_MAX_ADDITIONS,
    retention_score_column: str = "transport_survival_probability",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate admission ranking from stale-incumbent eviction value."""

    selected: list[pd.DataFrame] = []
    snapshots: list[dict[str, object]] = []

    for day, day_rows in scored_market.groupby("trading_day", sort=True):
        selector = RetentionAwareSubscriptionSelector(
            capacity=transport.SHORTLIST_BUDGET,
            max_additions=max_additions,
        )
        previous: set[str] = set()
        active_since: dict[str, int] = {}

        for timestamp, group in day_rows.groupby("t", sort=True):
            ranked = group.sort_values(
                ["active_priority", "ticker"],
                ascending=[False, True],
                kind="stable",
            )
            focus_pool = ranked.loc[
                ranked["transport_focus_eligible"].fillna(False).astype(bool)
            ]
            candidates = [
                RankedCandidate(
                    ticker=str(row.ticker),
                    score=float(row.active_priority),
                )
                for row in focus_pool.itertuples(index=False)
            ]
            if retention_score_column not in ranked.columns:
                raise ValueError(
                    f"missing retention score column: {retention_score_column}"
                )
            retention_scores = {}
            for row in ranked.itertuples(index=False):
                value = float(getattr(row, retention_score_column))
                if np.isfinite(value):
                    retention_scores[str(row.ticker)] = value
            selector.select(
                candidates,
                retention_scores=retention_scores,
            )
            subscribed = set(selector.active)
            desired = set(selector.desired)
            scoreable_names = set(ranked["ticker"].astype(str))

            additions = subscribed - previous
            removals = previous - subscribed
            for ticker in removals:
                active_since.pop(ticker, None)
            for ticker in subscribed:
                active_since.setdefault(ticker, int(timestamp))

            for ticker in sorted(subscribed):
                snapshots.append(
                    {
                        "trading_day": str(day),
                        "t": int(timestamp),
                        "ticker": ticker,
                        "transport_active": True,
                        "transport_desired_now": ticker in desired,
                        "scoreable_now": ticker in scoreable_names,
                    }
                )

            current = ranked.loc[
                ranked["ticker"].astype(str).isin(subscribed)
            ].copy()
            if not current.empty:
                current["focus_age_minutes"] = [
                    int((int(timestamp) - active_since[ticker]) / MINUTE_MS) + 1
                    for ticker in current["ticker"].astype(str)
                ]
                current["watch_run_age_minutes"] = current["focus_age_minutes"]
                current["stage1_hazard_probability"] = current[
                    "market_hazard_probability"
                ]
                current["transport_active"] = True
                current["transport_desired_now"] = (
                    current["ticker"].astype(str).isin(desired)
                )
                current["transport_addition"] = (
                    current["ticker"].astype(str).isin(additions)
                )
                selected.append(current)
            previous = set(subscribed)

    rows = (
        pd.concat(selected, ignore_index=True)
        if selected
        else scored_market.iloc[0:0].copy()
    )
    return rows, pd.DataFrame(snapshots)


def add_balanced_retention_value(scored: pd.DataFrame) -> pd.DataFrame:
    """Balance runner opportunity with multi-minute slot survivability."""

    result = scored.copy()
    actionable = pd.to_numeric(
        result["active_priority"], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    survival = pd.to_numeric(
        result["transport_survival_probability"], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    result["balanced_retention_value"] = actionable * survival
    return result


def balanced_active_rows_with_state(
    scored_market: pd.DataFrame,
    *,
    max_additions: int = PRIMARY_MAX_ADDITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _request149_retention_aware_active_rows_with_state(
        scored_market,
        max_additions=max_additions,
        retention_score_column="balanced_retention_value",
    )


def classify_subscription_state(
    trace: pd.DataFrame,
    scan: pd.DataFrame,
) -> dict[str, object]:
    annotated = _annotate_scan(scan).loc[
        :, ["trading_day", "ticker", "t", "already_runner"]
    ].copy()
    annotated["trading_day"] = annotated["trading_day"].astype(str)
    annotated["ticker"] = annotated["ticker"].astype(str).str.upper()
    raw = {
        (str(row.trading_day), str(row.ticker), int(row.t)): bool(row.already_runner)
        for row in annotated.itertuples(index=False)
    }

    counts = {
        "scoreable": 0,
        "runner_graduated": 0,
        "sparse_missing_bar": 0,
        "other_present_unscoreable": 0,
    }
    for row in trace.itertuples(index=False):
        if bool(row.scoreable_now):
            counts["scoreable"] += 1
            continue
        key = (str(row.trading_day), str(row.ticker).upper(), int(row.t))
        if key not in raw:
            counts["sparse_missing_bar"] += 1
        elif raw[key]:
            counts["runner_graduated"] += 1
        else:
            counts["other_present_unscoreable"] += 1

    total = int(len(trace))
    return {
        **counts,
        "subscription_rows": total,
        "sparse_waste_rate": (
            float(counts["sparse_missing_bar"] / total) if total else None
        ),
        "runner_graduation_rate": (
            float(counts["runner_graduated"] / total) if total else None
        ),
    }


def _sensitivity_summary(
    scored_parts: list[pd.DataFrame],
    eval_scan: pd.DataFrame,
) -> dict[str, object]:
    out: dict[str, object] = {}
    scored = pd.concat(scored_parts, ignore_index=True)
    for additions in SENSITIVITY_ADDITIONS:
        active, trace = balanced_active_rows_with_state(
            scored,
            max_additions=additions,
        )
        audit = transport.explicit_subscription_audit(trace)
        state = classify_subscription_state(trace, eval_scan)
        out[str(additions)] = {
            "active_retention_given_focus": None,
            "transport_audit": audit,
            "state_reason_audit": state,
            "active_rows": int(len(active)),
        }
    return out


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
        raise ValueError("request 150 fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_labelable, ignore_index=True))
    observability_model = fit_observability_model(
        pd.concat(fit_causal, ignore_index=True)
    )
    survival_rows = pd.concat(fit_survival, ignore_index=True)
    survival_model = fit_transport_survival_model(survival_rows)

    survival_fit = survival_rows.loc[
        survival_rows["trading_day"].astype(str).le(FIT_END)
    ].copy()
    survival_y = pd.to_numeric(
        survival_fit["transport_survival_3m_target"], errors="coerce"
    )
    labeled = survival_y.notna()
    survival_auc = (
        float(
            roc_auc_score(
                survival_y.loc[labeled].astype(int),
                survival_model.predict_proba(
                    survival_fit.loc[labeled, BASELINE_FEATURES].replace(
                        [np.inf, -np.inf], np.nan
                    )
                )[:, 1],
            )
        )
        if labeled.any() and survival_y.loc[labeled].nunique() > 1
        else None
    )

    candidate_parts: list[pd.DataFrame] = []
    eval_scans: list[pd.DataFrame] = []
    eval_focus_parts: list[pd.DataFrame] = []
    eval_scored_parts: list[pd.DataFrame] = []
    eval_primary_trace_parts: list[pd.DataFrame] = []
    eval_baseline_trace_parts: list[pd.DataFrame] = []
    eval_sensitivity_scored: list[pd.DataFrame] = []
    population_by_day: dict[str, object] = {}

    for day in daterange(start, end):
        day_text = day.isoformat()
        in_stage2_fit = STAGE2_FIT_START <= day_text <= STAGE2_FIT_END
        in_eval = day_text in EVAL_DAYS
        if (not in_stage2_fit and not in_eval) or not is_us_equity_trading_day(day):
            continue

        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        causal_market = build_market_hazard_frame(scan).copy()
        scored, focus, _ = select_learned_focus(
            causal_market.loc[:, HAZARD_COLUMNS].copy(),
            market_model,
        )
        scored = add_actionable_priority(scored, observability_model)
        scored = mark_focus_eligibility(scored, focus)
        scored = add_transport_survival_probability(scored, survival_model)

        scored = add_balanced_retention_value(scored)
        primary, primary_trace = balanced_active_rows_with_state(scored)
        primary = add_unconditional_cross_target(primary)
        candidate_parts.append(primary)

        if in_eval:
            baseline, baseline_trace = transport.active_observation_rows_with_state(
                scored,
                score_column="active_priority",
                desired_eligible_column="transport_focus_eligible",
            )
            eval_scans.append(scan)
            eval_focus_parts.append(focus)
            eval_scored_parts.append(scored)
            eval_primary_trace_parts.append(primary_trace)
            eval_baseline_trace_parts.append(baseline_trace)
            eval_sensitivity_scored.append(scored)
            population_by_day[day_text] = _population_summary(causal_market)

    candidates, second_audit = add_second_features(
        pd.concat(candidate_parts, ignore_index=True),
        second_client,
    )
    minute_model, second_model = fit_stage2(candidates)
    scored_candidates = add_stage2_scores(candidates, minute_model, second_model)
    eval_active = scored_candidates.loc[
        scored_candidates["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()

    eval_scan = pd.concat(eval_scans, ignore_index=True)
    eval_focus = pd.concat(eval_focus_parts, ignore_index=True)
    eval_scored = pd.concat(eval_scored_parts, ignore_index=True)
    primary_subscription_trace = pd.concat(
        eval_primary_trace_parts, ignore_index=True
    )
    baseline_subscription_trace = pd.concat(
        eval_baseline_trace_parts, ignore_index=True
    )

    runtime_trace, runtime_audit = build_learned_runtime_trace(
        eval_active,
        eval_active,
    )
    second_coverage = float(
        pd.to_numeric(
            eval_active.get("active_seconds_60"), errors="coerce"
        ).fillna(0).gt(0).mean()
    )

    evaluation = evaluate_causal_handoff(
        eval_scan,
        eval_scored,
        eval_focus,
        eval_active,
        runtime_trace,
        primary_subscription_trace,
        runtime_audit,
        second_coverage,
        population_by_day,
    )

    primary_state = classify_subscription_state(
        primary_subscription_trace, eval_scan
    )
    baseline_state = classify_subscription_state(
        baseline_subscription_trace, eval_scan
    )

    hot = evaluation["hot"]
    primary_gate = bool(
        evaluation["promotion_gate_pass"]
        and float(evaluation["active_retention_given_focus"]) >= 0.95
        and float(primary_state["sparse_waste_rate"]) <= REQUEST149_SPARSE_WASTE
        and float(evaluation["transport_audit"]["mean_scoreable_active_occupancy"])
        >= REQUEST149_SCOREABLE_OCCUPANCY
        and float(hot["hot_within_1m_rate"]) >= REQUEST147_HOT_1M
        and float(hot["hot_within_2m_rate"]) >= REQUEST147_HOT_2M
        and float(hot["hot_within_5m_rate"]) >= REQUEST147_HOT_5M
    )

    sensitivity: dict[str, object] = {}
    all_eval_scored = pd.concat(eval_sensitivity_scored, ignore_index=True)
    for additions in SENSITIVITY_ADDITIONS:
        sens_active, sens_trace = balanced_active_rows_with_state(
            all_eval_scored,
            max_additions=additions,
        )
        sensitivity[str(additions)] = {
            "active_retention_given_focus": transport._retention_given(
                eval_focus, sens_active, eval_scan
            ),
            "transport_audit": transport.explicit_subscription_audit(sens_trace),
            "state_reason_audit": classify_subscription_state(
                sens_trace, eval_scan
            ),
        }

    summary: dict[str, object] = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "retention_value_definition": (
            "active_priority * transport_survival_probability"
        ),
        "opens_new_dates": False,
        "survival_horizon_minutes": SURVIVAL_HORIZON_MINUTES,
        "primary_max_additions": PRIMARY_MAX_ADDITIONS,
        "sensitivity_additions": list(SENSITIVITY_ADDITIONS),
        "survival_fit_auc_in_sample_diagnostic_only": survival_auc,
        "evaluation": evaluation,
        "primary_state_reason_audit": primary_state,
        "request147_state_reason_audit": baseline_state,
        "request149_sparse_waste_floor": REQUEST149_SPARSE_WASTE,
        "request149_scoreable_occupancy_floor": REQUEST149_SCOREABLE_OCCUPANCY,
        "sensitivity": sensitivity,
        "second_audit": second_audit,
        "promotion_gate_pass": primary_gate,
        "flatfile_stats": store.stats.to_dict(),
        "second_client_stats": second_client.stats.to_dict(),
    }
    return runtime_trace, summary


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
        store,
        scan_client,
        second_client,
        args.start,
        args.end,
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
