"""Request 147: actionable active transport on the corrected causal market population."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import active_transport_integrated_hierarchy as transport
from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
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
from .market_calendar import is_us_equity_trading_day
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
from .second_path_attention_probe import SECOND_CACHE_DIR
from .targeted_second_hot_reranker import add_second_features

REQUEST_ID = 147

REQUEST146_ACTIVE_RETENTION = 0.9360146252285192
REQUEST146_UNSCOREABLE_RATE = 0.06348717948717948
REQUEST146_MEAN_SCOREABLE_OCCUPANCY = 18.73025641025641
REQUEST146_HOT_1M = 0.49537512846865367
REQUEST146_HOT_2M = 0.5858170606372045
REQUEST146_HOT_5M = 0.6536485097636177


def fit_observability_model(rows: pd.DataFrame) -> HistGradientBoostingClassifier:
    """Predict whether a current ticker will print an exact next-minute bar.

    The label is future information used only for supervised fitting. Inference
    consumes only the same causal BASELINE_FEATURES used by the market hazard.
    """

    fit = rows.loc[rows["trading_day"].astype(str).le(FIT_END)].copy()
    if fit.empty:
        raise ValueError("observability fit population is empty")
    target = fit["has_exact_next_minute"].fillna(False).astype(int)
    if target.nunique() < 2:
        raise ValueError("observability fit needs both scoreable classes")

    model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    model.fit(
        fit[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan),
        target,
    )
    return model


def add_actionable_priority(
    scored_market: pd.DataFrame,
    observability_model: HistGradientBoostingClassifier,
) -> pd.DataFrame:
    """Factor conditional runner hazard by next-minute observability.

    market_hazard_probability estimates crossing risk among labelable current
    states. Multiplying by causal P(next-minute bar) gives an actionable
    next-step utility without inspecting whether the future bar actually exists.
    """

    result = scored_market.copy()
    result["next_minute_observability_probability"] = (
        observability_model.predict_proba(
            result[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan)
        )[:, 1]
    )
    hazard = pd.to_numeric(
        result["market_hazard_probability"], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    observable = pd.to_numeric(
        result["next_minute_observability_probability"], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    result["active_priority"] = hazard * observable
    return result


def add_unconditional_cross_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Make the stage-2 target the actual next wall-clock-minute event.

    If no aggregate prints in the next minute, an exact next-minute runner
    crossing cannot occur, so the event target is causally well-defined as 0.
    """

    result = frame.copy()
    raw = pd.to_numeric(result["target_next_cross"], errors="coerce")
    result["target_next_cross"] = raw.fillna(0).astype(int)
    return result


def _stronger_gate(summary: dict[str, object]) -> bool:
    evaluation = summary["evaluation"]
    transport_audit = evaluation["transport_audit"]
    hot = evaluation["hot"]

    return bool(
        evaluation["promotion_gate_pass"]
        and float(evaluation["active_retention_given_focus"]) >= 0.95
        and float(transport_audit["temporarily_unscoreable_subscription_rate"])
        < REQUEST146_UNSCOREABLE_RATE
        and float(transport_audit["mean_scoreable_active_occupancy"])
        > REQUEST146_MEAN_SCOREABLE_OCCUPANCY
        and float(hot["hot_within_1m_rate"]) >= REQUEST146_HOT_1M
        and float(hot["hot_within_2m_rate"]) >= REQUEST146_HOT_2M
        and float(hot["hot_within_5m_rate"]) >= REQUEST146_HOT_5M
    )


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    second_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fit_labelable: list[pd.DataFrame] = []
    fit_causal: list[pd.DataFrame] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text > FIT_END or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        fit_labelable.append(
            build_market_hazard_rows(scan).loc[:, HAZARD_COLUMNS].copy()
        )
        causal = build_market_hazard_frame(scan)
        fit_causal.append(causal.copy())

    if not fit_labelable or not fit_causal:
        raise ValueError("request 147 market fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_labelable, ignore_index=True))
    observability_model = fit_observability_model(
        pd.concat(fit_causal, ignore_index=True)
    )

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
        causal_market = build_market_hazard_frame(scan).copy()
        scored, focus, _ = select_learned_focus(
            causal_market.loc[:, HAZARD_COLUMNS].copy(),
            market_model,
        )
        scored = add_actionable_priority(scored, observability_model)
        active, subscription_trace = transport.active_observation_rows_with_state(
            scored,
            score_column="active_priority",
        )
        active = add_unconditional_cross_target(active)
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
        raise ValueError("request 147 produced incomplete fit/evaluation state")

    candidates, second_audit = add_second_features(
        pd.concat(candidate_parts, ignore_index=True),
        second_client,
    )
    minute_model, second_model = fit_stage2(candidates)
    scored_candidates = add_stage2_scores(
        candidates,
        minute_model,
        second_model,
    )
    eval_active = scored_candidates.loc[
        scored_candidates["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()

    eval_scan = pd.concat(eval_scans, ignore_index=True)
    eval_focus = pd.concat(eval_focus_parts, ignore_index=True)
    eval_scored = pd.concat(eval_scored_parts, ignore_index=True)
    subscription_trace = pd.concat(eval_subscription_parts, ignore_index=True)

    found = sorted(eval_scan["trading_day"].astype(str).unique())
    if found != EVAL_DAYS:
        raise ValueError(f"expected request 147 eval days {EVAL_DAYS}, found {found}")

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

    evaluation = evaluate_causal_handoff(
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

    obs_probability = pd.to_numeric(
        eval_scored["next_minute_observability_probability"],
        errors="coerce",
    )
    actual_observable = eval_scored["has_exact_next_minute"].fillna(False).astype(bool)
    priority = pd.to_numeric(eval_scored["active_priority"], errors="coerce")

    summary: dict[str, object] = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "market_fit_end": FIT_END,
        "stage2_fit_start": STAGE2_FIT_START,
        "stage2_fit_end": STAGE2_FIT_END,
        "eval_days": EVAL_DAYS,
        "active_priority_definition": (
            "market_hazard_probability * next_minute_observability_probability"
        ),
        "population_by_day": population_by_day,
        "second_audit": second_audit,
        "observability": {
            "eval_rows": int(len(eval_scored)),
            "actual_next_minute_observable_rate": (
                float(actual_observable.mean()) if len(eval_scored) else None
            ),
            "mean_predicted_observability_probability": (
                float(obs_probability.mean()) if len(obs_probability) else None
            ),
            "mean_active_priority": (
                float(priority.mean()) if len(priority) else None
            ),
        },
        "request146_baseline": {
            "active_retention_given_focus": REQUEST146_ACTIVE_RETENTION,
            "temporarily_unscoreable_subscription_rate": REQUEST146_UNSCOREABLE_RATE,
            "mean_scoreable_active_occupancy": REQUEST146_MEAN_SCOREABLE_OCCUPANCY,
            "hot_within_1m_rate": REQUEST146_HOT_1M,
            "hot_within_2m_rate": REQUEST146_HOT_2M,
            "hot_within_5m_rate": REQUEST146_HOT_5M,
        },
        "evaluation": evaluation,
        "improvement_gate_pass": False,
        "flatfile_stats": store.stats.to_dict(),
        "second_client_stats": second_client.stats.to_dict(),
    }
    summary["improvement_gate_pass"] = _stronger_gate(summary)
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
