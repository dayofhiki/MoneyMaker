"""Request 148: revalidate first-HOT economic opportunity on actionable transport."""

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
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hierarchical_attention_runtime import (
    STAGE2_FIT_END,
    STAGE2_FIT_START,
    add_stage2_scores,
    fit_stage2,
)
from .hot_economic_opportunity import (
    CAL_DAYS,
    ENTRY_FEATURES,
    EVAL_DAYS,
    FIT_DAYS,
    OPPORTUNITY_DAYS,
    _evaluate,
    _first_hot_feature_rows,
    _fit_models,
    fit_policy_opportunity_selector,
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

REQUEST_ID = 148


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
        fit_causal.append(build_market_hazard_frame(scan))

    if not fit_labelable or not fit_causal:
        raise ValueError("request 148 market fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_labelable, ignore_index=True))
    observability_model = fit_observability_model(
        pd.concat(fit_causal, ignore_index=True)
    )

    candidate_parts: list[pd.DataFrame] = []
    scan_parts: list[pd.DataFrame] = []
    needed_days = set(OPPORTUNITY_DAYS)

    for day in daterange(start, end):
        day_text = day.isoformat()
        in_stage2_fit = STAGE2_FIT_START <= day_text <= STAGE2_FIT_END
        in_opportunity = day_text in needed_days
        if (not in_stage2_fit and not in_opportunity) or not is_us_equity_trading_day(day):
            continue

        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        causal_market = build_market_hazard_frame(scan)
        scored, focus, _ = select_learned_focus(
            causal_market.loc[:, HAZARD_COLUMNS].copy(),
            market_model,
        )
        scored = add_actionable_priority(scored, observability_model)
        scored = mark_focus_eligibility(scored, focus)
        active_rows, _ = transport.active_observation_rows_with_state(
            scored,
            score_column="active_priority",
            desired_eligible_column="transport_focus_eligible",
        )
        active_rows = add_unconditional_cross_target(active_rows)
        candidate_parts.append(active_rows)
        if in_opportunity:
            scan_parts.append(scan)

    if not candidate_parts or not scan_parts:
        raise ValueError("request 148 produced no candidate/opportunity rows")

    candidates, second_audit = add_second_features(
        pd.concat(candidate_parts, ignore_index=True),
        second_client,
    )
    minute_model, second_model = fit_stage2(candidates)
    scored = add_stage2_scores(candidates, minute_model, second_model)
    opportunity_candidates = scored.loc[
        scored["trading_day"].astype(str).isin(OPPORTUNITY_DAYS)
    ].copy()
    opportunity_scan = pd.concat(scan_parts, ignore_index=True)

    found = sorted(opportunity_scan["trading_day"].astype(str).unique())
    if found != OPPORTUNITY_DAYS:
        raise ValueError(
            f"expected request 148 opportunity days {OPPORTUNITY_DAYS}, found {found}"
        )

    features, runtime_audit = _first_hot_feature_rows(
        opportunity_candidates,
        opportunity_scan,
    )
    missing = set(ENTRY_FEATURES) - set(features.columns)
    if missing:
        raise ValueError(f"request 148 first-HOT features missing: {sorted(missing)}")

    # Regression remains frozen from request 140B. The executable classifier
    # threshold is policy-safe: it is frozen across all CAL feature rows rather
    # than only rows whose future economic label happens to exist.
    _, regressor, offset, _, low, high = _fit_models(features)
    classifier, probability_gate = fit_policy_opportunity_selector(features)
    evaluation = _evaluate(
        features,
        classifier,
        regressor,
        offset,
        probability_gate,
    )

    day = features["trading_day"].astype(str)
    calibration = features.loc[day.isin(CAL_DAYS)]
    eval_rows = features.loc[day.isin(EVAL_DAYS)]
    summary: dict[str, object] = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "source_attention_design": "request147_actionable_active_transport",
        "fit_days": FIT_DAYS,
        "calibration_days": CAL_DAYS,
        "eval_days": EVAL_DAYS,
        "entry_features": ENTRY_FEATURES,
        "entry_feature_change_from_140b": False,
        "regression_fit_winsor_low_pct": low,
        "regression_fit_winsor_high_pct": high,
        "regression_calibration_offset_pct": offset,
        "calibration_first_hot_rows": int(len(calibration)),
        "evaluation_first_hot_rows": int(len(eval_rows)),
        "second_audit": second_audit,
        "runtime_audit": runtime_audit,
        "evaluation": evaluation,
        "flatfile_stats": store.stats.to_dict(),
        "second_client_stats": second_client.stats.to_dict(),
    }
    return features, summary


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
