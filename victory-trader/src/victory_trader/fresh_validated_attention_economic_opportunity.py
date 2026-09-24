"""Request 162: fresh economic-opportunity revalidation on validated attention.

Freeze request-160/161 attention and re-run the request-148 first-HOT economic
learnability question. Economic fit/calibration reuse already-opened dates;
evaluation opens only 2026-06-08 through 2026-06-12.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .actionable_active_handoff import add_unconditional_cross_target
from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .focus180_temporal_retrain import select_focus_rows
from .hierarchical_active_features import (
    BASELINE_ACTIVE_FEATURES,
    FOCUS_CAP,
    HORIZON,
    fit_model,
    score_model,
    select_active,
)
from .hierarchical_attention_runtime import (
    STAGE2_FIT_END,
    STAGE2_FIT_START,
    add_stage2_scores,
    fit_stage2,
)
from .hot_economic_opportunity import (
    CAL_DAYS,
    CLASSIFIER_KWARGS,
    ENTRY_FEATURES,
    FIT_DAYS,
    _first_hot_feature_rows,
    _fit_models,
    _safe_ap,
    _safe_auc,
    _safe_brier,
    _safe_spearman,
    fit_policy_opportunity_selector,
)
from .market_calendar import is_us_equity_trading_day
from .market_wide_focus_hazard import (
    FIT_END,
    HAZARD_COLUMNS,
    build_market_hazard_rows,
    fit_market_hazard,
)
from .massive_client import MassiveClient
from .multi_day import daterange
from .recurrent_watch_hazard import _state_history
from .second_path_attention_probe import SECOND_CACHE_DIR
from .targeted_second_hot_reranker import add_second_features
from .temporal_active_admission import add_cross_within_horizon_targets

REQUEST_ID = 162
FRESH_EVAL_DAYS = [
    "2026-06-08",
    "2026-06-09",
    "2026-06-10",
    "2026-06-11",
    "2026-06-12",
]
OPPORTUNITY_DAYS = FIT_DAYS + CAL_DAYS + FRESH_EVAL_DAYS
CANDIDATE_THRESHOLD = 0.003093198572895277
CANDIDATE_FEATURES = [
    *BASELINE_ACTIVE_FEATURES,
    "market_hazard_probability",
]


def _build_frozen_active(
    rows: pd.DataFrame,
    market_model,
    active_model,
) -> pd.DataFrame:
    focus = select_focus_rows(rows, market_model, FOCUS_CAP)
    focus = score_model(
        focus,
        active_model,
        CANDIDATE_FEATURES,
        "candidate_probability",
    )
    active = select_active(
        focus,
        "candidate_probability",
        CANDIDATE_THRESHOLD,
    ).copy()

    # Under the validated dynamic-threshold architecture every emitted row is
    # desired Active now; there is no fixed-slot retained-only population.
    active["transport_desired_now"] = True

    # Preserve the old HOT reranker feature contract. Its stage-1 probability
    # represented upstream next-step market hazard; use the learned market-wide
    # hazard already frozen in the validated attention architecture.
    active["stage1_hazard_probability"] = pd.to_numeric(
        active["market_hazard_probability"],
        errors="coerce",
    )

    # State-age features are derived causally from consecutive dynamic Active
    # membership. All rows enter the HOT allocator as WATCH candidates.
    active["state"] = "watch"
    active = _state_history(active)

    # Keep request-148 stage-2 label semantics unchanged.
    active = add_unconditional_cross_target(active)
    return active


def _evaluate_fresh(
    frame: pd.DataFrame,
    classifier,
    regressor,
    offset: float,
    probability_gate: float,
) -> dict[str, object]:
    evaluation = frame.loc[
        frame["trading_day"].astype(str).isin(FRESH_EVAL_DAYS)
    ].copy()
    target = pd.to_numeric(
        evaluation["oracle_best_base_pct"],
        errors="coerce",
    )
    valid = target.notna()
    evaluation = evaluation.loc[valid].copy()
    target = pd.to_numeric(
        evaluation["oracle_best_base_pct"],
        errors="coerce",
    )
    y = target.gt(0).astype(int)

    x = evaluation[ENTRY_FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    probability = classifier.predict_proba(x)[:, 1]
    predicted_value = regressor.predict(x) + offset
    evaluation["opportunity_probability"] = probability
    evaluation["predicted_oracle_base_pct"] = predicted_value
    evaluation["selected_top_quartile"] = probability >= probability_gate

    auc = _safe_auc(y, evaluation["opportunity_probability"])
    ap = _safe_ap(y, evaluation["opportunity_probability"])
    brier = _safe_brier(y, evaluation["opportunity_probability"])
    spearman = _safe_spearman(
        evaluation["oracle_best_base_pct"],
        evaluation["predicted_oracle_base_pct"],
    )

    selected = evaluation.loc[
        evaluation["selected_top_quartile"]
    ].copy()
    all_mean = float(target.mean()) if len(evaluation) else None
    all_positive = float(y.mean()) if len(evaluation) else None
    selected_target = pd.to_numeric(
        selected["oracle_best_base_pct"],
        errors="coerce",
    )
    selected_positive = selected_target.gt(0)

    by_day: dict[str, object] = {}
    auc_positive_days = 0
    spearman_positive_days = 0
    selected_mean_nonlower_days = 0
    support_ok = True
    coverage_ok = True

    for day in FRESH_EVAL_DAYS:
        source = frame.loc[
            frame["trading_day"].astype(str).eq(day)
        ].copy()
        source_labeled = pd.to_numeric(
            source["oracle_best_base_pct"],
            errors="coerce",
        ).notna()
        day_eval = evaluation.loc[
            evaluation["trading_day"].astype(str).eq(day)
        ].copy()

        if len(day_eval) < 100:
            support_ok = False
        coverage = (
            float(source_labeled.mean()) if len(source) else 0.0
        )
        if coverage < 0.95:
            coverage_ok = False

        day_target = pd.to_numeric(
            day_eval["oracle_best_base_pct"],
            errors="coerce",
        )
        day_y = day_target.gt(0).astype(int)
        day_auc = _safe_auc(
            day_y,
            day_eval["opportunity_probability"],
        )
        day_spearman = _safe_spearman(
            day_eval["oracle_best_base_pct"],
            day_eval["predicted_oracle_base_pct"],
        )
        if day_auc is not None and day_auc > 0.5:
            auc_positive_days += 1
        if day_spearman is not None and day_spearman > 0:
            spearman_positive_days += 1

        day_selected = day_eval.loc[
            day_eval["selected_top_quartile"]
        ]
        day_selected_target = pd.to_numeric(
            day_selected["oracle_best_base_pct"],
            errors="coerce",
        )
        day_all_mean = (
            float(day_target.mean()) if len(day_eval) else None
        )
        day_selected_mean = (
            float(day_selected_target.mean())
            if len(day_selected)
            else None
        )
        if (
            day_all_mean is not None
            and day_selected_mean is not None
            and day_selected_mean >= day_all_mean
        ):
            selected_mean_nonlower_days += 1

        by_day[day] = {
            "first_hot_rows": int(len(source)),
            "labeled_rows": int(len(day_eval)),
            "label_coverage": coverage,
            "positive_rate": (
                float(day_y.mean()) if len(day_eval) else None
            ),
            "auc": day_auc,
            "spearman": day_spearman,
            "all_oracle_mean_pct": day_all_mean,
            "selected_rows": int(len(day_selected)),
            "selected_oracle_mean_pct": day_selected_mean,
            "selected_positive_rate": (
                float(day_selected_target.gt(0).mean())
                if len(day_selected)
                else None
            ),
        }

    selected_mean = (
        float(selected_target.mean()) if len(selected) else None
    )
    selected_positive_rate = (
        float(selected_positive.mean()) if len(selected) else None
    )

    gate = bool(
        support_ok
        and coverage_ok
        and auc is not None
        and auc >= 0.55
        and auc_positive_days >= 4
        and spearman is not None
        and spearman >= 0.05
        and spearman_positive_days >= 4
        and selected_mean is not None
        and all_mean is not None
        and selected_mean > 0
        and selected_mean > all_mean
        and selected_positive_rate is not None
        and all_positive is not None
        and selected_positive_rate >= all_positive + 0.05
        and selected_mean_nonlower_days >= 4
    )

    return {
        "eval_days": FRESH_EVAL_DAYS,
        "labeled_rows": int(len(evaluation)),
        "opportunity_positive_rate": all_positive,
        "oracle_base_mean_pct": all_mean,
        "classifier_auc": auc,
        "classifier_average_precision": ap,
        "classifier_brier": brier,
        "value_spearman": spearman,
        "calibration_probability_p75_gate": probability_gate,
        "selected_rows": int(len(selected)),
        "selected_rate": (
            float(len(selected) / len(evaluation))
            if len(evaluation)
            else None
        ),
        "selected_oracle_base_mean_pct": selected_mean,
        "selected_positive_rate": selected_positive_rate,
        "auc_above_random_days": auc_positive_days,
        "positive_spearman_days": spearman_positive_days,
        "selected_mean_nonlower_days": selected_mean_nonlower_days,
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
    fit_hazard: list[pd.DataFrame] = []
    fit_temporal: list[pd.DataFrame] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text > FIT_END or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        fit_hazard.append(
            build_market_hazard_rows(scan).loc[:, HAZARD_COLUMNS].copy()
        )
        fit_temporal.append(
            add_cross_within_horizon_targets(
                scan,
                horizons=(HORIZON,),
            )
        )

    if not fit_hazard or not fit_temporal:
        raise ValueError("request 162 frozen attention fit is empty")

    market_model = fit_market_hazard(
        pd.concat(fit_hazard, ignore_index=True)
    )
    temporal_fit = pd.concat(fit_temporal, ignore_index=True)
    fit_focus180 = select_focus_rows(
        temporal_fit,
        market_model,
        FOCUS_CAP,
    )
    active_model = fit_model(
        fit_focus180,
        CANDIDATE_FEATURES,
    )

    candidate_parts: list[pd.DataFrame] = []
    scan_parts: list[pd.DataFrame] = []
    needed_days = set(OPPORTUNITY_DAYS)

    for day in daterange(start, end):
        day_text = day.isoformat()
        in_stage2_fit = STAGE2_FIT_START <= day_text <= STAGE2_FIT_END
        in_opportunity = day_text in needed_days
        if (
            (not in_stage2_fit and not in_opportunity)
            or not is_us_equity_trading_day(day)
        ):
            continue

        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        rows = add_cross_within_horizon_targets(
            scan,
            horizons=(HORIZON,),
        )
        active = _build_frozen_active(
            rows,
            market_model,
            active_model,
        )
        candidate_parts.append(active)
        if in_opportunity:
            scan_parts.append(scan)

    if not candidate_parts or not scan_parts:
        raise ValueError(
            "request 162 produced no active candidates/opportunity scans"
        )

    candidates = pd.concat(candidate_parts, ignore_index=True)
    candidates, second_audit = add_second_features(
        candidates,
        second_client,
    )
    minute_model, second_model = fit_stage2(candidates)
    scored = add_stage2_scores(
        candidates,
        minute_model,
        second_model,
    )

    opportunity_candidates = scored.loc[
        scored["trading_day"].astype(str).isin(OPPORTUNITY_DAYS)
    ].copy()
    opportunity_scan = pd.concat(scan_parts, ignore_index=True)

    found = sorted(
        opportunity_scan["trading_day"].astype(str).unique()
    )
    if found != OPPORTUNITY_DAYS:
        raise ValueError(
            f"expected request 162 opportunity days {OPPORTUNITY_DAYS}, "
            f"found {found}"
        )

    features, runtime_audit = _first_hot_feature_rows(
        opportunity_candidates,
        opportunity_scan,
    )
    missing = set(ENTRY_FEATURES) - set(features.columns)
    if missing:
        raise ValueError(
            f"request 162 first-HOT features missing: {sorted(missing)}"
        )

    _, regressor, offset, _, low, high = _fit_models(features)
    classifier, probability_gate = fit_policy_opportunity_selector(
        features
    )
    evaluation = _evaluate_fresh(
        features,
        classifier,
        regressor,
        offset,
        probability_gate,
    )

    day = features["trading_day"].astype(str)
    fit_rows = features.loc[day.isin(FIT_DAYS)]
    cal_rows = features.loc[day.isin(CAL_DAYS)]
    eval_rows = features.loc[day.isin(FRESH_EVAL_DAYS)]

    summary: dict[str, object] = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": True,
        "fresh_economic_validation": True,
        "source_attention_design": (
            "request160_161_focus180_dynamic_active_hazard_probability"
        ),
        "focus_cap": FOCUS_CAP,
        "active_horizon_minutes": HORIZON,
        "active_threshold": CANDIDATE_THRESHOLD,
        "active_features": CANDIDATE_FEATURES,
        "hot_budget": 10,
        "opportunity_fit_days": FIT_DAYS,
        "opportunity_calibration_days": CAL_DAYS,
        "fresh_eval_days": FRESH_EVAL_DAYS,
        "entry_features": ENTRY_FEATURES,
        "entry_feature_change_from_140b": False,
        "opportunity_classifier_model": CLASSIFIER_KWARGS,
        "regression_fit_winsor_low_pct": low,
        "regression_fit_winsor_high_pct": high,
        "regression_calibration_offset_pct": offset,
        "fit_first_hot_rows": int(len(fit_rows)),
        "calibration_first_hot_rows": int(len(cal_rows)),
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
    output.to_parquet(
        args.output,
        index=False,
        compression="zstd",
    )
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
