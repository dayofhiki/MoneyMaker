"""Request 163: extend already-opened economic development history, then
freshly evaluate June 8-12 without changing the validated attention or economic
model family.

Reason for the change: Request 162 stopped before fitting because the new
selective attention path yielded only 1292/1275 labeled first-HOT episodes in
its original 5-day fit/calibration blocks, below the frozen >=1500 safety floor.
Request 163 keeps that floor and all model/gate definitions, but uses more
already-opened May history.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .focus180_temporal_retrain import select_focus_rows
from .fresh_validated_attention_economic_opportunity import (
    CANDIDATE_FEATURES,
    CANDIDATE_THRESHOLD,
    FRESH_EVAL_DAYS,
    _build_frozen_active,
    _evaluate_fresh,
)
from .hierarchical_active_features import FOCUS_CAP, HORIZON, fit_model
from .hierarchical_attention_runtime import (
    STAGE2_FIT_END,
    STAGE2_FIT_START,
    add_stage2_scores,
    fit_stage2,
)
from .hot_economic_opportunity import (
    CLASSIFIER_KWARGS,
    ENTRY_FEATURES,
    REGRESSOR_KWARGS,
    _first_hot_feature_rows,
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
from .second_path_attention_probe import SECOND_CACHE_DIR
from .targeted_second_hot_reranker import add_second_features
from .temporal_active_admission import add_cross_within_horizon_targets

REQUEST_ID = 163

# All of these dates were already opened in earlier research before Request 163.
EXTENDED_FIT_DAYS = [
    "2026-04-30",
    "2026-05-01",
    "2026-05-04",
    "2026-05-05",
    "2026-05-06",
    "2026-05-07",
    "2026-05-08",
]
EXTENDED_CAL_DAYS = [
    "2026-05-11",
    "2026-05-12",
    "2026-05-13",
    "2026-05-14",
    "2026-05-15",
    "2026-05-18",
    "2026-05-19",
    "2026-05-20",
]
OPPORTUNITY_DAYS = EXTENDED_FIT_DAYS + EXTENDED_CAL_DAYS + FRESH_EVAL_DAYS
MIN_DEVELOPMENT_LABELS = 1500


def _fit_extended_models(frame: pd.DataFrame):
    day = frame["trading_day"].astype(str)
    fit = frame.loc[day.isin(EXTENDED_FIT_DAYS)].copy()
    cal = frame.loc[day.isin(EXTENDED_CAL_DAYS)].copy()

    fit_target = pd.to_numeric(fit["oracle_best_base_pct"], errors="coerce")
    cal_target = pd.to_numeric(cal["oracle_best_base_pct"], errors="coerce")
    fit = fit.loc[fit_target.notna()].copy()
    cal = cal.loc[cal_target.notna()].copy()
    fit_target = pd.to_numeric(fit["oracle_best_base_pct"], errors="coerce")
    cal_target = pd.to_numeric(cal["oracle_best_base_pct"], errors="coerce")

    if len(fit) < MIN_DEVELOPMENT_LABELS or len(cal) < MIN_DEVELOPMENT_LABELS:
        raise ValueError(
            "request 163 keeps >=1500 labeled rows in fit/cal, "
            f"got {len(fit)}/{len(cal)}"
        )

    y_fit = fit_target.gt(0).astype(int)
    if y_fit.nunique() < 2:
        raise ValueError("request 163 fit labels need both classes")

    classifier = HistGradientBoostingClassifier(**CLASSIFIER_KWARGS)
    classifier.fit(
        fit[ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan),
        y_fit,
    )

    low, high = np.quantile(fit_target.to_numpy(dtype=float), [0.005, 0.995])
    regressor = HistGradientBoostingRegressor(**REGRESSOR_KWARGS)
    regressor.fit(
        fit[ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit_target.clip(lower=low, upper=high),
    )
    cal_prediction = regressor.predict(
        cal[ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan)
    )
    offset = float(np.mean(cal_target.to_numpy(dtype=float) - cal_prediction))
    return (
        classifier,
        regressor,
        offset,
        float(low),
        float(high),
        int(len(fit)),
        int(len(cal)),
    )


def _fit_extended_policy_selector(frame: pd.DataFrame):
    day = frame["trading_day"].astype(str)
    fit = frame.loc[day.isin(EXTENDED_FIT_DAYS)].copy()
    calibration = frame.loc[day.isin(EXTENDED_CAL_DAYS)].copy()

    fit_target = pd.to_numeric(fit["oracle_best_base_pct"], errors="coerce")
    fit = fit.loc[fit_target.notna()].copy()
    y_fit = pd.to_numeric(
        fit["oracle_best_base_pct"], errors="coerce"
    ).gt(0).astype(int)

    if len(fit) < MIN_DEVELOPMENT_LABELS:
        raise ValueError(
            f"request 163 policy fit has only {len(fit)} labeled rows"
        )
    if calibration.empty:
        raise ValueError("request 163 policy calibration rows are empty")
    if y_fit.nunique() < 2:
        raise ValueError("request 163 policy fit needs both classes")

    classifier = HistGradientBoostingClassifier(**CLASSIFIER_KWARGS)
    classifier.fit(
        fit[ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan),
        y_fit,
    )
    cal_probability = classifier.predict_proba(
        calibration[ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    gate = float(np.quantile(cal_probability, 0.75))
    return classifier, gate


def _fit_frozen_attention(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    start: date,
    end: date,
):
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
            add_cross_within_horizon_targets(scan, horizons=(HORIZON,))
        )
    if not fit_hazard or not fit_temporal:
        raise ValueError("request 163 attention fit population is empty")
    market_model = fit_market_hazard(pd.concat(fit_hazard, ignore_index=True))
    temporal_fit = pd.concat(fit_temporal, ignore_index=True)
    fit_focus180 = select_focus_rows(temporal_fit, market_model, FOCUS_CAP)
    active_model = fit_model(fit_focus180, CANDIDATE_FEATURES)
    return market_model, active_model


def _role_days(role: str, start: date, end: date) -> list[str]:
    if role == "stage2":
        return [
            day.isoformat()
            for day in daterange(start, end)
            if STAGE2_FIT_START <= day.isoformat() <= STAGE2_FIT_END
            and is_us_equity_trading_day(day)
        ]
    if role == "opportunity":
        return list(OPPORTUNITY_DAYS)
    raise ValueError(f"unknown role: {role}")


def prepare(
    role: str,
    start: date,
    end: date,
    candidates_output: Path,
    audit_output: Path,
    scan_output: Path | None,
) -> int:
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

    market_model, active_model = _fit_frozen_attention(
        store, scan_client, start, end
    )
    wanted = set(_role_days(role, start, end))
    candidate_parts: list[pd.DataFrame] = []
    scan_parts: list[pd.DataFrame] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text not in wanted or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        rows = add_cross_within_horizon_targets(scan, horizons=(HORIZON,))
        active = _build_frozen_active(rows, market_model, active_model)
        candidate_parts.append(active)
        if role == "opportunity":
            scan_parts.append(scan)

    if not candidate_parts:
        raise ValueError(f"request 163 {role} candidates are empty")

    candidates = pd.concat(candidate_parts, ignore_index=True)
    enriched, second_audit = add_second_features(candidates, second_client)

    candidates_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(candidates_output, index=False, compression="zstd")
    audit = {
        "request_id": REQUEST_ID,
        "role": role,
        "days": sorted(enriched["trading_day"].astype(str).unique()),
        "candidate_rows": int(len(enriched)),
        "second_audit": second_audit,
        "flatfile_stats": store.stats.to_dict(),
        "second_client_stats": second_client.stats.to_dict(),
    }
    audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )

    if role == "opportunity":
        if scan_output is None:
            raise ValueError("opportunity role needs --scan-output")
        scan = pd.concat(scan_parts, ignore_index=True)
        found = sorted(scan["trading_day"].astype(str).unique())
        if found != OPPORTUNITY_DAYS:
            raise ValueError(
                f"expected opportunity days {OPPORTUNITY_DAYS}, found {found}"
            )
        scan_output.parent.mkdir(parents=True, exist_ok=True)
        scan.to_parquet(scan_output, index=False, compression="zstd")

    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def evaluate(
    stage2_candidates_path: Path,
    opportunity_candidates_path: Path,
    opportunity_scan_path: Path,
    stage2_audit_path: Path,
    opportunity_audit_path: Path,
    output: Path,
    summary_path: Path,
) -> int:
    stage2_candidates = pd.read_parquet(stage2_candidates_path)
    opportunity_candidates = pd.read_parquet(opportunity_candidates_path)
    opportunity_scan = pd.read_parquet(opportunity_scan_path)

    minute_model, second_model = fit_stage2(stage2_candidates)
    scored = add_stage2_scores(
        opportunity_candidates, minute_model, second_model
    )
    features, runtime_audit = _first_hot_feature_rows(scored, opportunity_scan)

    missing = set(ENTRY_FEATURES) - set(features.columns)
    if missing:
        raise ValueError(
            f"request 163 first-HOT features missing: {sorted(missing)}"
        )

    (
        _,
        regressor,
        offset,
        low,
        high,
        labeled_fit_rows,
        labeled_cal_rows,
    ) = _fit_extended_models(features)
    classifier, probability_gate = _fit_extended_policy_selector(features)

    # This is the first point where June 8-12 outcome labels affect metrics.
    evaluation = _evaluate_fresh(
        features,
        classifier,
        regressor,
        offset,
        probability_gate,
    )

    day = features["trading_day"].astype(str)
    stage2_audit = json.loads(stage2_audit_path.read_text(encoding="utf-8"))
    opportunity_audit = json.loads(
        opportunity_audit_path.read_text(encoding="utf-8")
    )

    summary = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "parent_request": 162,
        "reason_for_change": (
            "keep the frozen >=1500 development support floor by extending "
            "already-opened history; no June outcome-driven tuning"
        ),
        "research_contract_changed": "development_history_only",
        "source_attention_design": (
            "request160_161_focus180_dynamic_active_hazard_probability"
        ),
        "focus_cap": FOCUS_CAP,
        "active_horizon_minutes": HORIZON,
        "active_threshold": CANDIDATE_THRESHOLD,
        "active_features": CANDIDATE_FEATURES,
        "hot_budget": 10,
        "opportunity_fit_days": EXTENDED_FIT_DAYS,
        "opportunity_calibration_days": EXTENDED_CAL_DAYS,
        "fresh_eval_days": FRESH_EVAL_DAYS,
        "min_development_labels": MIN_DEVELOPMENT_LABELS,
        "labeled_fit_rows": labeled_fit_rows,
        "labeled_calibration_rows": labeled_cal_rows,
        "entry_features": ENTRY_FEATURES,
        "entry_feature_change": False,
        "economic_definition_change": False,
        "classifier_hyperparameter_change": False,
        "regressor_hyperparameter_change": False,
        "promotion_gate_change": False,
        "regression_fit_winsor_low_pct": low,
        "regression_fit_winsor_high_pct": high,
        "regression_calibration_offset_pct": offset,
        "fit_first_hot_rows": int(day.isin(EXTENDED_FIT_DAYS).sum()),
        "calibration_first_hot_rows": int(day.isin(EXTENDED_CAL_DAYS).sum()),
        "evaluation_first_hot_rows": int(day.isin(FRESH_EVAL_DAYS).sum()),
        "stage2_prepare_audit": stage2_audit,
        "opportunity_prepare_audit": opportunity_audit,
        "runtime_audit": runtime_audit,
        "evaluation": evaluation,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False, compression="zstd")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("role", choices=["stage2", "opportunity"])
    prep.add_argument("start", type=date.fromisoformat)
    prep.add_argument("end", type=date.fromisoformat)
    prep.add_argument("--candidates-output", type=Path, required=True)
    prep.add_argument("--audit-output", type=Path, required=True)
    prep.add_argument("--scan-output", type=Path)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--stage2-candidates", type=Path, required=True)
    ev.add_argument("--opportunity-candidates", type=Path, required=True)
    ev.add_argument("--opportunity-scan", type=Path, required=True)
    ev.add_argument("--stage2-audit", type=Path, required=True)
    ev.add_argument("--opportunity-audit", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--summary", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(
            args.role,
            args.start,
            args.end,
            args.candidates_output,
            args.audit_output,
            args.scan_output,
        )
    return evaluate(
        args.stage2_candidates,
        args.opportunity_candidates,
        args.opportunity_scan,
        args.stage2_audit,
        args.opportunity_audit,
        args.output,
        args.summary,
    )


if __name__ == "__main__":
    raise SystemExit(main())
