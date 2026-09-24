"""Request 154: decompose the Focus-60 -> Active-20 admission score.

Development-only on request-151/152-opened sessions. No later date is opened.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .actionable_active_handoff import (
    add_actionable_priority,
    fit_observability_model,
    mark_focus_eligibility,
)
from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .balanced_retention_active_allocator import (
    add_balanced_retention_value,
    add_transport_survival_probability,
    build_transport_survival_rows,
    fit_transport_survival_model,
)
from .config import load_settings, require_flatfile_credentials
from .direct_active_admission import (
    EVAL_DAYS,
    MATERIAL_GAIN,
    TARGET_RETENTION,
    _variant_metrics,
    add_direct_admission_probability,
    fit_direct_admission_model,
)
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
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

REQUEST_ID = 154
VARIANTS = (
    "legacy",
    "hazard_only",
    "global_direct",
    "focus_direct",
)


def _apply_admission_score(
    scored: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    result = scored.copy()
    score = pd.to_numeric(result[score_column], errors="coerce").clip(
        lower=0.0, upper=1.0
    )
    survival = pd.to_numeric(
        result["transport_survival_probability"], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    result["active_priority"] = score
    result["balanced_retention_value"] = score * survival
    return result


def _diagnose_family(
    cap20: dict[str, float | None],
) -> dict[str, object]:
    legacy = cap20.get("legacy")
    if legacy is None:
        return {"diagnosis": "insufficient_support", "best_variant": None}

    candidates: list[tuple[float, str]] = []
    gains: list[tuple[float, float, str]] = []
    for name in VARIANTS:
        value = cap20.get(name)
        if value is None:
            continue
        gain = float(value - legacy)
        gains.append((gain, float(value), name))
        if name != "legacy" and value >= TARGET_RETENTION and gain >= MATERIAL_GAIN:
            candidates.append((float(value), name))

    if candidates:
        candidates.sort(reverse=True)
        return {
            "diagnosis": "structural_candidate_found",
            "best_variant": candidates[0][1],
        }

    nonlegacy = [item for item in gains if item[2] != "legacy"]
    if not nonlegacy:
        return {"diagnosis": "insufficient_support", "best_variant": None}
    best_gain, _, best_name = max(nonlegacy)
    if best_gain >= MATERIAL_GAIN:
        return {
            "diagnosis": "partial_structural_gain",
            "best_variant": best_name,
        }
    return {
        "diagnosis": "feature_information_bottleneck",
        "best_variant": best_name,
    }


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    start: date,
    end: date,
) -> dict[str, object]:
    fit_hazard: list[pd.DataFrame] = []
    fit_causal: list[pd.DataFrame] = []
    fit_survival: list[pd.DataFrame] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text > FIT_END or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        fit_hazard.append(build_market_hazard_rows(scan).loc[:, HAZARD_COLUMNS].copy())
        fit_causal.append(build_market_hazard_frame(scan))
        fit_survival.append(build_transport_survival_rows(scan))

    if not fit_hazard or not fit_causal:
        raise ValueError("request 154 fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_hazard, ignore_index=True))
    causal_fit = pd.concat(fit_causal, ignore_index=True)
    observability_model = fit_observability_model(causal_fit)
    global_direct_model = fit_direct_admission_model(causal_fit)
    _, focus_fit, _ = select_learned_focus(
        causal_fit.loc[:, HAZARD_COLUMNS].copy(),
        market_model,
    )
    focus_direct_model = fit_direct_admission_model(focus_fit)
    survival_model = fit_transport_survival_model(
        pd.concat(fit_survival, ignore_index=True)
    )

    eval_scans: list[pd.DataFrame] = []
    eval_focus: list[pd.DataFrame] = []
    eval_scored: list[pd.DataFrame] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text not in EVAL_DAYS or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        causal = build_market_hazard_frame(scan).copy()
        scored, focus, _ = select_learned_focus(
            causal.loc[:, HAZARD_COLUMNS].copy(),
            market_model,
        )
        scored = add_actionable_priority(scored, observability_model)
        scored = add_direct_admission_probability(scored, global_direct_model)
        scored["focus_direct_admission_probability"] = (
            focus_direct_model.predict_proba(
                scored[
                    [
                        "attention_score",
                        "attention_rank",
                        "return_from_previous_close_pct",
                        "minute_body_return_pct",
                        "minute_range_pct",
                        "log_minute_volume",
                        "log_minute_transactions",
                        "minute_return_1m_pct",
                        "return_accel_1m_pct",
                        "volume_ratio_prev1",
                        "transactions_ratio_prev1",
                    ]
                ].replace([np.inf, -np.inf], np.nan)
            )[:, 1]
        )
        scored = mark_focus_eligibility(scored, focus)
        scored = add_transport_survival_probability(scored, survival_model)
        scored = add_balanced_retention_value(scored)
        eval_scans.append(scan)
        eval_focus.append(focus)
        eval_scored.append(scored)

    scan = pd.concat(eval_scans, ignore_index=True)
    focus = pd.concat(eval_focus, ignore_index=True)
    scored = pd.concat(eval_scored, ignore_index=True)
    found = sorted(scan["trading_day"].astype(str).unique())
    if found != EVAL_DAYS:
        raise ValueError(f"expected request 154 eval days {EVAL_DAYS}, found {found}")

    score_columns = {
        "legacy": "active_priority",
        "hazard_only": "market_hazard_probability",
        "global_direct": "direct_admission_probability",
        "focus_direct": "focus_direct_admission_probability",
    }

    focus_rows = scored.loc[
        scored["transport_focus_eligible"].fillna(False).astype(bool)
    ].copy()
    y = pd.to_numeric(
        focus_rows["target_next_cross"], errors="coerce"
    ).fillna(0).astype(int)

    metrics: dict[str, object] = {}
    average_precision: dict[str, float | None] = {}
    cap20: dict[str, float | None] = {}
    legacy_original = scored["active_priority"].copy()
    for name, column in score_columns.items():
        variant = (
            scored.copy()
            if name == "legacy"
            else _apply_admission_score(scored, column)
        )
        if name == "legacy":
            variant["active_priority"] = legacy_original
        vm = _variant_metrics(variant, focus, scan)
        metrics[name] = vm
        cap20[name] = vm["20"]["active_retention_given_focus"]
        average_precision[name] = (
            float(average_precision_score(y, focus_rows[column]))
            if int(y.sum()) > 0
            else None
        )

    diagnosis = _diagnose_family(cap20)
    legacy20 = cap20["legacy"]
    gains = {
        name: (
            float(value - legacy20)
            if value is not None and legacy20 is not None
            else None
        )
        for name, value in cap20.items()
    }

    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "diagnostic_only": True,
        "eval_days": EVAL_DAYS,
        "variants": list(VARIANTS),
        "target_retention": TARGET_RETENTION,
        "material_gain_threshold": MATERIAL_GAIN,
        "focus_average_precision": average_precision,
        "metrics": metrics,
        "cap20_retention": cap20,
        "cap20_gain_vs_legacy": gains,
        **diagnosis,
        "interpretation_rule": {
            "structural_candidate_found": (
                "a nonlegacy structural score reaches 95% and gains at least 1pp"
            ),
            "partial_structural_gain": (
                "best structural score gains at least 1pp but remains below 95%"
            ),
            "feature_information_bottleneck": (
                "no structural score gains 1pp; enrich causal admission information"
            ),
        },
        "flatfile_stats": store.stats.to_dict(),
        "scan_client_stats": scan_client.stats.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
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
    summary = run_probe(store, scan_client, args.start, args.end)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
