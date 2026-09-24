"""Request 156: test dynamic absolute-threshold Active sizing.

Development-only on already-opened request-151/152 sessions. Focus-60 remains
frozen so this request isolates the Active-20 question.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import active_transport_integrated_hierarchy as transport
from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .config import load_settings, require_flatfile_credentials
from .direct_active_admission import EVAL_DAYS
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
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
from .temporal_active_admission import (
    add_cross_within_horizon_targets,
    fit_temporal_admission_model,
    _focus_rows,
)

REQUEST_ID = 156
HORIZON = 3
THRESHOLD_QUANTILES = (0.25, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95)
TARGET_RETENTION = 0.95
FIXED_TOPK = 20


def derive_thresholds(
    focus_fit: pd.DataFrame,
    model,
) -> list[dict[str, float]]:
    features = focus_fit[
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
    probs = model.predict_proba(features)[:, 1]
    thresholds: list[dict[str, float]] = []
    seen: set[float] = set()
    for quantile in THRESHOLD_QUANTILES:
        value = float(np.quantile(probs, quantile))
        rounded = round(value, 12)
        if rounded in seen:
            continue
        seen.add(rounded)
        thresholds.append(
            {
                "fit_quantile": float(quantile),
                "threshold": value,
            }
        )
    return thresholds


def score_eval_focus(
    temporal_rows: pd.DataFrame,
    market_model,
    temporal_model,
) -> pd.DataFrame:
    scored, focus, _ = select_learned_focus(
        temporal_rows.loc[:, HAZARD_COLUMNS].copy(),
        market_model,
    )
    features = scored[
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
    scored["cross_within_3m_probability"] = temporal_model.predict_proba(features)[:, 1]
    focus_keys = focus.loc[:, ["trading_day", "ticker", "t"]].copy()
    result = temporal_rows.merge(
        scored.loc[
            :,
            [
                "trading_day",
                "ticker",
                "t",
                "cross_within_3m_probability",
            ],
        ],
        on=["trading_day", "ticker", "t"],
        how="inner",
        validate="one_to_one",
    )
    return result.merge(
        focus_keys.assign(focus_eligible=True),
        on=["trading_day", "ticker", "t"],
        how="inner",
        validate="one_to_one",
    )


def _decision_counts(
    selected: pd.DataFrame,
    focus: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    timeline = (
        focus.loc[:, ["trading_day", "t"]]
        .drop_duplicates()
        .sort_values(["trading_day", "t"])
    )
    counts = (
        selected.groupby(["trading_day", "t"], sort=False)
        .size()
        .rename("active_count")
        .reset_index()
    )
    merged = timeline.merge(counts, on=["trading_day", "t"], how="left")
    values = merged["active_count"].fillna(0).to_numpy(dtype=float)
    return values, len(merged)


def _capture_rate(
    focus: pd.DataFrame,
    active: pd.DataFrame,
    scan: pd.DataFrame,
) -> float | None:
    return transport._retention_given(focus, active, scan)


def _policy_metrics(
    selected: pd.DataFrame,
    focus: pd.DataFrame,
    scan: pd.DataFrame,
) -> dict[str, object]:
    counts, decisions = _decision_counts(selected, focus)
    target = pd.to_numeric(
        selected.get("target_cross_within_3m", pd.Series(dtype=float)),
        errors="coerce",
    )
    labelable = target.notna()
    positives = int((target.loc[labelable] == 1).sum()) if int(labelable.sum()) else 0
    selected_rows = int(len(selected))
    labelable_rows = int(labelable.sum())
    positive_rate = (
        float(positives / labelable_rows)
        if labelable_rows
        else None
    )
    capture = _capture_rate(focus, selected, scan)
    return {
        "active_retention_given_focus": capture,
        "selected_rows": selected_rows,
        "labelable_selected_rows": labelable_rows,
        "selected_3m_positive_rate": positive_rate,
        "selected_3m_false_attention_rate": (
            float(1.0 - positive_rate) if positive_rate is not None else None
        ),
        "decisions": decisions,
        "mean_active_count": float(np.mean(counts)) if len(counts) else 0.0,
        "median_active_count": float(np.median(counts)) if len(counts) else 0.0,
        "p90_active_count": float(np.quantile(counts, 0.90)) if len(counts) else 0.0,
        "p95_active_count": float(np.quantile(counts, 0.95)) if len(counts) else 0.0,
        "max_active_count": int(np.max(counts)) if len(counts) else 0,
        "min_active_count": int(np.min(counts)) if len(counts) else 0,
        "zero_active_decision_rate": float(np.mean(counts == 0)) if len(counts) else None,
        "over_20_decision_rate": float(np.mean(counts > FIXED_TOPK)) if len(counts) else None,
        "active_count_std": float(np.std(counts)) if len(counts) else 0.0,
    }


def _topk_policy(
    focus: pd.DataFrame,
    k: int = FIXED_TOPK,
) -> pd.DataFrame:
    ranked = focus.sort_values(
        ["trading_day", "t", "cross_within_3m_probability", "ticker"],
        ascending=[True, True, False, True],
        kind="stable",
    ).copy()
    ranked["rank"] = ranked.groupby(["trading_day", "t"], sort=False).cumcount() + 1
    return ranked.loc[ranked["rank"].le(k)].copy()


def _best_dynamic(
    threshold_metrics: list[dict[str, object]],
    fixed_metrics: dict[str, object],
) -> tuple[dict[str, object] | None, str]:
    fixed_retention = fixed_metrics["active_retention_given_focus"]
    feasible = [
        row for row in threshold_metrics
        if row["metrics"]["mean_active_count"] <= FIXED_TOPK
    ]
    if not feasible:
        return None, "no_threshold_at_or_below_fixed_burden"

    feasible.sort(
        key=lambda row: (
            float(row["metrics"]["active_retention_given_focus"] or -1.0),
            -float(row["metrics"]["mean_active_count"]),
        ),
        reverse=True,
    )
    best = feasible[0]
    best_retention = best["metrics"]["active_retention_given_focus"]
    best_mean = best["metrics"]["mean_active_count"]

    if (
        best_retention is not None
        and fixed_retention is not None
        and best_retention >= TARGET_RETENTION
        and best_mean < FIXED_TOPK
    ):
        diagnosis = "dynamic_threshold_reaches_target_with_lower_burden"
    elif (
        best_retention is not None
        and fixed_retention is not None
        and best_retention >= fixed_retention
        and best_mean < FIXED_TOPK
    ):
        diagnosis = "dynamic_threshold_dominates_fixed20"
    else:
        diagnosis = "fixed20_not_refuted"
    return best, diagnosis


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    start: date,
    end: date,
) -> dict[str, object]:
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
        raise ValueError("request 156 fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_hazard, ignore_index=True))
    temporal_fit = pd.concat(fit_temporal, ignore_index=True)
    focus_fit = _focus_rows(temporal_fit, market_model)
    temporal_model = fit_temporal_admission_model(
        focus_fit,
        horizon=HORIZON,
    )
    thresholds = derive_thresholds(focus_fit, temporal_model)

    eval_scans: list[pd.DataFrame] = []
    eval_focus: list[pd.DataFrame] = []
    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text not in EVAL_DAYS or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        temporal_rows = add_cross_within_horizon_targets(
            scan,
            horizons=(HORIZON,),
        )
        focus = score_eval_focus(
            temporal_rows,
            market_model,
            temporal_model,
        )
        eval_scans.append(scan)
        eval_focus.append(focus)

    scan = pd.concat(eval_scans, ignore_index=True)
    focus = pd.concat(eval_focus, ignore_index=True)
    found = sorted(scan["trading_day"].astype(str).unique())
    if found != EVAL_DAYS:
        raise ValueError(f"expected request 156 eval days {EVAL_DAYS}, found {found}")

    fixed = _topk_policy(focus)
    fixed_metrics = _policy_metrics(fixed, focus, scan)

    threshold_metrics: list[dict[str, object]] = []
    for spec in thresholds:
        threshold = float(spec["threshold"])
        selected = focus.loc[
            pd.to_numeric(
                focus["cross_within_3m_probability"],
                errors="coerce",
            ).ge(threshold)
        ].copy()
        threshold_metrics.append(
            {
                **spec,
                "metrics": _policy_metrics(selected, focus, scan),
            }
        )

    best, diagnosis = _best_dynamic(threshold_metrics, fixed_metrics)
    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "diagnostic_only": True,
        "eval_days": EVAL_DAYS,
        "focus_budget": 60,
        "fixed_topk": FIXED_TOPK,
        "horizon_minutes": HORIZON,
        "threshold_source": "fit-period Focus-60 score quantiles only",
        "threshold_quantiles": list(THRESHOLD_QUANTILES),
        "fixed_top20": fixed_metrics,
        "dynamic_thresholds": threshold_metrics,
        "best_dynamic_at_or_below_20_mean": best,
        "diagnosis": diagnosis,
        "interpretation_rule": {
            "dynamic_threshold_reaches_target_with_lower_burden": (
                "a fixed absolute threshold reaches >=95% retention while averaging under 20 Active names"
            ),
            "dynamic_threshold_dominates_fixed20": (
                "a fixed absolute threshold matches/exceeds Top-20 retention while averaging under 20 Active names"
            ),
            "fixed20_not_refuted": (
                "no tested fit-frozen threshold matches Top-20 retention at mean Active <=20"
            ),
            "no_threshold_at_or_below_fixed_burden": (
                "tested thresholds never average <=20 Active names"
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
