"""Request 157: optimize the fixed learned-Focus count before fresh validation.

Development-only on the already-opened May block. The request-156 Active
threshold is frozen so this request changes only the Focus Top-K ceiling.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import MINUTE_MS
from .config import load_settings, require_flatfile_credentials
from .direct_active_admission import EVAL_DAYS
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hierarchical_attention_runtime import MODEL_KWARGS
from .market_calendar import is_us_equity_trading_day
from .market_wide_focus_hazard import (
    BASELINE_FEATURES,
    FIT_END,
    HAZARD_COLUMNS,
    build_market_hazard_rows,
    fit_market_hazard,
)
from .massive_client import MassiveClient
from .multi_day import daterange
from .temporal_active_admission import (
    add_cross_within_horizon_targets,
    fit_temporal_admission_model,
    _focus_rows,
)

REQUEST_ID = 157
HORIZON = 3
BASELINE_FOCUS_CAP = 60
FOCUS_CAPS: tuple[int | None, ...] = (20, 30, 45, 60, 75, 90, 120, 180, None)
ACTIVE_THRESHOLD = 0.0040281217293971616
FOCUS_NEAR_MAX_TOLERANCE = 0.0025
ACTIVE_NEAR_MAX_TOLERANCE = 0.0050


def score_market_rows(rows: pd.DataFrame, market_model, temporal_model) -> pd.DataFrame:
    result = rows.copy()
    result["market_hazard_probability"] = market_model.predict_proba(
        result[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    result["cross_within_3m_probability"] = temporal_model.predict_proba(
        result[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    return result


def select_focus(scored: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    ranked = scored.sort_values(
        ["trading_day", "t", "market_hazard_probability", "ticker"],
        ascending=[True, True, False, True],
        kind="stable",
    ).copy()
    if cap is None:
        return ranked
    return ranked.groupby(["trading_day", "t"], sort=False).head(cap).copy()


def select_dynamic_active(focus: pd.DataFrame) -> pd.DataFrame:
    return focus.loc[
        pd.to_numeric(
            focus["cross_within_3m_probability"], errors="coerce"
        ).ge(ACTIVE_THRESHOLD)
    ].copy()


def _timeline_counts(
    selected: pd.DataFrame,
    timeline_rows: pd.DataFrame,
) -> np.ndarray:
    timeline = (
        timeline_rows.loc[:, ["trading_day", "t"]]
        .drop_duplicates()
        .sort_values(["trading_day", "t"])
    )
    counts = (
        selected.groupby(["trading_day", "t"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    merged = timeline.merge(counts, on=["trading_day", "t"], how="left")
    return merged["count"].fillna(0).to_numpy(dtype=float)


def _count_metrics(selected: pd.DataFrame, timeline_rows: pd.DataFrame) -> dict[str, object]:
    counts = _timeline_counts(selected, timeline_rows)
    return {
        "mean": float(np.mean(counts)) if len(counts) else 0.0,
        "median": float(np.median(counts)) if len(counts) else 0.0,
        "p90": float(np.quantile(counts, 0.90)) if len(counts) else 0.0,
        "p95": float(np.quantile(counts, 0.95)) if len(counts) else 0.0,
        "min": int(np.min(counts)) if len(counts) else 0,
        "max": int(np.max(counts)) if len(counts) else 0,
        "std": float(np.std(counts)) if len(counts) else 0.0,
        "zero_rate": float(np.mean(counts == 0)) if len(counts) else None,
    }


def _supported_capture(
    scored: pd.DataFrame,
    selected: pd.DataFrame,
    scan: pd.DataFrame,
) -> dict[str, object]:
    scored_keys = {
        (str(r.trading_day), str(r.ticker).upper(), int(r.t))
        for r in scored.itertuples(index=False)
    }
    selected_keys = {
        (str(r.trading_day), str(r.ticker).upper(), int(r.t))
        for r in selected.itertuples(index=False)
    }
    total = 0
    supported = 0
    captured = 0
    crossings = scan.loc[
        scan["runner_cross_now"].fillna(False).astype(bool),
        ["trading_day", "ticker", "t"],
    ]
    for row in crossings.itertuples(index=False):
        total += 1
        prior = (
            str(row.trading_day),
            str(row.ticker).upper(),
            int(row.t) - MINUTE_MS,
        )
        if prior not in scored_keys:
            continue
        supported += 1
        if prior in selected_keys:
            captured += 1
    return {
        "runner_crossings": total,
        "supported_exact_prior_crossings": supported,
        "captured_supported_crossings": captured,
        "supported_capture_rate": (
            float(captured / supported) if supported else None
        ),
        "total_crossing_capture_rate": (
            float(captured / total) if total else None
        ),
    }


def _active_precision(active: pd.DataFrame) -> dict[str, object]:
    target = pd.to_numeric(
        active.get("target_cross_within_3m", pd.Series(dtype=float)),
        errors="coerce",
    )
    labelable = target.notna()
    n = int(labelable.sum())
    positives = int((target.loc[labelable] == 1).sum()) if n else 0
    return {
        "labelable_rows": n,
        "positive_rows": positives,
        "three_minute_positive_rate": float(positives / n) if n else None,
    }


def _policy_metrics(
    scored: pd.DataFrame,
    focus: pd.DataFrame,
    active: pd.DataFrame,
    scan: pd.DataFrame,
) -> dict[str, object]:
    focus_capture = _supported_capture(scored, focus, scan)
    active_capture = _supported_capture(scored, active, scan)
    focus_rate = focus_capture["supported_capture_rate"]
    active_rate = active_capture["supported_capture_rate"]
    conditional = (
        float(active_rate / focus_rate)
        if focus_rate not in (None, 0) and active_rate is not None
        else None
    )
    return {
        "focus_count": _count_metrics(focus, scored),
        "active_count": _count_metrics(active, scored),
        "focus_capture": focus_capture,
        "active_capture": active_capture,
        "active_capture_given_focus": conditional,
        "active_selection_quality": _active_precision(active),
    }


def choose_knee(results: list[dict[str, object]]) -> tuple[dict[str, object] | None, str]:
    focus_rates = [
        float(r["metrics"]["focus_capture"]["supported_capture_rate"])
        for r in results
        if r["metrics"]["focus_capture"]["supported_capture_rate"] is not None
    ]
    active_rates = [
        float(r["metrics"]["active_capture"]["supported_capture_rate"])
        for r in results
        if r["metrics"]["active_capture"]["supported_capture_rate"] is not None
    ]
    if not focus_rates or not active_rates:
        return None, "insufficient_support"

    best_focus = max(focus_rates)
    best_active = max(active_rates)
    finite = [
        r
        for r in results
        if r["focus_cap"] is not None
        and r["metrics"]["focus_capture"]["supported_capture_rate"]
        >= best_focus - FOCUS_NEAR_MAX_TOLERANCE
        and r["metrics"]["active_capture"]["supported_capture_rate"]
        >= best_active - ACTIVE_NEAR_MAX_TOLERANCE
    ]
    if not finite:
        return None, "focus_cap_ceiling_unresolved"

    finite.sort(key=lambda r: int(r["focus_cap"]))
    knee = finite[0]
    cap = int(knee["focus_cap"])
    if cap < BASELINE_FOCUS_CAP:
        diagnosis = "focus_cap_reduction_supported"
    elif cap == BASELINE_FOCUS_CAP:
        diagnosis = "focus60_near_knee"
    else:
        diagnosis = "focus_cap_increase_supported"
    return knee, diagnosis


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
        raise ValueError("request 157 fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_hazard, ignore_index=True))
    temporal_fit = pd.concat(fit_temporal, ignore_index=True)
    focus60_fit = _focus_rows(temporal_fit, market_model)
    temporal_model = fit_temporal_admission_model(
        focus60_fit,
        horizon=HORIZON,
    )

    eval_scans: list[pd.DataFrame] = []
    eval_scored: list[pd.DataFrame] = []
    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text not in EVAL_DAYS or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        temporal_rows = add_cross_within_horizon_targets(
            scan, horizons=(HORIZON,)
        )
        scored = score_market_rows(temporal_rows, market_model, temporal_model)
        eval_scans.append(scan)
        eval_scored.append(scored)

    scan = pd.concat(eval_scans, ignore_index=True)
    scored = pd.concat(eval_scored, ignore_index=True)
    found = sorted(scan["trading_day"].astype(str).unique())
    if found != EVAL_DAYS:
        raise ValueError(f"expected request 157 eval days {EVAL_DAYS}, found {found}")

    results: list[dict[str, object]] = []
    for cap in FOCUS_CAPS:
        focus = select_focus(scored, cap)
        active = select_dynamic_active(focus)
        results.append(
            {
                "focus_cap": cap,
                "label": "unbounded" if cap is None else f"top_{cap}",
                "metrics": _policy_metrics(scored, focus, active, scan),
            }
        )

    knee, diagnosis = choose_knee(results)
    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "diagnostic_only": True,
        "eval_days": EVAL_DAYS,
        "focus_caps": list(FOCUS_CAPS),
        "baseline_focus_cap": BASELINE_FOCUS_CAP,
        "active_policy": {
            "horizon_minutes": HORIZON,
            "threshold": ACTIVE_THRESHOLD,
            "source": "request 156 q65 winner, frozen before request 157",
        },
        "temporal_model_training_population": "fit-period learned Focus-60 only",
        "focus_near_max_tolerance": FOCUS_NEAR_MAX_TOLERANCE,
        "active_near_max_tolerance": ACTIVE_NEAR_MAX_TOLERANCE,
        "results": results,
        "recommended_knee": knee,
        "diagnosis": diagnosis,
        "interpretation_rule": {
            "focus_cap_reduction_supported": (
                "a cap below 60 is the smallest finite cap within 0.25pp of best Focus capture "
                "and 0.50pp of best downstream Active capture"
            ),
            "focus60_near_knee": (
                "60 is the smallest finite cap satisfying both near-max capture criteria"
            ),
            "focus_cap_increase_supported": (
                "a cap above 60 is required to satisfy both near-max capture criteria"
            ),
            "focus_cap_ceiling_unresolved": (
                "no tested finite cap is sufficiently close to the unbounded capture frontier"
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
