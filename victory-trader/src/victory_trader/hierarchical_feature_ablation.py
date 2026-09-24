"""Request 160: ablate upstream Focus-context features for Active admission.

Development-only on already-opened May sessions. Compare market-hazard
probability and market rank separately and together under matched fit-period
observation burden.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .config import load_settings, require_flatfile_credentials
from .direct_active_admission import EVAL_DAYS
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hierarchical_active_features import (
    BASELINE_ACTIVE_FEATURES,
    FOCUS_CAP,
    HORIZON,
    MAX_BURDEN_RATIO,
    TARGET_CAPTURE,
    TARGET_SELECTED_FRACTION,
    fit_model,
    policy_metrics,
    score_model,
    select_active,
    threshold_for_fraction,
)
from .focus180_temporal_retrain import select_focus_rows
from .market_calendar import is_us_equity_trading_day
from .market_wide_focus_hazard import (
    FIT_END,
    HAZARD_COLUMNS,
    build_market_hazard_rows,
    fit_market_hazard,
)
from .massive_client import MassiveClient
from .multi_day import daterange
from .temporal_active_admission import add_cross_within_horizon_targets

REQUEST_ID = 160
TAIL_BASELINE_CAPTURED = 4

FEATURE_SETS = {
    "baseline": list(BASELINE_ACTIVE_FEATURES),
    "hazard_probability_only": [
        *BASELINE_ACTIVE_FEATURES,
        "market_hazard_probability",
    ],
    "hazard_rank_only": [
        *BASELINE_ACTIVE_FEATURES,
        "market_rank",
    ],
    "hazard_probability_and_rank": [
        *BASELINE_ACTIVE_FEATURES,
        "market_hazard_probability",
        "market_rank",
    ],
}


def diagnose(results: dict[str, dict[str, object]]) -> dict[str, object]:
    baseline = results["baseline"]
    base_rate = baseline["metrics"]["active_capture"]["supported_capture_rate"]
    base_mean = float(baseline["metrics"]["active_count"]["mean"])
    if base_rate is None:
        return {"diagnosis": "insufficient_support"}

    candidates: list[tuple[str, float, int, float]] = []
    for name, result in results.items():
        if name == "baseline":
            continue
        metrics = result["metrics"]
        rate = metrics["active_capture"]["supported_capture_rate"]
        if rate is None:
            continue
        tail = int(
            metrics["rank_bucket_capture"]["rank_61_180"]["captured"]
        )
        mean = float(metrics["active_count"]["mean"])
        burden_ratio = mean / base_mean if base_mean > 0 else float("inf")
        if (
            rate >= TARGET_CAPTURE
            and tail >= TAIL_BASELINE_CAPTURED
            and burden_ratio <= MAX_BURDEN_RATIO
        ):
            candidates.append((name, float(rate), tail, burden_ratio))

    if candidates:
        candidates.sort(
            key=lambda x: (x[1], x[2], -x[3]),
            reverse=True,
        )
        best = candidates[0]
        return {
            "diagnosis": "hierarchical_ablation_candidate",
            "best_variant": best[0],
            "best_capture": best[1],
            "best_tail_captured": best[2],
            "best_burden_ratio": best[3],
        }

    best_name = None
    best_rate = float(base_rate)
    best_tail = TAIL_BASELINE_CAPTURED
    for name, result in results.items():
        if name == "baseline":
            continue
        rate = result["metrics"]["active_capture"]["supported_capture_rate"]
        if rate is not None and float(rate) > best_rate:
            best_rate = float(rate)
            best_name = name
            best_tail = int(
                result["metrics"]["rank_bucket_capture"]["rank_61_180"][
                    "captured"
                ]
            )
    if best_name is not None and best_rate >= TARGET_CAPTURE:
        return {
            "diagnosis": "overall_gain_tail_tradeoff",
            "best_variant": best_name,
            "best_capture": best_rate,
            "best_tail_captured": best_tail,
        }
    return {
        "diagnosis": "hierarchical_features_not_decisive",
        "best_variant": best_name,
        "best_capture": best_rate,
        "best_tail_captured": best_tail,
    }


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
        raise ValueError("request 160 fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_hazard, ignore_index=True))
    temporal_fit = pd.concat(fit_temporal, ignore_index=True)
    fit_focus180 = select_focus_rows(
        temporal_fit,
        market_model,
        FOCUS_CAP,
    )

    models = {
        name: fit_model(fit_focus180, features)
        for name, features in FEATURE_SETS.items()
    }

    fit_scored = fit_focus180.copy()
    thresholds: dict[str, float] = {}
    for name, features in FEATURE_SETS.items():
        column = f"{name}_probability"
        fit_scored = score_model(
            fit_scored,
            models[name],
            features,
            column,
        )
        thresholds[name] = threshold_for_fraction(
            fit_scored,
            column,
            TARGET_SELECTED_FRACTION,
        )

    eval_scans: list[pd.DataFrame] = []
    eval_support: list[pd.DataFrame] = []
    eval_focus: list[pd.DataFrame] = []
    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text not in EVAL_DAYS or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        rows = add_cross_within_horizon_targets(
            scan,
            horizons=(HORIZON,),
        )
        focus = select_focus_rows(rows, market_model, FOCUS_CAP)
        for name, features in FEATURE_SETS.items():
            focus = score_model(
                focus,
                models[name],
                features,
                f"{name}_probability",
            )
        eval_scans.append(scan)
        eval_support.append(rows)
        eval_focus.append(focus)

    scan = pd.concat(eval_scans, ignore_index=True)
    support_rows = pd.concat(eval_support, ignore_index=True)
    focus = pd.concat(eval_focus, ignore_index=True)
    found = sorted(scan["trading_day"].astype(str).unique())
    if found != EVAL_DAYS:
        raise ValueError(f"expected request 160 eval days {EVAL_DAYS}, found {found}")

    results: dict[str, dict[str, object]] = {}
    for name, features in FEATURE_SETS.items():
        column = f"{name}_probability"
        active = select_active(
            focus,
            column,
            thresholds[name],
        )
        results[name] = {
            "features": features,
            "threshold": thresholds[name],
            "metrics": policy_metrics(
                support_rows,
                focus,
                active,
                scan,
                column,
            ),
        }

    diagnosis = diagnose(results)
    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "diagnostic_only": True,
        "eval_days": EVAL_DAYS,
        "focus_cap": FOCUS_CAP,
        "horizon_minutes": HORIZON,
        "target_selected_fraction": TARGET_SELECTED_FRACTION,
        "results": results,
        "target_capture": TARGET_CAPTURE,
        "tail_baseline_captured": TAIL_BASELINE_CAPTURED,
        "max_burden_ratio": MAX_BURDEN_RATIO,
        **diagnosis,
        "interpretation_rule": {
            "hierarchical_ablation_candidate": (
                "a variant reaches >=95% overall capture without reducing the "
                "4/14 tail baseline and stays within 10% mean Active burden"
            ),
            "overall_gain_tail_tradeoff": (
                "a variant reaches >=95% overall capture but only by worsening "
                "rank-61-180 tail capture"
            ),
            "hierarchical_features_not_decisive": (
                "no ablated hierarchical feature set reaches a clean candidate"
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
