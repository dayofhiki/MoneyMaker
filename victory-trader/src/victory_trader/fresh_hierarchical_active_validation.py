"""Request 161: fresh validation of the frozen hierarchical Active candidate.

This is the first chronologically later validation block after Requests 151-160.
No June outcome may change Focus cap, features, models, thresholds, or dates.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .focus_cap_frontier import _supported_capture
from .focus180_temporal_retrain import select_focus_rows
from .hierarchical_active_features import (
    BASELINE_ACTIVE_FEATURES,
    FOCUS_CAP,
    HORIZON,
    fit_model,
    policy_metrics,
    score_model,
    select_active,
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
from .temporal_active_admission import add_cross_within_horizon_targets

REQUEST_ID = 161
FRESH_EVAL_DAYS = [
    "2026-06-01",
    "2026-06-02",
    "2026-06-03",
    "2026-06-04",
    "2026-06-05",
]
BASELINE_THRESHOLD = 0.003429286145316252
CANDIDATE_THRESHOLD = 0.003093198572895277
TARGET_FOCUS_CAPTURE = 0.99
TARGET_ACTIVE_CAPTURE = 0.95
MIN_SUPPORTED_CROSSINGS = 100

CANDIDATE_FEATURES = [
    *BASELINE_ACTIVE_FEATURES,
    "market_hazard_probability",
]


def _daily_capture(
    support_rows: pd.DataFrame,
    focus: pd.DataFrame,
    active: pd.DataFrame,
    scan: pd.DataFrame,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for day in FRESH_EVAL_DAYS:
        support_day = support_rows.loc[
            support_rows["trading_day"].astype(str).eq(day)
        ]
        focus_day = focus.loc[focus["trading_day"].astype(str).eq(day)]
        active_day = active.loc[active["trading_day"].astype(str).eq(day)]
        scan_day = scan.loc[scan["trading_day"].astype(str).eq(day)]
        out[day] = {
            "focus": _supported_capture(support_day, focus_day, scan_day),
            "active": _supported_capture(support_day, active_day, scan_day),
        }
    return out


def _diagnose(
    focus_capture: dict[str, object],
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    focus_rate = focus_capture["supported_capture_rate"]
    base_rate = baseline["active_capture"]["supported_capture_rate"]
    cand_rate = candidate["active_capture"]["supported_capture_rate"]
    support = int(focus_capture["supported_exact_prior_crossings"])

    if (
        focus_rate is None
        or base_rate is None
        or cand_rate is None
        or support < MIN_SUPPORTED_CROSSINGS
    ):
        return {"diagnosis": "fresh_validation_insufficient_support"}

    feature_gain = float(cand_rate - base_rate)
    if (
        focus_rate >= TARGET_FOCUS_CAPTURE
        and cand_rate >= TARGET_ACTIVE_CAPTURE
        and feature_gain >= 0.0
    ):
        diagnosis = "fresh_candidate_validated"
    elif (
        focus_rate >= TARGET_FOCUS_CAPTURE
        and cand_rate >= TARGET_ACTIVE_CAPTURE
    ):
        diagnosis = "fresh_absolute_pass_feature_uplift_not_confirmed"
    elif focus_rate < TARGET_FOCUS_CAPTURE:
        diagnosis = "fresh_focus_bottleneck"
    else:
        diagnosis = "fresh_active_candidate_failed"

    return {
        "diagnosis": diagnosis,
        "fresh_feature_capture_gain": feature_gain,
        "fresh_supported_crossings": support,
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
        raise ValueError("request 161 fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_hazard, ignore_index=True))
    temporal_fit = pd.concat(fit_temporal, ignore_index=True)
    fit_focus180 = select_focus_rows(
        temporal_fit,
        market_model,
        FOCUS_CAP,
    )

    baseline_model = fit_model(
        fit_focus180,
        list(BASELINE_ACTIVE_FEATURES),
    )
    candidate_model = fit_model(
        fit_focus180,
        CANDIDATE_FEATURES,
    )

    eval_scans: list[pd.DataFrame] = []
    eval_support: list[pd.DataFrame] = []
    eval_focus: list[pd.DataFrame] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text not in FRESH_EVAL_DAYS or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        rows = add_cross_within_horizon_targets(
            scan,
            horizons=(HORIZON,),
        )
        focus = select_focus_rows(rows, market_model, FOCUS_CAP)
        focus = score_model(
            focus,
            baseline_model,
            list(BASELINE_ACTIVE_FEATURES),
            "baseline_probability",
        )
        focus = score_model(
            focus,
            candidate_model,
            CANDIDATE_FEATURES,
            "candidate_probability",
        )
        eval_scans.append(scan)
        eval_support.append(rows)
        eval_focus.append(focus)

    if not eval_scans:
        raise ValueError("request 161 fresh evaluation population is empty")

    scan = pd.concat(eval_scans, ignore_index=True)
    support_rows = pd.concat(eval_support, ignore_index=True)
    focus = pd.concat(eval_focus, ignore_index=True)
    found = sorted(scan["trading_day"].astype(str).unique())
    if found != FRESH_EVAL_DAYS:
        raise ValueError(
            f"expected request 161 fresh days {FRESH_EVAL_DAYS}, found {found}"
        )

    baseline_active = select_active(
        focus,
        "baseline_probability",
        BASELINE_THRESHOLD,
    )
    candidate_active = select_active(
        focus,
        "candidate_probability",
        CANDIDATE_THRESHOLD,
    )

    focus_capture = _supported_capture(support_rows, focus, scan)
    baseline_metrics = policy_metrics(
        support_rows,
        focus,
        baseline_active,
        scan,
        "baseline_probability",
    )
    candidate_metrics = policy_metrics(
        support_rows,
        focus,
        candidate_active,
        scan,
        "candidate_probability",
    )
    diagnosis = _diagnose(
        focus_capture,
        baseline_metrics,
        candidate_metrics,
    )

    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": True,
        "fresh_validation": True,
        "eval_days": FRESH_EVAL_DAYS,
        "fit_end": FIT_END,
        "focus_cap": FOCUS_CAP,
        "horizon_minutes": HORIZON,
        "frozen_candidate_features": CANDIDATE_FEATURES,
        "frozen_thresholds": {
            "baseline": BASELINE_THRESHOLD,
            "candidate": CANDIDATE_THRESHOLD,
        },
        "targets": {
            "focus_supported_capture": TARGET_FOCUS_CAPTURE,
            "active_supported_capture": TARGET_ACTIVE_CAPTURE,
            "min_supported_crossings": MIN_SUPPORTED_CROSSINGS,
        },
        "focus_capture": focus_capture,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "by_day": _daily_capture(
            support_rows,
            focus,
            candidate_active,
            scan,
        ),
        **diagnosis,
        "interpretation_rule": {
            "fresh_candidate_validated": (
                "Focus180 keeps >=99% supported capture, frozen candidate keeps "
                ">=95% supported capture, and candidate does not underperform "
                "the frozen baseline on this fresh block"
            ),
            "fresh_absolute_pass_feature_uplift_not_confirmed": (
                "candidate keeps >=95% and Focus keeps >=99%, but candidate "
                "underperforms baseline; architecture passes but feature uplift "
                "does not replicate"
            ),
            "fresh_focus_bottleneck": (
                "Focus180 itself falls below 99% supported capture"
            ),
            "fresh_active_candidate_failed": (
                "Focus remains adequate but frozen Active candidate falls below 95%"
            ),
            "fresh_validation_insufficient_support": (
                "fresh block has fewer than 100 supported exact-prior crossings "
                "or missing metrics"
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
