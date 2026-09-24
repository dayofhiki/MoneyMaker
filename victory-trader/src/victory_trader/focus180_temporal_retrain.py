"""Request 158: retrain the 3-minute Active model on learned Focus-180.

Development-only on the already-opened May block. Focus cap 180 is frozen from
request 157. The primary comparison holds fit-period observation burden roughly
constant so any gain is not merely caused by watching more names.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import MINUTE_MS
from .config import load_settings, require_flatfile_credentials
from .direct_active_admission import EVAL_DAYS
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .focus_cap_frontier import _active_precision, _count_metrics, _supported_capture
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
)

REQUEST_ID = 158
HORIZON = 3
FOCUS_CAP = 180
BASELINE_TRAINING_CAP = 60
BASELINE_THRESHOLD = 0.0040281217293971616
TARGET_CAPTURE = 0.95
MATERIAL_GAIN = 0.005
MAX_BURDEN_RATIO = 1.10


def select_focus_rows(
    target_rows: pd.DataFrame,
    market_model,
    cap: int,
) -> pd.DataFrame:
    scored = target_rows.copy()
    scored["market_hazard_probability"] = market_model.predict_proba(
        scored[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    scored = scored.sort_values(
        ["trading_day", "t", "market_hazard_probability", "ticker"],
        ascending=[True, True, False, True],
        kind="stable",
    ).copy()
    scored["market_rank"] = (
        scored.groupby(["trading_day", "t"], sort=False).cumcount() + 1
    )
    return scored.loc[scored["market_rank"].le(cap)].copy()


def add_model_probability(
    rows: pd.DataFrame,
    model,
    column: str,
) -> pd.DataFrame:
    result = rows.copy()
    result[column] = model.predict_proba(
        result[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    return result


def _fit_mean_count(selected: pd.DataFrame, focus: pd.DataFrame) -> float:
    return float(_count_metrics(selected, focus)["mean"])


def derive_burden_matched_threshold(
    fit_focus180: pd.DataFrame,
    baseline_model,
    broad_model,
) -> dict[str, float]:
    scored = add_model_probability(
        fit_focus180, baseline_model, "baseline_probability"
    )
    scored = add_model_probability(
        scored, broad_model, "broad_probability"
    )
    baseline_selected = scored.loc[
        pd.to_numeric(scored["baseline_probability"], errors="coerce").ge(
            BASELINE_THRESHOLD
        )
    ].copy()
    baseline_fraction = float(len(baseline_selected) / len(scored))
    target_quantile = float(np.clip(1.0 - baseline_fraction, 0.0, 1.0))
    broad_scores = pd.to_numeric(
        scored["broad_probability"], errors="raise"
    ).to_numpy()
    threshold = float(np.quantile(broad_scores, target_quantile))
    broad_selected = scored.loc[
        pd.to_numeric(scored["broad_probability"], errors="coerce").ge(threshold)
    ].copy()
    return {
        "baseline_fit_selected_fraction": baseline_fraction,
        "baseline_fit_mean_active_count": _fit_mean_count(
            baseline_selected, scored
        ),
        "broad_fit_quantile": target_quantile,
        "broad_threshold": threshold,
        "broad_fit_mean_active_count": _fit_mean_count(broad_selected, scored),
    }


def _policy_metrics(
    focus: pd.DataFrame,
    active: pd.DataFrame,
    scan: pd.DataFrame,
) -> dict[str, object]:
    capture = _supported_capture(focus, active, scan)
    return {
        "active_capture": capture,
        "active_count": _count_metrics(active, focus),
        "active_selection_quality": _active_precision(active),
    }


def _rank_bucket_capture(
    focus: pd.DataFrame,
    active: pd.DataFrame,
    scan: pd.DataFrame,
) -> dict[str, object]:
    focus_rank = {
        (str(r.trading_day), str(r.ticker).upper(), int(r.t)): int(r.market_rank)
        for r in focus.itertuples(index=False)
    }
    active_keys = {
        (str(r.trading_day), str(r.ticker).upper(), int(r.t))
        for r in active.itertuples(index=False)
    }
    buckets = {
        "rank_1_60": {"supported": 0, "captured": 0},
        "rank_61_180": {"supported": 0, "captured": 0},
    }
    crossings = scan.loc[
        scan["runner_cross_now"].fillna(False).astype(bool),
        ["trading_day", "ticker", "t"],
    ]
    for row in crossings.itertuples(index=False):
        prior = (
            str(row.trading_day),
            str(row.ticker).upper(),
            int(row.t) - MINUTE_MS,
        )
        rank = focus_rank.get(prior)
        if rank is None:
            continue
        bucket = "rank_1_60" if rank <= 60 else "rank_61_180"
        buckets[bucket]["supported"] += 1
        if prior in active_keys:
            buckets[bucket]["captured"] += 1

    for values in buckets.values():
        support = int(values["supported"])
        values["capture_rate"] = (
            float(values["captured"] / support) if support else None
        )
    return buckets


def _ranking_ap(focus: pd.DataFrame, column: str) -> float | None:
    target = pd.to_numeric(
        focus["target_cross_within_3m"], errors="coerce"
    )
    mask = target.notna()
    if int(mask.sum()) == 0 or target.loc[mask].nunique() < 2:
        return None
    return float(
        average_precision_score(
            target.loc[mask].astype(int),
            pd.to_numeric(focus.loc[mask, column], errors="raise"),
        )
    )


def _diagnose(
    baseline: dict[str, object],
    broad: dict[str, object],
) -> dict[str, object]:
    base_capture = baseline["active_capture"]["supported_capture_rate"]
    broad_capture = broad["active_capture"]["supported_capture_rate"]
    base_mean = float(baseline["active_count"]["mean"])
    broad_mean = float(broad["active_count"]["mean"])
    if base_capture is None or broad_capture is None:
        return {"diagnosis": "insufficient_support"}

    gain = float(broad_capture - base_capture)
    burden_ratio = float(broad_mean / base_mean) if base_mean > 0 else None

    if (
        broad_capture >= TARGET_CAPTURE
        and gain >= MATERIAL_GAIN
        and burden_ratio is not None
        and burden_ratio <= MAX_BURDEN_RATIO
    ):
        diagnosis = "focus180_training_candidate"
    elif gain >= MATERIAL_GAIN:
        diagnosis = "focus180_training_partial_gain"
    elif gain < 0:
        diagnosis = "focus180_training_regression"
    else:
        diagnosis = "focus180_training_no_material_gain"

    return {
        "diagnosis": diagnosis,
        "primary_capture_gain": gain,
        "primary_eval_burden_ratio": burden_ratio,
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
        raise ValueError("request 158 fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_hazard, ignore_index=True))
    temporal_fit = pd.concat(fit_temporal, ignore_index=True)
    fit_focus60 = select_focus_rows(
        temporal_fit, market_model, BASELINE_TRAINING_CAP
    )
    fit_focus180 = select_focus_rows(temporal_fit, market_model, FOCUS_CAP)

    baseline_model = fit_temporal_admission_model(
        fit_focus60, horizon=HORIZON
    )
    broad_model = fit_temporal_admission_model(
        fit_focus180, horizon=HORIZON
    )
    burden_match = derive_burden_matched_threshold(
        fit_focus180,
        baseline_model,
        broad_model,
    )

    eval_scans: list[pd.DataFrame] = []
    eval_focus: list[pd.DataFrame] = []
    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text not in EVAL_DAYS or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        rows = add_cross_within_horizon_targets(scan, horizons=(HORIZON,))
        focus = select_focus_rows(rows, market_model, FOCUS_CAP)
        focus = add_model_probability(
            focus, baseline_model, "baseline_probability"
        )
        focus = add_model_probability(
            focus, broad_model, "broad_probability"
        )
        eval_scans.append(scan)
        eval_focus.append(focus)

    scan = pd.concat(eval_scans, ignore_index=True)
    focus = pd.concat(eval_focus, ignore_index=True)
    found = sorted(scan["trading_day"].astype(str).unique())
    if found != EVAL_DAYS:
        raise ValueError(f"expected request 158 eval days {EVAL_DAYS}, found {found}")

    baseline_active = focus.loc[
        pd.to_numeric(focus["baseline_probability"], errors="coerce").ge(
            BASELINE_THRESHOLD
        )
    ].copy()
    broad_threshold = float(burden_match["broad_threshold"])
    broad_active = focus.loc[
        pd.to_numeric(focus["broad_probability"], errors="coerce").ge(
            broad_threshold
        )
    ].copy()
    broad_same_threshold = focus.loc[
        pd.to_numeric(focus["broad_probability"], errors="coerce").ge(
            BASELINE_THRESHOLD
        )
    ].copy()

    baseline_metrics = _policy_metrics(focus, baseline_active, scan)
    broad_metrics = _policy_metrics(focus, broad_active, scan)
    broad_same_metrics = _policy_metrics(focus, broad_same_threshold, scan)

    diagnosis = _diagnose(baseline_metrics, broad_metrics)

    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "diagnostic_only": True,
        "eval_days": EVAL_DAYS,
        "focus_cap": FOCUS_CAP,
        "horizon_minutes": HORIZON,
        "baseline_training_population": "fit-period learned Focus-60",
        "broad_training_population": "fit-period learned Focus-180",
        "primary_threshold_policy": "fit-period burden-matched",
        "baseline_threshold": BASELINE_THRESHOLD,
        "burden_match": burden_match,
        "baseline_focus60_model": {
            "metrics": baseline_metrics,
            "eval_ap_on_focus180": _ranking_ap(
                focus, "baseline_probability"
            ),
            "rank_bucket_capture": _rank_bucket_capture(
                focus, baseline_active, scan
            ),
        },
        "broad_focus180_model_primary": {
            "metrics": broad_metrics,
            "eval_ap_on_focus180": _ranking_ap(
                focus, "broad_probability"
            ),
            "rank_bucket_capture": _rank_bucket_capture(
                focus, broad_active, scan
            ),
        },
        "broad_focus180_model_same_numeric_threshold_diagnostic": {
            "threshold": BASELINE_THRESHOLD,
            "metrics": broad_same_metrics,
            "rank_bucket_capture": _rank_bucket_capture(
                focus, broad_same_threshold, scan
            ),
        },
        "target_capture": TARGET_CAPTURE,
        "material_gain": MATERIAL_GAIN,
        "max_burden_ratio": MAX_BURDEN_RATIO,
        **diagnosis,
        "interpretation_rule": {
            "focus180_training_candidate": (
                "burden-matched Focus-180 training reaches >=95% supported capture, "
                "gains >=0.50pp over the Focus-60-trained baseline, and eval mean "
                "Active burden stays within 10%"
            ),
            "focus180_training_partial_gain": (
                "burden-matched Focus-180 training gains >=0.50pp but misses one "
                "of the target/burden gates"
            ),
            "focus180_training_no_material_gain": (
                "burden-matched Focus-180 training gains less than 0.50pp"
            ),
            "focus180_training_regression": (
                "burden-matched Focus-180 training reduces supported capture"
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
