"""Request 159: hierarchical Focus-context features for Active admission.

Development-only on already-opened May sessions. Focus=180 is frozen. Compare
the Request-158 Focus180 temporal model against the same model augmented with
the upstream learned market-hazard probability and rank.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import MINUTE_MS
from .config import load_settings, require_flatfile_credentials
from .direct_active_admission import EVAL_DAYS
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .focus_cap_frontier import _active_precision, _count_metrics, _supported_capture
from .focus180_temporal_retrain import select_focus_rows
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
from .temporal_active_admission import add_cross_within_horizon_targets

REQUEST_ID = 159
HORIZON = 3
FOCUS_CAP = 180
TARGET_SELECTED_FRACTION = 0.1291012691012691
TARGET_CAPTURE = 0.95
TAIL_BASELINE_CAPTURED = 4
TAIL_TARGET_CAPTURED = 7
MAX_BURDEN_RATIO = 1.10

BASELINE_ACTIVE_FEATURES = list(BASELINE_FEATURES)
HIERARCHICAL_ACTIVE_FEATURES = [
    *BASELINE_FEATURES,
    "market_hazard_probability",
    "market_rank",
]


def fit_model(
    rows: pd.DataFrame,
    features: list[str],
) -> HistGradientBoostingClassifier:
    fit = rows.loc[rows["trading_day"].astype(str).le(FIT_END)].copy()
    target = pd.to_numeric(fit["target_cross_within_3m"], errors="coerce")
    fit = fit.loc[target.notna()].copy()
    y = pd.to_numeric(
        fit["target_cross_within_3m"], errors="raise"
    ).astype(int)
    if fit.empty or y.nunique() < 2 or int(y.sum()) == 0:
        raise ValueError("request 159 fit needs both classes")
    model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    model.fit(
        fit[features].replace([np.inf, -np.inf], np.nan),
        y,
    )
    return model


def score_model(
    rows: pd.DataFrame,
    model,
    features: list[str],
    column: str,
) -> pd.DataFrame:
    result = rows.copy()
    result[column] = model.predict_proba(
        result[features].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    return result


def threshold_for_fraction(
    rows: pd.DataFrame,
    probability_column: str,
    selected_fraction: float = TARGET_SELECTED_FRACTION,
) -> float:
    quantile = float(np.clip(1.0 - selected_fraction, 0.0, 1.0))
    scores = pd.to_numeric(rows[probability_column], errors="raise").to_numpy()
    return float(np.quantile(scores, quantile))


def select_active(
    rows: pd.DataFrame,
    probability_column: str,
    threshold: float,
) -> pd.DataFrame:
    return rows.loc[
        pd.to_numeric(rows[probability_column], errors="coerce").ge(threshold)
    ].copy()


def rank_bucket_capture(
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


def ranking_ap(
    focus: pd.DataFrame,
    probability_column: str,
) -> float | None:
    target = pd.to_numeric(
        focus["target_cross_within_3m"], errors="coerce"
    )
    mask = target.notna()
    if int(mask.sum()) == 0 or target.loc[mask].nunique() < 2:
        return None
    return float(
        average_precision_score(
            target.loc[mask].astype(int),
            pd.to_numeric(
                focus.loc[mask, probability_column],
                errors="raise",
            ),
        )
    )


def policy_metrics(
    support_rows: pd.DataFrame,
    focus: pd.DataFrame,
    active: pd.DataFrame,
    scan: pd.DataFrame,
    probability_column: str,
) -> dict[str, object]:
    return {
        "active_capture": _supported_capture(support_rows, active, scan),
        "active_count": _count_metrics(active, focus),
        "active_selection_quality": _active_precision(active),
        "eval_ap_on_focus180": ranking_ap(focus, probability_column),
        "rank_bucket_capture": rank_bucket_capture(focus, active, scan),
    }


def diagnose(
    baseline: dict[str, object],
    hierarchical: dict[str, object],
) -> dict[str, object]:
    base_rate = baseline["active_capture"]["supported_capture_rate"]
    new_rate = hierarchical["active_capture"]["supported_capture_rate"]
    if base_rate is None or new_rate is None:
        return {"diagnosis": "insufficient_support"}

    base_mean = float(baseline["active_count"]["mean"])
    new_mean = float(hierarchical["active_count"]["mean"])
    burden_ratio = float(new_mean / base_mean) if base_mean > 0 else None

    base_tail = int(
        baseline["rank_bucket_capture"]["rank_61_180"]["captured"]
    )
    new_tail = int(
        hierarchical["rank_bucket_capture"]["rank_61_180"]["captured"]
    )
    tail_gain = new_tail - base_tail
    overall_gain = float(new_rate - base_rate)

    if (
        new_rate >= TARGET_CAPTURE
        and new_tail >= TAIL_TARGET_CAPTURED
        and burden_ratio is not None
        and burden_ratio <= MAX_BURDEN_RATIO
    ):
        diagnosis = "hierarchical_focus_context_candidate"
    elif tail_gain >= 2:
        diagnosis = "hierarchical_focus_context_partial_tail_gain"
    elif overall_gain < 0 or tail_gain < 0:
        diagnosis = "hierarchical_focus_context_regression"
    else:
        diagnosis = "hierarchical_focus_context_no_mechanism_gain"

    return {
        "diagnosis": diagnosis,
        "overall_capture_gain": overall_gain,
        "tail_capture_gain_count": tail_gain,
        "eval_burden_ratio": burden_ratio,
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
        raise ValueError("request 159 fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_hazard, ignore_index=True))
    temporal_fit = pd.concat(fit_temporal, ignore_index=True)
    fit_focus180 = select_focus_rows(
        temporal_fit,
        market_model,
        FOCUS_CAP,
    )

    baseline_model = fit_model(
        fit_focus180,
        BASELINE_ACTIVE_FEATURES,
    )
    hierarchical_model = fit_model(
        fit_focus180,
        HIERARCHICAL_ACTIVE_FEATURES,
    )

    fit_focus180 = score_model(
        fit_focus180,
        baseline_model,
        BASELINE_ACTIVE_FEATURES,
        "baseline_probability",
    )
    fit_focus180 = score_model(
        fit_focus180,
        hierarchical_model,
        HIERARCHICAL_ACTIVE_FEATURES,
        "hierarchical_probability",
    )
    baseline_threshold = threshold_for_fraction(
        fit_focus180,
        "baseline_probability",
    )
    hierarchical_threshold = threshold_for_fraction(
        fit_focus180,
        "hierarchical_probability",
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
        focus = score_model(
            focus,
            baseline_model,
            BASELINE_ACTIVE_FEATURES,
            "baseline_probability",
        )
        focus = score_model(
            focus,
            hierarchical_model,
            HIERARCHICAL_ACTIVE_FEATURES,
            "hierarchical_probability",
        )
        eval_scans.append(scan)
        eval_support.append(rows)
        eval_focus.append(focus)

    scan = pd.concat(eval_scans, ignore_index=True)
    support_rows = pd.concat(eval_support, ignore_index=True)
    focus = pd.concat(eval_focus, ignore_index=True)
    found = sorted(scan["trading_day"].astype(str).unique())
    if found != EVAL_DAYS:
        raise ValueError(f"expected request 159 eval days {EVAL_DAYS}, found {found}")

    baseline_active = select_active(
        focus,
        "baseline_probability",
        baseline_threshold,
    )
    hierarchical_active = select_active(
        focus,
        "hierarchical_probability",
        hierarchical_threshold,
    )

    baseline_metrics = policy_metrics(
        support_rows,
        focus,
        baseline_active,
        scan,
        "baseline_probability",
    )
    hierarchical_metrics = policy_metrics(
        support_rows,
        focus,
        hierarchical_active,
        scan,
        "hierarchical_probability",
    )
    diagnosis = diagnose(
        baseline_metrics,
        hierarchical_metrics,
    )

    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "diagnostic_only": True,
        "eval_days": EVAL_DAYS,
        "focus_cap": FOCUS_CAP,
        "horizon_minutes": HORIZON,
        "target_selected_fraction": TARGET_SELECTED_FRACTION,
        "thresholds_from_fit_period_only": {
            "baseline": baseline_threshold,
            "hierarchical": hierarchical_threshold,
        },
        "baseline_features": BASELINE_ACTIVE_FEATURES,
        "hierarchical_features": HIERARCHICAL_ACTIVE_FEATURES,
        "baseline": baseline_metrics,
        "hierarchical": hierarchical_metrics,
        "target_capture": TARGET_CAPTURE,
        "tail_baseline_captured_reference": TAIL_BASELINE_CAPTURED,
        "tail_target_captured": TAIL_TARGET_CAPTURED,
        "max_burden_ratio": MAX_BURDEN_RATIO,
        **diagnosis,
        "interpretation_rule": {
            "hierarchical_focus_context_candidate": (
                "hierarchical features reach >=95% supported capture, capture "
                "at least 7/14 rank-61-180 runners, and stay within 10% mean "
                "Active burden"
            ),
            "hierarchical_focus_context_partial_tail_gain": (
                "rank-61-180 capture improves by at least two runners but the "
                "full candidate gate does not pass"
            ),
            "hierarchical_focus_context_no_mechanism_gain": (
                "rank-61-180 capture improves by fewer than two runners without regression"
            ),
            "hierarchical_focus_context_regression": (
                "overall or rank-61-180 capture regresses"
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
