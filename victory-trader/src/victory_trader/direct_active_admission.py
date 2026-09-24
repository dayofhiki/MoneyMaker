"""Request 153: attribute Active admission misses and test direct one-step admission.

This request reuses only the request-151/152 evaluation sessions.  It opens no
new date and cannot promote a production policy.
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

from . import active_transport_integrated_hierarchy as transport
from .actionable_active_handoff import (
    add_actionable_priority,
    fit_observability_model,
    mark_focus_eligibility,
)
from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import MINUTE_MS
from .balanced_retention_active_allocator import (
    add_balanced_retention_value,
    add_transport_survival_probability,
    balanced_active_rows_with_state,
    build_transport_survival_rows,
    fit_transport_survival_model,
)
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hierarchical_attention_runtime import MODEL_KWARGS
from .market_calendar import is_us_equity_trading_day
from .market_wide_focus_hazard import (
    BASELINE_FEATURES,
    FIT_END,
    HAZARD_COLUMNS,
    build_market_hazard_frame,
    build_market_hazard_rows,
    fit_market_hazard,
    select_learned_focus,
)
from .massive_client import MassiveClient
from .multi_day import daterange

REQUEST_ID = 153
EVAL_DAYS = [
    "2026-05-21",
    "2026-05-22",
    "2026-05-26",
    "2026-05-27",
    "2026-05-28",
]
TARGET_RETENTION = 0.95
MATERIAL_GAIN = 0.01


def fit_direct_admission_model(rows: pd.DataFrame) -> HistGradientBoostingClassifier:
    """Fit unconditional P(exact next-minute runner crossing).

    A missing exact next-minute bar is a true zero for the executable one-minute
    event, rather than an unknown conditional-hazard label.
    """

    fit = rows.loc[rows["trading_day"].astype(str).le(FIT_END)].copy()
    if fit.empty:
        raise ValueError("direct admission fit population is empty")
    target = pd.to_numeric(fit["target_next_cross"], errors="coerce").fillna(0)
    y = target.astype(int)
    if y.nunique() < 2 or int(y.sum()) == 0:
        raise ValueError("direct admission fit needs both classes")
    model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    model.fit(
        fit[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan),
        y,
    )
    return model


def add_direct_admission_probability(
    scored: pd.DataFrame,
    model: HistGradientBoostingClassifier,
) -> pd.DataFrame:
    result = scored.copy()
    result["direct_admission_probability"] = model.predict_proba(
        result[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    return result


def _direct_variant(scored: pd.DataFrame) -> pd.DataFrame:
    """Reuse the validated allocator while replacing only admission/value score."""

    result = scored.copy()
    result["legacy_active_priority"] = result["active_priority"]
    direct = pd.to_numeric(
        result["direct_admission_probability"], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    survival = pd.to_numeric(
        result["transport_survival_probability"], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    result["active_priority"] = direct
    result["balanced_retention_value"] = direct * survival
    return result


def _ranked_focus(scored: pd.DataFrame, score_column: str) -> pd.DataFrame:
    focus = scored.loc[
        scored["transport_focus_eligible"].fillna(False).astype(bool)
    ].copy()
    focus = focus.sort_values(
        ["trading_day", "t", score_column, "ticker"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    focus["admission_rank"] = (
        focus.groupby(["trading_day", "t"], sort=False).cumcount() + 1
    )
    return focus


def _crossing_attribution(
    scored: pd.DataFrame,
    scan: pd.DataFrame,
) -> dict[str, object]:
    legacy = _ranked_focus(scored, "active_priority")
    direct = _ranked_focus(scored, "direct_admission_probability")
    scored_keys = {
        (str(r.trading_day), str(r.ticker).upper(), int(r.t))
        for r in scored.itertuples(index=False)
    }
    focus_keys = {
        (str(r.trading_day), str(r.ticker).upper(), int(r.t))
        for r in legacy.itertuples(index=False)
    }
    legacy_rank = {
        (str(r.trading_day), str(r.ticker).upper(), int(r.t)): int(r.admission_rank)
        for r in legacy.itertuples(index=False)
    }
    direct_rank = {
        (str(r.trading_day), str(r.ticker).upper(), int(r.t)): int(r.admission_rank)
        for r in direct.itertuples(index=False)
    }

    total = 0
    supported = 0
    in_focus = 0
    legacy_ranks: list[int] = []
    direct_ranks: list[int] = []
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
        if prior not in focus_keys:
            continue
        in_focus += 1
        legacy_ranks.append(legacy_rank[prior])
        direct_ranks.append(direct_rank[prior])

    def rank_summary(values: list[int]) -> dict[str, object]:
        if not values:
            return {
                "count": 0,
                "top20": 0,
                "top20_rate": None,
                "rank_median": None,
                "rank_p90": None,
                "rank_buckets": {},
            }
        arr = np.asarray(values, dtype=float)
        return {
            "count": len(values),
            "top20": int((arr <= 20).sum()),
            "top20_rate": float((arr <= 20).mean()),
            "rank_median": float(np.median(arr)),
            "rank_p90": float(np.quantile(arr, 0.90)),
            "rank_buckets": {
                "1_10": int(((arr >= 1) & (arr <= 10)).sum()),
                "11_20": int(((arr >= 11) & (arr <= 20)).sum()),
                "21_40": int(((arr >= 21) & (arr <= 40)).sum()),
                "41_60": int(((arr >= 41) & (arr <= 60)).sum()),
            },
        }

    return {
        "runner_crossings": total,
        "supported_exact_prior_pre_runner": supported,
        "focus_exact_prior_crossings": in_focus,
        "unsupported_or_not_pre_runner": total - supported,
        "focus_selection_misses_given_support": supported - in_focus,
        "legacy_rank": rank_summary(legacy_ranks),
        "direct_rank": rank_summary(direct_ranks),
    }


def _variant_metrics(
    scored: pd.DataFrame,
    focus: pd.DataFrame,
    scan: pd.DataFrame,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for cap in (8, 20):
        active, trace = balanced_active_rows_with_state(
            scored,
            max_additions=cap,
        )
        pooled = transport._retention_given(focus, active, scan)
        by_day: dict[str, float | None] = {}
        for day in EVAL_DAYS:
            by_day[day] = transport._retention_given(
                focus.loc[focus["trading_day"].astype(str).eq(day)],
                active.loc[active["trading_day"].astype(str).eq(day)],
                scan.loc[scan["trading_day"].astype(str).eq(day)],
            )
        out[str(cap)] = {
            "active_retention_given_focus": pooled,
            "by_day_active_retention_given_focus": by_day,
            "transport_audit": transport.explicit_subscription_audit(trace),
        }
    return out


def _diagnose(
    legacy_cap20: float | None,
    direct_cap20: float | None,
) -> str:
    if legacy_cap20 is None or direct_cap20 is None:
        return "insufficient_support"
    gain = direct_cap20 - legacy_cap20
    if direct_cap20 >= TARGET_RETENTION and gain >= MATERIAL_GAIN:
        return "direct_admission_candidate"
    if gain >= MATERIAL_GAIN:
        return "partial_ranking_improvement"
    return "direct_formulation_insufficient"


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
        raise ValueError("request 153 fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_hazard, ignore_index=True))
    causal_fit = pd.concat(fit_causal, ignore_index=True)
    observability_model = fit_observability_model(causal_fit)
    direct_model = fit_direct_admission_model(causal_fit)
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
        scored = add_direct_admission_probability(scored, direct_model)
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
        raise ValueError(f"expected request 153 eval days {EVAL_DAYS}, found {found}")

    legacy_metrics = _variant_metrics(scored, focus, scan)
    direct_scored = _direct_variant(scored)
    direct_metrics = _variant_metrics(direct_scored, focus, scan)

    focus_rows = scored.loc[
        scored["transport_focus_eligible"].fillna(False).astype(bool)
    ].copy()
    y = pd.to_numeric(
        focus_rows["target_next_cross"], errors="coerce"
    ).fillna(0).astype(int)
    legacy_ap = (
        float(average_precision_score(y, focus_rows["active_priority"]))
        if int(y.sum()) > 0
        else None
    )
    direct_ap = (
        float(
            average_precision_score(
                y,
                focus_rows["direct_admission_probability"],
            )
        )
        if int(y.sum()) > 0
        else None
    )

    attribution = _crossing_attribution(scored, scan)
    legacy20 = legacy_metrics["20"]["active_retention_given_focus"]
    direct20 = direct_metrics["20"]["active_retention_given_focus"]

    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "diagnostic_only": True,
        "eval_days": EVAL_DAYS,
        "target_retention": TARGET_RETENTION,
        "material_gain_threshold": MATERIAL_GAIN,
        "legacy_score": "market_hazard_probability * next_minute_observability_probability",
        "direct_score": "direct P(exact next-minute runner crossing)",
        "focus_average_precision": {
            "legacy": legacy_ap,
            "direct": direct_ap,
        },
        "crossing_attribution": attribution,
        "legacy": legacy_metrics,
        "direct": direct_metrics,
        "cap20_retention_gain": (
            float(direct20 - legacy20)
            if legacy20 is not None and direct20 is not None
            else None
        ),
        "diagnosis": _diagnose(legacy20, direct20),
        "interpretation_rule": {
            "direct_admission_candidate": (
                "direct cap20 reaches 95% and gains at least 1pp over legacy cap20"
            ),
            "partial_ranking_improvement": (
                "direct gains at least 1pp but remains below the 95% target"
            ),
            "direct_formulation_insufficient": (
                "direct gains less than 1pp; a richer admission intervention is needed"
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
