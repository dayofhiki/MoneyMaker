"""Request 152: diagnose the residual Active transport capacity frontier.

This request reuses only the already-opened request-151 evaluation sessions.
It is diagnostic-only: no policy is promoted and no later date is opened.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import active_transport_integrated_hierarchy as transport
from .actionable_active_handoff import (
    add_actionable_priority,
    fit_observability_model,
    mark_focus_eligibility,
)
from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .balanced_retention_active_allocator import (
    add_balanced_retention_value,
    add_transport_survival_probability,
    balanced_active_rows_with_state,
    build_transport_survival_rows,
    classify_subscription_state,
    fit_transport_survival_model,
)
from .config import load_settings, require_flatfile_credentials
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

REQUEST_ID = 152
EVAL_DAYS = [
    "2026-05-21",
    "2026-05-22",
    "2026-05-26",
    "2026-05-27",
    "2026-05-28",
]
CAPACITY_FRONTIER = (5, 8, 10, 12, 16, 20)
TARGET_RETENTION = 0.95
MATERIAL_GAIN = 0.01


def _diagnose_frontier(
    cap8_retention: float | None,
    cap20_retention: float | None,
) -> str:
    if cap8_retention is None or cap20_retention is None:
        return "insufficient_support"
    if cap20_retention < TARGET_RETENTION:
        return "selection_or_admission_bottleneck"
    if cap20_retention - cap8_retention >= MATERIAL_GAIN:
        return "residual_transport_capacity_bottleneck"
    return "mixed_or_near_plateau"


def _minimal_cap_meeting_target(frontier: dict[str, object]) -> int | None:
    for cap in CAPACITY_FRONTIER:
        value = frontier[str(cap)]["active_retention_given_focus"]
        if value is not None and float(value) >= TARGET_RETENTION:
            return cap
    return None


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    start: date,
    end: date,
) -> dict[str, object]:
    fit_labelable: list[pd.DataFrame] = []
    fit_causal: list[pd.DataFrame] = []
    fit_survival: list[pd.DataFrame] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text > FIT_END or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        fit_labelable.append(
            build_market_hazard_rows(scan).loc[:, HAZARD_COLUMNS].copy()
        )
        fit_causal.append(build_market_hazard_frame(scan))
        fit_survival.append(build_transport_survival_rows(scan))

    if not fit_labelable:
        raise ValueError("request 152 fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_labelable, ignore_index=True))
    observability_model = fit_observability_model(
        pd.concat(fit_causal, ignore_index=True)
    )
    survival_model = fit_transport_survival_model(
        pd.concat(fit_survival, ignore_index=True)
    )

    eval_scans: list[pd.DataFrame] = []
    eval_focus_parts: list[pd.DataFrame] = []
    eval_scored_parts: list[pd.DataFrame] = []

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
        scored = mark_focus_eligibility(scored, focus)
        scored = add_transport_survival_probability(scored, survival_model)
        scored = add_balanced_retention_value(scored)

        eval_scans.append(scan)
        eval_focus_parts.append(focus)
        eval_scored_parts.append(scored)

    eval_scan = pd.concat(eval_scans, ignore_index=True)
    eval_focus = pd.concat(eval_focus_parts, ignore_index=True)
    eval_scored = pd.concat(eval_scored_parts, ignore_index=True)

    found = sorted(eval_scan["trading_day"].astype(str).unique())
    if found != EVAL_DAYS:
        raise ValueError(f"expected request 152 eval days {EVAL_DAYS}, found {found}")

    frontier: dict[str, object] = {}
    for cap in CAPACITY_FRONTIER:
        active, trace = balanced_active_rows_with_state(
            eval_scored,
            max_additions=cap,
        )
        pooled_retention = transport._retention_given(
            eval_focus,
            active,
            eval_scan,
        )
        by_day: dict[str, float | None] = {}
        for day in EVAL_DAYS:
            focus_day = eval_focus.loc[
                eval_focus["trading_day"].astype(str).eq(day)
            ]
            active_day = active.loc[
                active["trading_day"].astype(str).eq(day)
            ]
            scan_day = eval_scan.loc[
                eval_scan["trading_day"].astype(str).eq(day)
            ]
            by_day[day] = transport._retention_given(
                focus_day,
                active_day,
                scan_day,
            )

        frontier[str(cap)] = {
            "active_retention_given_focus": pooled_retention,
            "by_day_active_retention_given_focus": by_day,
            "transport_audit": transport.explicit_subscription_audit(trace),
            "state_reason_audit": classify_subscription_state(trace, eval_scan),
        }

    cap8 = frontier["8"]["active_retention_given_focus"]
    cap20 = frontier["20"]["active_retention_given_focus"]
    min_cap = _minimal_cap_meeting_target(frontier)

    summary: dict[str, object] = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "eval_days": EVAL_DAYS,
        "diagnostic_only": True,
        "retention_value": "active_priority * transport_survival_probability",
        "capacity_frontier": list(CAPACITY_FRONTIER),
        "target_retention": TARGET_RETENTION,
        "material_gain_threshold": MATERIAL_GAIN,
        "frontier": frontier,
        "minimal_cap_meeting_95pct": min_cap,
        "cap8_to_cap20_retention_gain": (
            float(cap20 - cap8)
            if cap8 is not None and cap20 is not None
            else None
        ),
        "diagnosis": _diagnose_frontier(cap8, cap20),
        "interpretation_rule": {
            "selection_or_admission_bottleneck": (
                "cap20 retention remains below 95%; perfect minute-level "
                "transport cannot satisfy the target"
            ),
            "residual_transport_capacity_bottleneck": (
                "cap20 reaches 95% and improves over cap8 by at least 1pp"
            ),
            "mixed_or_near_plateau": (
                "cap20 reaches 95% but adds less than 1pp over cap8"
            ),
        },
        "flatfile_stats": store.stats.to_dict(),
        "scan_client_stats": scan_client.stats.to_dict(),
    }
    return summary


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
