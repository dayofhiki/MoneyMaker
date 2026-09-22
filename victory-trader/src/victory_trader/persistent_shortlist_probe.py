"""Causal persistence probe for learned focus top-20 observation demand."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hierarchical_attention_runtime import _prior_population_capture
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
from .second_path_attention_probe import BASELINE_FEATURES

EVAL_DAYS = [
    "2026-03-25",
    "2026-03-26",
    "2026-03-27",
    "2026-03-30",
    "2026-03-31",
]
FOCUS_BUDGET = 60
SHORTLIST_BUDGET = 20
INCUMBENT_RANK_BUFFER = 40
MAX_SESSION_TICKERS = 300


def shortlist_audit(shortlist: pd.DataFrame) -> dict[str, float | int | None]:
    replacements: list[int] = []
    retentions: list[float] = []
    max_occupancy = 0
    for _, day_rows in shortlist.groupby("trading_day", sort=True):
        previous: set[str] = set()
        for _, group in day_rows.groupby("t", sort=True):
            current = set(group["ticker"].astype(str))
            max_occupancy = max(max_occupancy, len(current))
            replacements.append(len(current - previous))
            if previous:
                retentions.append(len(previous & current) / len(previous))
            previous = current
    ticker_days = int(
        shortlist.loc[:, ["trading_day", "ticker"]].drop_duplicates().shape[0]
    )
    return {
        "ticker_day_requests": ticker_days,
        "mean_set_retention": float(np.mean(retentions)) if retentions else None,
        "mean_slots_replaced_per_decision": (
            float(np.mean(replacements)) if replacements else None
        ),
        "max_occupancy": max_occupancy,
    }


def select_persistent_shortlist(
    focus: pd.DataFrame,
    *,
    incumbent_rank_buffer: int = INCUMBENT_RANK_BUFFER,
    max_session_tickers: int = MAX_SESSION_TICKERS,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Apply rank hysteresis under a causal session admission budget."""

    if max_session_tickers < SHORTLIST_BUDGET:
        raise ValueError("session ticker budget must cover the shortlist")

    selected: list[pd.DataFrame] = []
    for _, day_rows in focus.groupby("trading_day", sort=True):
        incumbents: dict[str, pd.Series] = {}
        admitted: set[str] = set()
        for _, group in day_rows.groupby("t", sort=True):
            ranked = group.sort_values(
                ["market_hazard_probability", "ticker"],
                ascending=[False, True],
                kind="stable",
            ).copy()
            ranked["focus_hazard_rank"] = np.arange(1, len(ranked) + 1)
            retained = ranked.loc[
                ranked["ticker"].astype(str).isin(incumbents)
                & ranked["focus_hazard_rank"].le(incumbent_rank_buffer)
            ].copy()
            retained = retained.sort_values(
                ["market_hazard_probability", "ticker"],
                ascending=[False, True],
                kind="stable",
            ).head(SHORTLIST_BUDGET)
            retained_names = set(retained["ticker"].astype(str))
            fill_rows: list[pd.Series] = []
            for _, candidate in ranked.loc[
                ~ranked["ticker"].astype(str).isin(retained_names)
            ].iterrows():
                if len(retained) + len(fill_rows) >= SHORTLIST_BUDGET:
                    break
                ticker = str(candidate["ticker"])
                if ticker not in admitted and len(admitted) >= max_session_tickers:
                    continue
                admitted.add(ticker)
                fill_rows.append(candidate)
                retained_names.add(ticker)
            fill = (
                pd.DataFrame(fill_rows, columns=ranked.columns)
                if fill_rows
                else ranked.iloc[0:0].copy()
            )
            current = pd.concat([retained, fill], ignore_index=True)

            if len(current) < SHORTLIST_BUDGET:
                stale_rows: list[pd.Series] = []
                current_names = set(current["ticker"].astype(str))
                stale_candidates = sorted(
                    (
                        row.copy()
                        for ticker, row in incumbents.items()
                        if ticker not in current_names
                    ),
                    key=lambda row: (
                        -float(row["market_hazard_probability"]),
                        str(row["ticker"]),
                    ),
                )
                for row in stale_candidates[: SHORTLIST_BUDGET - len(current)]:
                    row["t"] = group["t"].iloc[0]
                    row["focus_hazard_rank"] = FOCUS_BUDGET + 1
                    if "target_next_cross" in row.index:
                        row["target_next_cross"] = False
                    stale_rows.append(row)
                if stale_rows:
                    current = pd.concat(
                        [current, pd.DataFrame(stale_rows, columns=ranked.columns)],
                        ignore_index=True,
                    )
            selected.append(current)
            incumbents = {
                str(row["ticker"]): row.copy() for _, row in current.iterrows()
            }
    output = (
        pd.concat(selected, ignore_index=True)
        if selected
        else focus.iloc[0:0].copy()
    )
    return output, shortlist_audit(output)


def evaluate_persistence(
    eval_scan: pd.DataFrame,
    focus: pd.DataFrame,
    stateless: pd.DataFrame,
    persistent: pd.DataFrame,
    stateless_audit: dict[str, float | int | None],
    persistent_audit: dict[str, float | int | None],
) -> dict[str, object]:
    focus_metrics = _prior_population_capture(focus, eval_scan)
    stateless_metrics = _prior_population_capture(stateless, eval_scan)
    persistent_metrics = _prior_population_capture(persistent, eval_scan)
    focus_rate = focus_metrics["prior_minute_rate"]
    persistent_rate = persistent_metrics["prior_minute_rate"]
    conditional_capture = (
        float(persistent_rate / focus_rate)
        if focus_rate not in (None, 0) and persistent_rate is not None
        else None
    )
    stateless_requests = int(stateless_audit["ticker_day_requests"])
    persistent_requests = int(persistent_audit["ticker_day_requests"])
    request_reduction = (
        float(1.0 - persistent_requests / stateless_requests)
        if stateless_requests
        else None
    )

    by_day: dict[str, object] = {}
    noninferior_days = 0
    support_ok = True
    for day in EVAL_DAYS:
        scan_day = eval_scan.loc[eval_scan["trading_day"].astype(str).eq(day)]
        stateless_day = stateless.loc[
            stateless["trading_day"].astype(str).eq(day)
        ]
        persistent_day = persistent.loc[
            persistent["trading_day"].astype(str).eq(day)
        ]
        old = _prior_population_capture(stateless_day, scan_day)
        new = _prior_population_capture(persistent_day, scan_day)
        if int(old["runner_crossings"]) < 10:
            support_ok = False
        if (
            old["prior_minute_rate"] is not None
            and new["prior_minute_rate"] is not None
            and new["prior_minute_rate"] >= old["prior_minute_rate"] - 0.02
        ):
            noninferior_days += 1
        by_day[day] = {"stateless": old, "persistent": new}

    stateless_rate = stateless_metrics["prior_minute_rate"]
    retention = persistent_audit["mean_set_retention"]
    gate = bool(
        support_ok
        and stateless_rate is not None
        and persistent_rate is not None
        and persistent_rate >= stateless_rate - 0.02
        and noninferior_days >= 4
        and conditional_capture is not None
        and conditional_capture >= 0.95
        and request_reduction is not None
        and request_reduction >= 0.25
        and retention is not None
        and retention >= 0.70
        and int(persistent_audit["max_occupancy"]) <= SHORTLIST_BUDGET
    )
    return {
        "eval_days": EVAL_DAYS,
        "focus": focus_metrics,
        "stateless_shortlist": stateless_metrics,
        "persistent_shortlist": persistent_metrics,
        "persistent_capture_conditional_on_focus_rate": conditional_capture,
        "noninferior_capture_days": noninferior_days,
        "ticker_day_request_reduction_rate": request_reduction,
        "stateless_audit": stateless_audit,
        "persistent_audit": persistent_audit,
        "by_day": by_day,
        "promotion_gate_pass": gate,
    }


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fit_rows: list[pd.DataFrame] = []
    eval_scans: list[pd.DataFrame] = []
    eval_focus: list[pd.DataFrame] = []
    eval_stateless: list[pd.DataFrame] = []
    day_summaries: list[dict[str, object]] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text > FIT_END or not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, scan_client, day)
        rows = build_market_hazard_rows(scan)
        fit_rows.append(rows.loc[:, HAZARD_COLUMNS].copy())
        day_summaries.append(
            {
                "trading_day": day_text,
                "phase": "focus_fit",
                "scan_rows": int(len(scan)),
                "hazard_rows": int(len(rows)),
                "runner_crossings": int(scan["runner_cross_now"].sum()),
                "point_in_time_common_stocks": scan_summary.get(
                    "point_in_time_common_stocks"
                ),
            }
        )
    if not fit_rows:
        raise ValueError("focus fit period produced no rows")
    model = fit_market_hazard(pd.concat(fit_rows, ignore_index=True))

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text not in EVAL_DAYS or not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, scan_client, day)
        market_rows = build_market_hazard_rows(scan).loc[:, HAZARD_COLUMNS]
        _, focus, stateless = select_learned_focus(market_rows, model)
        eval_scans.append(scan)
        eval_focus.append(focus)
        eval_stateless.append(stateless)
        day_summaries.append(
            {
                "trading_day": day_text,
                "phase": "evaluation",
                "scan_rows": int(len(scan)),
                "focus_rows": int(len(focus)),
                "stateless_shortlist_rows": int(len(stateless)),
                "runner_crossings": int(scan["runner_cross_now"].sum()),
                "point_in_time_common_stocks": scan_summary.get(
                    "point_in_time_common_stocks"
                ),
            }
        )
    if not eval_scans:
        raise ValueError("evaluation period produced no rows")
    eval_scan = pd.concat(eval_scans, ignore_index=True)
    found_days = sorted(eval_scan["trading_day"].astype(str).unique())
    if found_days != EVAL_DAYS:
        raise ValueError(f"expected eval days {EVAL_DAYS}, found {found_days}")

    focus = pd.concat(eval_focus, ignore_index=True)
    stateless = pd.concat(eval_stateless, ignore_index=True)
    persistent, persistent_audit = select_persistent_shortlist(focus)
    stateless_audit = shortlist_audit(stateless)
    evaluation = evaluate_persistence(
        eval_scan,
        focus,
        stateless,
        persistent,
        stateless_audit,
        persistent_audit,
    )
    output = persistent.loc[
        :,
        [
            "trading_day",
            "ticker",
            "t",
            "market_hazard_probability",
            "focus_hazard_rank",
            "target_next_cross",
        ],
    ].copy()
    summary: dict[str, object] = {
        "schema_version": 3,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "fit_end": FIT_END,
        "eval_days": EVAL_DAYS,
        "focus_budget": FOCUS_BUDGET,
        "shortlist_budget": SHORTLIST_BUDGET,
        "incumbent_rank_buffer": INCUMBENT_RANK_BUFFER,
        "max_session_tickers": MAX_SESSION_TICKERS,
        "features": BASELINE_FEATURES,
        "days": day_summaries,
        "flatfile_stats": store.stats.to_dict(),
        "evaluation": evaluation,
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe persistence for learned focus top-20 observation."
    )
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key), FLATFILE_CACHE_DIR
    )
    scan_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=0.0,
    )
    output, summary = run_probe(store, scan_client, args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
