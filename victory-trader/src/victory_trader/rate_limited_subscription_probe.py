"""Fresh validation of rate-limited desired-to-active subscription tracking."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import MINUTE_MS
from .bounded_turnover_shortlist_probe import select_shortlist, shortlist_audit
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .focus_shortlist_diagnostic import score_market_rows
from .hierarchical_attention_runtime import _prior_population_capture
from .market_calendar import is_us_equity_trading_day
from .market_wide_focus_hazard import (
    FIT_END,
    FOCUS_BUDGET,
    HAZARD_COLUMNS,
    SHORTLIST_BUDGET,
    build_market_hazard_rows,
    fit_market_hazard,
)
from .massive_client import MassiveClient
from .multi_day import daterange
from .observation_transport import (
    RateLimitedSubscriptionSelector,
    RankedCandidate,
)
from .second_path_attention_probe import BASELINE_FEATURES

EVAL_DAYS = [
    "2026-04-16",
    "2026-04-17",
    "2026-04-20",
    "2026-04-21",
    "2026-04-22",
]
MAX_ADDITIONS = 5


def select_rate_limited_active(
    focus: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_rows: list[dict[str, object]] = []
    desired_rows: list[pd.DataFrame] = []

    for trading_day, day_rows in focus.groupby("trading_day", sort=True):
        selector = RateLimitedSubscriptionSelector(
            capacity=SHORTLIST_BUDGET,
            max_additions=MAX_ADDITIONS,
        )
        for timestamp, group in day_rows.groupby("t", sort=True):
            ranked = group.sort_values(
                ["market_hazard_probability", "ticker"],
                ascending=[False, True],
                kind="stable",
            ).copy()
            candidates = [
                RankedCandidate(
                    ticker=str(row.ticker),
                    score=float(row.market_hazard_probability),
                )
                for row in ranked.itertuples(index=False)
            ]
            active = selector.select(candidates)
            desired = set(selector.desired)
            desired_rows.append(
                ranked.loc[
                    ranked["ticker"].astype(str).isin(desired)
                ].head(SHORTLIST_BUDGET)
            )
            by_ticker = {
                str(row.ticker): row._asdict()
                for row in ranked.itertuples(index=False)
            }
            for ticker in active:
                if ticker in by_ticker:
                    item = dict(by_ticker[ticker])
                else:
                    item = {
                        "trading_day": str(trading_day),
                        "ticker": ticker,
                        "t": int(timestamp),
                        "market_hazard_probability": np.nan,
                        "focus_hazard_rank": FOCUS_BUDGET + 1,
                        "target_next_cross": 0,
                    }
                item["transport_desired_now"] = ticker in desired
                active_rows.append(item)

    active_frame = pd.DataFrame(active_rows)
    desired_frame = (
        pd.concat(desired_rows, ignore_index=True)
        if desired_rows
        else focus.iloc[0:0].copy()
    )
    return active_frame, desired_frame


def _captured_crossings(
    selection: pd.DataFrame,
    scan: pd.DataFrame,
) -> set[tuple[str, str, int]]:
    keys = set(
        zip(
            selection["trading_day"].astype(str),
            selection["ticker"].astype(str),
            selection["t"].astype(int),
            strict=False,
        )
    )
    captured: set[tuple[str, str, int]] = set()
    crossings = scan.loc[
        scan["runner_cross_now"].fillna(False).astype(bool),
        ["trading_day", "ticker", "t"],
    ]
    for row in crossings.itertuples(index=False):
        prior_key = (
            str(row.trading_day),
            str(row.ticker),
            int(row.t) - MINUTE_MS,
        )
        if prior_key in keys:
            captured.add(
                (str(row.trading_day), str(row.ticker), int(row.t))
            )
    return captured


def _retention_given_focus(
    focus: pd.DataFrame,
    selection: pd.DataFrame,
    scan: pd.DataFrame,
) -> float | None:
    focus_crossings = _captured_crossings(focus, scan)
    selected_crossings = _captured_crossings(selection, scan)
    if not focus_crossings:
        return None
    return float(len(focus_crossings & selected_crossings) / len(focus_crossings))


def active_audit(
    active: pd.DataFrame,
    desired: pd.DataFrame,
) -> dict[str, object]:
    base = shortlist_audit(active)
    mismatch_rates: list[float] = []
    desired_lookup = {
        (str(day), int(t)): set(group["ticker"].astype(str))
        for (day, t), group in desired.groupby(
            ["trading_day", "t"], sort=False
        )
    }
    for (day, t), group in active.groupby(["trading_day", "t"], sort=False):
        active_set = set(group["ticker"].astype(str))
        desired_set = desired_lookup.get((str(day), int(t)), set())
        denominator = max(len(desired_set), 1)
        mismatch_rates.append(
            len(active_set.symmetric_difference(desired_set)) / denominator
        )
    base["mean_active_desired_symmetric_difference_rate"] = (
        float(np.mean(mismatch_rates)) if mismatch_rates else None
    )
    return base


def evaluate(
    eval_scan: pd.DataFrame,
    focus: pd.DataFrame,
    old_shortlist: pd.DataFrame,
    active: pd.DataFrame,
    desired: pd.DataFrame,
) -> dict[str, object]:
    focus_metrics = _prior_population_capture(focus, eval_scan)
    old_metrics = _prior_population_capture(old_shortlist, eval_scan)
    active_metrics = _prior_population_capture(active, eval_scan)
    desired_metrics = _prior_population_capture(desired, eval_scan)
    active_retention = _retention_given_focus(focus, active, eval_scan)
    old_retention = _retention_given_focus(focus, old_shortlist, eval_scan)
    desired_retention = _retention_given_focus(focus, desired, eval_scan)
    old_audit = shortlist_audit(old_shortlist)
    desired_audit = shortlist_audit(desired)
    rate_audit = active_audit(active, desired)

    support_ok = True
    nonlower_days = 0
    by_day: dict[str, object] = {}
    for day in EVAL_DAYS:
        scan_day = eval_scan.loc[eval_scan["trading_day"].astype(str).eq(day)]
        focus_day = focus.loc[focus["trading_day"].astype(str).eq(day)]
        old_day = old_shortlist.loc[
            old_shortlist["trading_day"].astype(str).eq(day)
        ]
        active_day = active.loc[active["trading_day"].astype(str).eq(day)]
        desired_day = desired.loc[desired["trading_day"].astype(str).eq(day)]
        f = _prior_population_capture(focus_day, scan_day)
        old = _prior_population_capture(old_day, scan_day)
        act = _prior_population_capture(active_day, scan_day)
        des = _prior_population_capture(desired_day, scan_day)
        if int(f["runner_crossings"]) < 10:
            support_ok = False
        if (
            old["prior_minute_rate"] is not None
            and act["prior_minute_rate"] is not None
            and act["prior_minute_rate"] >= old["prior_minute_rate"]
        ):
            nonlower_days += 1
        by_day[day] = {
            "focus": f,
            "rank40_hysteresis": old,
            "desired_current_top20": des,
            "rate_limited_active": act,
            "active_retention_given_focus": _retention_given_focus(
                focus_day, active_day, scan_day
            ),
        }

    old_rate = old_metrics["prior_minute_rate"]
    active_rate = active_metrics["prior_minute_rate"]
    gate = bool(
        support_ok
        and old_rate is not None
        and active_rate is not None
        and active_rate > old_rate
        and active_retention is not None
        and active_retention >= 0.95
        and nonlower_days >= 4
        and int(rate_audit["max_occupancy"]) <= SHORTLIST_BUDGET
        and rate_audit["mean_additions_per_decision"] is not None
        and float(rate_audit["mean_additions_per_decision"]) <= 5.0
    )
    return {
        "eval_days": EVAL_DAYS,
        "focus": focus_metrics,
        "rank40_hysteresis": old_metrics,
        "desired_current_top20": desired_metrics,
        "rate_limited_active": active_metrics,
        "rank40_retention_given_focus": old_retention,
        "desired_retention_given_focus": desired_retention,
        "active_retention_given_focus": active_retention,
        "rank40_audit": old_audit,
        "desired_audit": desired_audit,
        "rate_limited_audit": rate_audit,
        "nonlower_capture_days": nonlower_days,
        "transport_gate_pass": gate,
        "by_day": by_day,
    }


def run_probe(
    store: MassiveFlatFileStore,
    rest_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fit_rows: list[pd.DataFrame] = []
    eval_rows: list[pd.DataFrame] = []
    eval_scans: list[pd.DataFrame] = []
    days: list[dict[str, object]] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        in_fit = day_text <= FIT_END
        in_eval = day_text in EVAL_DAYS
        if (not in_fit and not in_eval) or not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, rest_client, day)
        rows = build_market_hazard_rows(scan).loc[:, HAZARD_COLUMNS].copy()
        if in_fit:
            fit_rows.append(rows)
        if in_eval:
            eval_rows.append(rows)
            eval_scans.append(scan)
        days.append(
            {
                "trading_day": day_text,
                "phase": "fit" if in_fit else "evaluation",
                "scan_rows": int(len(scan)),
                "hazard_rows": int(len(rows)),
                "runner_crossings": int(scan["runner_cross_now"].sum()),
                "point_in_time_common_stocks": scan_summary.get(
                    "point_in_time_common_stocks"
                ),
            }
        )

    if not fit_rows or not eval_rows or not eval_scans:
        raise ValueError("rate-limited validation requires fit and evaluation rows")
    eval_scan = pd.concat(eval_scans, ignore_index=True)
    found_days = sorted(eval_scan["trading_day"].astype(str).unique())
    if found_days != EVAL_DAYS:
        raise ValueError(f"expected eval days {EVAL_DAYS}, found {found_days}")

    model = fit_market_hazard(pd.concat(fit_rows, ignore_index=True))
    scored = score_market_rows(pd.concat(eval_rows, ignore_index=True), model)
    focus = scored.loc[scored["focus_hazard_rank"].le(FOCUS_BUDGET)].copy()
    old_shortlist = select_shortlist(focus, bounded_turnover=False)
    active, desired = select_rate_limited_active(focus)
    evaluation = evaluate(
        eval_scan, focus, old_shortlist, active, desired
    )
    summary = {
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "fit_end": FIT_END,
        "eval_days": EVAL_DAYS,
        "focus_budget": FOCUS_BUDGET,
        "shortlist_budget": SHORTLIST_BUDGET,
        "max_additions_per_update": MAX_ADDITIONS,
        "features": BASELINE_FEATURES,
        "days": days,
        "flatfile_stats": store.stats.to_dict(),
        "evaluation": evaluation,
    }
    return active, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate rate-limited active subscription tracking."
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
    rest_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=0.0,
    )
    output, summary = run_probe(
        store, rest_client, args.start, args.end
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
