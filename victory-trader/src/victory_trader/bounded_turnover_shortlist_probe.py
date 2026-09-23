"""Fresh-block validation of bounded-turnover shortlist admission."""

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
    BoundedTurnoverSelector,
    RankHysteresisSelector,
    RankedCandidate,
)
from .second_path_attention_probe import BASELINE_FEATURES

EVAL_DAYS = [
    "2026-04-09",
    "2026-04-10",
    "2026-04-13",
    "2026-04-14",
    "2026-04-15",
]
INCUMBENT_RANK_LIMIT = 40
MAX_REPLACEMENTS = 5


def select_shortlist(
    focus: pd.DataFrame,
    *,
    bounded_turnover: bool,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for _, day_rows in focus.groupby("trading_day", sort=True):
        if bounded_turnover:
            selector = BoundedTurnoverSelector(
                capacity=SHORTLIST_BUDGET,
                incumbent_rank_limit=INCUMBENT_RANK_LIMIT,
                max_replacements=MAX_REPLACEMENTS,
            )
        else:
            selector = RankHysteresisSelector(
                capacity=SHORTLIST_BUDGET,
                incumbent_rank_limit=INCUMBENT_RANK_LIMIT,
            )
        for _, group in day_rows.groupby("t", sort=True):
            candidates = [
                RankedCandidate(
                    ticker=str(row.ticker),
                    score=float(row.market_hazard_probability),
                )
                for row in group.itertuples(index=False)
            ]
            chosen = set(selector.select(candidates))
            selected.append(
                group.loc[group["ticker"].astype(str).isin(chosen)].copy()
            )
    return (
        pd.concat(selected, ignore_index=True)
        if selected
        else focus.iloc[0:0].copy()
    )


def shortlist_audit(shortlist: pd.DataFrame) -> dict[str, object]:
    additions: list[int] = []
    retention: list[float] = []
    occupancies: list[int] = []
    ticker_days = int(
        shortlist.loc[:, ["trading_day", "ticker"]].drop_duplicates().shape[0]
    )
    for _, day_rows in shortlist.groupby("trading_day", sort=True):
        previous: set[str] = set()
        for _, group in day_rows.groupby("t", sort=True):
            current = set(group["ticker"].astype(str))
            occupancies.append(len(current))
            additions.append(len(current - previous))
            if previous:
                retention.append(len(current & previous) / len(previous))
            previous = current
    return {
        "ticker_day_requests": ticker_days,
        "decisions": len(occupancies),
        "max_occupancy": max(occupancies, default=0),
        "mean_additions_per_decision": (
            float(np.mean(additions)) if additions else None
        ),
        "max_additions_per_decision": max(additions, default=0),
        "mean_set_retention": (
            float(np.mean(retention)) if retention else None
        ),
    }


def _conditional(shortlist: dict[str, object], focus: dict[str, object]) -> float | None:
    denominator = int(focus["prior_minute_count"])
    numerator = int(shortlist["prior_minute_count"])
    return float(numerator / denominator) if denominator else None


def evaluate(
    eval_scan: pd.DataFrame,
    focus: pd.DataFrame,
    old_shortlist: pd.DataFrame,
    bounded_shortlist: pd.DataFrame,
) -> dict[str, object]:
    focus_metrics = _prior_population_capture(focus, eval_scan)
    old_metrics = _prior_population_capture(old_shortlist, eval_scan)
    bounded_metrics = _prior_population_capture(bounded_shortlist, eval_scan)
    old_conditional = _conditional(old_metrics, focus_metrics)
    bounded_conditional = _conditional(bounded_metrics, focus_metrics)
    old_audit = shortlist_audit(old_shortlist)
    bounded_audit = shortlist_audit(bounded_shortlist)

    by_day: dict[str, object] = {}
    support_ok = True
    nonlower_days = 0
    for day in EVAL_DAYS:
        scan_day = eval_scan.loc[eval_scan["trading_day"].astype(str).eq(day)]
        focus_day = focus.loc[focus["trading_day"].astype(str).eq(day)]
        old_day = old_shortlist.loc[
            old_shortlist["trading_day"].astype(str).eq(day)
        ]
        bounded_day = bounded_shortlist.loc[
            bounded_shortlist["trading_day"].astype(str).eq(day)
        ]
        f = _prior_population_capture(focus_day, scan_day)
        old = _prior_population_capture(old_day, scan_day)
        new = _prior_population_capture(bounded_day, scan_day)
        if int(f["runner_crossings"]) < 10:
            support_ok = False
        if (
            old["prior_minute_rate"] is not None
            and new["prior_minute_rate"] is not None
            and new["prior_minute_rate"] >= old["prior_minute_rate"]
        ):
            nonlower_days += 1
        by_day[day] = {
            "focus": f,
            "rank40_hysteresis": old,
            "bounded_turnover": new,
        }

    old_rate = old_metrics["prior_minute_rate"]
    new_rate = bounded_metrics["prior_minute_rate"]
    gate = bool(
        support_ok
        and old_rate is not None
        and new_rate is not None
        and new_rate > old_rate
        and bounded_conditional is not None
        and bounded_conditional >= 0.95
        and nonlower_days >= 4
        and int(bounded_audit["max_occupancy"]) <= SHORTLIST_BUDGET
        and bounded_audit["mean_additions_per_decision"] is not None
        and float(bounded_audit["mean_additions_per_decision"]) <= 5.0
    )
    return {
        "eval_days": EVAL_DAYS,
        "focus": focus_metrics,
        "rank40_hysteresis": old_metrics,
        "bounded_turnover": bounded_metrics,
        "rank40_conditional_on_focus_rate": old_conditional,
        "bounded_conditional_on_focus_rate": bounded_conditional,
        "rank40_audit": old_audit,
        "bounded_audit": bounded_audit,
        "nonlower_capture_days": nonlower_days,
        "selector_gate_pass": gate,
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
        raise ValueError("selector validation requires fit and evaluation rows")
    eval_scan = pd.concat(eval_scans, ignore_index=True)
    found_days = sorted(eval_scan["trading_day"].astype(str).unique())
    if found_days != EVAL_DAYS:
        raise ValueError(f"expected eval days {EVAL_DAYS}, found {found_days}")

    model = fit_market_hazard(pd.concat(fit_rows, ignore_index=True))
    scored = score_market_rows(pd.concat(eval_rows, ignore_index=True), model)
    focus = scored.loc[scored["focus_hazard_rank"].le(FOCUS_BUDGET)].copy()
    old_shortlist = select_shortlist(focus, bounded_turnover=False)
    bounded_shortlist = select_shortlist(focus, bounded_turnover=True)
    evaluation = evaluate(
        eval_scan,
        focus,
        old_shortlist,
        bounded_shortlist,
    )

    old_keys = set(
        zip(
            old_shortlist["trading_day"].astype(str),
            old_shortlist["ticker"].astype(str),
            old_shortlist["t"].astype(int),
            strict=False,
        )
    )
    output = bounded_shortlist.loc[
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
    output["in_rank40_hysteresis"] = [
        (str(row.trading_day), str(row.ticker), int(row.t)) in old_keys
        for row in output.itertuples(index=False)
    ]
    summary = {
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "fit_end": FIT_END,
        "eval_days": EVAL_DAYS,
        "focus_budget": FOCUS_BUDGET,
        "shortlist_budget": SHORTLIST_BUDGET,
        "incumbent_rank_limit": INCUMBENT_RANK_LIMIT,
        "max_replacements_per_decision": MAX_REPLACEMENTS,
        "features": BASELINE_FEATURES,
        "days": days,
        "flatfile_stats": store.stats.to_dict(),
        "evaluation": evaluation,
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate bounded-turnover shortlist on a fresh block."
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
