"""Diagnose request-131 focus and shortlist misses without tuning on April."""

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
from .integrated_learned_focus_hierarchy import learned_shortlist
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
from .second_path_attention_probe import BASELINE_FEATURES

EVAL_DAYS = [
    "2026-04-01",
    "2026-04-02",
    "2026-04-06",
    "2026-04-07",
    "2026-04-08",
]


def score_market_rows(rows: pd.DataFrame, model) -> pd.DataFrame:
    scored = rows.copy()
    scored["market_hazard_probability"] = model.predict_proba(
        scored[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    scored = scored.sort_values(
        ["trading_day", "t", "market_hazard_probability", "ticker"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    scored["focus_hazard_rank"] = scored.groupby(
        ["trading_day", "t"], sort=False
    ).cumcount() + 1
    return scored


def _selection_keys(frame: pd.DataFrame) -> set[tuple[str, str, int]]:
    return set(
        zip(
            frame["trading_day"].astype(str),
            frame["ticker"].astype(str),
            frame["t"].astype(int),
            strict=False,
        )
    )


def _capture(frame: pd.DataFrame, scan: pd.DataFrame) -> dict[str, object]:
    return _prior_population_capture(frame, scan)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _shortlist_churn(shortlist: pd.DataFrame) -> dict[str, object]:
    additions: list[int] = []
    retentions: list[float] = []
    ticker_days = int(
        shortlist.loc[:, ["trading_day", "ticker"]].drop_duplicates().shape[0]
    )
    for _, day_rows in shortlist.groupby("trading_day", sort=True):
        previous: set[str] = set()
        for _, group in day_rows.groupby("t", sort=True):
            current = set(group["ticker"].astype(str))
            additions.append(len(current - previous))
            if previous:
                retentions.append(len(current & previous) / len(previous))
            previous = current
    return {
        "ticker_day_requests": ticker_days,
        "mean_additions_per_decision": (
            float(np.mean(additions)) if additions else None
        ),
        "mean_set_retention": (
            float(np.mean(retentions)) if retentions else None
        ),
    }


def _positive_rank_summary(positives: pd.DataFrame) -> dict[str, object]:
    ranks = pd.to_numeric(positives["focus_hazard_rank"], errors="coerce").dropna()
    if ranks.empty:
        return {"positives": 0}
    return {
        "positives": int(len(ranks)),
        "rank_le_20_rate": float(ranks.le(20).mean()),
        "rank_le_40_rate": float(ranks.le(40).mean()),
        "rank_le_60_rate": float(ranks.le(60).mean()),
        "rank_le_80_rate": float(ranks.le(80).mean()),
        "rank_le_100_rate": float(ranks.le(100).mean()),
        "median_rank": float(ranks.median()),
        "p75_rank": float(ranks.quantile(0.75)),
        "p90_rank": float(ranks.quantile(0.90)),
    }


def diagnose_selection(
    scored: pd.DataFrame,
    eval_scan: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    focus = scored.loc[scored["focus_hazard_rank"].le(FOCUS_BUDGET)].copy()
    stateless = focus.loc[
        focus["focus_hazard_rank"].le(SHORTLIST_BUDGET)
    ].copy()
    hysteresis = learned_shortlist(focus)

    stateless_keys = _selection_keys(stateless)
    hysteresis_keys = _selection_keys(hysteresis)
    positives = scored.loc[scored["target_next_cross"].astype(int).eq(1)].copy()
    positives["in_focus"] = positives["focus_hazard_rank"].le(FOCUS_BUDGET)
    positives["in_stateless_top20"] = [
        key in stateless_keys
        for key in zip(
            positives["trading_day"].astype(str),
            positives["ticker"].astype(str),
            positives["t"].astype(int),
            strict=False,
        )
    ]
    positives["in_hysteresis_top20"] = [
        key in hysteresis_keys
        for key in zip(
            positives["trading_day"].astype(str),
            positives["ticker"].astype(str),
            positives["t"].astype(int),
            strict=False,
        )
    ]
    positives["blocked_current_top20"] = (
        positives["focus_hazard_rank"].le(SHORTLIST_BUDGET)
        & ~positives["in_hysteresis_top20"]
    )

    focus_metrics = _capture(focus, eval_scan)
    stateless_metrics = _capture(stateless, eval_scan)
    hysteresis_metrics = _capture(hysteresis, eval_scan)
    focus_count = int(focus_metrics["prior_minute_count"])
    stateless_count = int(stateless_metrics["prior_minute_count"])
    hysteresis_count = int(hysteresis_metrics["prior_minute_count"])

    by_day: dict[str, object] = {}
    for day in EVAL_DAYS:
        scan_day = eval_scan.loc[eval_scan["trading_day"].astype(str).eq(day)]
        focus_day = focus.loc[focus["trading_day"].astype(str).eq(day)]
        stateless_day = stateless.loc[
            stateless["trading_day"].astype(str).eq(day)
        ]
        hysteresis_day = hysteresis.loc[
            hysteresis["trading_day"].astype(str).eq(day)
        ]
        positive_day = positives.loc[
            positives["trading_day"].astype(str).eq(day)
        ]
        by_day[day] = {
            "focus": _capture(focus_day, scan_day),
            "stateless_top20": _capture(stateless_day, scan_day),
            "hysteresis_top20": _capture(hysteresis_day, scan_day),
            "positive_rank": _positive_rank_summary(positive_day),
            "blocked_current_top20_count": int(
                positive_day["blocked_current_top20"].sum()
            ),
        }

    summary = {
        "eval_days": EVAL_DAYS,
        "focus": focus_metrics,
        "stateless_top20": stateless_metrics,
        "hysteresis_top20": hysteresis_metrics,
        "stateless_conditional_on_focus_rate": _safe_ratio(
            stateless_count, focus_count
        ),
        "hysteresis_conditional_on_focus_rate": _safe_ratio(
            hysteresis_count, focus_count
        ),
        "hysteresis_capture_delta_vs_stateless": (
            hysteresis_count - stateless_count
        ),
        "blocked_current_top20_count": int(
            positives["blocked_current_top20"].sum()
        ),
        "blocked_current_top20_rate_of_positives": float(
            positives["blocked_current_top20"].mean()
        ) if len(positives) else None,
        "positive_rank": _positive_rank_summary(positives),
        "stateless_churn": _shortlist_churn(stateless),
        "hysteresis_churn": _shortlist_churn(hysteresis),
        "by_day": by_day,
    }
    return positives, summary


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
                "phase": "fit" if in_fit else "diagnostic",
                "scan_rows": int(len(scan)),
                "hazard_rows": int(len(rows)),
                "runner_crossings": int(scan["runner_cross_now"].sum()),
                "point_in_time_common_stocks": scan_summary.get(
                    "point_in_time_common_stocks"
                ),
            }
        )

    if not fit_rows or not eval_rows or not eval_scans:
        raise ValueError("diagnostic requires fit rows and all opened April sessions")
    scan = pd.concat(eval_scans, ignore_index=True)
    found_days = sorted(scan["trading_day"].astype(str).unique())
    if found_days != EVAL_DAYS:
        raise ValueError(f"expected eval days {EVAL_DAYS}, found {found_days}")

    model = fit_market_hazard(pd.concat(fit_rows, ignore_index=True))
    scored = score_market_rows(pd.concat(eval_rows, ignore_index=True), model)
    positives, evaluation = diagnose_selection(scored, scan)
    output = positives.loc[
        :,
        [
            "trading_day",
            "ticker",
            "t",
            "market_hazard_probability",
            "focus_hazard_rank",
            "in_focus",
            "in_stateless_top20",
            "in_hysteresis_top20",
            "blocked_current_top20",
        ],
    ].copy()
    summary = {
        "schema_version": 1,
        "diagnostic_only": True,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "fit_end": FIT_END,
        "eval_days": EVAL_DAYS,
        "focus_budget": FOCUS_BUDGET,
        "shortlist_budget": SHORTLIST_BUDGET,
        "features": BASELINE_FEATURES,
        "days": days,
        "flatfile_stats": store.stats.to_dict(),
        "evaluation": evaluation,
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose opened request-131 focus/shortlist misses."
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
