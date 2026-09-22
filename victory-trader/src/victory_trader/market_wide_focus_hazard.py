"""Causal market-wide next-step hazard probe for learned focus admission."""

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
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hierarchical_attention_runtime import (
    FOCUS_CONFIG,
    MODEL_KWARGS,
    _prior_population_capture,
    build_focus_rows,
)
from .market_calendar import is_us_equity_trading_day
from .massive_client import MassiveClient
from .multi_day import daterange
from .second_path_attention_probe import BASELINE_FEATURES, _annotate_scan

FIT_END = "2026-01-16"
EVAL_DAYS = [
    "2026-02-25",
    "2026-02-26",
    "2026-02-27",
    "2026-03-02",
    "2026-03-03",
]
FOCUS_BUDGET = 60
SHORTLIST_BUDGET = 20
HAZARD_COLUMNS = [
    "trading_day",
    "ticker",
    "t",
    *BASELINE_FEATURES,
    "target_next_cross",
]


def build_market_hazard_rows(scan: pd.DataFrame) -> pd.DataFrame:
    """Build causal market-wide rows and exact next-minute crossing targets."""

    frame = _annotate_scan(scan)
    frame["attention_rank"] = frame.groupby(
        ["trading_day", "t"], sort=False
    )["attention_score"].rank(method="first", ascending=False)
    ordered = frame.sort_values(["ticker", "t"], kind="stable")
    groups = ordered.groupby("ticker", sort=False)
    next_t = groups["t"].shift(-1)
    next_cross = groups["runner_cross_now"].shift(-1).astype("boolean")
    ordered["has_exact_next_minute"] = (
        pd.to_numeric(next_t, errors="coerce")
        - pd.to_numeric(ordered["t"], errors="coerce")
    ).eq(MINUTE_MS)
    ordered["target_next_cross"] = (
        next_cross.fillna(False).astype(bool)
        & ordered["has_exact_next_minute"]
    ).astype(int)
    rows = ordered.loc[
        ~ordered["already_runner"].fillna(True)
        & ordered["has_exact_next_minute"].fillna(False)
    ].copy()
    missing = set(BASELINE_FEATURES) - set(rows.columns)
    if missing:
        raise ValueError(f"market hazard rows missing columns: {sorted(missing)}")
    return rows


def fit_market_hazard(rows: pd.DataFrame) -> HistGradientBoostingClassifier:
    fit = rows.loc[rows["trading_day"].astype(str).le(FIT_END)].copy()
    if fit.empty or int(fit["target_next_cross"].sum()) == 0:
        raise ValueError("market-wide hazard fit is empty or has no positives")
    model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    model.fit(
        fit[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit["target_next_cross"].astype(int),
    )
    return model


def select_learned_focus(
    rows: pd.DataFrame,
    model: HistGradientBoostingClassifier,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = rows.copy()
    scored["market_hazard_probability"] = model.predict_proba(
        scored[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    ranked = scored.sort_values(
        ["trading_day", "t", "market_hazard_probability", "ticker"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    focus = ranked.groupby(["trading_day", "t"], sort=False).head(
        FOCUS_BUDGET
    )
    shortlist = focus.groupby(["trading_day", "t"], sort=False).head(
        SHORTLIST_BUDGET
    )
    return scored, focus.copy(), shortlist.copy()


def evaluate_focus_gate(
    eval_rows: pd.DataFrame,
    eval_scan: pd.DataFrame,
    baseline_focus: pd.DataFrame,
    learned_focus: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> dict[str, object]:
    baseline = _prior_population_capture(baseline_focus, eval_scan)
    learned = _prior_population_capture(learned_focus, eval_scan)
    top20 = _prior_population_capture(shortlist, eval_scan)
    learned_rate = learned["prior_minute_rate"]
    top20_rate = top20["prior_minute_rate"]
    conditional_top20 = (
        float(top20_rate / learned_rate)
        if learned_rate not in (None, 0) and top20_rate is not None
        else None
    )

    y = eval_rows["target_next_cross"].astype(int)
    model_pr_auc = float(
        average_precision_score(y, eval_rows["market_hazard_probability"])
    )
    attention_pr_auc = float(average_precision_score(y, eval_rows["attention_score"]))

    by_day: dict[str, object] = {}
    nonlower_days = 0
    support_ok = True
    for day in EVAL_DAYS:
        scan_day = eval_scan.loc[eval_scan["trading_day"].astype(str).eq(day)]
        baseline_day = baseline_focus.loc[
            baseline_focus["trading_day"].astype(str).eq(day)
        ]
        learned_day = learned_focus.loc[
            learned_focus["trading_day"].astype(str).eq(day)
        ]
        b = _prior_population_capture(baseline_day, scan_day)
        learned_metrics = _prior_population_capture(learned_day, scan_day)
        if int(b["runner_crossings"]) < 10:
            support_ok = False
        if (
            b["prior_minute_rate"] is not None
            and learned_metrics["prior_minute_rate"] is not None
            and learned_metrics["prior_minute_rate"] >= b["prior_minute_rate"]
        ):
            nonlower_days += 1
        by_day[day] = {"baseline": b, "learned": learned_metrics}

    baseline_rate = baseline["prior_minute_rate"]
    max_focus = int(
        learned_focus.groupby(["trading_day", "t"], sort=False).size().max()
    )
    max_shortlist = int(
        shortlist.groupby(["trading_day", "t"], sort=False).size().max()
    )
    gate = bool(
        support_ok
        and baseline_rate is not None
        and learned_rate is not None
        and learned_rate > baseline_rate
        and learned_rate >= 0.50
        and nonlower_days >= 4
        and model_pr_auc > attention_pr_auc
        and conditional_top20 is not None
        and conditional_top20 >= 0.95
        and max_focus <= FOCUS_BUDGET
        and max_shortlist <= SHORTLIST_BUDGET
    )
    return {
        "eval_days": EVAL_DAYS,
        "baseline_focus": baseline,
        "learned_focus": learned,
        "learned_top20": top20,
        "top20_conditional_on_learned_focus_rate": conditional_top20,
        "attention_score_pr_auc": attention_pr_auc,
        "market_hazard_pr_auc": model_pr_auc,
        "nonlower_focus_capture_days": nonlower_days,
        "max_focus_occupancy": max_focus,
        "max_shortlist_occupancy": max_shortlist,
        "by_day": by_day,
        "promotion_gate_pass": gate,
    }


def run_probe(
    store: MassiveFlatFileStore,
    rest_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    needed = {day.isoformat() for day in daterange(start, end) if day.isoformat() <= FIT_END}
    needed.update(EVAL_DAYS)
    eval_scans: list[pd.DataFrame] = []
    rows: list[pd.DataFrame] = []
    baseline_focus_rows: list[pd.DataFrame] = []
    day_summaries: list[dict[str, object]] = []
    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text not in needed or not is_us_equity_trading_day(day):
            continue
        scan, scan_summary = build_flatfile_scan_day(store, rest_client, day)
        market_rows = build_market_hazard_rows(scan)
        rows.append(market_rows.loc[:, HAZARD_COLUMNS].copy())
        if day_text in EVAL_DAYS:
            eval_scans.append(scan)
            baseline_focus, _ = build_focus_rows(scan)
            baseline_focus_rows.append(baseline_focus)
        day_summaries.append(
            {
                "trading_day": day_text,
                "scan_rows": int(len(scan)),
                "hazard_rows": int(len(market_rows)),
                "runner_crossings": int(scan["runner_cross_now"].sum()),
                "point_in_time_common_stocks": scan_summary.get(
                    "point_in_time_common_stocks"
                ),
            }
        )

    if not rows or not eval_scans or not baseline_focus_rows:
        raise ValueError("requested range does not contain fit and evaluation sessions")
    all_rows = pd.concat(rows, ignore_index=True)
    eval_scan = pd.concat(eval_scans, ignore_index=True)
    found_days = sorted(eval_scan["trading_day"].astype(str).unique())
    if found_days != EVAL_DAYS:
        raise ValueError(f"expected eval days {EVAL_DAYS}, found {found_days}")
    model = fit_market_hazard(all_rows)
    eval_rows = all_rows.loc[
        all_rows["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()
    baseline_focus = pd.concat(baseline_focus_rows, ignore_index=True)
    scored, learned_focus, shortlist = select_learned_focus(eval_rows, model)
    evaluation = evaluate_focus_gate(
        scored,
        eval_scan,
        baseline_focus,
        learned_focus,
        shortlist,
    )
    output = learned_focus.loc[
        :,
        [
            "trading_day",
            "ticker",
            "t",
            "market_hazard_probability",
            "target_next_cross",
        ],
    ].copy()
    top20_keys = set(
        zip(
            shortlist["trading_day"].astype(str),
            shortlist["ticker"].astype(str),
            shortlist["t"].astype(int),
            strict=False,
        )
    )
    output["in_top20"] = [
        (str(row.trading_day), str(row.ticker), int(row.t)) in top20_keys
        for row in output.itertuples(index=False)
    ]
    summary: dict[str, object] = {
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "fit_end": FIT_END,
        "eval_days": EVAL_DAYS,
        "focus_budget": FOCUS_BUDGET,
        "shortlist_budget": SHORTLIST_BUDGET,
        "features": BASELINE_FEATURES,
        "model": MODEL_KWARGS,
        "attention_config": {
            "watch_enter_score": FOCUS_CONFIG.watch_enter_score,
            "watch_exit_score": FOCUS_CONFIG.watch_exit_score,
            "max_watch": FOCUS_CONFIG.max_watch,
        },
        "days": day_summaries,
        "fit_rows": int(
            all_rows["trading_day"].astype(str).le(FIT_END).sum()
        ),
        "eval_rows": int(len(eval_rows)),
        "flatfile_stats": store.stats.to_dict(),
        "evaluation": evaluation,
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe learned market-wide focus admission.")
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
    output, summary = run_probe(store, rest_client, args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
