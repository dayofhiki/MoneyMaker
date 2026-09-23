"""Diagnose request-140B economic-label coverage without opening new dates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DAYS = [
    "2026-05-14",
    "2026-05-15",
    "2026-05-18",
    "2026-05-19",
    "2026-05-20",
]


def _finite_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    return {
        "count": int(len(values)),
        "median": float(values.median()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    day = frame["trading_day"].astype(str)
    evaluation = frame.loc[day.isin(EVAL_DAYS)].copy()
    oracle = pd.to_numeric(evaluation["oracle_best_base_pct"], errors="coerce")
    entry = evaluation["entry_reference_available"].fillna(False).astype(bool)
    labeled = oracle.notna()
    missing = ~labeled
    no_entry = missing & ~entry
    no_exit_after_entry = missing & entry

    bins = [-np.inf, 1, 2, 5, 10, 30, np.inf]
    labels = ["<=1", "(1,2]", "(2,5]", "(5,10]", "(10,30]", ">30"]
    close_bucket = pd.cut(
        pd.to_numeric(evaluation["minutes_to_close"], errors="coerce"),
        bins=bins,
        labels=labels,
        right=True,
    )

    by_day: dict[str, object] = {}
    for day_text in EVAL_DAYS:
        mask = day.eq(day_text)
        group = evaluation.loc[mask].copy()
        group_oracle = pd.to_numeric(group["oracle_best_base_pct"], errors="coerce")
        group_entry = group["entry_reference_available"].fillna(False).astype(bool)
        group_missing = group_oracle.isna()
        group_no_entry = group_missing & ~group_entry
        group_no_exit = group_missing & group_entry
        by_day[day_text] = {
            "rows": int(len(group)),
            "labeled": int(group_oracle.notna().sum()),
            "coverage": float(group_oracle.notna().mean()) if len(group) else None,
            "missing": int(group_missing.sum()),
            "missing_entry_reference": int(group_no_entry.sum()),
            "entry_but_no_future_exit": int(group_no_exit.sum()),
            "missing_minutes_to_close": _finite_summary(
                group.loc[group_missing, "minutes_to_close"]
            ),
            "labeled_minutes_to_close": _finite_summary(
                group.loc[~group_missing, "minutes_to_close"]
            ),
            "missing_current_price": _finite_summary(
                group.loc[group_missing, "current_price"]
            ),
            "labeled_current_price": _finite_summary(
                group.loc[~group_missing, "current_price"]
            ),
            "missing_active_seconds_60": _finite_summary(
                group.loc[group_missing, "active_seconds_60"]
            ),
            "labeled_active_seconds_60": _finite_summary(
                group.loc[~group_missing, "active_seconds_60"]
            ),
        }

    bucket_table: dict[str, object] = {}
    for label in labels:
        bucket_mask = close_bucket.astype(str).eq(label)
        group = evaluation.loc[bucket_mask]
        group_oracle = pd.to_numeric(group["oracle_best_base_pct"], errors="coerce")
        bucket_table[label] = {
            "rows": int(len(group)),
            "missing": int(group_oracle.isna().sum()),
            "coverage": float(group_oracle.notna().mean()) if len(group) else None,
        }

    return {
        "schema_version": 1,
        "source": "request-140B artifact",
        "opens_new_dates": False,
        "eval_days": EVAL_DAYS,
        "rows": int(len(evaluation)),
        "labeled": int(labeled.sum()),
        "coverage": float(labeled.mean()) if len(evaluation) else None,
        "missing": int(missing.sum()),
        "missing_entry_reference": int(no_entry.sum()),
        "entry_but_no_future_exit": int(no_exit_after_entry.sum()),
        "missing_minutes_to_close": _finite_summary(
            evaluation.loc[missing, "minutes_to_close"]
        ),
        "labeled_minutes_to_close": _finite_summary(
            evaluation.loc[labeled, "minutes_to_close"]
        ),
        "coverage_by_minutes_to_close_bucket": bucket_table,
        "by_day": by_day,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    result = summarize(frame)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
