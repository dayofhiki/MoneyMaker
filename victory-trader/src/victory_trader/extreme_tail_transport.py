"""Request237: frozen extreme-tail transport test.

Request236, using calibration outcomes only, found one predeclared economically
positive pocket: the top 1% of the frozen Request232 regression score.

Request237 freezes that exact family and quantile. It computes the 99th
percentile score threshold using calibration rows only, then applies that
unchanged threshold to the already-designated June 8-12 fresh block.

No alternate fraction, score family, or threshold is inspected after fresh
outcomes are opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .extended_history_economic_opportunity import EXTENDED_CAL_DAYS
from .fresh_validated_attention_economic_opportunity import FRESH_EVAL_DAYS

REQUEST_ID = 237
SOURCE_RUN = 36251239308
FROZEN_QUANTILE = 0.99
SCORE_COLUMN = "predicted_persistent_tradability_pct"

MIN_FRESH_SELECTED_ROWS = 8
MIN_FRESH_DAYS_WITH_SELECTION = 3
MIN_FRESH_POSITIVE_MEAN_DAYS = 3
MIN_FRESH_POSITIVE_RATE = 0.45


def _metrics(frame: pd.DataFrame) -> dict[str, object]:
    target = pd.to_numeric(
        frame["persistent_tradability_pct"],
        errors="coerce",
    )
    valid = target.notna()
    work = frame.loc[valid].copy()
    target = pd.to_numeric(
        work["persistent_tradability_pct"],
        errors="coerce",
    )

    by_day: dict[str, object] = {}
    day_means: list[float] = []
    positive_days = 0
    days_with_selection = 0

    for day in sorted(work["trading_day"].astype(str).unique()):
        part = work.loc[
            work["trading_day"].astype(str).eq(day)
        ].copy()
        y = pd.to_numeric(
            part["persistent_tradability_pct"],
            errors="coerce",
        ).dropna()
        mean = float(y.mean()) if len(y) else None
        if mean is not None:
            day_means.append(mean)
            days_with_selection += 1
            positive_days += int(mean > 0)
        by_day[day] = {
            "rows": int(len(y)),
            "mean_pct": mean,
            "median_pct": (
                float(y.median()) if len(y) else None
            ),
            "positive_rate": (
                float(y.gt(0).mean()) if len(y) else None
            ),
            "severe_loss_rate_lt_minus2": (
                float(y.lt(-2).mean()) if len(y) else None
            ),
            "severe_loss_rate_lt_minus5": (
                float(y.lt(-5).mean()) if len(y) else None
            ),
        }

    return {
        "rows": int(len(target)),
        "mean_pct": (
            float(target.mean()) if len(target) else None
        ),
        "day_balanced_mean_pct": (
            float(sum(day_means) / len(day_means))
            if day_means else None
        ),
        "median_pct": (
            float(target.median()) if len(target) else None
        ),
        "positive_rate": (
            float(target.gt(0).mean()) if len(target) else None
        ),
        "severe_loss_rate_lt_minus2": (
            float(target.lt(-2).mean()) if len(target) else None
        ),
        "severe_loss_rate_lt_minus5": (
            float(target.lt(-5).mean()) if len(target) else None
        ),
        "days_with_selection": days_with_selection,
        "positive_mean_days": positive_days,
        "by_day": by_day,
    }


def evaluate(
    scored_hot_path: Path,
    output_path: Path,
) -> int:
    frame = pd.read_parquet(scored_hot_path)

    calibration = frame.loc[
        frame["trading_day"].astype(str).isin(EXTENDED_CAL_DAYS)
    ].copy()
    cal_score = pd.to_numeric(
        calibration[SCORE_COLUMN],
        errors="coerce",
    )
    cal_target = pd.to_numeric(
        calibration["persistent_tradability_pct"],
        errors="coerce",
    )
    usable_cal = calibration.loc[
        cal_score.notna() & cal_target.notna()
    ].copy()
    threshold = float(
        pd.to_numeric(
            usable_cal[SCORE_COLUMN],
            errors="coerce",
        ).quantile(FROZEN_QUANTILE)
    )

    calibration_selected = usable_cal.loc[
        pd.to_numeric(
            usable_cal[SCORE_COLUMN],
            errors="coerce",
        ).ge(threshold)
    ].copy()

    fresh = frame.loc[
        frame["trading_day"].astype(str).isin(FRESH_EVAL_DAYS)
    ].copy()
    fresh_target = pd.to_numeric(
        fresh["persistent_tradability_pct"],
        errors="coerce",
    )
    fresh_score = pd.to_numeric(
        fresh[SCORE_COLUMN],
        errors="coerce",
    )
    evaluable_fresh = fresh.loc[
        fresh_target.notna() & fresh_score.notna()
    ].copy()
    fresh_selected = evaluable_fresh.loc[
        pd.to_numeric(
            evaluable_fresh[SCORE_COLUMN],
            errors="coerce",
        ).ge(threshold)
    ].copy()

    calibration_metrics = _metrics(calibration_selected)
    fresh_population = _metrics(evaluable_fresh)
    fresh_metrics = _metrics(fresh_selected)

    fresh_mean = fresh_metrics["mean_pct"]
    fresh_day_mean = fresh_metrics["day_balanced_mean_pct"]
    fresh_positive_rate = fresh_metrics["positive_rate"]

    checks = {
        "min_selected_rows": (
            fresh_metrics["rows"] >= MIN_FRESH_SELECTED_ROWS
        ),
        "min_days_with_selection": (
            fresh_metrics["days_with_selection"]
            >= MIN_FRESH_DAYS_WITH_SELECTION
        ),
        "selected_mean_positive": (
            fresh_mean is not None
            and float(fresh_mean) > 0
        ),
        "day_balanced_mean_positive": (
            fresh_day_mean is not None
            and float(fresh_day_mean) > 0
        ),
        "min_positive_mean_days": (
            fresh_metrics["positive_mean_days"]
            >= MIN_FRESH_POSITIVE_MEAN_DAYS
        ),
        "min_positive_rate": (
            fresh_positive_rate is not None
            and float(fresh_positive_rate)
            >= MIN_FRESH_POSITIVE_RATE
        ),
    }
    gate = bool(all(checks.values()))

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "source_run": SOURCE_RUN,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "frozen_rule": {
            "score_column": SCORE_COLUMN,
            "calibration_quantile": FROZEN_QUANTILE,
            "calibration_score_threshold": threshold,
            "chosen_from_request236_fresh_blind_diagnostic": True,
            "fresh_used_to_choose_rule": False,
        },
        "calibration_selected": calibration_metrics,
        "fresh_population": fresh_population,
        "fresh_selected": fresh_metrics,
        "fresh_gate": {
            "checks": checks,
            "pass": gate,
            "min_selected_rows": MIN_FRESH_SELECTED_ROWS,
            "min_days_with_selection": (
                MIN_FRESH_DAYS_WITH_SELECTION
            ),
            "min_positive_mean_days": (
                MIN_FRESH_POSITIVE_MEAN_DAYS
            ),
            "min_positive_rate": MIN_FRESH_POSITIVE_RATE,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze the extreme-tail admission family and test a causal "
            "risk-controlled ENTER/HOLD/EXIT policy inside those episodes."
        ),
        "next_boundary_if_fail": (
            "the calibration-only extreme tail does not transport to "
            "June; the current tradability representation is not stable "
            "enough for admission and must be redesigned before timing."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-hot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(args.scored_hot, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
