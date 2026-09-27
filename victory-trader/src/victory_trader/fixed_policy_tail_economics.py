"""Request250: calibration-only tail economics for Request249 fixed policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .extended_history_economic_opportunity import EXTENDED_CAL_DAYS
from .full_hot_fixed_policy_value import (
    attach_fixed_value,
    train,
    xframe,
)

REQUEST_ID = 250
FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
FAMILY_ORDER = ("predicted_value", "joint_percentile", "positive_probability")
MIN_SELECTED_ROWS = 40
MIN_POSITIVE_RATE = 0.35
MIN_POSITIVE_MEAN_DAYS = 5


def score_calibration(first_hot: pd.DataFrame) -> pd.DataFrame:
    model = train(first_hot)
    cal = first_hot.loc[
        first_hot.trading_day.astype(str).isin(EXTENDED_CAL_DAYS)
    ].copy()
    target = pd.to_numeric(
        cal.fixed_first_watch_value_pct,
        errors="coerce",
    )
    cal = cal.loc[target.notna()].copy()
    x = xframe(cal, model.columns)
    cal["predicted_value"] = model.regressor.predict(x) + model.offset
    cal["positive_probability"] = model.classifier.predict_proba(x)[:, 1]
    cal["predicted_value_percentile"] = cal.predicted_value.rank(
        pct=True,
        method="average",
    )
    cal["positive_probability_percentile"] = cal.positive_probability.rank(
        pct=True,
        method="average",
    )
    cal["joint_percentile"] = (
        cal.predicted_value_percentile
        + cal.positive_probability_percentile
    ) / 2.0
    return cal


def metrics(
    frame: pd.DataFrame,
    family: str,
    fraction: float,
) -> tuple[dict, float]:
    score = pd.to_numeric(frame[family], errors="coerce")
    valid = score.notna()
    population = frame.loc[valid].copy()
    score = score.loc[valid]
    threshold = float(np.quantile(score.to_numpy(float), 1.0 - fraction))
    selected = population.loc[
        pd.to_numeric(population[family], errors="coerce").ge(threshold)
    ].copy()
    target = pd.to_numeric(
        population.fixed_first_watch_value_pct,
        errors="coerce",
    )
    chosen_target = pd.to_numeric(
        selected.fixed_first_watch_value_pct,
        errors="coerce",
    )
    day_means = []
    positive_days = 0
    by_day = {}
    for day in EXTENDED_CAL_DAYS:
        chosen_day = selected.loc[
            selected.trading_day.astype(str).eq(str(day))
        ]
        day_target = pd.to_numeric(
            chosen_day.fixed_first_watch_value_pct,
            errors="coerce",
        )
        day_mean = float(day_target.mean()) if len(day_target) else None
        if day_mean is not None:
            day_means.append(day_mean)
            positive_days += int(day_mean > 0)
        by_day[str(day)] = {
            "selected_rows": int(len(chosen_day)),
            "selected_mean_pct": day_mean,
            "selected_positive_rate": (
                float(day_target.gt(0).mean()) if len(day_target) else None
            ),
        }

    selected_mean = (
        float(chosen_target.mean()) if len(chosen_target) else None
    )
    day_balanced_mean = (
        float(np.mean(day_means)) if day_means else None
    )
    selected_positive = (
        float(chosen_target.gt(0).mean()) if len(chosen_target) else None
    )
    severe = (
        float(chosen_target.le(-2).mean()) if len(chosen_target) else None
    )
    population_severe = (
        float(target.le(-2).mean()) if len(target) else None
    )
    checks = {
        "min_selected_rows": len(selected) >= MIN_SELECTED_ROWS,
        "selected_mean_positive": (
            selected_mean is not None and selected_mean > 0
        ),
        "day_balanced_mean_positive": (
            day_balanced_mean is not None and day_balanced_mean > 0
        ),
        "selected_positive_rate": (
            selected_positive is not None
            and selected_positive >= MIN_POSITIVE_RATE
        ),
        "positive_mean_days": positive_days >= MIN_POSITIVE_MEAN_DAYS,
        "severe_loss_not_worse": (
            severe is not None
            and population_severe is not None
            and severe <= population_severe
        ),
    }
    return {
        "family": family,
        "fraction": fraction,
        "threshold": threshold,
        "population_rows": int(len(population)),
        "population_mean_pct": float(target.mean()) if len(target) else None,
        "population_positive_rate": (
            float(target.gt(0).mean()) if len(target) else None
        ),
        "population_severe_loss_rate_le_minus2": population_severe,
        "selected_rows": int(len(selected)),
        "selected_rate": (
            float(len(selected) / len(population)) if len(population) else None
        ),
        "selected_mean_pct": selected_mean,
        "day_balanced_selected_mean_pct": day_balanced_mean,
        "selected_positive_rate": selected_positive,
        "selected_severe_loss_rate_le_minus2": severe,
        "positive_selected_mean_days": positive_days,
        "by_day": by_day,
        "checks": checks,
        "passes": bool(all(checks.values())),
    }, threshold


def choose_rule(results: list[dict]) -> dict | None:
    passing = [r for r in results if r["passes"]]
    if not passing:
        return None
    family_rank = {name: i for i, name in enumerate(FAMILY_ORDER)}
    passing.sort(
        key=lambda r: (
            -float(r["fraction"]),
            family_rank[str(r["family"])],
        )
    )
    return passing[0]


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    required = {
        "trading_day",
        "ticker",
        "t",
        "fixed_first_watch_value_pct",
    }
    # Request233 is the frozen full first-HOT population. Its old outcome
    # columns are ignored; Request250 attaches the new Request249 fixed policy
    # target from the original scan.
    base_required = {"trading_day", "ticker", "t"}
    missing = base_required - set(first_hot.columns)
    if missing:
        raise ValueError(
            f"Request233 first-HOT artifact missing: {sorted(missing)}"
        )
    first_hot = first_hot.drop(
        columns=["fixed_first_watch_value_pct", "fixed_entry_elapsed_minutes"],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, scan)
    if not required.issubset(first_hot.columns):
        raise ValueError("Request250 fixed target attachment failed")
    calibration = score_calibration(first_hot)

    results = []
    for family in FAMILY_ORDER:
        for fraction in FRACTIONS:
            row, _ = metrics(calibration, family, fraction)
            results.append(row)
    chosen = choose_rule(results)

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "input_population": (
            "frozen Request233 full first-HOT artifact; old Request232/233 "
            "outcomes and scores are not used for the Request250 gate"
        ),
        "calibration_days": list(EXTENDED_CAL_DAYS),
        "predeclared_fractions": list(FRACTIONS),
        "score_families": list(FAMILY_ORDER),
        "first_hot_rows": int(len(first_hot)),
        "calibration_rows": int(len(calibration)),
        "results": results,
        "chosen_rule": (
            {
                "family": chosen["family"],
                "fraction": chosen["fraction"],
                "threshold": chosen["threshold"],
                "selected_rows": chosen["selected_rows"],
                "selected_mean_pct": chosen["selected_mean_pct"],
                "day_balanced_selected_mean_pct": (
                    chosen["day_balanced_selected_mean_pct"]
                ),
                "selected_positive_rate": chosen["selected_positive_rate"],
                "selected_severe_loss_rate_le_minus2": (
                    chosen["selected_severe_loss_rate_le_minus2"]
                ),
                "positive_selected_mean_days": (
                    chosen["positive_selected_mean_days"]
                ),
            }
            if chosen is not None
            else None
        ),
        "calibration_gate_pass": chosen is not None,
        "next_boundary": (
            "freeze rule and open June15-19"
            if chosen is not None
            else "add causal downside-survivability representation"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0

