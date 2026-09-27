"""Request253: causal pullback/rebound entry audit on full-HOT candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .attention_replay import MINUTE_MS
from .extended_history_economic_opportunity import (
    EXTENDED_CAL_DAYS,
    EXTENDED_FIT_DAYS,
)
from .full_hot_fixed_policy_value import (
    attach_fixed_value,
    event_lookup,
    usable_columns,
    xframe,
)
from .recurrent_wait_entry_action_value import _base_return

REQUEST_ID = 253
ENTRY_WINDOW_MINUTES = 10
RISK_CAP_MINUTES = 30
EVENT_HORIZONS = (1, 2, 3, 5)
CANDIDATE_FRACTIONS = (0.05, 0.10, 0.20)
RULE_ORDER = (
    "pullback_1",
    "pullback_2",
    "pullback_3",
    "rebound_2_1",
)
MIN_RESOLVED_ROWS = 40
MIN_ENTRY_COVERAGE = 0.25
MIN_POSITIVE_RATE = 0.40
MIN_POSITIVE_MEAN_DAYS = 5
MAX_SEVERE_RATE = 0.20
MIN_GAIN_VS_IMMEDIATE = 0.50


def fit_candidate_model(
    first_hot: pd.DataFrame,
) -> tuple[HistGradientBoostingClassifier, tuple[str, ...]]:
    fit = first_hot.loc[
        first_hot.trading_day.astype(str).isin(EXTENDED_FIT_DAYS)
    ].copy()
    target = pd.to_numeric(
        fit.fixed_first_watch_value_pct,
        errors="coerce",
    )
    fit = fit.loc[target.notna()].copy()
    target = pd.to_numeric(
        fit.fixed_first_watch_value_pct,
        errors="coerce",
    )
    columns = usable_columns(fit)
    y = target.gt(0).astype(int)
    if y.nunique() != 2:
        raise ValueError("Request253 candidate model needs both classes")
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=20261350,
    )
    model.fit(xframe(fit, columns), y)
    return model, columns


def pick_entry(
    events: list[tuple[int, float]],
    rule: str,
) -> tuple[int, float] | None:
    if not events:
        return None
    if rule == "immediate":
        return events[0]

    running_high = float(events[0][1])
    activated = False
    pullback_low = float(events[0][1])
    drawdown_required = {
        "pullback_1": 0.01,
        "pullback_2": 0.02,
        "pullback_3": 0.03,
    }.get(rule)

    for t, price in events[1:]:
        price = float(price)
        running_high = max(running_high, price)
        drawdown = (
            1.0 - price / running_high
            if running_high > 0
            else 0.0
        )
        if drawdown_required is not None:
            if drawdown >= drawdown_required:
                return int(t), price
            continue

        if rule == "rebound_2_1":
            if not activated and drawdown >= 0.02:
                activated = True
                pullback_low = price
                continue
            if activated:
                pullback_low = min(pullback_low, price)
                if (
                    pullback_low > 0
                    and price >= pullback_low * 1.01
                ):
                    return int(t), price
            continue

        raise ValueError(f"unknown Request253 entry rule: {rule}")
    return None


def fixed_value_after_entry(
    episode: list[tuple[int, float]],
    hot_t: int,
    entry: tuple[int, float] | None,
) -> float:
    if entry is None:
        return np.nan
    entry_t, entry_open = entry
    risk_cap_t = hot_t + RISK_CAP_MINUTES * MINUTE_MS
    future = [
        (t, price)
        for t, price in episode
        if entry_t < t <= risk_cap_t
    ]
    if len(future) < max(EVENT_HORIZONS):
        return np.nan
    values = []
    for horizon in EVENT_HORIZONS:
        exit_t, exit_open = future[horizon - 1]
        if exit_t > risk_cap_t:
            return np.nan
        value = _base_return(entry_open, exit_open)
        if not np.isfinite(value):
            return np.nan
        values.append(float(value))
    return float(np.mean(values))


def attach_rule_values(
    calibration: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    result = calibration.copy()
    lookup = event_lookup(scan)
    rules = ("immediate", *RULE_ORDER)
    for rule in rules:
        result[f"{rule}_value_pct"] = np.nan
        result[f"{rule}_entry_elapsed_minutes"] = np.nan

    for index, row in result.iterrows():
        day = str(row.trading_day)
        ticker = str(row.ticker).upper()
        hot_t = int(row.t)
        episode = lookup.get((day, ticker), [])
        limit_t = hot_t + ENTRY_WINDOW_MINUTES * MINUTE_MS
        watch_events = [
            (t, price)
            for t, price in episode
            if hot_t < t <= limit_t
        ]
        for rule in rules:
            entry = pick_entry(watch_events, rule)
            value = fixed_value_after_entry(
                episode,
                hot_t,
                entry,
            )
            result.at[index, f"{rule}_value_pct"] = value
            if entry is not None:
                result.at[
                    index,
                    f"{rule}_entry_elapsed_minutes",
                ] = float((entry[0] - hot_t) / MINUTE_MS)
    return result


def value_metrics(
    selected: pd.DataFrame,
    value_column: str,
) -> dict:
    target = pd.to_numeric(
        selected[value_column],
        errors="coerce",
    )
    valid = target.notna()
    resolved = selected.loc[valid].copy()
    target = pd.to_numeric(
        resolved[value_column],
        errors="coerce",
    )
    day_means = []
    positive_days = 0
    for day in EXTENDED_CAL_DAYS:
        day_target = pd.to_numeric(
            resolved.loc[
                resolved.trading_day.astype(str).eq(str(day)),
                value_column,
            ],
            errors="coerce",
        )
        if len(day_target):
            mean = float(day_target.mean())
            day_means.append(mean)
            positive_days += int(mean > 0)

    return {
        "candidate_rows": int(len(selected)),
        "resolved_rows": int(len(resolved)),
        "entry_resolution_rate": (
            float(len(resolved) / len(selected))
            if len(selected)
            else 0.0
        ),
        "mean_pct": float(target.mean()) if len(target) else None,
        "day_balanced_mean_pct": (
            float(np.mean(day_means)) if day_means else None
        ),
        "positive_rate": (
            float(target.gt(0).mean()) if len(target) else None
        ),
        "severe_loss_rate_le_minus2": (
            float(target.le(-2).mean()) if len(target) else None
        ),
        "positive_mean_days": positive_days,
    }


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, scan)
    model, columns = fit_candidate_model(first_hot)

    calibration = first_hot.loc[
        first_hot.trading_day.astype(str).isin(EXTENDED_CAL_DAYS)
    ].copy()
    target = pd.to_numeric(
        calibration.fixed_first_watch_value_pct,
        errors="coerce",
    )
    calibration = calibration.loc[target.notna()].copy()
    calibration["candidate_probability"] = model.predict_proba(
        xframe(calibration, columns)
    )[:, 1]
    calibration = attach_rule_values(calibration, scan)

    combinations = []
    passing = []
    thresholds = {}
    for fraction in CANDIDATE_FRACTIONS:
        threshold = float(
            np.quantile(
                calibration.candidate_probability.to_numpy(float),
                1.0 - fraction,
            )
        )
        thresholds[str(fraction)] = threshold
        selected = calibration.loc[
            calibration.candidate_probability.ge(threshold)
        ].copy()
        immediate = value_metrics(
            selected,
            "immediate_value_pct",
        )
        for rule in RULE_ORDER:
            candidate = value_metrics(
                selected,
                f"{rule}_value_pct",
            )
            gain = (
                candidate["mean_pct"] - immediate["mean_pct"]
                if candidate["mean_pct"] is not None
                and immediate["mean_pct"] is not None
                else None
            )
            checks = {
                "min_resolved_rows": (
                    candidate["resolved_rows"] >= MIN_RESOLVED_ROWS
                ),
                "entry_resolution_rate": (
                    candidate["entry_resolution_rate"]
                    >= MIN_ENTRY_COVERAGE
                ),
                "mean_positive": (
                    candidate["mean_pct"] is not None
                    and candidate["mean_pct"] > 0
                ),
                "day_balanced_mean_positive": (
                    candidate["day_balanced_mean_pct"] is not None
                    and candidate["day_balanced_mean_pct"] > 0
                ),
                "positive_rate": (
                    candidate["positive_rate"] is not None
                    and candidate["positive_rate"]
                    >= MIN_POSITIVE_RATE
                ),
                "positive_mean_days": (
                    candidate["positive_mean_days"]
                    >= MIN_POSITIVE_MEAN_DAYS
                ),
                "severe_loss_rate": (
                    candidate["severe_loss_rate_le_minus2"] is not None
                    and candidate["severe_loss_rate_le_minus2"]
                    <= MAX_SEVERE_RATE
                ),
                "gain_vs_immediate": (
                    gain is not None
                    and gain >= MIN_GAIN_VS_IMMEDIATE
                ),
            }
            row = {
                "candidate_fraction": fraction,
                "candidate_threshold": threshold,
                "rule": rule,
                "immediate": immediate,
                "candidate": candidate,
                "mean_gain_vs_immediate_pct": gain,
                "checks": checks,
                "passes": bool(all(checks.values())),
            }
            combinations.append(row)
            if row["passes"]:
                passing.append(row)

    chosen = None
    if passing:
        rule_rank = {
            rule: index
            for index, rule in enumerate(RULE_ORDER)
        }
        passing.sort(
            key=lambda row: (
                -float(row["candidate_fraction"]),
                rule_rank[row["rule"]],
            )
        )
        chosen = passing[0]

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "calibration_rows": int(len(calibration)),
        "candidate_fractions": list(CANDIDATE_FRACTIONS),
        "candidate_thresholds": thresholds,
        "entry_window_minutes": ENTRY_WINDOW_MINUTES,
        "risk_cap_minutes": RISK_CAP_MINUTES,
        "event_horizons": list(EVENT_HORIZONS),
        "combinations": combinations,
        "chosen_rule": (
            {
                "candidate_fraction": chosen["candidate_fraction"],
                "candidate_threshold": chosen["candidate_threshold"],
                "rule": chosen["rule"],
                "candidate": chosen["candidate"],
                "immediate": chosen["immediate"],
                "mean_gain_vs_immediate_pct": (
                    chosen["mean_gain_vs_immediate_pct"]
                ),
            }
            if chosen is not None
            else None
        ),
        "calibration_gate_pass": chosen is not None,
        "next_boundary": (
            "freeze rule and open June15-19"
            if chosen is not None
            else "learn richer causal entry state or redesign candidate state"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
