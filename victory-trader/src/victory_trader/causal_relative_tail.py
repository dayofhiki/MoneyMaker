"""Request238: causal relative extreme-tail admission.

Request237 showed that the calibration-derived absolute top-1% score threshold
does not transport to the already-opened June development block, despite the
Request232 ranking remaining strong there.

Request238 tests whether the useful signal is relative rather than absolute.
At each HOT timestamp, it compares the frozen Request232 regression score with
the 99th percentile of a rolling window of previously observed HOT scores only.
The rolling window length is frozen from the median number of calibration HOT
rows per day. Same-timestamp peers do not see each other's scores.

June 8-12 has already been opened by earlier requests, so this is explicitly a
development diagnostic, not a fresh validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .extended_history_economic_opportunity import EXTENDED_CAL_DAYS
from .fresh_validated_attention_economic_opportunity import FRESH_EVAL_DAYS

REQUEST_ID = 238
SOURCE_RUN = 36251239308
SCORE_COLUMN = "predicted_persistent_tradability_pct"
TAIL_QUANTILE = 0.99


def _metrics(frame: pd.DataFrame) -> dict[str, object]:
    target = pd.to_numeric(
        frame["persistent_tradability_pct"],
        errors="coerce",
    )
    work = frame.loc[target.notna()].copy()
    target = pd.to_numeric(
        work["persistent_tradability_pct"],
        errors="coerce",
    )
    by_day: dict[str, object] = {}
    day_means: list[float] = []
    for day in FRESH_EVAL_DAYS:
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
            float(np.mean(day_means)) if day_means else None
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
        "positive_mean_days": int(
            sum(
                1
                for value in day_means
                if value > 0
            )
        ),
        "days_with_selection": int(len(day_means)),
        "by_day": by_day,
    }


def _calibration_window(frame: pd.DataFrame) -> int:
    calibration = frame.loc[
        frame["trading_day"].astype(str).isin(EXTENDED_CAL_DAYS)
    ].copy()
    per_day = (
        calibration.groupby(
            calibration["trading_day"].astype(str)
        )[SCORE_COLUMN]
        .count()
        .astype(int)
    )
    if per_day.empty:
        raise ValueError("request238 calibration score counts are empty")
    return max(50, int(round(float(per_day.median()))))


def apply_relative_tail(
    frame: pd.DataFrame,
    window_rows: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["relative_tail_threshold"] = np.nan
    result["request238_selected"] = False

    calibration = result.loc[
        result["trading_day"].astype(str).isin(EXTENDED_CAL_DAYS)
    ].copy()
    calibration = calibration.sort_values(
        ["trading_day", "t", "ticker"],
        kind="stable",
    )
    history = [
        float(x)
        for x in pd.to_numeric(
            calibration[SCORE_COLUMN],
            errors="coerce",
        ).dropna().tail(window_rows)
    ]

    development = result.loc[
        result["trading_day"].astype(str).isin(FRESH_EVAL_DAYS)
    ].copy()
    development = development.sort_values(
        ["trading_day", "t", "ticker"],
        kind="stable",
    )

    for _, same_t in development.groupby(
        ["trading_day", "t"],
        sort=True,
    ):
        if len(history) >= 20:
            threshold = float(
                np.quantile(
                    np.asarray(history[-window_rows:], dtype=float),
                    TAIL_QUANTILE,
                )
            )
        else:
            threshold = np.nan

        scores = pd.to_numeric(
            same_t[SCORE_COLUMN],
            errors="coerce",
        )
        for idx, score in scores.items():
            result.at[idx, "relative_tail_threshold"] = threshold
            result.at[idx, "request238_selected"] = bool(
                np.isfinite(threshold)
                and np.isfinite(score)
                and float(score) >= threshold
            )

        new_scores = [
            float(x)
            for x in scores.dropna().to_numpy(dtype=float)
        ]
        history.extend(new_scores)
        if len(history) > window_rows:
            history = history[-window_rows:]

    return result


def evaluate(
    scored_hot_path: Path,
    output_path: Path,
) -> int:
    frame = pd.read_parquet(scored_hot_path)
    window_rows = _calibration_window(frame)
    scored = apply_relative_tail(frame, window_rows)

    development = scored.loc[
        scored["trading_day"].astype(str).isin(FRESH_EVAL_DAYS)
    ].copy()
    selected = development.loc[
        development["request238_selected"]
        .fillna(False)
        .astype(bool)
    ].copy()

    metrics = _metrics(selected)
    population = _metrics(development)
    gate = bool(
        metrics["rows"] >= 8
        and metrics["mean_pct"] is not None
        and float(metrics["mean_pct"]) > 0
        and metrics["day_balanced_mean_pct"] is not None
        and float(metrics["day_balanced_mean_pct"]) > 0
        and metrics["positive_rate"] is not None
        and float(metrics["positive_rate"]) >= 0.45
        and metrics["positive_mean_days"] >= 3
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "source_run": SOURCE_RUN,
        "opens_new_dates": False,
        "development_only": True,
        "promotion_eligible": False,
        "fresh_validation_claimed": False,
        "rule": {
            "score": SCORE_COLUMN,
            "tail_quantile": TAIL_QUANTILE,
            "window_rows": window_rows,
            "window_definition": (
                "median calibration HOT rows per day"
            ),
            "threshold_uses_only_prior_scores": True,
            "same_timestamp_peer_scores_excluded": True,
            "outcomes_used_for_threshold": False,
        },
        "development_population": population,
        "development_selected": metrics,
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze causal relative-tail admission and open a genuinely "
            "new holdout block before building entry/exit timing."
        ),
        "next_boundary_if_fail": (
            "relative score adaptation does not rescue economics; "
            "redesign the tradability target/representation before "
            "further admission or timing work."
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
