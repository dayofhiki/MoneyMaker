"""Request236: calibration-only tail economics diagnostic.

Requests232-235 show strong relative tradability ranking but no stable positive
admission boundary. Before changing the target or representation, Request236
asks a narrower question on already-opened calibration dates only:

Does any extreme top tail of the frozen Request232 scores have positive economic
value, or is the negative mean caused by a small number of severe losses?

No fresh June outcomes are opened. This is a diagnostic, not a policy search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .extended_history_economic_opportunity import EXTENDED_CAL_DAYS

REQUEST_ID = 236
SOURCE_RUN = 36251239308
TOP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25)


def _metrics(frame: pd.DataFrame) -> dict[str, object]:
    y = pd.to_numeric(
        frame["persistent_tradability_pct"],
        errors="coerce",
    ).dropna()
    if y.empty:
        return {
            "rows": 0,
            "mean_pct": None,
            "day_balanced_mean_pct": None,
            "median_pct": None,
            "positive_rate": None,
            "severe_loss_rate_lt_minus2": None,
            "severe_loss_rate_lt_minus5": None,
            "mean_positive_pct": None,
            "mean_negative_pct": None,
            "p10_pct": None,
            "p90_pct": None,
            "positive_mean_days": 0,
        }

    day_means = (
        frame.assign(
            _target=pd.to_numeric(
                frame["persistent_tradability_pct"],
                errors="coerce",
            )
        )
        .dropna(subset=["_target"])
        .groupby(frame["trading_day"].astype(str))["_target"]
        .mean()
    )
    positive = y.loc[y.gt(0)]
    negative = y.loc[y.le(0)]
    return {
        "rows": int(len(y)),
        "mean_pct": float(y.mean()),
        "day_balanced_mean_pct": float(day_means.mean()),
        "median_pct": float(y.median()),
        "positive_rate": float(y.gt(0).mean()),
        "severe_loss_rate_lt_minus2": float(y.lt(-2).mean()),
        "severe_loss_rate_lt_minus5": float(y.lt(-5).mean()),
        "mean_positive_pct": (
            float(positive.mean()) if len(positive) else None
        ),
        "mean_negative_pct": (
            float(negative.mean()) if len(negative) else None
        ),
        "p10_pct": float(y.quantile(0.10)),
        "p90_pct": float(y.quantile(0.90)),
        "positive_mean_days": int(day_means.gt(0).sum()),
    }


def _tail_table(
    frame: pd.DataFrame,
    score_column: str,
) -> dict[str, object]:
    work = frame.copy()
    score = pd.to_numeric(work[score_column], errors="coerce")
    target = pd.to_numeric(
        work["persistent_tradability_pct"],
        errors="coerce",
    )
    work = work.loc[score.notna() & target.notna()].copy()
    work["_score"] = pd.to_numeric(
        work[score_column],
        errors="coerce",
    )
    work = work.sort_values(
        ["_score", "trading_day", "ticker"],
        ascending=[False, True, True],
        kind="stable",
    )

    curves: dict[str, object] = {}
    for fraction in TOP_FRACTIONS:
        n = max(1, int(np.ceil(len(work) * fraction)))
        selected = work.head(n).copy()
        curves[f"{int(fraction * 100)}pct"] = {
            "fraction": fraction,
            "score_floor": float(selected["_score"].min()),
            **_metrics(selected),
        }
    return {
        "score_column": score_column,
        "population": _metrics(work),
        "top_tail": curves,
    }


def _joint_rank(frame: pd.DataFrame) -> pd.Series:
    value = pd.to_numeric(
        frame["predicted_persistent_tradability_pct"],
        errors="coerce",
    )
    probability = pd.to_numeric(
        frame["tradability_positive_probability"],
        errors="coerce",
    )
    value_rank = value.rank(pct=True, method="average")
    prob_rank = probability.rank(pct=True, method="average")
    return (value_rank + prob_rank) / 2.0


def evaluate(
    scored_hot_path: Path,
    output_path: Path,
) -> int:
    frame = pd.read_parquet(scored_hot_path)
    calibration = frame.loc[
        frame["trading_day"].astype(str).isin(EXTENDED_CAL_DAYS)
    ].copy()
    calibration["joint_tradability_rank"] = _joint_rank(calibration)

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "source_run": SOURCE_RUN,
        "opens_new_dates": False,
        "fresh_outcomes_opened": False,
        "diagnostic_only": True,
        "question": (
            "Does the frozen Request232 score contain any positive "
            "economic top-tail pocket, and are losses dominated by "
            "rare severe downside?"
        ),
        "calibration_days": list(EXTENDED_CAL_DAYS),
        "population": _metrics(calibration),
        "regression_score": _tail_table(
            calibration,
            "predicted_persistent_tradability_pct",
        ),
        "positive_probability": _tail_table(
            calibration,
            "tradability_positive_probability",
        ),
        "joint_rank": _tail_table(
            calibration,
            "joint_tradability_rank",
        ),
        "interpretation_rule": {
            "positive_tail_evidence": (
                "At least one predeclared top tail has positive overall "
                "mean, positive day-balanced mean, and >=5 positive-mean "
                "calibration days."
            ),
            "downside_asymmetry_evidence": (
                "Top-tail positive rate materially improves while mean "
                "stays negative and severe-loss frequency or mean "
                "negative outcome remains large."
            ),
        },
        "next_if_positive_tail": (
            "Freeze the strongest predeclared tail family and test a "
            "risk-controlled entry/exit policy without threshold search."
        ),
        "next_if_downside_asymmetry": (
            "Separate upside opportunity from downside survivability; "
            "learn an entry-survivability/risk head instead of another "
            "global admission threshold."
        ),
        "next_if_no_tail_signal": (
            "Request232 ranking is statistically real but economically "
            "misaligned; redesign the tradability target/representation."
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
