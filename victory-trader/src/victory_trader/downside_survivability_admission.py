"""Request251: causal downside-survivability admission on Request249 target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .extended_history_economic_opportunity import (
    EXTENDED_CAL_DAYS,
    EXTENDED_FIT_DAYS,
)
from .fixed_policy_tail_economics import metrics
from .full_hot_fixed_policy_value import (
    attach_fixed_value,
    train,
    xframe,
)

REQUEST_ID = 251
SEVERE_LOSS_THRESHOLD = -2.0
MIN_RISK_AUC = 0.58
MAX_SELECTED_SEVERE_RATE = 0.15
FRACTIONS = (0.02, 0.05, 0.10, 0.15, 0.20)


def fit_risk_model(
    first_hot: pd.DataFrame,
    columns: tuple[str, ...],
) -> HistGradientBoostingClassifier:
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
    y = target.le(SEVERE_LOSS_THRESHOLD).astype(int)
    if y.nunique() != 2:
        raise ValueError("Request251 risk fit requires both classes")
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=20261351,
    )
    model.fit(xframe(fit, columns), y)
    return model


def score_calibration(first_hot: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    opportunity = train(first_hot)
    risk_model = fit_risk_model(first_hot, opportunity.columns)

    cal = first_hot.loc[
        first_hot.trading_day.astype(str).isin(EXTENDED_CAL_DAYS)
    ].copy()
    target = pd.to_numeric(
        cal.fixed_first_watch_value_pct,
        errors="coerce",
    )
    cal = cal.loc[target.notna()].copy()
    target = pd.to_numeric(
        cal.fixed_first_watch_value_pct,
        errors="coerce",
    )
    x = xframe(cal, opportunity.columns)

    cal["predicted_value"] = (
        opportunity.regressor.predict(x) + opportunity.offset
    )
    cal["positive_probability"] = (
        opportunity.classifier.predict_proba(x)[:, 1]
    )
    cal["severe_loss_probability"] = (
        risk_model.predict_proba(x)[:, 1]
    )
    severe = target.le(SEVERE_LOSS_THRESHOLD).astype(int)
    risk_auc = (
        float(roc_auc_score(severe, cal.severe_loss_probability))
        if severe.nunique() == 2
        else 0.5
    )

    cal["predicted_value_percentile"] = cal.predicted_value.rank(
        pct=True,
        method="average",
    )
    cal["positive_probability_percentile"] = (
        cal.positive_probability.rank(pct=True, method="average")
    )
    cal["survivability_percentile"] = (
        (1.0 - cal.severe_loss_probability).rank(
            pct=True,
            method="average",
        )
    )
    cal["joint_percentile_baseline"] = (
        cal.predicted_value_percentile
        + cal.positive_probability_percentile
    ) / 2.0
    cal["risk_adjusted_opportunity"] = (
        cal.predicted_value_percentile
        + cal.positive_probability_percentile
        + cal.survivability_percentile
    ) / 3.0
    return cal, risk_auc


def choose_rule(
    calibration: pd.DataFrame,
    risk_auc: float,
) -> tuple[dict | None, list[dict], list[dict]]:
    candidate_results = []
    baseline_results = []
    passing = []
    for fraction in FRACTIONS:
        candidate, _ = metrics(
            calibration,
            "risk_adjusted_opportunity",
            fraction,
        )
        baseline, _ = metrics(
            calibration,
            "joint_percentile_baseline",
            fraction,
        )
        candidate["risk_auc"] = risk_auc
        candidate["extra_checks"] = {
            "risk_auc": risk_auc >= MIN_RISK_AUC,
            "severe_loss_rate": (
                candidate["selected_severe_loss_rate_le_minus2"] is not None
                and candidate["selected_severe_loss_rate_le_minus2"]
                <= MAX_SELECTED_SEVERE_RATE
            ),
        }
        candidate["passes_251"] = bool(
            candidate["passes"]
            and all(candidate["extra_checks"].values())
        )
        candidate_results.append(candidate)
        baseline_results.append(baseline)
        if candidate["passes_251"]:
            passing.append(candidate)

    if not passing:
        return None, candidate_results, baseline_results
    passing.sort(key=lambda r: -float(r["fraction"]))
    return passing[0], candidate_results, baseline_results


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
    calibration, risk_auc = score_calibration(first_hot)
    chosen, candidate_results, baseline_results = choose_rule(
        calibration,
        risk_auc,
    )

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "first_hot_rows": int(len(first_hot)),
        "calibration_rows": int(len(calibration)),
        "severe_loss_threshold_pct": SEVERE_LOSS_THRESHOLD,
        "calibration_severe_loss_rate": float(
            pd.to_numeric(
                calibration.fixed_first_watch_value_pct,
                errors="coerce",
            ).le(SEVERE_LOSS_THRESHOLD).mean()
        ),
        "severe_loss_auc": risk_auc,
        "fractions": list(FRACTIONS),
        "risk_adjusted_results": candidate_results,
        "joint_baseline_results": baseline_results,
        "chosen_rule": (
            {
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
            else "redesign upstream first-HOT representation"
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
