"""Request252: distributional expected economic value on fixed first-WATCH returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score

from .extended_history_economic_opportunity import (
    EXTENDED_CAL_DAYS,
    EXTENDED_FIT_DAYS,
)
from .fixed_policy_tail_economics import metrics
from .full_hot_fixed_policy_value import (
    attach_fixed_value,
    usable_columns,
    xframe,
)

REQUEST_ID = 252
FRACTIONS = (0.02, 0.05, 0.10, 0.15, 0.20)
MIN_WIN_AUC = 0.60
MIN_EV_SPEARMAN = 0.10
MAX_SELECTED_SEVERE_RATE = 0.20


def fit_heads(first_hot: pd.DataFrame):
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
    x = xframe(fit, columns)
    y_win = target.gt(0).astype(int)
    if y_win.nunique() != 2:
        raise ValueError("Request252 win head needs both classes")

    win_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=20261352,
    )
    win_model.fit(x, y_win)

    positive = target.gt(0)
    gain_fit = fit.loc[positive].copy()
    gain_target = pd.to_numeric(
        gain_fit.fixed_first_watch_value_pct,
        errors="coerce",
    ).to_numpy(float)
    if len(gain_fit) < 150:
        raise ValueError(
            f"Request252 gain head has only {len(gain_fit)} rows"
        )
    gain_low, gain_high = np.quantile(gain_target, [0.01, 0.99])
    gain_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261353,
    )
    gain_model.fit(
        xframe(gain_fit, columns),
        np.clip(gain_target, gain_low, gain_high),
    )

    loss_fit = fit.loc[~positive].copy()
    loss_target = -pd.to_numeric(
        loss_fit.fixed_first_watch_value_pct,
        errors="coerce",
    ).to_numpy(float)
    if len(loss_fit) < 500:
        raise ValueError(
            f"Request252 loss head has only {len(loss_fit)} rows"
        )
    loss_low, loss_high = np.quantile(loss_target, [0.005, 0.995])
    loss_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261354,
    )
    loss_model.fit(
        xframe(loss_fit, columns),
        np.clip(loss_target, loss_low, loss_high),
    )
    support = {
        "fit_rows": int(len(fit)),
        "fit_positive_rows": int(positive.sum()),
        "fit_nonpositive_rows": int((~positive).sum()),
        "fit_positive_rate": float(positive.mean()),
        "gain_winsor": [float(gain_low), float(gain_high)],
        "loss_winsor": [float(loss_low), float(loss_high)],
    }
    return win_model, gain_model, loss_model, columns, support


def score_calibration(first_hot: pd.DataFrame):
    win_model, gain_model, loss_model, columns, support = fit_heads(
        first_hot
    )
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
    x = xframe(cal, columns)
    win_probability = win_model.predict_proba(x)[:, 1]
    gain = np.maximum(gain_model.predict(x), 0.0)
    loss = np.maximum(loss_model.predict(x), 0.0)
    expected_value = win_probability * gain - (
        1.0 - win_probability
    ) * loss

    cal["win_probability"] = win_probability
    cal["predicted_gain_pct"] = gain
    cal["predicted_loss_pct"] = loss
    cal["distributional_expected_value"] = expected_value

    actual_win = target.gt(0).astype(int)
    win_auc = (
        float(roc_auc_score(actual_win, win_probability))
        if actual_win.nunique() == 2
        else 0.5
    )
    ev_spearman = float(
        pd.Series(expected_value).corr(
            pd.Series(target.to_numpy(float)),
            method="spearman",
        )
    )
    return cal, support, win_auc, ev_spearman


def choose_rule(
    calibration: pd.DataFrame,
    win_auc: float,
    ev_spearman: float,
):
    results = []
    passing = []
    for fraction in FRACTIONS:
        row, _ = metrics(
            calibration,
            "distributional_expected_value",
            fraction,
        )
        row["model_checks"] = {
            "win_auc": win_auc >= MIN_WIN_AUC,
            "expected_value_spearman": (
                np.isfinite(ev_spearman)
                and ev_spearman >= MIN_EV_SPEARMAN
            ),
            "severe_loss_rate": (
                row["selected_severe_loss_rate_le_minus2"] is not None
                and row["selected_severe_loss_rate_le_minus2"]
                <= MAX_SELECTED_SEVERE_RATE
            ),
        }
        row["passes_252"] = bool(
            row["passes"] and all(row["model_checks"].values())
        )
        results.append(row)
        if row["passes_252"]:
            passing.append(row)
    if not passing:
        return None, results
    passing.sort(key=lambda r: -float(r["fraction"]))
    return passing[0], results


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
    calibration, support, win_auc, ev_spearman = score_calibration(
        first_hot
    )
    chosen, results = choose_rule(
        calibration,
        win_auc,
        ev_spearman,
    )

    target = pd.to_numeric(
        calibration.fixed_first_watch_value_pct,
        errors="coerce",
    )
    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "support": support,
        "calibration_rows": int(len(calibration)),
        "calibration_population_mean_pct": float(target.mean()),
        "calibration_population_positive_rate": float(target.gt(0).mean()),
        "win_auc": win_auc,
        "expected_value_spearman": ev_spearman,
        "fractions": list(FRACTIONS),
        "results": results,
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
            "freeze model and open June15-19"
            if chosen is not None
            else "redesign upstream first-HOT state representation"
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
