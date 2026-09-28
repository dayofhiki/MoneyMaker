"""Request270: policy-consistent causal continuation-horizon diagnostic."""

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

from .causal_hold_exit_signal import build_position_states, path_columns
from .causal_minute_controller_rebuild import causal_execution_scan
from .full_hot_fixed_policy_value import attach_fixed_value
from .learned_pullback_entry import CONTROLLER_TRAIN_DAYS, TEST_DAYS, _weights, controller_x
from .rich_post_hot_state import score_and_select_candidates

REQUEST_ID = 270
MODEL_FIT_DAYS = ("2026-05-05", "2026-05-06", "2026-05-07")
CALIBRATION_DAY = "2026-05-08"
HORIZONS = (1, 2, 3, 5)
MIN_CAL_AUC = 0.55
MIN_CAL_SPEARMAN = 0.03
MIN_TEST_ROWS = 500
MIN_TEST_AUC = 0.58
MIN_TEST_SPEARMAN = 0.05
MIN_TEST_SPREAD = 0.25


def attach_continuation_targets(states: pd.DataFrame) -> pd.DataFrame:
    result = states.copy()
    for horizon in HORIZONS:
        result[f"continuation_{horizon}_advantage_pct"] = np.nan

    for _, group in result.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        ordered = group.sort_values("state_t")
        if ordered.empty:
            continue
        values = pd.to_numeric(
            ordered.exit_now_pct,
            errors="coerce",
        ).to_numpy(float)
        indices = ordered.index.to_list()
        last = len(values) - 1
        for position, index in enumerate(indices):
            if not np.isfinite(values[position]):
                continue
            for horizon in HORIZONS:
                future_position = min(position + horizon, last)
                future_value = values[future_position]
                if np.isfinite(future_value):
                    result.at[
                        index,
                        f"continuation_{horizon}_advantage_pct",
                    ] = float(future_value - values[position])
    return result


def fit_horizon_model(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    horizon: int,
):
    target_column = f"continuation_{horizon}_advantage_pct"
    target = pd.to_numeric(frame[target_column], errors="coerce")
    fit = frame.loc[target.notna()].copy()
    y = pd.to_numeric(fit[target_column], errors="coerce").to_numpy(float)
    label = y > 0
    if len(fit) < 500 or len(np.unique(label)) != 2:
        raise ValueError(
            f"Request270 horizon {horizon} lacks support/classes: {len(fit)}"
        )
    low, high = np.quantile(y, [0.005, 0.995])
    weights = _weights(fit)
    reg = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261420 + horizon,
    )
    cls = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261430 + horizon,
    )
    x = controller_x(fit, columns)
    reg.fit(x, np.clip(y, low, high), sample_weight=weights)
    cls.fit(x, label.astype(int), sample_weight=weights)
    return reg, cls, {
        "rows": int(len(fit)),
        "target_mean_pct": float(np.mean(y)),
        "positive_rate": float(np.mean(label)),
        "winsor": [float(low), float(high)],
    }


def signal_report(
    frame: pd.DataFrame,
    reg,
    cls,
    columns: tuple[str, ...],
    horizon: int,
) -> dict:
    target_column = f"continuation_{horizon}_advantage_pct"
    target = pd.to_numeric(frame[target_column], errors="coerce")
    work = frame.loc[target.notna()].copy()
    target = pd.to_numeric(work[target_column], errors="coerce")
    x = controller_x(work, columns)
    pred = reg.predict(x)
    prob = cls.predict_proba(x)[:, 1]
    y = target.gt(0).astype(int)
    auc = (
        float(roc_auc_score(y, prob))
        if y.nunique() == 2
        else None
    )
    spearman = pd.Series(pred).corr(
        pd.Series(target.to_numpy(float)),
        method="spearman",
    )
    scored = pd.DataFrame(
        {
            "target": target.to_numpy(float),
            "pred": pred,
        }
    )
    q25 = float(scored.pred.quantile(0.25))
    q75 = float(scored.pred.quantile(0.75))
    bottom = scored.loc[scored.pred.le(q25), "target"]
    top = scored.loc[scored.pred.ge(q75), "target"]
    spread = (
        float(top.mean() - bottom.mean())
        if len(top) and len(bottom)
        else None
    )
    return {
        "horizon_states": horizon,
        "rows": int(len(work)),
        "target_mean_pct": float(target.mean()),
        "target_positive_rate": float(y.mean()),
        "auc": auc,
        "spearman": None if pd.isna(spearman) else float(spearman),
        "top_quartile_mean_pct": float(top.mean()) if len(top) else None,
        "bottom_quartile_mean_pct": (
            float(bottom.mean()) if len(bottom) else None
        ),
        "top_bottom_quartile_spread_pct": spread,
    }


def choose_horizon(calibration_reports: list[dict]) -> int | None:
    passing = [
        report
        for report in calibration_reports
        if report["auc"] is not None
        and report["spearman"] is not None
        and report["auc"] >= MIN_CAL_AUC
        and report["spearman"] >= MIN_CAL_SPEARMAN
    ]
    if not passing:
        return None
    passing.sort(
        key=lambda report: (
            -float(report["auc"]),
            -float(report["spearman"]),
            int(report["horizon_states"]),
        )
    )
    return int(passing[0]["horizon_states"])


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    causal_scan = causal_execution_scan(scan)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, causal_scan)
    scored, train_threshold, test_threshold = score_and_select_candidates(
        first_hot
    )

    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(CONTROLLER_TRAIN_DAYS)
        & scored.candidate_probability.ge(train_threshold)
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_probability.ge(test_threshold)
    ].copy()

    train_states = attach_continuation_targets(
        build_position_states(train_selected, causal_scan)
    )
    test_states = attach_continuation_targets(
        build_position_states(test_selected, causal_scan)
    )
    fit_states = train_states.loc[
        train_states.trading_day.astype(str).isin(MODEL_FIT_DAYS)
    ].copy()
    calibration_states = train_states.loc[
        train_states.trading_day.astype(str).eq(CALIBRATION_DAY)
    ].copy()
    columns = path_columns(fit_states)

    models = {}
    fit_support = {}
    calibration_reports = []
    for horizon in HORIZONS:
        reg, cls, support = fit_horizon_model(
            fit_states,
            columns,
            horizon,
        )
        models[horizon] = (reg, cls)
        fit_support[str(horizon)] = support
        calibration_reports.append(
            signal_report(
                calibration_states,
                reg,
                cls,
                columns,
                horizon,
            )
        )

    chosen_horizon = choose_horizon(calibration_reports)
    test_reports = []
    for horizon in HORIZONS:
        reg, cls = models[horizon]
        test_reports.append(
            signal_report(
                test_states,
                reg,
                cls,
                columns,
                horizon,
            )
        )

    chosen_test = next(
        (
            report
            for report in test_reports
            if report["horizon_states"] == chosen_horizon
        ),
        None,
    )
    checks = {
        "calibration_horizon_selected": chosen_horizon is not None,
        "test_rows": (
            chosen_test is not None
            and chosen_test["rows"] >= MIN_TEST_ROWS
        ),
        "test_auc": (
            chosen_test is not None
            and chosen_test["auc"] is not None
            and chosen_test["auc"] >= MIN_TEST_AUC
        ),
        "test_spearman": (
            chosen_test is not None
            and chosen_test["spearman"] is not None
            and chosen_test["spearman"] >= MIN_TEST_SPEARMAN
        ),
        "test_quartile_spread": (
            chosen_test is not None
            and chosen_test["top_bottom_quartile_spread_pct"] is not None
            and chosen_test["top_bottom_quartile_spread_pct"]
            >= MIN_TEST_SPREAD
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "causal_reference_contract": True,
        "train_candidate_threshold_top20": train_threshold,
        "test_candidate_threshold_top5": test_threshold,
        "support": {
            "train_candidate_episodes": int(len(train_selected)),
            "test_candidate_episodes": int(len(test_selected)),
            "fit_position_states": int(len(fit_states)),
            "calibration_position_states": int(len(calibration_states)),
            "test_position_states": int(len(test_states)),
            "feature_count": len(columns),
        },
        "fit_support_by_horizon": fit_support,
        "calibration_reports": calibration_reports,
        "chosen_horizon_states": chosen_horizon,
        "test_reports": test_reports,
        "chosen_test_report": chosen_test,
        "checks": checks,
        "continuation_signal_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "Each target is policy-consistent with holding a fixed number of "
            "future observed states and then exiting, with terminal fallback. "
            "May8 selects the horizon; May11-20 does not influence selection."
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
