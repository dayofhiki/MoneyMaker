"""Request272: cross-fitted fitted-policy HOLD/EXIT improvement."""

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

from .causal_continuation_horizon import attach_continuation_targets
from .causal_hold_exit_controller import (
    _base_record,
    policy_metrics,
    static_policy_rows,
)
from .causal_hold_exit_signal import build_position_states, path_columns
from .causal_minute_controller_rebuild import causal_execution_scan
from .full_hot_fixed_policy_value import attach_fixed_value
from .learned_pullback_entry import CONTROLLER_TRAIN_DAYS, TEST_DAYS, _weights, controller_x
from .rich_post_hot_state import score_and_select_candidates

REQUEST_ID = 272
TRAIN_DAYS = tuple(CONTROLLER_TRAIN_DAYS)
HOLD_PROBABILITY_THRESHOLD = 0.50
OOF_TARGET_COLUMN = "policy_consistent_hold_advantage_pct"
MIN_OOF_ROWS = 3000
MIN_POSITIVE_RATE = 0.40
MAX_SEVERE_RATE = 0.15
MIN_POSITIVE_DAYS = 5
MIN_GAIN_VS_BASE = 0.10
MIN_GAIN_VS_STATIC = 0.10


def fit_advantage_model(
    frame: pd.DataFrame,
    target_column: str,
    columns: tuple[str, ...],
    seed: int,
):
    target = pd.to_numeric(frame[target_column], errors="coerce")
    fit = frame.loc[target.notna()].copy()
    y = pd.to_numeric(fit[target_column], errors="coerce").to_numpy(float)
    label = y > 0
    if len(fit) < 500 or len(np.unique(label)) != 2:
        raise ValueError(
            f"Request272 target {target_column} lacks support/classes: {len(fit)}"
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
        random_state=seed,
    )
    cls = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed + 1,
    )
    x = controller_x(fit, columns)
    reg.fit(x, np.clip(y, low, high), sample_weight=weights)
    cls.fit(x, label.astype(int), sample_weight=weights)
    return reg, cls, {
        "rows": int(len(fit)),
        "mean_pct": float(np.mean(y)),
        "positive_rate": float(np.mean(label)),
        "winsor": [float(low), float(high)],
    }


def recurrent_policy_rows(
    states: pd.DataFrame,
    reg,
    cls,
    columns: tuple[str, ...],
    *,
    policy_name: str,
) -> pd.DataFrame:
    work = states.copy()
    x = controller_x(work, columns)
    work["predicted_advantage_pct"] = reg.predict(x)
    work["hold_probability"] = cls.predict_proba(x)[:, 1]
    rows = []
    for _, group in work.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        ordered = group.sort_values("state_t").reset_index(drop=True)
        if ordered.empty:
            continue
        chosen = ordered.iloc[-1]
        reason = "terminal_cap"
        for position, (_, state) in enumerate(ordered.iterrows()):
            if position == len(ordered) - 1:
                chosen = state
                reason = "terminal_cap"
                break
            hold = (
                float(state.predicted_advantage_pct) > 0
                and float(state.hold_probability)
                >= HOLD_PROBABILITY_THRESHOLD
            )
            if not hold:
                chosen = state
                reason = "model_exit"
                break
        record = _base_record(ordered, chosen, policy_name)
        record["exit_reason"] = reason
        rows.append(record)
    return pd.DataFrame(rows)


def attach_policy_values_from_base(
    heldout: pd.DataFrame,
    reg,
    cls,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    result = heldout.copy()
    result[OOF_TARGET_COLUMN] = np.nan
    work = heldout.copy()
    x = controller_x(work, columns)
    work["_predicted_advantage"] = reg.predict(x)
    work["_hold_probability"] = cls.predict_proba(x)[:, 1]

    for _, group in work.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        ordered = group.sort_values("state_t")
        if ordered.empty:
            continue
        indices = ordered.index.to_list()
        exit_values = pd.to_numeric(
            ordered.exit_now_pct,
            errors="coerce",
        ).to_numpy(float)
        pred = pd.to_numeric(
            ordered._predicted_advantage,
            errors="coerce",
        ).to_numpy(float)
        prob = pd.to_numeric(
            ordered._hold_probability,
            errors="coerce",
        ).to_numpy(float)
        n = len(ordered)

        downstream_values = np.full(n, np.nan, dtype=float)
        for start in range(n):
            exit_position = n - 1
            for position in range(start, n):
                if position == n - 1:
                    exit_position = position
                    break
                hold = (
                    np.isfinite(pred[position])
                    and pred[position] > 0
                    and np.isfinite(prob[position])
                    and prob[position] >= HOLD_PROBABILITY_THRESHOLD
                )
                if not hold:
                    exit_position = position
                    break
            downstream_values[start] = exit_values[exit_position]

        for position, index in enumerate(indices):
            if not np.isfinite(exit_values[position]):
                continue
            if position == n - 1:
                target = 0.0
            else:
                next_policy_value = downstream_values[position + 1]
                target = (
                    float(next_policy_value - exit_values[position])
                    if np.isfinite(next_policy_value)
                    else np.nan
                )
            result.at[index, OOF_TARGET_COLUMN] = target
    return result


def build_cross_fitted_targets(
    train_states: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[pd.DataFrame, list[dict]]:
    pieces = []
    audits = []
    for fold_index, heldout_day in enumerate(TRAIN_DAYS):
        fit = train_states.loc[
            ~train_states.trading_day.astype(str).eq(str(heldout_day))
        ].copy()
        heldout = train_states.loc[
            train_states.trading_day.astype(str).eq(str(heldout_day))
        ].copy()
        reg, cls, support = fit_advantage_model(
            fit,
            "continuation_1_advantage_pct",
            columns,
            20261440 + fold_index * 10,
        )
        labeled = attach_policy_values_from_base(
            heldout,
            reg,
            cls,
            columns,
        )
        pieces.append(labeled)
        target = pd.to_numeric(
            labeled[OOF_TARGET_COLUMN],
            errors="coerce",
        )
        audits.append(
            {
                "heldout_day": str(heldout_day),
                "base_fit_support": support,
                "heldout_rows": int(len(labeled)),
                "oof_target_rows": int(target.notna().sum()),
                "oof_target_mean_pct": (
                    float(target.mean()) if target.notna().any() else None
                ),
                "oof_target_positive_rate": (
                    float(target.gt(0).mean()) if target.notna().any() else None
                ),
            }
        )
    return pd.concat(pieces, ignore_index=True), audits


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
    rows_output: Path,
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
        scored.trading_day.astype(str).isin(TRAIN_DAYS)
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
    columns = path_columns(train_states)

    base_reg, base_cls, base_support = fit_advantage_model(
        train_states,
        "continuation_1_advantage_pct",
        columns,
        20261430,
    )
    oof_states, fold_audit = build_cross_fitted_targets(
        train_states,
        columns,
    )
    improved_reg, improved_cls, improved_support = fit_advantage_model(
        oof_states,
        OOF_TARGET_COLUMN,
        columns,
        20261490,
    )

    base_rows = recurrent_policy_rows(
        test_states,
        base_reg,
        base_cls,
        columns,
        policy_name="base_one_step",
    )
    improved_rows = recurrent_policy_rows(
        test_states,
        improved_reg,
        improved_cls,
        columns,
        policy_name="policy_iteration_1",
    )
    static_rows = {
        mode: static_policy_rows(test_states, mode)
        for mode in ("first", "fifth", "terminal")
    }

    base_metrics = policy_metrics(base_rows)
    improved_metrics = policy_metrics(improved_rows)
    static_metrics = {
        mode: policy_metrics(rows)
        for mode, rows in static_rows.items()
    }
    best_static_mean = max(
        report["mean_pct"]
        for report in static_metrics.values()
        if report["mean_pct"] is not None
    )
    gain_vs_base = (
        improved_metrics["mean_pct"] - base_metrics["mean_pct"]
    )
    gain_vs_static = (
        improved_metrics["mean_pct"] - best_static_mean
    )
    oof_target = pd.to_numeric(
        oof_states[OOF_TARGET_COLUMN],
        errors="coerce",
    )

    checks = {
        "oof_target_rows": int(oof_target.notna().sum()) >= MIN_OOF_ROWS,
        "improved_mean_positive": improved_metrics["mean_pct"] > 0,
        "day_balanced_mean_positive": (
            improved_metrics["day_balanced_mean_pct"] > 0
        ),
        "positive_rate": (
            improved_metrics["positive_rate"] >= MIN_POSITIVE_RATE
        ),
        "severe_loss_rate": (
            improved_metrics["severe_loss_rate_le_minus2"]
            <= MAX_SEVERE_RATE
        ),
        "positive_mean_days": (
            improved_metrics["positive_mean_days"] >= MIN_POSITIVE_DAYS
        ),
        "gain_vs_base": gain_vs_base >= MIN_GAIN_VS_BASE,
        "gain_vs_best_static": gain_vs_static >= MIN_GAIN_VS_STATIC,
    }

    all_rows = [base_rows, improved_rows]
    for mode, rows in static_rows.items():
        tagged = rows.copy()
        tagged["policy"] = mode
        all_rows.append(tagged)

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
            "train_position_states": int(len(train_states)),
            "test_position_states": int(len(test_states)),
            "feature_count": len(columns),
            "base_target_support": base_support,
            "oof_target_rows": int(oof_target.notna().sum()),
            "oof_target_mean_pct": float(oof_target.mean()),
            "oof_target_positive_rate": float(oof_target.gt(0).mean()),
            "improved_target_support": improved_support,
        },
        "cross_fit_folds": fold_audit,
        "policies": {
            "base_one_step": base_metrics,
            "policy_iteration_1": improved_metrics,
            "static_first_exit": static_metrics["first"],
            "static_fifth_exit": static_metrics["fifth"],
            "static_terminal_exit": static_metrics["terminal"],
        },
        "best_static_mean_pct": best_static_mean,
        "mean_gain_vs_base_one_step_pct": gain_vs_base,
        "mean_gain_vs_best_static_pct": gain_vs_static,
        "checks": checks,
        "economic_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "Policy-consistent HOLD labels are generated out-of-fold by "
            "holding one state and then following the frozen base recurrent "
            "policy. May11-20 is never used for target generation or fitting."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.concat(all_rows, ignore_index=True).to_parquet(
        rows_output,
        index=False,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
