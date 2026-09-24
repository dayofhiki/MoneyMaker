"""Request 180: honest patience override on recurrent EXIT baseline.

Development-only reuse of the already-opened Request-178 block. The short
one-minute controller remains the default. A compact override model may rescue
a baseline EXIT for exactly one minute using only short HOLD probability and
the independently estimated consensus remaining-opportunity score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .day_balanced_hurdle_expected_value import (
    train_day_balanced_classifier,
    train_day_balanced_magnitude,
)
from .direct_recurrent_hold_exit_advantage import (
    EPISODE_KEYS,
    build_trajectories,
    matched_difference,
    trajectory_metrics,
)
from .fresh_consensus_hurdle_value import (
    FRESH_DAYS,
    _score_day_ev,
    _score_row_ev,
)
from .market_regime_hurdle_expected_value import _train_magnitude_head
from .market_regime_position_value import (
    attach_market_regime,
    build_market_regime,
)
from .market_regime_positive_value import train_classifier
from .selected_hot_position_value_observability import (
    apply_excess_target,
    fit_minute_baselines,
    predict_hold,
    train_hold_model,
)

REQUEST_ID = 180
MODEL_SEED = 20261080
MIN_START_COVERAGE = 0.80
MIN_COMPLETION_COVERAGE = 0.90
MIN_HONEST_TARGET_ROWS = 500
MAX_HOLD_MINUTES = 30


@dataclass(frozen=True)
class OverrideModel:
    model: HistGradientBoostingRegressor
    winsor_low: float
    winsor_high: float
    training_rows: int


def split_calibration_days(
    calibration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    days = sorted(calibration["trading_day"].astype(str).unique())
    midpoint = len(days) // 2
    first = days[:midpoint]
    second = days[midpoint:]
    if len(first) < 4 or len(second) < 4:
        raise ValueError(
            f"request 180 needs >=4 calibration days per half, got "
            f"{len(first)} and {len(second)}"
        )
    a = calibration.loc[
        calibration["trading_day"].astype(str).isin(first)
    ].copy()
    b = calibration.loc[
        calibration["trading_day"].astype(str).isin(second)
    ].copy()
    return a, b, first, second


def _train_consensus_heads(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
):
    row_classifier = train_classifier(fit, calibration)
    row_positive = _train_magnitude_head(
        fit,
        calibration,
        positive_class=True,
        seed=20261080,
    )
    row_nonpositive = _train_magnitude_head(
        fit,
        calibration,
        positive_class=False,
        seed=20261081,
    )
    day_classifier = train_day_balanced_classifier(
        fit,
        calibration,
    )
    day_positive = train_day_balanced_magnitude(
        fit,
        calibration,
        positive_class=True,
        seed=20261082,
    )
    day_nonpositive = train_day_balanced_magnitude(
        fit,
        calibration,
        positive_class=False,
        seed=20261083,
    )
    return (
        row_classifier,
        row_positive,
        row_nonpositive,
        day_classifier,
        day_positive,
        day_nonpositive,
    )


def _score_consensus(
    frame: pd.DataFrame,
    heads,
) -> np.ndarray:
    (
        row_classifier,
        row_positive,
        row_nonpositive,
        day_classifier,
        day_positive,
        day_nonpositive,
    ) = heads
    row = _score_row_ev(
        frame,
        row_classifier,
        row_positive,
        row_nonpositive,
    )
    day = _score_day_ev(
        frame,
        day_classifier,
        day_positive,
        day_nonpositive,
    )
    return np.minimum(row, day)


def _minute_map(group: pd.DataFrame) -> dict[int, pd.Series]:
    work = group.copy()
    work["_minute"] = pd.to_numeric(
        work["minutes_held"],
        errors="coerce",
    )
    work = work.loc[
        work["_minute"].notna()
        & work["_minute"].mod(1).eq(0)
    ].copy()
    work["_minute"] = work["_minute"].astype(int)
    return {
        int(row["_minute"]): row
        for _, row in work.sort_values("_minute", kind="stable").iterrows()
    }


def build_honest_patience_targets(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = scored.copy()
    frame["patience_override_advantage_pct"] = np.nan
    frame["teacher_realized_value_pct"] = np.nan

    target_values: dict[int, float] = {}
    teacher_values: dict[int, float] = {}

    for _, group in frame.groupby(EPISODE_KEYS, sort=False):
        by_minute = _minute_map(group)
        if not by_minute:
            continue
        local_teacher: dict[int, float] = {}

        for minute in range(MAX_HOLD_MINUTES - 1, 0, -1):
            row = by_minute.get(minute)
            if row is None:
                continue
            idx = int(row.name)
            current_exit = pd.to_numeric(
                pd.Series([row.get("exit_now_base_return_pct")]),
                errors="coerce",
            ).iloc[0]
            short = pd.to_numeric(
                pd.Series([row.get("short_hold_probability")]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(current_exit):
                continue

            next_value = np.nan
            if minute < MAX_HOLD_MINUTES - 1:
                next_row = by_minute.get(minute + 1)
                if next_row is not None:
                    next_idx = int(next_row.name)
                    if next_idx in local_teacher:
                        next_value = local_teacher[next_idx]
            if pd.isna(next_value):
                next_return = pd.to_numeric(
                    pd.Series([row.get("next_minute_base_return_pct")]),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(next_return):
                    next_value = float(next_return)

            teacher_hold = bool(
                pd.notna(short)
                and np.isfinite(float(short))
                and float(short) > 0.5
            )
            if teacher_hold and pd.notna(next_value):
                realized = float(next_value)
            else:
                realized = float(current_exit)

            local_teacher[idx] = realized
            teacher_values[idx] = realized

            if (
                not teacher_hold
                and pd.notna(next_value)
                and np.isfinite(float(next_value))
            ):
                target_values[idx] = (
                    float(next_value) - float(current_exit)
                )

    if teacher_values:
        teacher_series = pd.Series(teacher_values, dtype=float)
        frame.loc[
            teacher_series.index,
            "teacher_realized_value_pct",
        ] = teacher_series
    if target_values:
        target_series = pd.Series(target_values, dtype=float)
        frame.loc[
            target_series.index,
            "patience_override_advantage_pct",
        ] = target_series

    target = pd.to_numeric(
        frame["patience_override_advantage_pct"],
        errors="coerce",
    )
    valid = target.notna()
    return frame, {
        "rows": int(len(frame)),
        "honest_target_rows": int(valid.sum()),
        "honest_target_mean_pct": (
            float(target.loc[valid].mean()) if valid.any() else None
        ),
        "honest_target_positive_rate": (
            float(target.loc[valid].gt(0).mean())
            if valid.any()
            else None
        ),
    }


def _override_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        :,
        ["short_hold_probability", "consensus_hurdle_ev_pct"],
    ].apply(pd.to_numeric, errors="coerce")


def train_override_model(
    honest: pd.DataFrame,
) -> OverrideModel:
    target = pd.to_numeric(
        honest["patience_override_advantage_pct"],
        errors="coerce",
    )
    features = _override_features(honest)
    valid = target.notna() & features.notna().all(axis=1)
    train = honest.loc[valid].copy()
    if len(train) < MIN_HONEST_TARGET_ROWS:
        raise ValueError(
            f"request 180 needs >={MIN_HONEST_TARGET_ROWS} honest targets, "
            f"got {len(train)}"
        )

    y = pd.to_numeric(
        train["patience_override_advantage_pct"],
        errors="coerce",
    ).astype(float)
    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=100,
        l2_regularization=2.0,
        random_state=MODEL_SEED,
    )
    model.fit(
        _override_features(train),
        y.clip(lower=low, upper=high),
    )
    return OverrideModel(
        model=model,
        winsor_low=float(low),
        winsor_high=float(high),
        training_rows=int(len(train)),
    )


def predict_override(
    frame: pd.DataFrame,
    fitted: OverrideModel,
) -> np.ndarray:
    x = _override_features(frame)
    result = np.full(len(frame), np.nan, dtype=float)
    valid = x.notna().all(axis=1)
    if valid.any():
        result[valid.to_numpy()] = fitted.model.predict(x.loc[valid])
    return result


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    trajectories_output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    history_scan = pd.read_parquet(history_scan_path)

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError("request 180 requires complete request 178 shards")

    fresh = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    if sorted(fresh["trading_day"].astype(str).unique()) != list(FRESH_DAYS):
        raise ValueError("request 180 fresh days differ from request 178")

    historical_days = set(
        fit["trading_day"].astype(str)
    ) | set(calibration["trading_day"].astype(str))
    historical_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(historical_days)
    ].copy()
    historical_regime = build_market_regime(historical_scan)
    fresh_regime = build_market_regime(fresh_scan)

    fit = attach_market_regime(fit, historical_regime)
    calibration = attach_market_regime(
        calibration,
        historical_regime,
    )
    fresh = attach_market_regime(fresh, fresh_regime)

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(calibration, baselines)
    fresh = apply_excess_target(fresh, baselines)

    cal_a, cal_b, cal_a_days, cal_b_days = split_calibration_days(
        calibration
    )

    partial_short = train_hold_model(fit, cal_a)
    partial_heads = _train_consensus_heads(fit, cal_a)
    scored_b = cal_b.copy()
    scored_b["short_hold_probability"] = predict_hold(
        scored_b,
        partial_short,
    )
    scored_b["consensus_hurdle_ev_pct"] = _score_consensus(
        scored_b,
        partial_heads,
    )
    honest_b, honest_diag = build_honest_patience_targets(scored_b)
    override_model = train_override_model(honest_b)

    honest_valid = honest_b.loc[
        pd.to_numeric(
            honest_b["patience_override_advantage_pct"],
            errors="coerce",
        ).notna()
    ].copy()
    honest_valid["override_fit_prediction_pct"] = predict_override(
        honest_valid,
        override_model,
    )

    full_short = train_hold_model(fit, calibration)
    full_heads = _train_consensus_heads(fit, calibration)
    scored = fresh.copy()
    scored["short_hold_probability"] = predict_hold(
        scored,
        full_short,
    )
    scored["consensus_hurdle_ev_pct"] = _score_consensus(
        scored,
        full_heads,
    )
    scored["predicted_patience_override_pct"] = predict_override(
        scored,
        override_model,
    )

    short_hold = pd.to_numeric(
        scored["short_hold_probability"],
        errors="coerce",
    ).gt(0.5)
    override_hold = (
        ~short_hold
        & pd.to_numeric(
            scored["predicted_patience_override_pct"],
            errors="coerce",
        ).gt(0)
    )
    scored["candidate_hold_decision"] = np.where(
        short_hold | override_hold,
        1.0,
        0.0,
    )

    candidate, coverage = build_trajectories(
        scored,
        score_column="candidate_hold_decision",
        threshold=0.5,
        policy="honest_patience_override",
    )
    comparator, comparator_coverage = build_trajectories(
        scored,
        score_column="short_hold_probability",
        threshold=0.5,
        policy="request142_hold_classifier",
    )

    candidate_metrics = trajectory_metrics(
        candidate,
        policy="honest_patience_override",
    )
    comparator_metrics = trajectory_metrics(
        comparator,
        policy="request142_hold_classifier",
    )
    difference = matched_difference(candidate, comparator)

    c_completion = candidate_metrics.get("completion_coverage")
    diff_day = difference.get("day_balanced_difference_pct")
    diff_low = difference["bootstrap"].get("ci_low_pct")
    c_severe = candidate_metrics.get("severe_loss_rate")
    b_severe = comparator_metrics.get("severe_loss_rate")

    controller_gate = bool(
        coverage["start_state_coverage"] is not None
        and float(coverage["start_state_coverage"]) >= MIN_START_COVERAGE
        and c_completion is not None
        and float(c_completion) >= MIN_COMPLETION_COVERAGE
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
        and c_severe is not None
        and b_severe is not None
        and float(c_severe) <= float(b_severe)
    )

    c_day = candidate_metrics.get("day_balanced_base_mean_pct")
    c_boot_low = candidate_metrics.get("day_bootstrap", {}).get(
        "ci_low_pct"
    )
    economic_gate = bool(
        controller_gate
        and c_day is not None
        and float(c_day) > 0
        and c_boot_low is not None
        and float(c_boot_low) > 0
    )

    override_pred = pd.to_numeric(
        scored["predicted_patience_override_pct"],
        errors="coerce",
    )
    override_fit_pred = pd.to_numeric(
        honest_valid["override_fit_prediction_pct"],
        errors="coerce",
    )
    honest_target = pd.to_numeric(
        honest_valid["patience_override_advantage_pct"],
        errors="coerce",
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "entry_policy_changed": False,
        "calibration_split": {
            "calibration_a_days": cal_a_days,
            "calibration_b_days": cal_b_days,
        },
        "honest_target_diagnostics": {
            **honest_diag,
            "override_training_rows": override_model.training_rows,
            "winsor_low_pct": override_model.winsor_low,
            "winsor_high_pct": override_model.winsor_high,
            "fit_prediction_mean_pct": (
                float(override_fit_pred.mean())
                if override_fit_pred.notna().any()
                else None
            ),
            "fit_prediction_positive_rate": (
                float(override_fit_pred.gt(0).mean())
                if override_fit_pred.notna().any()
                else None
            ),
            "fit_prediction_spearman": (
                float(
                    override_fit_pred.corr(
                        honest_target,
                        method="spearman",
                    )
                )
                if int(
                    (override_fit_pred.notna() & honest_target.notna()).sum()
                ) >= 20
                else None
            ),
        },
        "fresh_decision_diagnostics": {
            "rows": int(len(scored)),
            "short_hold_rate": float(short_hold.mean()),
            "override_hold_rate": float(override_hold.mean()),
            "combined_hold_rate": float(
                (short_hold | override_hold).mean()
            ),
            "override_prediction_mean_pct": (
                float(override_pred.mean())
                if override_pred.notna().any()
                else None
            ),
            "override_prediction_positive_rate": (
                float(override_pred.gt(0).mean())
                if override_pred.notna().any()
                else None
            ),
        },
        "candidate_coverage": coverage,
        "comparator_coverage": comparator_coverage,
        "candidate": candidate_metrics,
        "comparator": comparator_metrics,
        "matched_candidate_minus_comparator": difference,
        "frozen_gates": {
            "min_start_coverage": MIN_START_COVERAGE,
            "min_completion_coverage": MIN_COMPLETION_COVERAGE,
            "controller_difference_must_be_positive": True,
            "controller_difference_bootstrap_low_must_be_positive": True,
            "severe_loss_rate_no_worse_than_comparator": True,
            "economic_day_balanced_base_must_be_positive": True,
            "economic_bootstrap_low_must_be_positive": True,
        },
        "controller_improvement_gate_pass": controller_gate,
        "economic_gate_pass": economic_gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.concat(
        [candidate, comparator],
        ignore_index=True,
    ).to_parquet(
        trajectories_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--history-scan", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectories-output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.trajectories_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
