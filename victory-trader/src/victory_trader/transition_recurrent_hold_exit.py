"""Request 210: transition-aware direct recurrent HOLD/EXIT."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .direct_recurrent_hold_exit_advantage import (
    BOOTSTRAP_SEED,
    FEATURES,
    MIN_ADVANTAGE_SPEARMAN,
    MIN_GOOD_DAYS,
    MIN_START_COVERAGE,
    advantage_diagnostics,
    build_trajectories,
    matched_difference,
    predict_advantage as predict_baseline_advantage,
    train_advantage_model as train_baseline_advantage_model,
    trajectory_metrics,
)
from .fresh_consensus_hurdle_value import FRESH_DAYS
from .market_regime_position_value import (
    attach_market_regime,
    build_market_regime,
)

REQUEST_ID = 210
MODEL_SEED = 20261116

POSITION_TRANSITION_FEATURES = (
    "position_return_change_1m_pct",
    "position_return_accel_1m_pct",
    "position_drawdown_change_1m_pct",
    "position_recovery_change_1m_pct",
    "position_attention_change_1m",
    "position_rank_change_1m",
    "position_log_volume_change_1m",
    "position_log_transactions_change_1m",
    "position_active_seconds_change_1m",
    "position_sec_last5_return_change_1m",
    "position_sec_last10_return_change_1m",
    "position_close_vs_vwap_change_1m",
    "position_volume_burst5_change_1m",
    "position_transactions_burst5_change_1m",
)
TRANSITION_FEATURES = tuple(
    dict.fromkeys([*FEATURES, *POSITION_TRANSITION_FEATURES])
)


@dataclass(frozen=True)
class TransitionAdvantageModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def _number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(float(value)):
        return np.nan
    return float(value)


def attach_position_transitions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in POSITION_TRANSITION_FEATURES:
        result[column] = np.nan

    keys = ["trading_day", "ticker", "hot_t"]
    for _, group in result.groupby(keys, sort=False):
        ordered = group.sort_values("minutes_held", kind="stable")
        previous_row: pd.Series | None = None
        previous_return_change = np.nan

        for index, row in ordered.iterrows():
            if previous_row is not None:
                current_return = _number(row, "entry_to_current_close_pct")
                previous_return = _number(
                    previous_row, "entry_to_current_close_pct"
                )
                if np.isfinite(current_return) and np.isfinite(previous_return):
                    delta = current_return - previous_return
                    result.at[index, "position_return_change_1m_pct"] = delta
                    if np.isfinite(previous_return_change):
                        result.at[index, "position_return_accel_1m_pct"] = (
                            delta - previous_return_change
                        )
                    previous_return_change = delta

                pairs = (
                    (
                        "drawdown_from_peak_pct",
                        "position_drawdown_change_1m_pct",
                    ),
                    (
                        "recovery_from_trough_pct",
                        "position_recovery_change_1m_pct",
                    ),
                    ("attention_score", "position_attention_change_1m"),
                    ("attention_rank", "position_rank_change_1m"),
                    ("log_minute_volume", "position_log_volume_change_1m"),
                    (
                        "log_minute_transactions",
                        "position_log_transactions_change_1m",
                    ),
                    ("active_seconds_60", "position_active_seconds_change_1m"),
                    (
                        "sec_last5_return_pct",
                        "position_sec_last5_return_change_1m",
                    ),
                    (
                        "sec_last10_return_pct",
                        "position_sec_last10_return_change_1m",
                    ),
                    (
                        "sec_close_vs_vwap_pct",
                        "position_close_vs_vwap_change_1m",
                    ),
                    (
                        "sec_volume_burst_5",
                        "position_volume_burst5_change_1m",
                    ),
                    (
                        "sec_transactions_burst_5",
                        "position_transactions_burst5_change_1m",
                    ),
                )
                for source, target in pairs:
                    current = _number(row, source)
                    previous = _number(previous_row, source)
                    if np.isfinite(current) and np.isfinite(previous):
                        result.at[index, target] = current - previous
            previous_row = row
    return result


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    day = frame["trading_day"].astype(str)
    counts = day.map(day.value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(dtype=float)
    return weights / float(np.mean(weights))


def train_transition_advantage_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> TransitionAdvantageModel:
    target = pd.to_numeric(fit["hold_advantage_1m_pct"], errors="coerce")
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError("request 210 insufficient fit action-value support")

    columns = tuple(
        column
        for column in TRANSITION_FEATURES
        if column in train.columns
        and pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=MODEL_SEED,
    )
    model.fit(
        _feature_frame(train, columns),
        y.clip(lower=low, upper=high),
        sample_weight=_day_weights(train),
    )

    cal_target = pd.to_numeric(
        calibration["hold_advantage_1m_pct"], errors="coerce"
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    raw = model.predict(_feature_frame(cal, columns))
    residual = cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    offset = float(np.average(residual, weights=_day_weights(cal)))
    return TransitionAdvantageModel(
        model=model,
        feature_columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_transition_advantage(
    frame: pd.DataFrame,
    fitted: TransitionAdvantageModel,
) -> np.ndarray:
    return (
        fitted.model.predict(_feature_frame(frame, fitted.feature_columns))
        + fitted.offset
    )


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
        raise ValueError("request 210 requires complete request 178 shards")

    fresh = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )

    historical_days = set(fit["trading_day"].astype(str)) | set(
        calibration["trading_day"].astype(str)
    )
    historical_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(historical_days)
    ].copy()

    historical_regime = build_market_regime(historical_scan)
    fresh_regime = build_market_regime(fresh_scan)
    fit = attach_market_regime(fit, historical_regime)
    calibration = attach_market_regime(calibration, historical_regime)
    fresh = attach_market_regime(fresh, fresh_regime)

    fit = attach_position_transitions(fit)
    calibration = attach_position_transitions(calibration)
    fresh = attach_position_transitions(fresh)

    baseline_model = train_baseline_advantage_model(fit, calibration)
    transition_model = train_transition_advantage_model(fit, calibration)

    scored = fresh.copy()
    scored["baseline_direct_advantage_pct"] = predict_baseline_advantage(
        scored, baseline_model
    )
    scored["predicted_hold_advantage_1m_pct"] = (
        predict_transition_advantage(scored, transition_model)
    )

    transition_diagnostics = advantage_diagnostics(scored)
    baseline_diag_frame = scored.copy()
    baseline_diag_frame["predicted_hold_advantage_1m_pct"] = (
        baseline_diag_frame["baseline_direct_advantage_pct"]
    )
    baseline_diagnostics = advantage_diagnostics(baseline_diag_frame)

    candidate, coverage = build_trajectories(
        scored,
        score_column="predicted_hold_advantage_1m_pct",
        threshold=0.0,
        policy="transition_direct_expected_advantage",
    )
    comparator, comparator_coverage = build_trajectories(
        scored,
        score_column="baseline_direct_advantage_pct",
        threshold=0.0,
        policy="request179_direct_expected_advantage",
    )

    candidate_metrics = trajectory_metrics(
        candidate, policy="transition_direct_expected_advantage"
    )
    comparator_metrics = trajectory_metrics(
        comparator, policy="request179_direct_expected_advantage"
    )
    difference = matched_difference(candidate, comparator)

    spearman = transition_diagnostics["advantage_spearman"]
    selected_mean = transition_diagnostics[
        "selected_realized_advantage_mean_pct"
    ]
    candidate_day = candidate_metrics.get("day_balanced_base_mean_pct")
    comparator_day = comparator_metrics.get("day_balanced_base_mean_pct")
    diff_day = difference.get("day_balanced_difference_pct")
    diff_low = difference["bootstrap"].get("ci_low_pct")

    gate = bool(
        coverage["start_state_coverage"] is not None
        and float(coverage["start_state_coverage"]) >= MIN_START_COVERAGE
        and spearman is not None
        and float(spearman) >= MIN_ADVANTAGE_SPEARMAN
        and selected_mean is not None
        and float(selected_mean) > 0
        and int(transition_diagnostics["good_days"]) >= MIN_GOOD_DAYS
        and candidate_day is not None
        and comparator_day is not None
        and float(candidate_day) > float(comparator_day)
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
        and float(candidate_day) > 0
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "representation": "Request179 state + causal within-position transition features",
        "transition_feature_count": len(POSITION_TRANSITION_FEATURES),
        "transition_feature_names": list(POSITION_TRANSITION_FEATURES),
        "candidate_feature_count": len(
            transition_model.feature_columns
        ),
        "transition_advantage_diagnostics": transition_diagnostics,
        "request179_retrained_diagnostics": baseline_diagnostics,
        "candidate_coverage": coverage,
        "comparator_coverage": comparator_coverage,
        "candidate": candidate_metrics,
        "comparator_request179": comparator_metrics,
        "matched_candidate_minus_request179": difference,
        "frozen_gate": {
            "min_start_coverage": MIN_START_COVERAGE,
            "min_advantage_spearman": MIN_ADVANTAGE_SPEARMAN,
            "min_good_days": MIN_GOOD_DAYS,
            "selected_advantage_mean_must_be_positive": True,
            "candidate_must_beat_request179": True,
            "matched_bootstrap_low_must_be_positive": True,
            "candidate_day_balanced_base_must_be_positive": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.concat([candidate, comparator], ignore_index=True).to_parquet(
        trajectories_output_path, index=False, compression="zstd"
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
    parser.add_argument("--trajectories-output", type=Path, required=True)
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
