"""Request 208: causal pullback/turn transition features for recurrent entry timing."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .recurrent_wait_entry_action_value import MAX_WAIT_MINUTES, WATCH_FEATURES, build_watch_states
from .relative_recurrent_entry_timing import (
    BOOTSTRAP_SAMPLES,
    _matched_difference,
    _metrics,
    _policy_trades,
    _safe_spearman,
    attach_relative_advantage,
)
from .wait_reobserve_entry_confirmation import _day_weights, _feature_frame

REQUEST_ID = 208
MODEL_SEED = 20261111
MIN_STATE_COVERAGE = 0.80
MIN_TRADES = 100
MIN_WAIT_USAGE = 0.10
MIN_SPEARMAN = 0.05
MIN_IMPROVED_DAYS = 4

TRANSITION_FEATURES = (
    "watch_price_change_1m_pct",
    "watch_price_change_accel_pct",
    "watch_return_from_start_pct",
    "watch_drawdown_from_peak_pct",
    "watch_rebound_from_trough_pct",
    "watch_attention_change_1m",
    "watch_attention_drawdown_from_peak",
    "watch_rank_change_1m",
    "watch_log_volume_change_1m",
    "watch_log_transactions_change_1m",
    "watch_active_seconds_change_1m",
    "watch_sec_last5_return_change_1m",
    "watch_sec_last10_return_change_1m",
    "watch_close_vs_vwap_change_1m",
    "watch_volume_burst5_change_1m",
    "watch_transactions_burst5_change_1m",
)
MODEL_FEATURES = tuple(dict.fromkeys([*WATCH_FEATURES, *TRANSITION_FEATURES]))


@dataclass(frozen=True)
class TransitionModel:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def _number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else np.nan


def _pct_change(current: float, previous: float) -> float:
    if not np.isfinite(current) or not np.isfinite(previous) or previous <= 0:
        return np.nan
    return float((current / previous - 1.0) * 100.0)


def attach_transition_features(states: pd.DataFrame) -> pd.DataFrame:
    result = states.copy()
    for column in TRANSITION_FEATURES:
        result[column] = np.nan

    for _, group in result.groupby(EPISODE_KEYS, sort=False):
        ordered = group.sort_values("minutes_since_hot", kind="stable")
        first_price = np.nan
        price_peak = np.nan
        price_trough = np.nan
        attention_peak = np.nan
        previous_row: pd.Series | None = None
        previous_price_change = np.nan

        for index, row in ordered.iterrows():
            price = _number(row, "current_price")
            attention = _number(row, "attention_score")
            if np.isfinite(price):
                if not np.isfinite(first_price):
                    first_price = price
                price_peak = price if not np.isfinite(price_peak) else max(price_peak, price)
                price_trough = price if not np.isfinite(price_trough) else min(price_trough, price)
            if np.isfinite(attention):
                attention_peak = (
                    attention
                    if not np.isfinite(attention_peak)
                    else max(attention_peak, attention)
                )

            result.at[index, "watch_return_from_start_pct"] = _pct_change(
                price, first_price
            )
            result.at[index, "watch_drawdown_from_peak_pct"] = _pct_change(
                price, price_peak
            )
            result.at[index, "watch_rebound_from_trough_pct"] = _pct_change(
                price, price_trough
            )
            if np.isfinite(attention) and np.isfinite(attention_peak):
                result.at[index, "watch_attention_drawdown_from_peak"] = (
                    attention - attention_peak
                )

            if previous_row is not None:
                previous_price = _number(previous_row, "current_price")
                price_change = _pct_change(price, previous_price)
                result.at[index, "watch_price_change_1m_pct"] = price_change
                if np.isfinite(price_change) and np.isfinite(previous_price_change):
                    result.at[index, "watch_price_change_accel_pct"] = (
                        price_change - previous_price_change
                    )

                pairs = (
                    ("attention_score", "watch_attention_change_1m"),
                    ("attention_rank", "watch_rank_change_1m"),
                    ("log_minute_volume", "watch_log_volume_change_1m"),
                    ("log_minute_transactions", "watch_log_transactions_change_1m"),
                    ("active_seconds_60", "watch_active_seconds_change_1m"),
                    ("sec_last5_return_pct", "watch_sec_last5_return_change_1m"),
                    ("sec_last10_return_pct", "watch_sec_last10_return_change_1m"),
                    ("sec_close_vs_vwap_pct", "watch_close_vs_vwap_change_1m"),
                    ("sec_volume_burst_5", "watch_volume_burst5_change_1m"),
                    (
                        "sec_transactions_burst_5",
                        "watch_transactions_burst5_change_1m",
                    ),
                )
                for source, target in pairs:
                    current = _number(row, source)
                    previous = _number(previous_row, source)
                    if np.isfinite(current) and np.isfinite(previous):
                        result.at[index, target] = current - previous
                previous_price_change = price_change
            previous_row = row
    return result


def train_model(fit: pd.DataFrame, calibration: pd.DataFrame) -> TransitionModel:
    target = pd.to_numeric(fit["enter_vs_wait1_advantage_pct"], errors="coerce")
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 300:
        raise ValueError("request 208 needs >=300 relative-advantage fit rows")

    columns = tuple(
        column
        for column in MODEL_FEATURES
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
        calibration["enter_vs_wait1_advantage_pct"], errors="coerce"
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 150:
        raise ValueError("request 208 needs >=150 calibration rows")
    raw = model.predict(_feature_frame(cal, columns))
    residual = cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    offset = float(np.average(residual, weights=_day_weights(cal)))
    return TransitionModel(
        model=model,
        columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict(frame: pd.DataFrame, fitted: TransitionModel) -> np.ndarray:
    return fitted.model.predict(_feature_frame(frame, fitted.columns)) + fitted.offset


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit = attach_transition_features(
        attach_relative_advantage(build_watch_states(pd.read_parquet(fit_positions_path)))
    )
    calibration = attach_transition_features(
        attach_relative_advantage(
            build_watch_states(pd.read_parquet(calibration_positions_path))
        )
    )
    model = train_model(fit, calibration)

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(position_paths) != len(FRESH_DAYS):
        raise ValueError("request 208 requires complete request 178 position shards")
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths], ignore_index=True
    )
    fresh = attach_transition_features(
        attach_relative_advantage(build_watch_states(fresh_positions))
    )
    if sorted(fresh["trading_day"].astype(str).unique()) != list(FRESH_DAYS):
        raise ValueError("request 208 fresh days differ from request 178")

    fresh["predicted_relative_advantage_pct"] = predict(fresh, model)
    spearman = _safe_spearman(
        fresh["enter_vs_wait1_advantage_pct"],
        fresh["predicted_relative_advantage_pct"],
    )

    total_episodes = int(
        fresh_positions.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    minute1 = int(
        fresh.loc[
            pd.to_numeric(fresh["minutes_since_hot"], errors="coerce").eq(1),
            EPISODE_KEYS,
        ].drop_duplicates().shape[0]
    )
    coverage = float(minute1 / total_episodes) if total_episodes else None

    baseline, baseline_counts = _policy_trades(fresh, recurrent=False)
    recurrent, recurrent_counts = _policy_trades(fresh, recurrent=True)
    baseline["policy"] = "always_enter_min1"
    recurrent["policy"] = "transition_relative_recurrent"

    baseline_metrics = _metrics(baseline)
    recurrent_metrics = _metrics(recurrent)
    difference = _matched_difference(recurrent, baseline)
    wait_usage = (
        float(recurrent_counts["wait_actions"] / recurrent_counts["episodes"])
        if recurrent_counts["episodes"]
        else None
    )

    diff_day = difference.get("day_balanced_difference_pct")
    diff_low = difference.get("bootstrap", {}).get("ci_low_pct")
    improved_days = int(difference.get("improved_days", 0))
    rec_severe = recurrent_metrics.get("severe_loss_rate")
    base_severe = baseline_metrics.get("severe_loss_rate")
    rec_trades = int(recurrent_metrics.get("trades", 0))

    gate = bool(
        coverage is not None
        and coverage >= MIN_STATE_COVERAGE
        and rec_trades >= MIN_TRADES
        and wait_usage is not None
        and wait_usage >= MIN_WAIT_USAGE
        and spearman is not None
        and spearman >= MIN_SPEARMAN
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
        and improved_days >= MIN_IMPROVED_DAYS
        and rec_severe is not None
        and base_severe is not None
        and float(rec_severe) <= float(base_severe)
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "hypothesis": "Explicit causal pullback/turn transition geometry improves recurrent entry timing.",
        "fresh_relative_advantage_spearman": spearman,
        "state_coverage": coverage,
        "transition_features": list(TRANSITION_FEATURES),
        "model_features": list(model.columns),
        "model_offset": model.offset,
        "baseline": baseline_metrics,
        "recurrent": recurrent_metrics,
        "baseline_path_counts": baseline_counts,
        "recurrent_path_counts": recurrent_counts,
        "wait_usage_per_episode": wait_usage,
        "matched_recurrent_minus_baseline": difference,
        "frozen_gate": {
            "min_state_coverage": MIN_STATE_COVERAGE,
            "min_trades": MIN_TRADES,
            "min_wait_usage": MIN_WAIT_USAGE,
            "min_fresh_relative_advantage_spearman": MIN_SPEARMAN,
            "matched_day_difference_must_be_positive": True,
            "matched_bootstrap_low_must_be_positive": True,
            "min_improved_days": MIN_IMPROVED_DAYS,
            "severe_loss_no_worse_than_baseline": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.concat([baseline, recurrent], ignore_index=True).to_parquet(
        rows_output_path, index=False, compression="zstd"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument("--calibration-positions", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.fresh_dir,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
