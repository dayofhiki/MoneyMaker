"""Request 209: opportunity admission gate composed with Request-208 timing."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .pullback_turn_transition_entry import (
    MODEL_FEATURES,
    attach_transition_features,
    predict as predict_timing,
    train_model as train_timing_model,
)
from .recurrent_wait_entry_action_value import build_watch_states
from .relative_recurrent_entry_timing import (
    _metrics,
    _policy_trades,
    attach_relative_advantage,
)
from .wait_reobserve_entry_confirmation import _day_weights, _feature_frame

REQUEST_ID = 209
CLASSIFIER_SEED = 20261112
POSITIVE_SEED = 20261113
NONPOSITIVE_SEED = 20261114
BOOTSTRAP_SEED = 20261115
BOOTSTRAP_SAMPLES = 10_000
MIN_STATE_COVERAGE = 0.80
MIN_ADMISSION_RATE = 0.05
MAX_ADMISSION_RATE = 0.70
MIN_TRADES = 30
MIN_POSITIVE_DAYS = 4


@dataclass(frozen=True)
class HurdleAdmissionModel:
    classifier: HistGradientBoostingClassifier
    platt: LogisticRegression
    positive: HistGradientBoostingRegressor
    nonpositive: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    positive_offset: float
    nonpositive_offset: float


def attach_episode_opportunity(states: pd.DataFrame) -> pd.DataFrame:
    result = states.copy()
    result["episode_best_entry_3m_base_pct"] = np.nan

    for _, group in result.groupby(EPISODE_KEYS, sort=False):
        values = pd.to_numeric(group["enter_3m_base_pct"], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        best = float(valid.max())
        result.loc[group.index, "episode_best_entry_3m_base_pct"] = best
    return result


def minute1_states(states: pd.DataFrame) -> pd.DataFrame:
    minute = pd.to_numeric(states["minutes_since_hot"], errors="coerce")
    return states.loc[minute.eq(1)].copy()


def _fit_regressor(
    frame: pd.DataFrame,
    target: pd.Series,
    columns: tuple[str, ...],
    *,
    seed: int,
) -> HistGradientBoostingRegressor:
    y = target.astype(float)
    if len(frame) < 100:
        raise ValueError("request 209 needs >=100 rows per magnitude branch")
    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _feature_frame(frame, columns),
        y.clip(lower=low, upper=high),
        sample_weight=_day_weights(frame),
    )
    return model


def train_admission_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> HurdleAdmissionModel:
    target = pd.to_numeric(
        fit["episode_best_entry_3m_base_pct"], errors="coerce"
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 300:
        raise ValueError("request 209 needs >=300 fit admission rows")

    columns = tuple(
        column
        for column in MODEL_FEATURES
        if column in train.columns
        and pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    label = y.gt(0).astype(int)
    if label.nunique() < 2:
        raise ValueError("request 209 admission fit target has one class")

    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=CLASSIFIER_SEED,
    )
    classifier.fit(
        _feature_frame(train, columns),
        label,
        sample_weight=_day_weights(train),
    )

    positive_mask = y.gt(0)
    nonpositive_mask = ~positive_mask
    positive = _fit_regressor(
        train.loc[positive_mask],
        y.loc[positive_mask],
        columns,
        seed=POSITIVE_SEED,
    )
    nonpositive = _fit_regressor(
        train.loc[nonpositive_mask],
        y.loc[nonpositive_mask],
        columns,
        seed=NONPOSITIVE_SEED,
    )

    cal_target = pd.to_numeric(
        calibration["episode_best_entry_3m_base_pct"], errors="coerce"
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    cal_y = cal_target.loc[cal_valid].astype(float)
    if len(cal) < 150 or cal_y.gt(0).nunique() < 2:
        raise ValueError("request 209 needs valid two-class calibration rows")

    raw_p = np.clip(
        classifier.predict_proba(_feature_frame(cal, columns))[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw_p / (1.0 - raw_p)).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=CLASSIFIER_SEED,
    )
    platt.fit(logits, cal_y.gt(0).astype(int))

    pos_cal = cal.loc[cal_y.gt(0)].copy()
    neg_cal = cal.loc[~cal_y.gt(0)].copy()
    if len(pos_cal) < 50 or len(neg_cal) < 50:
        raise ValueError("request 209 needs >=50 calibration rows per branch")

    pos_raw = positive.predict(_feature_frame(pos_cal, columns))
    neg_raw = nonpositive.predict(_feature_frame(neg_cal, columns))
    positive_offset = float(
        pd.to_numeric(
            pos_cal["episode_best_entry_3m_base_pct"], errors="coerce"
        ).mean()
        - float(np.mean(pos_raw))
    )
    nonpositive_offset = float(
        pd.to_numeric(
            neg_cal["episode_best_entry_3m_base_pct"], errors="coerce"
        ).mean()
        - float(np.mean(neg_raw))
    )

    return HurdleAdmissionModel(
        classifier=classifier,
        platt=platt,
        positive=positive,
        nonpositive=nonpositive,
        columns=columns,
        positive_offset=positive_offset,
        nonpositive_offset=nonpositive_offset,
    )


def predict_admission_ev(
    frame: pd.DataFrame,
    fitted: HurdleAdmissionModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = _feature_frame(frame, fitted.columns)
    raw_p = np.clip(
        fitted.classifier.predict_proba(features)[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw_p / (1.0 - raw_p)).reshape(-1, 1)
    probability = fitted.platt.predict_proba(logits)[:, 1]
    positive = fitted.positive.predict(features) + fitted.positive_offset
    nonpositive = (
        fitted.nonpositive.predict(features) + fitted.nonpositive_offset
    )
    expected = probability * positive + (1.0 - probability) * nonpositive
    return expected, probability, positive, nonpositive


def _admitted_timing_policy(
    scored: pd.DataFrame,
    admitted_keys: set[tuple[str, str, int]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    filtered = scored.loc[
        [
            (str(row.trading_day), str(row.ticker), int(row.hot_t))
            in admitted_keys
            for row in scored.itertuples(index=False)
        ]
    ].copy()
    trades, counts = _policy_trades(filtered, recurrent=True)
    counts["admitted_episodes"] = len(admitted_keys)
    return trades, counts


def _day_bootstrap(trades: pd.DataFrame) -> dict[str, float | int | None]:
    if trades.empty:
        return {
            "days": 0,
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    value = pd.to_numeric(trades["realized_base_return_pct"], errors="coerce")
    daily = (
        trades.assign(_value=value)
        .groupby(trades["trading_day"].astype(str))["_value"]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(daily) < 2:
        return {
            "days": int(len(daily)),
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": float(daily.mean()) if len(daily) else None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(daily), size=(BOOTSTRAP_SAMPLES, len(daily))
    )
    draws = daily[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(daily)),
        "samples": BOOTSTRAP_SAMPLES,
        "mean_pct": float(daily.mean()),
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit = attach_transition_features(
        attach_episode_opportunity(
            attach_relative_advantage(
                build_watch_states(pd.read_parquet(fit_positions_path))
            )
        )
    )
    calibration = attach_transition_features(
        attach_episode_opportunity(
            attach_relative_advantage(
                build_watch_states(pd.read_parquet(calibration_positions_path))
            )
        )
    )

    admission_model = train_admission_model(
        minute1_states(fit),
        minute1_states(calibration),
    )
    timing_model = train_timing_model(fit, calibration)

    paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(paths) != len(FRESH_DAYS):
        raise ValueError("request 209 requires complete request 178 position shards")
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in paths], ignore_index=True
    )
    fresh = attach_transition_features(
        attach_episode_opportunity(
            attach_relative_advantage(build_watch_states(fresh_positions))
        )
    )
    if sorted(fresh["trading_day"].astype(str).unique()) != list(FRESH_DAYS):
        raise ValueError("request 209 fresh days differ from request 178")

    fresh["predicted_relative_advantage_pct"] = predict_timing(
        fresh, timing_model
    )

    first = minute1_states(fresh)
    expected, probability, positive, nonpositive = predict_admission_ev(
        first, admission_model
    )
    first["predicted_episode_opportunity_ev_pct"] = expected
    first["predicted_episode_positive_probability"] = probability
    first["predicted_episode_positive_magnitude_pct"] = positive
    first["predicted_episode_nonpositive_magnitude_pct"] = nonpositive
    first["admitted"] = expected > 0

    admitted = first.loc[first["admitted"]].copy()
    admitted_keys = {
        (str(row.trading_day), str(row.ticker), int(row.hot_t))
        for row in admitted.itertuples(index=False)
    }

    ungated, ungated_counts = _policy_trades(fresh, recurrent=True)
    gated, gated_counts = _admitted_timing_policy(fresh, admitted_keys)
    ungated["policy"] = "transition_timing_ungated"
    gated["policy"] = "opportunity_admission_plus_transition_timing"

    total_episodes = int(
        fresh_positions.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    coverage = float(len(first) / total_episodes) if total_episodes else None
    admission_rate = float(len(admitted) / len(first)) if len(first) else None

    ungated_metrics = _metrics(ungated)
    gated_metrics = _metrics(gated)
    bootstrap = _day_bootstrap(gated)
    gated_day = gated_metrics.get("day_balanced_mean_pct")
    ungated_day = ungated_metrics.get("day_balanced_mean_pct")
    gated_severe = gated_metrics.get("severe_loss_rate")
    ungated_severe = ungated_metrics.get("severe_loss_rate")
    by_day = gated_metrics.get("by_day", {})
    positive_days = sum(float(value) > 0 for value in by_day.values())
    gated_trades = int(gated_metrics.get("trades", 0))

    gate = bool(
        coverage is not None
        and coverage >= MIN_STATE_COVERAGE
        and admission_rate is not None
        and MIN_ADMISSION_RATE <= admission_rate <= MAX_ADMISSION_RATE
        and gated_trades >= MIN_TRADES
        and gated_day is not None
        and float(gated_day) > 0
        and ungated_day is not None
        and float(gated_day) > float(ungated_day)
        and positive_days >= MIN_POSITIVE_DAYS
        and bootstrap["ci_low_pct"] is not None
        and float(bootstrap["ci_low_pct"]) > 0
        and gated_severe is not None
        and ungated_severe is not None
        and float(gated_severe) <= float(ungated_severe)
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "architecture": "minute-1 opportunity hurdle admission -> Request-208 transition timing",
        "state_coverage": coverage,
        "minute1_states": int(len(first)),
        "admitted_episodes": int(len(admitted)),
        "admission_rate": admission_rate,
        "admission_prediction": {
            "ev_mean_pct": float(np.mean(expected)) if len(expected) else None,
            "positive_probability_mean": (
                float(np.mean(probability)) if len(probability) else None
            ),
            "positive_magnitude_mean_pct": (
                float(np.mean(positive)) if len(positive) else None
            ),
            "nonpositive_magnitude_mean_pct": (
                float(np.mean(nonpositive)) if len(nonpositive) else None
            ),
        },
        "ungated_transition_timing": ungated_metrics,
        "gated_policy": gated_metrics,
        "gated_day_bootstrap": bootstrap,
        "positive_gated_days": int(positive_days),
        "ungated_path_counts": ungated_counts,
        "gated_path_counts": gated_counts,
        "frozen_gate": {
            "min_state_coverage": MIN_STATE_COVERAGE,
            "min_admission_rate": MIN_ADMISSION_RATE,
            "max_admission_rate": MAX_ADMISSION_RATE,
            "min_trades": MIN_TRADES,
            "gated_day_balanced_base_must_be_positive": True,
            "must_improve_vs_ungated_transition_timing": True,
            "min_positive_days": MIN_POSITIVE_DAYS,
            "bootstrap_low_must_be_positive": True,
            "severe_loss_no_worse_than_ungated": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    combined = pd.concat([ungated, gated], ignore_index=True)
    combined.to_parquet(rows_output_path, index=False, compression="zstd")
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
