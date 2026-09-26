"""Request 218: learned chronological sequence representation for HOLD ranking.

Request217 showed that hand-crafted 3/5-observation path summaries did not
recover HOLD-value ranking. Request218 keeps the same event-time POSITION
population and within-day ranking target, but replaces manual path summaries
with a learned fixed-length chronological representation.

The candidate receives:
* the current Request216 state vector;
* the last 8 actually observed POSITION states, including the current state;
* per-observation causal time gaps / age and padding masks.

The sequence is presented in chronological lag slots and a small MLP learns the
interactions directly. No future observation timestamp, price, gap, label, or
best-future quantity enters the input.

This is a development-only representation diagnostic. It does not open new
dates and does not claim a profitable trading policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from .event_time_position_replay import (
    EVENT_FEATURES,
    predict_event_advantage,
    train_event_advantage_model,
)
from .fresh_consensus_hurdle_value import FRESH_DAYS
from .recent_path_position_ranking import (
    CANDIDATE_FEATURES as HANDCRAFTED_FEATURES,
    PATH_SOURCES,
    attach_rank_target,
    attach_recent_path_features,
    predict_rank,
    ranking_diagnostics,
    train_rank_model,
    _prepare_event_frames,
)

REQUEST_ID = 218
MODEL_SEED = 20261128
SEQUENCE_LENGTH = 8

SEQUENCE_SOURCES = tuple(
    dict.fromkeys(
        [
            *PATH_SOURCES,
            "elapsed_since_previous_observation_min",
            "minutes_held",
        ]
    )
)

MIN_SPEARMAN = 0.05
MIN_GAIN_VS_CURRENT_MLP = 0.03
MIN_GAIN_VS_HANDCRAFTED = 0.03
MIN_GOOD_DAYS = 3
MIN_TOP10_POSITIVE_RATE_GAIN = 0.05
MIN_CALIBRATION_SPEARMAN = 0.02


@dataclass(frozen=True)
class MLPSequenceRankModel:
    model: MLPRegressor
    feature_columns: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray


def sequence_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for lag in range(SEQUENCE_LENGTH):
        names.append(f"seq_lag{lag}_mask")
        names.append(f"seq_lag{lag}_age_min")
        for source in SEQUENCE_SOURCES:
            names.append(f"seq_lag{lag}_{source}")
    return tuple(names)


SEQUENCE_FEATURES = sequence_feature_names()
LEARNED_SEQUENCE_FEATURES = tuple(
    dict.fromkeys([*EVENT_FEATURES, *SEQUENCE_FEATURES])
)


def attach_sequence_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach only current/past observed states in fixed chronological lag slots."""
    result = frame.copy()
    for column in SEQUENCE_FEATURES:
        result[column] = np.nan

    keys = ["trading_day", "ticker", "hot_t"]
    for _, group in result.groupby(keys, sort=False):
        ordered = group.sort_values("state_t", kind="stable")
        indices = list(ordered.index)
        times = pd.to_numeric(
            ordered["state_t"], errors="coerce"
        ).to_numpy(dtype=float)

        source_values: dict[str, np.ndarray] = {}
        for source in SEQUENCE_SOURCES:
            series = (
                ordered[source]
                if source in ordered.columns
                else pd.Series(np.nan, index=ordered.index)
            )
            source_values[source] = pd.to_numeric(
                series, errors="coerce"
            ).to_numpy(dtype=float)

        for position, index in enumerate(indices):
            current_t = times[position]
            for lag in range(SEQUENCE_LENGTH):
                history_position = position - lag
                if history_position < 0:
                    result.at[index, f"seq_lag{lag}_mask"] = 0.0
                    continue

                history_t = times[history_position]
                result.at[index, f"seq_lag{lag}_mask"] = 1.0
                if np.isfinite(current_t) and np.isfinite(history_t):
                    result.at[index, f"seq_lag{lag}_age_min"] = float(
                        (current_t - history_t) / 60_000.0
                    )

                for source in SEQUENCE_SOURCES:
                    value = source_values[source][history_position]
                    if np.isfinite(value):
                        result.at[
                            index, f"seq_lag{lag}_{source}"
                        ] = float(value)
    return result


def _usable_columns(
    frame: pd.DataFrame,
    feature_family: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        column
        for column in feature_family
        if column in frame.columns
        and pd.to_numeric(
            frame[column], errors="coerce"
        ).notna().any()
    )


def _fit_scaler(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = (
        frame.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    center = np.zeros(matrix.shape[1], dtype=float)
    scale = np.ones(matrix.shape[1], dtype=float)

    for col in range(matrix.shape[1]):
        values = matrix[:, col]
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        median = float(np.median(finite))
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        spread = float(q75 - q25)
        if not np.isfinite(spread) or spread <= 1e-6:
            spread = float(np.std(finite))
        if not np.isfinite(spread) or spread <= 1e-6:
            spread = 1.0
        center[col] = median
        scale[col] = spread
    return center, scale


def _transform(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    matrix = (
        frame.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float, copy=True)
    )
    missing = ~np.isfinite(matrix)
    if missing.any():
        matrix[missing] = np.take(center, np.where(missing)[1])
    matrix = (matrix - center) / scale
    return np.clip(matrix, -10.0, 10.0)


def train_mlp_rank_model(
    fit: pd.DataFrame,
    feature_family: tuple[str, ...],
) -> MLPSequenceRankModel:
    target = pd.to_numeric(
        fit["hold_advantage_day_rank"], errors="coerce"
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].to_numpy(dtype=float)
    if len(train) < 1000:
        raise ValueError("request 218 insufficient sequence-rank support")

    columns = _usable_columns(train, feature_family)
    if not columns:
        raise ValueError("request 218 has no usable model features")

    center, scale = _fit_scaler(train, columns)
    x = _transform(train, columns, center, scale)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=250,
        shuffle=True,
        random_state=MODEL_SEED,
        tol=1e-4,
        n_iter_no_change=20,
        early_stopping=False,
    )
    model.fit(x, y)
    return MLPSequenceRankModel(
        model=model,
        feature_columns=columns,
        center=center,
        scale=scale,
    )


def predict_mlp_rank(
    frame: pd.DataFrame,
    fitted: MLPSequenceRankModel,
) -> np.ndarray:
    x = _transform(
        frame,
        fitted.feature_columns,
        fitted.center,
        fitted.scale,
    )
    return fitted.model.predict(x)


def _validate_sequence_contract() -> None:
    forbidden_exact = {
        "next_event_t",
        "next_event_base_return_pct",
        "hold_advantage_event_pct",
        "time_to_next_observation",
        "event_exit_minute_after_hot",
    }
    overlap = forbidden_exact.intersection(LEARNED_SEQUENCE_FEATURES)
    if overlap:
        raise ValueError(
            f"request 218 future fields leaked into sequence: {sorted(overlap)}"
        )


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    states_output_path: Path,
) -> int:
    _validate_sequence_contract()

    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )
    history_scan = pd.read_parquet(history_scan_path)

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError("request 218 requires complete request178 shards")

    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    found_days = sorted(
        fresh_positions["trading_day"].astype(str).unique()
    )
    if found_days != list(FRESH_DAYS):
        raise ValueError(
            f"request 218 expected {FRESH_DAYS}, found {found_days}"
        )

    fit, calibration, fresh = _prepare_event_frames(
        fit_positions,
        calibration_positions,
        fresh_positions,
        history_scan,
        fresh_scan,
    )
    fit = attach_rank_target(fit)
    calibration = attach_rank_target(calibration)
    fresh = attach_rank_target(fresh)

    fit_sequence = attach_sequence_features(fit)
    calibration_sequence = attach_sequence_features(calibration)
    fresh_sequence = attach_sequence_features(fresh)

    current_model = train_mlp_rank_model(
        fit_sequence, EVENT_FEATURES
    )
    sequence_model = train_mlp_rank_model(
        fit_sequence, LEARNED_SEQUENCE_FEATURES
    )

    fit_handcrafted = attach_recent_path_features(fit.copy())
    fresh_handcrafted = attach_recent_path_features(fresh.copy())
    handcrafted_model = train_rank_model(
        fit_handcrafted, HANDCRAFTED_FEATURES
    )

    request216_model = train_event_advantage_model(
        fit, calibration
    )

    calibration_sequence["current_mlp_rank_score"] = (
        predict_mlp_rank(calibration_sequence, current_model)
    )
    calibration_sequence["sequence_mlp_rank_score"] = (
        predict_mlp_rank(calibration_sequence, sequence_model)
    )

    fresh_sequence["current_mlp_rank_score"] = predict_mlp_rank(
        fresh_sequence, current_model
    )
    fresh_sequence["sequence_mlp_rank_score"] = predict_mlp_rank(
        fresh_sequence, sequence_model
    )
    fresh_sequence["request217_handcrafted_rank_score"] = predict_rank(
        fresh_handcrafted, handcrafted_model
    )
    fresh_sequence["request216_expected_advantage_score"] = (
        predict_event_advantage(fresh, request216_model)
    )

    current_diag = ranking_diagnostics(
        fresh_sequence, "current_mlp_rank_score"
    )
    candidate_diag = ranking_diagnostics(
        fresh_sequence, "sequence_mlp_rank_score"
    )
    handcrafted_diag = ranking_diagnostics(
        fresh_sequence, "request217_handcrafted_rank_score"
    )
    request216_diag = ranking_diagnostics(
        fresh_sequence, "request216_expected_advantage_score"
    )
    calibration_candidate_diag = ranking_diagnostics(
        calibration_sequence, "sequence_mlp_rank_score"
    )

    candidate_s = candidate_diag.get("spearman")
    current_s = current_diag.get("spearman")
    handcrafted_s = handcrafted_diag.get("spearman")
    calibration_s = calibration_candidate_diag.get("spearman")
    top10 = candidate_diag["top10_by_day"]
    top20 = candidate_diag["top20_by_day"]
    overall_positive = candidate_diag.get("overall_positive_rate")
    top10_positive = top10.get("positive_rate")
    same_age = candidate_diag.get("same_age_median_spearman")

    gain_current = (
        float(candidate_s) - float(current_s)
        if candidate_s is not None and current_s is not None
        else None
    )
    gain_handcrafted = (
        float(candidate_s) - float(handcrafted_s)
        if candidate_s is not None and handcrafted_s is not None
        else None
    )

    gate = bool(
        candidate_s is not None
        and float(candidate_s) >= MIN_SPEARMAN
        and gain_current is not None
        and float(gain_current) >= MIN_GAIN_VS_CURRENT_MLP
        and gain_handcrafted is not None
        and float(gain_handcrafted) >= MIN_GAIN_VS_HANDCRAFTED
        and calibration_s is not None
        and float(calibration_s) >= MIN_CALIBRATION_SPEARMAN
        and int(candidate_diag["good_days"]) >= MIN_GOOD_DAYS
        and top10.get("day_balanced_mean_advantage_pct") is not None
        and float(top10["day_balanced_mean_advantage_pct"]) > 0
        and top20.get("day_balanced_mean_advantage_pct") is not None
        and float(top20["day_balanced_mean_advantage_pct"]) > 0
        and overall_positive is not None
        and top10_positive is not None
        and float(top10_positive) - float(overall_positive)
        >= MIN_TOP10_POSITIVE_RATE_GAIN
        and same_age is not None
        and float(same_age) > 0
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "hypothesis": (
            "a learned fixed-length chronological state representation "
            "can recover HOLD-value ordering that hand-crafted summaries "
            "and current-state models missed"
        ),
        "target": (
            "within-day percentile rank of realized next-executable-event "
            "BASE HOLD advantage"
        ),
        "sequence": {
            "observations": SEQUENCE_LENGTH,
            "sources": list(SEQUENCE_SOURCES),
            "sequence_feature_count": len(SEQUENCE_FEATURES),
            "includes_current_state_vector": True,
            "padding_mask": True,
            "causal_age_features": True,
            "future_information_used": False,
        },
        "model": {
            "family": "MLPRegressor",
            "hidden_layers": [64, 32],
            "activation": "relu",
            "alpha": 0.01,
            "max_iter": 250,
            "current_only_feature_count": len(
                current_model.feature_columns
            ),
            "candidate_feature_count": len(
                sequence_model.feature_columns
            ),
            "candidate_iterations": int(sequence_model.model.n_iter_),
        },
        "fit_rows": int(len(fit_sequence)),
        "calibration_rows": int(len(calibration_sequence)),
        "fresh_rows": int(len(fresh_sequence)),
        "comparators": {
            "request216_expected_value": request216_diag,
            "current_state_mlp_rank": current_diag,
            "request217_handcrafted_rank": handcrafted_diag,
        },
        "candidate_learned_sequence_rank": candidate_diag,
        "calibration_candidate_learned_sequence_rank": (
            calibration_candidate_diag
        ),
        "candidate_minus_current_mlp_spearman": gain_current,
        "candidate_minus_handcrafted_spearman": gain_handcrafted,
        "frozen_gate": {
            "min_candidate_spearman": MIN_SPEARMAN,
            "min_gain_vs_current_mlp": MIN_GAIN_VS_CURRENT_MLP,
            "min_gain_vs_handcrafted": MIN_GAIN_VS_HANDCRAFTED,
            "min_calibration_spearman": MIN_CALIBRATION_SPEARMAN,
            "min_good_days": MIN_GOOD_DAYS,
            "top10_day_balanced_advantage_positive": True,
            "top20_day_balanced_advantage_positive": True,
            "min_top10_positive_rate_gain": (
                MIN_TOP10_POSITIVE_RATE_GAIN
            ),
            "same_age_median_spearman_positive": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze the learned sequence representation, calibrate an "
            "absolute HOLD/EXIT value decision on fit/calibration only, "
            "then replay recurrent event-time trajectories"
        ),
        "next_boundary_if_fail": (
            "do not add more path features; challenge the one-step HOLD "
            "target itself with multi-event continuation / option-value "
            "objectives before opening new dates"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    states_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fresh_sequence.to_parquet(
        states_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument(
        "--calibration-positions", type=Path, required=True
    )
    parser.add_argument("--history-scan", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.states_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
