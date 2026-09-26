"""Request 219: multi-event remaining-option-value diagnostic.

Request216-218 repeatedly failed to learn a stable one-step HOLD advantage even
after event-time repair, hand-crafted path features, and a learned recent-state
sequence. Request219 changes the *target* while freezing the representation and
development block.

The new diagnostic target asks a different question:

    If we keep the position alive now, how much additional executable exit
    opportunity exists over the next five actually observed POSITION events?

For each current state we compute an offline upper-bound label equal to the best
BASE exit return among the next five executable observations minus EXIT-now.
This uses future outcomes only as a supervised-learning label. It is never an
input feature and it is explicitly NOT a deployable exit policy because the
best future exit is known only in hindsight.

The experiment keeps the Request218 current-state and 8-observation sequence
representations frozen. Model choice between those two representations is made
using calibration data only, then the chosen representation is evaluated once
on the already-open Request178 development dates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from .direct_recurrent_hold_exit_advantage import (
    MAX_HOLD_MINUTES,
    _safe_spearman,
)
from .event_time_position_replay import EVENT_FEATURES
from .learned_sequence_position_ranking import (
    LEARNED_SEQUENCE_FEATURES,
    MLPSequenceRankModel,
    _fit_scaler,
    _transform,
    attach_sequence_features,
    predict_mlp_rank,
    train_mlp_rank_model,
)
from .recent_path_position_ranking import (
    _prepare_event_frames,
    attach_rank_target,
)

REQUEST_ID = 219
MODEL_SEED = 20261128
OPTION_HORIZON_EVENTS = 5

MIN_CALIBRATION_SPEARMAN = 0.05
MIN_FRESH_SPEARMAN = 0.08
MIN_GAIN_VS_OLD_TARGET = 0.05
MIN_GOOD_DAYS = 3
MIN_TOP10_DAY_BALANCED_LIFT_PCT = 0.10
MIN_TOP20_DAY_BALANCED_LIFT_PCT = 0.05
MIN_TOP10_POSITIVE_RATE_LIFT = 0.05
MIN_SAME_AGE_MEDIAN_SPEARMAN = 0.03


@dataclass(frozen=True)
class TargetRankModel:
    model: MLPRegressor
    feature_columns: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray


def attach_option_value_target(
    frame: pd.DataFrame,
    *,
    horizon_events: int = OPTION_HORIZON_EVENTS,
) -> pd.DataFrame:
    """Attach a fixed-horizon hindsight upper bound for remaining exit option."""
    if horizon_events < 1:
        raise ValueError("horizon_events must be positive")

    result = frame.copy()
    result["option_value_5event_pct"] = np.nan
    result["option_best_future_exit_pct"] = np.nan
    result["option_horizon_last_t"] = np.nan

    keys = ["trading_day", "ticker", "hot_t"]
    for _, group in result.groupby(keys, sort=False):
        ordered = group.sort_values("state_t", kind="stable")
        indices = list(ordered.index)
        current_returns = pd.to_numeric(
            ordered["exit_now_base_return_pct"], errors="coerce"
        ).to_numpy(dtype=float)
        held = pd.to_numeric(
            ordered["minutes_held"], errors="coerce"
        ).to_numpy(dtype=float)
        times = pd.to_numeric(
            ordered["state_t"], errors="coerce"
        ).to_numpy(dtype=float)

        for position, index in enumerate(indices):
            end = position + 1 + horizon_events
            if end > len(indices):
                continue
            current = current_returns[position]
            future = current_returns[position + 1 : end]
            future_held = held[position + 1 : end]
            if (
                not np.isfinite(current)
                or len(future) != horizon_events
                or not np.isfinite(future).all()
                or not np.isfinite(future_held).all()
                or float(future_held[-1]) > MAX_HOLD_MINUTES + 1e-9
            ):
                continue

            best = float(np.max(future))
            result.at[index, "option_best_future_exit_pct"] = best
            result.at[index, "option_value_5event_pct"] = float(
                best - current
            )
            if np.isfinite(times[end - 1]):
                result.at[index, "option_horizon_last_t"] = float(
                    times[end - 1]
                )
    return result


def attach_option_rank_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Within-day percentile rank of the five-event remaining option value."""
    result = frame.copy()
    actual = pd.to_numeric(
        result["option_value_5event_pct"], errors="coerce"
    )
    result["_option_actual"] = actual
    valid = actual.notna()
    result.loc[valid, "option_value_day_rank"] = (
        result.loc[valid]
        .groupby(
            result.loc[valid, "trading_day"].astype(str),
            sort=False,
        )["_option_actual"]
        .rank(method="average", pct=True)
    )
    return result.drop(columns=["_option_actual"])


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


def train_target_rank_model(
    fit: pd.DataFrame,
    feature_family: tuple[str, ...],
) -> TargetRankModel:
    target = pd.to_numeric(
        fit["option_value_day_rank"], errors="coerce"
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].to_numpy(dtype=float)
    if len(train) < 1000:
        raise ValueError("request 219 insufficient option-rank support")

    columns = _usable_columns(train, feature_family)
    if not columns:
        raise ValueError("request 219 has no usable model features")

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
    return TargetRankModel(
        model=model,
        feature_columns=columns,
        center=center,
        scale=scale,
    )


def predict_target_rank(
    frame: pd.DataFrame,
    fitted: TargetRankModel,
) -> np.ndarray:
    x = _transform(
        frame,
        fitted.feature_columns,
        fitted.center,
        fitted.scale,
    )
    return fitted.model.predict(x)


def _age_bucket(minutes_held: pd.Series) -> pd.Series:
    held = pd.to_numeric(minutes_held, errors="coerce")
    return pd.cut(
        held,
        bins=[-np.inf, 2.0, 5.0, 10.0, 20.0, np.inf],
        labels=["<=2", "2-5", "5-10", "10-20", ">20"],
        right=True,
    )


def _top_fraction_by_day(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
) -> dict[str, object]:
    selected_parts: list[pd.DataFrame] = []
    lifts: list[float] = []
    positive_rate_lifts: list[float] = []
    by_day: dict[str, object] = {}

    days = sorted(frame["trading_day"].dropna().astype(str).unique())
    for day in days:
        part = frame.loc[
            frame["trading_day"].astype(str).eq(day)
        ].copy()
        part["_actual"] = pd.to_numeric(
            part["option_value_5event_pct"], errors="coerce"
        )
        part["_score"] = pd.to_numeric(
            part[score_column], errors="coerce"
        )
        valid = part.loc[
            part["_actual"].notna() & part["_score"].notna()
        ].copy()
        if valid.empty:
            by_day[day] = {
                "rows": 0,
                "selected": 0,
                "overall_mean_option_value_pct": None,
                "top_mean_option_value_pct": None,
                "mean_lift_pct": None,
                "overall_positive_rate": None,
                "top_positive_rate": None,
                "positive_rate_lift": None,
            }
            continue

        valid = valid.sort_values(
            "_score", ascending=False, kind="stable"
        )
        count = max(1, int(np.ceil(len(valid) * fraction)))
        top = valid.head(count).copy()
        selected_parts.append(top)

        overall_mean = float(valid["_actual"].mean())
        top_mean = float(top["_actual"].mean())
        overall_positive = float(valid["_actual"].gt(0).mean())
        top_positive = float(top["_actual"].gt(0).mean())
        lift = float(top_mean - overall_mean)
        positive_lift = float(top_positive - overall_positive)
        lifts.append(lift)
        positive_rate_lifts.append(positive_lift)

        by_day[day] = {
            "rows": int(len(valid)),
            "selected": int(len(top)),
            "overall_mean_option_value_pct": overall_mean,
            "top_mean_option_value_pct": top_mean,
            "mean_lift_pct": lift,
            "overall_positive_rate": overall_positive,
            "top_positive_rate": top_positive,
            "positive_rate_lift": positive_lift,
        }

    if not selected_parts:
        return {
            "rows": 0,
            "mean_option_value_pct": None,
            "positive_rate": None,
            "day_balanced_mean_lift_pct": None,
            "day_balanced_positive_rate_lift": None,
            "by_day": by_day,
        }

    selected = pd.concat(selected_parts, ignore_index=True)
    return {
        "rows": int(len(selected)),
        "mean_option_value_pct": float(
            selected["_actual"].mean()
        ),
        "positive_rate": float(
            selected["_actual"].gt(0).mean()
        ),
        "day_balanced_mean_lift_pct": float(np.mean(lifts)),
        "day_balanced_positive_rate_lift": float(
            np.mean(positive_rate_lifts)
        ),
        "by_day": by_day,
    }


def option_diagnostics(
    frame: pd.DataFrame,
    score_column: str,
) -> dict[str, object]:
    actual = pd.to_numeric(
        frame["option_value_5event_pct"], errors="coerce"
    )
    score = pd.to_numeric(frame[score_column], errors="coerce")
    valid = actual.notna() & score.notna()
    work = frame.loc[valid].copy()
    work["_actual"] = actual.loc[valid]
    work["_score"] = score.loc[valid]

    by_day: dict[str, object] = {}
    good_days = 0
    days = sorted(work["trading_day"].astype(str).unique())
    for day in days:
        part = work.loc[
            work["trading_day"].astype(str).eq(day)
        ]
        corr = _safe_spearman(part["_actual"], part["_score"])
        count = max(1, int(np.ceil(len(part) * 0.10)))
        top = part.sort_values(
            "_score", ascending=False, kind="stable"
        ).head(count)
        overall_mean = float(part["_actual"].mean())
        top_mean = float(top["_actual"].mean())
        lift = float(top_mean - overall_mean)
        good = bool(
            corr is not None
            and corr > 0
            and lift > 0
        )
        good_days += int(good)
        by_day[day] = {
            "rows": int(len(part)),
            "spearman": corr,
            "overall_mean_option_value_pct": overall_mean,
            "top10_mean_option_value_pct": top_mean,
            "top10_mean_lift_pct": lift,
            "top10_positive_rate": float(
                top["_actual"].gt(0).mean()
            ),
            "both_positive": good,
        }

    age = _age_bucket(work["minutes_held"])
    same_age: dict[str, object] = {}
    age_values: list[float] = []
    for label in ["<=2", "2-5", "5-10", "10-20", ">20"]:
        part = work.loc[age.astype(str).eq(label)]
        corr = (
            _safe_spearman(part["_actual"], part["_score"])
            if len(part) >= 20
            else None
        )
        if corr is not None:
            age_values.append(float(corr))
        same_age[label] = {
            "rows": int(len(part)),
            "spearman": corr,
        }

    return {
        "rows": int(len(frame)),
        "evaluable_rows": int(len(work)),
        "spearman": _safe_spearman(
            work["_actual"], work["_score"]
        ),
        "overall_mean_option_value_pct": (
            float(work["_actual"].mean()) if len(work) else None
        ),
        "overall_positive_rate": (
            float(work["_actual"].gt(0).mean())
            if len(work)
            else None
        ),
        "good_days": int(good_days),
        "by_day": by_day,
        "top10_by_day": _top_fraction_by_day(
            frame, score_column, 0.10
        ),
        "top20_by_day": _top_fraction_by_day(
            frame, score_column, 0.20
        ),
        "same_age_bucket": same_age,
        "same_age_median_spearman": (
            float(np.median(age_values))
            if age_values
            else None
        ),
    }


def _select_on_calibration(
    current_diag: dict[str, object],
    sequence_diag: dict[str, object],
) -> str:
    current = current_diag.get("spearman")
    sequence = sequence_diag.get("spearman")
    current_value = (
        float(current) if current is not None else -np.inf
    )
    sequence_value = (
        float(sequence) if sequence is not None else -np.inf
    )
    return (
        "learned_sequence"
        if sequence_value > current_value
        else "current_state"
    )


def _validate_contract() -> None:
    forbidden = {
        "next_event_t",
        "next_event_base_return_pct",
        "hold_advantage_event_pct",
        "option_value_5event_pct",
        "option_best_future_exit_pct",
        "option_horizon_last_t",
        "time_to_next_observation",
    }
    overlap = forbidden.intersection(LEARNED_SEQUENCE_FEATURES)
    if overlap:
        raise ValueError(
            f"request 219 future label leaked into features: {sorted(overlap)}"
        )


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    states_output_path: Path,
) -> int:
    _validate_contract()

    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )
    history_scan = pd.read_parquet(history_scan_path)

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    if len(position_paths) != 5 or len(scan_paths) != 5:
        raise ValueError("request 219 requires complete request178 shards")

    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )

    fit, calibration, fresh = _prepare_event_frames(
        fit_positions,
        calibration_positions,
        fresh_positions,
        history_scan,
        fresh_scan,
    )

    fit = attach_option_rank_target(
        attach_option_value_target(fit)
    )
    calibration = attach_option_rank_target(
        attach_option_value_target(calibration)
    )
    fresh = attach_option_rank_target(
        attach_option_value_target(fresh)
    )

    fit_seq = attach_sequence_features(fit)
    calibration_seq = attach_sequence_features(calibration)
    fresh_seq = attach_sequence_features(fresh)

    current_model = train_target_rank_model(
        fit_seq, EVENT_FEATURES
    )
    sequence_model = train_target_rank_model(
        fit_seq, LEARNED_SEQUENCE_FEATURES
    )

    calibration_seq["option_current_score"] = (
        predict_target_rank(calibration_seq, current_model)
    )
    calibration_seq["option_sequence_score"] = (
        predict_target_rank(calibration_seq, sequence_model)
    )
    fresh_seq["option_current_score"] = predict_target_rank(
        fresh_seq, current_model
    )
    fresh_seq["option_sequence_score"] = predict_target_rank(
        fresh_seq, sequence_model
    )

    calibration_current_diag = option_diagnostics(
        calibration_seq, "option_current_score"
    )
    calibration_sequence_diag = option_diagnostics(
        calibration_seq, "option_sequence_score"
    )
    selected = _select_on_calibration(
        calibration_current_diag,
        calibration_sequence_diag,
    )

    selected_score = (
        "option_sequence_score"
        if selected == "learned_sequence"
        else "option_current_score"
    )
    fresh_current_diag = option_diagnostics(
        fresh_seq, "option_current_score"
    )
    fresh_sequence_diag = option_diagnostics(
        fresh_seq, "option_sequence_score"
    )
    fresh_selected_diag = (
        fresh_sequence_diag
        if selected == "learned_sequence"
        else fresh_current_diag
    )

    old_fit = attach_rank_target(fit_seq.copy())
    old_model: MLPSequenceRankModel = train_mlp_rank_model(
        old_fit, LEARNED_SEQUENCE_FEATURES
    )
    fresh_seq["request218_old_target_score"] = predict_mlp_rank(
        fresh_seq, old_model
    )
    old_target_diag = option_diagnostics(
        fresh_seq, "request218_old_target_score"
    )

    selected_s = fresh_selected_diag.get("spearman")
    old_s = old_target_diag.get("spearman")
    calibration_selected_diag = (
        calibration_sequence_diag
        if selected == "learned_sequence"
        else calibration_current_diag
    )
    calibration_s = calibration_selected_diag.get("spearman")
    top10 = fresh_selected_diag["top10_by_day"]
    top20 = fresh_selected_diag["top20_by_day"]
    same_age = fresh_selected_diag.get(
        "same_age_median_spearman"
    )

    gain_old = (
        float(selected_s) - float(old_s)
        if selected_s is not None and old_s is not None
        else None
    )

    gate = bool(
        calibration_s is not None
        and float(calibration_s) >= MIN_CALIBRATION_SPEARMAN
        and selected_s is not None
        and float(selected_s) >= MIN_FRESH_SPEARMAN
        and gain_old is not None
        and float(gain_old) >= MIN_GAIN_VS_OLD_TARGET
        and int(fresh_selected_diag["good_days"])
        >= MIN_GOOD_DAYS
        and top10.get("day_balanced_mean_lift_pct") is not None
        and float(top10["day_balanced_mean_lift_pct"])
        >= MIN_TOP10_DAY_BALANCED_LIFT_PCT
        and top20.get("day_balanced_mean_lift_pct") is not None
        and float(top20["day_balanced_mean_lift_pct"])
        >= MIN_TOP20_DAY_BALANCED_LIFT_PCT
        and top10.get("day_balanced_positive_rate_lift")
        is not None
        and float(top10["day_balanced_positive_rate_lift"])
        >= MIN_TOP10_POSITIVE_RATE_LIFT
        and same_age is not None
        and float(same_age) >= MIN_SAME_AGE_MEDIAN_SPEARMAN
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "hypothesis": (
            "the recurrent HOLD bottleneck is partly caused by the "
            "one-step target; a five-executable-event remaining-option "
            "target should be materially more learnable from causal state "
            "information"
        ),
        "target": {
            "name": "five_event_remaining_option_upper_bound",
            "horizon_executable_events": OPTION_HORIZON_EVENTS,
            "definition": (
                "best BASE exit return among the next five actually "
                "observed executable POSITION states minus BASE EXIT-now"
            ),
            "requires_full_horizon": True,
            "future_used_only_for_label": True,
            "deployable_policy": False,
            "warning": (
                "This is a hindsight upper-bound diagnostic, not an "
                "oracle exit rule to deploy."
            ),
        },
        "representation": {
            "current_state_features": len(EVENT_FEATURES),
            "sequence_representation": "frozen Request218",
            "sequence_observations": 8,
            "model_family": "MLPRegressor",
            "hidden_layers": [64, 32],
            "model_seed": MODEL_SEED,
        },
        "fit_rows": int(len(fit_seq)),
        "fit_target_rows": int(
            pd.to_numeric(
                fit_seq["option_value_5event_pct"],
                errors="coerce",
            ).notna().sum()
        ),
        "calibration_rows": int(len(calibration_seq)),
        "calibration_target_rows": int(
            pd.to_numeric(
                calibration_seq["option_value_5event_pct"],
                errors="coerce",
            ).notna().sum()
        ),
        "fresh_rows": int(len(fresh_seq)),
        "fresh_target_rows": int(
            pd.to_numeric(
                fresh_seq["option_value_5event_pct"],
                errors="coerce",
            ).notna().sum()
        ),
        "calibration_model_selection": {
            "selected_representation": selected,
            "current_state": calibration_current_diag,
            "learned_sequence": calibration_sequence_diag,
        },
        "fresh_comparators": {
            "current_state_new_target": fresh_current_diag,
            "learned_sequence_new_target": fresh_sequence_diag,
            "request218_sequence_old_one_step_target": (
                old_target_diag
            ),
        },
        "fresh_selected_representation": selected,
        "fresh_selected_diagnostics": fresh_selected_diag,
        "selected_minus_old_target_spearman": gain_old,
        "frozen_gate": {
            "min_calibration_spearman": MIN_CALIBRATION_SPEARMAN,
            "min_fresh_spearman": MIN_FRESH_SPEARMAN,
            "min_gain_vs_request218_old_target": (
                MIN_GAIN_VS_OLD_TARGET
            ),
            "min_good_days": MIN_GOOD_DAYS,
            "min_top10_day_balanced_lift_pct": (
                MIN_TOP10_DAY_BALANCED_LIFT_PCT
            ),
            "min_top20_day_balanced_lift_pct": (
                MIN_TOP20_DAY_BALANCED_LIFT_PCT
            ),
            "min_top10_positive_rate_lift": (
                MIN_TOP10_POSITIVE_RATE_LIFT
            ),
            "min_same_age_median_spearman": (
                MIN_SAME_AGE_MEDIAN_SPEARMAN
            ),
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "replace hindsight max with a causal fitted continuation-value "
            "recursion, calibrate HOLD/EXIT only on fit/calibration, and "
            "replay full event-time trajectories"
        ),
        "next_boundary_if_fail": (
            "the available state is not reliably predicting even a "
            "multi-event opportunity upper bound; revisit entry-conditioned "
            "state construction and higher-resolution information before "
            "attempting another exit policy"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    states_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fresh_seq.to_parquet(
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
