"""Request 222: pullback-onset redecision action-value controller.

Request221 found a transferable signal pocket specifically at pullback onset.
Request222 stops scoring every POSITION state and asks a concrete local action
question only when a rising position first turns down:

    EXIT NOW
    versus
    WAIT for up to three actually observed events.

WAIT ends at the first causal reacceleration event, where the controller gets a
new decision point, or at the third observed event if no reacceleration appears.
The local target is the BASE mark-to-market value at that redecision/timeout
state minus EXIT-now. This is a deterministic, event-triggered action template,
not a hindsight best-price target.

The experiment is still development-only and does not claim full-trade P&L.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from .entry_conditioned_rich_second import (
    ENTRY_CONDITIONED_FEATURES,
    attach_entry_conditioned_features,
)
from .event_time_position_replay import EVENT_FEATURES
from .learned_sequence_position_ranking import (
    _fit_scaler,
    _transform,
)
from .transition_conditioned_option_value import (
    attach_transition_phase,
    _prepare_historical,
)

REQUEST_ID = 222
MODEL_SEED = 20261130
MAX_WAIT_EVENTS = 3
MAX_HOLD_MINUTES = 30.0
BOOTSTRAP_SEED = 20261222
BOOTSTRAP_SAMPLES = 10_000

MIN_TARGET_COVERAGE = 0.70
MIN_SPEARMAN = 0.05
MIN_SPEARMAN_GAIN = 0.02
MIN_WAIT_RATE = 0.10
MAX_WAIT_RATE = 0.80
MIN_GOOD_DAYS = 3


@dataclass(frozen=True)
class PullbackValueModel:
    model: MLPRegressor
    feature_columns: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    offset: float
    winsor_low: float
    winsor_high: float


def attach_pullback_wait_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach causal-template WAIT value at pullback-onset states."""
    result = frame.copy()
    result["pullback_wait_advantage_pct"] = np.nan
    result["pullback_resolution"] = None
    result["pullback_resolution_events"] = np.nan
    result["pullback_resolution_t"] = np.nan

    keys = ["trading_day", "ticker", "hot_t"]
    for _, group in result.groupby(keys, sort=False):
        ordered = group.sort_values("state_t", kind="stable")
        indices = list(ordered.index)
        phases = ordered["transition_phase"].astype(str).tolist()
        exits = pd.to_numeric(
            ordered["exit_now_base_return_pct"], errors="coerce"
        ).to_numpy(dtype=float)
        held = pd.to_numeric(
            ordered["minutes_held"], errors="coerce"
        ).to_numpy(dtype=float)
        times = pd.to_numeric(
            ordered["state_t"], errors="coerce"
        ).to_numpy(dtype=float)

        for pos, index in enumerate(indices):
            if phases[pos] != "pullback_onset":
                continue
            current_exit = exits[pos]
            if not np.isfinite(current_exit):
                continue

            available = len(indices) - pos - 1
            if available <= 0:
                continue

            target_pos: int | None = None
            reason: str | None = None
            for step in range(1, min(MAX_WAIT_EVENTS, available) + 1):
                future_pos = pos + step
                if phases[future_pos] == "reacceleration":
                    target_pos = future_pos
                    reason = "reacceleration_redecision"
                    break

            if target_pos is None:
                if available < MAX_WAIT_EVENTS:
                    continue
                target_pos = pos + MAX_WAIT_EVENTS
                reason = "three_event_timeout"

            target_exit = exits[target_pos]
            target_held = held[target_pos]
            if (
                not np.isfinite(target_exit)
                or not np.isfinite(target_held)
                or float(target_held) > MAX_HOLD_MINUTES + 1e-9
            ):
                continue

            result.at[index, "pullback_wait_advantage_pct"] = float(
                target_exit - current_exit
            )
            result.at[index, "pullback_resolution"] = reason
            result.at[index, "pullback_resolution_events"] = int(
                target_pos - pos
            )
            if np.isfinite(times[target_pos]):
                result.at[index, "pullback_resolution_t"] = float(
                    times[target_pos]
                )
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


def _day_balanced_offset(
    frame: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    residual = pd.DataFrame(
        {
            "day": frame["trading_day"].astype(str).to_numpy(),
            "residual": actual - predicted,
        }
    )
    daily = residual.groupby("day", sort=True)["residual"].mean()
    return float(daily.mean()) if len(daily) else 0.0


def train_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_family: tuple[str, ...],
) -> PullbackValueModel:
    fit_target = pd.to_numeric(
        fit["pullback_wait_advantage_pct"], errors="coerce"
    )
    train = fit.loc[fit_target.notna()].copy()
    y = fit_target.loc[fit_target.notna()].to_numpy(dtype=float)
    if len(train) < 500:
        raise ValueError(
            f"request 222 insufficient fit pullback support: {len(train)}"
        )

    columns = _usable_columns(train, feature_family)
    center, scale = _fit_scaler(train, columns)
    x = _transform(train, columns, center, scale)
    low, high = np.quantile(y, [0.005, 0.995])

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
    model.fit(x, np.clip(y, low, high))

    cal_target = pd.to_numeric(
        calibration["pullback_wait_advantage_pct"],
        errors="coerce",
    )
    cal = calibration.loc[cal_target.notna()].copy()
    actual = cal_target.loc[cal_target.notna()].to_numpy(dtype=float)
    if len(cal) < 200:
        raise ValueError(
            f"request 222 insufficient calibration support: {len(cal)}"
        )
    raw = model.predict(_transform(cal, columns, center, scale))
    offset = _day_balanced_offset(cal, actual, raw)

    return PullbackValueModel(
        model=model,
        feature_columns=columns,
        center=center,
        scale=scale,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_model(
    frame: pd.DataFrame,
    fitted: PullbackValueModel,
) -> np.ndarray:
    return (
        fitted.model.predict(
            _transform(
                frame,
                fitted.feature_columns,
                fitted.center,
                fitted.scale,
            )
        )
        + fitted.offset
    )


def _safe_spearman(
    actual: pd.Series,
    predicted: pd.Series,
) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna()
    if int(valid.sum()) < 20:
        return None
    value = a.loc[valid].corr(p.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _bootstrap_daily(
    daily: np.ndarray,
    seed: int,
) -> dict[str, object]:
    values = np.asarray(daily, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {
            "days": int(len(values)),
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": float(values.mean()) if len(values) else None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_SAMPLES, len(values)),
    )
    draws = values[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(values)),
        "samples": BOOTSTRAP_SAMPLES,
        "mean_pct": float(values.mean()),
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def action_diagnostics(
    frame: pd.DataFrame,
    score_column: str,
) -> dict[str, object]:
    actual = pd.to_numeric(
        frame["pullback_wait_advantage_pct"], errors="coerce"
    )
    predicted = pd.to_numeric(frame[score_column], errors="coerce")
    valid = actual.notna() & predicted.notna()
    work = frame.loc[valid].copy()
    work["_actual"] = actual.loc[valid]
    work["_pred"] = predicted.loc[valid]
    work["_wait"] = work["_pred"].gt(0)
    work["_policy_uplift"] = np.where(
        work["_wait"],
        work["_actual"],
        0.0,
    )

    selected = work.loc[work["_wait"]].copy()
    daily_policy = (
        work.groupby(
            work["trading_day"].astype(str), sort=True
        )["_policy_uplift"]
        .mean()
    )
    daily_wait = (
        work.groupby(
            work["trading_day"].astype(str), sort=True
        )["_actual"]
        .mean()
    )

    by_day: dict[str, object] = {}
    good_days = 0
    for day, part in work.groupby(
        work["trading_day"].astype(str),
        sort=True,
    ):
        corr = _safe_spearman(part["_actual"], part["_pred"])
        wait = part.loc[part["_wait"]]
        policy_mean = float(part["_policy_uplift"].mean())
        selected_mean = (
            float(wait["_actual"].mean()) if len(wait) else None
        )
        good = bool(
            corr is not None
            and corr > 0
            and policy_mean > 0
            and selected_mean is not None
            and selected_mean > 0
        )
        good_days += int(good)
        by_day[str(day)] = {
            "rows": int(len(part)),
            "spearman": corr,
            "predicted_wait_rate": float(part["_wait"].mean()),
            "selected_wait_advantage_mean_pct": selected_mean,
            "policy_uplift_mean_pct": policy_mean,
            "always_wait_advantage_mean_pct": float(
                part["_actual"].mean()
            ),
            "both_positive": good,
        }

    reasons = {
        str(k): int(v)
        for k, v in work["pullback_resolution"]
        .value_counts(dropna=False)
        .to_dict()
        .items()
    }

    return {
        "rows": int(len(frame)),
        "evaluable_rows": int(len(work)),
        "target_coverage": (
            float(len(work) / len(frame)) if len(frame) else None
        ),
        "spearman": _safe_spearman(work["_actual"], work["_pred"]),
        "predicted_wait_rows": int(work["_wait"].sum()),
        "predicted_wait_rate": (
            float(work["_wait"].mean()) if len(work) else None
        ),
        "selected_wait_advantage_mean_pct": (
            float(selected["_actual"].mean())
            if len(selected)
            else None
        ),
        "selected_wait_advantage_p05_pct": (
            float(selected["_actual"].quantile(0.05))
            if len(selected)
            else None
        ),
        "policy_uplift_mean_pct": (
            float(work["_policy_uplift"].mean())
            if len(work)
            else None
        ),
        "policy_day_balanced_uplift_pct": (
            float(daily_policy.mean())
            if len(daily_policy)
            else None
        ),
        "policy_day_bootstrap": _bootstrap_daily(
            daily_policy.to_numpy(dtype=float),
            BOOTSTRAP_SEED,
        ),
        "always_wait_day_balanced_advantage_pct": (
            float(daily_wait.mean()) if len(daily_wait) else None
        ),
        "good_days": int(good_days),
        "resolution_reasons": reasons,
        "by_day": by_day,
    }


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    history_scan_path: Path,
    fresh_states_path: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )
    history_scan = pd.read_parquet(history_scan_path)
    fresh = pd.read_parquet(fresh_states_path).copy()

    fit, calibration = _prepare_historical(
        fit_positions,
        calibration_positions,
        history_scan,
    )
    fit = attach_pullback_wait_target(
        attach_entry_conditioned_features(
            attach_transition_phase(fit)
        )
    )
    calibration = attach_pullback_wait_target(
        attach_entry_conditioned_features(
            attach_transition_phase(calibration)
        )
    )
    fresh = attach_pullback_wait_target(fresh)

    fit_pullback = fit.loc[
        fit["transition_phase"].astype(str).eq("pullback_onset")
    ].copy()
    cal_pullback = calibration.loc[
        calibration["transition_phase"].astype(str).eq(
            "pullback_onset"
        )
    ].copy()
    fresh_pullback = fresh.loc[
        fresh["transition_phase"].astype(str).eq("pullback_onset")
    ].copy()

    baseline_model = train_model(
        fit_pullback,
        cal_pullback,
        EVENT_FEATURES,
    )
    candidate_model = train_model(
        fit_pullback,
        cal_pullback,
        ENTRY_CONDITIONED_FEATURES,
    )

    cal_pullback["baseline_wait_score"] = predict_model(
        cal_pullback, baseline_model
    )
    cal_pullback["candidate_wait_score"] = predict_model(
        cal_pullback, candidate_model
    )
    fresh_pullback["baseline_wait_score"] = predict_model(
        fresh_pullback, baseline_model
    )
    fresh_pullback["candidate_wait_score"] = predict_model(
        fresh_pullback, candidate_model
    )

    calibration_baseline = action_diagnostics(
        cal_pullback, "baseline_wait_score"
    )
    calibration_candidate = action_diagnostics(
        cal_pullback, "candidate_wait_score"
    )
    fresh_baseline = action_diagnostics(
        fresh_pullback, "baseline_wait_score"
    )
    fresh_candidate = action_diagnostics(
        fresh_pullback, "candidate_wait_score"
    )

    base_s = fresh_baseline.get("spearman")
    candidate_s = fresh_candidate.get("spearman")
    gain = (
        float(candidate_s) - float(base_s)
        if candidate_s is not None and base_s is not None
        else None
    )
    wait_rate = fresh_candidate.get("predicted_wait_rate")
    selected_mean = fresh_candidate.get(
        "selected_wait_advantage_mean_pct"
    )
    day_uplift = fresh_candidate.get(
        "policy_day_balanced_uplift_pct"
    )
    bootstrap_low = fresh_candidate[
        "policy_day_bootstrap"
    ].get("ci_low_pct")
    base_day_uplift = fresh_baseline.get(
        "policy_day_balanced_uplift_pct"
    )

    gate = bool(
        fresh_candidate.get("target_coverage") is not None
        and float(fresh_candidate["target_coverage"])
        >= MIN_TARGET_COVERAGE
        and candidate_s is not None
        and float(candidate_s) >= MIN_SPEARMAN
        and gain is not None
        and float(gain) >= MIN_SPEARMAN_GAIN
        and wait_rate is not None
        and MIN_WAIT_RATE <= float(wait_rate) <= MAX_WAIT_RATE
        and selected_mean is not None
        and float(selected_mean) > 0
        and day_uplift is not None
        and float(day_uplift) > 0
        and base_day_uplift is not None
        and float(day_uplift) > float(base_day_uplift)
        and bootstrap_low is not None
        and float(bootstrap_low) > 0
        and int(fresh_candidate["good_days"]) >= MIN_GOOD_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "event_scope": "pullback_onset only",
        "action_template": {
            "exit_now": "realize BASE return at the pullback-onset state",
            "wait": (
                "wait until first causal reacceleration within the next "
                "three observed events; if none appears, exit at the "
                "third observed event"
            ),
            "reacceleration_is_redecision_not_final_policy_exit": True,
            "future_information_used_as_input": False,
        },
        "target": (
            "BASE value at deterministic redecision/timeout state minus "
            "BASE EXIT-now"
        ),
        "fit_pullback_rows": int(len(fit_pullback)),
        "calibration_pullback_rows": int(len(cal_pullback)),
        "fresh_pullback_rows": int(len(fresh_pullback)),
        "models": {
            "baseline_feature_count": len(
                baseline_model.feature_columns
            ),
            "candidate_feature_count": len(
                candidate_model.feature_columns
            ),
            "family": "MLPRegressor",
            "hidden_layers": [64, 32],
            "seed": MODEL_SEED,
        },
        "calibration": {
            "baseline": calibration_baseline,
            "entry_conditioned": calibration_candidate,
        },
        "fresh": {
            "baseline": fresh_baseline,
            "entry_conditioned": fresh_candidate,
        },
        "fresh_candidate_minus_baseline_spearman": gain,
        "frozen_gate": {
            "min_target_coverage": MIN_TARGET_COVERAGE,
            "min_candidate_spearman": MIN_SPEARMAN,
            "min_spearman_gain_vs_current_state": MIN_SPEARMAN_GAIN,
            "min_wait_rate": MIN_WAIT_RATE,
            "max_wait_rate": MAX_WAIT_RATE,
            "selected_wait_advantage_must_be_positive": True,
            "policy_day_balanced_uplift_must_be_positive": True,
            "policy_must_beat_current_state_baseline": True,
            "policy_bootstrap_ci_low_must_be_positive": True,
            "min_good_days": MIN_GOOD_DAYS,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "integrate this pullback decision as one node in a recurrent "
            "event-time controller; at reacceleration, hand control to a "
            "continuation/exit value model."
        ),
        "next_boundary_if_fail": (
            "the pullback pocket ranks long-horizon opportunity but does "
            "not support this deterministic wait action; audit pullback "
            "depth/duration and define the redecision event from learned "
            "hazard rather than a fixed three-event timeout."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fresh_pullback.to_parquet(
        rows_output_path,
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
    parser.add_argument("--fresh-states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.history_scan,
        args.fresh_states,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
