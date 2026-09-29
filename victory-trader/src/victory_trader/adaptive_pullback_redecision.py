"""Request 223: adaptive pullback redecision value.

Request221 found a transferable information pocket at pullback onset. Request222
then showed that a fixed "wait up to three observed events" action is not
transferable. Request223 removes that fixed timeout and lets the controller
reconsider WAIT versus EXIT at every actually observed state inside the
pullback.

A pullback segment begins at a causal pullback_onset and ends at:
* the first causal reacceleration event, which becomes a new decision point; or
* the first observed state at/after the 30-minute risk cap.

For every pre-terminal state, the supervised target is terminal BASE value
minus EXIT-now. Future outcomes are used only as labels. The deployed-style
local policy itself uses only the current score: WAIT while predicted
continuation value is positive, otherwise EXIT.

This remains a development-only local controller diagnostic. Reacceleration is
a handoff point, not a final full-trade exit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from .entry_conditioned_rich_second import ENTRY_CONDITIONED_FEATURES
from .event_time_position_replay import EVENT_FEATURES
from .learned_sequence_position_ranking import _fit_scaler, _transform
from .pullback_redecision_controller import (
    BOOTSTRAP_SAMPLES,
    MAX_HOLD_MINUTES,
)
from .transition_conditioned_option_value import (
    _prepare_historical,
    attach_entry_conditioned_features,
    attach_transition_phase,
)

REQUEST_ID = 223
MODEL_SEED = 20261223
BOOTSTRAP_SEED = 20261223

CONTEXT_FEATURES = (
    "pullback_age_events",
    "pullback_elapsed_min",
    "pullback_return_from_onset_pct",
    "pullback_worst_from_onset_pct",
    "pullback_recovery_from_worst_pct",
)
BASELINE_FEATURES = tuple(
    dict.fromkeys([*EVENT_FEATURES, *CONTEXT_FEATURES])
)
CANDIDATE_FEATURES = tuple(
    dict.fromkeys([*ENTRY_CONDITIONED_FEATURES, *CONTEXT_FEATURES])
)

MIN_SEGMENT_COVERAGE = 0.75
MIN_TARGET_STATE_COVERAGE = 0.75
MIN_WAIT_ACTION_RATE = 0.10
MAX_WAIT_ACTION_RATE = 0.90
MIN_VALUE_SPEARMAN = 0.02
MIN_GOOD_DAYS = 3


@dataclass(frozen=True)
class AdaptiveValueModel:
    model: MLPRegressor
    feature_columns: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    offset: float
    winsor_low: float
    winsor_high: float


def build_pullback_wait_states(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create causal pullback segments with variable redecision horizons."""
    records: list[dict[str, object]] = []
    segment_total = 0
    segment_resolved = 0
    raw_wait_states = 0

    keys = ["trading_day", "ticker", "hot_t"]
    for episode_key, group in frame.groupby(keys, sort=False):
        ordered = group.sort_values("state_t", kind="stable").copy()
        ordered = ordered.reset_index(drop=True)
        phases = ordered["transition_phase"].astype(str).tolist()
        held = pd.to_numeric(
            ordered["minutes_held"], errors="coerce"
        ).to_numpy(dtype=float)
        exits = pd.to_numeric(
            ordered["exit_now_base_return_pct"], errors="coerce"
        ).to_numpy(dtype=float)
        times = pd.to_numeric(
            ordered["state_t"], errors="coerce"
        ).to_numpy(dtype=float)

        for onset_pos, phase in enumerate(phases):
            if phase != "pullback_onset":
                continue
            segment_total += 1

            terminal_pos: int | None = None
            terminal_reason: str | None = None
            for pos in range(onset_pos + 1, len(ordered)):
                if phases[pos] == "reacceleration":
                    terminal_pos = pos
                    terminal_reason = "reacceleration_redecision"
                    break
                if np.isfinite(held[pos]) and held[pos] >= MAX_HOLD_MINUTES:
                    terminal_pos = pos
                    terminal_reason = "forced_30m_cap"
                    break

            if terminal_pos is None:
                continue
            terminal_exit = exits[terminal_pos]
            terminal_t = times[terminal_pos]
            if not np.isfinite(terminal_exit) or not np.isfinite(terminal_t):
                continue
            segment_resolved += 1

            onset_exit = exits[onset_pos]
            onset_t = times[onset_pos]
            if not np.isfinite(onset_exit) or not np.isfinite(onset_t):
                continue

            worst_from_onset = 0.0
            segment_id = (
                f"{episode_key[0]}|{str(episode_key[1]).upper()}|"
                f"{int(episode_key[2])}|{int(onset_t)}"
            )

            for pos in range(onset_pos, terminal_pos):
                current_exit = exits[pos]
                current_t = times[pos]
                if not np.isfinite(current_exit) or not np.isfinite(current_t):
                    continue

                delta = float(current_exit - onset_exit)
                worst_from_onset = min(worst_from_onset, delta)
                recovery = float(delta - worst_from_onset)
                raw_wait_states += 1

                item = ordered.iloc[pos].to_dict()
                item.update(
                    {
                        "pullback_segment_id": segment_id,
                        "pullback_onset_t": int(onset_t),
                        "pullback_terminal_t": int(terminal_t),
                        "pullback_terminal_reason": terminal_reason,
                        "pullback_terminal_exit_pct": float(terminal_exit),
                        "pullback_onset_exit_pct": float(onset_exit),
                        "pullback_age_events": int(pos - onset_pos),
                        "pullback_elapsed_min": float(
                            (current_t - onset_t) / 60_000.0
                        ),
                        "pullback_return_from_onset_pct": delta,
                        "pullback_worst_from_onset_pct": float(
                            worst_from_onset
                        ),
                        "pullback_recovery_from_worst_pct": recovery,
                        "adaptive_wait_value_pct": float(
                            terminal_exit - current_exit
                        ),
                    }
                )
                records.append(item)

    states = pd.DataFrame(records)
    summary = {
        "pullback_segments_total": int(segment_total),
        "pullback_segments_resolved": int(segment_resolved),
        "segment_coverage": (
            float(segment_resolved / segment_total)
            if segment_total
            else None
        ),
        "wait_states": int(raw_wait_states),
    }
    return states, summary


def _usable_columns(
    frame: pd.DataFrame,
    feature_family: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        column
        for column in feature_family
        if column in frame.columns
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
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


def train_value_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_family: tuple[str, ...],
) -> AdaptiveValueModel:
    target = pd.to_numeric(
        fit["adaptive_wait_value_pct"], errors="coerce"
    )
    train = fit.loc[target.notna()].copy()
    y = target.loc[target.notna()].to_numpy(dtype=float)
    if len(train) < 1000:
        raise ValueError(
            f"request 223 insufficient fit wait states: {len(train)}"
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
        calibration["adaptive_wait_value_pct"], errors="coerce"
    )
    cal = calibration.loc[cal_target.notna()].copy()
    actual = cal_target.loc[cal_target.notna()].to_numpy(dtype=float)
    if len(cal) < 500:
        raise ValueError(
            f"request 223 insufficient calibration wait states: {len(cal)}"
        )
    raw = model.predict(_transform(cal, columns, center, scale))
    offset = _day_balanced_offset(cal, actual, raw)

    return AdaptiveValueModel(
        model=model,
        feature_columns=columns,
        center=center,
        scale=scale,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_value(
    frame: pd.DataFrame,
    fitted: AdaptiveValueModel,
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
    values: np.ndarray,
    seed: int,
) -> dict[str, object]:
    daily = np.asarray(values, dtype=float)
    daily = daily[np.isfinite(daily)]
    if len(daily) < 2:
        return {
            "days": int(len(daily)),
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": float(daily.mean()) if len(daily) else None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(
        0,
        len(daily),
        size=(BOOTSTRAP_SAMPLES, len(daily)),
    )
    draws = daily[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(daily)),
        "samples": BOOTSTRAP_SAMPLES,
        "mean_pct": float(daily.mean()),
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def state_diagnostics(
    states: pd.DataFrame,
    score_column: str,
) -> dict[str, object]:
    actual = pd.to_numeric(
        states["adaptive_wait_value_pct"], errors="coerce"
    )
    score = pd.to_numeric(states[score_column], errors="coerce")
    valid = actual.notna() & score.notna()
    work = states.loc[valid].copy()
    work["_actual"] = actual.loc[valid]
    work["_score"] = score.loc[valid]
    return {
        "rows": int(len(states)),
        "evaluable_rows": int(len(work)),
        "spearman": _safe_spearman(work["_actual"], work["_score"]),
        "wait_action_rate": (
            float(work["_score"].gt(0).mean()) if len(work) else None
        ),
        "selected_wait_value_mean_pct": (
            float(work.loc[work["_score"].gt(0), "_actual"].mean())
            if work["_score"].gt(0).any()
            else None
        ),
    }


def simulate_adaptive_policy(
    states: pd.DataFrame,
    score_column: str,
    *,
    policy: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for segment_id, group in states.groupby(
        "pullback_segment_id", sort=False
    ):
        work = group.sort_values("state_t", kind="stable").copy()
        first = work.iloc[0]
        onset_exit = float(first["pullback_onset_exit_pct"])
        terminal_exit = float(first["pullback_terminal_exit_pct"])
        terminal_t = int(first["pullback_terminal_t"])
        terminal_reason = str(first["pullback_terminal_reason"])

        exit_return = terminal_exit
        exit_t = terminal_t
        reason = terminal_reason
        waits = 0

        for _, row in work.iterrows():
            score = pd.to_numeric(
                pd.Series([row.get(score_column)]),
                errors="coerce",
            ).iloc[0]
            current_exit = pd.to_numeric(
                pd.Series([row.get("exit_now_base_return_pct")]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(score) or pd.isna(current_exit):
                exit_return = float(current_exit) if pd.notna(current_exit) else onset_exit
                exit_t = int(row["state_t"])
                reason = "unscoreable_exit"
                break
            if float(score) <= 0:
                exit_return = float(current_exit)
                exit_t = int(row["state_t"])
                reason = "model_exit"
                break
            waits += 1

        records.append(
            {
                "pullback_segment_id": str(segment_id),
                "trading_day": str(first["trading_day"]),
                "ticker": str(first["ticker"]).upper(),
                "hot_t": int(first["hot_t"]),
                "policy": policy,
                "local_uplift_pct": float(exit_return - onset_exit),
                "exit_return_pct": float(exit_return),
                "onset_exit_pct": onset_exit,
                "exit_t": int(exit_t),
                "wait_decisions": int(waits),
                "exit_reason": reason,
            }
        )
    return pd.DataFrame(records)


def fixed_three_comparator(
    full_frame: pd.DataFrame,
) -> pd.DataFrame:
    from .pullback_redecision_controller import attach_pullback_wait_target

    marked = attach_pullback_wait_target(full_frame)
    onset = marked.loc[
        marked["transition_phase"].astype(str).eq("pullback_onset")
    ].copy()
    onset = onset.loc[
        pd.to_numeric(
            onset["pullback_wait_advantage_pct"], errors="coerce"
        ).notna()
    ].copy()
    onset["pullback_segment_id"] = (
        onset["trading_day"].astype(str)
        + "|"
        + onset["ticker"].astype(str).str.upper()
        + "|"
        + onset["hot_t"].astype(int).astype(str)
        + "|"
        + onset["state_t"].astype("int64").astype(str)
    )
    return pd.DataFrame(
        {
            "pullback_segment_id": onset["pullback_segment_id"],
            "trading_day": onset["trading_day"].astype(str),
            "ticker": onset["ticker"].astype(str).str.upper(),
            "hot_t": onset["hot_t"].astype(int),
            "policy": "request222_fixed3_wait",
            "local_uplift_pct": pd.to_numeric(
                onset["pullback_wait_advantage_pct"],
                errors="coerce",
            ),
        }
    )


def policy_metrics(
    policy_rows: pd.DataFrame,
) -> dict[str, object]:
    if policy_rows.empty:
        return {"segments": 0}
    values = pd.to_numeric(
        policy_rows["local_uplift_pct"], errors="coerce"
    )
    valid = values.notna()
    work = policy_rows.loc[valid].copy()
    work["_value"] = values.loc[valid]
    daily = work.groupby(
        work["trading_day"].astype(str), sort=True
    )["_value"].mean()
    by_day = {
        str(day): {
            "segments": int(len(part)),
            "mean_uplift_pct": float(part["_value"].mean()),
        }
        for day, part in work.groupby(
            work["trading_day"].astype(str), sort=True
        )
    }
    good_days = int(sum(v["mean_uplift_pct"] > 0 for v in by_day.values()))
    return {
        "segments": int(len(work)),
        "mean_uplift_pct": float(work["_value"].mean()),
        "median_uplift_pct": float(work["_value"].median()),
        "p05_uplift_pct": float(work["_value"].quantile(0.05)),
        "positive_rate": float(work["_value"].gt(0).mean()),
        "day_balanced_uplift_pct": float(daily.mean()),
        "day_bootstrap": _bootstrap_daily(
            daily.to_numpy(dtype=float),
            BOOTSTRAP_SEED,
        ),
        "good_days": good_days,
        "by_day": by_day,
    }


def matched_difference(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
) -> dict[str, object]:
    left = candidate.loc[
        :, ["pullback_segment_id", "trading_day", "local_uplift_pct"]
    ].copy()
    right = comparator.loc[
        :, ["pullback_segment_id", "local_uplift_pct"]
    ].copy()
    merged = left.merge(
        right,
        on="pullback_segment_id",
        suffixes=("_candidate", "_comparator"),
        validate="one_to_one",
    )
    if merged.empty:
        return {
            "matched_segments": 0,
            "day_balanced_difference_pct": None,
            "bootstrap": _bootstrap_daily(
                np.asarray([], dtype=float), BOOTSTRAP_SEED + 1
            ),
        }
    merged["difference_pct"] = (
        pd.to_numeric(
            merged["local_uplift_pct_candidate"], errors="coerce"
        )
        - pd.to_numeric(
            merged["local_uplift_pct_comparator"], errors="coerce"
        )
    )
    daily = merged.groupby(
        merged["trading_day"].astype(str), sort=True
    )["difference_pct"].mean()
    return {
        "matched_segments": int(len(merged)),
        "mean_difference_pct": float(merged["difference_pct"].mean()),
        "candidate_better_rate": float(
            merged["difference_pct"].gt(0).mean()
        ),
        "day_balanced_difference_pct": float(daily.mean()),
        "bootstrap": _bootstrap_daily(
            daily.to_numpy(dtype=float), BOOTSTRAP_SEED + 1
        ),
    }


def _choose_representation_on_calibration(
    baseline_policy: dict[str, object],
    candidate_policy: dict[str, object],
) -> str:
    base = baseline_policy.get("day_balanced_uplift_pct")
    cand = candidate_policy.get("day_balanced_uplift_pct")
    base_value = float(base) if base is not None else -np.inf
    cand_value = float(cand) if cand is not None else -np.inf
    return "entry_conditioned" if cand_value > base_value else "current_state"


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
    fresh_full = pd.read_parquet(fresh_states_path).copy()

    fit_full, calibration_full = _prepare_historical(
        fit_positions,
        calibration_positions,
        history_scan,
    )
    fit_full = attach_entry_conditioned_features(
        attach_transition_phase(fit_full)
    )
    calibration_full = attach_entry_conditioned_features(
        attach_transition_phase(calibration_full)
    )
    if "transition_phase" not in fresh_full.columns:
        fresh_full = attach_transition_phase(fresh_full)

    fit_states, fit_support = build_pullback_wait_states(fit_full)
    cal_states, cal_support = build_pullback_wait_states(calibration_full)
    fresh_states, fresh_support = build_pullback_wait_states(fresh_full)

    baseline_model = train_value_model(
        fit_states, cal_states, BASELINE_FEATURES
    )
    candidate_model = train_value_model(
        fit_states, cal_states, CANDIDATE_FEATURES
    )

    for states in (cal_states, fresh_states):
        states["current_state_score"] = predict_value(
            states, baseline_model
        )
        states["entry_conditioned_score"] = predict_value(
            states, candidate_model
        )

    cal_baseline_policy_rows = simulate_adaptive_policy(
        cal_states,
        "current_state_score",
        policy="adaptive_current_state",
    )
    cal_candidate_policy_rows = simulate_adaptive_policy(
        cal_states,
        "entry_conditioned_score",
        policy="adaptive_entry_conditioned",
    )
    cal_baseline_policy = policy_metrics(cal_baseline_policy_rows)
    cal_candidate_policy = policy_metrics(cal_candidate_policy_rows)
    selected = _choose_representation_on_calibration(
        cal_baseline_policy, cal_candidate_policy
    )
    fresh_baseline_policy_rows = simulate_adaptive_policy(
        fresh_states,
        "current_state_score",
        policy="adaptive_current_state",
    )
    fresh_candidate_policy_rows = simulate_adaptive_policy(
        fresh_states,
        "entry_conditioned_score",
        policy="adaptive_entry_conditioned",
    )
    selected_rows = (
        fresh_candidate_policy_rows
        if selected == "entry_conditioned"
        else fresh_baseline_policy_rows
    )

    fixed3_rows = fixed_three_comparator(fresh_full)
    fresh_baseline_policy = policy_metrics(
        fresh_baseline_policy_rows
    )
    fresh_candidate_policy = policy_metrics(
        fresh_candidate_policy_rows
    )
    selected_policy = (
        fresh_candidate_policy
        if selected == "entry_conditioned"
        else fresh_baseline_policy
    )
    fixed3_policy = policy_metrics(fixed3_rows)
    difference = matched_difference(selected_rows, fixed3_rows)

    cal_baseline_state = state_diagnostics(
        cal_states, "current_state_score"
    )
    cal_candidate_state = state_diagnostics(
        cal_states, "entry_conditioned_score"
    )
    fresh_baseline_state = state_diagnostics(
        fresh_states, "current_state_score"
    )
    fresh_candidate_state = state_diagnostics(
        fresh_states, "entry_conditioned_score"
    )
    selected_state = (
        fresh_candidate_state
        if selected == "entry_conditioned"
        else fresh_baseline_state
    )

    segment_coverage = fresh_support.get("segment_coverage")
    state_coverage = selected_state.get("evaluable_rows")
    state_coverage = (
        float(state_coverage / len(fresh_states))
        if len(fresh_states)
        else None
    )
    wait_rate = selected_state.get("wait_action_rate")
    spearman = selected_state.get("spearman")
    policy_day = selected_policy.get("day_balanced_uplift_pct")
    policy_low = selected_policy.get("day_bootstrap", {}).get(
        "ci_low_pct"
    )
    diff_day = difference.get("day_balanced_difference_pct")
    diff_low = difference.get("bootstrap", {}).get("ci_low_pct")

    gate = bool(
        segment_coverage is not None
        and float(segment_coverage) >= MIN_SEGMENT_COVERAGE
        and state_coverage is not None
        and float(state_coverage) >= MIN_TARGET_STATE_COVERAGE
        and wait_rate is not None
        and MIN_WAIT_ACTION_RATE
        <= float(wait_rate)
        <= MAX_WAIT_ACTION_RATE
        and spearman is not None
        and float(spearman) >= MIN_VALUE_SPEARMAN
        and policy_day is not None
        and float(policy_day) > 0
        and policy_low is not None
        and float(policy_low) > 0
        and int(selected_policy.get("good_days", 0)) >= MIN_GOOD_DAYS
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "scope": "adaptive management inside pullback segments",
        "terminal_rule": (
            "first causal reacceleration, otherwise first observed state "
            "at/after the 30-minute risk cap"
        ),
        "action_rule": (
            "at each observed pullback state, WAIT iff predicted terminal "
            "continuation value > 0; otherwise EXIT"
        ),
        "reacceleration_is_redecision_point": True,
        "fixed_wait_horizon_used": False,
        "context_features": list(CONTEXT_FEATURES),
        "support": {
            "fit": fit_support,
            "calibration": cal_support,
            "fresh": fresh_support,
        },
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
            "current_state": {
                "state": cal_baseline_state,
                "policy": cal_baseline_policy,
            },
            "entry_conditioned": {
                "state": cal_candidate_state,
                "policy": cal_candidate_policy,
            },
            "selected_representation": selected,
        },
        "fresh": {
            "current_state": {
                "state": fresh_baseline_state,
                "policy": fresh_baseline_policy,
            },
            "entry_conditioned": {
                "state": fresh_candidate_state,
                "policy": fresh_candidate_policy,
            },
            "selected_representation": selected,
            "selected_state": selected_state,
            "selected_policy": selected_policy,
            "request222_fixed3": fixed3_policy,
            "selected_minus_fixed3": difference,
        },
        "frozen_gate": {
            "min_segment_coverage": MIN_SEGMENT_COVERAGE,
            "min_target_state_coverage": MIN_TARGET_STATE_COVERAGE,
            "min_wait_action_rate": MIN_WAIT_ACTION_RATE,
            "max_wait_action_rate": MAX_WAIT_ACTION_RATE,
            "min_value_spearman": MIN_VALUE_SPEARMAN,
            "selected_policy_day_balanced_uplift_positive": True,
            "selected_policy_bootstrap_ci_low_positive": True,
            "min_good_days": MIN_GOOD_DAYS,
            "selected_must_beat_request222_fixed3": True,
            "matched_difference_bootstrap_ci_low_positive": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze the adaptive pullback node and connect its "
            "reacceleration handoff to a continuation/exit controller, "
            "then replay complete Request208-entry trajectories."
        ),
        "next_boundary_if_fail": (
            "stop fitting generic pullback value over all pullbacks; "
            "audit which pullback geometries carry the Request221 "
            "opportunity signal and restrict decisions to that event subset."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fresh_states.to_parquet(
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
