"""Request 226: entry survivability gate before POSITION management.

Request225 showed that one additional post-entry observation makes pullback
quality substantially more predictable, but paying for that information while
already in the position is economically too late. Request226 moves the question
upstream.

For each Request208-selected entry, define the first post-entry stress event as:
* the first observed state whose BASE return is below the previous observed
  state (entry itself is return 0); or
* if no such adverse step occurs first, the first observed state at/after the
  30-minute risk cap.

The supervised target is the BASE return at that first stress event. Inputs are
only the causal Request208 WATCH state at the selected entry. A positive target
means the entry built/retained positive cushion through its first adverse
observation.

The candidate policy is intentionally simple and frozen:
* Request208 proposes an entry;
* admit it only when calibrated predicted first-stress return > 0;
* otherwise ABSTAIN from that episode.

Fresh evaluation compares the candidate with the frozen Request208 fixed-3m
trade on the exact same proposed entries. No new dates are opened.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .attention_replay import MINUTE_MS
from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .event_time_position_replay import reanchor_event_positions
from .integrated_entry_hold_exit import choose_entries, score_entry_states
from .pullback_turn_transition_entry import (
    MODEL_FEATURES,
    attach_transition_features as attach_watch_transitions,
    predict as predict_entry,
    train_model as train_entry_model,
)
from .recurrent_wait_entry_action_value import build_watch_states
from .relative_recurrent_entry_timing import attach_relative_advantage
from .state_action_risk import SEVERE_LOSS_PCT
from .wait_reobserve_entry_confirmation import _day_weights, _feature_frame

REQUEST_ID = 226
MODEL_SEED = 20261226
BOOTSTRAP_SEED = 20261226
BOOTSTRAP_SAMPLES = 10_000
MAX_HOLD_MINUTES = 30.0
EPSILON = 1e-9

ENTRY_SURVIVABILITY_FEATURES = tuple(
    dict.fromkeys(
        [
            *MODEL_FEATURES,
            "predicted_relative_advantage_pct",
            "minutes_since_hot",
        ]
    )
)

MIN_LABEL_COVERAGE = 0.80
MIN_STRESS_SPEARMAN = 0.08
MIN_ADMITTED_RATE = 0.15
MAX_ADMITTED_RATE = 0.75
MIN_ADMITTED_STRESS_POSITIVE_RATE = 0.60
MIN_GOOD_DAYS = 4


@dataclass(frozen=True)
class SurvivabilityModel:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def selected_entry_states(
    scored_states: pd.DataFrame,
    entries: pd.DataFrame,
) -> pd.DataFrame:
    keys = entries.loc[
        :, EPISODE_KEYS + ["entry_minute_after_hot", "realized_base_return_pct"]
    ].copy()
    states = scored_states.merge(
        keys,
        left_on=EPISODE_KEYS + ["minutes_since_hot"],
        right_on=EPISODE_KEYS + ["entry_minute_after_hot"],
        how="inner",
        validate="one_to_one",
    )
    return states


def attach_first_stress_labels(
    entry_states: pd.DataFrame,
    positions: pd.DataFrame,
    entries: pd.DataFrame,
) -> pd.DataFrame:
    result = entry_states.copy()
    result["first_stress_return_pct"] = np.nan
    result["first_stress_minutes"] = np.nan
    result["pre_stress_peak_return_pct"] = np.nan
    result["first_stress_giveback_pct"] = np.nan
    result["first_stress_reason"] = None
    result["first_stress_t"] = np.nan

    event = reanchor_event_positions(positions, entries)
    if event.empty:
        return result

    labels: list[dict[str, object]] = []
    for keys, group in event.groupby(EPISODE_KEYS, sort=False):
        work = group.sort_values("state_t", kind="stable").copy()
        returns = pd.to_numeric(
            work["exit_now_base_return_pct"], errors="coerce"
        ).to_numpy(dtype=float)
        held = pd.to_numeric(
            work["minutes_held"], errors="coerce"
        ).to_numpy(dtype=float)
        times = pd.to_numeric(
            work["state_t"], errors="coerce"
        ).to_numpy(dtype=float)

        previous = 0.0
        peak = 0.0
        chosen: int | None = None
        reason: str | None = None

        for pos in range(len(work)):
            current = returns[pos]
            minute = held[pos]
            if not np.isfinite(current) or not np.isfinite(minute):
                continue

            if current < previous - EPSILON:
                chosen = pos
                reason = (
                    "immediate_adverse"
                    if pos == 0
                    else "first_adverse_observation"
                )
                break

            peak = max(peak, float(current))
            previous = float(current)

            if minute + EPSILON >= MAX_HOLD_MINUTES:
                chosen = pos
                reason = "no_adverse_before_30m_cap"
                break

        if chosen is None:
            continue

        current = returns[chosen]
        minute = held[chosen]
        timestamp = times[chosen]
        if not (
            np.isfinite(current)
            and np.isfinite(minute)
            and np.isfinite(timestamp)
        ):
            continue

        peak_before = peak
        if reason == "no_adverse_before_30m_cap":
            peak_before = max(peak_before, float(current))

        labels.append(
            {
                "trading_day": str(keys[0]),
                "ticker": str(keys[1]).upper(),
                "hot_t": int(keys[2]),
                "first_stress_return_pct": float(current),
                "first_stress_minutes": float(minute),
                "pre_stress_peak_return_pct": float(peak_before),
                "first_stress_giveback_pct": float(
                    current - peak_before
                ),
                "first_stress_reason": str(reason),
                "first_stress_t": int(timestamp),
            }
        )

    if not labels:
        return result

    label_frame = pd.DataFrame(labels)
    merged = result.drop(
        columns=[
            "first_stress_return_pct",
            "first_stress_minutes",
            "pre_stress_peak_return_pct",
            "first_stress_giveback_pct",
            "first_stress_reason",
            "first_stress_t",
        ]
    ).merge(
        label_frame,
        on=EPISODE_KEYS,
        how="left",
        validate="one_to_one",
    )
    return merged


def _usable_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in ENTRY_SURVIVABILITY_FEATURES
        if column in frame.columns
        and pd.to_numeric(
            frame[column], errors="coerce"
        ).notna().any()
    )


def train_survivability_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> SurvivabilityModel:
    target = pd.to_numeric(
        fit["first_stress_return_pct"], errors="coerce"
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].to_numpy(dtype=float)
    if len(train) < 300:
        raise ValueError(
            f"request 226 insufficient fit labels: {len(train)}"
        )

    columns = _usable_columns(train)
    low, high = np.quantile(y, [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=MODEL_SEED,
    )
    model.fit(
        _feature_frame(train, columns),
        np.clip(y, low, high),
        sample_weight=_day_weights(train),
    )

    cal_target = pd.to_numeric(
        calibration["first_stress_return_pct"],
        errors="coerce",
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 150:
        raise ValueError(
            f"request 226 insufficient calibration labels: {len(cal)}"
        )
    raw = model.predict(_feature_frame(cal, columns))
    residual = (
        cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    )
    offset = float(
        np.average(residual, weights=_day_weights(cal))
    )
    return SurvivabilityModel(
        model=model,
        columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_survivability(
    frame: pd.DataFrame,
    fitted: SurvivabilityModel,
) -> np.ndarray:
    return (
        fitted.model.predict(
            _feature_frame(frame, fitted.columns)
        )
        + fitted.offset
    )


def _safe_spearman(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = a.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    value = a.loc[valid].corr(s.loc[valid], method="spearman")
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


def survivability_diagnostics(
    frame: pd.DataFrame,
) -> dict[str, object]:
    actual = pd.to_numeric(
        frame["first_stress_return_pct"], errors="coerce"
    )
    score = pd.to_numeric(
        frame["predicted_first_stress_return_pct"],
        errors="coerce",
    )
    valid = actual.notna() & score.notna()
    work = frame.loc[valid].copy()
    work["_actual"] = actual.loc[valid]
    work["_score"] = score.loc[valid]
    work["_admit"] = work["_score"].gt(0)

    admitted = work.loc[work["_admit"]]
    reasons = {
        str(k): int(v)
        for k, v in work["first_stress_reason"]
        .value_counts(dropna=False)
        .to_dict()
        .items()
    }
    by_day = {}
    good_days = 0
    for day, part in work.groupby(
        work["trading_day"].astype(str), sort=True
    ):
        corr = _safe_spearman(part["_actual"], part["_score"])
        selected = part.loc[part["_admit"]]
        selected_mean = (
            float(selected["_actual"].mean())
            if len(selected)
            else None
        )
        good = bool(
            corr is not None
            and corr > 0
            and selected_mean is not None
            and selected_mean > 0
        )
        good_days += int(good)
        by_day[str(day)] = {
            "rows": int(len(part)),
            "spearman": corr,
            "admitted_rate": float(part["_admit"].mean()),
            "admitted_stress_mean_pct": selected_mean,
            "admitted_stress_positive_rate": (
                float(selected["_actual"].gt(0).mean())
                if len(selected)
                else None
            ),
        }

    return {
        "rows": int(len(frame)),
        "evaluable_rows": int(len(work)),
        "label_coverage": (
            float(len(work) / len(frame)) if len(frame) else None
        ),
        "spearman": _safe_spearman(
            work["_actual"], work["_score"]
        ),
        "admitted_rows": int(work["_admit"].sum()),
        "admitted_rate": (
            float(work["_admit"].mean()) if len(work) else None
        ),
        "admitted_stress_mean_pct": (
            float(admitted["_actual"].mean())
            if len(admitted)
            else None
        ),
        "admitted_stress_positive_rate": (
            float(admitted["_actual"].gt(0).mean())
            if len(admitted)
            else None
        ),
        "overall_stress_mean_pct": (
            float(work["_actual"].mean()) if len(work) else None
        ),
        "median_first_stress_minutes": (
            float(
                pd.to_numeric(
                    work["first_stress_minutes"],
                    errors="coerce",
                ).median()
            )
            if len(work)
            else None
        ),
        "stress_reasons": reasons,
        "good_days": int(good_days),
        "by_day": by_day,
    }


def gate_policy(
    scored_entries: pd.DataFrame,
) -> pd.DataFrame:
    result = scored_entries.loc[
        :,
        EPISODE_KEYS
        + [
            "entry_minute_after_hot",
            "realized_base_return_pct",
            "predicted_first_stress_return_pct",
            "first_stress_return_pct",
        ],
    ].copy()
    result["admitted"] = pd.to_numeric(
        result["predicted_first_stress_return_pct"],
        errors="coerce",
    ).gt(0)
    realized = pd.to_numeric(
        result["realized_base_return_pct"], errors="coerce"
    )
    result["baseline_return_pct"] = realized
    result["candidate_return_pct"] = np.where(
        result["admitted"],
        realized,
        0.0,
    )
    result["difference_pct"] = (
        result["candidate_return_pct"]
        - result["baseline_return_pct"]
    )
    return result


def policy_metrics(
    rows: pd.DataFrame,
    column: str,
) -> dict[str, object]:
    values = pd.to_numeric(rows[column], errors="coerce")
    work = rows.loc[values.notna()].copy()
    work["_value"] = values.loc[values.notna()]
    if work.empty:
        return {"trades": 0}

    daily = work.groupby(
        work["trading_day"].astype(str), sort=True
    )["_value"].mean()
    return {
        "episodes": int(len(work)),
        "mean_pct": float(work["_value"].mean()),
        "day_balanced_mean_pct": float(daily.mean()),
        "positive_rate": float(work["_value"].gt(0).mean()),
        "severe_loss_rate": float(
            work["_value"].le(SEVERE_LOSS_PCT).mean()
        ),
        "p05_pct": float(work["_value"].quantile(0.05)),
        "by_day": {
            str(k): float(v) for k, v in daily.items()
        },
    }


def matched_difference(
    rows: pd.DataFrame,
) -> dict[str, object]:
    diff = pd.to_numeric(rows["difference_pct"], errors="coerce")
    work = rows.loc[diff.notna()].copy()
    work["_diff"] = diff.loc[diff.notna()]
    daily = work.groupby(
        work["trading_day"].astype(str), sort=True
    )["_diff"].mean()
    return {
        "episodes": int(len(work)),
        "mean_difference_pct": float(work["_diff"].mean()),
        "candidate_better_rate": float(
            work["_diff"].gt(0).mean()
        ),
        "day_balanced_difference_pct": float(daily.mean()),
        "improved_days": int((daily > 0).sum()),
        "bootstrap": _bootstrap_daily(
            daily.to_numpy(dtype=float),
            BOOTSTRAP_SEED,
        ),
    }


def _prepare_entry_bundle(
    fit_positions: pd.DataFrame,
    calibration_positions: pd.DataFrame,
    fresh_positions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    fit_watch = attach_watch_transitions(
        attach_relative_advantage(
            build_watch_states(fit_positions)
        )
    )
    cal_watch = attach_watch_transitions(
        attach_relative_advantage(
            build_watch_states(calibration_positions)
        )
    )
    entry_model = train_entry_model(fit_watch, cal_watch)

    fit_watch["predicted_relative_advantage_pct"] = predict_entry(
        fit_watch, entry_model
    )
    cal_watch["predicted_relative_advantage_pct"] = predict_entry(
        cal_watch, entry_model
    )
    fresh_watch = score_entry_states(
        fresh_positions, entry_model
    )

    fit_entries = choose_entries(fit_watch)
    cal_entries = choose_entries(cal_watch)
    fresh_entries = choose_entries(fresh_watch)

    return (
        selected_entry_states(fit_watch, fit_entries),
        selected_entry_states(cal_watch, cal_entries),
        selected_entry_states(fresh_watch, fresh_entries),
        fit_entries,
        cal_entries,
        fresh_entries,
    )


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(position_paths) != len(FRESH_DAYS):
        raise ValueError(
            "request 226 requires complete Request178 position shards"
        )
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    found_days = sorted(
        fresh_positions["trading_day"].astype(str).unique()
    )
    if found_days != list(FRESH_DAYS):
        raise ValueError(
            f"request 226 expected {FRESH_DAYS}, found {found_days}"
        )

    (
        fit_states,
        cal_states,
        fresh_states,
        fit_entries,
        cal_entries,
        fresh_entries,
    ) = _prepare_entry_bundle(
        fit_positions,
        calibration_positions,
        fresh_positions,
    )

    fit_states = attach_first_stress_labels(
        fit_states, fit_positions, fit_entries
    )
    cal_states = attach_first_stress_labels(
        cal_states, calibration_positions, cal_entries
    )
    fresh_states = attach_first_stress_labels(
        fresh_states, fresh_positions, fresh_entries
    )

    model = train_survivability_model(
        fit_states, cal_states
    )
    cal_states["predicted_first_stress_return_pct"] = (
        predict_survivability(cal_states, model)
    )
    fresh_states["predicted_first_stress_return_pct"] = (
        predict_survivability(fresh_states, model)
    )

    calibration_diag = survivability_diagnostics(cal_states)
    fresh_diag = survivability_diagnostics(fresh_states)

    policy = gate_policy(fresh_states)
    baseline_metrics = policy_metrics(
        policy, "baseline_return_pct"
    )
    candidate_metrics = policy_metrics(
        policy, "candidate_return_pct"
    )
    difference = matched_difference(policy)

    coverage = fresh_diag.get("label_coverage")
    spearman = fresh_diag.get("spearman")
    admitted_rate = fresh_diag.get("admitted_rate")
    admitted_positive = fresh_diag.get(
        "admitted_stress_positive_rate"
    )
    candidate_day = candidate_metrics.get(
        "day_balanced_mean_pct"
    )
    baseline_day = baseline_metrics.get(
        "day_balanced_mean_pct"
    )
    diff_day = difference.get("day_balanced_difference_pct")
    diff_low = difference.get("bootstrap", {}).get("ci_low_pct")
    candidate_severe = candidate_metrics.get("severe_loss_rate")
    baseline_severe = baseline_metrics.get("severe_loss_rate")

    gate = bool(
        coverage is not None
        and float(coverage) >= MIN_LABEL_COVERAGE
        and spearman is not None
        and float(spearman) >= MIN_STRESS_SPEARMAN
        and admitted_rate is not None
        and MIN_ADMITTED_RATE
        <= float(admitted_rate)
        <= MAX_ADMITTED_RATE
        and admitted_positive is not None
        and float(admitted_positive)
        >= MIN_ADMITTED_STRESS_POSITIVE_RATE
        and int(fresh_diag.get("good_days", 0))
        >= MIN_GOOD_DAYS
        and candidate_day is not None
        and baseline_day is not None
        and float(candidate_day) > 0
        and float(candidate_day) > float(baseline_day)
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
        and candidate_severe is not None
        and baseline_severe is not None
        and float(candidate_severe) <= float(baseline_severe)
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "entry_policy": "frozen Request208 proposals",
        "candidate_intervention": (
            "admit Request208 entry only if calibrated predicted "
            "BASE return at first post-entry stress is positive"
        ),
        "first_stress_definition": {
            "adverse_event": (
                "first observed post-entry BASE return below the "
                "previous observed return, with entry return anchored at 0"
            ),
            "fallback": (
                "first observed state at/after the 30-minute risk cap"
            ),
            "future_information_used_as_input": False,
            "hindsight_best_price_used": False,
        },
        "target": "BASE return at first stress observation",
        "feature_count": len(model.columns),
        "model": {
            "family": "HistGradientBoostingRegressor",
            "learning_rate": 0.05,
            "max_iter": 180,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 60,
            "l2_regularization": 2.0,
            "seed": MODEL_SEED,
            "semantic_admission_boundary_pct": 0.0,
        },
        "support": {
            "fit_request208_entries": int(len(fit_entries)),
            "calibration_request208_entries": int(len(cal_entries)),
            "fresh_request208_entries": int(len(fresh_entries)),
        },
        "calibration_survivability": calibration_diag,
        "fresh_survivability": fresh_diag,
        "fresh_policy": {
            "request208_fixed3m_baseline": baseline_metrics,
            "survivability_gated_fixed3m": candidate_metrics,
            "matched_candidate_minus_request208": difference,
        },
        "frozen_gate": {
            "min_label_coverage": MIN_LABEL_COVERAGE,
            "min_fresh_stress_spearman": MIN_STRESS_SPEARMAN,
            "min_admitted_rate": MIN_ADMITTED_RATE,
            "max_admitted_rate": MAX_ADMITTED_RATE,
            "min_admitted_stress_positive_rate": (
                MIN_ADMITTED_STRESS_POSITIVE_RATE
            ),
            "min_good_days": MIN_GOOD_DAYS,
            "candidate_day_balanced_mean_must_be_positive": True,
            "candidate_must_beat_request208": True,
            "matched_bootstrap_ci_low_must_be_positive": True,
            "severe_loss_no_worse_than_request208": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze survivability admission, then test whether the "
            "survivability value can be blended with Request208 timing "
            "to enter earlier on strong-cushion setups before the first "
            "pullback, followed by full recurrent replay."
        ),
        "next_boundary_if_fail": (
            "entry-time observables do not reliably predict first-stress "
            "cushion; revisit the Request208 entry objective itself using "
            "multi-event pre-entry continuation value rather than fixed "
            "three-minute timing."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    policy.to_parquet(
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
