"""Request 216: event-time recurrent POSITION replay.

Development-only structural experiment on the already-opened Request178 dates.

Request212 re-anchored positions to the Request208 causal entry controller but
required an exact next-minute POSITION state for every HOLD transition.
Request216 keeps the Request208 entry controller, model capacity, feature family,
BASE costs and zero semantic HOLD boundary frozen while changing POSITION replay
to event time:

* decisions are made only at actually observed/executable POSITION states;
* silent clock time is forced HOLD/no-action, never a synthetic decision;
* HOLD value is EXIT-at-next-observed-event minus EXIT-now;
* time since the previous observed event is a causal feature;
* time to the next observed event is never a feature;
* a 30-minute risk cap exits at the first executable observation at or after
  the cap, and unresolved positions are never converted to cash.

No new market dates are opened and this experiment is not promotion eligible.
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
from .direct_recurrent_hold_exit_advantage import (
    EPISODE_KEYS,
    FEATURES,
    MAX_HOLD_MINUTES,
    MIN_ADVANTAGE_SPEARMAN,
    MIN_GOOD_DAYS,
    MIN_START_COVERAGE,
    advantage_diagnostics,
    build_trajectories,
    matched_difference,
    trajectory_metrics,
)
from .fresh_consensus_hurdle_value import FRESH_DAYS
from .integrated_entry_hold_exit import (
    choose_entries,
    reanchor_positions,
    score_entry_states,
)
from .market_regime_position_value import (
    attach_market_regime,
    build_market_regime,
)
from .pullback_turn_transition_entry import (
    attach_transition_features as attach_watch_transitions,
    predict as predict_entry,
    train_model as train_entry_model,
)
from .recurrent_wait_entry_action_value import (
    _base_return,
    build_watch_states,
)
from .relative_recurrent_entry_timing import attach_relative_advantage
from .transition_recurrent_hold_exit import (
    POSITION_TRANSITION_FEATURES,
    attach_position_transitions,
    predict_transition_advantage,
    train_transition_advantage_model,
)

REQUEST_ID = 216
MODEL_SEED = 20261126
EPSILON = 1e-9

EVENT_FEATURES = tuple(
    dict.fromkeys(
        [
            *FEATURES,
            *POSITION_TRANSITION_FEATURES,
            "elapsed_since_previous_observation_min",
        ]
    )
)


@dataclass(frozen=True)
class EventAdvantageModel:
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


def _positive(value: object) -> bool:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(
        pd.notna(parsed)
        and np.isfinite(float(parsed))
        and float(parsed) > 0
    )


def _entry_map(entries: pd.DataFrame) -> dict[tuple[str, str, int], int]:
    return {
        (
            str(row["trading_day"]),
            str(row["ticker"]).upper(),
            int(row["hot_t"]),
        ): int(row["entry_minute_after_hot"])
        for _, row in entries.iterrows()
    }


def reanchor_event_positions(
    position_rows: pd.DataFrame,
    entries: pd.DataFrame,
) -> pd.DataFrame:
    """Re-anchor Request208 entries and create next-observed-event HOLD labels."""
    selected = _entry_map(entries)
    records: list[dict[str, object]] = []

    for keys, group in position_rows.groupby(EPISODE_KEYS, sort=False):
        key = (str(keys[0]), str(keys[1]).upper(), int(keys[2]))
        chosen_minute = selected.get(key)
        if chosen_minute is None:
            continue

        work = group.copy()
        work["_minute"] = pd.to_numeric(
            work["minutes_held"], errors="coerce"
        )
        work["_state_t"] = pd.to_numeric(work["state_t"], errors="coerce")
        work["_exit_open"] = pd.to_numeric(
            work["exit_reference_open"], errors="coerce"
        )
        work = work.loc[
            work["_minute"].notna()
            & work["_state_t"].notna()
            & work["_exit_open"].gt(0)
        ].copy()
        if work.empty:
            continue

        work = work.sort_values(
            ["_state_t", "_minute"], kind="stable"
        ).drop_duplicates("_state_t", keep="first")

        entry_candidates = work.loc[
            (work["_minute"] - float(chosen_minute)).abs() <= EPSILON
        ]
        if entry_candidates.empty:
            continue
        entry_row = entry_candidates.iloc[0]
        entry_t = int(entry_row["_state_t"])
        entry_open = float(entry_row["_exit_open"])
        if not _positive(entry_open):
            continue

        future = work.loc[work["_state_t"].gt(entry_t)].copy()
        if future.empty:
            continue
        future = future.sort_values("_state_t", kind="stable").reset_index(
            drop=True
        )

        running_high = np.nan
        running_low = np.nan
        previous_t = entry_t

        for event_index, (_, row) in enumerate(future.iterrows(), start=1):
            current_t = int(row["_state_t"])
            current_exit_open = float(row["_exit_open"])
            current_exit = _base_return(entry_open, current_exit_open)

            current_close_log = _number(row, "log_current_close")
            current_close = (
                float(np.exp(current_close_log))
                if np.isfinite(current_close_log)
                else np.nan
            )
            if _positive(current_close):
                running_high = (
                    current_close
                    if not np.isfinite(running_high)
                    else max(running_high, current_close)
                )
                running_low = (
                    current_close
                    if not np.isfinite(running_low)
                    else min(running_low, current_close)
                )

            next_event_return = np.nan
            next_event_t = np.nan
            if event_index < len(future):
                next_row = future.iloc[event_index]
                next_event_t = int(next_row["_state_t"])
                next_event_return = _base_return(
                    entry_open, float(next_row["_exit_open"])
                )

            record = row.drop(
                labels=["_minute", "_state_t", "_exit_open"],
                errors="ignore",
            ).to_dict()
            record["entry_actual_t"] = entry_t
            record["entry_open"] = entry_open
            record["log_entry_price"] = float(np.log(entry_open))
            record["minutes_held"] = float(
                (current_t - entry_t) / MINUTE_MS
            )
            record["event_index"] = int(event_index)
            record["elapsed_since_previous_observation_min"] = float(
                (current_t - previous_t) / MINUTE_MS
            )
            record["exit_now_base_return_pct"] = current_exit
            record["next_event_base_return_pct"] = next_event_return
            record["next_event_t"] = next_event_t
            record["hold_advantage_event_pct"] = (
                float(next_event_return - current_exit)
                if pd.notna(next_event_return) and pd.notna(current_exit)
                else np.nan
            )
            record["entry_to_current_close_pct"] = (
                float(current_close / entry_open - 1.0) * 100.0
                if _positive(current_close)
                else np.nan
            )
            record["running_max_return_pct"] = (
                float(running_high / entry_open - 1.0) * 100.0
                if _positive(running_high)
                else np.nan
            )
            record["running_min_return_pct"] = (
                float(running_low / entry_open - 1.0) * 100.0
                if _positive(running_low)
                else np.nan
            )
            record["drawdown_from_peak_pct"] = (
                float(current_close / running_high - 1.0) * 100.0
                if _positive(current_close) and _positive(running_high)
                else np.nan
            )
            record["recovery_from_trough_pct"] = (
                float(current_close / running_low - 1.0) * 100.0
                if _positive(current_close) and _positive(running_low)
                else np.nan
            )
            records.append(record)
            previous_t = current_t

    return pd.DataFrame(records)


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric, errors="coerce"
    )


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    day = frame["trading_day"].astype(str)
    counts = day.map(day.value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(dtype=float)
    mean = float(np.mean(weights))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("invalid equal-day weights")
    return weights / mean


def train_event_advantage_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> EventAdvantageModel:
    target = pd.to_numeric(
        fit["hold_advantage_event_pct"], errors="coerce"
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError("request 216 insufficient fit event-action support")

    columns = tuple(
        column
        for column in EVENT_FEATURES
        if column in train.columns
        and pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    if not columns:
        raise ValueError("request 216 has no usable event-time features")

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
        calibration["hold_advantage_event_pct"], errors="coerce"
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 500:
        raise ValueError(
            "request 216 insufficient calibration event-action support"
        )
    raw = model.predict(_feature_frame(cal, columns))
    residual = cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    offset = float(np.average(residual, weights=_day_weights(cal)))
    return EventAdvantageModel(
        model=model,
        feature_columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_event_advantage(
    frame: pd.DataFrame,
    fitted: EventAdvantageModel,
) -> np.ndarray:
    return (
        fitted.model.predict(_feature_frame(frame, fitted.feature_columns))
        + fitted.offset
    )


def event_advantage_diagnostics(scored: pd.DataFrame) -> dict[str, object]:
    diagnostic = scored.copy()
    diagnostic["hold_advantage_1m_pct"] = pd.to_numeric(
        diagnostic["hold_advantage_event_pct"], errors="coerce"
    )
    diagnostic["predicted_hold_advantage_1m_pct"] = pd.to_numeric(
        diagnostic["predicted_hold_advantage_event_pct"], errors="coerce"
    )
    result = advantage_diagnostics(diagnostic)
    result["target_semantics"] = (
        "BASE EXIT at next executable observation minus BASE EXIT now"
    )
    return result


def build_event_trajectories(
    scored: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    groups: dict[tuple[str, str, int], pd.DataFrame] = {}
    for keys, group in scored.groupby(EPISODE_KEYS, sort=False):
        key = (str(keys[0]), str(keys[1]).upper(), int(keys[2]))
        groups[key] = group.sort_values("state_t", kind="stable").copy()

    rows: list[dict[str, object]] = []
    for _, entry in entries.iterrows():
        key = (
            str(entry["trading_day"]),
            str(entry["ticker"]).upper(),
            int(entry["hot_t"]),
        )
        group = groups.get(key)
        if group is None or group.empty:
            continue

        work = group.reset_index(drop=True)
        first = work.iloc[0]
        entry_t = int(first["entry_actual_t"])
        status = "unresolved"
        exit_reason = "unknown"
        realized = np.nan
        exit_t = np.nan
        hold_decisions = 0
        observed_decisions = 0
        forced_silence_minutes = 0.0

        for index, row in work.iterrows():
            current_t = int(row["state_t"])
            held_minutes = float((current_t - entry_t) / MINUTE_MS)
            current_exit = _number(row, "exit_now_base_return_pct")
            if not np.isfinite(current_exit):
                status = "unresolved"
                exit_reason = "missing_current_exit"
                break

            if held_minutes + EPSILON >= MAX_HOLD_MINUTES:
                status = "completed"
                exit_reason = "forced_30m_event_cap"
                realized = current_exit
                exit_t = current_t
                break

            score = _number(row, "predicted_hold_advantage_event_pct")
            observed_decisions += 1
            if not np.isfinite(score):
                status = "completed"
                exit_reason = "unscoreable_exit"
                realized = current_exit
                exit_t = current_t
                break

            if score <= threshold:
                status = "completed"
                exit_reason = "model_exit"
                realized = current_exit
                exit_t = current_t
                break

            hold_decisions += 1
            if index + 1 >= len(work):
                status = "unresolved"
                exit_reason = "no_next_executable_observation"
                break

            next_row = work.iloc[index + 1]
            next_t = int(next_row["state_t"])
            gap_min = float((next_t - current_t) / MINUTE_MS)
            forced_silence_minutes += max(0.0, gap_min - 1.0)

        rows.append(
            {
                "trading_day": key[0],
                "ticker": key[1],
                "hot_t": key[2],
                "policy": "event_time_expected_advantage",
                "status": status,
                "exit_reason": exit_reason,
                "base_net_return_pct": (
                    float(realized) if pd.notna(realized) else np.nan
                ),
                "minutes_held": (
                    float((int(exit_t) - entry_t) / MINUTE_MS)
                    if pd.notna(exit_t)
                    else np.nan
                ),
                "hold_decisions": int(hold_decisions),
                "observed_decisions": int(observed_decisions),
                "forced_silence_minutes": float(forced_silence_minutes),
                "entry_actual_t": int(entry_t),
                "exit_t": int(exit_t) if pd.notna(exit_t) else np.nan,
            }
        )

    columns = [
        *EPISODE_KEYS,
        "policy",
        "status",
        "exit_reason",
        "base_net_return_pct",
        "minutes_held",
        "hold_decisions",
        "observed_decisions",
        "forced_silence_minutes",
        "entry_actual_t",
        "exit_t",
    ]
    trajectories = pd.DataFrame(rows, columns=columns)
    expected = int(len(entries))
    started = int(len(trajectories))
    return trajectories, {
        "selected_entries": expected,
        "started_episodes": started,
        "start_state_coverage": (
            float(started / expected) if expected else None
        ),
        "episodes_with_forced_silence": (
            int(trajectories["forced_silence_minutes"].gt(0).sum())
            if len(trajectories)
            else 0
        ),
        "mean_forced_silence_minutes": (
            float(trajectories["forced_silence_minutes"].mean())
            if len(trajectories)
            else None
        ),
    }


def fixed_three_minute_trajectories(
    entries: pd.DataFrame,
) -> pd.DataFrame:
    rows = entries.loc[
        :, EPISODE_KEYS + ["realized_base_return_pct"]
    ].copy()
    rows["policy"] = "request208_fixed3m"
    rows["status"] = np.where(
        pd.to_numeric(
            rows["realized_base_return_pct"], errors="coerce"
        ).notna(),
        "completed",
        "unresolved",
    )
    rows["exit_reason"] = "fixed_3m"
    rows["base_net_return_pct"] = pd.to_numeric(
        rows["realized_base_return_pct"], errors="coerce"
    )
    rows["minutes_held"] = 3.0
    rows["hold_decisions"] = 0
    rows["entry_actual_t"] = np.nan
    rows["exit_t"] = np.nan
    return rows.drop(columns=["realized_base_return_pct"])


def _prepare_entries(
    fit_positions: pd.DataFrame,
    calibration_positions: pd.DataFrame,
    fresh_positions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_states = attach_watch_transitions(
        attach_relative_advantage(build_watch_states(fit_positions))
    )
    calibration_states = attach_watch_transitions(
        attach_relative_advantage(
            build_watch_states(calibration_positions)
        )
    )
    entry_model = train_entry_model(fit_states, calibration_states)

    fit_states["predicted_relative_advantage_pct"] = predict_entry(
        fit_states, entry_model
    )
    calibration_states["predicted_relative_advantage_pct"] = predict_entry(
        calibration_states, entry_model
    )
    fresh_states = score_entry_states(fresh_positions, entry_model)
    return (
        choose_entries(fit_states),
        choose_entries(calibration_states),
        choose_entries(fresh_states),
    )


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    trajectories_output_path: Path,
) -> int:
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
        raise ValueError("request 216 requires complete request178 shards")

    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    found_days = sorted(fresh_positions["trading_day"].astype(str).unique())
    if found_days != list(FRESH_DAYS):
        raise ValueError(
            f"request 216 expected {FRESH_DAYS}, found {found_days}"
        )

    fit_entries, calibration_entries, fresh_entries = _prepare_entries(
        fit_positions, calibration_positions, fresh_positions
    )

    exact_fit = reanchor_positions(fit_positions, fit_entries)
    exact_calibration = reanchor_positions(
        calibration_positions, calibration_entries
    )
    exact_fresh = reanchor_positions(fresh_positions, fresh_entries)

    event_fit = reanchor_event_positions(fit_positions, fit_entries)
    event_calibration = reanchor_event_positions(
        calibration_positions, calibration_entries
    )
    event_fresh = reanchor_event_positions(
        fresh_positions, fresh_entries
    )

    historical_days = set(
        fit_positions["trading_day"].astype(str)
    ) | set(calibration_positions["trading_day"].astype(str))
    historical_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(historical_days)
    ].copy()
    historical_regime = build_market_regime(historical_scan)
    fresh_regime = build_market_regime(fresh_scan)

    exact_fit = attach_market_regime(exact_fit, historical_regime)
    exact_calibration = attach_market_regime(
        exact_calibration, historical_regime
    )
    exact_fresh = attach_market_regime(exact_fresh, fresh_regime)
    event_fit = attach_market_regime(event_fit, historical_regime)
    event_calibration = attach_market_regime(
        event_calibration, historical_regime
    )
    event_fresh = attach_market_regime(event_fresh, fresh_regime)

    exact_fit = attach_position_transitions(exact_fit)
    exact_calibration = attach_position_transitions(exact_calibration)
    exact_fresh = attach_position_transitions(exact_fresh)
    event_fit = attach_position_transitions(event_fit)
    event_calibration = attach_position_transitions(event_calibration)
    event_fresh = attach_position_transitions(event_fresh)

    exact_model = train_transition_advantage_model(
        exact_fit, exact_calibration
    )
    exact_scored = exact_fresh.copy()
    exact_scored["predicted_hold_advantage_1m_pct"] = (
        predict_transition_advantage(exact_scored, exact_model)
    )
    request212, request212_coverage = build_trajectories(
        exact_scored,
        score_column="predicted_hold_advantage_1m_pct",
        threshold=0.0,
        policy="request212_exact_minute_dynamic_hold_exit",
    )

    event_model = train_event_advantage_model(
        event_fit, event_calibration
    )
    event_scored = event_fresh.copy()
    event_scored["predicted_hold_advantage_event_pct"] = (
        predict_event_advantage(event_scored, event_model)
    )
    diagnostics = event_advantage_diagnostics(event_scored)
    candidate, candidate_coverage = build_event_trajectories(
        event_scored, fresh_entries
    )
    fixed3m = fixed_three_minute_trajectories(fresh_entries)

    candidate_metrics = trajectory_metrics(
        candidate, policy="event_time_expected_advantage"
    )
    request212_metrics = trajectory_metrics(
        request212,
        policy="request212_exact_minute_dynamic_hold_exit",
    )
    fixed3m_metrics = trajectory_metrics(
        fixed3m, policy="request208_fixed3m"
    )
    vs_request212 = matched_difference(candidate, request212)
    vs_fixed3m = matched_difference(candidate, fixed3m)

    spearman = diagnostics.get("advantage_spearman")
    selected_mean = diagnostics.get(
        "selected_realized_advantage_mean_pct"
    )
    candidate_day = candidate_metrics.get(
        "day_balanced_base_mean_pct"
    )
    request212_day = request212_metrics.get(
        "day_balanced_base_mean_pct"
    )
    fixed3m_day = fixed3m_metrics.get(
        "day_balanced_base_mean_pct"
    )
    diff_212 = vs_request212.get("day_balanced_difference_pct")
    diff_212_low = vs_request212.get("bootstrap", {}).get("ci_low_pct")
    diff_fixed = vs_fixed3m.get("day_balanced_difference_pct")
    diff_fixed_low = vs_fixed3m.get("bootstrap", {}).get("ci_low_pct")

    gate = bool(
        candidate_coverage["start_state_coverage"] is not None
        and float(candidate_coverage["start_state_coverage"])
        >= MIN_START_COVERAGE
        and spearman is not None
        and float(spearman) >= MIN_ADVANTAGE_SPEARMAN
        and selected_mean is not None
        and float(selected_mean) > 0
        and int(diagnostics["good_days"]) >= MIN_GOOD_DAYS
        and candidate_day is not None
        and request212_day is not None
        and fixed3m_day is not None
        and float(candidate_day) > 0
        and float(candidate_day) > float(request212_day)
        and float(candidate_day) > float(fixed3m_day)
        and diff_212 is not None
        and float(diff_212) > 0
        and diff_212_low is not None
        and float(diff_212_low) > 0
        and diff_fixed is not None
        and float(diff_fixed) > 0
        and diff_fixed_low is not None
        and float(diff_fixed_low) > 0
    )

    event_gap = pd.to_numeric(
        event_scored["elapsed_since_previous_observation_min"],
        errors="coerce",
    )
    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "entry_policy": (
            "Request208 causal transition-relative recurrent ENTER/WAIT"
        ),
        "position_policy": (
            "HOLD iff predicted BASE advantage to next executable "
            "observation > 0; silent time forces no-action HOLD"
        ),
        "causality": {
            "decisions_only_on_observed_states": True,
            "time_since_previous_observation_feature": True,
            "time_to_next_observation_feature": False,
            "unresolved_positions_zero_filled": False,
        },
        "fit_selected_entries": int(len(fit_entries)),
        "calibration_selected_entries": int(len(calibration_entries)),
        "fresh_selected_entries": int(len(fresh_entries)),
        "event_time_rows": {
            "fit": int(len(event_fit)),
            "calibration": int(len(event_calibration)),
            "fresh": int(len(event_fresh)),
        },
        "event_gap_minutes": {
            "mean": (
                float(event_gap.mean()) if event_gap.notna().any() else None
            ),
            "p90": (
                float(event_gap.quantile(0.90))
                if event_gap.notna().any()
                else None
            ),
            "max": (
                float(event_gap.max()) if event_gap.notna().any() else None
            ),
            "greater_than_one_minute_rows": int(event_gap.gt(1.0).sum()),
        },
        "event_hold_advantage_diagnostics": diagnostics,
        "candidate_coverage": candidate_coverage,
        "request212_coverage": request212_coverage,
        "candidate_event_time": candidate_metrics,
        "comparator_request212_exact_minute": request212_metrics,
        "comparator_request208_fixed3m": fixed3m_metrics,
        "matched_event_minus_request212": vs_request212,
        "matched_event_minus_fixed3m": vs_fixed3m,
        "model": {
            "feature_count": len(event_model.feature_columns),
            "adds_only_elapsed_previous_observation": True,
            "offset_pct": event_model.offset,
            "winsor_low_pct": event_model.winsor_low,
            "winsor_high_pct": event_model.winsor_high,
        },
        "frozen_gate": {
            "min_start_coverage": MIN_START_COVERAGE,
            "min_advantage_spearman": MIN_ADVANTAGE_SPEARMAN,
            "min_good_days": MIN_GOOD_DAYS,
            "selected_advantage_mean_must_be_positive": True,
            "candidate_day_balanced_base_must_be_positive": True,
            "candidate_must_beat_request212": True,
            "candidate_must_beat_fixed3m": True,
            "both_matched_bootstrap_lows_must_be_positive": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.concat(
        [
            candidate.assign(comparator="candidate_event_time"),
            request212.assign(comparator="request212_exact_minute"),
            fixed3m.assign(comparator="request208_fixed3m"),
        ],
        ignore_index=True,
    ).to_parquet(
        trajectories_output_path, index=False, compression="zstd"
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
    parser.add_argument(
        "--trajectories-output", type=Path, required=True
    )
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.trajectories_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
