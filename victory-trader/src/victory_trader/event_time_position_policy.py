"""Request 216: event-time POSITION replay and executable HOLD/EXIT.

This development-only experiment keeps Request208 entry timing frozen and replays
the selected positions on the same already-opened Request178 dates. Unlike
Request212, HOLD is not defined as "advance to the next exact integer minute".
A decision can be taken only at an observed/executable position state. If the
clock is silent, the position remains open until the next executable event.

The learning target is therefore the realized advantage of HOLDing from the
current executable event to the next executable event, versus EXIT now.
Future event spacing is an outcome only and is never exposed as a feature.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .direct_recurrent_hold_exit_advantage import (
    MAX_HOLD_MINUTES,
    MIN_ADVANTAGE_SPEARMAN,
    MIN_GOOD_DAYS,
    MIN_START_COVERAGE,
    advantage_diagnostics,
)
from .fresh_consensus_hurdle_value import FRESH_DAYS
from .integrated_entry_hold_exit import (
    choose_entries,
    dynamic_to_trades,
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
from .relative_recurrent_entry_timing import (
    _matched_difference,
    _metrics,
    attach_relative_advantage,
)
from .transition_recurrent_hold_exit import (
    POSITION_TRANSITION_FEATURES,
    TRANSITION_FEATURES,
    attach_position_transitions,
)

REQUEST_ID = 216
MODEL_SEED = 20261126
EPISODE_KEYS = ["trading_day", "ticker", "hot_t"]

EVENT_STATE_FEATURES = (
    "event_prev_gap_minutes",
    "event_prev_stale_minutes",
    "event_prev_gap_log1p",
    "event_prev_gap_gt1",
    "event_elapsed_minutes",
)
EVENT_RATE_FEATURES = tuple(
    f"{column}_per_elapsed_minute"
    for column in POSITION_TRANSITION_FEATURES
)
MODEL_FEATURES = tuple(
    dict.fromkeys(
        [
            *TRANSITION_FEATURES,
            *EVENT_STATE_FEATURES,
            *EVENT_RATE_FEATURES,
        ]
    )
)


@dataclass(frozen=True)
class EventHoldModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def _number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(
        pd.Series([row.get(column)]), errors="coerce"
    ).iloc[0]
    if pd.isna(value) or not np.isfinite(float(value)):
        return np.nan
    return float(value)


def _positive(value: object) -> bool:
    parsed = pd.to_numeric(
        pd.Series([value]), errors="coerce"
    ).iloc[0]
    return bool(
        pd.notna(parsed)
        and np.isfinite(float(parsed))
        and float(parsed) > 0
    )


def _pct_change(current: float, reference: float) -> float:
    if not _positive(current) or not _positive(reference):
        return np.nan
    return float((current / reference - 1.0) * 100.0)


def reanchor_event_positions(
    position_rows: pd.DataFrame,
    entries: pd.DataFrame,
) -> pd.DataFrame:
    """Re-anchor to chosen entry and preserve every later observed event."""
    entry_map = {
        (
            str(row["trading_day"]),
            str(row["ticker"]).upper(),
            int(row["hot_t"]),
        ): float(row["entry_minute_after_hot"])
        for _, row in entries.iterrows()
    }
    records: list[dict[str, object]] = []

    for keys, group in position_rows.groupby(EPISODE_KEYS, sort=False):
        key = (str(keys[0]), str(keys[1]).upper(), int(keys[2]))
        chosen = entry_map.get(key)
        if chosen is None:
            continue

        work = group.copy()
        work["_source_minute"] = pd.to_numeric(
            work["minutes_held"], errors="coerce"
        )
        work["_state_t"] = pd.to_numeric(
            work["state_t"], errors="coerce"
        )
        work = work.loc[
            work["_source_minute"].notna()
            & work["_state_t"].notna()
        ].sort_values(
            ["_source_minute", "_state_t"],
            kind="stable",
        )
        if work.empty:
            continue

        entry_candidates = work.loc[
            np.isclose(
                work["_source_minute"].to_numpy(dtype=float),
                float(chosen),
                atol=1e-9,
                rtol=0.0,
            )
        ]
        if entry_candidates.empty:
            continue
        entry_row = entry_candidates.iloc[0]
        entry_open = _number(entry_row, "exit_reference_open")
        entry_t = _number(entry_row, "state_t")
        if not _positive(entry_open) or not np.isfinite(entry_t):
            continue

        future = work.loc[
            work["_source_minute"].gt(float(chosen))
        ].copy()
        if future.empty:
            continue

        ordered = list(future.iterrows())
        running_high = float(entry_open)
        running_low = float(entry_open)
        previous_source_minute = float(chosen)

        for position, (index, row) in enumerate(ordered):
            source_minute = float(row["_source_minute"])
            elapsed = source_minute - float(chosen)
            if elapsed <= 0:
                continue

            current_exit_open = _number(
                row, "exit_reference_open"
            )
            current_close_log = _number(row, "log_current_close")
            current_close = (
                float(np.exp(current_close_log))
                if np.isfinite(current_close_log)
                else np.nan
            )
            if _positive(current_close):
                running_high = max(running_high, current_close)
                running_low = min(running_low, current_close)

            current_return = _base_return(
                entry_open, current_exit_open
            )

            next_row = (
                ordered[position + 1][1]
                if position + 1 < len(ordered)
                else None
            )
            next_source_minute = (
                float(next_row["_source_minute"])
                if next_row is not None
                else np.nan
            )
            next_exit_open = (
                _number(next_row, "exit_reference_open")
                if next_row is not None
                else np.nan
            )
            next_return = _base_return(
                entry_open, next_exit_open
            )
            hold_advantage = (
                float(next_return - current_return)
                if pd.notna(next_return)
                and pd.notna(current_return)
                else np.nan
            )
            next_gap = (
                float(next_source_minute - source_minute)
                if np.isfinite(next_source_minute)
                else np.nan
            )
            previous_gap = source_minute - previous_source_minute

            record = row.drop(
                labels=["_source_minute", "_state_t"],
                errors="ignore",
            ).to_dict()
            record["source_minutes_held"] = source_minute
            record["entry_actual_t"] = int(entry_t)
            record["entry_open"] = float(entry_open)
            record["log_entry_price"] = float(
                np.log(entry_open)
            )
            record["minutes_held"] = float(elapsed)
            record["event_elapsed_minutes"] = float(elapsed)
            record["event_prev_gap_minutes"] = float(previous_gap)
            record["event_prev_stale_minutes"] = max(
                0.0, float(previous_gap) - 1.0
            )
            record["event_prev_gap_log1p"] = float(
                np.log1p(max(0.0, previous_gap))
            )
            record["event_prev_gap_gt1"] = float(
                previous_gap > 1.0 + 1e-9
            )
            record["exit_now_base_return_pct"] = current_return
            record["next_event_base_return_pct"] = next_return
            record["hold_advantage_next_event_pct"] = hold_advantage
            record["next_event_delta_minutes"] = next_gap
            record["entry_to_current_close_pct"] = _pct_change(
                current_close, entry_open
            )
            record["running_max_return_pct"] = _pct_change(
                running_high, entry_open
            )
            record["running_min_return_pct"] = _pct_change(
                running_low, entry_open
            )
            record["drawdown_from_peak_pct"] = _pct_change(
                current_close, running_high
            )
            record["recovery_from_trough_pct"] = _pct_change(
                current_close, running_low
            )
            records.append(record)
            previous_source_minute = source_minute

    return pd.DataFrame(records)


def attach_event_transition_rates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = attach_position_transitions(frame)
    gap = pd.to_numeric(
        result["event_prev_gap_minutes"], errors="coerce"
    )
    valid_gap = gap.where(gap.gt(0))
    for source, target in zip(
        POSITION_TRANSITION_FEATURES,
        EVENT_RATE_FEATURES,
        strict=True,
    ):
        values = pd.to_numeric(
            result.get(source), errors="coerce"
        )
        result[target] = values / valid_gap
    return result


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


def train_event_hold_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> EventHoldModel:
    target = pd.to_numeric(
        fit["hold_advantage_next_event_pct"], errors="coerce"
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError(
            "request 216 insufficient event-time fit support"
        )

    columns = tuple(
        column
        for column in MODEL_FEATURES
        if column in train.columns
        and pd.to_numeric(
            train[column], errors="coerce"
        ).notna().any()
    )
    if not columns:
        raise ValueError("request 216 has no usable features")

    low, high = np.quantile(
        y.to_numpy(dtype=float), [0.005, 0.995]
    )
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
        calibration["hold_advantage_next_event_pct"],
        errors="coerce",
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 500:
        raise ValueError(
            "request 216 insufficient event-time calibration support"
        )
    raw = model.predict(_feature_frame(cal, columns))
    residual = (
        cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    )
    offset = float(
        np.average(residual, weights=_day_weights(cal))
    )
    return EventHoldModel(
        model=model,
        feature_columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_event_hold(
    frame: pd.DataFrame,
    fitted: EventHoldModel,
) -> np.ndarray:
    return (
        fitted.model.predict(
            _feature_frame(frame, fitted.feature_columns)
        )
        + fitted.offset
    )


def build_event_trajectories(
    scored: pd.DataFrame,
    *,
    score_column: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    total = int(
        scored.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    rows: list[dict[str, object]] = []

    for _, group in scored.groupby(EPISODE_KEYS, sort=False):
        work = group.copy()
        work["_elapsed"] = pd.to_numeric(
            work["minutes_held"], errors="coerce"
        )
        work = work.loc[
            work["_elapsed"].notna() & work["_elapsed"].gt(0)
        ].sort_values(
            ["_elapsed", "state_t"], kind="stable"
        )
        if work.empty:
            continue

        observed = list(work.iterrows())
        first = observed[0][1]
        status = "unresolved"
        reason = "unknown"
        realized = np.nan
        exit_t = np.nan
        hold_decisions = 0

        for position, (_, row) in enumerate(observed):
            elapsed = float(row["_elapsed"])
            current_return = _number(
                row, "exit_now_base_return_pct"
            )
            if pd.isna(current_return):
                status = "unresolved"
                reason = "missing_current_exit"
                break

            if elapsed >= MAX_HOLD_MINUTES:
                status = "completed"
                reason = "forced_cap_observed"
                realized = float(current_return)
                exit_t = int(row["state_t"])
                break

            score = _number(row, score_column)
            if not np.isfinite(score):
                status = "completed"
                reason = "unscoreable_exit"
                realized = float(current_return)
                exit_t = int(row["state_t"])
                break

            if score <= 0:
                status = "completed"
                reason = "model_exit"
                realized = float(current_return)
                exit_t = int(row["state_t"])
                break

            hold_decisions += 1
            if position + 1 >= len(observed):
                status = "unresolved"
                reason = "no_later_executable_event"
                break

            next_row = observed[position + 1][1]
            next_elapsed = float(next_row["_elapsed"])
            if next_elapsed >= MAX_HOLD_MINUTES:
                next_return = _number(
                    next_row, "exit_now_base_return_pct"
                )
                if pd.notna(next_return):
                    status = "completed"
                    reason = "forced_cap_next_event"
                    realized = float(next_return)
                    exit_t = int(next_row["state_t"])
                else:
                    status = "unresolved"
                    reason = "missing_cap_execution"
                break

        rows.append(
            {
                "trading_day": str(first["trading_day"]),
                "ticker": str(first["ticker"]).upper(),
                "hot_t": int(first["hot_t"]),
                "policy": "request216_event_time_hold_exit",
                "status": status,
                "exit_reason": reason,
                "base_net_return_pct": (
                    float(realized)
                    if pd.notna(realized)
                    else np.nan
                ),
                "minutes_held": (
                    float(
                        (
                            int(exit_t)
                            - int(first["entry_actual_t"])
                        )
                        / 60_000
                    )
                    if pd.notna(exit_t)
                    else np.nan
                ),
                "hold_decisions": int(hold_decisions),
                "entry_actual_t": int(first["entry_actual_t"]),
                "exit_t": (
                    int(exit_t)
                    if pd.notna(exit_t)
                    else np.nan
                ),
            }
        )

    trajectories = pd.DataFrame(rows)
    started = int(
        trajectories.loc[:, EPISODE_KEYS]
        .drop_duplicates()
        .shape[0]
    ) if len(trajectories) else 0
    return trajectories, {
        "total_position_episodes": total,
        "started_episodes": started,
        "start_state_coverage": (
            float(started / total) if total else None
        ),
    }


def event_gap_diagnostics(
    frame: pd.DataFrame,
) -> dict[str, object]:
    previous_gap = pd.to_numeric(
        frame["event_prev_gap_minutes"], errors="coerce"
    )
    target = pd.to_numeric(
        frame["hold_advantage_next_event_pct"],
        errors="coerce",
    )
    next_gap = pd.to_numeric(
        frame["next_event_delta_minutes"],
        errors="coerce",
    )
    return {
        "rows": int(len(frame)),
        "evaluable_hold_targets": int(target.notna().sum()),
        "hold_target_coverage": (
            float(target.notna().mean()) if len(frame) else None
        ),
        "rows_arriving_after_silence": int(
            previous_gap.gt(1.0 + 1e-9).sum()
        ),
        "previous_gap_mean_min": (
            float(previous_gap.mean())
            if previous_gap.notna().any()
            else None
        ),
        "previous_gap_p90_min": (
            float(previous_gap.quantile(0.90))
            if previous_gap.notna().any()
            else None
        ),
        "next_gap_mean_min_outcome_only": (
            float(next_gap.mean())
            if next_gap.notna().any()
            else None
        ),
        "next_gap_p90_min_outcome_only": (
            float(next_gap.quantile(0.90))
            if next_gap.notna().any()
            else None
        ),
    }


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

    fit_entry_states = attach_watch_transitions(
        attach_relative_advantage(
            build_watch_states(fit_positions)
        )
    )
    calibration_entry_states = attach_watch_transitions(
        attach_relative_advantage(
            build_watch_states(calibration_positions)
        )
    )
    entry_model = train_entry_model(
        fit_entry_states, calibration_entry_states
    )
    fit_entry_states[
        "predicted_relative_advantage_pct"
    ] = predict_entry(fit_entry_states, entry_model)
    calibration_entry_states[
        "predicted_relative_advantage_pct"
    ] = predict_entry(calibration_entry_states, entry_model)

    fit_entries = choose_entries(fit_entry_states)
    calibration_entries = choose_entries(
        calibration_entry_states
    )

    position_paths = sorted(
        fresh_dir.glob("*-positions.parquet")
    )
    scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError(
            "request 216 requires complete request178 shards"
        )

    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    fresh_entry_states = score_entry_states(
        fresh_positions, entry_model
    )
    fresh_entries = choose_entries(fresh_entry_states)

    fit = reanchor_event_positions(
        fit_positions, fit_entries
    )
    calibration = reanchor_event_positions(
        calibration_positions, calibration_entries
    )
    fresh = reanchor_event_positions(
        fresh_positions, fresh_entries
    )

    historical_days = set(
        fit["trading_day"].astype(str)
    ) | set(calibration["trading_day"].astype(str))
    historical_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(
            historical_days
        )
    ].copy()
    historical_regime = build_market_regime(
        historical_scan
    )
    fresh_regime = build_market_regime(fresh_scan)
    fit = attach_market_regime(fit, historical_regime)
    calibration = attach_market_regime(
        calibration, historical_regime
    )
    fresh = attach_market_regime(fresh, fresh_regime)

    fit = attach_event_transition_rates(fit)
    calibration = attach_event_transition_rates(calibration)
    fresh = attach_event_transition_rates(fresh)

    model = train_event_hold_model(fit, calibration)
    scored = fresh.copy()
    scored["predicted_hold_advantage_next_event_pct"] = (
        predict_event_hold(scored, model)
    )

    diagnostic_frame = scored.copy()
    diagnostic_frame["hold_advantage_1m_pct"] = (
        diagnostic_frame["hold_advantage_next_event_pct"]
    )
    diagnostic_frame[
        "predicted_hold_advantage_1m_pct"
    ] = diagnostic_frame[
        "predicted_hold_advantage_next_event_pct"
    ]
    diagnostics = advantage_diagnostics(
        diagnostic_frame
    )

    trajectories, coverage = build_event_trajectories(
        scored,
        score_column=(
            "predicted_hold_advantage_next_event_pct"
        ),
    )
    dynamic_trades = dynamic_to_trades(trajectories)
    comparator_trades = fresh_entries.loc[
        :,
        EPISODE_KEYS + ["realized_base_return_pct"],
    ].copy()

    dynamic_metrics = _metrics(dynamic_trades)
    fixed_metrics = _metrics(comparator_trades)
    difference = _matched_difference(
        dynamic_trades, comparator_trades
    )

    completion = (
        float(
            trajectories["status"].eq("completed").mean()
        )
        if len(trajectories)
        else 0.0
    )
    candidate_day = dynamic_metrics.get(
        "day_balanced_mean_pct"
    )
    fixed_day = fixed_metrics.get(
        "day_balanced_mean_pct"
    )
    diff_day = difference.get(
        "day_balanced_difference_pct"
    )
    diff_low = difference.get("bootstrap", {}).get(
        "ci_low_pct"
    )
    spearman = diagnostics.get("advantage_spearman")
    selected_mean = diagnostics.get(
        "selected_realized_advantage_mean_pct"
    )

    gate = bool(
        coverage.get("start_state_coverage") is not None
        and float(coverage["start_state_coverage"])
        >= MIN_START_COVERAGE
        and completion >= 0.90
        and spearman is not None
        and float(spearman) >= MIN_ADVANTAGE_SPEARMAN
        and selected_mean is not None
        and float(selected_mean) > 0
        and int(diagnostics["good_days"]) >= MIN_GOOD_DAYS
        and candidate_day is not None
        and fixed_day is not None
        and float(candidate_day) > 0
        and float(candidate_day) > float(fixed_day)
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
        "entry_policy": (
            "Request208 transition-relative recurrent ENTER/WAIT"
        ),
        "hold_policy": (
            "event-time next-executable-observation "
            "expected HOLD advantage"
        ),
        "fit_selected_entries": int(len(fit_entries)),
        "calibration_selected_entries": int(
            len(calibration_entries)
        ),
        "fresh_selected_entries": int(len(fresh_entries)),
        "fit_event_rows": int(len(fit)),
        "calibration_event_rows": int(len(calibration)),
        "fresh_event_rows": int(len(fresh)),
        "event_gap_diagnostics": event_gap_diagnostics(
            scored
        ),
        "hold_advantage_diagnostics": diagnostics,
        "candidate_coverage": {
            **coverage,
            "completion_coverage": completion,
        },
        "candidate_dynamic_event_exit": dynamic_metrics,
        "comparator_request208_fixed3m": fixed_metrics,
        "matched_dynamic_minus_fixed3m": difference,
        "request212_reference": {
            "dynamic_mean_pct": -1.246285490355756,
            "dynamic_day_balanced_mean_pct": -1.2057664016698741,
            "fixed3m_day_balanced_mean_pct": -1.1142578582498444,
            "hold_advantage_spearman": -0.012274457502498405,
            "selected_realized_advantage_mean_pct": -0.055206504637855404,
            "dynamic_minus_fixed_day_balanced_pct": -0.08458885748618297,
        },
        "frozen_gate": {
            "min_start_coverage": MIN_START_COVERAGE,
            "min_completion_coverage": 0.90,
            "min_advantage_spearman": MIN_ADVANTAGE_SPEARMAN,
            "min_good_days": MIN_GOOD_DAYS,
            "selected_advantage_mean_must_be_positive": True,
            "dynamic_day_balanced_must_be_positive": True,
            "dynamic_must_beat_fixed3m": True,
            "matched_bootstrap_low_must_be_positive": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    trajectories_output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    trajectories.to_parquet(
        trajectories_output_path,
        index=False,
        compression="zstd",
    )
    scored.to_parquet(
        output_path.with_name(
            output_path.stem + "-states.parquet"
        ),
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-positions", type=Path, required=True
    )
    parser.add_argument(
        "--calibration-positions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--history-scan", type=Path, required=True
    )
    parser.add_argument(
        "--fresh-dir", type=Path, required=True
    )
    parser.add_argument(
        "--output", type=Path, required=True
    )
    parser.add_argument(
        "--trajectories-output",
        type=Path,
        required=True,
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
