"""Request 212: integrate Request-208 entry timing with recurrent HOLD/EXIT.

Development-only structural diagnostic. Re-anchor historical and fresh POSITION
rows to the entry minute chosen by the Request-208 transition controller, then
retrain the Request-210 one-minute expected-advantage HOLD/EXIT model on those
entry-aligned trajectories. Compare dynamic exits with Request-208's frozen
three-minute exit on the same chosen entries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .direct_recurrent_hold_exit_advantage import (
    EPISODE_KEYS,
    MIN_ADVANTAGE_SPEARMAN,
    MIN_GOOD_DAYS,
    MIN_START_COVERAGE,
    advantage_diagnostics,
    build_trajectories,
)
from .fresh_consensus_hurdle_value import FRESH_DAYS
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
    _policy_trades,
    attach_relative_advantage,
)
from .transition_recurrent_hold_exit import (
    attach_position_transitions,
    predict_transition_advantage,
    train_transition_advantage_model,
)

REQUEST_ID = 212


def score_entry_states(
    position_rows: pd.DataFrame,
    entry_model: object,
) -> pd.DataFrame:
    states = attach_watch_transitions(
        attach_relative_advantage(build_watch_states(position_rows))
    )
    states["predicted_relative_advantage_pct"] = predict_entry(
        states, entry_model
    )
    return states


def choose_entries(scored_states: pd.DataFrame) -> pd.DataFrame:
    entries, _ = _policy_trades(scored_states, recurrent=True)
    return entries.loc[
        :,
        EPISODE_KEYS + ["entry_minute_after_hot", "realized_base_return_pct"],
    ].copy()


def _number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(
        pd.Series([row.get(column)]), errors="coerce"
    ).iloc[0]
    if pd.isna(value) or not np.isfinite(float(value)):
        return np.nan
    return float(value)


def _positive(value: object) -> bool:
    return bool(
        pd.notna(value)
        and np.isfinite(float(value))
        and float(value) > 0
    )


def reanchor_positions(
    position_rows: pd.DataFrame,
    entries: pd.DataFrame,
) -> pd.DataFrame:
    entry_map = {
        (
            str(row["trading_day"]),
            str(row["ticker"]).upper(),
            int(row["hot_t"]),
        ): int(row["entry_minute_after_hot"])
        for _, row in entries.iterrows()
    }
    records: list[dict[str, object]] = []

    for keys, group in position_rows.groupby(EPISODE_KEYS, sort=False):
        key = (str(keys[0]), str(keys[1]).upper(), int(keys[2]))
        chosen_minute = entry_map.get(key)
        if chosen_minute is None:
            continue

        work = group.copy()
        minute = pd.to_numeric(work["minutes_held"], errors="coerce")
        work = work.loc[
            minute.notna() & minute.mod(1).eq(0)
        ].copy()
        work["_minute"] = pd.to_numeric(
            work["minutes_held"], errors="coerce"
        ).astype(int)
        by_minute = {
            int(row["_minute"]): row
            for _, row in work.sort_values(
                "_minute", kind="stable"
            ).iterrows()
        }

        entry_row = by_minute.get(chosen_minute)
        if entry_row is None:
            continue
        new_entry_open = _number(entry_row, "exit_reference_open")
        entry_t = _number(entry_row, "state_t")
        if not _positive(new_entry_open) or not np.isfinite(entry_t):
            continue

        running_high = np.nan
        running_low = np.nan
        max_old_minute = max(by_minute) if by_minute else chosen_minute
        for old_minute in range(chosen_minute + 1, max_old_minute + 1):
            row = by_minute.get(old_minute)
            if row is None:
                continue

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

            current_exit_open = _number(row, "exit_reference_open")
            next_row = by_minute.get(old_minute + 1)
            next_exit_open = (
                _number(next_row, "exit_reference_open")
                if next_row is not None
                else np.nan
            )
            exit_now = _base_return(
                new_entry_open, current_exit_open
            )
            next_exit = _base_return(
                new_entry_open, next_exit_open
            )

            record = row.drop(
                labels=["_minute"], errors="ignore"
            ).to_dict()
            new_minute = old_minute - chosen_minute
            record["entry_actual_t"] = int(entry_t)
            record["entry_open"] = float(new_entry_open)
            record["log_entry_price"] = float(
                np.log(new_entry_open)
            )
            record["minutes_held"] = float(new_minute)
            record["exit_now_base_return_pct"] = exit_now
            record["next_minute_base_return_pct"] = next_exit
            record["hold_advantage_1m_pct"] = (
                float(next_exit - exit_now)
                if pd.notna(next_exit) and pd.notna(exit_now)
                else np.nan
            )
            record["entry_to_current_close_pct"] = (
                float(current_close / new_entry_open - 1.0) * 100.0
                if _positive(current_close)
                else np.nan
            )
            record["running_max_return_pct"] = (
                float(running_high / new_entry_open - 1.0) * 100.0
                if _positive(running_high)
                else np.nan
            )
            record["running_min_return_pct"] = (
                float(running_low / new_entry_open - 1.0) * 100.0
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

    return pd.DataFrame(records)


def dynamic_to_trades(
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    completed = trajectories.loc[
        trajectories["status"].eq("completed")
    ].copy()
    if completed.empty:
        return pd.DataFrame(
            columns=EPISODE_KEYS + ["realized_base_return_pct"]
        )
    return completed.loc[
        :, EPISODE_KEYS + ["base_net_return_pct"]
    ].rename(
        columns={"base_net_return_pct": "realized_base_return_pct"}
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

    fit_entry_states = attach_watch_transitions(
        attach_relative_advantage(build_watch_states(fit_positions))
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
    fit = reanchor_positions(fit_positions, fit_entries)
    calibration = reanchor_positions(
        calibration_positions, calibration_entries
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
            "request 212 requires complete request 178 shards"
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
    fresh = reanchor_positions(
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
    historical_regime = build_market_regime(historical_scan)
    fresh_regime = build_market_regime(fresh_scan)
    fit = attach_market_regime(fit, historical_regime)
    calibration = attach_market_regime(
        calibration, historical_regime
    )
    fresh = attach_market_regime(fresh, fresh_regime)

    fit = attach_position_transitions(fit)
    calibration = attach_position_transitions(calibration)
    fresh = attach_position_transitions(fresh)

    hold_model = train_transition_advantage_model(
        fit, calibration
    )
    scored = fresh.copy()
    scored["predicted_hold_advantage_1m_pct"] = (
        predict_transition_advantage(scored, hold_model)
    )
    diagnostics = advantage_diagnostics(scored)

    candidate, coverage = build_trajectories(
        scored,
        score_column="predicted_hold_advantage_1m_pct",
        threshold=0.0,
        policy="request208_entry_plus_dynamic_hold_exit",
    )
    candidate_trades = dynamic_to_trades(candidate)
    comparator_trades = fresh_entries.loc[
        :,
        EPISODE_KEYS + ["realized_base_return_pct"],
    ].copy()

    candidate_metrics = _metrics(candidate_trades)
    comparator_metrics = _metrics(comparator_trades)
    difference = _matched_difference(
        candidate_trades, comparator_trades
    )

    spearman = diagnostics["advantage_spearman"]
    selected_mean = diagnostics[
        "selected_realized_advantage_mean_pct"
    ]
    candidate_day = candidate_metrics.get(
        "day_balanced_mean_pct"
    )
    comparator_day = comparator_metrics.get(
        "day_balanced_mean_pct"
    )
    diff_day = difference.get(
        "day_balanced_difference_pct"
    )
    diff_low = difference.get("bootstrap", {}).get(
        "ci_low_pct"
    )

    gate = bool(
        coverage["start_state_coverage"] is not None
        and float(coverage["start_state_coverage"])
        >= MIN_START_COVERAGE
        and spearman is not None
        and float(spearman) >= MIN_ADVANTAGE_SPEARMAN
        and selected_mean is not None
        and float(selected_mean) > 0
        and int(diagnostics["good_days"]) >= MIN_GOOD_DAYS
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
        "diagnostic_only_reason": (
            "entry model is reused on development phases; "
            "fresh validation remains previously opened"
        ),
        "entry_policy": (
            "Request208 transition-relative recurrent ENTER/WAIT"
        ),
        "hold_policy": (
            "Request210 transition one-minute expected advantage"
        ),
        "fit_selected_entries": int(len(fit_entries)),
        "calibration_selected_entries": int(
            len(calibration_entries)
        ),
        "fresh_selected_entries": int(len(fresh_entries)),
        "fit_reanchored_rows": int(len(fit)),
        "calibration_reanchored_rows": int(
            len(calibration)
        ),
        "fresh_reanchored_rows": int(len(fresh)),
        "hold_advantage_diagnostics": diagnostics,
        "candidate_coverage": coverage,
        "candidate_dynamic_exit": candidate_metrics,
        "comparator_request208_fixed3m": comparator_metrics,
        "matched_dynamic_minus_fixed3m": difference,
        "frozen_gate": {
            "min_start_coverage": MIN_START_COVERAGE,
            "min_advantage_spearman": (
                MIN_ADVANTAGE_SPEARMAN
            ),
            "min_good_days": MIN_GOOD_DAYS,
            "selected_advantage_mean_must_be_positive": True,
            "dynamic_must_beat_fixed3m": True,
            "matched_bootstrap_low_must_be_positive": True,
            "candidate_day_balanced_must_be_positive": True,
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
    candidate.to_parquet(
        trajectories_output_path,
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
