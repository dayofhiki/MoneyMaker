"""Request 215: event-time executable recurrent entry.

Development-only structural experiment on the already-opened Request178 dates.
Request214 required an exact checkpoint at every minute and an exact +3 minute
exit reference. Request215 keeps the same causal features, model capacity,
relative ENTER-vs-WAIT timing model, BASE costs and five-minute decision window,
but changes the replay clock:

* a silent exact-minute checkpoint is an explicit STALE/WAIT clock state rather
  than an immediate terminal coverage failure;
* ENTER remains possible only at an actually observed/executable checkpoint;
* the fixed three-minute diagnostic exit is filled at the first actually
  executable observation at or after the three-minute target;
* unresolved entries remain unresolved if no later executable observation
  exists, and are never priced as cash.

The fixed exit still isolates entry research. This is not yet a complete
recurrent HOLD/EXIT or portfolio backtest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .executable_recurrent_entry import (
    DECISION_COLUMNS,
    add_execution_labels,
    decide as decide_request214,
    development_gate,
    policy_report,
)
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS, net_round_trip_return_pct
from .opportunity_admission_transition_timing import (
    predict_admission_ev,
    train_admission_model,
)
from .pullback_turn_transition_entry import (
    attach_transition_features,
    predict as predict_timing,
    train_model as train_timing_model,
)
from .recurrent_wait_entry_action_value import (
    MAX_WAIT_MINUTES,
    _base_return,
    build_watch_states,
)
from .relative_recurrent_entry_timing import (
    _safe_spearman,
    attach_relative_advantage,
)
from .supply_admission_transition_timing import _difference_bootstrap

REQUEST_ID = 215
HOLD_MINUTES = 3
EPSILON = 1e-9


def prepare(position_rows: pd.DataFrame) -> pd.DataFrame:
    """Keep Request214 causal state construction and timing features frozen."""
    return attach_transition_features(
        attach_relative_advantage(build_watch_states(position_rows))
    )


def _finite_positive(value: object) -> bool:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(pd.notna(parsed) and np.isfinite(float(parsed)) and float(parsed) > 0)


def _position_paths(
    positions: pd.DataFrame,
) -> dict[tuple[str, str, int], list[tuple[float, float]]]:
    paths: dict[tuple[str, str, int], list[tuple[float, float]]] = {}
    for keys, group in positions.groupby(EPISODE_KEYS, sort=False):
        rows: list[tuple[float, float]] = []
        for _, row in group.iterrows():
            minute = pd.to_numeric(
                pd.Series([row.get("minutes_held")]), errors="coerce"
            ).iloc[0]
            price = pd.to_numeric(
                pd.Series([row.get("exit_reference_open")]), errors="coerce"
            ).iloc[0]
            if (
                pd.notna(minute)
                and np.isfinite(float(minute))
                and _finite_positive(price)
            ):
                rows.append((float(minute), float(price)))
        rows.sort(key=lambda item: item[0])
        paths[(str(keys[0]), str(keys[1]).upper(), int(keys[2]))] = rows
    return paths


def attach_event_time_execution(
    states: pd.DataFrame,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """Replace exact +3m outcomes with first executable observation after +3m.

    The outcome is a label only. No future exit information is added to the
    decision feature set.
    """
    result = states.copy()
    result["enter_3m_exact_base_pct"] = pd.to_numeric(
        result.get("enter_3m_base_pct"), errors="coerce"
    )
    result["enter_3m_event_base_pct"] = np.nan
    result["event_exit_minute_after_hot"] = np.nan
    result["event_exit_delay_after_3m_min"] = np.nan
    result["gross_return_pct"] = np.nan
    result["stress_return_pct"] = np.nan

    paths = _position_paths(positions)
    stress = next(
        scenario for scenario in DEFAULT_EXECUTION_SCENARIOS
        if scenario.name == "stress"
    )

    for index, row in result.iterrows():
        key = (
            str(row["trading_day"]),
            str(row["ticker"]).upper(),
            int(row["hot_t"]),
        )
        entry_minute = int(row["minutes_since_hot"])
        path = paths.get(key, [])
        entry_candidates = [
            (minute, price)
            for minute, price in path
            if abs(minute - float(entry_minute)) <= EPSILON
        ]
        if not entry_candidates:
            continue
        _, entry_price = entry_candidates[0]
        target = float(entry_minute + HOLD_MINUTES)
        exit_candidates = [
            (minute, price)
            for minute, price in path
            if minute + EPSILON >= target
        ]
        if not exit_candidates:
            continue

        exit_minute, exit_price = exit_candidates[0]
        base_return = _base_return(entry_price, exit_price)
        gross_return = (exit_price / entry_price - 1.0) * 100.0
        result.at[index, "enter_3m_event_base_pct"] = base_return
        result.at[index, "event_exit_minute_after_hot"] = exit_minute
        result.at[index, "event_exit_delay_after_3m_min"] = max(
            0.0, exit_minute - target
        )
        result.at[index, "gross_return_pct"] = gross_return
        result.at[index, "stress_return_pct"] = net_round_trip_return_pct(
            entry_price, gross_return, stress
        )

    return result


def _source_keys(source_episodes: pd.DataFrame) -> pd.DataFrame:
    keys = source_episodes.loc[:, EPISODE_KEYS].drop_duplicates().copy()
    keys["ticker"] = keys["ticker"].astype(str).str.upper()
    return keys.sort_values(EPISODE_KEYS, kind="stable").reset_index(drop=True)


def decide_event_time(
    scored: pd.DataFrame,
    source_episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Replay the five-minute entry clock while preserving silent minutes.

    Silent minutes are WAIT states in which no ENTER action can be executed.
    A later observed checkpoint may still trigger an entry. Observed states with
    non-finite model outputs remain explicit prediction failures.
    """
    work = scored.loc[:, DECISION_COLUMNS].copy()
    if work.duplicated(EPISODE_KEYS + ["minutes_since_hot"]).any():
        raise ValueError("duplicate episode checkpoint")

    groups: dict[tuple[str, str, int], dict[int, pd.Series]] = {}
    for keys, group in work.groupby(EPISODE_KEYS, sort=False):
        key = (str(keys[0]), str(keys[1]).upper(), int(keys[2]))
        groups[key] = {
            int(row["minutes_since_hot"]): row
            for _, row in group.sort_values(
                "minutes_since_hot", kind="stable"
            ).iterrows()
        }

    records: list[dict[str, object]] = []
    for _, source in _source_keys(source_episodes).iterrows():
        key = (
            str(source["trading_day"]),
            str(source["ticker"]).upper(),
            int(source["hot_t"]),
        )
        by_minute = groups.get(key, {})
        record: dict[str, object] = {
            "trading_day": key[0],
            "ticker": key[1],
            "hot_t": key[2],
            "action": "ABSTAIN",
            "entry_minute_after_hot": np.nan,
            "wait_actions": 0,
            "stale_wait_actions": 0,
            "observed_wait_actions": 0,
            "observed_checkpoints": 0,
        }
        for minute in range(1, MAX_WAIT_MINUTES + 1):
            row = by_minute.get(minute)
            if row is None:
                record["stale_wait_actions"] += 1
                if minute < MAX_WAIT_MINUTES:
                    record["wait_actions"] += 1
                continue

            record["observed_checkpoints"] += 1
            ev = float(row["predicted_entry_ev_pct"])
            advantage = float(row["predicted_relative_advantage_pct"])
            if not np.isfinite(ev) or (
                minute < MAX_WAIT_MINUTES and not np.isfinite(advantage)
            ):
                record["action"] = "PREDICTION_MISS"
                break

            if ev > 0 and (
                minute == MAX_WAIT_MINUTES or advantage >= 0
            ):
                record["action"] = "ENTER"
                record["entry_minute_after_hot"] = minute
                break

            if minute < MAX_WAIT_MINUTES:
                record["wait_actions"] += 1
                record["observed_wait_actions"] += 1

        if int(record["observed_checkpoints"]) == 0:
            record["action"] = "NO_OBSERVED_STATE"
        records.append(record)

    return pd.DataFrame(
        records,
        columns=EPISODE_KEYS
        + [
            "action",
            "entry_minute_after_hot",
            "wait_actions",
            "stale_wait_actions",
            "observed_wait_actions",
            "observed_checkpoints",
        ],
    )


def clock_diagnostics(
    decisions: pd.DataFrame,
    observed_states: pd.DataFrame,
) -> dict[str, object]:
    episodes = int(len(decisions))
    expected = episodes * MAX_WAIT_MINUTES
    observed = int(
        observed_states.loc[
            observed_states["minutes_since_hot"].between(
                1, MAX_WAIT_MINUTES
            ),
            EPISODE_KEYS + ["minutes_since_hot"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    stale = max(0, expected - observed)
    return {
        "episodes": episodes,
        "expected_clock_checkpoints": expected,
        "observed_checkpoints": observed,
        "stale_clock_checkpoints": stale,
        "stale_checkpoint_rate": (
            float(stale / expected) if expected else None
        ),
        "stale_wait_actions": int(decisions["stale_wait_actions"].sum()),
        "observed_wait_actions": int(
            decisions["observed_wait_actions"].sum()
        ),
        "episodes_with_stale_wait": int(
            decisions["stale_wait_actions"].gt(0).sum()
        ),
        "episodes_without_observed_state": int(
            decisions["observed_checkpoints"].eq(0).sum()
        ),
    }


def _cash_difference(report: dict) -> dict[str, object]:
    daily = report.get("all_opportunity_daily_base_pct")
    if not daily:
        return {}
    candidate = pd.Series(daily, dtype=float)
    comparator = pd.Series(0.0, index=candidate.index, dtype=float)
    return _difference_bootstrap(candidate, comparator)


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    fresh_dir: Path,
    output: Path,
) -> dict[str, object]:
    paths = [
        fit_path,
        calibration_path,
        *sorted(fresh_dir.glob("*-positions.parquet")),
    ]
    if len(paths) != 7:
        raise ValueError(
            "request 215 requires exactly five request178 development shards"
        )

    position_frames = [pd.read_parquet(path) for path in paths]
    fit_positions = position_frames[0]
    calibration_positions = position_frames[1]
    fresh_positions = pd.concat(position_frames[2:], ignore_index=True)

    fit = prepare(fit_positions)
    calibration = prepare(calibration_positions)
    fresh = prepare(fresh_positions)

    found_days = sorted(fresh["trading_day"].astype(str).unique())
    if found_days != list(FRESH_DAYS):
        raise ValueError(
            f"request 215 expected {FRESH_DAYS}, found {found_days}"
        )
    if not (
        max(fit["trading_day"]) < min(calibration["trading_day"])
        <= max(calibration["trading_day"]) < min(fresh["trading_day"])
    ):
        raise ValueError(
            "fit/calibration/evaluation dates must be strictly ordered"
        )

    fit_event = attach_event_time_execution(fit, fit_positions)
    calibration_event = attach_event_time_execution(
        calibration, calibration_positions
    )
    fresh_event = attach_event_time_execution(fresh, fresh_positions)

    timing_model = train_timing_model(fit, calibration)
    exact_value_model = train_admission_model(
        fit, calibration, target_column="enter_3m_base_pct"
    )
    event_value_model = train_admission_model(
        fit_event,
        calibration_event,
        target_column="enter_3m_event_base_pct",
    )

    timing_prediction = predict_timing(fresh, timing_model)
    exact_prediction = predict_admission_ev(
        fresh, exact_value_model
    )[0]
    event_prediction = predict_admission_ev(
        fresh_event, event_value_model
    )[0]

    # Reproduce Request214 semantics on the same frozen inputs.
    exact_scored = add_execution_labels(
        fresh.copy(), fresh_positions
    )
    exact_scored["predicted_relative_advantage_pct"] = timing_prediction
    exact_scored["predicted_entry_ev_pct"] = exact_prediction
    request214_decisions = decide_request214(exact_scored)
    request214_report, _ = policy_report(
        request214_decisions, exact_scored
    )

    # Request215 changes only the event-time execution/accounting semantics.
    event_scored = fresh_event.copy()
    event_scored["predicted_relative_advantage_pct"] = timing_prediction
    event_scored["predicted_entry_ev_pct"] = event_prediction
    event_scored["enter_3m_base_pct"] = pd.to_numeric(
        event_scored["enter_3m_event_base_pct"], errors="coerce"
    )
    source_episodes = fresh_positions.loc[:, EPISODE_KEYS].drop_duplicates()
    event_decisions = decide_event_time(
        event_scored, source_episodes
    )
    event_report, event_settled = policy_report(
        event_decisions, event_scored
    )
    event_vs_cash = _cash_difference(event_report)
    gate = development_gate(event_report, event_vs_cash)

    exact_valid = pd.to_numeric(
        fresh_event["enter_3m_exact_base_pct"], errors="coerce"
    ).notna()
    event_valid = pd.to_numeric(
        fresh_event["enter_3m_event_base_pct"], errors="coerce"
    ).notna()
    delays = pd.to_numeric(
        fresh_event.loc[
            event_valid, "event_exit_delay_after_3m_min"
        ],
        errors="coerce",
    )
    target = pd.to_numeric(
        fresh_event["enter_3m_event_base_pct"], errors="coerce"
    )

    result: dict[str, object] = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "development_gate_pass": bool(gate),
        "promotion_gate_pass": False,
        "change_from_request214": (
            "silent exact-minute checkpoints become STALE/WAIT; "
            "fixed 3m diagnostic exits use first executable observation "
            "at or after the target instead of requiring exact +3m"
        ),
        "frozen_components": {
            "attention_and_hot": True,
            "causal_feature_family": True,
            "relative_entry_timing_model": True,
            "hurdle_entry_model_capacity": True,
            "base_execution_costs": True,
            "max_wait_minutes": MAX_WAIT_MINUTES,
            "diagnostic_hold_minutes": HOLD_MINUTES,
            "decision_boundary": (
                "EV>0 and (relative advantage>=0 or minute=5)"
            ),
        },
        "request214_exact_replay": request214_report,
        "event_time_policy": event_report,
        "event_time_minus_cash": event_vs_cash,
        "clock": clock_diagnostics(event_decisions, fresh_event),
        "outcome_accounting": {
            "watch_states": int(len(fresh_event)),
            "exact_3m_resolved_states": int(exact_valid.sum()),
            "event_time_3m_resolved_states": int(event_valid.sum()),
            "newly_resolved_states": int((event_valid & ~exact_valid).sum()),
            "exact_3m_coverage": float(exact_valid.mean())
            if len(exact_valid)
            else None,
            "event_time_3m_coverage": float(event_valid.mean())
            if len(event_valid)
            else None,
            "event_exit_delay_mean_min": float(delays.mean())
            if delays.notna().any()
            else None,
            "event_exit_delay_p90_min": float(delays.quantile(0.90))
            if delays.notna().any()
            else None,
            "event_exit_delayed_states": int(delays.gt(0).sum()),
            "event_positive_realized_states": int(
                target.loc[event_valid].gt(0).sum()
            ),
            "event_positive_realized_rate": float(
                target.loc[event_valid].gt(0).mean()
            )
            if int(event_valid.sum())
            else None,
        },
        "entry_value_model": {
            "spearman": _safe_spearman(
                fresh_event["enter_3m_event_base_pct"],
                pd.Series(event_prediction, index=fresh_event.index),
            ),
            "mean_predicted_ev_pct": float(
                np.mean(event_prediction)
            ),
            "predicted_ev_range_pct": [
                float(np.min(event_prediction)),
                float(np.max(event_prediction)),
            ],
            "positive_predicted_states": int(
                np.asarray(event_prediction).astype(float).reshape(-1)
                .__gt__(0)
                .sum()
            ),
        },
        "frozen_gate": {
            "min_resolved_trades": 30,
            "min_positive_days": 4,
            "positive_day_balanced_base": True,
            "positive_vs_cash_daily_bootstrap_low": True,
            "complete_episode_and_entry_outcome_accounting": True,
        },
        "runtime": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "scipy",
                    "joblib",
                    "threadpoolctl",
                ]
            },
        },
        "inputs": [
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
        "dates": {
            "fit": sorted(fit["trading_day"].astype(str).unique()),
            "calibration": sorted(
                calibration["trading_day"].astype(str).unique()
            ),
            "development": found_days,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fresh_event.assign(
        predicted_relative_advantage_pct=timing_prediction,
        predicted_entry_ev_pct=event_prediction,
    ).to_parquet(
        output.with_name(output.stem + "-states.parquet"),
        index=False,
        compression="zstd",
    )
    event_settled.to_parquet(
        output.with_name(output.stem + "-decisions.parquet"),
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-positions", type=Path, required=True
    )
    parser.add_argument(
        "--calibration-positions", type=Path, required=True
    )
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.fresh_dir,
        args.output,
    )


if __name__ == "__main__":
    main()
