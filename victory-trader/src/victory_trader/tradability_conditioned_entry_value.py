"""Request246: learn ENTER versus WAIT value inside the frozen tradable set.

Admission is the frozen Request245 model/0.60 threshold. The downstream HOLD/EXIT
teacher is the Request243 earlier teacher trained only on Apr30-May4. Entry-value
models train only on May5-8 and are evaluated on already-open May11-20.
No future-best-state target is used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .causal_tradability_admission import (
    TRAIN_DAYS,
    THRESHOLD_DAYS,
    execution_report,
    fetch_second_paths,
    fit_admission,
    label_tradability,
    probability,
)
from .chronological_action_value import (
    BASE,
    STRESS,
    CAL_DAYS,
    CAUSAL_SOURCE_FEATURES,
    EXIT_DEADLINE,
    FIT_DAYS,
    KEYS,
    PATH_FEATURES,
    build_states,
    features,
    fit_policy,
    rollout,
)
from .config import load_settings
from .massive_client import MassiveClient
from .second_execution_reconstruction import (
    EXPIRY_MS,
    PRIMARY_LATENCY_MS,
    first_observed_open,
    modeled_return,
    reconstruct,
)

REQUEST_ID = 246
ADMISSION_THRESHOLD = 0.60
VALUE_TRAIN_DAYS = THRESHOLD_DAYS
TEST_DAYS = CAL_DAYS
MINUTE_MS = 60_000
MIN_ENTER_TARGET_ROWS = 150
MIN_WAIT_TARGET_ROWS = 150
MIN_LEARNED_ENTRIES = 20
MIN_ENTRY_FILL_COVERAGE = 0.70
MIN_CLOSED_RESOLUTION = 0.50
MIN_CLOSED_POSITIVE_RATE = 0.50


def downstream_exit_t(part: pd.DataFrame, state_index: int) -> int:
    later = part.loc[part.index > state_index]
    exits = later.loc[later.held_action.astype(str).eq("EXIT")]
    if len(exits):
        return int(exits.iloc[0].state_t)
    return int(part.iloc[0].hot_t) + EXIT_DEADLINE * MINUTE_MS


def executable_return(
    *,
    path: tuple[np.ndarray, np.ndarray],
    entry_t: int,
    exit_t: int,
    scenario=BASE,
) -> float:
    times, opens = path
    _, entry_ref = first_observed_open(
        times,
        opens,
        decision_t=int(entry_t),
        latency_ms=PRIMARY_LATENCY_MS,
        expiry_ms=EXPIRY_MS,
    )
    if entry_ref is None:
        return np.nan
    _, exit_ref = first_observed_open(
        times,
        opens,
        decision_t=int(exit_t),
        latency_ms=PRIMARY_LATENCY_MS,
        expiry_ms=EXPIRY_MS,
    )
    if exit_ref is None:
        return np.nan
    return modeled_return(float(entry_ref), float(exit_ref), scenario)


def build_value_targets(
    teacher_scored: pd.DataFrame,
    admission_probs: np.ndarray,
    paths: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    work = teacher_scored.reset_index(drop=True).copy()
    work["tradability_probability"] = admission_probs
    rows = []
    for key, part in work.groupby(KEYS, sort=False):
        part = part.sort_values("state_t")
        admitted_indices = [
            idx
            for idx, row in part.iterrows()
            if bool(row["can_enter"])
            and not bool(row["terminal"])
            and float(row["tradability_probability"]) >= ADMISSION_THRESHOLD
        ]
        value_by_index: dict[int, float] = {}
        stress_by_index: dict[int, float] = {}
        day, ticker, _ = key
        path = paths.get(
            (str(day), str(ticker).upper()),
            (np.array([], dtype=np.int64), np.array([], dtype=float)),
        )
        for idx in admitted_indices:
            exit_t = downstream_exit_t(part, idx)
            value_by_index[idx] = executable_return(
                path=path,
                entry_t=int(work.at[idx, "state_t"]),
                exit_t=exit_t,
                scenario=BASE,
            )
            stress_by_index[idx] = executable_return(
                path=path,
                entry_t=int(work.at[idx, "state_t"]),
                exit_t=exit_t,
                scenario=STRESS,
            )

        for position, idx in enumerate(admitted_indices):
            row = work.loc[idx].to_dict()
            row["enter_value"] = value_by_index[idx]
            row["enter_stress_value"] = stress_by_index[idx]
            if position + 1 < len(admitted_indices):
                next_idx = admitted_indices[position + 1]
                row["wait_value"] = value_by_index[next_idx]
                row["wait_target_source"] = "next_admitted_enter"
            else:
                row["wait_value"] = 0.0
                row["wait_target_source"] = "cash_no_later_admitted_state"
            rows.append(row)
    return pd.DataFrame(rows)


def value_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        c
        for c in dict.fromkeys(
            [*CAUSAL_SOURCE_FEATURES, *PATH_FEATURES, "tradability_probability"]
        )
        if c in frame and pd.to_numeric(frame[c], errors="coerce").notna().any()
    )


def fit_value_models(targets: pd.DataFrame):
    if sorted(targets.trading_day.astype(str).unique()) != VALUE_TRAIN_DAYS:
        raise ValueError("entry-value training dates changed")
    columns = value_columns(targets)
    models = {}
    support = {}
    for j, target in enumerate(("enter_value", "wait_value")):
        valid = np.isfinite(pd.to_numeric(targets[target], errors="coerce"))
        train = targets.loc[valid].copy()
        episodes = len(train[KEYS].drop_duplicates())
        minimum = MIN_ENTER_TARGET_ROWS if target == "enter_value" else MIN_WAIT_TARGET_ROWS
        if len(train) < minimum:
            raise ValueError(f"{target}: only {len(train)} finite targets")
        y = train[target].to_numpy(float)
        low, high = np.quantile(y, [0.005, 0.995])
        episode_sizes = train.groupby(KEYS)["state_t"].transform("size").to_numpy(float)
        episodes_per_day = train[KEYS].drop_duplicates().groupby("trading_day").size()
        weights = (
            1
            / episode_sizes
            / train.trading_day.map(episodes_per_day).to_numpy(float)
        )
        weights /= weights.mean()
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=7,
            min_samples_leaf=40,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=20261346 + j,
        )
        model.fit(
            features(train, columns),
            np.clip(y, low, high),
            sample_weight=weights,
        )
        models[target] = model
        support[target] = {
            "rows": int(len(train)),
            "episodes": int(episodes),
            "mean_pct": float(np.mean(y)),
            "positive_rate": float(np.mean(y > 0)),
            "winsor_low": float(low),
            "winsor_high": float(high),
        }
    return models, columns, support


def predict_values(models, columns, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = features(frame, columns)
    return (
        models["enter_value"].predict(x),
        models["wait_value"].predict(x),
    )


def make_decisions(
    teacher_scored: pd.DataFrame,
    admission_probs: np.ndarray,
    *,
    value_models=None,
    value_model_columns: tuple[str, ...] = (),
    first_admitted: bool = False,
) -> pd.DataFrame:
    work = teacher_scored.reset_index(drop=True).copy()
    work["tradability_probability"] = admission_probs
    if value_models is not None:
        q_enter, q_wait = predict_values(value_models, value_model_columns, work)
        work["predicted_enter_value"] = q_enter
        work["predicted_wait_value"] = q_wait
    else:
        work["predicted_enter_value"] = np.nan
        work["predicted_wait_value"] = np.nan

    rows = []
    for key, part in work.groupby(KEYS, sort=False):
        part = part.sort_values("state_t")
        entry_index = None
        wait_actions = 0
        for idx, row in part.iterrows():
            if not bool(row["can_enter"]) or bool(row["terminal"]):
                continue
            admitted = (
                float(row["tradability_probability"]) >= ADMISSION_THRESHOLD
            )
            if not admitted:
                wait_actions += 1
                continue
            if first_admitted:
                entry_index = idx
                break
            enter_q = float(row["predicted_enter_value"])
            wait_q = float(row["predicted_wait_value"])
            if enter_q > 0 and enter_q >= wait_q:
                entry_index = idx
                break
            wait_actions += 1

        day, ticker, hot_t = key
        record = {
            "trading_day": day,
            "ticker": ticker,
            "hot_t": hot_t,
            "entered": entry_index is not None,
            "unresolved": False,
            "net_pct": np.nan,
            "stress_pct": np.nan,
            "entry_t": None,
            "exit_t": None,
            "wait_actions": wait_actions,
            "hold_actions": 0,
            "entry_decision_t": None,
            "exit_decision_t": None,
            "deadline_liquidation": False,
            "policy_exit_missing": False,
            "entry_tradability_probability": None,
            "predicted_enter_value": None,
            "predicted_wait_value": None,
        }
        if entry_index is None:
            rows.append(record)
            continue

        entry_row = work.loc[entry_index]
        record["entry_t"] = int(entry_row["state_t"])
        record["entry_decision_t"] = int(entry_row["state_t"])
        record["entry_tradability_probability"] = float(
            entry_row["tradability_probability"]
        )
        if value_models is not None:
            record["predicted_enter_value"] = float(
                entry_row["predicted_enter_value"]
            )
            record["predicted_wait_value"] = float(
                entry_row["predicted_wait_value"]
            )

        later = part.loc[part.index > entry_index]
        exits = later.loc[later.held_action.astype(str).eq("EXIT")]
        if len(exits):
            exit_index = exits.index[0]
            record["exit_t"] = int(work.at[exit_index, "state_t"])
            record["exit_decision_t"] = int(work.at[exit_index, "state_t"])
            between = later.loc[later.index < exit_index]
            record["hold_actions"] = int(
                between.held_action.astype(str).eq("HOLD").sum()
            )
        else:
            record["policy_exit_missing"] = True
            record["deadline_liquidation"] = True
            record["exit_decision_t"] = int(hot_t) + EXIT_DEADLINE * MINUTE_MS
            record["hold_actions"] = int(
                later.held_action.astype(str).eq("HOLD").sum()
            )
        rows.append(record)
    return pd.DataFrame(rows)


def paired_delta(learned: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    left = learned.loc[
        learned.execution_status.eq("closed"),
        [*KEYS, "reconstructed_base_pct"],
    ].rename(columns={"reconstructed_base_pct": "learned_pct"})
    right = baseline.loc[
        baseline.execution_status.eq("closed"),
        [*KEYS, "reconstructed_base_pct"],
    ].rename(columns={"reconstructed_base_pct": "baseline_pct"})
    paired = left.merge(right, on=KEYS, how="inner")
    return {
        "paired_closed_episodes": int(len(paired)),
        "mean_delta_pct": (
            float((paired.learned_pct - paired.baseline_pct).mean())
            if len(paired)
            else None
        ),
    }


def evaluate(
    fit_positions: Path,
    calibration_positions: Path,
    output: Path,
    rows_output: Path,
) -> int:
    fit_states = build_states(pd.read_parquet(fit_positions))
    test_states = build_states(pd.read_parquet(calibration_positions))
    if sorted(fit_states.trading_day.astype(str).unique()) != FIT_DAYS:
        raise ValueError("fit dates changed")
    if sorted(test_states.trading_day.astype(str).unique()) != TEST_DAYS:
        raise ValueError("test dates changed")

    all_states = pd.concat([fit_states, test_states], ignore_index=True)
    client = MassiveClient(
        load_settings().massive_api_key,
        cache_dir=Path("data/cache/request246-second-bars"),
        request_interval_seconds=0.05,
    )
    paths = fetch_second_paths(all_states, client)
    labeled = label_tradability(all_states, paths)

    early_labeled = labeled.loc[
        labeled.trading_day.astype(str).isin(TRAIN_DAYS)
    ].copy()
    admission_model, admission_columns = fit_admission(early_labeled)

    early_states = fit_states.loc[
        fit_states.trading_day.astype(str).isin(TRAIN_DAYS)
    ].reset_index(drop=True)
    teacher = fit_policy(rollout(early_states, None))

    value_states = fit_states.loc[
        fit_states.trading_day.astype(str).isin(VALUE_TRAIN_DAYS)
    ].reset_index(drop=True)
    value_teacher_scored = rollout(value_states, teacher, "full")
    value_admission_probs = probability(
        admission_model, admission_columns, value_teacher_scored
    )
    targets = build_value_targets(
        value_teacher_scored, value_admission_probs, paths
    )
    value_models, value_model_columns, support = fit_value_models(targets)

    test_teacher_scored = rollout(test_states, teacher, "full")
    test_admission_probs = probability(
        admission_model, admission_columns, test_teacher_scored
    )

    learned_decisions = make_decisions(
        test_teacher_scored,
        test_admission_probs,
        value_models=value_models,
        value_model_columns=value_model_columns,
    )
    baseline_decisions = make_decisions(
        test_teacher_scored,
        test_admission_probs,
        first_admitted=True,
    )
    learned_rows = reconstruct(
        learned_decisions, paths, latency_ms=PRIMARY_LATENCY_MS
    )
    baseline_rows = reconstruct(
        baseline_decisions, paths, latency_ms=PRIMARY_LATENCY_MS
    )
    learned_report = execution_report(learned_rows)
    baseline_report = execution_report(baseline_rows)
    paired = paired_delta(learned_rows, baseline_rows)

    learned_mean = learned_report["closed_mean_base_pct"]
    baseline_mean = baseline_report["closed_mean_base_pct"]
    checks = {
        "enter_target_support": support["enter_value"]["rows"] >= MIN_ENTER_TARGET_ROWS,
        "wait_target_support": support["wait_value"]["rows"] >= MIN_WAIT_TARGET_ROWS,
        "learned_entries": learned_report["entries"] >= MIN_LEARNED_ENTRIES,
        "entry_fill_coverage": (
            learned_report["entry_fill_coverage"] >= MIN_ENTRY_FILL_COVERAGE
        ),
        "closed_resolution_rate": (
            learned_report["closed_resolution_rate"] >= MIN_CLOSED_RESOLUTION
        ),
        "closed_mean_beats_first_admitted": (
            learned_mean is not None
            and baseline_mean is not None
            and learned_mean > baseline_mean
        ),
        "paired_mean_beats_first_admitted": (
            (paired["mean_delta_pct"] or 0) > 0
        ),
        "closed_base_mean_positive": (
            learned_mean is not None and learned_mean > 0
        ),
        "closed_positive_rate": (
            (learned_report["closed_positive_rate"] or 0)
            >= MIN_CLOSED_POSITIVE_RATE
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "development_only": True,
        "admission_threshold": ADMISSION_THRESHOLD,
        "admission_training_days": TRAIN_DAYS,
        "value_training_days": VALUE_TRAIN_DAYS,
        "test_days": TEST_DAYS,
        "downstream_teacher_days": TRAIN_DAYS,
        "value_feature_count": len(value_model_columns),
        "support": support,
        "training_targets": {
            "admitted_states": int(len(targets)),
            "enter_finite": int(np.isfinite(targets.enter_value).sum()),
            "wait_finite": int(np.isfinite(targets.wait_value).sum()),
            "enter_mean_pct": float(
                pd.to_numeric(targets.enter_value, errors="coerce").mean()
            ),
            "wait_mean_pct": float(
                pd.to_numeric(targets.wait_value, errors="coerce").mean()
            ),
        },
        "test": {
            "first_admitted_baseline": baseline_report,
            "learned_enter_wait": learned_report,
            "paired_comparison": paired,
        },
        "checks": checks,
        "economic_gate_pass": bool(all(checks.values())),
        "api_stats": client.stats.to_dict(),
        "interpretation": (
            "Tradability is frozen from Request245. This request tests entry timing "
            "only under one frozen earlier HOLD/EXIT teacher. It is development "
            "evidence, not promotion or live-readiness evidence."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.concat(
        [
            baseline_rows.assign(policy="first_admitted"),
            learned_rows.assign(policy="learned_enter_wait"),
        ],
        ignore_index=True,
    ).to_parquet(rows_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument("--calibration-positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
