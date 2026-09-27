"""Request247: learn HOLD versus EXIT after Request246 entry timing.

Admission and entry timing are reconstructed exactly from Requests245/246.
The new position target is one-step chronological continuation: sell at the next
observed state versus sell now. There is no best-future-state target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .causal_tradability_admission import (
    TRAIN_DAYS,
    execution_report,
    fetch_second_paths,
    fit_admission,
    label_tradability,
    probability,
)
from .chronological_action_value import (
    BASE,
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
from .execution_costs import modeled_sell_fill
from .massive_client import MassiveClient
from .second_execution_reconstruction import (
    EXPIRY_MS,
    PRIMARY_LATENCY_MS,
    first_observed_open,
    reconstruct,
)
from .tradability_conditioned_entry_value import (
    ADMISSION_THRESHOLD,
    VALUE_TRAIN_DAYS,
    build_value_targets,
    fit_value_models,
    make_decisions,
    paired_delta,
)

REQUEST_ID = 247
POSITION_TRAIN_DAYS = VALUE_TRAIN_DAYS
TEST_DAYS = CAL_DAYS
MINUTE_MS = 60_000
MIN_POSITION_TARGET_ROWS = 1000
MIN_JOINT_ENTRIES = 20
MIN_ENTRY_FILL_COVERAGE = 0.70
MIN_CLOSED_RESOLUTION = 0.50
MIN_CLOSED_POSITIVE_RATE = 0.50


def one_step_hold_advantage(
    path: tuple[np.ndarray, np.ndarray],
    current_t: int,
    next_t: int,
) -> float:
    times, opens = path
    _, current_ref = first_observed_open(
        times,
        opens,
        decision_t=int(current_t),
        latency_ms=PRIMARY_LATENCY_MS,
        expiry_ms=EXPIRY_MS,
    )
    _, next_ref = first_observed_open(
        times,
        opens,
        decision_t=int(next_t),
        latency_ms=PRIMARY_LATENCY_MS,
        expiry_ms=EXPIRY_MS,
    )
    if current_ref is None or next_ref is None:
        return np.nan
    now_sell = modeled_sell_fill(float(current_ref), BASE)
    later_sell = modeled_sell_fill(float(next_ref), BASE)
    if now_sell <= 0:
        return np.nan
    return float((later_sell / now_sell - 1.0) * 100.0)


def build_position_targets(
    states: pd.DataFrame,
    admission_probs: np.ndarray,
    paths: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    work = states.reset_index(drop=True).copy()
    work["tradability_probability"] = admission_probs
    rows = []
    for key, part in work.groupby(KEYS, sort=False):
        part = part.sort_values("state_t")
        ids = list(part.index)
        day, ticker, _ = key
        path = paths.get(
            (str(day), str(ticker).upper()),
            (np.array([], dtype=np.int64), np.array([], dtype=float)),
        )
        for a, b in zip(ids[:-1], ids[1:]):
            if bool(work.at[a, "terminal"]):
                continue
            record = work.loc[a].to_dict()
            record["next_state_t"] = int(work.at[b, "state_t"])
            record["hold_advantage"] = one_step_hold_advantage(
                path,
                int(work.at[a, "state_t"]),
                int(work.at[b, "state_t"]),
            )
            rows.append(record)
    return pd.DataFrame(rows)


def position_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        c
        for c in dict.fromkeys(
            [*CAUSAL_SOURCE_FEATURES, *PATH_FEATURES, "tradability_probability"]
        )
        if c in frame and pd.to_numeric(frame[c], errors="coerce").notna().any()
    )


def fit_position_model(targets: pd.DataFrame):
    if sorted(targets.trading_day.astype(str).unique()) != POSITION_TRAIN_DAYS:
        raise ValueError("position-value training dates changed")
    valid = np.isfinite(pd.to_numeric(targets.hold_advantage, errors="coerce"))
    train = targets.loc[valid].copy()
    if len(train) < MIN_POSITION_TARGET_ROWS:
        raise ValueError(f"only {len(train)} finite position targets")
    columns = position_columns(train)
    y = train.hold_advantage.to_numpy(float)
    low, high = np.quantile(y, [0.005, 0.995])
    episode_sizes = train.groupby(KEYS)["state_t"].transform("size").to_numpy(float)
    episodes_per_day = train[KEYS].drop_duplicates().groupby("trading_day").size()
    weights = 1 / episode_sizes / train.trading_day.map(episodes_per_day).to_numpy(float)
    weights /= weights.mean()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=60,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261347,
    )
    model.fit(
        features(train, columns),
        np.clip(y, low, high),
        sample_weight=weights,
    )
    support = {
        "rows": int(len(train)),
        "episodes": int(len(train[KEYS].drop_duplicates())),
        "mean_pct": float(np.mean(y)),
        "positive_rate": float(np.mean(y > 0)),
        "winsor_low": float(low),
        "winsor_high": float(high),
        "unresolved_rows": int((~valid).sum()),
    }
    return model, columns, support


def apply_position_policy(
    entry_decisions: pd.DataFrame,
    states: pd.DataFrame,
    admission_probs: np.ndarray,
    position_model,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    work = states.reset_index(drop=True).copy()
    work["tradability_probability"] = admission_probs
    work["predicted_hold_advantage"] = position_model.predict(
        features(work, columns)
    )
    decisions = entry_decisions.copy()
    for idx, decision in decisions.loc[decisions.entered].iterrows():
        mask = (
            work.trading_day.astype(str).eq(str(decision.trading_day))
            & work.ticker.astype(str).eq(str(decision.ticker))
            & pd.to_numeric(work.hot_t).eq(int(decision.hot_t))
        )
        part = work.loc[mask].sort_values("state_t")
        entry_t = int(decision.entry_decision_t)
        later = part.loc[pd.to_numeric(part.state_t).gt(entry_t)]
        chosen = None
        hold_count = 0
        for _, row in later.iterrows():
            if bool(row["terminal"]):
                chosen = int(row["state_t"])
                break
            if float(row["predicted_hold_advantage"]) > 0:
                hold_count += 1
                continue
            chosen = int(row["state_t"])
            break

        decisions.at[idx, "hold_actions"] = hold_count
        if chosen is None:
            decisions.at[idx, "exit_t"] = None
            decisions.at[idx, "exit_decision_t"] = (
                int(decision.hot_t) + EXIT_DEADLINE * MINUTE_MS
            )
            decisions.at[idx, "deadline_liquidation"] = True
            decisions.at[idx, "policy_exit_missing"] = True
        else:
            decisions.at[idx, "exit_t"] = chosen
            decisions.at[idx, "exit_decision_t"] = chosen
            decisions.at[idx, "deadline_liquidation"] = False
            decisions.at[idx, "policy_exit_missing"] = False
    return decisions


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
        cache_dir=Path("data/cache/request247-second-bars"),
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
    entry_targets = build_value_targets(
        value_teacher_scored, value_admission_probs, paths
    )
    entry_models, entry_columns, entry_support = fit_value_models(entry_targets)

    position_admission_probs = probability(
        admission_model, admission_columns, value_states
    )
    position_targets = build_position_targets(
        value_states, position_admission_probs, paths
    )
    position_model, pos_columns, position_support = fit_position_model(
        position_targets
    )

    test_teacher_scored = rollout(test_states, teacher, "full")
    test_admission_probs = probability(
        admission_model, admission_columns, test_teacher_scored
    )
    entry_decisions = make_decisions(
        test_teacher_scored,
        test_admission_probs,
        value_models=entry_models,
        value_model_columns=entry_columns,
    )

    baseline_rows = reconstruct(
        entry_decisions, paths, latency_ms=PRIMARY_LATENCY_MS
    )
    joint_decisions = apply_position_policy(
        entry_decisions,
        test_teacher_scored,
        test_admission_probs,
        position_model,
        pos_columns,
    )
    joint_rows = reconstruct(
        joint_decisions, paths, latency_ms=PRIMARY_LATENCY_MS
    )

    baseline_report = execution_report(baseline_rows)
    joint_report = execution_report(joint_rows)
    paired = paired_delta(joint_rows, baseline_rows)

    joint_mean = joint_report["closed_mean_base_pct"]
    baseline_mean = baseline_report["closed_mean_base_pct"]
    checks = {
        "position_target_support": position_support["rows"] >= MIN_POSITION_TARGET_ROWS,
        "joint_entries": joint_report["entries"] >= MIN_JOINT_ENTRIES,
        "entry_fill_coverage": (
            joint_report["entry_fill_coverage"] >= MIN_ENTRY_FILL_COVERAGE
        ),
        "closed_resolution_rate": (
            joint_report["closed_resolution_rate"] >= MIN_CLOSED_RESOLUTION
        ),
        "closed_mean_beats_request246": (
            joint_mean is not None
            and baseline_mean is not None
            and joint_mean > baseline_mean
        ),
        "paired_mean_beats_request246": (
            (paired["mean_delta_pct"] or 0) > 0
        ),
        "closed_base_mean_positive": (
            joint_mean is not None and joint_mean > 0
        ),
        "closed_positive_rate": (
            (joint_report["closed_positive_rate"] or 0)
            >= MIN_CLOSED_POSITIVE_RATE
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "development_only": True,
        "admission_threshold": ADMISSION_THRESHOLD,
        "entry_model_support": entry_support,
        "position_model_support": position_support,
        "test": {
            "request246_entry_plus_teacher_hold": baseline_report,
            "joint_entry_plus_learned_position": joint_report,
            "paired_comparison": paired,
        },
        "checks": checks,
        "economic_gate_pass": bool(all(checks.values())),
        "api_stats": client.stats.to_dict(),
        "interpretation": (
            "This is one isolated position-policy improvement on top of frozen "
            "Request245/246 admission and entry timing. No joint retraining or "
            "promotion claim is made."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.concat(
        [
            baseline_rows.assign(policy="request246_baseline"),
            joint_rows.assign(policy="learned_position"),
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
