"""Request246: direct action-advantage classification.

Replaces Request243's comparison of independently scaled value regressors with
binary action-preference questions. Training targets are forced-action realized
continuations under an earlier chronological teacher, never best-future prices.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .chronological_action_value import (
    BASE,
    CAL_DAYS,
    CAUSAL_SOURCE_FEATURES,
    KEYS,
    PATH_FEATURES,
    build_states,
    chronological_fit,
    episode_results,
    features,
    metrics,
    modeled_sell_fill,
    net,
    rollout,
)
from .action_value_calibration_audit import reachable_masks

REQUEST_ID = 246
THRESHOLD = 0.5
MIN_ENTRIES = 20
MIN_RESOLVED = 20
MIN_WAIT_ACTIONS = 5
MIN_ACTION_PRECISION = 0.55
MAX_SEVERE_LOSS = 0.48


@dataclass
class ConstantBinary:
    probability: float

    def predict_proba(self, x):
        p = np.repeat(float(self.probability), len(x))
        return np.column_stack([1 - p, p])


@dataclass
class DirectPolicy:
    enter_model: object
    wait_model: object
    hold_model: object
    columns: tuple[str, ...]
    support: dict


def finite(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.notna() & np.isfinite(x)


def episode_day_weights(frame: pd.DataFrame) -> np.ndarray:
    sizes = frame.groupby(KEYS)["state_t"].transform("size").to_numpy(float)
    ep_per_day = frame[KEYS].drop_duplicates().groupby("trading_day").size()
    weights = 1 / sizes / frame.trading_day.map(ep_per_day).to_numpy(float)
    return weights / weights.mean()


def fit_binary(frame: pd.DataFrame, label: pd.Series, columns, seed: int):
    work = frame.copy()
    y = label.loc[work.index].astype(int)
    positive = int(y.sum())
    negative = int(len(y) - positive)
    support = {
        "rows": int(len(work)),
        "episodes": int(work[KEYS].drop_duplicates().shape[0]),
        "positive_rows": positive,
        "negative_rows": negative,
        "positive_rate": float(y.mean()) if len(y) else None,
    }
    if not len(work):
        return ConstantBinary(0.0), support
    if positive == 0 or negative == 0:
        return ConstantBinary(float(positive > 0)), support

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=60,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(
        features(work, columns),
        y.to_numpy(),
        sample_weight=episode_day_weights(work),
    )
    return model, support


def positive_probability(model, x) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] != 2:
        raise ValueError("binary model must expose two probabilities")
    return np.asarray(proba[:, 1], dtype=float)


def fit_direct_policy(student_targets: pd.DataFrame) -> DirectPolicy:
    columns = tuple(
        c
        for c in dict.fromkeys([*CAUSAL_SOURCE_FEATURES, *PATH_FEATURES])
        if c in student_targets
        and pd.to_numeric(student_targets[c], errors="coerce").notna().any()
    )

    flat_valid = (
        student_targets.can_enter.astype(bool)
        & ~student_targets.terminal.astype(bool)
        & finite(student_targets.enter_value)
        & finite(student_targets.wait_value)
    )
    flat = student_targets.loc[flat_valid].copy()
    enter_label = pd.to_numeric(flat.enter_value).gt(
        np.maximum(pd.to_numeric(flat.wait_value), 0.0)
    )
    enter_model, enter_support = fit_binary(
        flat, enter_label, columns, seed=20261460
    )

    wait_population = flat.loc[~enter_label].copy()
    wait_label = pd.to_numeric(wait_population.wait_value).gt(0)
    wait_model, wait_support = fit_binary(
        wait_population, wait_label, columns, seed=20261461
    )

    hold_valid = (
        ~student_targets.terminal.astype(bool)
        & finite(student_targets.hold_advantage)
    )
    hold_frame = student_targets.loc[hold_valid].copy()
    hold_label = pd.to_numeric(hold_frame.hold_advantage).gt(0)
    hold_model, hold_support = fit_binary(
        hold_frame, hold_label, columns, seed=20261462
    )

    return DirectPolicy(
        enter_model=enter_model,
        wait_model=wait_model,
        hold_model=hold_model,
        columns=columns,
        support={
            "enter_best": enter_support,
            "wait_positive_given_enter_not_best": wait_support,
            "hold_better": hold_support,
        },
    )


def direct_actions(frame: pd.DataFrame, policy: DirectPolicy):
    x = features(frame, policy.columns)
    p_enter = positive_probability(policy.enter_model, x)
    p_wait = positive_probability(policy.wait_model, x)
    p_hold = positive_probability(policy.hold_model, x)

    allowed = frame.can_enter.to_numpy(bool) & ~frame.terminal.to_numpy(bool)
    enter = allowed & (p_enter >= THRESHOLD)
    wait = allowed & ~enter & (p_wait >= THRESHOLD)
    flat = np.where(enter, "ENTER", np.where(wait, "WAIT", "ABSTAIN"))
    held = np.where(
        (p_hold >= THRESHOLD) & ~frame.terminal.to_numpy(bool),
        "HOLD",
        "EXIT",
    )
    return flat, held, p_enter, p_wait, p_hold


def rollout_direct(frame: pd.DataFrame, policy: DirectPolicy) -> pd.DataFrame:
    out = frame.reset_index(drop=True).copy()
    decision_columns = list(
        dict.fromkeys(
            [*CAUSAL_SOURCE_FEATURES, *PATH_FEATURES, "can_enter", "terminal"]
        )
    )
    flat, held, p_enter, p_wait, p_hold = direct_actions(
        out[decision_columns], policy
    )
    out["p_enter_best"] = p_enter
    out["p_wait_positive"] = p_wait
    out["p_hold_better"] = p_hold
    out["flat_action"] = flat
    out["held_action"] = held
    out["enter_value"] = np.nan
    out["wait_value"] = np.nan
    out["hold_advantage"] = np.nan
    out["policy_value"] = np.nan
    out["policy_entry_index"] = -1
    out["policy_exit_index"] = -1
    out["forced_enter_exit_index"] = -1

    for _, group in out.groupby(KEYS, sort=False):
        ids = list(group.index)
        n = len(ids)
        held_exit = [-1] * n
        flat_value = [0.0] * n
        flat_entry = [-1] * n
        flat_exit = [-1] * n
        for k in range(n - 1, -1, -1):
            i = ids[k]
            next_exit = held_exit[k + 1] if k + 1 < n else -1
            held_exit[k] = i if held[i] == "EXIT" else next_exit

            enter_value = np.nan
            hold_advantage = np.nan
            if next_exit >= 0:
                enter_value = net(
                    out.at[i, "execution_open"],
                    out.at[next_exit, "execution_open"],
                )
                now = out.at[i, "execution_open"]
                later = out.at[next_exit, "execution_open"]
                if (
                    pd.notna(now)
                    and pd.notna(later)
                    and np.isfinite(float(now))
                    and np.isfinite(float(later))
                    and float(now) > 0
                    and float(later) > 0
                ):
                    now_sell = modeled_sell_fill(float(now), BASE)
                    if now_sell > 0:
                        hold_advantage = (
                            modeled_sell_fill(float(later), BASE) / now_sell - 1
                        ) * 100

            wait_value = flat_value[k + 1] if k + 1 < n else 0.0
            if bool(out.at[i, "can_enter"]) and not bool(out.at[i, "terminal"]):
                out.at[i, "enter_value"] = enter_value
                out.at[i, "wait_value"] = wait_value
                out.at[i, "forced_enter_exit_index"] = next_exit
            if not bool(out.at[i, "terminal"]):
                out.at[i, "hold_advantage"] = hold_advantage

            if flat[i] == "ENTER":
                flat_value[k] = enter_value
                flat_entry[k] = i
                flat_exit[k] = next_exit
            elif flat[i] == "WAIT" and k + 1 < n:
                flat_value[k] = flat_value[k + 1]
                flat_entry[k] = flat_entry[k + 1]
                flat_exit[k] = flat_exit[k + 1]

            out.at[i, "policy_value"] = flat_value[k]
            out.at[i, "policy_entry_index"] = flat_entry[k]
            out.at[i, "policy_exit_index"] = flat_exit[k]
    return out


def decision_quality(scored: pd.DataFrame) -> dict:
    flat_reachable, position_reachable = reachable_masks(scored)
    flat_valid = (
        flat_reachable
        & scored.can_enter.astype(bool)
        & ~scored.terminal.astype(bool)
        & finite(scored.enter_value)
        & finite(scored.wait_value)
    )
    flat = scored.loc[flat_valid].copy()
    enter = flat.flat_action.eq("ENTER")
    wait = flat.flat_action.eq("WAIT")
    abstain = flat.flat_action.eq("ABSTAIN")
    enter_adv = pd.to_numeric(flat.enter_value) - np.maximum(
        pd.to_numeric(flat.wait_value), 0.0
    )
    wait_adv = pd.to_numeric(flat.wait_value) - np.maximum(
        pd.to_numeric(flat.enter_value), 0.0
    )
    best = np.where(
        pd.to_numeric(flat.enter_value) > np.maximum(pd.to_numeric(flat.wait_value), 0.0),
        "ENTER",
        np.where(
            pd.to_numeric(flat.wait_value) > np.maximum(pd.to_numeric(flat.enter_value), 0.0),
            "WAIT",
            "ABSTAIN",
        ),
    )

    hold_valid = (
        position_reachable
        & finite(scored.hold_advantage)
    )
    position = scored.loc[hold_valid].copy()
    chosen_hold = position.held_action.eq("HOLD")

    return {
        "flat_comparable_rows": int(len(flat)),
        "flat_best_action_accuracy": (
            float(np.mean(flat.flat_action.astype(str).to_numpy() == best))
            if len(flat)
            else None
        ),
        "chosen_enter_rows": int(enter.sum()),
        "chosen_enter_advantage_positive_rate": (
            float(enter_adv.loc[enter].gt(0).mean()) if enter.any() else None
        ),
        "chosen_enter_mean_advantage_pct": (
            float(enter_adv.loc[enter].mean()) if enter.any() else None
        ),
        "chosen_wait_rows": int(wait.sum()),
        "chosen_wait_advantage_positive_rate": (
            float(wait_adv.loc[wait].gt(0).mean()) if wait.any() else None
        ),
        "chosen_wait_mean_advantage_pct": (
            float(wait_adv.loc[wait].mean()) if wait.any() else None
        ),
        "chosen_abstain_rows": int(abstain.sum()),
        "position_comparable_rows": int(len(position)),
        "chosen_hold_rows": int(chosen_hold.sum()),
        "chosen_hold_advantage_positive_rate": (
            float(pd.to_numeric(position.loc[chosen_hold, "hold_advantage"]).gt(0).mean())
            if chosen_hold.any()
            else None
        ),
        "chosen_hold_mean_advantage_pct": (
            float(pd.to_numeric(position.loc[chosen_hold, "hold_advantage"]).mean())
            if chosen_hold.any()
            else None
        ),
    }


def passes(value, threshold):
    return value is not None and np.isfinite(value) and value >= threshold


def evaluate(
    fit_positions: Path,
    calibration_positions: Path,
    output: Path,
    rows_output: Path,
) -> int:
    fit_states = build_states(pd.read_parquet(fit_positions))
    cal_states = build_states(pd.read_parquet(calibration_positions))
    if sorted(cal_states.trading_day.astype(str).unique()) != CAL_DAYS:
        raise ValueError("calibration dates changed")

    _, request243_student, student_targets = chronological_fit(fit_states)
    policy = fit_direct_policy(student_targets)

    direct_scored = rollout_direct(cal_states, policy)
    direct_episodes = episode_results(direct_scored)
    direct_metrics = metrics(direct_episodes, CAL_DAYS)
    quality = decision_quality(direct_scored)

    old_scored = rollout(cal_states, request243_student, "full")
    old_episodes = episode_results(old_scored)
    old_metrics = metrics(old_episodes, CAL_DAYS)

    checks = {
        "min_entries": direct_metrics["entries"] >= MIN_ENTRIES,
        "min_resolved_trades": direct_metrics["resolved_trades"] >= MIN_RESOLVED,
        "positive_resolved_trade_mean": (
            direct_metrics["resolved_trade_mean_pct"] is not None
            and direct_metrics["resolved_trade_mean_pct"] > 0
        ),
        "uses_wait": direct_metrics["wait_actions"] >= MIN_WAIT_ACTIONS,
        "chosen_enter_advantage_precision": passes(
            quality["chosen_enter_advantage_positive_rate"], MIN_ACTION_PRECISION
        ),
        "chosen_hold_advantage_precision": passes(
            quality["chosen_hold_advantage_positive_rate"], MIN_ACTION_PRECISION
        ),
        "severe_loss_rate": (
            direct_metrics["severe_loss_rate_le_minus2"] is not None
            and direct_metrics["severe_loss_rate_le_minus2"] <= MAX_SEVERE_LOSS
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "development_only": True,
        "threshold": THRESHOLD,
        "class_balancing": False,
        "training_support": policy.support,
        "direct_policy": direct_metrics,
        "request243_refit_comparator": old_metrics,
        "decision_quality": quality,
        "checks": checks,
        "development_gate_pass": bool(all(checks.values())),
        "limitations": [
            "resolved-trade means condition on different subsets and are not paired portfolio comparisons",
            "WAIT support is inherited from the earlier chronological teacher",
            "no reentry, shared capital, full-HOT admission or live execution",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    direct_scored.to_parquet(rows_output, index=False)
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
