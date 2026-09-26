"""Request245: audit same-policy action-value calibration without changing actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .chronological_action_value import (
    CAL_DAYS,
    KEYS,
    TARGETS,
    build_states,
    chronological_fit,
    rollout,
)

REQUEST_ID = 245
MIN_SPEARMAN = 0.20
MIN_CHOSEN_ADVANTAGE_POSITIVE_RATE = 0.55


def finite_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & np.isfinite(values)


def spearman(x: pd.Series, y: pd.Series) -> float | None:
    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair.x.nunique() < 2 or pair.y.nunique() < 2:
        return None
    return float(pair.x.rank(method="average").corr(pair.y.rank(method="average")))


def support_summary(frame: pd.DataFrame, target: str) -> dict:
    values = pd.to_numeric(frame[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = values.dropna()
    quantiles = {}
    if len(valid):
        for q in (0, 0.01, 0.25, 0.5, 0.75, 0.99, 1):
            quantiles[str(q)] = float(valid.quantile(q))
    return {
        "rows": int(len(valid)),
        "episodes": int(frame.loc[values.notna(), KEYS].drop_duplicates().shape[0]),
        "missing_rows": int(values.isna().sum()),
        "positive_rate": float(valid.gt(0).mean()) if len(valid) else None,
        "zero_rate": float(valid.eq(0).mean()) if len(valid) else None,
        "mean": float(valid.mean()) if len(valid) else None,
        "std": float(valid.std(ddof=0)) if len(valid) else None,
        "quantiles": quantiles,
    }


def reachable_masks(scored: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    flat = pd.Series(False, index=scored.index)
    position = pd.Series(False, index=scored.index)
    for _, group in scored.groupby(KEYS, sort=False):
        ids = list(group.index)
        for i in ids:
            flat.at[i] = True
            if str(scored.at[i, "flat_action"]) != "WAIT":
                break
        if not ids:
            continue
        entry = int(scored.at[ids[0], "policy_entry_index"])
        exit_index = int(scored.at[ids[0], "policy_exit_index"])
        if entry < 0:
            continue
        for i in ids:
            if i <= entry:
                continue
            if exit_index >= 0 and i > exit_index:
                break
            position.at[i] = True
    return flat, position


def value_metrics(
    scored: pd.DataFrame,
    *,
    target: str,
    predicted: str,
    mask: pd.Series,
) -> dict:
    valid = mask & finite_series(scored[target]) & finite_series(scored[predicted])
    sub = scored.loc[valid, KEYS + [target, predicted]].copy()
    if not len(sub):
        return {
            "rows": 0,
            "episodes": 0,
            "spearman": None,
            "mean_prediction": None,
            "mean_realized": None,
            "mean_bias": None,
            "mae": None,
            "predicted_positive_rate": None,
            "realized_positive_rate": None,
            "positive_sign_precision": None,
        }
    pred = pd.to_numeric(sub[predicted], errors="coerce")
    real = pd.to_numeric(sub[target], errors="coerce")
    positive = pred.gt(0)
    return {
        "rows": int(len(sub)),
        "episodes": int(sub[KEYS].drop_duplicates().shape[0]),
        "spearman": spearman(pred, real),
        "mean_prediction": float(pred.mean()),
        "mean_realized": float(real.mean()),
        "mean_bias": float((pred - real).mean()),
        "mae": float((pred - real).abs().mean()),
        "predicted_positive_rate": float(positive.mean()),
        "realized_positive_rate": float(real.gt(0).mean()),
        "positive_sign_precision": (
            float(real.loc[positive].gt(0).mean()) if positive.any() else None
        ),
    }


def flat_advantage_metrics(scored: pd.DataFrame, mask: pd.Series) -> dict:
    needed = [
        "enter_value",
        "wait_value",
        "predicted_enter_value",
        "predicted_wait_value",
    ]
    valid = mask & scored.can_enter.astype(bool) & ~scored.terminal.astype(bool)
    for column in needed:
        valid &= finite_series(scored[column])
    sub = scored.loc[valid].copy()
    if not len(sub):
        return {
            "rows": 0,
            "episodes": 0,
            "spearman": None,
            "chosen_enter_rows": 0,
            "chosen_enter_advantage_positive_rate": None,
            "chosen_wait_rows": 0,
            "chosen_wait_advantage_positive_rate": None,
            "realized_best_action_accuracy": None,
        }

    sub["predicted_advantage"] = (
        pd.to_numeric(sub.predicted_enter_value)
        - pd.to_numeric(sub.predicted_wait_value)
    )
    sub["realized_advantage"] = (
        pd.to_numeric(sub.enter_value) - pd.to_numeric(sub.wait_value)
    )
    enter_alt = np.maximum(pd.to_numeric(sub.wait_value), 0.0)
    wait_alt = np.maximum(pd.to_numeric(sub.enter_value), 0.0)
    sub["enter_advantage_over_best_alt"] = pd.to_numeric(sub.enter_value) - enter_alt
    sub["wait_advantage_over_best_alt"] = pd.to_numeric(sub.wait_value) - wait_alt

    enter_rows = sub.flat_action.eq("ENTER")
    wait_rows = sub.flat_action.eq("WAIT")

    real_enter = pd.to_numeric(sub.enter_value).to_numpy(float)
    real_wait = pd.to_numeric(sub.wait_value).to_numpy(float)
    real_best = np.where(
        (real_enter > real_wait) & (real_enter > 0),
        "ENTER",
        np.where(real_wait > 0, "WAIT", "ABSTAIN"),
    )
    accuracy = float(np.mean(sub.flat_action.astype(str).to_numpy() == real_best))

    return {
        "rows": int(len(sub)),
        "episodes": int(sub[KEYS].drop_duplicates().shape[0]),
        "spearman": spearman(sub.predicted_advantage, sub.realized_advantage),
        "mean_predicted_advantage": float(sub.predicted_advantage.mean()),
        "mean_realized_advantage": float(sub.realized_advantage.mean()),
        "mean_advantage_bias": float(
            (sub.predicted_advantage - sub.realized_advantage).mean()
        ),
        "chosen_enter_rows": int(enter_rows.sum()),
        "chosen_enter_advantage_positive_rate": (
            float(sub.loc[enter_rows, "enter_advantage_over_best_alt"].gt(0).mean())
            if enter_rows.any()
            else None
        ),
        "chosen_enter_mean_advantage_pct": (
            float(sub.loc[enter_rows, "enter_advantage_over_best_alt"].mean())
            if enter_rows.any()
            else None
        ),
        "chosen_wait_rows": int(wait_rows.sum()),
        "chosen_wait_advantage_positive_rate": (
            float(sub.loc[wait_rows, "wait_advantage_over_best_alt"].gt(0).mean())
            if wait_rows.any()
            else None
        ),
        "chosen_wait_mean_advantage_pct": (
            float(sub.loc[wait_rows, "wait_advantage_over_best_alt"].mean())
            if wait_rows.any()
            else None
        ),
        "realized_best_action_accuracy": accuracy,
    }


def hold_decision_metrics(scored: pd.DataFrame, mask: pd.Series) -> dict:
    valid = (
        mask
        & finite_series(scored.hold_advantage)
        & finite_series(scored.predicted_hold_advantage)
    )
    sub = scored.loc[valid].copy()
    hold = sub.held_action.eq("HOLD")
    exit_now = sub.held_action.eq("EXIT")
    return {
        "rows": int(len(sub)),
        "episodes": int(sub[KEYS].drop_duplicates().shape[0]) if len(sub) else 0,
        "spearman": (
            spearman(sub.predicted_hold_advantage, sub.hold_advantage)
            if len(sub)
            else None
        ),
        "chosen_hold_rows": int(hold.sum()),
        "chosen_hold_advantage_positive_rate": (
            float(pd.to_numeric(sub.loc[hold, "hold_advantage"]).gt(0).mean())
            if hold.any()
            else None
        ),
        "chosen_hold_mean_advantage_pct": (
            float(pd.to_numeric(sub.loc[hold, "hold_advantage"]).mean())
            if hold.any()
            else None
        ),
        "chosen_exit_rows": int(exit_now.sum()),
        "chosen_exit_correct_rate": (
            float(pd.to_numeric(sub.loc[exit_now, "hold_advantage"]).le(0).mean())
            if exit_now.any()
            else None
        ),
    }


def enough(value: float | None, threshold: float) -> bool:
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

    teacher, student, student_targets = chronological_fit(fit_states)
    scored = rollout(cal_states, student, "full")
    flat_reachable, position_reachable = reachable_masks(scored)
    scored["flat_reachable"] = flat_reachable
    scored["position_reachable"] = position_reachable

    enter = value_metrics(
        scored,
        target="enter_value",
        predicted="predicted_enter_value",
        mask=flat_reachable,
    )
    wait = value_metrics(
        scored,
        target="wait_value",
        predicted="predicted_wait_value",
        mask=flat_reachable,
    )
    hold = value_metrics(
        scored,
        target="hold_advantage",
        predicted="predicted_hold_advantage",
        mask=position_reachable,
    )
    flat_adv = flat_advantage_metrics(scored, flat_reachable)
    hold_decisions = hold_decision_metrics(scored, position_reachable)

    support = {target: support_summary(student_targets, target) for target in TARGETS}
    enter_std = support["enter_value"]["std"]
    wait_std = support["wait_value"]["std"]
    wait_to_enter_std_ratio = (
        float(wait_std / enter_std)
        if wait_std is not None and enter_std is not None and enter_std > 0
        else None
    )

    checks = {
        "enter_spearman": enough(enter["spearman"], MIN_SPEARMAN),
        "flat_advantage_spearman": enough(flat_adv["spearman"], MIN_SPEARMAN),
        "chosen_enter_advantage_positive_rate": enough(
            flat_adv["chosen_enter_advantage_positive_rate"],
            MIN_CHOSEN_ADVANTAGE_POSITIVE_RATE,
        ),
        "hold_spearman": enough(hold["spearman"], MIN_SPEARMAN),
        "chosen_hold_advantage_positive_rate": enough(
            hold_decisions["chosen_hold_advantage_positive_rate"],
            MIN_CHOSEN_ADVANTAGE_POSITIVE_RATE,
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "development_only": True,
        "calibration_days": CAL_DAYS,
        "teacher_training_days": list(teacher.training_days),
        "student_training_days": list(student.training_days),
        "student_target_support": support,
        "wait_to_enter_target_std_ratio": wait_to_enter_std_ratio,
        "reachable_counts": {
            "flat_rows": int(flat_reachable.sum()),
            "position_rows": int(position_reachable.sum()),
            "episodes": int(scored[KEYS].drop_duplicates().shape[0]),
        },
        "enter_value": enter,
        "wait_value": wait,
        "hold_value": hold,
        "flat_enter_vs_wait": flat_adv,
        "hold_decisions": hold_decisions,
        "checks": checks,
        "adequacy_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "Diagnostic only. Realized values are same-policy causal continuations "
            "from Request243 bookkeeping, never a best-future-price oracle."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    scored.to_parquet(rows_output, index=False)
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
