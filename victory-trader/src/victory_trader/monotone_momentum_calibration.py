"""R307: nested first-cash probability calibration, preserving R306 signal."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .preentry_momentum_observability import CLOCK_FEATURES, SNAPSHOTS, fit_probability, metrics, snapshots
from .risk_reachable_hold_value import validate_fit_days

REQUEST_ID = 307


@dataclass(frozen=True)
class Map:
    intercept: float
    slope: float
    prior: float
    fallback: bool


def probability_input(probabilities) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("invalid probability input")
    return logit(np.clip(p, 1e-6, 1-1e-6))


def apply_map(probabilities, fitted: Map) -> np.ndarray:
    if fitted.slope < 0:
        raise ValueError("probability inversion forbidden")
    return expit(fitted.intercept+fitted.slope*probability_input(probabilities))


def fit_map(first: pd.DataFrame) -> tuple[Map, dict]:
    if first.empty or first.duplicated(KEYS).any():
        raise ValueError("calibration needs one original snapshot per episode")
    x = probability_input(first.inner_probability)
    y, w = truth(first[CEILING], 5).astype(float), _episode_day_weights(first)
    prior = float(np.average(y, weights=w))
    smoothed = float((np.sum(w*y)+1)/(np.sum(w)+2))
    fallback = Map(float(logit(smoothed)), 0., prior, True)
    audit = {"episodes": len(first), "positive_episodes": int(y.sum()), "fit_days": sorted(first.trading_day.unique()),
             "weighted_first_snapshot_prevalence": prior, "smoothed_fallback_prevalence": smoothed}
    if len(np.unique(y)) != 2:
        return fallback, {**audit, "optimizer_success": False, "reason": "single_class"}
    def objective(parameters):
        intercept, slope = parameters
        z = intercept+slope*x
        loss = np.sum(w*(np.logaddexp(0, z)-y*z))+.5*slope*slope
        residual = w*(expit(z)-y)
        gradient = np.array([residual.sum(), np.dot(residual, x)+slope])
        return float(loss), gradient
    optimized = minimize(objective, [float(logit(np.clip(prior, 1e-6, 1-1e-6))), 1.],
                         jac=True, method="L-BFGS-B", bounds=[(None, None), (0., None)],
                         options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
    if not optimized.success or not np.isfinite(optimized.x).all():
        return fallback, {**audit, "optimizer_success": False, "reason": str(optimized.message)}
    fitted = Map(float(optimized.x[0]), float(optimized.x[1]), prior, False)
    return fitted, {**audit, "optimizer_success": True, "intercept": fitted.intercept,
                   "slope": fitted.slope, "constant_map": fitted.slope <= 1e-8}


def predict_probability(frame: pd.DataFrame, model, prior: float, features: tuple[str, ...]) -> np.ndarray:
    return model.predict_proba(frame[list(features)])[:, 1] if model is not None else np.full(len(frame), prior)


def calibrated_crossfit(source: pd.DataFrame, features: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    if set(source.trading_day.astype(str)) != set(CROSSFIT_DAYS) or source.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("R307 requires unique May5-8 development states")
    pieces, folds = [], {}
    for outer, day in enumerate(CROSSFIT_DAYS):
        train, held = source.loc[source.trading_day.ne(day)], source.loc[source.trading_day.eq(day)].copy()
        train_days = sorted(train.trading_day.unique())
        validate_fit_days(train_days, excluded=[day])
        model, prior = fit_probability(train, features, 5, 20264100+outer*100+5)
        replay = predict_probability(held, model, prior, features)
        error = float(np.max(np.abs(replay-held.C_p_net_5.to_numpy(float))))
        if error > 1e-9:
            raise ValueError("saved R306 C score mismatch")
        inner_parts, inner_folds = [], {}
        for inner, scored_day in enumerate(train_days):
            fitting = train.loc[train.trading_day.ne(scored_day)]
            scoring = train.loc[train.trading_day.eq(scored_day)].copy()
            fit_days = sorted(fitting.trading_day.unique())
            validate_fit_days(fit_days, excluded=[day, scored_day])
            seed = 20264200+outer*100+inner*10+5
            inner_model, inner_prior = fit_probability(fitting, features, 5, seed)
            scoring["inner_probability"] = predict_probability(scoring, inner_model, inner_prior, features)
            inner_parts.append(snapshots(scoring, 0))
            inner_folds[scored_day] = {"fit_days": fit_days, "seed": seed, "single_class_fallback": inner_model is None}
        calibration = pd.concat(inner_parts, ignore_index=True)
        mapping, audit = fit_map(calibration)
        held["E_p_net_5"] = apply_map(held.C_p_net_5.to_numpy(float), mapping)
        held["E_prior_net_5"] = mapping.prior
        pieces.append(held)
        folds[day] = {"outer_fit_days": train_days, "maximum_saved_C_error": error,
                      "inner_folds": inner_folds, "calibration": audit,
                      "intercept": mapping.intercept, "slope": mapping.slope, "fallback": mapping.fallback}
        print(f"R307 nested calibration fold complete {day}, slope={mapping.slope:.6f}", flush=True)
    return pd.concat(pieces, ignore_index=True), folds


def diagnostic(frame: pd.DataFrame, arm: str) -> dict:
    report = metrics(frame, arm, 5)
    if frame.empty:
        return report
    p, y, w = frame[f"{arm}_p_net_5"].to_numpy(float), truth(frame[CEILING], 5), _episode_day_weights(frame)
    report["predicted_mean"] = float(np.average(p, weights=w))
    report["observed_mean"] = float(np.average(y, weights=w))
    report["probability_bins"] = []
    for low, high in ((0, .05), (.05, .1), (.1, .2), (.2, .5), (.5, 1.000001)):
        keep = (p >= low) & (p < high)
        report["probability_bins"].append({"lower": low, "upper": min(high, 1.), "states": int(keep.sum()),
            "predicted": float(np.average(p[keep], weights=w[keep])) if keep.any() else None,
            "observed": float(np.average(y[keep], weights=w[keep])) if keep.any() else None})
    return report


def brier_intervals(first: pd.DataFrame) -> dict:
    output = {}
    weights = _episode_day_weights(first)
    y = truth(first[CEILING], 5).astype(float)
    gain = (y-first.E_p_net_5.to_numpy(float))**2-(y-first.C_p_net_5.to_numpy(float))**2
    for i, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        clusters = list(first.groupby(columns, sort=False).indices.values())
        rng = np.random.default_rng(20264250+i)
        values = []
        for _ in range(1000):
            rows = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
            values.append(float(np.average(gain[rows], weights=weights[rows])))
        output["day" if i == 0 else "ticker_day"] = {"clusters": len(clusters), "E_minus_C_brier_ci95": np.quantile(values, [.025, .975]).tolist()}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("states", "summary", "output", "states-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_parquet(args.states)
    previous = json.loads(args.summary.read_text())
    if previous.get("request_id") != 306 or len(source[KEYS].drop_duplicates()) != 97 or not source.label_complete.all():
        raise ValueError("exact full R306 candidate support required")
    features = tuple(previous["baseline_features"])+CLOCK_FEATURES
    scored, folds = calibrated_crossfit(source, features)
    reports = {"all_states": {arm: diagnostic(scored, arm) for arm in ("C", "D", "E")}}
    for delay in SNAPSHOTS:
        chosen = snapshots(scored, delay)
        reports[f"snapshot_{delay}s"] = {"episodes": len(chosen), "missing_of97": 97-len(chosen),
            "arms": {arm: diagnostic(chosen, arm) for arm in ("C", "D", "E")},
            "per_day": {str(day): {arm: diagnostic(group, arm) for arm in ("C", "D", "E")} for day, group in chosen.groupby("trading_day")}}
        if delay == 0:
            reports[f"snapshot_{delay}s"]["paired_brier_intervals"] = brier_intervals(chosen)
    initial = reports["snapshot_0s"]
    raw, control, calibrated = (initial["arms"][arm] for arm in ("C", "D", "E"))
    rank_equal = []
    for day, group in snapshots(scored, 0).groupby("trading_day"):
        y = truth(group[CEILING], 5)
        if len(np.unique(y)) == 2:
            rank_equal.append(abs(roc_auc_score(y, group.C_p_net_5)-roc_auc_score(y, group.E_p_net_5)) <= 1e-9)
    checks = {"exact_R306_C_scores": all(f["maximum_saved_C_error"] <= 1e-9 for f in folds.values()),
              "first_brier_beats_C": calibrated["brier"] < raw["brier"],
              "first_brier_beats_cost_control": calibrated["brier"] < control["brier"],
              "first_brier_beats_first_snapshot_training_prior": calibrated["brier"] < calibrated["prior_brier"],
              "first_auc_not_below_raw_C": calibrated["auc"] >= raw["auc"],
              "all_calibration_slopes_positive": all(f["slope"] > 1e-8 for f in folds.values()),
              "within_day_first_rank_equal": all(rank_equal)}
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
        "network_requests": 0, "June_HOLD_opened": False, "entries_retuned": False,
        "reference_R306_run": 37211254516, "features": list(features), "folds": folds,
        "observability": reports, "checks": checks, "research_gate_pass": all(checks.values()),
        "limitations": "Nested calibration prevents own/outer-day target reuse, but base two-day calibration versus three-day deployment can shift score distributions. First-snapshot map transport to later states is secondary. Constant maps are failures of discrimination. Nine first winners,97 conditional upstream-OOF episodes and four reused days do not validate profitability or justify promotion. No tuning, inversion, policy or threshold selection."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.states_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.states_output, index=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({"checks": checks, "first_raw": raw, "first_calibrated": calibrated}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
