"""R308: fixed clock/first-snapshot weighting on the unchanged R306 input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .monotone_momentum_calibration import diagnostic, predict_probability
from .preentry_momentum_observability import CLOCK_FEATURES, SNAPSHOTS, fit_probability, snapshots
from .risk_reachable_hold_value import validate_fit_days

REQUEST_ID = 308
ARMS = ("U", "T", "E")
PHASES = ("0_20s", "20_60s", "60_120s", "120s_plus")


def timing(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty or frame.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("unique nonempty cash states required")
    elapsed = (frame.decision_t-frame.groupby(KEYS).decision_t.transform("min")).to_numpy(float)/1000
    if not np.isfinite(elapsed).all():
        raise ValueError("finite decision clocks required")
    return np.searchsorted([20., 60., 120.], elapsed, side="right"), elapsed == 0


def sample_weights(frame: pd.DataFrame, arm: str) -> np.ndarray:
    """Equal day and episode mass; labels and feature values never select weights."""
    if arm not in ARMS:
        raise ValueError("unknown weighting arm")
    phase, first = timing(frame)
    if arm == "U":
        return _episode_day_weights(frame)
    work = frame[KEYS].reset_index(drop=True).assign(phase=phase)
    counts = work.groupby([*KEYS, "phase"]).phase.transform("size").to_numpy(float)
    phases = work.groupby(KEYS).phase.transform("nunique").to_numpy(float)
    within = 1/(counts*phases)
    if arm == "E":
        within = .5*within+.5*first.astype(float)
    episodes = work[KEYS].drop_duplicates().trading_day.value_counts()
    weights = within/work.trading_day.map(episodes).to_numpy(float)
    return weights/weights.mean()


def exposure(frame: pd.DataFrame, weights: np.ndarray) -> dict:
    phase, first = timing(frame)
    return {"first_snapshot_mass": float(weights[first].sum()/weights.sum()),
            "phase_mass": {label: float(weights[phase == i].sum()/weights.sum()) for i, label in enumerate(PHASES)},
            "weighted_net5_prevalence": float(np.average(truth(frame[CEILING], 5), weights=weights))}


def fit_weighted(train: pd.DataFrame, features: tuple[str, ...], arm: str, seed: int):
    y, weights = truth(train[CEILING], 5), sample_weights(train, arm)
    prior = float(np.average(y, weights=weights))
    if len(np.unique(y)) != 2:
        return None, prior
    model = HistGradientBoostingClassifier(learning_rate=.04, max_iter=220, max_leaf_nodes=15,
        min_samples_leaf=75, l2_regularization=3., early_stopping=False, random_state=seed)
    model.fit(train[list(features)], y, sample_weight=weights)
    return model, prior


def weighted_crossfit(source: pd.DataFrame, features: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    if set(source.trading_day.astype(str)) != set(CROSSFIT_DAYS):
        raise ValueError("R308 requires May5-8 development states")
    timing(source)
    parts, folds = [], {}
    for fold, day in enumerate(CROSSFIT_DAYS):
        train = source.loc[source.trading_day.ne(day)]
        held = source.loc[source.trading_day.eq(day)].copy()
        days = sorted(train.trading_day.unique())
        validate_fit_days(days, excluded=[day])
        seed = 20264100+fold*100+5
        baseline, prior = fit_probability(train, features, 5, seed)
        held["U_p_net_5"] = predict_probability(held, baseline, prior, features)
        error = float(np.max(np.abs(held.U_p_net_5.to_numpy()-held.C_p_net_5.to_numpy())))
        if error > 1e-9:
            raise ValueError("saved R306 C score mismatch")
        held["U_prior_net_5"] = prior
        first_train = snapshots(train, 0)
        held["first_training_prior"] = float(np.average(truth(first_train[CEILING], 5), weights=_episode_day_weights(first_train)))
        audits = {"U": exposure(train, sample_weights(train, "U"))}
        for arm in ("T", "E"):
            model, arm_prior = fit_weighted(train, features, arm, seed)
            held[f"{arm}_p_net_5"] = predict_probability(held, model, arm_prior, features)
            held[f"{arm}_prior_net_5"] = arm_prior
            audits[arm] = {**exposure(train, sample_weights(train, arm)), "single_class_fallback": model is None}
        folds[day] = {"fit_days": days, "seed": seed, "train_states": len(train), "train_episodes": len(first_train),
                      "train_first_positive_episodes": int(truth(first_train[CEILING], 5).sum()),
                      "maximum_saved_C_error": error, "weight_exposure": audits}
        parts.append(held)
        print(f"R308 weighting fold complete {day}", flush=True)
    return pd.concat(parts, ignore_index=True), folds


def paired_intervals(first: pd.DataFrame) -> dict:
    output = {}
    y, w = truth(first[CEILING], 5), _episode_day_weights(first)
    p = {arm: first[f"{arm}_p_net_5"].to_numpy(float) for arm in ARMS}
    for i, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        clusters = list(first.groupby(columns, sort=False).indices.values())
        rng = np.random.default_rng(20264300+i)
        values = {arm: {"auc": [], "brier": []} for arm in ("T", "E")}
        for _ in range(1000):
            rows = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
            both = len(np.unique(y[rows])) == 2
            baseline_auc = roc_auc_score(y[rows], p["U"][rows], sample_weight=w[rows]) if both else None
            for arm in ("T", "E"):
                if both:
                    values[arm]["auc"].append(float(roc_auc_score(y[rows], p[arm][rows], sample_weight=w[rows])-baseline_auc))
                gain = (y[rows]-p[arm][rows])**2-(y[rows]-p["U"][rows])**2
                values[arm]["brier"].append(float(np.average(gain, weights=w[rows])))
        output["day" if i == 0 else "ticker_day"] = {"clusters": len(clusters), "arms": {
            arm: {"auc_minus_U_ci95": np.quantile(result["auc"], [.025, .975]).tolist(),
                  "brier_minus_U_ci95": np.quantile(result["brier"], [.025, .975]).tolist(),
                  "auc_valid_draws": len(result["auc"])} for arm, result in values.items()}}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("states", "summary", "output", "states-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    source, previous = pd.read_parquet(args.states), json.loads(args.summary.read_text())
    if previous.get("request_id") != 306 or len(source[KEYS].drop_duplicates()) != 97 or not source.label_complete.all():
        raise ValueError("exact full R306 candidate support required")
    features = tuple(previous["baseline_features"])+CLOCK_FEATURES
    scored, folds = weighted_crossfit(source, features)
    reports = {"all_states": {arm: diagnostic(scored, arm) for arm in ARMS}}
    for delay in SNAPSHOTS:
        chosen = snapshots(scored, delay)
        w, y = _episode_day_weights(chosen), truth(chosen[CEILING], 5)
        reports[f"snapshot_{delay}s"] = {"episodes": len(chosen), "missing_of97": 97-len(chosen),
            "first_training_prior_brier": float(np.average((y-chosen.first_training_prior.to_numpy())**2, weights=w)),
            "arms": {arm: diagnostic(chosen, arm) for arm in ARMS},
            "per_day": {str(day): {arm: diagnostic(group, arm) for arm in ARMS} for day, group in chosen.groupby("trading_day")}}
        if delay == 0:
            reports["snapshot_0s"]["paired_intervals"] = paired_intervals(chosen)
    initial = reports["snapshot_0s"]
    raw, early = initial["arms"]["U"], initial["arms"]["E"]
    days = [r for r in initial["per_day"].values() if r["U"]["auc"] is not None]
    checks = {"exact_R306_C_scores": all(f["maximum_saved_C_error"] <= 1e-9 for f in folds.values()),
        "first_E_auc_beats_U": early["auc"] > raw["auc"], "first_E_ap_beats_U": early["ap"] > raw["ap"],
        "first_E_brier_beats_U": early["brier"] < raw["brier"],
        "first_E_brier_beats_first_training_prior": early["brier"] < initial["first_training_prior_brier"],
        "E_auc_improves_at_least_two_evaluable_days": sum(r["E"]["auc"] > r["U"]["auc"] for r in days) >= 2,
        "ticker_day_E_auc_gain_ci_positive": initial["paired_intervals"]["ticker_day"]["arms"]["E"]["auc_minus_U_ci95"][0] > 0}
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
        "reference_R306_run": 37211254516, "network_requests": 0, "June_HOLD_opened": False, "entries_retuned": False,
        "features": list(features), "arms": {"U": "Original within-episode uniform states", "T": "Equal mass per available fixed clock phase",
            "E": "Fixed 0.5 first-snapshot point mass + 0.5 T distribution; primary arm"},
        "clock_phase_boundaries_seconds": [20, 60, 120], "folds": folds, "observability": reports,
        "checks": checks, "research_gate_pass": all(checks.values()),
        "limitations": "Same 97 upstream-OOF-conditioned episodes, nine first winners and four reused May days. Reweighting creates no independent examples. First clock starts at first saved decision, not HOT; unavailable phases are not imputed. All historical features, labels, trees, seeds and policy remain fixed; row-based tree bins/minimum leaf sizes remain correlated-state constraints. No post-result fraction, phase, parameter, feature, threshold or policy selection. No executable profit claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.states_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.states_output, index=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({"checks": checks, "first_arms": initial["arms"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
