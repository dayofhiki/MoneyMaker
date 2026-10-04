"""R309: preregistered feature/linear-pipeline factorial on frozen May data."""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .monotone_momentum_calibration import diagnostic, predict_probability
from .preentry_momentum_observability import CLOCK_FEATURES, SNAPSHOTS, baseline_columns, fit_probability, metrics, snapshots, validate_states
from .risk_reachable_hold_value import validate_fit_days

REQUEST_ID = 309
ARMS = ("U", "V", "W", "X", "F")
COMPACT_FEATURES = (
    "momentum_return_30s", "momentum_efficiency_30s", "momentum_volume_rate_ratio_30s",
    "momentum_transactions_rate_ratio_30s", "momentum_signed_volume_proxy_30s",
    "momentum_reclaim_prior_high_30s", "momentum_activity_fraction_30s",
    "momentum_gap_s", "known_cost_drag_pct", "momentum_acceleration_10_30",
)


def independent_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = _episode_day_weights(frame)
    return weights*len(frame[KEYS].drop_duplicates())/weights.sum()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    index = np.searchsorted(np.cumsum(weights[order]), q*weights.sum(), side="left")
    return float(values[order[min(index, len(order)-1)]])


@dataclass(frozen=True)
class Transform:
    features: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    all_missing: tuple[str, ...]

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[list(self.features)].to_numpy(float)
        if np.isinf(values).any():
            raise ValueError("infinite causal feature")
        missing = np.isnan(values)
        filled = np.where(missing, self.median, np.clip(values, self.lower, self.upper))
        return np.c_[(filled-self.mean)/self.scale, missing.astype(float)]

    def audit(self) -> dict:
        return {"raw_features": list(self.features), "processed_dimensions": 2*len(self.features),
                "all_training_missing": list(self.all_missing),
                **{name: getattr(self, name).tolist() for name in ("lower", "upper", "median", "mean", "scale")}}


def fit_transform(train: pd.DataFrame, features: tuple[str, ...], weights: np.ndarray) -> Transform:
    values = train[list(features)].to_numpy(float)
    if np.isinf(values).any() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("finite features and positive finite weights required")
    lower, upper, median, mean, scale, missing = [], [], [], [], [], []
    for j, feature in enumerate(features):
        v = values[:, j]
        available = np.isfinite(v)
        if available.any():
            lo, med, hi = [weighted_quantile(v[available], weights[available], q) for q in (.01, .5, .99)]
        else:
            lo = med = hi = 0.
            missing.append(feature)
        filled = np.where(available, np.clip(v, lo, hi), med)
        center = float(np.average(filled, weights=weights))
        std = float(np.sqrt(np.average((filled-center)**2, weights=weights)))
        lower.append(lo)
        upper.append(hi)
        median.append(med)
        mean.append(center)
        scale.append(std if std > 1e-12 else 1.)
    return Transform(features, *(np.asarray(v) for v in (lower, upper, median, mean, scale)), tuple(missing))


def fit_linear(train: pd.DataFrame, features: tuple[str, ...], seed: int):
    weights, y = independent_weights(train), truth(train[CEILING], 5)
    transform = fit_transform(train, features, weights)
    prior = float(np.average(y, weights=weights))
    audit = {"train_rows": len(train), "train_episodes": len(train[KEYS].drop_duplicates()),
             "train_positive_episodes": len(train.loc[y, KEYS].drop_duplicates()),
             "weight_sum": float(weights.sum()), "prior": prior, "preprocessing": transform.audit(),
             "single_class_fallback": len(np.unique(y)) != 2}
    if len(np.unique(y)) != 2:
        return None, transform, prior, audit
    model = LogisticRegression(C=.1, l1_ratio=0., solver="lbfgs", fit_intercept=True,
                               max_iter=2000, tol=1e-8, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(transform.apply(train), y, sample_weight=weights)
    audit.update({"iterations": int(model.n_iter_.max()), "intercept": float(model.intercept_[0]),
                  "coefficients": model.coef_[0].tolist()})
    return model, transform, prior, audit


def predict_linear(frame: pd.DataFrame, model, transform: Transform, prior: float) -> np.ndarray:
    x = transform.apply(frame)
    return model.predict_proba(x)[:, 1] if model is not None else np.full(len(frame), prior)


def representation_crossfit(source: pd.DataFrame, full_features: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    if set(source.trading_day.astype(str)) != set(CROSSFIT_DAYS) or source.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("R309 requires unique May5-8 development states")
    if not set(COMPACT_FEATURES).issubset(full_features):
        raise ValueError("compact input must be a causal subset of original features")
    pieces, folds = [], {}
    for fold, day in enumerate(CROSSFIT_DAYS):
        train, held = source.loc[source.trading_day.ne(day)], source.loc[source.trading_day.eq(day)].copy()
        fit_days = sorted(train.trading_day.unique())
        validate_fit_days(fit_days, excluded=[day])
        seed = 20264100+fold*100+5
        audit = {"fit_days": fit_days, "seed": seed, "models": {}}
        for arm, features in (("U", full_features), ("V", COMPACT_FEATURES)):
            model, prior = fit_probability(train, features, 5, seed)
            held[f"{arm}_p_net_5"] = predict_probability(held, model, prior, features)
            held[f"{arm}_prior_net_5"] = prior
            audit["models"][arm] = {"raw_dimensions": len(features), "single_class_fallback": model is None, "prior": prior}
        error = float(np.max(np.abs(held.U_p_net_5.to_numpy()-held.C_p_net_5.to_numpy())))
        if error > 1e-9:
            raise ValueError("saved R306 C score mismatch")
        first_train = snapshots(train, 0)
        held["first_training_prior"] = float(np.average(truth(first_train[CEILING], 5), weights=_episode_day_weights(first_train)))
        for arm, features, fitting in (("W", full_features, train), ("X", COMPACT_FEATURES, train), ("F", COMPACT_FEATURES, first_train)):
            model, transform, prior, linear_audit = fit_linear(fitting, features, seed)
            held[f"{arm}_p_net_5"] = predict_linear(held, model, transform, prior)
            held[f"{arm}_prior_net_5"] = prior
            audit["models"][arm] = linear_audit
        audit["maximum_saved_C_error"] = error
        pieces.append(held)
        folds[day] = audit
        print(f"R309 compact representation fold complete {day}", flush=True)
    return pd.concat(pieces, ignore_index=True), folds


def paired_intervals(first: pd.DataFrame) -> dict:
    comparisons = [(arm, "U") for arm in ("V", "W", "X", "F")]+[("X", "V"), ("X", "W")]
    y, w = truth(first[CEILING], 5), _episode_day_weights(first)
    p = {arm: first[f"{arm}_p_net_5"].to_numpy(float) for arm in ARMS}
    output = {}
    for i, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        clusters = list(first.groupby(columns, sort=False).indices.values())
        rng = np.random.default_rng(20264400+i)
        values = {f"{a}_minus_{b}": {"auc": [], "brier": []} for a, b in comparisons}
        for _ in range(1000):
            rows = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
            both = len(np.unique(y[rows])) == 2
            aucs = {arm: roc_auc_score(y[rows], p[arm][rows], sample_weight=w[rows]) for arm in ARMS} if both else {}
            for a, b in comparisons:
                key = f"{a}_minus_{b}"
                if both:
                    values[key]["auc"].append(float(aucs[a]-aucs[b]))
                gain = (y[rows]-p[a][rows])**2-(y[rows]-p[b][rows])**2
                values[key]["brier"].append(float(np.average(gain, weights=w[rows])))
        output["day" if i == 0 else "ticker_day"] = {"clusters": len(clusters), "comparisons": {
            key: {"auc_ci95": np.quantile(v["auc"], [.025, .975]).tolist(),
                  "brier_ci95": np.quantile(v["brier"], [.025, .975]).tolist(),
                  "auc_valid_draws": len(v["auc"])} for key, v in values.items()}}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("states", "summary", "output", "states-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    source, previous = pd.read_parquet(args.states), json.loads(args.summary.read_text())
    validate_states(source)
    if previous.get("request_id") != 306 or not source.label_complete.all() or not np.isfinite(source[CEILING]).all():
        raise ValueError("full finite R306 candidate labels required")
    baseline = baseline_columns(source)
    if baseline != tuple(previous["baseline_features"]):
        raise ValueError("saved causal feature allowlist mismatch")
    full_features = baseline+CLOCK_FEATURES
    scored, folds = representation_crossfit(source, full_features)
    reports = {"all_states": {arm: metrics(scored, arm, 5) for arm in ARMS}}
    for delay in SNAPSHOTS:
        chosen = snapshots(scored, delay)
        w, y = _episode_day_weights(chosen), truth(chosen[CEILING], 5)
        reports[f"snapshot_{delay}s"] = {"episodes": len(chosen), "missing_of97": 97-len(chosen),
            "first_training_prior_brier": float(np.average((y-chosen.first_training_prior.to_numpy())**2, weights=w)),
            "arms": {arm: diagnostic(chosen, arm) if delay == 0 else metrics(chosen, arm, 5) for arm in ARMS},
            "per_day": {str(day): {arm: metrics(group, arm, 5) for arm in ARMS} for day, group in chosen.groupby("trading_day")}}
        if delay == 0:
            reports["snapshot_0s"]["paired_intervals"] = paired_intervals(chosen)
    initial = reports["snapshot_0s"]
    raw, compact = initial["arms"]["U"], initial["arms"]["X"]
    days = [r for r in initial["per_day"].values() if r["U"]["auc"] is not None]
    checks = {"exact_R306_C_scores": all(f["maximum_saved_C_error"] <= 1e-9 for f in folds.values()),
        "first_X_auc_beats_U": compact["auc"] > raw["auc"], "first_X_ap_beats_U": compact["ap"] > raw["ap"],
        "first_X_brier_beats_U": compact["brier"] < raw["brier"],
        "first_X_brier_beats_first_training_prior": compact["brier"] < initial["first_training_prior_brier"],
        "X_auc_improves_at_least_two_evaluable_days": sum(r["X"]["auc"]-r["U"]["auc"] > 1e-9 for r in days) >= 2,
        "ticker_day_X_auc_gain_ci_positive": initial["paired_intervals"]["ticker_day"]["comparisons"]["X_minus_U"]["auc_ci95"][0] > 0}
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
        "reference_R306_run": 37211254516, "network_requests": 0, "June_HOLD_opened": False, "entries_retuned": False,
        "full_features": list(full_features), "compact_features": list(COMPACT_FEATURES),
        "arms": {"U": "99-input original tree", "V": "10-input original tree", "W": "99-input fixed linear pipeline",
                 "X": "10-input fixed linear pipeline, PRIMARY", "F": "10-input linear pipeline, first-only training, secondary"},
        "logistic_config": {"C": .1, "l1_ratio": 0., "solver": "lbfgs", "max_iter": 2000, "tol": 1e-8,
                            "weight_sum": "independent training episode count", "clip_quantiles": [.01, .99]},
        "folds": folds, "observability": reports, "checks": checks, "research_gate_pass": all(checks.values()),
        "mechanism_diagnostics": {"first_X_auc_beats_V": compact["auc"] > initial["arms"]["V"]["auc"],
                                  "first_X_auc_beats_W": compact["auc"] > initial["arms"]["W"]["auc"]},
        "limitations": "Four reused days,97 upstream-OOF-conditioned episodes and nine first winners. Model+preprocessing is one package; missing indicators double linear dimensions. Per-episode regularization mass does not remove trajectory correlation or guarantee sufficient shrinkage. F is secondary, not a fallback arm to choose. No future/own-day fitting, outcome-based feature selection, parameter search, policy/threshold/calibration changes, June access or profit claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.states_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.states_output, index=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({"checks": checks, "first_arms": initial["arms"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
