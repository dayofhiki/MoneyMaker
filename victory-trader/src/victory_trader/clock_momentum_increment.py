"""R310: fixed clock-signal increment with within-day first-state ranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .compact_momentum_representation import fit_linear, predict_linear
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .monotone_momentum_calibration import diagnostic
from .preentry_momentum_observability import CLOCK_FEATURES, SNAPSHOTS, baseline_columns, metrics, snapshots, validate_states
from .risk_reachable_hold_value import validate_fit_days

REQUEST_ID = 310
ARMS = ("N", "Q", "M", "W")
BACKGROUND = ("log_current_price", "minutes_since_open", "minutes_to_close", "return_from_previous_close_pct", "known_cost_drag_pct")
COVERAGE = ("momentum_gap_s", *(f"momentum_activity_fraction_{s}s" for s in (10, 30, 120)))
CONTROL = BACKGROUND+COVERAGE
SIGNAL = tuple(name for name in CLOCK_FEATURES if name not in CONTROL)
MOMENTUM = CONTROL+SIGNAL
CONTRASTS = (("M", "Q"), ("Q", "N"), ("M", "N"), ("W", "Q"))


def within_day_auc(frame: pd.DataFrame, arm: str, weights: np.ndarray | None = None) -> float | None:
    """Weighted SAME-day episode pairs; external weights allow bootstrap draws."""
    if weights is None:
        if frame.duplicated(KEYS).any():
            raise ValueError("one original snapshot per episode required")
        weights = _episode_day_weights(frame) if len(frame) else np.array([])
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(frame) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("positive finite evaluation weights required")
    y, p = truth(frame[CEILING], 5), frame[f"{arm}_p_net_5"].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("finite probabilities in [0,1] required")
    positive, negative = np.flatnonzero(y), np.flatnonzero(~y)
    days = frame.trading_day.to_numpy()
    same = days[positive, None] == days[None, negative]
    pair_weights = weights[positive, None]*weights[None, negative]*same
    denominator = pair_weights.sum()
    if denominator == 0:
        return None
    delta = p[positive, None]-p[None, negative]
    rank = (delta > 0)+.5*(delta == 0)
    return float(np.sum(pair_weights*rank)/denominator)


def increment_crossfit(source: pd.DataFrame, full_features: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    if set(source.trading_day.astype(str)) != set(CROSSFIT_DAYS) or source.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("R310 requires unique May5-8 development states")
    if not set(MOMENTUM).issubset(full_features):
        raise ValueError("fixed packages must be causal subsets of original inputs")
    pieces, folds = [], {}
    for fold, day in enumerate(CROSSFIT_DAYS):
        train, held = source.loc[source.trading_day.ne(day)], source.loc[source.trading_day.eq(day)].copy()
        fit_days = sorted(train.trading_day.unique())
        validate_fit_days(fit_days, excluded=[day])
        seed = 20264100+fold*100+5
        saved = held.W_p_net_5.to_numpy(float).copy()
        audit = {"fit_days": fit_days, "seed": seed, "models": {}}
        for arm, features in (("N", BACKGROUND), ("Q", CONTROL), ("M", MOMENTUM), ("W", full_features)):
            model, transform, prior, model_audit = fit_linear(train, features, seed)
            held[f"{arm}_p_net_5"] = predict_linear(held, model, transform, prior)
            held[f"{arm}_prior_net_5"] = prior
            audit["models"][arm] = model_audit
        error = float(np.max(np.abs(held.W_p_net_5.to_numpy()-saved)))
        if error > 1e-9:
            raise ValueError("saved R309 W score mismatch")
        first = snapshots(train, 0)
        held["first_training_prior"] = float(np.average(truth(first[CEILING], 5), weights=_episode_day_weights(first)))
        audit["maximum_saved_W_error"] = error
        folds[day] = audit
        pieces.append(held)
        print(f"R310 fixed clock increment fold complete {day}", flush=True)
    return pd.concat(pieces, ignore_index=True), folds


def snapshot_report(frame: pd.DataFrame, arm: str, detailed: bool = False) -> dict:
    report = diagnostic(frame, arm) if detailed else metrics(frame, arm, 5)
    report["within_day_auc"] = within_day_auc(frame, arm)
    return report


def interval(values: list[float]) -> list[float] | None:
    return np.quantile(values, [.025, .975]).tolist() if values else None


def paired_intervals(first: pd.DataFrame) -> dict:
    y, w = truth(first[CEILING], 5), _episode_day_weights(first)
    p = {arm: first[f"{arm}_p_net_5"].to_numpy(float) for arm in ARMS}
    output = {}
    for i, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        clusters = list(first.groupby(columns, sort=False).indices.values())
        rng = np.random.default_rng(20264500+i)
        values = {f"{a}_minus_{b}": {"auc": [], "within_day_auc": [], "brier": []} for a, b in CONTRASTS}
        for _ in range(1000):
            rows = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
            sample = first.iloc[rows]
            both = len(np.unique(y[rows])) == 2
            aucs = {a: roc_auc_score(y[rows], p[a][rows], sample_weight=w[rows]) for a in ARMS} if both else {}
            within = {a: within_day_auc(sample, a, w[rows]) for a in ARMS}
            for a, b in CONTRASTS:
                result = values[f"{a}_minus_{b}"]
                if both:
                    result["auc"].append(float(aucs[a]-aucs[b]))
                if within[a] is not None:
                    result["within_day_auc"].append(float(within[a]-within[b]))
                difference = (y[rows]-p[a][rows])**2-(y[rows]-p[b][rows])**2
                result["brier"].append(float(np.average(difference, weights=w[rows])))
        output["day" if i == 0 else "ticker_day"] = {"clusters": len(clusters), "comparisons": {
            key: {f"{metric}_ci95": interval(v) for metric, v in values.items()} |
                 {f"{metric}_valid_draws": len(v) for metric, v in values.items()}
            for key, values in values.items()}}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("states", "summary", "output", "states-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    source, previous = pd.read_parquet(args.states), json.loads(args.summary.read_text())
    validate_states(source)
    if previous.get("request_id") != 309 or not source.label_complete.all() or not np.isfinite(source[CEILING]).all():
        raise ValueError("full finite R309 states carrying original R306 labels required")
    full_features = baseline_columns(source)+CLOCK_FEATURES
    if full_features != tuple(previous["full_features"]):
        raise ValueError("saved causal feature allowlist mismatch")
    scored, folds = increment_crossfit(source, full_features)
    reports = {"all_states": {arm: metrics(scored, arm, 5) for arm in ARMS}}
    for delay in SNAPSHOTS:
        chosen = snapshots(scored, delay)
        w, y = _episode_day_weights(chosen), truth(chosen[CEILING], 5)
        reports[f"snapshot_{delay}s"] = {"episodes": len(chosen), "missing_of97": 97-len(chosen),
            "first_training_prior_brier": float(np.average((y-chosen.first_training_prior.to_numpy())**2, weights=w)),
            "arms": {a: snapshot_report(chosen, a, delay == 0) for a in ARMS},
            "per_day": {str(day): {a: snapshot_report(group, a) for a in ARMS} for day, group in chosen.groupby("trading_day")}}
        if delay == 0:
            reports["snapshot_0s"]["paired_intervals"] = paired_intervals(chosen)
    initial = reports["snapshot_0s"]
    primary, control = initial["arms"]["M"], initial["arms"]["Q"]
    days = [r for r in initial["per_day"].values() if r["Q"]["auc"] is not None]
    ci = initial["paired_intervals"]["ticker_day"]["comparisons"]["M_minus_Q"]["within_day_auc_ci95"]
    checks = {"exact_R309_W_scores": all(f["maximum_saved_W_error"] <= 1e-9 for f in folds.values()),
        "first_M_within_day_auc_beats_Q": primary["within_day_auc"] > control["within_day_auc"],
        "first_M_pooled_auc_beats_Q": primary["auc"] > control["auc"],
        "first_M_ap_beats_Q": primary["ap"] > control["ap"],
        "first_M_brier_beats_Q": primary["brier"] < control["brier"],
        "first_M_brier_beats_first_training_prior": primary["brier"] < initial["first_training_prior_brier"],
        "M_auc_improves_at_least_two_evaluable_days": sum(r["M"]["auc"]-r["Q"]["auc"] > 1e-9 for r in days) >= 2,
        "ticker_day_M_within_day_auc_gain_ci_positive": ci is not None and ci[0] > 0}
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
        "reference_R309_run": 37217714691, "network_requests": 0, "June_HOLD_opened": False, "entries_retuned": False,
        "packages": {"N": list(BACKGROUND), "Q": list(CONTROL), "M": list(MOMENTUM), "W": list(full_features)},
        "added_clock_signal_features": list(SIGNAL), "primary_contrast": "M minus Q at first saved observation",
        "pipeline": "Exact unchanged R309 fit_linear: fixed L2 C=.1, training-only preprocessing, episode-count weight sum",
        "folds": folds, "observability": reports, "checks": checks, "research_gate_pass": all(checks.values()),
        "limitations": "Exploratory reused four days,97 upstream-OOF-conditioned episodes and nine first winners. Within-day pair endpoints remove day score offsets, not shortlist conditioning or correlations. N includes previous-close return; Q covers print quality; M adds the whole clock feature+missing-indicator package, not pure causal momentum. Whole99 W is a replay reference, not a selectable fallback. No parameter/feature/seed/threshold/calibration/policy tuning, June access or executable profit claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.states_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.states_output, index=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({"checks": checks, "first_arms": initial["arms"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
