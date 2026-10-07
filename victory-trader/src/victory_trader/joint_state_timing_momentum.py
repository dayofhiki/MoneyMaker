"""R323: a fixed convex mixture of absolute and within-episode logistic loss."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .canonical_population_momentum import verify_checkpoint
from .compact_momentum_representation import fit_transform, independent_weights
from .elapsed_momentum_evidence import complete_mask
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS
from .lagged_minute_context import CROSSFIT_DAYS
from .nearby_state_momentum import LocalCache, local_manifest
from .state_evolution_momentum import PACKAGES as PREVIOUS_PACKAGES, RankCache, metrics
from .within_episode_momentum import frozen_transform, metrics as raw_metrics, pair_manifest, verify_pairs

REQUEST_ID = 323
C_VALUE = .1
PACKAGES = {"J": PREVIOUS_PACKAGES["T"], "C": PREVIOUS_PACKAGES["T"],
            "H": PREVIOUS_PACKAGES["G"], "D": PREVIOUS_PACKAGES["G"]}
ALPHAS = {"J": .5, "C": 0., "H": .5, "D": 0.}
ARMS = (*PACKAGES, *(a+"0" for a in PACKAGES))
METRICS = ("between_episode_same_day_auc", "auc", "ap", "brier", "within_episode_auc", "local_pair_rank")
CONTRASTS = (("J", "C"), ("J", "H"), ("J", "D"), ("J", "J0"), ("C", "D"), ("H", "D"))


def loss_gradient(theta, x, y, weights, delta, pair_weights, alpha, c=C_VALUE):
    """Normalized losses with an unpenalized intercept and fixed L2 mass."""
    beta, intercept = theta[:-1], theta[-1]
    mass, pair_mass = weights.sum(), pair_weights.sum()
    state_logit = x@beta+intercept
    state_loss = np.average(np.logaddexp(0., state_logit)-y*state_logit, weights=weights)
    residual = (expit(state_logit)-y)*weights/mass
    gradient = (1-alpha)*(x.T@residual)
    relative_loss = 0.
    if alpha:
        if pair_mass <= 0:
            raise ValueError("positive pair mass required for joint objective")
        pair_logit = delta@beta
        relative_loss = np.average(np.logaddexp(0., -pair_logit), weights=pair_weights)
        gradient += alpha*(delta.T@((expit(pair_logit)-1)*pair_weights/pair_mass))
    regularization = .5*np.dot(beta, beta)/(c*mass)
    value = (1-alpha)*state_loss+alpha*relative_loss+regularization
    gradient += beta/(c*mass)
    return float(value), np.r_[gradient, (1-alpha)*residual.sum()]


@dataclass(frozen=True)
class JointLogit:
    coefficients: np.ndarray
    intercept: float

    def decision_function(self, x):
        return x@self.coefficients+self.intercept

    def predict(self, x):
        return expit(self.decision_function(x))


def solve(x, y, weights, delta, pair_weights, alpha):
    if alpha not in (0., .5) or x.ndim != 2 or len(y) != len(x) or len(weights) != len(x):
        raise ValueError("fixed mixture and aligned state design required")
    if delta.ndim != 2 or delta.shape[1] != x.shape[1] or len(pair_weights) != len(delta):
        raise ValueError("aligned relative design required")
    if not all(np.isfinite(v).all() for v in (x, y, weights, delta, pair_weights)) or (weights <= 0).any() or (pair_weights <= 0).any():
        raise ValueError("finite design and positive independent weights required")
    if len(np.unique(y)) != 2 or not np.isin(y, [0, 1]).all():
        raise ValueError("two absolute label classes required")
    options = {"maxiter": 2000, "maxls": 50, "gtol": 1e-8, "ftol": 64*np.finfo(float).eps}
    result = minimize(loss_gradient, np.zeros(x.shape[1]+1), args=(x, y, weights, delta, pair_weights, alpha),
                      method="L-BFGS-B", jac=True, options=options)
    value, gradient = loss_gradient(result.x, x, y, weights, delta, pair_weights, alpha)
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(value) or np.max(np.abs(gradient)) > 1e-6:
        raise ValueError("fixed optimizer failed convergence contract")
    model = JointLogit(result.x[:-1].copy(), float(result.x[-1]))
    state_loss = float(np.average(np.logaddexp(0., model.decision_function(x))-y*model.decision_function(x), weights=weights))
    pair_loss = float(np.average(np.logaddexp(0., -(delta@model.coefficients)), weights=pair_weights)) if len(delta) else None
    audit = {"method": "L-BFGS-B", "options": options, "zero_start": True, "C": C_VALUE, "alpha": alpha,
        "iterations": int(result.nit), "function_evaluations": int(result.nfev), "success": bool(result.success),
        "message": str(result.message), "objective": value, "normalized_gradient_inf": float(np.max(np.abs(gradient))),
        "absolute_mean_loss": state_loss, "relative_mean_loss": pair_loss,
        "regularization": float(.5*np.dot(model.coefficients, model.coefficients)/(C_VALUE*weights.sum()))}
    return model, audit


def fit(states: pd.DataFrame, pairs: pd.DataFrame, features: tuple[str, ...], alpha: float, frozen: dict):
    if states.empty or not set(states.trading_day).issubset(CROSSFIT_DAYS) or features not in tuple(PACKAGES.values()):
        raise ValueError("registered training dates and features required")
    if not states.index.is_unique or states.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("unique causal training states required")
    verify_pairs(states, pairs)
    train = states.loc[complete_mask(states)]
    weights = independent_weights(train)
    transform = frozen_transform(frozen)
    replay = fit_transform(train, features, weights)
    if transform.features != features or transform.all_missing != replay.all_missing:
        raise ValueError("frozen feature transform changed")
    errors = {n: float(np.max(np.abs(getattr(transform, n)-getattr(replay, n)))) for n in ("lower", "upper", "median", "mean", "scale")}
    if max(errors.values()) > 1e-9:
        raise ValueError("frozen transform does not replay training scope")
    x = transform.apply(train)
    delta = transform.apply(states.loc[pairs.positive_index])-transform.apply(states.loc[pairs.negative_index])
    model, optimizer = solve(x, truth(train[CEILING], 5).astype(int), weights, delta, pairs.pair_weight.to_numpy(), alpha)
    prior = float(np.average(truth(train[CEILING], 5), weights=weights))
    mass, mixed_mass = float(weights.sum()), float(pairs.pair_weight.sum())
    audit = {"fit_days": sorted(train.trading_day.unique()), "preprocessing": transform.audit(), "transform_replay_errors": errors,
        "complete_training_states": len(train), "training_episodes": len(train[KEYS].drop_duplicates()),
        "positive_training_episodes": len(train.loc[truth(train[CEILING], 5), KEYS].drop_duplicates()),
        "mixed_training_episodes": len(pairs[KEYS].drop_duplicates()), "unordered_pairs": len(pairs),
        "state_weight_sum": mass, "pair_weight_sum": mixed_mass, "effective_total_mass": mass,
        "absolute_component_mass": (1-alpha)*mass, "relative_component_mass": alpha*mass,
        "pair_mass_rescaling": mass/mixed_mass if alpha and mixed_mass else 0.,
        "prior": prior, "coefficients": model.coefficients.tolist(), "intercept": model.intercept, "optimizer": optimizer}
    return model, transform, audit


def score(training, pairs, evaluation, previous):
    scored, fits = evaluation.copy(), {}
    for arm, features in PACKAGES.items():
        reference = "T" if arm in ("J", "C") else "G"
        model, transform, audit = fit(training, pairs, features, ALPHAS[arm], previous["fits"][reference]["preprocessing"])
        scored[f"R323_{arm}_score"] = model.decision_function(transform.apply(evaluation))
        scored[f"R323_{arm}_p_net_5"] = model.predict(transform.apply(evaluation))
        scored[f"R323_{arm}_prior_net_5"] = audit["prior"]
        fits[arm] = audit
    first = scored.sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    for arm in PACKAGES:
        columns = [f"R323_{arm}_score", f"R323_{arm}_p_net_5"]
        ref = first[[*KEYS, *columns]].rename(columns={c: c.replace(f"R323_{arm}_", f"R323_{arm}0_") for c in columns})
        scored = scored.merge(ref, on=KEYS, how="left", validate="many_to_one", sort=False)
        scored[f"R323_{arm}0_prior_net_5"] = scored[f"R323_{arm}_prior_net_5"]
    errors = {a: float(np.max(np.abs(scored[f"R323_{a}_p_net_5"]-scored[f"{r}_p_net_5"]))) for a, r in (("C", "T"), ("D", "G"))}
    if max(errors.values()) > 1e-7:
        raise ValueError("real state-only boundary differs from frozen R319 classifier")
    return scored, fits, errors


def probability_frame(frame):
    columns = [f"R323_{a}_{s}" for a in ARMS for s in ("p_net_5", "prior_net_5")]
    return frame[[*KEYS, "decision_t", "observation_available", "label_complete", CEILING, *columns]].rename(columns={c: c.removeprefix("R323_") for c in columns})


def raw_frame(frame):
    columns = [f"R323_{a}_score" for a in ARMS]
    return frame[[*KEYS, "decision_t", "observation_available", "label_complete", CEILING, *columns]].rename(columns={c: c.removeprefix("R323_") for c in columns})


def report_metrics(frame, pairs):
    result = metrics(probability_frame(frame), ARMS)
    local = LocalCache(raw_frame(frame), pairs, ARMS).evaluate()
    return {a: result[a] | local[a] for a in ARMS}


def intervals(frame, pairs, draws=1000):
    complete = frame.loc[complete_mask(frame)]
    cache, local = RankCache(probability_frame(complete), ARMS), LocalCache(raw_frame(complete), pairs, ARMS)
    output = {}
    for scheme, columns, seed in (("day", ["trading_day"], 20265750), ("ticker_day", ["trading_day", "ticker"], 20265751)):
        groups = list(complete.groupby(columns, sort=False).indices.values())
        ids = np.zeros(len(complete), int)
        for j, index in enumerate(groups):
            ids[index] = j
        rng = np.random.default_rng(seed)
        values = {f"{a}_minus_{b}": {m: [] for m in METRICS} for a, b in CONTRASTS}
        for _ in range(draws):
            counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            multiplicity = counts[ids]
            ranked, nearby = cache.evaluate(multiplicity), local.evaluate(multiplicity)
            for a, b in CONTRASTS:
                for metric in METRICS:
                    scored = nearby if metric == "local_pair_rank" else ranked
                    x, y = scored[a][metric], scored[b][metric]
                    if x is not None and y is not None:
                        values[f"{a}_minus_{b}"][metric].append(x-y)
        output[scheme] = {"clusters": len(groups), "draws": draws, "comparisons": {
            c: {f"{m}_ci95": np.quantile(v, [.025, .975]).tolist() if v else None for m, v in item.items()} |
               {f"{m}_valid_draws": len(v) for m, v in item.items()} for c, item in values.items()}}
        print(f"R323 {scheme} bootstrap complete", flush=True)
    return output


def gate(report):
    a, rank = report["arms"], "between_episode_same_day_auc"
    def greater(x, y):
        return x is not None and y is not None and x > y
    checks = {"at_least200_training_episodes": report["training_episodes"] >= 200,
        "at_least30_positive_training_episodes": report["positive_training_episodes"] >= 30,
        "at_least30_mixed_training_episodes": report["mixed_training_episodes"] >= 30,
        "at_least50_mixed_evaluation_episodes": a["J"]["within_episode_evaluable_episodes"] >= 50,
        "at_least30_local_evaluation_episodes": a["J"]["local_evaluable_episodes"] >= 30,
        "at_least4_positive_evaluation_dates": sum(d["positive_episodes"] > 0 for d in report["per_day"].values()) >= 4,
        "J_admission_beats_C_H_D": all(greater(a["J"][rank], a[x][rank]) for x in ("C", "H", "D")),
        "J_timing_beats_C_H_and_half": all(greater(a["J"]["within_episode_auc"], v) for v in [a[x]["within_episode_auc"] for x in ("C", "H")]+[.5]),
        "J_local_beats_C_H_and_half": all(greater(a["J"]["local_pair_rank"], v) for v in [a[x]["local_pair_rank"] for x in ("C", "H")]+[.5]),
        "J_brier_beats_C_H_D_and_prior": all(greater(v, a["J"]["brier"]) for v in [a[x]["brier"] for x in ("C", "H", "D")]+[a["J"]["prior_brier"]])}
    days = [d["arms"] for d in report["per_day"].values() if d["arms"]["C"][rank] is not None]
    checks["J_C_admission_improves_majority_days"] = bool(days) and sum(greater(d["J"][rank], d["C"][rank]) for d in days) > len(days)/2
    for contrast, metric, positive in (("J_minus_C", rank, True), ("J_minus_C", "within_episode_auc", True),
                                      ("J_minus_C", "local_pair_rank", True), ("J_minus_H", rank, True), ("J_minus_C", "brier", False)):
        for scheme in ("day", "ticker_day"):
            ci = report["paired_intervals"][scheme]["comparisons"][contrast][f"{metric}_ci95"]
            checks[f"{scheme}_{contrast}_{metric}_ci_{'positive' if positive else 'negative'}"] = ci is not None and (ci[0] > 0 if positive else ci[1] < 0)
    return checks


def run(previous: Path, pinned: Path, output: Path):
    hashes = json.loads(pinned.read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((previous/name).read_bytes()).hexdigest() != digest:
            raise ValueError("preregistered input hash changed: "+name)
    prior = json.loads((previous/"official319/request319.json").read_text())
    if prior["request_id"] != 319 or prior["market_requests"] != 0:
        raise ValueError("official cached provenance required")
    training = pd.read_parquet(previous/"official319/request319-training-states.parquet").reset_index(drop=True)
    evaluation = pd.read_parquet(previous/"official321/request321-states.parquet")
    ledger = pd.read_parquet(previous/"official319/request319-episode-ledger.parquet")
    manifest = pd.read_parquet(previous/"official319/request319-clock-manifest.parquet")
    verify_checkpoint(manifest, evaluation[list(manifest)])
    pairs = pd.read_parquet(previous/"official320/request320-pair-manifest.parquet")
    pair_ledger = pd.read_parquet(previous/"official320/request320-pair-ledger.parquet")
    reproduced, reproduced_ledger = pair_manifest(training)
    verify_checkpoint(pairs, reproduced)
    verify_checkpoint(pair_ledger, reproduced_ledger)
    local_pairs = pd.read_parquet(previous/"official321/request321-local-pair-manifest.parquet")
    local_ledger = pd.read_parquet(previous/"official321/request321-local-pair-ledger.parquet")
    reproduced_local, reproduced_local_ledger = local_manifest(evaluation, ledger)
    verify_checkpoint(local_pairs, reproduced_local)
    verify_checkpoint(local_ledger, reproduced_local_ledger)
    complete = training.loc[complete_mask(training)]
    positives = len(complete.loc[truth(complete[CEILING], 5), KEYS].drop_duplicates())
    if len(training) != 9325 or len(complete) != 8792 or positives != 62 or len(evaluation) != 19530 or len(ledger) != 828 or len(pairs) != 47825 or len(local_pairs) != 8623 or set(evaluation.trading_day) != set(EVAL_DAYS):
        raise ValueError("registered support changed")
    scored, fits, errors = score(training, pairs, evaluation, prior)
    verify_checkpoint(evaluation, scored[list(evaluation)])
    first = scored.sort_values("decision_t").groupby(KEYS, sort=False).head(1).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    for suffix, frame in (("states", scored), ("first-states", first), ("episode-ledger", ledger),
                          ("pair-manifest", pairs), ("pair-ledger", pair_ledger), ("local-pair-manifest", local_pairs), ("local-pair-ledger", local_ledger)):
        frame.to_parquet(output/f"request323-{suffix}.parquet", index=False, compression="zstd")
    print("R323 fixed joint and boundary fits complete; prior labels/scores retained", flush=True)
    report = {"training_episodes": fits["J"]["training_episodes"], "positive_training_episodes": positives,
        "mixed_training_episodes": fits["J"]["mixed_training_episodes"], "observed_states": len(scored),
        "complete_states": int(complete_mask(scored).sum()), "complete_evaluation_episodes": len(scored.loc[complete_mask(scored), KEYS].drop_duplicates()),
        "ledger_episodes": len(ledger), "arms": report_metrics(scored, local_pairs),
        "per_day": {day: {"arms": report_metrics(scored.loc[scored.trading_day.eq(day)], local_pairs.loc[local_pairs.trading_day.eq(day)].reset_index(drop=True)),
            "positive_episodes": len(scored.loc[scored.trading_day.eq(day) & complete_mask(scored) & truth(scored[CEILING], 5), KEYS].drop_duplicates())} for day in EVAL_DAYS},
        "paired_intervals": intervals(scored, local_pairs)}
    saved_local = LocalCache(scored, local_pairs, ("W", "U", "T")).evaluate()
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
        "market_requests": 0, "June_HOLD_opened": False, "final_July_August_opened": False,
        "input_sha256": hashes, "packages": {a: list(f) for a, f in PACKAGES.items()}, "fits": fits,
        "comparison": report, "first": metrics(probability_frame(first), ARMS), "gate_checks": gate(report),
        "external_saved_references": {"probability_ranks": metrics(scored, ("T", "G", "Q", "X", "X0")),
            "local_raw_ranks": saved_local, "whole_raw_ranks": raw_metrics(scored, ("W", "U", "T"))},
        "integrity": {"pinned_hashes_verified": True, "saved_inputs_labels_missingness_scores_preserved": True,
            "all_training_and_local_pairs_replayed": True, "state_only_boundary_probability_errors": errors,
            "frozen_training_transforms_replayed": True},
        "limitations": "Reused May development.336 absolute episodes but only50 independent mixed episodes for pair loss, not47825 independent pairs. Joint alpha=.5 changes absolute/pair mass at fixed total336 and shared regularization; this is an objective comparison, not a causal market treatment. Candidate sigmoid outputs are not automatically calibrated. Local eligibility is label-conditional diagnostic, never online selection. Fixed-fit whole-cluster intervals omit fitting uncertainty. No feature/mixture/parameter/window search, policy/profits, promotion or sealed-date access."}
    (output/"request323.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("previous", "pinned-inputs", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.previous, args.pinned_inputs, args.output_dir)
    print(json.dumps({"arms": result["comparison"]["arms"], "gate_checks": result["gate_checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
