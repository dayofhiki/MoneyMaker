"""R325: fixed continuous-time rates with prediction-only latent evolution."""
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
from .compact_momentum_representation import fit_transform
from .elapsed_momentum_evidence import complete_mask
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS
from .lagged_minute_context import CROSSFIT_DAYS
from .latent_state_evolution_momentum import causal_states, frozen_initial, gate, transition_manifest, transition_support
from .nearby_state_momentum import LocalCache, local_manifest
from .state_evolution_momentum import HISTORY, PACKAGES as PREVIOUS_PACKAGES, RankCache, metrics
from .within_episode_momentum import metrics as raw_metrics

REQUEST_ID = 325
RATE_UNIT_S = 30.
C = .1
PACKAGES = {"F": PREVIOUS_PACKAGES["T"], "H": PREVIOUS_PACKAGES["G"], "A": HISTORY, "K": ()}
CORE_ARMS = ("F", "H", "A", "K", "B", "B0")
ARMS = (*CORE_ARMS, "P")
METRICS = ("between_episode_same_day_auc", "auc", "ap", "brier", "within_episode_auc", "local_pair_rank")
CONTRASTS = (("F", "H"), ("F", "A"), ("F", "K"), ("F", "B"), ("F", "B0"), ("H", "A"), ("F", "P"))
OPTIONS = {"maxiter": 2000, "maxls": 50, "ftol": 1e-13, "gtol": 1e-8}


def kernel(logits: np.ndarray, gap: np.ndarray):
    """Stable log-domain kernel; zero gap is identity, with no probability clamp."""
    eta, d = np.asarray(logits, float), np.asarray(gap, float)
    if eta.shape != (len(d), 2) or not np.isfinite(eta).all() or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("finite two-rate logits and nonnegative elapsed seconds required")
    difference = eta[:, 0]-eta[:, 1]
    if not np.isfinite(difference).all():
        raise ValueError("nonfinite rate log odds")
    pi, complement = expit(difference), expit(-difference)
    log_pi, log_complement = -np.logaddexp(0., -difference), -np.logaddexp(0., difference)
    log_d = np.full(len(d), -np.inf)
    positive = d > 0
    log_d[positive] = np.log(d[positive])
    log_h = np.logaddexp(eta[:, 0], eta[:, 1])-np.log(RATE_UNIT_S)+log_d
    h = np.full(len(d), np.inf)
    representable = log_h <= np.log(np.finfo(float).max)
    h[representable] = np.exp(log_h[representable])
    z, u = np.exp(-h), -np.expm1(-h)
    log_u, v = np.empty(len(d)), np.empty(len(d))
    small = log_h < np.log(1e-6)
    large = h > 700.
    middle = ~(small | large)
    log_u[small] = log_h[small]-.5*h[small]+h[small]**2/24
    v[small] = 1-.5*h[small]+h[small]**2/12
    log_u[large], v[large] = 0., 0.
    log_u[middle] = np.log(u[middle])
    v[middle] = h[middle]*z[middle]/u[middle]
    return {"h": h, "pi": pi, "complement": complement, "log_pi": log_pi,
        "log_complement": log_complement, "z": z, "u": u, "log_u": log_u, "v": v,
        "q01": pi*u, "q11": pi*u+z}


def loss_gradient(theta, x, gap, previous, current, weights):
    """Normalized weighted CTMC endpoint NLL + registered unnormalized L2."""
    x, w = np.asarray(x, float), np.asarray(weights, float)
    p, y, gap = np.asarray(previous), np.asarray(current), np.asarray(gap, float)
    if x.ndim != 2 or len(x) != len(w) or not len(w) or len(p) != len(w) or len(y) != len(w) or len(gap) != len(w):
        raise ValueError("aligned nonempty endpoint design required")
    if not np.isfinite(x).all() or not np.isfinite(w).all() or not np.isfinite(w.sum()) or (w <= 0).any() or (gap <= 0).any() or not np.isin(p, (0, 1)).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("finite design, positive weights/gaps and binary endpoints required")
    design = np.c_[x, np.ones(len(x))]
    parameters = np.asarray(theta, float).reshape(2, design.shape[1])
    k = kernel(design@parameters.T, gap)
    pi, complement, h = k["pi"], k["complement"], k["h"]
    zero = p == 0
    log_rho = np.where(zero, k["log_pi"], k["log_complement"])
    log_alternative = np.where(zero, k["log_complement"], k["log_pi"])
    g_pi, g_complement = np.c_[complement, -complement], np.c_[-pi, pi]
    g_rho = np.where(zero[:, None], g_pi, g_complement)
    g_alternative = np.where(zero[:, None], g_complement, g_pi)
    rate_share = np.c_[pi, complement]
    log_switch = log_rho+k["log_u"]
    log_carry = log_rho-h
    log_stay = np.logaddexp(log_alternative, log_carry)
    carry = np.exp(log_carry-log_stay)
    carried_h = np.zeros(len(h))
    finite = np.isfinite(h)
    carried_h[finite] = carry[finite]*h[finite]
    g_switch = g_rho+k["v"][:, None]*rate_share
    g_stay = np.exp(log_alternative-log_stay)[:, None]*g_alternative+carry[:, None]*g_rho-carried_h[:, None]*rate_share
    switched = p != y
    mass = w.sum()
    slopes = parameters[:, :-1]
    value = (-np.sum(w*np.where(switched, log_switch, log_stay))+np.sum(slopes**2)/(2*C))/mass
    derivative = -(w/mass)[:, None]*np.where(switched[:, None], g_switch, g_stay)
    gradient = derivative.T@design
    gradient[:, :-1] += slopes/(C*mass)
    return float(value), gradient.ravel()


@dataclass(frozen=True)
class RateModel:
    coefficients: np.ndarray
    intercepts: np.ndarray

    def logits(self, values):
        return values@self.coefficients.T+self.intercepts


def solve(x, manifest):
    previous, current = manifest.previous_label.to_numpy(int), manifest.current_label.to_numpy(int)
    weights, gap = manifest.conditional_weight.to_numpy(float), manifest.gap_ms.to_numpy(float)/1000
    cells = set(zip(previous, current))
    if cells != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        return None, {"no_support": True, "reason": "empty_or_single_class_condition_no_finite_rate_fit", "optimization_attempted": False}
    initial = np.zeros((2, x.shape[1]+1))
    initialization = {}
    for condition in (0, 1):
        chosen = previous == condition
        switched = current[chosen] != condition
        count = float(np.sum(weights[chosen]*switched))
        exposure = float(np.sum(weights[chosen]*gap[chosen]))
        initial[condition, -1] = np.log(RATE_UNIT_S*count/exposure)
        initialization[str(condition)] = {"weighted_switch_count": count, "weighted_exposure_s": exposure,
            "initial_log_rate_per30s": float(initial[condition, -1]), "weight_sum": float(weights[chosen].sum())}
    args = (x, gap, previous, current, weights)
    try:
        result = minimize(loss_gradient, initial.ravel(), args=args, method="L-BFGS-B", jac=True, options=OPTIONS)
        value, gradient = loss_gradient(result.x, *args)
    except (ValueError, FloatingPointError, OverflowError) as error:
        return None, {"no_support": False, "optimization_attempted": True, "optimization_success": False,
            "optimizer": "L-BFGS-B", "options": OPTIONS.copy(), "C": C, "rate_unit_s": RATE_UNIT_S,
            "initialization": initialization, "message": type(error).__name__+": "+str(error)}
    norm = float(np.max(np.abs(gradient)))
    success = bool(result.success and np.isfinite(result.x).all() and np.isfinite(value) and norm <= 1e-6)
    parameters = result.x.reshape(initial.shape)
    audit = {"no_support": False, "optimization_attempted": True, "optimization_success": success,
        "optimizer": "L-BFGS-B", "options": OPTIONS.copy(), "C": C, "rate_unit_s": RATE_UNIT_S,
        "initialization": initialization, "supervision_weight_sum": float(weights.sum()), "iterations": int(result.nit),
        "message": str(result.message), "normalized_objective": value, "normalized_gradient_inf": norm,
        "coefficients": parameters[:, :-1].tolist(), "intercepts": parameters[:, -1].tolist()}
    return (RateModel(parameters[:, :-1], parameters[:, -1]) if success else None), audit


def fit_rates(training: pd.DataFrame, manifest: pd.DataFrame):
    if training.empty or not set(training.trading_day).issubset(CROSSFIT_DAYS):
        raise ValueError("original fixed preceding training dates required")
    reproduced, _ = transition_manifest(training)
    verify_checkpoint(manifest.reset_index(drop=True), reproduced)
    bundles, audits = {}, {}
    if manifest.empty:
        return {a: (None, None) for a in PACKAGES}, {a: {"no_support": True, "reason": "empty_transition_manifest", "optimization_attempted": False} for a in PACKAGES}
    currents = training.loc[manifest.current_index]
    weights = manifest.transform_weight.to_numpy(float)
    for arm, features in PACKAGES.items():
        transform = fit_transform(currents, features, weights)
        model, audit = solve(transform.apply(currents), manifest)
        bundles[arm] = model, transform
        audits[arm] = audit | {"fit_days": sorted(currents.trading_day.unique()), "preprocessing": transform.audit(),
            "transform_states": len(currents), "transform_episodes": len(currents[KEYS].drop_duplicates()),
            "transform_weight_sum": float(weights.sum())}
    return bundles, audits


def infer(states: pd.DataFrame, bundles: dict, initial_audit: dict):
    """Current features + clocks + previous predictions; no outcomes are read."""
    result = causal_states(states)
    initial, replay_error = frozen_initial(result, initial_audit)
    result["R325_B_p_net_5"] = initial
    first = result.assign(_initial=initial).sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    first_map = first.set_index(KEYS)._initial.to_dict()
    result["R325_B0_p_net_5"] = [first_map[k] for k in result[KEYS].itertuples(index=False, name=None)]
    gap = (result.decision_t-result.state_previous_t).to_numpy(float)/1000
    missing_gap = np.isnan(gap)
    computational_gap = np.where(missing_gap, 0., gap)
    for arm in PACKAGES:
        model, transform = bundles[arm]
        if model is None:
            quantities = {n: np.full(len(result), np.nan) for n in ("q01", "q11", "pi", "z", "u")}
            log_rates = rates = np.full((len(result), 2), np.nan)
        else:
            logits = model.logits(transform.apply(result))
            quantities = kernel(logits, computational_gap)
            log_rates = logits-np.log(RATE_UNIT_S)
            with np.errstate(over="raise", invalid="raise"):
                rates = np.exp(log_rates)
            if not np.isfinite(rates).all():
                raise ValueError("nonfinite persisted rate")
        for j, name in enumerate(("onset", "decay")):
            result[f"R325_{arm}_log_{name}_rate_per_s"] = log_rates[:, j]
            result[f"R325_{arm}_{name}_rate_per_s"] = rates[:, j]
        result[f"R325_{arm}_stationary_p"] = quantities["pi"]
        for name in ("q01", "q11"):
            result[f"R325_{arm}_{name}"] = np.where(missing_gap, np.nan, quantities[name])
        probabilities = np.full(len(result), np.nan)
        for _, group in result.groupby(KEYS, sort=False):
            positions = result.index.get_indexer(group.sort_values("decision_t").index)
            probabilities[positions[0]] = initial[positions[0]]
            for j in range(1, len(positions)):
                now, previous = positions[j], positions[j-1]
                probabilities[now] = quantities["pi"][now]*quantities["u"][now]+probabilities[previous]*quantities["z"][now]
        result[f"R325_{arm}_p_net_5"] = probabilities
    for arm in CORE_ARMS:
        result[f"R325_{arm}_prior_net_5"] = initial_audit["prior"]
    return result, {"maximum_saved_G_score_replay_error": replay_error,
        "unscored_observations": {a: int(result[f"R325_{a}_p_net_5"].isna().sum()) for a in CORE_ARMS}}


def attach_predecessor(frame, saved):
    verify_checkpoint(saved[[*KEYS, "decision_t", "label_complete", CEILING]], frame[[*KEYS, "decision_t", "label_complete", CEILING]])
    result = frame.copy()
    result["R325_P_p_net_5"] = saved.R324_F_p_net_5.to_numpy(float)
    result["R325_P_prior_net_5"] = saved.R324_F_prior_net_5.to_numpy(float)
    return result


def probability_frame(frame, arms=ARMS):
    columns = [f"R325_{a}_{s}" for a in arms for s in ("p_net_5", "prior_net_5")]
    return frame[[*KEYS, "decision_t", "observation_available", "label_complete", CEILING, *columns]].rename(columns={c: c.removeprefix("R325_") for c in columns})


def raw_frame(frame, arms=ARMS):
    result = frame[[*KEYS, "decision_t", "observation_available", "label_complete", CEILING]].copy()
    for a in arms:
        result[f"{a}_score"] = frame[f"R325_{a}_p_net_5"]
    return result


def report_metrics(frame, pairs, arms=ARMS):
    finite = np.isfinite(frame[[f"R325_{a}_p_net_5" for a in arms]].to_numpy(float)).all(axis=1)
    ranked = frame.loc[finite]
    usable = pairs.loc[pairs.positive_index.isin(ranked.index) & pairs.negative_index.isin(ranked.index)].reset_index(drop=True)
    scored = metrics(probability_frame(ranked, arms), arms)
    local = LocalCache(raw_frame(ranked, arms), usable, arms).evaluate()
    return {a: scored[a] | local[a] for a in arms}


def intervals(frame, pairs, draws=1000):
    complete = frame.loc[complete_mask(frame)]
    cache, local = RankCache(probability_frame(complete), ARMS), LocalCache(raw_frame(complete), pairs, ARMS)
    output = {}
    for scheme, columns, seed in (("day", ["trading_day"], 20265950), ("ticker_day", ["trading_day", "ticker"], 20265951)):
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
        print(f"R325 {scheme} bootstrap complete", flush=True)
    return output


def chronological(training, saved, predecessor, prior):
    pieces, folds = [], {}
    for day in CROSSFIT_DAYS[1:]:
        preceding = training.loc[training.trading_day.lt(day)]
        manifest, _ = transition_manifest(preceding)
        bundles, fits = fit_rates(preceding, manifest)
        held = saved.loc[saved.trading_day.eq(day)]
        scored, audit = infer(held, bundles, prior["chronological_training"]["folds"][day]["G"])
        scored = attach_predecessor(scored, predecessor.loc[predecessor.trading_day.eq(day)])
        pieces.append(scored)
        folds[day] = {"fits": fits, "inference": audit, "training_support": transition_support(manifest)}
    return pd.concat(pieces, ignore_index=True), folds


def run(previous: Path, pinned: Path, output: Path):
    hashes = json.loads(pinned.read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((previous/name).read_bytes()).hexdigest() != digest:
            raise ValueError("preregistered input hash changed: "+name)
    prior = json.loads((previous/"official319/request319.json").read_text())
    predecessor_report = json.loads((previous/"official324/request324.json").read_text())
    if prior["request_id"] != 319 or predecessor_report["request_id"] != 324 or prior["market_requests"] or predecessor_report["market_requests"]:
        raise ValueError("official cached provenance required")
    training = causal_states(pd.read_parquet(previous/"official319/request319-training-states.parquet").reset_index(drop=True))
    evaluation = pd.read_parquet(previous/"official321/request321-states.parquet")
    predecessor = pd.read_parquet(previous/"official324/request324-states.parquet")
    verify_checkpoint(evaluation, predecessor[list(evaluation)])
    ledger = pd.read_parquet(previous/"official319/request319-episode-ledger.parquet")
    clock = pd.read_parquet(previous/"official319/request319-clock-manifest.parquet")
    verify_checkpoint(clock, evaluation[list(clock)])
    local_pairs = pd.read_parquet(previous/"official321/request321-local-pair-manifest.parquet")
    local_ledger = pd.read_parquet(previous/"official321/request321-local-pair-ledger.parquet")
    replay_pairs, replay_ledger = local_manifest(evaluation, ledger)
    verify_checkpoint(local_pairs, replay_pairs)
    verify_checkpoint(local_ledger, replay_ledger)
    transitions, transition_ledger = transition_manifest(training)
    verify_checkpoint(pd.read_parquet(previous/"official324/request324-transition-manifest.parquet"), transitions)
    support = transition_support(transitions)
    if len(training) != 9325 or len(evaluation) != 19530 or len(ledger) != 828 or len(transitions) != 8445 or support["episodes"] != 286 or len(local_pairs) != 8623 or set(evaluation.trading_day) != set(EVAL_DAYS):
        raise ValueError("registered population changed")
    output.mkdir(parents=True, exist_ok=True)
    transitions.to_parquet(output/"request325-transition-manifest.parquet", index=False, compression="zstd")
    transition_ledger.to_parquet(output/"request325-transition-ledger.parquet", index=False, compression="zstd")
    print("R325 original adjacent transitions/weights persisted before fits", flush=True)
    bundles, fits = fit_rates(training, transitions)
    (output/"request325-fit-audit.json").write_text(json.dumps(fits, indent=2, allow_nan=False)+"\n")
    if any(bundle[0] is None for bundle in bundles.values()):
        raise ValueError("registered full rate fit unsupported/failed; audit saved, no replacement evaluation")
    scored, inference = infer(evaluation, bundles, prior["fits"]["G"])
    if any(inference["unscored_observations"].values()):
        raise ValueError("full evaluation contains unsupported core inference")
    scored = attach_predecessor(scored, predecessor)
    verify_checkpoint(evaluation, scored[list(evaluation)])
    saved_chrono = pd.read_parquet(previous/"official321/request321-chronological-states.parquet")
    predecessor_chrono = pd.read_parquet(previous/"official324/request324-chronological-states.parquet")
    verify_checkpoint(saved_chrono, predecessor_chrono[list(saved_chrono)])
    chrono, folds = chronological(training, saved_chrono, predecessor_chrono, prior)
    verify_checkpoint(saved_chrono, chrono[list(saved_chrono)])
    first = scored.sort_values("decision_t").groupby(KEYS, sort=False).head(1).reset_index(drop=True)
    if not np.all(first[[f"R325_{a}_p_net_5" for a in ARMS]].to_numpy() == first.R325_B_p_net_5.to_numpy()[:, None]):
        raise ValueError("common first predictions changed")
    for suffix, frame in (("training-states", training), ("states", scored), ("chronological-states", chrono), ("first-states", first), ("episode-ledger", ledger)):
        frame.to_parquet(output/f"request325-{suffix}.parquet", index=False, compression="zstd")
    report = {"training_support": support, "observed_states": len(scored), "complete_states": int(complete_mask(scored).sum()),
        "complete_evaluation_episodes": len(scored.loc[complete_mask(scored), KEYS].drop_duplicates()), "ledger_episodes": len(ledger),
        "arms": report_metrics(scored, local_pairs),
        "per_day": {d: {"arms": report_metrics(scored.loc[scored.trading_day.eq(d)], local_pairs.loc[local_pairs.trading_day.eq(d)].reset_index(drop=True)),
            "positive_episodes": len(scored.loc[scored.trading_day.eq(d) & complete_mask(scored) & truth(scored[CEILING], 5), KEYS].drop_duplicates())} for d in EVAL_DAYS},
        "paired_intervals": intervals(scored, local_pairs)}
    for metric in METRICS:
        if abs(report["arms"]["P"][metric]-predecessor_report["comparison"]["arms"]["F"][metric]) > 1e-9:
            raise ValueError("fixed R324 rich predecessor metric changed")
    chrono_pairs, _ = local_manifest(chrono, chrono[KEYS].drop_duplicates())
    gap = (scored.decision_t-scored.state_previous_t)/1000
    long_gap = {n: metrics(probability_frame(scored.loc[m]), ARMS) for n, m in (("at_most30s", gap.le(30)), ("over30s", gap.gt(30)))}
    result = {"request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False, "market_requests": 0,
        "June_HOLD_opened": False, "final_July_August_opened": False, "input_sha256": hashes,
        "packages": {a: list(f) for a, f in PACKAGES.items()}, "fits": fits, "inference": inference,
        "comparison": report, "gate_checks": gate(report), "first": metrics(probability_frame(first), ARMS),
        "long_gap_diagnostic": long_gap, "chronological_training": {"observed_states": len(chrono),
            "common_finite_observations": int(np.isfinite(chrono[[f"R325_{a}_p_net_5" for a in ARMS]]).all(axis=1).sum()),
            "arms": report_metrics(chrono, chrono_pairs), "folds": folds},
        "external_saved_references": {"whole_raw_ranks": raw_metrics(scored, ("W", "U", "T")),
            "local_raw_ranks": LocalCache(scored, local_pairs, ("W", "U", "T")).evaluate()},
        "integrity": {"pinned_hashes_verified": True, "saved_inputs_labels_missingness_scores_preserved": True,
            "same_original_transition_manifest_and_weights": True, "inference_reads_no_outcomes": True,
            "all_first_predictions_equal": True, "chronological_preceding_fit_scope_verified": True},
        "limitations": "Reused May development, latent hindsight reachable net>=5 labels, not observed regimes/profits. Approximate two-state rates need not identify true Markov dynamics. Current features approximate the preceding interval; no unobserved paths are reconstructed.8445 transitions are286 independent episodes; conditional support279/54,onset/decay38/48. Parameterization/joint likelihood/no gap covariate change together, so no sole clock-consistency causal claim. Initial frozen G all-state classifier is not optimized for first observations. Fixed-fit intervals omit fitting uncertainty. No policy/threshold/window/feature search, fallback selection, promotion, sealed dates or live trades."}
    (output/"request325.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
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
