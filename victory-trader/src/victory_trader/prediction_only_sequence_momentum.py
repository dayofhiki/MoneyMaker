"""R326: matched sequence likelihood, adjacent likelihood and memory-free controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from . import elapsed_state_rate_momentum as predecessor
from .canonical_population_momentum import verify_checkpoint
from .compact_momentum_representation import fit_transform, independent_weights, weighted_quantile
from .elapsed_momentum_evidence import complete_mask
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS
from .lagged_minute_context import CROSSFIT_DAYS
from .latent_state_evolution_momentum import causal_states, frozen_initial, transition_manifest, transition_support
from .nearby_state_momentum import LocalCache, local_manifest
from .state_evolution_momentum import RankCache, metrics
from .within_episode_momentum import metrics as raw_metrics

REQUEST_ID = 326
C, RATE_UNIT_S, OPTIONS = predecessor.C, predecessor.RATE_UNIT_S, predecessor.OPTIONS
PACKAGES = predecessor.PACKAGES | {"M": predecessor.PACKAGES["F"], "S": predecessor.PACKAGES["F"]}
NEW_ARMS = tuple(PACKAGES)
CORE_ARMS = (*NEW_ARMS, "B", "B0")
ARMS = (*CORE_ARMS, "P")
METRICS = predecessor.METRICS
CONTRASTS = (("F", "H"), ("F", "A"), ("F", "K"), ("F", "M"), ("F", "S"), ("F", "B"), ("F", "B0"), ("H", "A"), ("F", "P"))
STATIC_PARAMETERS = dict(C=.2, l1_ratio=0., solver="lbfgs", max_iter=2000, tol=1e-8, fit_intercept=True, random_state=20266005)


@dataclass(frozen=True)
class History:
    previous: np.ndarray
    first: np.ndarray
    gap: np.ndarray
    steps: tuple[np.ndarray, ...]


def history(states):
    frame = causal_states(states)
    previous = np.full(len(frame), -1, int)
    groups = []
    for _, group in frame.groupby(KEYS, sort=False):
        positions = frame.index.get_indexer(group.sort_values("decision_t").index)
        previous[positions[1:]] = positions[:-1]
        groups.append(positions)
    steps = tuple(np.array([g[j] for g in groups if len(g) > j], int) for j in range(1, max(map(len, groups))))
    gap = (frame.decision_t-frame.state_previous_t).to_numpy(float)/1000
    return History(previous, previous < 0, np.nan_to_num(gap, nan=0.), steps)


def forward(logits, clock, initial, derivatives=False):
    """Log-domain evolution of own predictions; labels and censoring are absent."""
    initial = np.asarray(initial, float)
    if len(initial) != len(clock.first) or not np.isfinite(initial).all() or ((initial < 0) | (initial > 1)).any():
        raise ValueError("aligned finite initial probabilities required")
    k = predecessor.kernel(logits, clock.gap)
    lp, lc = np.full(len(initial), np.nan), np.full(len(initial), np.nan)
    with np.errstate(divide="ignore"):
        lp[clock.first], lc[clock.first] = np.log(initial[clock.first]), np.log1p(-initial[clock.first])
    gp, gc = np.zeros((len(initial), 2)), np.zeros((len(initial), 2))
    bp, bc = np.zeros(len(initial)), np.zeros(len(initial))
    share = np.c_[k["pi"], k["complement"]]
    gpi, gcomp = np.c_[k["complement"], -k["complement"]], np.c_[-k["pi"], k["pi"]]
    for now in clock.steps:
        prev = clock.previous[now]
        sp, sc = k["log_pi"][now]+k["log_u"][now], k["log_complement"][now]+k["log_u"][now]
        cp, cc = lp[prev]-k["h"][now], lc[prev]-k["h"][now]
        lp[now], lc[now] = np.logaddexp(sp, cp), np.logaddexp(sc, cc)
        if derivatives:
            ap, ac = np.exp(sp-lp[now]), np.exp(sc-lc[now])
            bp[now], bc[now] = np.exp(cp-lp[now]), np.exp(cc-lc[now])
            hp, hc = np.zeros(len(now)), np.zeros(len(now))
            finite = np.isfinite(k["h"][now])
            hp[finite] = bp[now[finite]]*k["h"][now[finite]]
            hc[finite] = bc[now[finite]]*k["h"][now[finite]]
            gp[now] = ap[:, None]*(gpi[now]+k["v"][now, None]*share[now])-hp[:, None]*share[now]
            gc[now] = ac[:, None]*(gcomp[now]+k["v"][now, None]*share[now])-hc[:, None]*share[now]
    return lp, lc, k, gp, gc, bp, bc


def sequence_loss_gradient(theta, x, clock, initial, labels, weights):
    x, w, y = np.asarray(x, float), np.asarray(weights, float), np.asarray(labels)
    if x.ndim != 2 or len(x) != len(w) or len(y) != len(w) or len(w) != len(clock.first) or not np.isfinite(x).all() or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0 or not np.isin(y[w > 0], (0, 1)).all():
        raise ValueError("aligned finite sequence design, binary targets and positive loss mass required")
    design = np.c_[x, np.ones(len(x))]
    parameters = np.asarray(theta, float).reshape(2, design.shape[1])
    lp, lc, _, gp, gc, bp, bc = forward(design@parameters.T, clock, initial, True)
    target, mass = w > 0, w.sum()
    loss = -np.sum(w[target]*np.where(y[target] == 1, lp[target], lc[target]))
    slopes = parameters[:, :-1]
    value = (loss+np.sum(slopes**2)/(2*C))/mass
    adjp, adjc = np.zeros(len(w)), np.zeros(len(w))
    adjp[target] = -w[target]*(y[target] == 1)/mass
    adjc[target] = -w[target]*(y[target] == 0)/mass
    derivative = np.zeros((len(w), 2))
    for now in reversed(clock.steps):
        prev = clock.previous[now]
        derivative[now] = adjp[now, None]*gp[now]+adjc[now, None]*gc[now]
        adjp[prev] += adjp[now]*bp[now]
        adjc[prev] += adjc[now]*bc[now]
    gradient = derivative.T@design
    gradient[:, :-1] += slopes/(C*mass)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("nonfinite sequence objective/gradient")
    return float(value), gradient.ravel()


def target_weights(training):
    result = np.zeros(len(training))
    complete = complete_mask(training).to_numpy()
    if complete.any():
        result[complete] = independent_weights(training.loc[complete])
    return result


def initialization(manifest, dimensions):
    if set(zip(manifest.previous_label, manifest.current_label)) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        return None, {}
    initial, audit = np.zeros((2, dimensions+1)), {}
    for condition in (0, 1):
        chosen = manifest.previous_label.eq(condition)
        selected = manifest.loc[chosen]
        weights = selected.conditional_weight.to_numpy(float)
        switches = selected.current_label.ne(condition).to_numpy()
        exposure = float(np.sum(weights*selected.gap_ms.to_numpy()/1000))
        count = float(np.sum(weights*switches))
        initial[condition, -1] = np.log(RATE_UNIT_S*count/exposure)
        audit[str(condition)] = {"weighted_switch_count": count, "weighted_exposure_s": exposure,
            "weight_sum": float(weights.sum()), "initial_log_rate_per30s": float(initial[condition, -1])}
    return initial, audit


def solve_sequence(x, manifest, clock, initial_p, labels, weights):
    initial, start = initialization(manifest, x.shape[1])
    if initial is None:
        return None, {"no_support": True, "reason": "empty_or_boundary_transition_support", "optimization_attempted": False}
    args = (x, clock, initial_p, labels, weights)
    audit = {"no_support": False, "optimization_attempted": True, "optimizer": "L-BFGS-B", "options": OPTIONS.copy(),
        "C": C, "rate_unit_s": RATE_UNIT_S, "initialization": start, "supervision_weight_sum": float(weights.sum()),
        "observed_sequence_steps": len(x), "first_loss_weight": float(weights[clock.first].sum()), "objective": "prediction_only_sequence_NLL"}
    try:
        result = minimize(sequence_loss_gradient, initial.ravel(), args=args, method="L-BFGS-B", jac=True, options=OPTIONS)
        value, gradient = sequence_loss_gradient(result.x, *args)
    except (ValueError, FloatingPointError, OverflowError) as error:
        return None, audit | {"optimization_success": False, "message": type(error).__name__+": "+str(error)}
    norm = float(np.max(np.abs(gradient)))
    success = bool(result.success and np.isfinite(result.x).all() and np.isfinite(value) and norm <= 1e-6)
    parameters = result.x.reshape(initial.shape)
    audit |= {"optimization_success": success, "iterations": int(result.nit), "message": str(result.message),
        "normalized_objective": value, "normalized_gradient_inf": norm,
        "coefficients": parameters[:, :-1].tolist(), "intercepts": parameters[:, -1].tolist()}
    return (predecessor.RateModel(parameters[:, :-1], parameters[:, -1]) if success else None), audit


def canonical_initial(reconstructed, saved):
    saved = np.asarray(saved, float)
    if saved.shape != reconstructed.shape or not np.isfinite(saved).all() or ((saved < 0) | (saved > 1)).any():
        raise ValueError("aligned finite canonical G required")
    error = float(np.max(np.abs(saved-reconstructed)))
    if error > 1e-9:
        raise ValueError("canonical frozen G replay changed")
    return saved.copy(), error


def prefix_initial(training, prior, canonical=None):
    initial, priors = np.zeros(len(training)), np.zeros(len(training))
    audits = {}
    for day, group in training.groupby("trading_day", sort=False):
        p, error = frozen_initial(group, prior["chronological_training"]["folds"][day]["G"])
        positions = training.index.get_indexer(group.index)
        canonical_error = None
        if canonical is not None:
            p, canonical_error = canonical_initial(p, np.asarray(canonical)[positions])
        initial[positions] = p
        priors[positions] = prior["chronological_training"]["folds"][day]["G"]["prior"]
        audits[day] = {"fit_days": prior["chronological_training"]["folds"][day]["G"]["fit_days"], "maximum_saved_G_score_replay_error": error, "maximum_canonical_G_replay_error": canonical_error}
    return initial, priors, audits


def fit_models(training, manifest, initial_p):
    if training.empty:
        return {a: (None, None) for a in NEW_ARMS}, {a: {"no_support": True, "reason": "no_preceding_out_of_time_meta_day", "optimization_attempted": False} for a in NEW_ARMS}
    if not set(training.trading_day).issubset(CROSSFIT_DAYS[1:]):
        raise ValueError("only registered out-of-time meta training days required")
    reproduced, _ = transition_manifest(training)
    verify_checkpoint(manifest.reset_index(drop=True), reproduced)
    if manifest.empty:
        return {a: (None, None) for a in NEW_ARMS}, {a: {"no_support": True, "reason": "empty_transition_manifest", "optimization_attempted": False} for a in NEW_ARMS}
    currents, clock = training.loc[manifest.current_index], history(training)
    weights, labels = target_weights(training), truth(training[CEILING], 5)
    transforms = {a: fit_transform(currents, PACKAGES[a], manifest.transform_weight.to_numpy()) for a in ("F", "H", "A", "K")}
    transforms["M"] = transforms["S"] = transforms["F"]
    bundles, audits = {}, {}
    for arm in NEW_ARMS:
        transform = transforms[arm]
        if arm == "M":
            model, audit = predecessor.solve(transform.apply(currents), manifest)
            audit["objective"] = "adjacent_conditional_endpoint_NLL"
        elif arm == "S":
            later = (weights > 0) & ~clock.first
            model = None
            audit = {"no_support": len(np.unique(labels[later])) < 2, "parameters": STATIC_PARAMETERS.copy(),
                "supervision_weight_sum": float(weights[later].sum()), "complete_mass_including_first": float(weights.sum()),
                "first_loss_weight": float(weights[clock.first].sum()), "objective": "stationary_logistic_NLL"}
            if not audit["no_support"]:
                try:
                    model = LogisticRegression(**STATIC_PARAMETERS)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(transform.apply(training.loc[later]), labels[later], sample_weight=weights[later])
                    audit |= {"optimization_success": True, "iterations": int(model.n_iter_.max()),
                        "coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}
                except (ValueError, ConvergenceWarning) as error:
                    model = None
                    audit |= {"optimization_success": False, "message": type(error).__name__+": "+str(error)}
        else:
            model, audit = solve_sequence(transform.apply(training), manifest, clock, initial_p, labels, weights)
        bundles[arm] = model, transform
        audits[arm] = audit | {"fit_days": sorted(training.trading_day.unique()), "preprocessing": transform.audit(),
            "transform_states": len(currents), "transform_episodes": len(currents[KEYS].drop_duplicates()),
            "transform_weight_sum": float(manifest.transform_weight.sum())}
        print(f"R326 {arm} fit: success={model is not None}, iterations={audit.get('iterations')}", flush=True)
    return bundles, audits


def infer_with_initial(states, bundles, current_g, priors):
    result, clock = causal_states(states), history(states)
    current_g = np.asarray(current_g, float)
    result["R326_B_p_net_5"], result["R326_B_prior_net_5"] = current_g, priors
    initial = current_g.copy()
    for now in clock.steps:
        initial[now] = initial[clock.previous[now]]
    result["R326_B0_p_net_5"], result["R326_B0_prior_net_5"] = initial, priors
    columns = {}
    for arm in NEW_ARMS:
        model, transform = bundles[arm]
        probabilities = np.full(len(result), np.nan)
        lp = lc = np.full(len(result), np.nan)
        quantities = {n: np.full(len(result), np.nan) for n in ("q01", "q11", "pi", "z")}
        log_rates = rates = np.full((len(result), 2), np.nan)
        if model is not None and arm == "S":
            probabilities = model.predict_proba(transform.apply(result))[:, 1]
        elif model is not None:
            logits = model.logits(transform.apply(result))
            lp, lc, quantities, *_ = forward(logits, clock, initial)
            probabilities = np.exp(lp)
            log_rates = logits-np.log(RATE_UNIT_S)
            with np.errstate(over="raise", invalid="raise"):
                rates = np.exp(log_rates)
            if not np.isfinite(rates).all():
                raise ValueError("nonfinite persisted rate")
        probabilities[clock.first] = initial[clock.first]
        with np.errstate(divide="ignore"):
            if arm == "S" or model is None:
                lp, lc = np.log(probabilities), np.log1p(-probabilities)
        columns[f"R326_{arm}_p_net_5"], columns[f"R326_{arm}_prior_net_5"] = probabilities, priors
        columns[f"R326_{arm}_log_p_net_5"], columns[f"R326_{arm}_log_1_minus_p"] = lp, lc
        columns[f"R326_{arm}_stationary_p"] = quantities["pi"]
        for name in ("q01", "q11", "z"):
            columns[f"R326_{arm}_{'retention' if name == 'z' else name}"] = np.where(clock.first, np.nan, quantities[name])
        for j, name in enumerate(("onset", "decay")):
            columns[f"R326_{arm}_log_{name}_rate_per_s"], columns[f"R326_{arm}_{name}_rate_per_s"] = log_rates[:, j], rates[:, j]
    result = pd.concat([result, pd.DataFrame(columns, index=result.index)], axis=1)
    return result, {"unscored_observations": {a: int(result[f"R326_{a}_p_net_5"].isna().sum()) for a in CORE_ARMS}}


def infer(states, bundles, initial_audit, canonical=None):
    p, error = frozen_initial(states, initial_audit)
    canonical_error = None
    if canonical is not None:
        p, canonical_error = canonical_initial(p, canonical)
    scored, audit = infer_with_initial(states, bundles, p, initial_audit["prior"])
    return scored, audit | {"maximum_saved_G_score_replay_error": error, "maximum_canonical_G_replay_error": canonical_error}


def attach_predecessor(frame, saved):
    verify_checkpoint(saved[[*KEYS, "decision_t", "label_complete", CEILING]], frame[[*KEYS, "decision_t", "label_complete", CEILING]])
    result = frame.copy()
    result["R326_P_p_net_5"] = saved.R325_F_p_net_5.to_numpy(float)
    result["R326_P_prior_net_5"] = saved.R325_F_prior_net_5.to_numpy(float)
    return result


def probability_frame(frame, arms=ARMS):
    columns = [f"R326_{a}_{s}" for a in arms for s in ("p_net_5", "prior_net_5")]
    return frame[[*KEYS, "decision_t", "observation_available", "label_complete", CEILING, *columns]].rename(columns={c: c.removeprefix("R326_") for c in columns})


def raw_frame(frame, arms=ARMS):
    result = frame[[*KEYS, "decision_t", "observation_available", "label_complete", CEILING]].copy()
    for a in arms:
        result[f"{a}_score"] = frame[f"R326_{a}_p_net_5"]
    return result


def report_metrics(frame, pairs, arms=ARMS):
    finite = np.isfinite(frame[[f"R326_{a}_p_net_5" for a in arms]].to_numpy(float)).all(axis=1)
    ranked = frame.loc[finite]
    usable = pairs.loc[pairs.positive_index.isin(ranked.index) & pairs.negative_index.isin(ranked.index)].reset_index(drop=True)
    scored = metrics(probability_frame(ranked, arms), arms)
    local = LocalCache(raw_frame(ranked, arms), usable, arms).evaluate()
    return {a: scored[a] | local[a] for a in arms}


def intervals(frame, pairs, draws=1000):
    complete = frame.loc[complete_mask(frame)]
    cache, local = RankCache(probability_frame(complete), ARMS), LocalCache(raw_frame(complete), pairs, ARMS)
    output = {}
    for scheme, columns, seed in (("day", ["trading_day"], 20266050), ("ticker_day", ["trading_day", "ticker"], 20266051)):
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
        print(f"R326 {scheme} bootstrap complete", flush=True)
    return output



def gate(report):
    checks = predecessor.gate(report)
    a = report['arms']
    replacements = {
        'F_admission_beats_H_A_K_B_B0': ('between_episode_same_day_auc', ('H', 'A', 'K', 'M', 'S', 'B', 'B0'), False),
        'F_timing_beats_H_A_K_B_and_half': ('within_episode_auc', ('H', 'A', 'K', 'M', 'S', 'B'), False),
        'F_local_beats_H_A_K_B_and_half': ('local_pair_rank', ('H', 'A', 'K', 'M', 'S', 'B'), False),
        'F_brier_beats_H_A_K_B_B0_and_prior': ('brier', ('H', 'A', 'K', 'M', 'S', 'B', 'B0'), True)}
    for key, (metric, controls, lower) in replacements.items():
        candidates = [a[x][metric] for x in controls]
        if 'half' in key:
            candidates.append(.5)
        if 'prior' in key:
            candidates.append(a['F']['prior_brier'])
        checks[key.replace('K_B', 'K_M_S_B')] = all(v is not None and a['F'][metric] is not None and (a['F'][metric] < v if lower else a['F'][metric] > v) for v in candidates)
        del checks[key]
    for control in ('M', 'S'):
        for metric in ('within_episode_auc', 'local_pair_rank'):
            for scheme in ('day', 'ticker_day'):
                ci = report['paired_intervals'][scheme]['comparisons'][f'F_minus_{control}'][f'{metric}_ci95']
                checks[f'{scheme}_F_minus_{control}_{metric}_ci_positive'] = ci is not None and ci[0] > 0
    if len(checks) != 32:
        raise ValueError('registered32 checks changed')
    return checks


def chronological(training, predecessor_states, prior):
    pieces, folds = [], {}
    for day in CROSSFIT_DAYS[1:]:
        preceding = training.loc[training.trading_day.lt(day)]
        manifest = transition_manifest(preceding)[0] if not preceding.empty else pd.DataFrame()
        p = prefix_initial(preceding, prior, predecessor_states.loc[preceding.index, "R325_B_p_net_5"].to_numpy())[0] if not preceding.empty else np.array([])
        bundles, fits = fit_models(preceding, manifest, p)
        held = training.loc[training.trading_day.eq(day)]
        scored, audit = infer(held, bundles, prior['chronological_training']['folds'][day]['G'], predecessor_states.loc[held.index, 'R325_B_p_net_5'].to_numpy())
        scored = attach_predecessor(scored, predecessor_states.loc[predecessor_states.trading_day.eq(day)])
        pieces.append(scored)
        folds[day] = {'fits': fits, 'inference': audit, 'fit_days': sorted(preceding.trading_day.unique()),
            'training_support': transition_support(manifest) if not preceding.empty else {'rows': 0, 'episodes': 0, 'cells': {}, 'conditional': {}}}
    return pd.concat(pieces).sort_index(), folds


def retention_diagnostic(frame):
    selected = frame.loc[complete_mask(frame) & frame.state_previous_t.notna()]
    weights = target_weights(frame)[(complete_mask(frame) & frame.state_previous_t.notna()).to_numpy()]
    result = {}
    for arm in ('F', 'H', 'A', 'K', 'M'):
        z = selected[f'R326_{arm}_retention'].to_numpy()
        result[arm] = {'complete_nonfirst_states': len(selected), 'weight_sum': float(weights.sum()),
            'weighted_mean': float(np.average(z, weights=weights)),
            **{f'q{int(q*100)}': weighted_quantile(z, weights, q) for q in (.1, .5, .9)}}
    return result


def run(previous: Path, pinned: Path, output: Path):
    hashes = json.loads(pinned.read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((previous/name).read_bytes()).hexdigest() != digest:
            raise ValueError('preregistered input hash changed: '+name)
    prior = json.loads((previous/'official319/request319.json').read_text())
    predecessor_report = json.loads((previous/'official325/request325.json').read_text())
    if prior['request_id'] != 319 or predecessor_report['request_id'] != 325 or prior['market_requests'] or predecessor_report['market_requests']:
        raise ValueError('official cached provenance required')
    training = causal_states(pd.read_parquet(previous/'official321/request321-chronological-states.parquet').reset_index(drop=True))
    evaluation = pd.read_parquet(previous/'official321/request321-states.parquet')
    saved = pd.read_parquet(previous/'official325/request325-states.parquet')
    saved_chrono = pd.read_parquet(previous/'official325/request325-chronological-states.parquet')
    verify_checkpoint(evaluation, saved[list(evaluation)])
    verify_checkpoint(training[list(saved_chrono.columns.intersection(training.columns))], saved_chrono[list(saved_chrono.columns.intersection(training.columns))])
    ledger = pd.read_parquet(previous/'official319/request319-episode-ledger.parquet')
    clock = pd.read_parquet(previous/'official319/request319-clock-manifest.parquet')
    verify_checkpoint(clock, evaluation[list(clock)])
    pairs = pd.read_parquet(previous/'official321/request321-local-pair-manifest.parquet')
    pair_ledger = pd.read_parquet(previous/'official321/request321-local-pair-ledger.parquet')
    replay_pairs, replay_ledger = local_manifest(evaluation, ledger)
    verify_checkpoint(pairs, replay_pairs)
    verify_checkpoint(pair_ledger, replay_ledger)
    transitions, transition_ledger = transition_manifest(training)
    support = transition_support(transitions)
    if len(training) != 7806 or int(complete_mask(training).sum()) != 7327 or len(evaluation) != 19530 or len(ledger) != 828 or len(transitions) != 7066 or support['episodes'] != 218 or len(pairs) != 8623 or set(evaluation.trading_day) != set(EVAL_DAYS):
        raise ValueError('registered population changed')
    if {k: v['rows'] for k, v in support['cells'].items()} != {'0_to_0': 6265, '0_to_1': 77, '1_to_0': 85, '1_to_1': 639}:
        raise ValueError('registered transition support changed')
    initial, priors, initial_audit = prefix_initial(training, prior, saved_chrono.R325_B_p_net_5.to_numpy())
    output.mkdir(parents=True, exist_ok=True)
    transitions.to_parquet(output/'request326-transition-manifest.parquet', index=False, compression='zstd')
    transition_ledger.to_parquet(output/'request326-transition-ledger.parquet', index=False, compression='zstd')
    print('R326 registered original population, prefix initials and weights verified before fits', flush=True)
    bundles, fits = fit_models(training, transitions, initial)
    audit_path = output/'request326-fit-audit.json'
    audit_path.write_text(json.dumps(fits, indent=2, allow_nan=False)+'\n')
    if any(b[0] is None for b in bundles.values()):
        raise ValueError('registered full fit unsupported/failed; audit saved, no replacement evaluation')
    scored, inference = infer(evaluation, bundles, prior['fits']['G'], saved.R325_B_p_net_5.to_numpy())
    if any(inference['unscored_observations'].values()):
        raise ValueError('full evaluation contains unsupported core inference')
    scored = attach_predecessor(scored, saved)
    verify_checkpoint(evaluation, scored[list(evaluation)])
    chrono, folds = chronological(training, saved_chrono, prior)
    verify_checkpoint(training, chrono[list(training)])
    first = scored.sort_values('decision_t').groupby(KEYS, sort=False).head(1).reset_index(drop=True)
    if not np.all(first[[f'R326_{a}_p_net_5' for a in ARMS]].to_numpy() == first.R326_B_p_net_5.to_numpy()[:, None]):
        raise ValueError('common first predictions changed')
    enriched_training, _ = infer_with_initial(training, bundles, initial, priors)
    enriched_training['R326_target_weight'] = target_weights(training)
    enriched_training['R326_prefix_G_current'] = initial
    enriched_training['R326_is_first'] = history(training).first
    for suffix, frame in (('training-states', enriched_training), ('states', scored), ('chronological-states', chrono), ('first-states', first), ('episode-ledger', ledger)):
        frame.to_parquet(output/f'request326-{suffix}.parquet', index=False, compression='zstd')
    report = {'training_support': support, 'observed_states': len(scored), 'complete_states': int(complete_mask(scored).sum()),
        'complete_evaluation_episodes': len(scored.loc[complete_mask(scored), KEYS].drop_duplicates()), 'ledger_episodes': len(ledger),
        'arms': report_metrics(scored, pairs),
        'per_day': {d: {'arms': report_metrics(scored.loc[scored.trading_day.eq(d)], pairs.loc[pairs.trading_day.eq(d)].reset_index(drop=True)),
            'positive_episodes': len(scored.loc[scored.trading_day.eq(d) & complete_mask(scored) & truth(scored[CEILING], 5), KEYS].drop_duplicates())} for d in EVAL_DAYS},
        'paired_intervals': intervals(scored, pairs)}
    for metric in METRICS:
        if abs(report['arms']['P'][metric]-predecessor_report['comparison']['arms']['F'][metric]) > 1e-9:
            raise ValueError('frozen R325 predecessor metric changed')
    chrono_pairs, _ = local_manifest(chrono, chrono[KEYS].drop_duplicates())
    fitted = chrono.loc[chrono.trading_day.gt(CROSSFIT_DAYS[1])]
    gap = (scored.decision_t-scored.state_previous_t)/1000
    result = {'request_id': REQUEST_ID, 'development_only': True, 'promotion_eligible': False, 'market_requests': 0,
        'June_HOLD_opened': False, 'final_July_August_opened': False, 'input_sha256': hashes,
        'packages': {a: list(f) for a, f in PACKAGES.items()}, 'fits': fits, 'prefix_initial_audit': initial_audit, 'inference': inference,
        'comparison': report, 'gate_checks': gate(report), 'first': metrics(probability_frame(first), ARMS),
        'retention_diagnostic': retention_diagnostic(scored),
        'long_gap_diagnostic': {n: metrics(probability_frame(scored.loc[m]), ARMS) for n, m in (('at_most30s', gap.le(30)), ('over30s', gap.gt(30)))},
        'chronological_training': {'observed_states': len(chrono),
            'common_finite_observations': int(np.isfinite(chrono[[f'R326_{a}_p_net_5' for a in ARMS]]).all(axis=1).sum()),
            'arms_common_finite': report_metrics(chrono, chrono_pairs), 'fitted_May7_May8_only': report_metrics(fitted, chrono_pairs),
            'per_day': {d: report_metrics(chrono.loc[chrono.trading_day.eq(d)], chrono_pairs) for d in CROSSFIT_DAYS[1:]}, 'folds': folds},
        'external_saved_references': {'whole_raw_ranks': raw_metrics(scored, ('W', 'U', 'T')), 'local_raw_ranks': LocalCache(scored, pairs, ('W', 'U', 'T')).evaluate()},
        'integrity': {'pinned_hashes_verified': True, 'saved_inputs_labels_missingness_scores_preserved': True,
            'inference_reads_no_outcomes': True, 'all_first_predictions_equal': True, 'strict_prefix_initials_replayed': True,
            'chronological_preceding_fit_scope_verified': True, 'shared_rich_transform_F_M_S': True},
        'limitations': 'Reused May development; latent hindsight reachable net>=5 targets, not realized profit. Conditional filtering can be valid; sequence objective and weights/support change together. Frozen initial G is an all-state classifier; full and prefix initial models differ.218 transition episodes,256 complete target episodes, not independent seconds. Censored observations propagate predictions. Current features approximate preceding intervals. Fixed-fit bootstrap omits fitting uncertainty. May6 has no preceding out-of-time meta day and4502 unsupported later scores per new arm; fitted May7/8-only reported separately. Fast-rate collapse is not trajectory evidence. No tuning, fallback selection, sealed dates, promotion or live trades.'}
    (output/'request326.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('previous', 'pinned-inputs', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.previous, args.pinned_inputs, args.output_dir)
    print(json.dumps({'arms': result['comparison']['arms'], 'gate_checks': result['gate_checks']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
