"""R327: observed feature-path innovation with fixed current and anchor controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from . import prediction_only_sequence_momentum as previous
from .canonical_population_momentum import verify_checkpoint
from .compact_momentum_representation import fit_transform, weighted_quantile
from .elapsed_momentum_evidence import complete_mask
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS
from .lagged_minute_context import CROSSFIT_DAYS
from .latent_state_evolution_momentum import causal_states, frozen_initial, transition_manifest, transition_support
from .nearby_state_momentum import LocalCache, local_manifest
from .state_evolution_momentum import LAG_INPUTS, RankCache, metrics
from .within_episode_momentum import frozen_transform, metrics as raw_metrics

REQUEST_ID = 327
TAU_S = 30.
CURRENT = previous.PACKAGES['F']
EMA = tuple('R327_EMA_innovation_'+f for f in LAG_INPUTS)
ANCHOR = tuple('R327_anchor_innovation_'+f for f in LAG_INPUTS)
NEW_ARMS = ('F', 'N', 'S')
ARMS = (*NEW_ARMS, 'C', 'D', 'B', 'B0')
METRICS = previous.METRICS
CONTRASTS = (('F', 'C'), ('F', 'N'), ('F', 'S'), ('F', 'D'), ('F', 'B'), ('F', 'B0'), ('N', 'C'), ('S', 'D'))
STATIC_PARAMETERS = previous.STATIC_PARAMETERS.copy()


def path_columns():
    return tuple('R327_'+kind+'_'+f for f in LAG_INPUTS for kind in ('ema_before', 'anchor_before',
        'last_finite_before_t', 'history_age_s', 'EMA_innovation', 'anchor_innovation', 'ema_after', 'anchor_after', 'last_finite_after_t'))


def feature_paths(states):
    """Emit innovations BEFORE update; every feature carries its last finite clock."""
    frame, clock = causal_states(states), previous.history(states)
    values = frame[list(LAG_INPUTS)].to_numpy(float)
    if np.isinf(values).any():
        raise ValueError('infinite observed feature')
    times = frame.decision_t.to_numpy(float)
    ema, anchor, last, innovation, fixed, prior_ema, prior_anchor, prior_t = (np.full_like(values, np.nan) for _ in range(8))
    ema[clock.first] = anchor[clock.first] = values[clock.first]
    last[clock.first] = np.where(np.isfinite(values[clock.first]), times[clock.first, None], np.nan)
    for now in clock.steps:
        prev = clock.previous[now]
        past, origin, current = ema[prev], anchor[prev], values[now]
        prior_ema[now], prior_anchor[now], prior_t[now] = past, origin, last[prev]
        valid, known = np.isfinite(current), np.isfinite(past)
        z = np.exp(-(times[now, None]-last[prev])/1000/TAU_S)
        innovation[now] = np.where(valid & known, current-past, np.nan)
        fixed[now] = np.where(valid & np.isfinite(origin), current-origin, np.nan)
        ema[now] = np.where(valid, np.where(known, z*past+(1-z)*current, current), past)
        anchor[now] = np.where(np.isfinite(origin), origin, current)
        last[now] = np.where(valid, times[now, None], last[prev])
    quantities = {'ema_before': prior_ema, 'anchor_before': prior_anchor, 'last_finite_before_t': prior_t,
        'history_age_s': (times[:, None]-prior_t)/1000, 'EMA_innovation': innovation, 'anchor_innovation': fixed,
        'ema_after': ema, 'anchor_after': anchor, 'last_finite_after_t': last}
    columns = {'R327_'+kind+'_'+f: v[:, j] for kind, v in quantities.items() for j, f in enumerate(LAG_INPUTS)}
    frame = frame.drop(columns=[c for c in columns if c in frame])
    return pd.concat([frame, pd.DataFrame(columns, index=frame.index)], axis=1)


def path_digest(frame, columns):
    return hashlib.sha256(np.ascontiguousarray(frame[list(columns)].to_numpy(float), dtype='<f8').tobytes()).hexdigest()


@dataclass(frozen=True)
class Blocks:
    current: object
    innovation: object

    def apply(self, frame):
        return np.c_[self.current.apply(frame), self.innovation.apply(frame)]

    def audit(self):
        return {'current_block': self.current.audit(), 'innovation_block': self.innovation.audit(),
            'processed_dimensions': 2*(len(self.current.features)+len(self.innovation.features))}


def frozen_current(training, manifest, audit):
    if audit['fit_days'] != sorted(training.trading_day.unique()):
        raise ValueError('current preprocessing must have identical preceding fit scope')
    transform = frozen_transform(audit['preprocessing'])
    reproduced = fit_transform(training.loc[manifest.current_index], CURRENT, manifest.transform_weight.to_numpy()).audit()
    saved = transform.audit()
    if saved['raw_features'] != list(CURRENT) or saved['all_training_missing'] != reproduced['all_training_missing']:
        raise ValueError('frozen current feature schema changed')
    for key in ('lower', 'upper', 'median', 'mean', 'scale'):
        np.testing.assert_allclose(saved[key], reproduced[key], atol=1e-9, rtol=0)
    return transform


def fit_models(training, manifest, initial, baseline_audit):
    if training.empty:
        return {a: (None, None) for a in NEW_ARMS}, {a: {'no_support': True, 'optimization_attempted': False,
            'reason': 'no_preceding_out_of_time_meta_day'} for a in NEW_ARMS}
    if not set(training.trading_day).issubset(CROSSFIT_DAYS[1:]):
        raise ValueError('fixed out-of-time meta training dates required')
    reproduced, _ = transition_manifest(training)
    verify_checkpoint(manifest.reset_index(drop=True), reproduced)
    if manifest.empty:
        return {a: (None, None) for a in NEW_ARMS}, {a: {'no_support': True, 'optimization_attempted': False,
            'reason': 'empty_transition_manifest'} for a in NEW_ARMS}
    training = feature_paths(training)
    current = frozen_current(training, manifest, baseline_audit)
    currents, clock = training.loc[manifest.current_index], previous.history(training)
    weights, labels = previous.target_weights(training), truth(training[CEILING], 5)
    transforms = {a: Blocks(current, fit_transform(currents, features, manifest.transform_weight.to_numpy())) for a, features in (('F', EMA), ('N', ANCHOR))}
    transforms['S'] = transforms['F']
    bundles, audits = {}, {}
    for arm in NEW_ARMS:
        transform = transforms[arm]
        if arm != 'S':
            model, audit = previous.solve_sequence(transform.apply(training), manifest, clock, initial, labels, weights)
        else:
            later = (weights > 0) & ~clock.first
            model = None
            audit = {'no_support': len(np.unique(labels[later])) < 2, 'optimization_attempted': False,
                'parameters': STATIC_PARAMETERS.copy(), 'supervision_weight_sum': float(weights[later].sum()),
                'complete_mass_including_first': float(weights.sum()), 'first_loss_weight': float(weights[clock.first].sum()),
                'objective': 'stationary_logistic_NLL'}
            if not audit['no_support']:
                audit['optimization_attempted'] = True
                try:
                    model = LogisticRegression(**STATIC_PARAMETERS)
                    x = transform.apply(training.loc[later])
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', ConvergenceWarning)
                        model.fit(x, labels[later], sample_weight=weights[later])
                    eta = model.decision_function(x)
                    later_loss = np.sum(weights[later]*(np.logaddexp(0., eta)-labels[later]*eta))
                    first = clock.first & (weights > 0)
                    first_loss = -np.sum(weights[first]*np.where(labels[first] == 1, np.log(initial[first]), np.log1p(-initial[first])))
                    raw = later_loss+first_loss+np.sum(model.coef_[0]**2)/(2*STATIC_PARAMETERS['C'])
                    derivative = weights[later]*(expit(eta)-labels[later])/weights.sum()
                    gradient = derivative@x+model.coef_[0]/(STATIC_PARAMETERS['C']*weights.sum())
                    audit |= {'optimization_success': True, 'iterations': int(model.n_iter_.max()),
                        'coefficients': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]),
                        'normalized_objective_including_first': float(raw/weights.sum()),
                        'normalized_gradient_inf': float(max(np.max(np.abs(gradient)), abs(derivative.sum())))}
                except (ValueError, ConvergenceWarning) as error:
                    model = None
                    audit |= {'optimization_success': False, 'message': type(error).__name__+': '+str(error)}
        bundles[arm] = model, transform
        audits[arm] = audit | {'fit_days': sorted(training.trading_day.unique()), 'preprocessing': transform.audit(),
            'transform_states': len(currents), 'transform_episodes': len(currents[KEYS].drop_duplicates()),
            'transform_weight_sum': float(manifest.transform_weight.sum()), 'tau_s': TAU_S}
        print(f"R327 {arm} fit: success={model is not None}, iterations={audit.get('iterations')}", flush=True)
    return bundles, audits


def attach_controls(frame, saved):
    """Frozen scores join by identities/clocks; no outcomes are read."""
    verify_checkpoint(frame[[*KEYS, 'decision_t']], saved[[*KEYS, 'decision_t']])
    columns = {}
    for old, new in (('F', 'C'), ('S', 'D'), ('B', 'B'), ('B0', 'B0')):
        prefix = f'R326_{old}_'
        columns |= {c.replace(prefix, f'R327_{new}_', 1): saved[c].to_numpy() for c in saved if c.startswith(prefix)}
    return pd.concat([frame, pd.DataFrame(columns, index=frame.index)], axis=1)


def infer_with_initial(states, bundles, current_g, priors, controls):
    result, clock = feature_paths(states), previous.history(states)
    initial = np.asarray(current_g, float).copy()
    for now in clock.steps:
        initial[now] = initial[clock.previous[now]]
    columns = {}
    for arm in NEW_ARMS:
        model, transform = bundles[arm]
        p = np.full(len(result), np.nan)
        lp = lc = np.full(len(result), np.nan)
        quantities = {n: np.full(len(result), np.nan) for n in ('pi', 'q01', 'q11', 'z')}
        log_rates = rates = np.full((len(result), 2), np.nan)
        if model is not None and arm == 'S':
            p = model.predict_proba(transform.apply(result))[:, 1]
        elif model is not None:
            logits = model.logits(transform.apply(result))
            lp, lc, quantities, *_ = previous.forward(logits, clock, initial)
            p = np.exp(lp)
            log_rates = logits-np.log(previous.RATE_UNIT_S)
            with np.errstate(over='raise', invalid='raise'):
                rates = np.exp(log_rates)
            if not np.isfinite(rates).all():
                raise ValueError('nonfinite persisted rate')
        p[clock.first] = initial[clock.first]
        if model is None or arm == 'S':
            with np.errstate(divide='ignore'):
                lp, lc = np.log(p), np.log1p(-p)
        columns[f'R327_{arm}_p_net_5'], columns[f'R327_{arm}_prior_net_5'] = p, priors
        columns[f'R327_{arm}_log_p_net_5'], columns[f'R327_{arm}_log_1_minus_p'] = lp, lc
        columns[f'R327_{arm}_stationary_p'] = quantities['pi']
        for name in ('q01', 'q11', 'z'):
            columns[f"R327_{arm}_{'retention' if name == 'z' else name}"] = np.where(clock.first, np.nan, quantities[name])
        for j, name in enumerate(('onset', 'decay')):
            columns[f'R327_{arm}_log_{name}_rate_per_s'], columns[f'R327_{arm}_{name}_rate_per_s'] = log_rates[:, j], rates[:, j]
    result = attach_controls(pd.concat([result, pd.DataFrame(columns, index=result.index)], axis=1), controls)
    if not np.array_equal(result.R327_B_p_net_5.to_numpy(), current_g):
        raise ValueError('unchanged canonical current G required')
    if not np.array_equal(result.R327_B0_p_net_5.to_numpy(), initial):
        raise ValueError('own first canonical G changed')
    return result, {'unscored_observations': {a: int(result[f'R327_{a}_p_net_5'].isna().sum()) for a in ARMS}}


def infer(states, bundles, initial_audit, controls):
    reconstructed, error = frozen_initial(states, initial_audit)
    canonical, canonical_error = previous.canonical_initial(reconstructed, controls.R326_B_p_net_5.to_numpy())
    result, audit = infer_with_initial(states, bundles, canonical, initial_audit['prior'], controls)
    return result, audit | {'maximum_saved_G_score_replay_error': error, 'maximum_canonical_G_replay_error': canonical_error}


def probability_frame(frame, arms=ARMS):
    columns = [f"R327_{a}_{s}" for a in arms for s in ("p_net_5", "prior_net_5")]
    return frame[[*KEYS, "decision_t", "observation_available", "label_complete", CEILING, *columns]].rename(columns={c: c.removeprefix("R327_") for c in columns})


def raw_frame(frame, arms=ARMS):
    result = frame[[*KEYS, "decision_t", "observation_available", "label_complete", CEILING]].copy()
    for a in arms:
        result[f"{a}_score"] = frame[f"R327_{a}_p_net_5"]
    return result


def report_metrics(frame, pairs, arms=ARMS):
    finite = np.isfinite(frame[[f"R327_{a}_p_net_5" for a in arms]].to_numpy(float)).all(axis=1)
    ranked = frame.loc[finite]
    usable = pairs.loc[pairs.positive_index.isin(ranked.index) & pairs.negative_index.isin(ranked.index)].reset_index(drop=True)
    scored = metrics(probability_frame(ranked, arms), arms)
    local = LocalCache(raw_frame(ranked, arms), usable, arms).evaluate()
    return {a: scored[a] | local[a] for a in arms}


def intervals(frame, pairs, draws=1000):
    complete = frame.loc[complete_mask(frame)]
    cache, local = RankCache(probability_frame(complete), ARMS), LocalCache(raw_frame(complete), pairs, ARMS)
    output = {}
    for scheme, columns, seed in (("day", ["trading_day"], 20266150), ("ticker_day", ["trading_day", "ticker"], 20266151)):
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
        print(f"R327 {scheme} bootstrap complete", flush=True)
    return output




def gate(report):
    a, support, rank = report['arms'], report['training_support'], 'between_episode_same_day_auc'
    cells = support['cells']
    def greater(x, y):
        return x is not None and y is not None and x > y
    checks = {'at_least20_onset_training_episodes': cells.get('0_to_1', {}).get('episodes', 0) >= 20,
        'at_least20_decay_training_episodes': cells.get('1_to_0', {}).get('episodes', 0) >= 20,
        'onset_decay_at_least3_training_dates': all(len(cells.get(c, {}).get('dates', [])) >= 3 for c in ('0_to_1', '1_to_0')),
        'at_least200_transition_training_episodes': support['episodes'] >= 200,
        'at_least50_mixed_evaluation_episodes': a['F']['within_episode_evaluable_episodes'] >= 50,
        'at_least30_local_evaluation_episodes': a['F']['local_evaluable_episodes'] >= 30,
        'at_least4_positive_evaluation_dates': sum(v['positive_episodes'] > 0 for v in report['per_day'].values()) >= 4,
        'F_admission_beats_N_C_S_D_B_B0': all(greater(a['F'][rank], a[x][rank]) for x in ('N', 'C', 'S', 'D', 'B', 'B0')),
        'F_timing_beats_N_C_S_D_B_and_half': all(greater(a['F']['within_episode_auc'], v) for v in [a[x]['within_episode_auc'] for x in ('N', 'C', 'S', 'D', 'B')]+[.5]),
        'F_local_beats_N_C_S_D_B_and_half': all(greater(a['F']['local_pair_rank'], v) for v in [a[x]['local_pair_rank'] for x in ('N', 'C', 'S', 'D', 'B')]+[.5]),
        'F_brier_beats_N_C_S_D_B_B0_and_prior': all(greater(v, a['F']['brier']) for v in [a[x]['brier'] for x in ('N', 'C', 'S', 'D', 'B', 'B0')]+[a['F']['prior_brier']])}
    days = [v['arms'] for v in report['per_day'].values() if v['arms']['B'][rank] is not None]
    checks['F_B_admission_improves_majority_dates'] = bool(days) and sum(greater(v['F'][rank], v['B'][rank]) for v in days) > len(days)/2
    requirements = [('F_minus_B', rank, True), ('F_minus_B', 'within_episode_auc', True), ('F_minus_B', 'local_pair_rank', True),
        ('F_minus_N', rank, True), ('F_minus_C', rank, True), ('F_minus_B', 'brier', False)]
    requirements += [(f'F_minus_{control}', metric, True) for control in ('C', 'N', 'S') for metric in ('within_episode_auc', 'local_pair_rank')]
    for contrast, metric, positive in requirements:
        for scheme in ('day', 'ticker_day'):
            ci = report['paired_intervals'][scheme]['comparisons'][contrast][f'{metric}_ci95']
            checks[f"{scheme}_{contrast}_{metric}_ci_{'positive' if positive else 'negative'}"] = ci is not None and (ci[0] > 0 if positive else ci[1] < 0)
    if len(checks) != 36:
        raise ValueError('registered36 checks changed')
    return checks


def persist_audit(output, audit):
    (output/'request327-fit-audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')


def chronological(training, saved_training, saved_chrono, prior, baseline_report, output, fit_audit):
    pieces, folds = [], {}
    for day in CROSSFIT_DAYS[1:]:
        preceding = training.loc[training.trading_day.lt(day)]
        manifest = transition_manifest(preceding)[0] if not preceding.empty else pd.DataFrame()
        p = previous.prefix_initial(preceding, prior, saved_training.loc[preceding.index, 'R326_B_p_net_5'].to_numpy())[0] if not preceding.empty else np.array([])
        baseline = baseline_report['chronological_training']['folds'][day]['fits']['F']
        bundles, fits = fit_models(preceding, manifest, p, baseline)
        folds[day] = {'fits': fits, 'fit_days': sorted(preceding.trading_day.unique()),
            'training_support': transition_support(manifest) if not preceding.empty else {'rows': 0, 'episodes': 0, 'cells': {}, 'conditional': {}}}
        fit_audit['folds'][day] = folds[day]
        persist_audit(output, fit_audit)
        if any(bundles[a][0] is None and not fits[a]['no_support'] for a in NEW_ARMS):
            raise ValueError('registered fold numerical failure audited; no replacement evaluation')
        held = training.loc[training.trading_day.eq(day)]
        scored, inference = infer(held, bundles, prior['chronological_training']['folds'][day]['G'], saved_chrono.loc[held.index])
        folds[day]['inference'] = inference
        pieces.append(scored)
    persist_audit(output, fit_audit)
    return pd.concat(pieces).sort_index(), folds


def diagnostics(frame, pairs):
    complete = complete_mask(frame).to_numpy()
    weights = previous.target_weights(frame)
    nonfirst = complete & frame.state_previous_t.notna().to_numpy()
    retention = {}
    for arm in ('F', 'N', 'C'):
        z = frame.loc[nonfirst, f'R327_{arm}_retention'].to_numpy()
        w = weights[nonfirst]
        retention[arm] = {'complete_nonfirst_states': int(nonfirst.sum()), 'weight_sum': float(w.sum()),
            'weighted_mean': float(np.average(z, weights=w)), **{f'q{int(q*100)}': weighted_quantile(z, w, q) for q in (.1, .5, .9)}}
    ages = {}
    for feature in LAG_INPUTS:
        v = frame['R327_history_age_s_'+feature].to_numpy(float)
        chosen = complete & np.isfinite(v)
        w = weights[chosen]
        ages[feature] = {'complete_targets': int(chosen.sum()), 'weight_sum': float(w.sum()),
            'weighted_mean_s': float(np.average(v[chosen], weights=w)),
            **{f'q{int(q*100)}_s': weighted_quantile(v[chosen], w, q) for q in (.1, .5, .9)}}
    available = np.isfinite(frame[list(EMA)].to_numpy(float))
    slices = {}
    for name, mask in (('all_seven', available.all(axis=1)), ('partial', available.any(axis=1) & ~available.all(axis=1)), ('none', ~available.any(axis=1))):
        subset = frame.loc[mask]
        slices[name] = {'observations': len(subset), 'complete_targets': int((mask & complete).sum()),
            'complete_episodes': len(frame.loc[mask & complete, KEYS].drop_duplicates()), 'original_complete_weight_mass': float(weights[mask].sum()),
            'arms': report_metrics(subset, pairs)}
    return {'retention': retention, 'feature_history_age': ages, 'innovation_availability_slices': slices}


def feature_history_table(training, evaluation, chronological_states):
    pieces = []
    for scope, frame in (('training', training), ('evaluation', evaluation), ('chronological', chronological_states)):
        selected = frame[[*KEYS, 'decision_t', 'observation_available', 'label_complete', CEILING, *path_columns()]].copy()
        selected.insert(0, 'source_index', frame.index.to_numpy())
        selected.insert(0, 'scope', scope)
        selected['original_complete_target_weight'] = previous.target_weights(frame)
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


def parquet(frame, output, suffix):
    frame.to_parquet(output/f'request327-{suffix}.parquet', index=False, compression='zstd', compression_level=19)


def run(previous_dir: Path, pinned: Path, output: Path):
    hashes = json.loads(pinned.read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((previous_dir/name).read_bytes()).hexdigest() != digest:
            raise ValueError('preregistered input changed: '+name)
    if (previous_dir/'official326/source-sha.txt').read_text().strip() != '41117f2efb3b1dc9eef8a81c1ee396f6a64991e9':
        raise ValueError('verified R326 source changed')
    prior = json.loads((previous_dir/'official319/request319.json').read_text())
    baseline_report = json.loads((previous_dir/'official326/request326.json').read_text())
    if prior['request_id'] != 319 or baseline_report['request_id'] != 326 or prior['market_requests'] or baseline_report['market_requests']:
        raise ValueError('official cached provenance required')
    training_raw = pd.read_parquet(previous_dir/'official321/request321-chronological-states.parquet').reset_index(drop=True)
    evaluation = pd.read_parquet(previous_dir/'official321/request321-states.parquet')
    saved_training = pd.read_parquet(previous_dir/'official326/request326-training-states.parquet')
    saved = pd.read_parquet(previous_dir/'official326/request326-states.parquet')
    saved_chrono = pd.read_parquet(previous_dir/'official326/request326-chronological-states.parquet')
    verify_checkpoint(training_raw, saved_training[list(training_raw)])
    verify_checkpoint(training_raw, saved_chrono[list(training_raw)])
    verify_checkpoint(evaluation, saved[list(evaluation)])
    ledger = pd.read_parquet(previous_dir/'official319/request319-episode-ledger.parquet')
    clocks = pd.read_parquet(previous_dir/'official319/request319-clock-manifest.parquet')
    verify_checkpoint(clocks, evaluation[list(clocks)])
    pairs = pd.read_parquet(previous_dir/'official321/request321-local-pair-manifest.parquet')
    pair_ledger = pd.read_parquet(previous_dir/'official321/request321-local-pair-ledger.parquet')
    replay_pairs, replay_ledger = local_manifest(evaluation, ledger)
    verify_checkpoint(pairs, replay_pairs)
    verify_checkpoint(pair_ledger, replay_ledger)
    training = feature_paths(training_raw)
    support_audit = json.loads((pinned.parent/'request327-training-support.json').read_text())
    expected = support_audit['training_path_sha256_little_endian_float64']
    if path_digest(training, EMA) != expected['EMA_innovation'] or path_digest(training, ANCHOR) != expected['first_anchor_innovation']:
        raise ValueError('registered training-only feature path fingerprints changed')
    transitions, transition_ledger = transition_manifest(training)
    support = transition_support(transitions)
    if len(training) != 7806 or int(complete_mask(training).sum()) != 7327 or len(evaluation) != 19530 or len(ledger) != 828 or len(transitions) != 7066 or support['episodes'] != 218 or len(pairs) != 8623 or set(evaluation.trading_day) != set(EVAL_DAYS):
        raise ValueError('registered population changed')
    if {k: v['rows'] for k, v in support['cells'].items()} != {'0_to_0': 6265, '0_to_1': 77, '1_to_0': 85, '1_to_1': 639}:
        raise ValueError('registered label support changed')
    initial, priors, initial_audit = previous.prefix_initial(training, prior, saved_training.R326_B_p_net_5.to_numpy())
    output.mkdir(parents=True, exist_ok=True)
    parquet(transitions, output, 'transition-manifest')
    parquet(transition_ledger, output, 'transition-ledger')
    print('R327 original support, weights, prefix G and observed feature path fingerprints verified before fitting', flush=True)
    bundles, fits = fit_models(training, transitions, initial, baseline_report['fits']['F'])
    fit_audit = {'full': fits, 'folds': {}}
    persist_audit(output, fit_audit)
    if any(b[0] is None for b in bundles.values()):
        raise ValueError('registered full fit unsupported/failed; audit saved, no replacement evaluation')
    scored, inference = infer(evaluation, bundles, prior['fits']['G'], saved)
    if any(inference['unscored_observations'].values()):
        raise ValueError('full evaluation contains unsupported scores')
    verify_checkpoint(evaluation, scored[list(evaluation)])
    chrono, folds = chronological(training, saved_training, saved_chrono, prior, baseline_report, output, fit_audit)
    verify_checkpoint(training_raw, chrono[list(training_raw)])
    first = scored.sort_values('decision_t').groupby(KEYS, sort=False).head(1).reset_index(drop=True)
    if not np.all(first[[f'R327_{a}_p_net_5' for a in ARMS]].to_numpy() == first.R327_B_p_net_5.to_numpy()[:, None]):
        raise ValueError('common canonical first predictions changed')
    enriched_training, _ = infer_with_initial(training, bundles, initial, priors, saved_training)
    enriched_training['R327_target_weight'] = previous.target_weights(training)
    enriched_training['R327_prefix_G_current'] = initial
    enriched_training['R327_is_first'] = previous.history(training).first
    feature_history = feature_history_table(enriched_training, scored, chrono)
    for suffix, frame in (('training-states', enriched_training), ('states', scored), ('chronological-states', chrono), ('first-states', first), ('episode-ledger', ledger), ('feature-history', feature_history)):
        parquet(frame, output, suffix)
    report = {'training_support': support, 'observed_states': len(scored), 'complete_states': int(complete_mask(scored).sum()),
        'complete_evaluation_episodes': len(scored.loc[complete_mask(scored), KEYS].drop_duplicates()), 'ledger_episodes': len(ledger),
        'arms': report_metrics(scored, pairs),
        'per_day': {d: {'arms': report_metrics(scored.loc[scored.trading_day.eq(d)], pairs.loc[pairs.trading_day.eq(d)].reset_index(drop=True)),
            'positive_episodes': len(scored.loc[scored.trading_day.eq(d) & complete_mask(scored) & truth(scored[CEILING], 5), KEYS].drop_duplicates())} for d in EVAL_DAYS},
        'paired_intervals': intervals(scored, pairs)}
    for new, old in (('C', 'F'), ('D', 'S'), ('B', 'B'), ('B0', 'B0')):
        for metric in METRICS:
            if abs(report['arms'][new][metric]-baseline_report['comparison']['arms'][old][metric]) > 1e-9:
                raise ValueError('fixed R326 control metric changed')
    chrono_pairs, _ = local_manifest(chrono, chrono[KEYS].drop_duplicates())
    fitted = chrono.loc[chrono.trading_day.gt(CROSSFIT_DAYS[1])]
    gap = (scored.decision_t-scored.state_previous_t)/1000
    result = {'request_id': REQUEST_ID, 'development_only': True, 'promotion_eligible': False, 'market_requests': 0,
        'June_HOLD_opened': False, 'final_July_August_opened': False, 'input_sha256': hashes, 'tau_s': TAU_S,
        'features': {'current': list(CURRENT), 'EMA_innovations': list(EMA), 'first_anchor_innovations': list(ANCHOR)},
        'fits': fits, 'prefix_initial_audit': initial_audit, 'inference': inference, 'comparison': report, 'gate_checks': gate(report),
        'first': metrics(probability_frame(first), ARMS), 'diagnostics': diagnostics(scored, pairs),
        'training_path_sha256': {'EMA_innovation': path_digest(training, EMA), 'first_anchor_innovation': path_digest(training, ANCHOR)},
        'long_gap_diagnostic': {n: metrics(probability_frame(scored.loc[m]), ARMS) for n, m in (('at_most30s', gap.le(30)), ('over30s', gap.gt(30)))},
        'chronological_training': {'observed_states': len(chrono),
            'common_finite_observations': int(np.isfinite(chrono[[f'R327_{a}_p_net_5' for a in ARMS]]).all(axis=1).sum()),
            'arms_common_finite': report_metrics(chrono, chrono_pairs), 'fitted_May7_May8_only': report_metrics(fitted, chrono_pairs),
            'per_day': {d: report_metrics(chrono.loc[chrono.trading_day.eq(d)], chrono_pairs) for d in CROSSFIT_DAYS[1:]}, 'folds': folds},
        'external_saved_references': {'whole_raw_ranks': raw_metrics(scored, ('W', 'U', 'T')), 'local_raw_ranks': LocalCache(scored, pairs, ('W', 'U', 'T')).evaluate()},
        'integrity': {'pinned_hashes_verified': True, 'training_path_fingerprints_verified': True, 'saved_inputs_labels_missingness_scores_preserved': True,
            'inference_and_feature_paths_read_no_outcomes': True, 'all_first_predictions_equal': True, 'strict_prefix_initials_replayed': True,
            'chronological_preceding_fit_scope_verified': True, 'frozen_current_blocks_unchanged': True, 'shared_EMA_blocks_F_S': True,
            'controls_not_refitted': True, 'feature_history_rows': len(feature_history)},
        'limitations': 'Reused May development, latent hindsight reachable net>=5 targets, not realized profit. Existing T38 already contains history/lag deltas; innovation is an extra observed-path summary, not pure removal of static information.EMA and anchor have matched extra dimension count but different information/scales.No sole order-only causal claim.Current features approximate preceding intervals; approximate rates need not identify real Markov regimes.Frozen G initial all-state classifier differs by fold.Fixed-fit bootstrap omits fitting uncertainty.May6 explicitly unsupported later predictions; fitted May7/8-only reported.No tau/feature/optimizer/policy search, fallback selection, sealed-date opening, promotion or live trades.'}
    (output/'request327.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
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
