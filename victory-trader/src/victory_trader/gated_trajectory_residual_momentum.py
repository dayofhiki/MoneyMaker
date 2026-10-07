"""R328: same-gate observed-path corrections on frozen out-of-time G."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from . import observed_path_innovation_momentum as old
from . import prediction_only_sequence_momentum as previous
from .canonical_population_momentum import verify_checkpoint
from .compact_momentum_representation import fit_transform, weighted_quantile
from .elapsed_momentum_evidence import complete_mask
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .frozen_momentum_may_transport import EVAL_DAYS
from .lagged_minute_context import CROSSFIT_DAYS
from .latent_state_evolution_momentum import frozen_initial, transition_manifest, transition_support
from .nearby_state_momentum import LocalCache, local_manifest
from .state_evolution_momentum import LAG_INPUTS, RankCache

REQUEST_ID = 328
NEW_ARMS = ('F', 'N', 'S', 'M')
ARMS = (*NEW_ARMS, 'B', 'B0')
FEATURES = {'F': old.EMA, 'N': old.ANCHOR, 'S': LAG_INPUTS}
CONTRASTS = (('F', 'N'), ('F', 'S'), ('F', 'M'), ('F', 'B'), ('F', 'B0'), ('N', 'B'), ('S', 'B'), ('M', 'B'))
METRICS = previous.METRICS
C = .1
OPTIONS = {'maxiter': 2000, 'maxls': 50, 'ftol': 1e-13, 'gtol': 1e-8}


def gate_mask(frame):
    ema = np.isfinite(frame[list(old.EMA)].to_numpy(float)).all(axis=1)
    anchor = np.isfinite(frame[list(old.ANCHOR)].to_numpy(float)).all(axis=1)
    if not np.array_equal(ema, anchor):
        raise ValueError('EMA/anchor gate mismatch')
    if ema[previous.history(frame).first].any():
        raise ValueError('first gate must be zero')
    return ema


def baseline_logits(probabilities, size):
    p = np.asarray(probabilities, float)
    if p.shape != (size,) or not np.isfinite(p).all() or ((p <= 0) | (p >= 1)).any():
        raise ValueError('aligned strictly interior frozen G required; no clipping')
    return np.log(p)-np.log1p(-p)


def design(frame, active, arm, transform):
    dimensions = 1 if arm == 'M' else 7
    x = np.zeros((len(frame), dimensions))
    if arm == 'M':
        x[active] = 1.
    elif transform is None:
        x[active] = np.nan
    else:
        if tuple(transform.features) != tuple(FEATURES[arm]):
            raise ValueError('registered residual feature schema changed')
        x[active] = transform.apply(frame.loc[active])[:, :7]
        if not np.isfinite(x).all():
            raise ValueError('finite active numeric coordinates required')
    return x


def offset_loss_gradient(beta, x, offset, labels, weights):
    w, y = np.asarray(weights, float), np.asarray(labels, float)
    if x.shape != (len(w), len(beta)) or offset.shape != w.shape or y.shape != w.shape:
        raise ValueError('aligned offset objective required')
    if not np.isfinite(x).all() or not np.isfinite(offset).all() or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('finite design/offset and positive original weight mass required')
    chosen = w > 0
    if not np.isfinite(y[chosen]).all() or not np.isin(y[chosen], [0., 1.]).all():
        raise ValueError('binary complete targets required')
    eta = offset[chosen]+x[chosen]@beta
    loss = np.sum(w[chosen]*(np.logaddexp(0., eta)-y[chosen]*eta))+np.sum(beta**2)/(2*C)
    gradient = x[chosen].T@(w[chosen]*(expit(eta)-y[chosen]))+beta/C
    return float(loss/w.sum()), gradient/w.sum()


def solve(x, offset, labels, weights, active):
    chosen = active & (weights > 0)
    audit = {'no_support': len(np.unique(labels[chosen])) < 2, 'optimization_attempted': False,
        'objective': 'frozen_G_offset_original_complete_NLL', 'C': C,
        'parameters': {'method': 'L-BFGS-B', 'options': OPTIONS.copy(), 'bounds': None, 'starts': 1, 'initialization': 'zeros'},
        'supervision_weight_sum': float(weights.sum()), 'active_weight_sum': float(weights[chosen].sum()),
        'inactive_constant_weight_sum': float(weights[~active].sum()), 'processed_dimensions': x.shape[1]}
    if audit['no_support']:
        return None, audit | {'reason': 'empty_or_single_class_active_targets'}
    initial = np.zeros(x.shape[1])
    audit |= {'optimization_attempted': True, 'initial_coefficients': initial.tolist()}
    try:
        result = minimize(offset_loss_gradient, initial, args=(x, offset, labels, weights), jac=True,
            method='L-BFGS-B', options=OPTIONS.copy())
        value, gradient = offset_loss_gradient(result.x, x, offset, labels, weights)
    except (ValueError, FloatingPointError) as error:
        return None, audit | {'optimization_success': False, 'message': type(error).__name__+': '+str(error)}
    finite = np.isfinite(result.x).all() and np.isfinite(value) and np.isfinite(gradient).all()
    success = bool(result.success and finite and np.max(np.abs(gradient)) <= 1e-6)
    audit |= {'optimization_success': success, 'solver_success': bool(result.success), 'message': str(result.message),
        'iterations': int(result.nit), 'coefficients': result.x.tolist() if np.isfinite(result.x).all() else None,
        'normalized_objective': value if np.isfinite(value) else None,
        'normalized_gradient_inf': float(np.max(np.abs(gradient))) if np.isfinite(gradient).all() else None}
    return result.x.copy() if success else None, audit


def fit_models(training, manifest, current_g):
    if training.empty:
        return {a: (None, None) for a in NEW_ARMS}, {a: {'no_support': True, 'optimization_attempted': False,
            'reason': 'no_preceding_out_of_time_meta_day'} for a in NEW_ARMS}
    if not set(training.trading_day).issubset(CROSSFIT_DAYS[1:]):
        raise ValueError('only fixed out-of-time meta dates required')
    verify_checkpoint(manifest.reset_index(drop=True), transition_manifest(training)[0])
    frame = old.feature_paths(training)
    active, offset = gate_mask(frame), baseline_logits(current_g, len(frame))
    weights, labels = previous.target_weights(frame), truth(frame[CEILING], 5)
    if manifest.empty:
        return {a: (None, None) for a in NEW_ARMS}, {a: {'no_support': True, 'optimization_attempted': False,
            'reason': 'empty_transition_transform_support'} for a in NEW_ARMS}
    currents = frame.loc[manifest.current_index]
    bundles, audits = {}, {}
    for arm in NEW_ARMS:
        transform = None if arm == 'M' else fit_transform(currents, FEATURES[arm], manifest.transform_weight.to_numpy())
        x = design(frame, active, arm, transform)
        beta, audit = solve(x, offset, labels, weights, active)
        bundles[arm] = beta, transform
        audits[arm] = audit | {'fit_days': sorted(frame.trading_day.unique()),
            'preprocessing': None if transform is None else transform.audit(),
            'numeric_columns_only': True, 'intercept': False if arm != 'M' else 'penalized_gate_scalar',
            'transform_states': len(currents), 'transform_episodes': len(currents[KEYS].drop_duplicates()),
            'transform_weight_sum': float(manifest.transform_weight.sum()), 'tau_s': old.TAU_S}
        print(f"R328 {arm}: success={beta is not None}, iterations={audit.get('iterations')}", flush=True)
    return bundles, audits


def infer_with_initial(states, bundles, current_g, priors):
    frame = old.feature_paths(states)
    active, offset = gate_mask(frame), baseline_logits(current_g, len(frame))
    g, clock = np.asarray(current_g, float), previous.history(frame)
    initial = g.copy()
    for now in clock.steps:
        initial[now] = initial[clock.previous[now]]
    columns = {'R328_gate': active, 'R328_G_offset_logit': offset}
    for arm in ARMS:
        if arm in NEW_ARMS:
            beta, transform = bundles[arm]
            x = design(frame, active, arm, transform)
            correction = np.zeros(len(frame))
            correction[active] = np.nan if beta is None else x[active]@beta
            eta = offset+correction
            p = expit(eta)
            copied = ~active | (correction == 0)
            p[copied] = g[copied]
            lp, lc = -np.logaddexp(0., -eta), -np.logaddexp(0., eta)
            lp[copied], lc[copied] = np.log(g[copied]), np.log1p(-g[copied])
            for j in range(x.shape[1]):
                columns[f'R328_{arm}_design_{j}'] = x[:, j]
            columns[f'R328_{arm}_correction'] = correction
        else:
            p = g.copy() if arm == 'B' else initial.copy()
            eta = baseline_logits(p, len(p))
            lp, lc = np.log(p), np.log1p(-p)
        columns[f'R328_{arm}_p_net_5'], columns[f'R328_{arm}_prior_net_5'] = p, priors
        columns[f'R328_{arm}_logit'] = eta
        columns[f'R328_{arm}_log_p_net_5'], columns[f'R328_{arm}_log_1_minus_p'] = lp, lc
        if arm in NEW_ARMS and not np.array_equal(p[~active], g[~active]):
            raise ValueError('gate0 canonical G copy changed')
    result = pd.concat([frame, pd.DataFrame(columns, index=frame.index)], axis=1)
    return result, {'unscored_observations': {a: int(result[f'R328_{a}_p_net_5'].isna().sum()) for a in ARMS},
        'active_observations': int(active.sum()), 'gate0_G_exact': True}


def infer(states, bundles, initial_audit, canonical):
    reconstructed, error = frozen_initial(states, initial_audit)
    g, canonical_error = previous.canonical_initial(reconstructed, canonical)
    scored, audit = infer_with_initial(states, bundles, g, initial_audit['prior'])
    return scored, audit | {'maximum_saved_G_score_replay_error': error, 'maximum_canonical_G_replay_error': canonical_error}


def legacy(frame):
    return frame.rename(columns={c: c.replace('R328_', 'R327_', 1) for c in frame if c.startswith('R328_')})


def report_metrics(frame, pairs):
    return old.report_metrics(legacy(frame), pairs, ARMS)


def intervals(frame, pairs, draws=1000):
    complete = frame.loc[complete_mask(frame)]
    adapted = legacy(complete)
    cache, local = RankCache(old.probability_frame(adapted, ARMS), ARMS), LocalCache(old.raw_frame(adapted, ARMS), pairs, ARMS)
    output = {}
    for scheme, columns, seed in (('day', ['trading_day'], 20266250), ('ticker_day', ['trading_day', 'ticker'], 20266251)):
        groups = list(complete.groupby(columns, sort=False).indices.values())
        ids = np.zeros(len(complete), int)
        for j, index in enumerate(groups):
            ids[index] = j
        rng = np.random.default_rng(seed)
        values = {f'{a}_minus_{b}': {m: [] for m in METRICS} for a, b in CONTRASTS}
        for _ in range(draws):
            counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            ranked, nearby = cache.evaluate(counts[ids]), local.evaluate(counts[ids])
            for a, b in CONTRASTS:
                for metric in METRICS:
                    scored = nearby if metric == 'local_pair_rank' else ranked
                    x, y = scored[a][metric], scored[b][metric]
                    if x is not None and y is not None:
                        values[f'{a}_minus_{b}'][metric].append(x-y)
        output[scheme] = {'clusters': len(groups), 'draws': draws, 'comparisons': {
            c: {f'{m}_ci95': np.quantile(v, [.025, .975]).tolist() if v else None for m, v in item.items()} |
            {f'{m}_valid_draws': len(v) for m, v in item.items()} for c, item in values.items()}}
        print(f'R328 {scheme} bootstrap complete', flush=True)
    return output


def support(frame):
    active = gate_mask(frame)
    complete, y, w = complete_mask(frame).to_numpy(), truth(frame[CEILING], 5), previous.target_weights(frame)
    return {'observations': len(frame), 'active_observations': int(active.sum()), 'complete_targets': int(complete.sum()),
        'complete_weight_mass': float(w.sum()), 'active_complete_targets': int((active & complete).sum()),
        'active_target_episodes': len(frame.loc[active & complete, KEYS].drop_duplicates()),
        'active_positive_episodes': len(frame.loc[active & complete & y, KEYS].drop_duplicates()),
        'active_weight_mass': float(w[active].sum()), 'inactive_constant_weight_mass': float(w[~active].sum()),
        'active_positive_dates': sorted(frame.loc[active & complete & y, 'trading_day'].unique()),
        'active_negative_dates': sorted(frame.loc[active & complete & ~y, 'trading_day'].unique())}


def gate(report, integrity):
    a, s, rank = report['arms'], report['gate_training_support'], 'between_episode_same_day_auc'
    def greater(x, y):
        return x is not None and y is not None and x > y
    checks = {'at_least150_active_training_episodes': s['active_target_episodes'] >= 150,
        'at_least30_active_positive_training_episodes': s['active_positive_episodes'] >= 30,
        'active_both_classes_at_least3_dates': all(len(s[k]) >= 3 for k in ('active_positive_dates', 'active_negative_dates')),
        'at_least200_original_transform_episodes': report['training_support']['episodes'] >= 200,
        'at_least50_mixed_evaluation_episodes': a['F']['within_episode_evaluable_episodes'] >= 50,
        'at_least30_local_evaluation_episodes': a['F']['local_evaluable_episodes'] >= 30,
        'at_least4_positive_evaluation_dates': sum(v['positive_episodes'] > 0 for v in report['per_day'].values()) >= 4,
        'gate0_and_first_G_exact': integrity['gate0_and_first_G_exact'],
        'strict_prefix_G_replayed': integrity['strict_prefix_G_replayed']}
    checks |= {'F_admission_beats_N_S_M_B_B0': all(greater(a['F'][rank], a[x][rank]) for x in ('N', 'S', 'M', 'B', 'B0')),
        'F_whole_beats_N_S_M_B_and_half': all(greater(a['F']['within_episode_auc'], v) for v in [a[x]['within_episode_auc'] for x in ('N', 'S', 'M', 'B')]+[.5]),
        'F_local_beats_N_S_M_B_and_half': all(greater(a['F']['local_pair_rank'], v) for v in [a[x]['local_pair_rank'] for x in ('N', 'S', 'M', 'B')]+[.5]),
        'F_brier_beats_N_S_M_B_B0_prior': all(greater(v, a['F']['brier']) for v in [a[x]['brier'] for x in ('N', 'S', 'M', 'B', 'B0')]+[a['F']['prior_brier']])}
    days = [v['arms'] for v in report['per_day'].values() if v['arms']['B'][rank] is not None]
    checks['F_B_admission_improves_majority_dates'] = bool(days) and sum(greater(v['F'][rank], v['B'][rank]) for v in days) > len(days)/2
    requirements = [('F_minus_B', m, True) for m in (rank, 'within_episode_auc', 'local_pair_rank')]+[('F_minus_B', 'brier', False)]
    requirements += [(f'F_minus_{control}', m, True) for control in ('N', 'S', 'M') for m in ('within_episode_auc', 'local_pair_rank', rank)]
    for contrast, metric, positive in requirements:
        for scheme in ('day', 'ticker_day'):
            ci = report['paired_intervals'][scheme]['comparisons'][contrast][f'{metric}_ci95']
            checks[f"{scheme}_{contrast}_{metric}_ci_{'positive' if positive else 'negative'}"] = ci is not None and (ci[0] > 0 if positive else ci[1] < 0)
    if len(checks) != 40:
        raise ValueError('registered40 checks changed')
    return checks


def persist_audit(output, audit):
    (output/'request328-fit-audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')


def chronological(training, saved, prior, output, audit):
    pieces, folds = [], {}
    for day in CROSSFIT_DAYS[1:]:
        preceding = training.loc[training.trading_day.lt(day)]
        manifest = transition_manifest(preceding)[0] if len(preceding) else pd.DataFrame()
        g = previous.prefix_initial(preceding, prior, saved.loc[preceding.index, 'R326_B_p_net_5'].to_numpy())[0] if len(preceding) else np.array([])
        bundles, fits = fit_models(preceding, manifest, g)
        folds[day] = {'fit_days': sorted(preceding.trading_day.unique()), 'fits': fits,
            'training_gate_support': support(old.feature_paths(preceding)) if len(preceding) else None}
        audit['folds'][day] = folds[day]
        persist_audit(output, audit)
        if any(bundles[a][0] is None and not fits[a]['no_support'] for a in NEW_ARMS):
            raise ValueError('fold numerical failure audited; no replacement evaluation')
        held = training.loc[training.trading_day.eq(day)]
        scored, inference = infer(held, bundles, prior['chronological_training']['folds'][day]['G'], saved.loc[held.index, 'R326_B_p_net_5'].to_numpy())
        folds[day]['inference'] = inference
        pieces.append(scored)
    persist_audit(output, audit)
    return pd.concat(pieces).sort_index(), folds


def tables(training, evaluation, chrono):
    history_pieces, design_pieces = [], []
    for scope_name, frame in (('training', training), ('evaluation', evaluation), ('chronological', chrono)):
        base = [*KEYS, 'decision_t', 'observation_available', 'label_complete', CEILING]
        coordinates = [f'R328_{a}_design_{j}' for a in NEW_ARMS for j in range(1 if a == 'M' else 7)]
        for columns, pieces in (([*base, *old.path_columns()], history_pieces),
                ([*base, 'R328_gate', 'R328_G_offset_logit', *coordinates], design_pieces)):
            selected = frame[columns].copy()
            selected.insert(0, 'source_index', frame.index.to_numpy())
            selected.insert(0, 'scope', scope_name)
            selected['original_complete_target_weight'] = previous.target_weights(frame)
            pieces.append(selected)
    return pd.concat(history_pieces, ignore_index=True), pd.concat(design_pieces, ignore_index=True)


def diagnostics(frame, pairs):
    active, complete, weights = gate_mask(frame), complete_mask(frame).to_numpy(), previous.target_weights(frame)
    slices = {}
    for name, mask in (('active_all_seven', active), ('inactive', ~active)):
        subset = frame.loc[mask]
        slices[name] = {'observations': len(subset), 'complete_targets': int((mask & complete).sum()),
            'complete_episodes': len(frame.loc[mask & complete, KEYS].drop_duplicates()),
            'original_complete_weight_mass': float(weights[mask].sum()), 'arms': report_metrics(subset, pairs)}
    corrections = {}
    for a in NEW_ARMS:
        v = frame[f'R328_{a}_correction'].to_numpy()
        selected = complete & active & np.isfinite(v)
        w = weights[selected]
        corrections[a] = {'complete_active_rows': int(selected.sum()), 'original_weight_mass': float(w.sum()),
            'weighted_mean': float(np.average(v[selected], weights=w)),
            **{f'q{int(q*100)}': weighted_quantile(v[selected], w, q) for q in (.1, .5, .9)}}
    return {'availability_slices': slices, 'logit_corrections': corrections}


def parquet(frame, output, suffix):
    frame.to_parquet(output/f'request328-{suffix}.parquet', index=False, compression='zstd', compression_level=19)


def run(previous_dir, pinned, output):
    pins = json.loads(pinned.read_text())
    for name, digest in pins.items():
        if hashlib.sha256((previous_dir/name).read_bytes()).hexdigest() != digest:
            raise ValueError('registered input changed: '+name)
    if (previous_dir/'official327/source-sha.txt').read_text().strip() != '6c2bb5518187c41d7a59a3e1d9d3085fd9ec50dd':
        raise ValueError('verified R327 source changed')
    prior = json.loads((previous_dir/'official319/request319.json').read_text())
    frozen = json.loads((previous_dir/'official326/request326.json').read_text())
    raw = pd.read_parquet(previous_dir/'official321/request321-chronological-states.parquet').reset_index(drop=True)
    evaluation = pd.read_parquet(previous_dir/'official321/request321-states.parquet')
    saved = pd.read_parquet(previous_dir/'official326/request326-training-states.parquet')
    eval_saved = pd.read_parquet(previous_dir/'official326/request326-states.parquet')
    verify_checkpoint(raw, saved[list(raw)])
    verify_checkpoint(evaluation, eval_saved[list(evaluation)])
    ledger = pd.read_parquet(previous_dir/'official319/request319-episode-ledger.parquet')
    clocks = pd.read_parquet(previous_dir/'official319/request319-clock-manifest.parquet')
    verify_checkpoint(clocks, evaluation[list(clocks)])
    pairs = pd.read_parquet(previous_dir/'official321/request321-local-pair-manifest.parquet')
    pair_ledger = pd.read_parquet(previous_dir/'official321/request321-local-pair-ledger.parquet')
    replay_pairs, replay_ledger = local_manifest(evaluation, ledger)
    verify_checkpoint(pairs, replay_pairs)
    verify_checkpoint(pair_ledger, replay_ledger)
    training = old.feature_paths(raw)
    transitions, transition_ledger = transition_manifest(training)
    audit_support = json.loads((pinned.parent/'request328-training-support.json').read_text())
    g, priors, prefix_audit = previous.prefix_initial(training, prior, saved.R326_B_p_net_5.to_numpy())
    active = gate_mask(training)
    path_hashes = {'EMA_innovation': old.path_digest(training, old.EMA), 'first_anchor_innovation': old.path_digest(training, old.ANCHOR)}
    if path_hashes != audit_support['training_path_sha256'] or hashlib.sha256(active.astype('uint8').tobytes()).hexdigest() != audit_support['gate_sha256_uint8'] or hashlib.sha256(np.ascontiguousarray(g, dtype='<f8').tobytes()).hexdigest() != audit_support['prefix_G_sha256_little_endian_float64']:
        raise ValueError('registered training path/gate/prefix G fingerprints changed')
    if len(raw) != 7806 or int(complete_mask(raw).sum()) != 7327 or len(evaluation) != 19530 or len(ledger) != 828 or len(transitions) != 7066 or len(pairs) != 8623 or set(evaluation.trading_day) != set(EVAL_DAYS):
        raise ValueError('registered original population changed')
    output.mkdir(parents=True, exist_ok=True)
    parquet(transitions, output, 'transition-manifest')
    parquet(transition_ledger, output, 'transition-ledger')
    print('R328 original scope, training paths/gate/prefix G authenticated before fitting', flush=True)
    bundles, fits = fit_models(training, transitions, g)
    fit_audit = {'full': fits, 'folds': {}}
    persist_audit(output, fit_audit)
    if any(b[0] is None for b in bundles.values()):
        raise ValueError('unsupported/failed full fit audited; no replacement evaluation')
    scored, inference = infer(evaluation, bundles, prior['fits']['G'], eval_saved.R326_B_p_net_5.to_numpy())
    verify_checkpoint(evaluation, scored[list(evaluation)])
    if any(inference['unscored_observations'].values()):
        raise ValueError('full fitted scores must be finite')
    chrono, folds = chronological(training, saved, prior, output, fit_audit)
    verify_checkpoint(raw, chrono[list(raw)])
    enriched, _ = infer_with_initial(training, bundles, g, priors)
    enriched['R328_original_target_weight'] = previous.target_weights(training)
    first = scored.loc[previous.history(scored).first].reset_index(drop=True)
    if not np.all(first[[f'R328_{a}_p_net_5' for a in ARMS]].to_numpy() == first.R328_B_p_net_5.to_numpy()[:, None]):
        raise ValueError('exact common first changed')
    feature_history, design_matrix = tables(enriched, scored, chrono)
    for suffix, frame in (('training-states', enriched), ('states', scored), ('chronological-states', chrono), ('first-states', first), ('episode-ledger', ledger), ('feature-history', feature_history), ('design-matrix', design_matrix)):
        parquet(frame, output, suffix)
    integrity = {'pinned_hashes_verified': True, 'path_gate_prefix_G_fingerprints_verified': True,
        'original_inputs_preserved': True, 'gate0_and_first_G_exact': True, 'strict_prefix_G_replayed': True,
        'label_free_causal_inference': True, 'original_weights_no_gate_renormalization': True,
        'same_gate_all_four_heads': True, 'no_missing_indicator_coefficients': True, 'no_G_refit': True}
    report = {'training_support': transition_support(transitions), 'gate_training_support': support(training),
        'observed_states': len(scored), 'complete_states': int(complete_mask(scored).sum()), 'ledger_episodes': len(ledger),
        'complete_evaluation_episodes': len(scored.loc[complete_mask(scored), KEYS].drop_duplicates()),
        'arms': report_metrics(scored, pairs),
        'per_day': {d: {'arms': report_metrics(scored.loc[scored.trading_day.eq(d)], pairs.loc[pairs.trading_day.eq(d)].reset_index(drop=True)),
            'positive_episodes': len(scored.loc[scored.trading_day.eq(d) & complete_mask(scored) & truth(scored[CEILING], 5), KEYS].drop_duplicates())} for d in EVAL_DAYS},
        'paired_intervals': intervals(scored, pairs)}
    for a in ('B', 'B0'):
        for metric in METRICS:
            if abs(report['arms'][a][metric]-frozen['comparison']['arms'][a][metric]) > 1e-9:
                raise ValueError('unchanged G control metric mismatch')
    chrono_pairs, _ = local_manifest(chrono, chrono[KEYS].drop_duplicates())
    fitted = chrono.loc[chrono.trading_day.gt(CROSSFIT_DAYS[1])]
    gap = (scored.decision_t-scored.state_previous_t)/1000
    result = {'request_id': REQUEST_ID, 'development_only': True, 'promotion_eligible': False, 'market_requests': 0,
        'June_HOLD_opened': False, 'final_July_August_opened': False, 'input_sha256': pins,
        'features': {a: list(f) for a, f in FEATURES.items()}, 'tau_s': old.TAU_S, 'fits': fits,
        'prefix_G_audit': prefix_audit, 'training_path_sha256': path_hashes, 'inference': inference,
        'comparison': report, 'gate_checks': gate(report, integrity), 'integrity': integrity,
        'diagnostics': diagnostics(scored, pairs),
        'long_gap_diagnostic': {n: report_metrics(scored.loc[m], pairs) for n, m in (('at_most30s', gap.le(30)), ('over30s', gap.gt(30)))},
        'chronological_training': {'observed_states': len(chrono), 'common_finite_observations': int(np.isfinite(chrono[[f'R328_{a}_p_net_5' for a in ARMS]]).all(axis=1).sum()),
            'arms_common_finite': report_metrics(chrono, chrono_pairs), 'fitted_May7_May8_only': report_metrics(fitted, chrono_pairs),
            'per_day': {d: report_metrics(chrono.loc[chrono.trading_day.eq(d)], chrono_pairs) for d in CROSSFIT_DAYS[1:]}, 'folds': folds},
        'limitations': 'Reused May development; hindsight reachable labels, not realized profit. G and current features already include history. Gate is a new hypothesis generated after R327, not established missingness causality. EMA/anchor/current have equal residual dimension but different information/scales; M has one dimension. Fixed-fit intervals omit fitting/gate-design uncertainty. No trajectory benefit from prior predicted probability is claimed. Unsupported May6 active predictions remain missing; inactive exact G is a defined architecture branch, not a fabricated fit. No gate/tau/C/optimizer search, fallback arm selection, raw acquisition, sealed dates, policy change, promotion or live trades.'}
    (output/'request328.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
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
