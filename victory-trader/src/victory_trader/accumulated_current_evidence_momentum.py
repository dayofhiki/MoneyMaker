"""R329: fixed accumulated current evidence against frozen current and anchor controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit

from . import gated_trajectory_residual_momentum as frozen
from .canonical_population_momentum import verify_checkpoint
from .compact_momentum_representation import fit_transform
from .within_episode_momentum import frozen_transform

old, previous = frozen.old, frozen.previous
KEYS, CEILING, LAG_INPUTS = frozen.KEYS, frozen.CEILING, frozen.LAG_INPUTS
complete_mask, truth = frozen.complete_mask, frozen.truth
transition_manifest, transition_support = frozen.transition_manifest, frozen.transition_support
local_manifest, LocalCache, RankCache = frozen.local_manifest, frozen.LocalCache, frozen.RankCache
EVAL_DAYS, CROSSFIT_DAYS = frozen.EVAL_DAYS, frozen.CROSSFIT_DAYS
REQUEST_ID, TAU_S = 329, 30.
NEW_ARMS = ('F', 'N')
ARMS = ('F', 'N', 'C', 'M', 'B', 'B0')
CONTRASTS = (('F', 'N'), ('F', 'C'), ('F', 'M'), ('F', 'B'), ('F', 'B0'), ('N', 'C'), ('N', 'B'), ('C', 'B'), ('M', 'B'))
METRICS = frozen.METRICS

def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()


def memory_basis(frame, transform):
    """Current evidence enters the EMA at this clock; gate0 only carries state."""
    paths = frozen.old.feature_paths(frame)
    active, clock = frozen.gate_mask(paths), frozen.previous.history(paths)
    current = frozen.design(paths, active, 'S', transform)
    times = paths.decision_t.to_numpy(float)
    ema, anchor, before = (np.full((len(frame), 7), np.nan) for _ in range(3))
    last = np.full(len(frame), np.nan)
    first_eligible = np.zeros(len(frame), bool)
    for now in clock.steps:
        past = clock.previous[now]
        before[now] = ema[past]
        ema[now], anchor[now], last[now] = ema[past], anchor[past], last[past]
        chosen = now[active[now]]
        if len(chosen):
            old = clock.previous[chosen]
            known = np.isfinite(last[old])
            z = np.exp(-(times[chosen]-last[old])/1000/30.)
            ema[chosen] = np.where(known[:, None], z[:, None]*ema[old]+(1-z[:, None])*current[chosen], current[chosen])
            anchor[chosen] = np.where(known[:, None], anchor[old], current[chosen])
            last[chosen] = times[chosen]
            first_eligible[chosen] = ~known
    accumulated, anchored = np.zeros_like(current), np.zeros_like(current)
    accumulated[active], anchored[active] = ema[active], anchor[active]
    if transform is not None and (not np.isfinite(accumulated).all() or not np.isfinite(anchored).all()):
        raise ValueError('finite gated memory coordinates required')
    np.testing.assert_array_equal(accumulated[first_eligible], current[first_eligible])
    np.testing.assert_array_equal(anchored[first_eligible], current[first_eligible])
    return active, current, accumulated, anchored, first_eligible, before, ema, last


def legacy(frame):
    return frame.rename(columns={c: c.replace('R329_', 'R327_', 1) for c in frame if c.startswith('R329_')})


def report_metrics(frame, pairs):
    return old.report_metrics(legacy(frame), pairs, ARMS)


def intervals(frame, pairs, draws=1000):
    complete = frame.loc[complete_mask(frame)]
    adapted = legacy(complete)
    cache, local = RankCache(old.probability_frame(adapted, ARMS), ARMS), LocalCache(old.raw_frame(adapted, ARMS), pairs, ARMS)
    output = {}
    for scheme, columns, seed in (('day', ['trading_day'], 20266350), ('ticker_day', ['trading_day', 'ticker'], 20266351)):
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
        print(f'R329 {scheme} bootstrap complete', flush=True)
    return output


def gate(report, integrity):
    a, s, rank = report['arms'], report['gate_training_support'], 'between_episode_same_day_auc'
    def greater(x, y):
        return x is not None and y is not None and x > y
    checks = {'at_least150_carried_training_episodes': s['complete_target_episodes'] >= 150,
        'at_least30_active_positive_training_episodes': s['positive_target_episodes'] >= 30,
        'active_both_classes_at_least3_dates': all(len(s[k]) >= 3 for k in ('positive_target_days', 'negative_target_days')),
        'at_least200_original_transform_episodes': report['training_support']['episodes'] >= 200,
        'at_least50_mixed_evaluation_episodes': a['F']['within_episode_evaluable_episodes'] >= 50,
        'at_least30_local_evaluation_episodes': a['F']['local_evaluable_episodes'] >= 30,
        'at_least4_positive_evaluation_dates': sum(v['positive_episodes'] > 0 for v in report['per_day'].values()) >= 4,
        'gate0_and_first_G_exact': integrity['gate0_and_first_G_exact'],
        'strict_prefix_G_replayed': integrity['strict_prefix_G_replayed']}
    checks |= {'F_admission_beats_N_C_M_B_B0': all(greater(a['F'][rank], a[x][rank]) for x in ('N', 'C', 'M', 'B', 'B0')),
        'F_whole_beats_N_C_M_B_and_half': all(greater(a['F']['within_episode_auc'], v) for v in [a[x]['within_episode_auc'] for x in ('N', 'C', 'M', 'B')]+[.5]),
        'F_local_beats_N_C_M_B_and_half': all(greater(a['F']['local_pair_rank'], v) for v in [a[x]['local_pair_rank'] for x in ('N', 'C', 'M', 'B')]+[.5]),
        'F_brier_beats_N_C_M_B_B0_prior': all(greater(v, a['F']['brier']) for v in [a[x]['brier'] for x in ('N', 'C', 'M', 'B', 'B0')]+[a['F']['prior_brier']])}
    days = [v['arms'] for v in report['per_day'].values() if v['arms']['B'][rank] is not None]
    checks['F_B_admission_improves_majority_dates'] = bool(days) and sum(greater(v['F'][rank], v['B'][rank]) for v in days) > len(days)/2
    requirements = [('F_minus_B', m, True) for m in (rank, 'within_episode_auc', 'local_pair_rank')]+[('F_minus_B', 'brier', False)]
    requirements += [(f'F_minus_{control}', m, True) for control in ('N', 'C', 'M') for m in ('within_episode_auc', 'local_pair_rank', rank)]
    for contrast, metric, positive in requirements:
        for scheme in ('day', 'ticker_day'):
            ci = report['paired_intervals'][scheme]['comparisons'][contrast][f'{metric}_ci95']
            checks[f"{scheme}_{contrast}_{metric}_ci_{'positive' if positive else 'negative'}"] = ci is not None and (ci[0] > 0 if positive else ci[1] < 0)
    if len(checks) != 40:
        raise ValueError('registered40 checks changed')
    return checks


def frozen_preprocessing(training, fit):
    if training.empty:
        return None
    if fit['fit_days'] != sorted(training.trading_day.unique()):
        raise ValueError('frozen current transform scope changed')
    transform = frozen_transform(fit['preprocessing'])
    manifest, _ = transition_manifest(training)
    replay = fit_transform(training.loc[manifest.current_index], LAG_INPUTS, manifest.transform_weight.to_numpy()).audit()
    for key in ('lower', 'upper', 'median', 'mean', 'scale'):
        np.testing.assert_allclose(transform.audit()[key], replay[key], atol=1e-9, rtol=0)
    if tuple(transform.features) != tuple(LAG_INPUTS):
        raise ValueError('frozen numeric schema changed')
    return transform


def path_hashes(basis):
    _, current, ema, anchor, _, before, after, last = basis
    return {k: digest(v) for k, v in {'current': current, 'EMA_evidence': ema,
        'first_eligible_anchor': anchor, 'EMA_before': before, 'EMA_after': after, 'last_eligible_t': last}.items()}


def fit_models(training, current_g, transform, registered_scope):
    if training.empty:
        return {a: None for a in NEW_ARMS}, {a: {'no_support': True, 'optimization_attempted': False,
            'reason': 'no_preceding_out_of_time_meta_day'} for a in NEW_ARMS}
    if not set(training.trading_day).issubset(CROSSFIT_DAYS[1:]):
        raise ValueError('fixed out-of-time meta dates required')
    basis = memory_basis(training, transform)
    if registered_scope is not None and path_hashes(basis) != registered_scope['path_sha256']:
        raise ValueError('registered pre-fit memory fingerprints changed')
    active, _, ema, anchor = basis[:4]
    offset = frozen.baseline_logits(current_g, len(training))
    weights, labels = previous.target_weights(training), truth(training[CEILING], 5)
    models, audits = {}, {}
    for arm, x in (('F', ema), ('N', anchor)):
        beta, audit = frozen.solve(x, offset, labels, weights, active)
        models[arm] = beta
        audits[arm] = audit | {'fit_days': sorted(training.trading_day.unique()),
            'preprocessing': transform.audit(), 'frozen_R328_S_transform': True,
            'numeric_columns_only': True, 'intercept': False, 'tau_s': TAU_S,
            'training_path_sha256': path_hashes(basis)}
        print(f"R329 {arm}: success={beta is not None}, iterations={audit.get('iterations')}", flush=True)
    return models, audits


def infer(states, models, transform, controls):
    identities = [*KEYS, 'decision_t']
    verify_checkpoint(states[identities], controls[identities])
    frame = old.feature_paths(states)
    basis = memory_basis(frame, transform)
    active, current, ema, anchor, first, before, after, last = basis
    g = controls.R328_B_p_net_5.to_numpy()
    offset = frozen.baseline_logits(g, len(frame))
    if not np.array_equal(active, controls.R328_gate.to_numpy()):
        raise ValueError('frozen gate mismatch')
    if transform is not None:
        np.testing.assert_array_equal(current, controls[[f'R328_S_design_{j}' for j in range(7)]].to_numpy())
    columns = {'R329_gate': active, 'R329_G_offset_logit': offset,
        'R329_first_eligible': first, 'R329_last_eligible_t': last}
    for name, values in (('current', current), ('F_design', ema), ('N_design', anchor), ('evidence_before', before), ('evidence_after', after)):
        for j in range(7):
            columns[f'R329_{name}_{j}'] = values[:, j]
    for arm in ARMS:
        if arm in NEW_ARMS:
            beta, x = models[arm], ema if arm == 'F' else anchor
            correction = np.zeros(len(frame))
            correction[active] = np.nan if beta is None else x[active]@beta
            eta = offset+correction
            p = expit(eta)
            copied = ~active | (correction == 0)
            p[copied] = g[copied]
            with np.errstate(invalid='ignore'):
                lp, lc = -np.logaddexp(0., -eta), -np.logaddexp(0., eta)
            lp[copied], lc[copied] = np.log(g[copied]), np.log1p(-g[copied])
            columns[f'R329_{arm}_correction'] = correction
            values = {'p_net_5': p, 'logit': eta, 'log_p_net_5': lp, 'log_1_minus_p': lc,
                'prior_net_5': controls.R328_B_prior_net_5.to_numpy()}
        else:
            source = 'S' if arm == 'C' else arm
            suffixes = ('p_net_5', 'prior_net_5', 'logit', 'log_p_net_5', 'log_1_minus_p')
            if arm in ('C', 'M'):
                suffixes += ('correction',)
            values = {suffix: controls[f'R328_{source}_{suffix}'].to_numpy(copy=True) for suffix in suffixes}
        for suffix, value in values.items():
            columns[f'R329_{arm}_{suffix}'] = value
        if not np.array_equal(values['p_net_5'][~active], g[~active]):
            # Own-first B0 is only required to agree at original first clocks.
            if arm != 'B0':
                raise ValueError('gate0 exact G copy changed')
    result = pd.concat([frame, pd.DataFrame(columns, index=frame.index)], axis=1)
    return result, {'unscored_observations': {a: int(result[f'R329_{a}_p_net_5'].isna().sum()) for a in ARMS},
        'active_observations': int(active.sum()), 'first_eligible_observations': int(first.sum()),
        'first_eligible_basis_exact': True, 'frozen_control_paths_exact': True}


def parquet(frame, output, suffix):
    path = output/f'request329-{suffix}.parquet'
    temporary = path.with_suffix('.parquet.tmp')
    frame.to_parquet(temporary, index=False, compression='zstd', compression_level=19)
    metadata = pq.ParquetFile(temporary).metadata
    if metadata.num_rows != len(frame) or metadata.num_columns != len(frame.columns):
        raise ValueError('complete parquet footer required before atomic replacement')
    os.replace(temporary, path)


def persist_audit(output, audit):
    (output/'request329-fit-audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')


def chronological(raw, saved, prior, frozen_audit, control, scopes, output, audit):
    pieces, folds = [], {}
    for day in CROSSFIT_DAYS[1:]:
        preceding = raw.loc[raw.trading_day.lt(day)]
        transform = frozen_preprocessing(preceding, frozen_audit['folds'][day]['fits']['S'])
        g = previous.prefix_initial(preceding, prior, saved.loc[preceding.index, 'R326_B_p_net_5'].to_numpy())[0] if len(preceding) else np.array([])
        models, fits = fit_models(preceding, g, transform, scopes[day])
        folds[day] = {'fit_days': sorted(preceding.trading_day.unique()), 'fits': fits}
        audit['folds'][day] = folds[day]
        persist_audit(output, audit)
        if any(models[a] is None and not fits[a]['no_support'] for a in NEW_ARMS):
            raise ValueError('fold numerical failure audited; no replacement evaluation')
        held = raw.loc[raw.trading_day.eq(day)]
        scored, inference = infer(held, models, transform, control.loc[held.index])
        folds[day]['inference'] = inference
        pieces.append(scored)
    persist_audit(output, audit)
    return pd.concat(pieces).sort_index(), folds


def evidence_history(training, evaluation, chrono):
    pieces = []
    for scope, frame in (('training', training), ('evaluation', evaluation), ('chronological', chrono)):
        columns = [*KEYS, 'decision_t', 'observation_available', 'label_complete', CEILING,
            *[c for c in frame if c.startswith('R329_') and ('design_' in c or 'current_' in c or 'evidence_' in c or c in ('R329_gate', 'R329_G_offset_logit', 'R329_first_eligible', 'R329_last_eligible_t'))]]
        selected = frame[columns].copy()
        selected.insert(0, 'source_index', frame.index.to_numpy())
        selected.insert(0, 'scope', scope)
        selected['original_complete_target_weight'] = previous.target_weights(frame)
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


def diagnostics(frame, pairs):
    active, first = frame.R329_gate.to_numpy(), frame.R329_first_eligible.to_numpy()
    complete, weights = complete_mask(frame).to_numpy(), previous.target_weights(frame)
    slices = {}
    for name, mask in (('eligible', active), ('inactive', ~active), ('first_eligible', first), ('carried', active & ~first)):
        slices[name] = {'observations': int(mask.sum()), 'complete_targets': int((mask & complete).sum()),
            'original_complete_weight_mass': float(weights[mask].sum()), 'arms': report_metrics(frame.loc[mask], pairs)}
    # Pair identities are joined to original source indices; no hindsight resampling.
    both = pairs.positive_index.isin(frame.index[active]) & pairs.negative_index.isin(frame.index[active])
    return {'availability_slices': slices, 'both_endpoints_eligible_pairs': int(both.sum()),
        'both_endpoints_eligible_timing': report_metrics(frame, pairs.loc[both].reset_index(drop=True)),
        'slices_are_descriptive_not_validation': True}


def run(previous_dir, pinned, output):
    pins = json.loads(pinned.read_text())
    for name, sha in pins.items():
        if hashlib.sha256((previous_dir/name).read_bytes()).hexdigest() != sha:
            raise ValueError('registered input changed: '+name)
    if (previous_dir/'official328/source-sha.txt').read_text().strip() != 'aa031411f1ab790c2c10d5f0a92dc9aa9225aead':
        raise ValueError('verified R328 source changed')
    def read(name):
        return pd.read_parquet(previous_dir/name)
    raw = read('official321/request321-chronological-states.parquet').reset_index(drop=True)
    evaluation = read('official321/request321-states.parquet')
    saved = read('official326/request326-training-states.parquet')
    controls = {scope: read(f'official328/request328-{suffix}.parquet') for scope, suffix in
        (('training', 'training-states'), ('evaluation', 'states'), ('chronological', 'chronological-states'))}
    ledger = read('official319/request319-episode-ledger.parquet')
    clocks = read('official319/request319-clock-manifest.parquet')
    pairs = read('official321/request321-local-pair-manifest.parquet')
    pair_ledger = read('official321/request321-local-pair-ledger.parquet')
    for frame, source in ((raw, saved), (raw, controls['training']), (raw, controls['chronological']), (evaluation, controls['evaluation'])):
        verify_checkpoint(frame, source[list(frame)])
    verify_checkpoint(clocks, evaluation[list(clocks)])
    replay_pairs, replay_ledger = local_manifest(evaluation, ledger)
    verify_checkpoint(pairs, replay_pairs)
    verify_checkpoint(pair_ledger, replay_ledger)
    transitions, transition_ledger = transition_manifest(raw)
    if len(raw) != 7806 or int(complete_mask(raw).sum()) != 7327 or len(evaluation) != 19530 or len(ledger) != 828 or len(transitions) != 7066 or len(pairs) != 8623 or set(evaluation.trading_day) != set(EVAL_DAYS):
        raise ValueError('registered original population changed')
    prior = json.loads((previous_dir/'official319/request319.json').read_text())
    frozen_audit = json.loads((previous_dir/'official328/request328-fit-audit.json').read_text())
    scopes = json.loads((pinned.parent/'request329-training-support.json').read_text())['scopes']
    g, priors, prefix_audit = previous.prefix_initial(raw, prior, saved.R326_B_p_net_5.to_numpy())
    np.testing.assert_array_equal(g, controls['training'].R328_B_p_net_5.to_numpy())
    # Authenticate every full/fold basis before the first new real fit.
    for scope, frame, fit in [('full', raw, frozen_audit['full']['S'])]+[(d, raw.loc[raw.trading_day.lt(d)], item['fits']['S']) for d, item in frozen_audit['folds'].items()]:
        transform = frozen_preprocessing(frame, fit)
        if len(frame) and path_hashes(memory_basis(frame, transform)) != scopes[scope]['path_sha256']:
            raise ValueError('registered full/fold basis changed before fitting')
    output.mkdir(parents=True, exist_ok=True)
    parquet(transitions, output, 'transition-manifest')
    parquet(transition_ledger, output, 'transition-ledger')
    print('R329 original scope, immutable controls/preprocessing and all training path hashes authenticated before fitting', flush=True)
    transform = frozen_preprocessing(raw, frozen_audit['full']['S'])
    models, fits = fit_models(raw, g, transform, scopes['full'])
    audit = {'full': fits, 'folds': {}}
    persist_audit(output, audit)
    if any(beta is None for beta in models.values()):
        raise ValueError('unsupported/failed full fit audited; no replacement evaluation')
    scored, inference = infer(evaluation, models, transform, controls['evaluation'])
    chrono, folds = chronological(raw, saved, prior, frozen_audit, controls['chronological'], scopes, output, audit)
    enriched, _ = infer(raw, models, transform, controls['training'])
    enriched['R329_original_target_weight'] = previous.target_weights(raw)
    first = scored.loc[previous.history(scored).first].reset_index(drop=True)
    np.testing.assert_array_equal(first[[f'R329_{a}_p_net_5' for a in ARMS]].to_numpy(), np.repeat(first.R329_B_p_net_5.to_numpy()[:, None], len(ARMS), axis=1))
    for suffix, frame in (('training-states', enriched), ('states', scored), ('chronological-states', chrono), ('first-states', first), ('episode-ledger', ledger), ('evidence-history', evidence_history(enriched, scored, chrono))):
        parquet(frame, output, suffix)
    integrity = {'pinned_hashes_verified': True, 'all_registered_full_fold_paths_verified_before_fitting': True,
        'original_inputs_preserved': True, 'gate0_and_first_G_exact': True, 'strict_prefix_G_replayed': True,
        'label_free_causal_inference': True, 'original_weights_no_gate_renormalization': True,
        'equal_first_eligible_bases': True, 'immutable_R328_preprocessing_and_controls': True, 'atomic_parquet_footers_verified': True}
    report = {'training_support': transition_support(transitions), 'gate_training_support': scopes['full']['carried_history'],
        'observed_states': len(scored), 'complete_states': int(complete_mask(scored).sum()), 'ledger_episodes': len(ledger),
        'complete_evaluation_episodes': len(scored.loc[complete_mask(scored), KEYS].drop_duplicates()),
        'arms': report_metrics(scored, pairs),
        'per_day': {d: {'arms': report_metrics(scored.loc[scored.trading_day.eq(d)], pairs.loc[pairs.trading_day.eq(d)].reset_index(drop=True)),
            'positive_episodes': len(scored.loc[scored.trading_day.eq(d) & complete_mask(scored) & truth(scored[CEILING], 5), KEYS].drop_duplicates())} for d in EVAL_DAYS},
        'paired_intervals': intervals(scored, pairs)}
    frozen_result = json.loads((previous_dir/'official328/request328.json').read_text())
    for a, b in (('C', 'S'), ('M', 'M'), ('B', 'B'), ('B0', 'B0')):
        for metric in METRICS:
            if abs(report['arms'][a][metric]-frozen_result['comparison']['arms'][b][metric]) > 1e-9:
                raise ValueError('pinned control metrics changed')
    chrono_pairs, _ = local_manifest(chrono, chrono[KEYS].drop_duplicates())
    fitted = chrono.loc[chrono.trading_day.gt(CROSSFIT_DAYS[1])]
    gap = (scored.decision_t-scored.state_previous_t)/1000
    result = {'request_id': REQUEST_ID, 'development_only': True, 'promotion_eligible': False, 'market_requests': 0,
        'June_HOLD_opened': False, 'final_July_August_opened': False, 'input_sha256': pins,
        'tau_s': TAU_S, 'fits': fits, 'prefix_G_audit': prefix_audit, 'inference': inference,
        'comparison': report, 'gate_checks': gate(report, integrity), 'integrity': integrity,
        'diagnostics': diagnostics(scored, pairs),
        'long_gap_diagnostic': {n: report_metrics(scored.loc[m], pairs) for n, m in (('at_most30s', gap.le(30)), ('over30s', gap.gt(30)))},
        'chronological_training': {'observed_states': len(chrono), 'common_finite_observations': int(np.isfinite(chrono[[f'R329_{a}_p_net_5' for a in ARMS]]).all(axis=1).sum()),
            'arms_common_finite': report_metrics(chrono, chrono_pairs), 'fitted_May7_May8_only': report_metrics(fitted, chrono_pairs),
            'per_day': {d: report_metrics(chrono.loc[chrono.trading_day.eq(d)], chrono_pairs) for d in CROSSFIT_DAYS[1:]}, 'folds': folds},
        'limitations': 'Reused May development; hindsight reachability is not realized P&L. Frozen G/current features already include history. Fixed-fit intervals omit model/design uncertainty. No tau/gate/C search, fallback arm selection, sealed dates, policy promotion or live trades. R329 failure binds the next priority to current-state continuous ENTER/WAIT policy; event specialist stays independent.'}
    (output/'request329.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
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

