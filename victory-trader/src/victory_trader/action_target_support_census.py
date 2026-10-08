"""R331: preregistered clock-only census followed by fixed ENTER observability.

No learner or action selector is fitted. Run clocks, publish their authenticated
manifest, then run outcomes under the unchanged R330 economic contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import current_state_action_policy as contract
from .frozen_entry_second_hold_exit import HARD_STOP_PCT, KEYS, net
from .pending_exit_integrity import session_limits

REQUEST_ID, BUCKET_MS = 331, 30000
TRAIN_DAYS = ('2026-05-05', '2026-05-06', '2026-05-07', '2026-05-08')
PRIMARY_DAYS = TRAIN_DAYS[1:]
EVAL_DAYS = tuple(contract.frozen.EVAL_DAYS)
ARMS = ('EARLY', 'ADDITIONAL_LATE', 'EXTENDED')
CLOCK_KEYS = [*KEYS, 'decision_t']
FLAGS = ('observed', 'any_complete_compatible', 'any_positive_compatible', 'any_nonpositive_compatible')
CLOCK_FILES = tuple(f'request331-{s}-{k}.parquet' for s in ('training', 'evaluation') for k in ('clocks', 'clock-ledger'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.parquet.tmp')
    frame.to_parquet(temporary, index=False, compression='zstd', compression_level=19)
    metadata = pq.ParquetFile(temporary).metadata
    if metadata.num_rows != len(frame) or metadata.num_columns != len(frame.columns):
        raise ValueError('complete readable parquet footer required')
    os.replace(temporary, path)


def write_json(value, path):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def load_inputs(previous, pinned):
    pins = json.loads(pinned.read_text())
    for name, sha in pins.items():
        if digest(previous/name) != sha:
            raise ValueError('registered input changed: '+name)
    sources = {311: 'a6034adba43fbd02334834f68341a64951957ba6',
        315: 'c271fbab9c3567c470fc852ef80c39fe36116713'}
    # R319 is authenticated by its pinned source file; retain its full hash verbatim.
    for n in (311, 315):
        if (previous/f'official{n}/source-sha.txt').read_text().strip() != sources[n]:
            raise ValueError('raw cache source changed')
    if (previous/'official330/source-sha.txt').read_text().strip() != '29b60b70ec02e68292f19c7d9c0af80017da1cc5':
        raise ValueError('canonical R330 source changed')
    def read(name):
        return pd.read_parquet(previous/name)
    cohort = read('official315/request315-training-cohort.parquet')
    training = cohort.loc[cohort.in_broad, KEYS].sort_values(KEYS).reset_index(drop=True)
    evaluation = read('official319/request319-episode-ledger.parquet')[KEYS].sort_values(KEYS).reset_index(drop=True)
    expected = dict(zip(TRAIN_DAYS, (102, 108, 97, 100)))
    if training.groupby('trading_day').size().to_dict() != expected or training.duplicated(KEYS).any():
        raise ValueError('407 original broad training identities required')
    if len(evaluation) != 828 or set(evaluation.trading_day) != set(EVAL_DAYS) or evaluation.duplicated(KEYS).any():
        raise ValueError('828 original evaluation identities required')
    raw = {'training': read('official315/request315-training-raw-seconds.parquet'),
        'evaluation': read('official311/request311-raw-seconds.parquet')}
    for scope, days in (('training', TRAIN_DAYS), ('evaluation', EVAL_DAYS)):
        if not set(raw[scope].trading_day).issubset(days) or raw[scope].duplicated(['trading_day', 'ticker', 't']).any():
            raise ValueError('unique cached allowed-date seconds required')
    return {'training': training, 'evaluation': evaluation}, raw, pins


def clock_manifest(identities, timestamps):
    """Only identifiers/timestamps enter clock choice; prices are inaccessible."""
    if identities.duplicated(KEYS).any() or timestamps.duplicated(['trading_day', 'ticker', 't']).any():
        raise ValueError('duplicate identities or second timestamps')
    times = {k: np.sort(g.t.to_numpy(np.int64)) for k, g in timestamps.groupby(['trading_day', 'ticker'], sort=False)}
    rows, ledger = [], []
    for key in identities[KEYS].sort_values(KEYS).itertuples(index=False, name=None):
        day, ticker, hot = key
        cap = min(int(hot)+3600000, session_limits(day)[1])
        starts = times.get((day, ticker), np.array([], np.int64))
        ends = starts+1000
        possible = ends[(starts >= hot) & (ends < cap)]
        early = possible[possible <= hot+contract.WATCH_MS]
        candidates = possible[possible > hot+contract.WATCH_MS]
        buckets = (candidates-hot-contract.WATCH_MS-1)//BUCKET_MS
        first = np.r_[True, np.diff(buckets) != 0] if len(candidates) else np.array([], bool)
        late = candidates[first]
        identity = dict(zip(KEYS, key))
        for phase, clocks in (('EARLY', early), ('ADDITIONAL_LATE', late)):
            rows.extend({**identity, 'decision_t': int(t), 'phase': phase} for t in clocks)
        ledger.append({**identity, 'terminal_t': int(cap), 'early_clocks': len(early),
            'late_clocks': len(late), 'extended_clocks': len(early)+len(late),
            'raw_pair_available': (day, ticker) in times,
            'no_observation': len(early)+len(late) == 0})
    return pd.DataFrame(rows, columns=[*CLOCK_KEYS, 'phase']).sort_values(CLOCK_KEYS).reset_index(drop=True), pd.DataFrame(ledger)


def verify_early(clocks, inherited):
    old = inherited[CLOCK_KEYS].sort_values(CLOCK_KEYS).reset_index(drop=True)
    early = clocks.loc[clocks.phase.eq('EARLY'), CLOCK_KEYS].sort_values(CLOCK_KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(early, old, check_dtype=False, check_exact=True)


def save_clocks(previous, pinned, output):
    identities, raw, pins = load_inputs(previous, pinned)
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in CLOCK_FILES):
        raise ValueError('refuse to overwrite a registered clock manifest')
    summaries = {}
    for scope in identities:
        clocks, ledger = clock_manifest(identities[scope], raw[scope][['trading_day', 'ticker', 't']])
        inherited = pd.read_parquet(previous/('official319/request319-training-states.parquet' if scope == 'training' else 'official319/request319-clock-manifest.parquet'))
        verify_early(clocks, inherited)
        atomic_parquet(clocks, output/f'request331-{scope}-clocks.parquet')
        atomic_parquet(ledger, output/f'request331-{scope}-clock-ledger.parquet')
        summaries[scope] = {'identities': len(ledger), 'early_clocks': int(ledger.early_clocks.sum()),
            'additional_late_clocks': int(ledger.late_clocks.sum()), 'no_observation_identities': int(ledger.no_observation.sum())}
    registration = {'request_id': REQUEST_ID, 'stage': 'timestamps_only_before_prices_and_targets',
        'input_sha256': pins, 'clock_sha256': {name: digest(output/name) for name in CLOCK_FILES}, 'scope': summaries}
    write_json(registration, output/'request331-clock-registration.json')
    return registration


def known_costs(clocks, cache):
    costs = []
    for state in clocks.itertuples():
        bars = cache[(state.trading_day, state.ticker)]
        right = int(np.searchsorted(bars.t.to_numpy(np.int64)+1000, state.decision_t, side='right'))
        if not right:
            raise ValueError('a completed causal price is required by clock membership')
        price = float(bars.c.iloc[right-1])
        costs.append(-net(price, price))
    return np.asarray(costs)


def label_clocks(clocks, raw, days):
    cache = contract.contexts(raw, days)
    # Known cost is computed and frozen before any future execution/outcome label.
    result = clocks.assign(known_cost_drag_pct=known_costs(clocks, cache))
    outcomes = []
    for state in clocks.itertuples():
        bars = cache[(state.trading_day, state.ticker)]
        out = contract.enter_outcome(bars, int(state.decision_t), int(state.hot_t), session_limits(state.trading_day)[1])
        times = bars.t.to_numpy(np.int64)
        entry = np.searchsorted(times, state.decision_t)
        out['next_entry_reference_gap_s'] = (float(times[entry])-state.decision_t)/1000 if entry < len(times) else np.nan
        exit_t = out['exit_submission_t']
        exit_index = np.searchsorted(times, exit_t) if np.isfinite(exit_t) else len(times)
        out['next_exit_reference_gap_s'] = (float(times[exit_index])-exit_t)/1000 if exit_index < len(times) else np.nan
        outcomes.append(out)
    result = pd.concat([result, pd.DataFrame(outcomes, index=result.index)], axis=1)
    result['cost_compatible'] = np.isfinite(result.known_cost_drag_pct) & result.known_cost_drag_pct.lt(-HARD_STOP_PCT)
    return result


def verify_r330(states, saved):
    early = states.loc[states.phase.eq('EARLY') & states.trading_day.isin(saved.trading_day.unique())]
    columns = [*CLOCK_KEYS, 'known_cost_drag_pct', 'enter_resolved', 'enter_reason',
        'entry_fill_t', 'entry_reference_price', 'entry_lag_s', 'exit_submission_t',
        'exit_fill_t', 'exit_reference_price', 'exit_lag_s', 'stop_triggered', 'cost_only_stop',
        *[f'enter_{s}_pct' for s in contract.SCENARIOS]]
    x, y = (frame[columns].sort_values(CLOCK_KEYS).reset_index(drop=True) for frame in (early, saved))
    if not x.isna().equals(y.isna()):
        raise ValueError('canonical early missingness changed')
    pd.testing.assert_frame_equal(x, y, check_dtype=False, atol=1e-9, rtol=0)


def choose_arm(states, arm):
    return states if arm == 'EXTENDED' else states.loc[states.phase.eq(arm)]


def identity_support(states, identities):
    ledger = identities[KEYS].copy()
    for arm in ARMS:
        frame = choose_arm(states, arm)
        valid = frame.enter_resolved & frame.cost_compatible
        positive, nonpositive = valid & frame.enter_base_pct.gt(0), valid & frame.enter_base_pct.le(0)
        aggregate = frame.assign(complete_compatible=valid, positive_compatible=positive, nonpositive_compatible=nonpositive).groupby(KEYS, sort=False).agg(
            rows=('decision_t', 'size'), cost_compatible_rows=('cost_compatible', 'sum'),
            complete_rows=('enter_resolved', 'sum'), complete_compatible_rows=('complete_compatible', 'sum'),
            positive_compatible_rows=('positive_compatible', 'sum'), nonpositive_compatible_rows=('nonpositive_compatible', 'sum'))
        for name in aggregate.columns:
            joined = ledger[KEYS].merge(aggregate[name].rename('value'), on=KEYS, how='left', validate='one_to_one')
            ledger[f'{arm}_{name}'] = joined.value.fillna(0).to_numpy(np.int64)
        ledger[f'{arm}_observed'] = ledger[f'{arm}_rows'].gt(0)
        for flag, count in zip(FLAGS[1:], ('complete_compatible_rows', 'positive_compatible_rows', 'nonpositive_compatible_rows')):
            ledger[f'{arm}_{flag}'] = ledger[f'{arm}_{count}'].gt(0)
        for label, mask in (('complete', frame.enter_resolved),
                            ('positive', frame.enter_resolved & frame.enter_base_pct.gt(0)),
                            ('nonpositive', frame.enter_resolved & frame.enter_base_pct.le(0))):
            keys = frame.loc[mask, KEYS].drop_duplicates().assign(supported=True)
            joined = ledger[KEYS].merge(keys, on=KEYS, how='left', validate='one_to_one')
            ledger[f'{arm}_any_{label}'] = joined.supported.fillna(False).to_numpy(bool)
        ledger[f'{arm}_compatible_resolution_with_original_weights'] = np.divide(
            ledger[f'{arm}_complete_compatible_rows'], ledger[f'{arm}_rows'],
            out=np.zeros(len(ledger)), where=ledger[f'{arm}_rows'].gt(0))
        ledger[f'{arm}_compatible_observation_mass'] = np.divide(
            ledger[f'{arm}_cost_compatible_rows'], ledger[f'{arm}_rows'],
            out=np.zeros(len(ledger)), where=ledger[f'{arm}_rows'].gt(0))
    for flag in FLAGS[1:]:
        early, union = (ledger[f'{arm}_{flag}'] for arm in ('EARLY', 'EXTENDED'))
        ledger[f'{flag}_change'] = np.select([~early & union, early & ~union], ['gained', 'lost'], default='unchanged')
        if (early & ~union).any():
            raise ValueError('union lost original early support')
    return ledger


def distribution(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    return {'known_count': len(finite), 'unknown_count': len(values)-len(finite),
        'full_mean': float(values.mean()) if len(values) and np.isfinite(values).all() else None,
        'conditional_known_mean': float(finite.mean()) if len(finite) else None,
        'conditional_known_quantiles': {str(q): float(v) for q, v in zip((0., .1, .5, .9, 1.), np.quantile(finite, (0., .1, .5, .9, 1.)))} if len(finite) else {}}


def support_metrics(states, identities, arm):
    frame = choose_arm(states, arm)
    valid = frame.enter_resolved.to_numpy(bool)
    compatible = frame.cost_compatible.to_numpy(bool)
    weights = contract.original_weights(frame) if len(frame) else np.array([])
    def episodes(mask):
        return len(frame.loc[mask, KEYS].drop_duplicates())
    def rate(n, d):
        return float(n/d) if d else None
    result = {'identities': len(identities), 'observed_clocks': len(frame), 'observed_episodes': episodes(np.ones(len(frame), bool)),
        'resolved_clocks': int(valid.sum()), 'resolved_episodes': episodes(valid),
        'row_resolution': rate(valid.sum(), len(frame)), 'causal_cost_compatible_clocks': int(compatible.sum()),
        'complete_compatible_clocks': int((valid & compatible).sum()), 'complete_compatible_episodes': episodes(valid & compatible),
        'compatible_row_resolution': rate((valid & compatible).sum(), compatible.sum()),
        'original_observed_episode_mass': float(weights.sum()), 'complete_original_weight_mass': float(weights[valid].sum()),
        'compatible_complete_original_weight_mass': float(weights[valid & compatible].sum()),
        'reason_counts': frame.enter_reason.value_counts().to_dict(), 'stop_triggered_clocks': int(frame.stop_triggered.sum()),
        'cost_only_stop_clocks': int(frame.cost_only_stop.sum()),
        'entry_fill_delay_s': distribution(frame.entry_lag_s), 'exit_fill_delay_s': distribution(frame.exit_lag_s),
        'next_entry_reference_gap_s': distribution(frame.next_entry_reference_gap_s),
        'next_exit_reference_gap_s': distribution(frame.next_exit_reference_gap_s), 'scenario_payoffs_pct': {},
        'phase_contribution_complete_weight_mass': {phase: float(weights[valid & frame.phase.eq(phase).to_numpy()].sum()) for phase in ARMS[:2]}}
    for name, sign in (('positive', frame.enter_base_pct.gt(0)), ('nonpositive', frame.enter_base_pct.le(0))):
        result[f'{name}_episodes'] = episodes(sign.to_numpy() & valid)
        result[f'{name}_compatible_episodes'] = episodes(sign.to_numpy() & valid & compatible)
        result[f'{name}_compatible_dates'] = sorted(frame.loc[sign & frame.enter_resolved & frame.cost_compatible, 'trading_day'].unique())
    for scenario in contract.SCENARIOS:
        values = frame[f'enter_{scenario}_pct'].to_numpy(float)
        result['scenario_payoffs_pct'][scenario] = {**distribution(values),
            'conditional_original_weighted_mean': float(np.average(values[valid], weights=weights[valid])) if valid.any() else None,
            'conditional_compatible_original_weighted_mean': float(np.average(values[valid & compatible], weights=weights[valid & compatible])) if (valid & compatible).any() else None}
    return result


def clustered_support(ledger, draws=1000):
    """All original identities remain, including unobserved/censored episodes."""
    days = ledger.trading_day.value_counts()
    weights = np.array([1/(len(days)*days[d]) for d in ledger.trading_day])
    result = {}
    for scheme, columns, seed in (('day', ['trading_day'], 20266460), ('ticker_day', ['trading_day', 'ticker'], 20266461)):
        groups = list(ledger.groupby(columns, sort=False).indices.values())
        ids = np.zeros(len(ledger), int)
        for j, group in enumerate(groups):
            ids[group] = j
        rng = np.random.default_rng(seed)
        samples = {}
        for _ in range(draws):
            counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            w = weights*counts[ids]
            for arm in ARMS:
                for flag in FLAGS:
                    samples.setdefault(f'{arm}_{flag}_rate', []).append(float(np.average(ledger[f'{arm}_{flag}'], weights=w)))
                denominator = float(np.sum(w*ledger[f'{arm}_compatible_observation_mass']))
                if denominator:
                    samples.setdefault(f'{arm}_compatible_row_resolution', []).append(float(np.sum(w*ledger[f'{arm}_compatible_resolution_with_original_weights'])/denominator))
            for flag in FLAGS:
                delta = ledger[f'EXTENDED_{flag}'].astype(float)-ledger[f'EARLY_{flag}'].astype(float)
                samples.setdefault(f'EXTENDED_minus_EARLY_{flag}_rate', []).append(float(np.average(delta, weights=w)))
        result[scheme] = {'clusters': len(groups), 'draws': draws,
            'metrics': {name: {'ci95': np.quantile(values, (.025, .975)).tolist(), 'valid_draws': len(values)} for name, values in samples.items()}}
    return result


def gates(primary, integrity):
    m = primary['arms']['EXTENDED']
    checks = {name: bool(value) for name, value in integrity.items()}
    checks.update(training_complete_compatible_episodes150=m['complete_compatible_episodes'] >= 150,
        training_positive_compatible_episodes30=m['positive_compatible_episodes'] >= 30,
        training_both_compatible_signs3_dates=all(set(m[f'{sign}_compatible_dates']) == set(PRIMARY_DAYS) for sign in ('positive', 'nonpositive')),
        training_compatible_row_resolution90=m['compatible_row_resolution'] is not None and m['compatible_row_resolution'] >= .9)
    if len(checks) != 10:
        raise ValueError('registered10 gates changed')
    return checks


def census(previous, pinned, output, clock_registration):
    identities, raw, pins = load_inputs(previous, pinned)
    registration = json.loads(clock_registration.read_text())
    if registration['input_sha256'] != pins or registration['stage'] != 'timestamps_only_before_prices_and_targets':
        raise ValueError('clock registration inputs/stage changed')
    if set(registration['clock_sha256']) != set(CLOCK_FILES):
        raise ValueError('complete clock registration required')
    for name, sha in registration['clock_sha256'].items():
        if digest(output/name) != sha:
            raise ValueError('registered clock manifest changed: '+name)
    frames, ledgers = {}, {}
    for scope, days in (('training', TRAIN_DAYS), ('evaluation', EVAL_DAYS)):
        clocks = pd.read_parquet(output/f'request331-{scope}-clocks.parquet')
        # Reconstruct the time-only schedule and all identity ledgers independently.
        expected, ledger = clock_manifest(identities[scope], raw[scope][['trading_day', 'ticker', 't']])
        pd.testing.assert_frame_equal(expected, clocks, check_exact=True)
        pd.testing.assert_frame_equal(ledger, pd.read_parquet(output/f'request331-{scope}-clock-ledger.parquet'), check_exact=True)
        frames[scope] = label_clocks(clocks, raw[scope], days)
        saved = pd.read_parquet(previous/f'official330/request330-{"training-states" if scope == "training" else "states"}.parquet')
        verify_r330(frames[scope], saved)
        ledgers[scope] = identity_support(frames[scope], identities[scope])
    report = {}
    for name, scope, days in (('training_all', 'training', TRAIN_DAYS), ('training_primary', 'training', PRIMARY_DAYS), ('evaluation', 'evaluation', EVAL_DAYS)):
        states = frames[scope].loc[frames[scope].trading_day.isin(days)]
        ledger = ledgers[scope].loc[ledgers[scope].trading_day.isin(days)].reset_index(drop=True)
        identity = identities[scope].loc[identities[scope].trading_day.isin(days)]
        report[name] = {'arms': {arm: support_metrics(states, identity, arm) for arm in ARMS},
            'per_day': {day: {arm: support_metrics(states.loc[states.trading_day.eq(day)], identity.loc[identity.trading_day.eq(day)], arm) for arm in ARMS} for day in days},
            'paired_changes': {flag: ledger[f'{flag}_change'].value_counts().to_dict() for flag in FLAGS[1:]},
            'clustered_support_intervals': clustered_support(ledger)}
    integrity = {'registered_inputs_verified': True, 'exact_populations': True,
        'early_clocks_preserved': True, 'early_R330_contract_reproduced': True,
        'clock_manifest_before_targets': True, 'causal_clock_cost_invariance_tests': True}
    result = {'request_id': REQUEST_ID, 'development_only': True, 'promotion_eligible': False,
        'market_requests': 0, 'June_HOLD_opened': False, 'final_July_August_opened': False,
        'model_fits': 0, 'input_sha256': pins, 'clock_registration_sha256': digest(clock_registration),
        'scope_reports': report, 'integrity': integrity,
        'limitations': 'Reused May census;3 primary dates. Conditional known payoffs are not policy or account returns. Next-print cost proxies omit NBBO/size and actual fills. Clusters quantify fixed-census support, not independent new validation. No policy or G transport to late phases.'}
    result['gate_checks'] = gates(report['training_primary'], integrity)
    for scope in frames:
        atomic_parquet(frames[scope], output/f'request331-{scope}-targets.parquet')
        atomic_parquet(ledgers[scope], output/f'request331-{scope}-support-ledger.parquet')
    write_json(result, output/'request331.json')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('clocks', 'outcomes'), required=True)
    for name in ('previous', 'pinned-inputs', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--clock-registration', type=Path)
    args = parser.parse_args()
    if args.stage == 'clocks':
        result = save_clocks(args.previous, args.pinned_inputs, args.output_dir)
    else:
        if args.clock_registration is None:
            parser.error('outcomes require the previously published clock registration')
        result = census(args.previous, args.pinned_inputs, args.output_dir, args.clock_registration)
    print(json.dumps(result.get('gate_checks', result), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
