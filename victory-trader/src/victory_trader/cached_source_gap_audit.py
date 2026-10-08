"""R332: immutable cached timestamp census; no prices, targets or fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import action_target_support_census as census
from .frozen_entry_second_hold_exit import KEYS
from .pending_exit_integrity import session_limits

CLOCK_KEYS = [*KEYS, 'decision_t', 'phase']
TIMING_FIELDS = ['cost_compatible', 'enter_resolved', 'enter_reason',
    'entry_fill_t', 'exit_submission_t', 'exit_fill_t',
    'next_entry_reference_gap_s', 'next_exit_reference_gap_s']
BINS = ('at_most3s', 'over3_through30s', 'over30_through60s', 'over60s', 'no_later_cached_reference')


def gap_bin(gap):
    if not np.isfinite(gap):
        return BINS[4]
    if gap < 0:
        raise ValueError('next reference cannot precede submission')
    return BINS[0] if gap <= 3 else BINS[1] if gap <= 30 else BINS[2] if gap <= 60 else BINS[3]


def next_reference(times, submission, boundary):
    """Boundary is HOT terminal for entry, regular close for exit diagnosis."""
    if not np.isfinite(submission):
        return np.nan, np.nan, 'no_submission'
    if times is None or not len(times):
        return np.nan, np.nan, 'raw_pair_absent'
    at = np.searchsorted(times, submission)
    if at == len(times):
        return np.nan, np.nan, 'no_later_cached_reference'
    stamp = int(times[at])
    gap = (stamp-submission)/1000
    category = 'at_or_after_boundary' if stamp >= boundary else 'within3s_before_boundary' if gap <= 3 else 'late_before_boundary'
    return float(stamp), float(gap), category


def time_cache(raw):
    if raw.duplicated(['trading_day', 'ticker', 't']).any() or not np.isfinite(raw.t).all():
        raise ValueError('unique finite cached timestamps required')
    cache = {}
    for key, group in raw.groupby(['trading_day', 'ticker'], sort=True):
        times = np.sort(group.t.to_numpy(np.int64))
        opening, closing = session_limits(key[0])
        if (times < opening).any() or (times+1000 > closing).any():
            raise ValueError('only original regular-session seconds allowed')
        cache[key] = times
    return cache


def identity_ledger(registered, cache):
    if registered.duplicated(KEYS).any():
        raise ValueError('unique original identities required')
    rows = []
    for state in registered.itertuples(index=False):
        times = cache.get((state.trading_day, state.ticker), np.array([], np.int64))
        cap = min(state.hot_t+3600000, session_limits(state.trading_day)[1])
        eligible = (times >= state.hot_t) & (times+1000 < cap)
        if int(state.terminal_t) != cap or bool(state.raw_pair_available) != bool(len(times)):
            raise ValueError('original raw-pair/terminal ledger changed')
        if bool(state.no_observation) != (state.extended_clocks == 0):
            raise ValueError('no-observation identities must be retained')
        rows.append({**{k: getattr(state, k) for k in KEYS}, 'terminal_t': cap,
            'raw_pair_available': bool(len(times)), 'full_session_cached_seconds': len(times),
            'first_cached_start_t': float(times[0]) if len(times) else np.nan,
            'last_cached_start_t': float(times[-1]) if len(times) else np.nan,
            'eligible_completed_seconds': int(eligible.sum()),
            'before_hot_seconds': int((times < state.hot_t).sum()),
            'at_or_after_terminal_seconds': int((times >= cap).sum()),
            'early_clocks': int(state.early_clocks), 'late_clocks': int(state.late_clocks),
            'extended_clocks': int(state.extended_clocks), 'no_observation': bool(state.no_observation),
            'response_completeness': 'UNKNOWN_NOT_RETAINED',
            'gap_cause': 'UNIDENTIFIED_NO_INDEPENDENT_STREAM'})
    return pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)


def clock_audit(clocks, timing, cache):
    if clocks.duplicated(CLOCK_KEYS).any() or timing.duplicated(CLOCK_KEYS).any():
        raise ValueError('unique original clocks required')
    left, right = (x[CLOCK_KEYS].sort_values(CLOCK_KEYS).reset_index(drop=True) for x in (clocks, timing))
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    frame = clocks.merge(timing[[*CLOCK_KEYS, *TIMING_FIELDS]], on=CLOCK_KEYS, how='left', validate='one_to_one').sort_values(CLOCK_KEYS).reset_index(drop=True)
    added = []
    for state in frame.itertuples(index=False):
        times = cache.get((state.trading_day, state.ticker))
        closing = session_limits(state.trading_day)[1]
        cap = min(state.hot_t+3600000, closing)
        entry = next_reference(times, state.decision_t, cap)
        exit_ref = next_reference(times, state.exit_submission_t, closing)
        # Existing economic labels and stop times are never recalculated.
        for gap, saved in ((entry[1], state.next_entry_reference_gap_s), (exit_ref[1], state.next_exit_reference_gap_s)):
            if not (np.isnan(gap) and np.isnan(saved)) and gap != saved:
                raise ValueError('registered timestamp gap changed')
        if bool(state.enter_resolved) != (state.enter_reason in ('stop', 'fixed60s_or_session_cap')):
            raise ValueError('registered resolution/reason inconsistency')
        added.append({'next_cached_entry_t': entry[0], 'entry_gap_bin': gap_bin(entry[1]),
            'entry_reference_anatomy': entry[2], 'next_cached_exit_t': exit_ref[0],
            'exit_gap_bin': gap_bin(exit_ref[1]) if np.isfinite(state.exit_submission_t) else 'no_submission',
            'exit_reference_anatomy': exit_ref[2], 'gap_cause': 'UNIDENTIFIED_NO_INDEPENDENT_STREAM'})
    return pd.concat([frame, pd.DataFrame(added)], axis=1)


def counts(frame, column):
    return {str(k): int(v) for k, v in frame[column].value_counts().sort_index().items()}


def clock_summary(frame):
    compatible = frame.loc[frame.cost_compatible]
    missing_entry = compatible.loc[compatible.enter_reason.eq('entry_missing_or_late')]
    missing_exit = compatible.loc[compatible.enter_reason.eq('exit_missing_or_late')]
    result = {'all_clocks': len(frame), 'compatible_clocks': len(compatible),
        'resolved_compatible_clocks': int(compatible.enter_resolved.sum()),
        'unresolved_compatible_clocks': int((~compatible.enter_resolved).sum()),
        'pooled_compatible_resolution': float(compatible.enter_resolved.mean()) if len(compatible) else None,
        'missing_entry_clocks': len(missing_entry), 'missing_exit_given_entry_clocks': len(missing_exit)}
    for name, subset in (('all', frame), ('compatible', compatible), ('compatible_missing_entry', missing_entry), ('compatible_missing_exit', missing_exit)):
        result[name] = {column: counts(subset, column) for column in ('enter_reason', 'entry_gap_bin', 'entry_reference_anatomy', 'exit_gap_bin', 'exit_reference_anatomy')}
    return result


def scope_summary(ledger, audit):
    result = {'identities': len(ledger), 'raw_pair_present_identities': int(ledger.raw_pair_available.sum()),
        'no_observation_identities': int(ledger.no_observation.sum()),
        'eligible_completed_seconds': int(ledger.eligible_completed_seconds.sum()),
        'identities_with_no_eligible_seconds': int(ledger.eligible_completed_seconds.eq(0).sum()),
        'independent_stream_reconciled_identities': 0, 'response_page_verified_identities': 0,
        'source_gap_cause_identified_identities': 0, 'arms': {}}
    for arm in census.ARMS:
        frame = census.choose_arm(audit, arm)
        result['arms'][arm] = clock_summary(frame)
    result['per_day'] = {str(day): {'identities': int(ledger.trading_day.eq(day).sum()),
        'no_observation_identities': int(ledger.loc[ledger.trading_day.eq(day), 'no_observation'].sum()),
        'arms': {arm: clock_summary(census.choose_arm(audit.loc[audit.trading_day.eq(day)], arm)) for arm in census.ARMS}} for day in sorted(ledger.trading_day.unique())}
    return result


def lineage_report(previous, evidence):
    required = ('request_bounds', 'adjusted', 'statuses', 'page_counts', 'terminal_next_url', 'payload_sha256')
    for source in evidence['original_sources'].values():
        if set(source['retained_response_evidence']) != set(required) or any(source['retained_response_evidence'][k] is not None for k in required) or source['independent_full_window_streams']:
            raise ValueError('new response/independent stream requires a new registered reconciliation')
    reported = {'311': json.loads((previous/'official311/request311.json').read_text())['acquisition'],
        '315': json.loads((previous/'official315/request315-acquisition.json').read_text())['acquisition']}
    return {'original_sources': evidence['original_sources'], 'acquisition_summaries_as_reported': reported,
        'reported_call_counts_are_not_per_pair_response_proof': True,
        'R315_summary_scope': 'original478 union;97 inherited and381 fetched;not audited407 subset',
        'response_completeness': 'UNKNOWN_NOT_RETAINED',
        'independent_full_window_streams': [], 'gap_cause': 'UNIDENTIFIED',
        'absence_scope': evidence['scope'], 'last_cached_bar_is_not_proof_of_request_truncation': True}


def run(previous, pinned, evidence_path, output):
    pins = json.loads(pinned.read_text())
    for name, sha in pins.items():
        if census.digest(previous/name) != sha:
            raise ValueError('registered input changed: '+name)
    sources = {311: 'a6034adba43fbd02334834f68341a64951957ba6', 315: 'c271fbab9c3567c470fc852ef80c39fe36116713', 331: '175645954cf2216781992a4619356c49b348332b'}
    for n, sha in sources.items():
        if (previous/f'official{n}/source-sha.txt').read_text().strip() != sha:
            raise ValueError('original source SHA changed')
    evidence = json.loads(evidence_path.read_text())
    cohort = pd.read_parquet(previous/'official315/request315-training-cohort.parquet', columns=[*KEYS, 'in_broad'])
    identities = {'training': cohort.loc[cohort.in_broad, KEYS],
        'evaluation': pd.read_parquet(previous/'official319/request319-episode-ledger.parquet', columns=KEYS)}
    if identities['training'].groupby('trading_day').size().to_dict() != dict(zip(census.TRAIN_DAYS, (102, 108, 97, 100))) or len(identities['evaluation']) != 828 or set(identities['evaluation'].trading_day) != set(census.EVAL_DAYS):
        raise ValueError('original407/828 census required')
    output.mkdir(parents=True, exist_ok=True)
    ledgers, audits = {}, {}
    for scope, raw_name, days in (('training', 'official315/request315-training-raw-seconds.parquet', census.TRAIN_DAYS), ('evaluation', 'official311/request311-raw-seconds.parquet', census.EVAL_DAYS)):
        raw = pd.read_parquet(previous/raw_name, columns=['trading_day', 'ticker', 't'])
        if not set(raw.trading_day).issubset(days):
            raise ValueError('sealed/out-of-scope date rejected')
        cache = time_cache(raw)
        registered = pd.read_parquet(previous/f'official331/request331-{scope}-clock-ledger.parquet')
        pd.testing.assert_frame_equal(identities[scope].sort_values(KEYS).reset_index(drop=True), registered[KEYS].sort_values(KEYS).reset_index(drop=True), check_dtype=False, check_exact=True)
        clocks = pd.read_parquet(previous/f'official331/request331-{scope}-clocks.parquet', columns=CLOCK_KEYS)
        timing = pd.read_parquet(previous/f'official331/request331-{scope}-targets.parquet', columns=[*CLOCK_KEYS, *TIMING_FIELDS])
        ledgers[scope] = identity_ledger(registered, cache)
        audits[scope] = clock_audit(clocks, timing, cache)
        observed = audits[scope].groupby(KEYS).size().rename('actual')
        joined = ledgers[scope].merge(observed, on=KEYS, how='left', validate='one_to_one')
        if not joined.actual.fillna(0).eq(joined.extended_clocks).all():
            raise ValueError('registered identity observation counts changed')
        for kind, frame in (('source-ledger', ledgers[scope]), ('clock-audit', audits[scope])):
            census.atomic_parquet(frame, output/f'request332-{scope}-{kind}.parquet')
    scopes = {'training_all': (ledgers['training'], audits['training']),
        'training_primary': (ledgers['training'].loc[ledgers['training'].trading_day.isin(census.PRIMARY_DAYS)], audits['training'].loc[audits['training'].trading_day.isin(census.PRIMARY_DAYS)]),
        'evaluation': (ledgers['evaluation'], audits['evaluation'])}
    result = {'request_id': 332, 'audit_completed': True, 'development_only': True,
        'new_market_requests': 0, 'new_economic_labels': 0, 'model_fits': 0,
        'promotion_eligible': False, 'sealed_dates_opened': False, 'input_sha256': pins,
        'source_evidence_sha256': census.digest(evidence_path), 'lineage': lineage_report(previous, evidence),
        'integrity_checks': {'all_input_hashes_and_original_sources_match': True,
            'all_original_identities_and_clocks_retained': True, 'registered_reference_gaps_exact': True,
            'only_timestamp_and_existing_timing_flags_read': True},
        'source_readiness': {'per_pair_response_completeness': 'UNKNOWN',
            'independent_same_window_reconciliation': 'UNAVAILABLE', 'missing_reference_causes': 'UNIDENTIFIED',
            'R331_90pct_resolution_failure_remains': True, 'new_value_fit_justified': False},
        'scopes': {name: scope_summary(*frames) for name, frames in scopes.items()}}
    result['output_sha256'] = {p.name: census.digest(p) for p in sorted(output.glob('request332-*.parquet'))}
    census.write_json(result, output/'request332.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--pinned-inputs', type=Path, required=True)
    parser.add_argument('--source-evidence', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.previous, args.pinned_inputs, args.source_evidence, args.output_dir)
    print(json.dumps({'request_id': 332, 'audit_completed': result['audit_completed'], 'source_readiness': result['source_readiness'], 'scope_identity_counts': {k: v['identities'] for k, v in result['scopes'].items()}}, indent=2))
