"""Post-result input/coverage audit; no outcome fitting or subgroup model selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from victory_trader.feasible_upside_observability import CEILING, truth
from victory_trader.compact_momentum_representation import weighted_quantile
from victory_trader.frozen_momentum_may_transport import (
    EVAL_DAYS, POPULATION_COLUMNS, causal_population, first_observation,
    freeze_models, score_available, validate_training_context,
)
from victory_trader.hierarchical_crack_entry_controller import _episode_day_weights
from victory_trader.pending_exit_integrity import session_limits
from victory_trader.preentry_momentum_observability import (
    CLOCK_FEATURES, clock_features, entry_labels, snapshots,
)


def quantiles(series):
    return {str(q): float(series.quantile(q)) for q in (.1, .5, .9)}


def input_summary(frame):
    weights = _episode_day_weights(frame)
    return {
        'rows': len(frame),
        'HOT_minutes_since_open': quantiles(frame.minutes_since_open),
        'decision_delay_s': quantiles((frame.decision_t-frame.hot_t)/1000),
        'entry_gap_s': quantiles(frame.label_entry_gap_s),
        'known_cost_drag_pct': quantiles(frame.known_cost_drag_pct),
        'activity_medians': {str(w): float(frame[f'momentum_activity_fraction_{w}s'].median()) for w in (10, 30, 120)},
        'equal_day_episode_weighted_activity_medians': {
            str(w): weighted_quantile(frame[f'momentum_activity_fraction_{w}s'].to_numpy(), weights, .5)
            for w in (10, 30, 120)
        },
        'missing_fractions': frame[list(CLOCK_FEATURES)].isna().mean().to_dict(),
        'equal_day_episode_weighted_missing_fractions': {
            c: float(np.average(frame[c].isna(), weights=weights)) for c in CLOCK_FEATURES
        },
        'constant_completed_print_gap': bool(frame.momentum_gap_s.eq(0).all()),
        'cost_only_stop_count': int(frame.label_stop_cost_only.fillna(False).sum()),
    }


def history_summary(frame, contexts):
    values = {w: {'counts': [], 'baseline_age': []} for w in (10, 30, 120)}
    for row in frame.itertuples():
        bars = contexts[(str(row.trading_day), str(row.ticker))][0]
        ends = bars.t.to_numpy(np.int64)+1000
        right = np.searchsorted(ends, int(row.decision_t), side='right')
        for window, out in values.items():
            left = np.searchsorted(ends, int(row.decision_t)-window*1000, side='right')
            out['counts'].append(int(right-left))
            if left:
                out['baseline_age'].append(float((int(row.decision_t)-ends[left-1])/1000))
    return {str(w): {
        'window_prints_median': float(np.median(v['counts'])),
        'at_most_one_print_count': int(sum(n <= 1 for n in v['counts'])),
        'return_baseline_available': len(v['baseline_age']),
        'return_baseline_age_s': quantiles(pd.Series(v['baseline_age'])),
        'return_baseline_older_than_twice_window_count': int(sum(t > 2*w for t in v['baseline_age'])),
    } for w, v in values.items()}


def contexts_for(raw):
    result = {}
    for key, bars in raw.groupby(['trading_day', 'ticker'], sort=False):
        opening, closing = session_limits(str(key[0]))
        assert bars.t.ge(opening).all() and bars.t.lt(closing).all()
        bars = bars.sort_values('t').reset_index(drop=True)
        assert not bars.t.duplicated().any()
        result[key] = (bars, opening, closing)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('official', 'training', 'training-raw', 'population', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    official = json.loads((args.official/'request311.json').read_text())
    train = pd.read_parquet(args.training/'request310-states.parquet')
    train_summary = json.loads((args.training/'request310.json').read_text())
    states = pd.read_parquet(args.official/'request311-states.parquet')
    cohort = pd.read_parquet(args.official/'request311-cohort.parquet')
    raw = pd.read_parquet(args.official/'request311-raw-seconds.parquet')
    source = pd.read_parquet(args.population, columns=list(POPULATION_COLUMNS))
    reference = pd.read_parquet(args.reference, columns=list(POPULATION_COLUMNS))
    expected, audit = causal_population(source)
    other, _ = causal_population(reference)
    for candidate in (cohort, states[list(cohort.columns)], other):
        pd.testing.assert_frame_equal(expected, candidate, check_dtype=False, atol=1e-12, rtol=1e-12)
    assert audit == official['population']
    alignment = validate_training_context(train, source)
    assert alignment == official['training_context_alignment']
    assert set(raw.trading_day) <= set(EVAL_DAYS)
    raw_keys = set(map(tuple, raw[['trading_day', 'ticker']].drop_duplicates().to_numpy()))
    assert raw_keys <= set(map(tuple, cohort[['trading_day', 'ticker']].to_numpy()))
    contexts = contexts_for(raw)
    missing = {}
    for row in states.itertuples():
        key = (str(row.trading_day), str(row.ticker))
        bars, _, closing = contexts.get(key, (pd.DataFrame(columns=['t']), 0, session_limits(key[0])[1]))
        decision = first_observation(bars, int(row.hot_t), closing)
        assert (decision is not None) == row.observation_available
        if decision is None:
            if int(row.hot_t) >= closing:
                reason = 'HOT_at_session_close'
            elif bars.loc[bars.t.ge(row.hot_t)].empty:
                reason = 'no_regular_print_at_or_after_HOT'
            else:
                reason = 'first_regular_print_5min_or_more_after_HOT'
            missing[reason] = missing.get(reason, 0)+1
            continue
        assert decision == row.decision_t
        labels = entry_labels(bars, decision, min(int(row.hot_t)+3600000, closing))
        assert labels['label_complete'] == row.label_complete
        assert np.isclose(labels[CEILING], getattr(row, CEILING), atol=1e-9, rtol=0, equal_nan=True)
        if not row.label_complete:
            entry = np.searchsorted(bars.t.to_numpy(), decision)
            cap = min(int(row.hot_t)+3600000, closing)
            if entry == len(bars) or bars.t.iloc[entry] >= cap:
                reason = 'no_entry_print_before_cap'
            elif not ((bars.t >= bars.t.iloc[entry]) & (bars.t+1000 <= cap)).any():
                reason = 'no_completed_entry_print_before_cap'
            else:
                reason = 'no_terminal_fill_after_submission'
            missing[reason] = missing.get(reason, 0)+1
    observed = states.loc[states.observation_available].copy()
    replay = clock_features(observed.drop(columns=list(CLOCK_FEATURES)), contexts)
    pd.testing.assert_frame_equal(replay[list(CLOCK_FEATURES)], observed[list(CLOCK_FEATURES)], check_dtype=False, atol=1e-9, rtol=1e-12)
    models, _, first_prior = freeze_models(train, train_summary)
    rescored = score_available(states, models, first_prior)
    errors = {a: float(np.nanmax(np.abs(states[f'{a}_p_net_5']-rescored[f'{a}_p_net_5']))) for a in models}
    assert max(errors.values()) < 1e-9
    assert states.loc[~states.observation_available, [f'{a}_p_net_5' for a in models]].isna().all().all()
    first = snapshots(train, 0)
    training_contexts = contexts_for(pd.read_parquet(args.training_raw))
    training_replay = clock_features(first.drop(columns=list(CLOCK_FEATURES)), training_contexts)
    pd.testing.assert_frame_equal(training_replay[list(CLOCK_FEATURES)], first[list(CLOCK_FEATURES)], check_dtype=False, atol=1e-9, rtol=1e-12)
    labeled = states.loc[states.label_complete].copy()
    weights = _episode_day_weights(labeled)
    cost_only = labeled.label_stop_cost_only.fillna(False).astype(bool)
    whole_known = labeled.label_whole_max_net_pct.notna()
    result = {
        'request_id': 311, 'posthoc_descriptive_only': True,
        'official_source': (args.official/'source-sha.txt').read_text().strip(),
        'input_sha256': {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            args.official/'request311.json', args.official/'request311-cohort.parquet',
            args.official/'request311-states.parquet', args.official/'request311-raw-seconds.parquet',
            args.training/'request310-states.parquet', args.training_raw,
        )},
        'verified_cohort_rows': len(cohort), 'verified_regular_raw_rows': len(raw),
        'verified_observations_labels_and_features': True,
        'maximum_prediction_replay_errors': errors,
        'training_context_alignment': alignment,
        'training_all_inputs': input_summary(train), 'training_first_inputs': input_summary(first),
        'evaluation_observed_inputs': input_summary(observed), 'evaluation_labeled_inputs': input_summary(labeled),
        'training_first_history': history_summary(first, training_contexts),
        'evaluation_observed_history': history_summary(observed, contexts),
        'missing_details': missing,
        'unobserved_HOT_time': quantiles(states.loc[~states.observation_available, 'minutes_since_open']),
        'incomplete_HOT_time': quantiles(observed.loc[~observed.label_complete, 'minutes_since_open']),
        'weighted_predicted_means': {a: float(np.average(labeled[f'{a}_p_net_5'], weights=weights)) for a in models},
        'weighted_observed_prevalence': float(np.average(truth(labeled[CEILING], 5), weights=weights)),
        'cost_only_stop_episodes': int(cost_only.sum()),
        'cost_only_stop_positive_episodes': int((cost_only & labeled[CEILING].ge(5)).sum()),
        'known_whole_horizon_episodes': int(whole_known.sum()),
        'whole_horizon_net5_but_stop_reachable_below5': int((whole_known & labeled.label_whole_max_net_pct.ge(5) & labeled[CEILING].lt(5)).sum()),
        'same_day_M_minus_N': official['transport']['arms']['M']['within_day_auc']-official['transport']['arms']['N']['within_day_auc'],
        'passed_checks': sum(official['checks'].values()),
        'limits': 'Descriptions after official outcomes, no refit on evaluation labels, policy/threshold/gate or subgroup choice. Medians unweighted descriptive; metric weights original equal-day/episode. Whole-horizon diagnostic is hindsight with unchanged cap/cost and incomplete whole paths reported; not executable.',
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
