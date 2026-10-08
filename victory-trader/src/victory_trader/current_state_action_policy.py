"""R330: cost-aware ENTER versus a fixed causal WAIT continuation policy."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import gated_trajectory_residual_momentum as frozen
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS
from .frozen_entry_second_hold_exit import HARD_STOP_PCT, KEYS, MAX_LAG_MS, net
from .pending_exit_integrity import session_limits
from .within_episode_momentum import frozen_transform

REQUEST_ID, HOLD_MS, WATCH_MS, ALPHA = 330, 60000, 300000, 10.
POLICIES = ('F', 'E', 'G', 'G0', 'CASH')
SCENARIOS = tuple(s.name for s in DEFAULT_EXECUTION_SCENARIOS)
INPUTS = ('G_logit', *[f'current_{j}' for j in range(7)], 'gate', 'cost_drag_stop_units')


def contexts(raw, allowed_days):
    if not set(raw.trading_day).issubset(allowed_days) or raw.duplicated(['trading_day', 'ticker', 't']).any():
        raise ValueError('unique cached allowed-date seconds required')
    if not np.isfinite(raw[['t', 'o', 'c']].to_numpy(float)).all() or (raw[['o', 'c']] <= 0).any().any():
        raise ValueError('finite positive cached references required')
    return {k: g.sort_values('t').reset_index(drop=True) for k, g in raw.groupby(['trading_day', 'ticker'], sort=False)}


def verify_clocks(states, identities, cache):
    actual = {k: g.sort_values('decision_t').decision_t.to_numpy(np.int64) for k, g in states.groupby(KEYS, sort=False)}
    for key in identities[KEYS].itertuples(index=False, name=None):
        day, ticker, hot = key
        bars = cache.get((day, ticker))
        cap = min(hot+3600000, session_limits(day)[1])
        if bars is None:
            expected = np.array([], np.int64)
        else:
            t = bars.t.to_numpy(np.int64)
            expected = t[(t >= hot) & (t+1000 <= hot+WATCH_MS) & (t+1000 < cap)]+1000
        np.testing.assert_array_equal(actual.get(key, np.array([], np.int64)), expected)


def enter_outcome(bars, decision_t, hot_t, closing):
    """Future execution/outcome label only; never called by action selection."""
    out = {'enter_resolved': False, 'enter_reason': 'entry_missing_or_late',
        'entry_fill_t': np.nan, 'entry_reference_price': np.nan, 'entry_lag_s': np.nan,
        'exit_submission_t': np.nan, 'exit_fill_t': np.nan, 'exit_reference_price': np.nan,
        'exit_lag_s': np.nan, 'stop_triggered': False, 'cost_only_stop': False,
        **{f'enter_{s}_pct': np.nan for s in SCENARIOS}}
    if bars is None or bars.empty:
        return out
    times, opens, closes = (bars[c].to_numpy() for c in ('t', 'o', 'c'))
    at = int(np.searchsorted(times, decision_t))
    cap = min(int(hot_t)+3600000, closing)
    if at == len(times) or times[at] >= cap or times[at]-decision_t > MAX_LAG_MS:
        return out
    entry_t, price = int(times[at]), float(opens[at])
    deadline = min(entry_t+HOLD_MS, cap)
    observed = np.flatnonzero((times >= entry_t) & (times+1000 <= deadline))
    losses = np.array([net(price, float(closes[j])) for j in observed])
    stop = np.flatnonzero(losses <= HARD_STOP_PCT)
    submission = int(times[observed[stop[0]]])+1000 if len(stop) else deadline
    out.update(entry_fill_t=entry_t, entry_reference_price=price, entry_lag_s=(entry_t-decision_t)/1000,
        exit_submission_t=submission, stop_triggered=bool(len(stop)), cost_only_stop=net(price, price) <= HARD_STOP_PCT,
        enter_reason='exit_missing_or_late')
    fill = int(np.searchsorted(times, submission))
    if fill == len(times) or times[fill]-submission > MAX_LAG_MS:
        return out
    exit_price = float(opens[fill])
    out.update(enter_resolved=True, enter_reason='stop' if len(stop) else 'fixed60s_or_session_cap',
        exit_fill_t=int(times[fill]), exit_reference_price=exit_price, exit_lag_s=(times[fill]-submission)/1000)
    for s in DEFAULT_EXECUTION_SCENARIOS:
        out[f'enter_{s.name}_pct'] = net(price, exit_price, s)
    return out


def baseline_signals(states, controls):
    identities = [*KEYS, 'decision_t']
    frozen.verify_checkpoint(states[identities], controls[identities])
    g, prior = (controls[f'R328_B_{s}'].to_numpy(float) for s in ('p_net_5', 'prior_net_5'))
    cost = states.known_cost_drag_pct.to_numpy(float)
    return np.isfinite(cost) & (cost < -HARD_STOP_PCT) & np.isfinite(g) & np.isfinite(prior) & (g >= prior)


def attach_wait_targets(states, signals):
    """The next causal pi0 entry is chosen by signals, never by its payoff."""
    result = states.copy()
    waits = {s: np.zeros(len(states)) for s in SCENARIOS}
    sources = np.full(len(states), -1, np.int64)
    for _, group in states.groupby(KEYS, sort=False):
        positions = states.index.get_indexer(group.sort_values('decision_t').index)
        chosen = -1
        for now in positions[::-1]:
            sources[now] = chosen
            if chosen >= 0:
                for s in SCENARIOS:
                    waits[s][now] = states[f'enter_{s}_pct'].iloc[chosen]
            if signals[now]:
                chosen = now
    result['wait_selected_next_position'] = sources
    for s in SCENARIOS:
        result[f'wait_{s}_pct'] = waits[s]
    return result


def targets(states, cache, controls):
    outcomes = [enter_outcome(cache.get((r.trading_day, r.ticker)), int(r.decision_t), int(r.hot_t), session_limits(r.trading_day)[1]) for r in states.itertuples()]
    enriched = pd.concat([states, pd.DataFrame(outcomes, index=states.index)], axis=1)
    return attach_wait_targets(enriched, baseline_signals(states, controls))


def original_weights(states):
    return 1/states.groupby(KEYS, sort=False).ticker.transform('size').to_numpy(float)


def current_inputs(states, controls, transform=None):
    frozen.verify_checkpoint(states[[*KEYS, 'decision_t']], controls[[*KEYS, 'decision_t']])
    gate = controls.R328_gate.to_numpy(bool)
    if transform is None:
        current = controls[[f'R328_S_design_{j}' for j in range(7)]].to_numpy(float)
    else:
        current = frozen.design(frozen.old.feature_paths(states), gate, 'S', transform)
    x = np.c_[controls.R328_B_logit.to_numpy(float), current, gate.astype(float), states.known_cost_drag_pct.to_numpy(float)/(-HARD_STOP_PCT)]
    if not np.isfinite(x).all():
        raise ValueError('finite current-state inputs required')
    return x


def fit_values(states, x):
    if states.empty:
        return None, {'no_support': True, 'reason': 'no_preceding_value_fit', 'optimization_attempted': False}
    y = states[['enter_base_pct', 'wait_base_pct']].to_numpy(float)
    chosen = np.isfinite(y).all(axis=1)
    weights = original_weights(states)
    if not chosen.any():
        return None, {'no_support': True, 'reason': 'no_common_finite_action_targets', 'optimization_attempted': False}
    z = np.c_[x[chosen], np.ones(chosen.sum())]
    w = weights[chosen]
    gram = z.T@(w[:, None]*z)+np.diag([ALPHA]*x.shape[1]+[0.])
    rhs = z.T@(w[:, None]*y[chosen])
    condition = float(np.linalg.cond(gram))
    beta = np.linalg.solve(gram, rhs)
    residual = float(np.max(np.abs(gram@beta-rhs)))
    if not np.isfinite(beta).all() or not np.isfinite(condition) or residual > 1e-8:
        raise ValueError('invalid one-shot ridge normal equations; no retry')
    pos, neg = chosen & (y[:, 0] > 0), chosen & (y[:, 0] <= 0)
    return beta, {'no_support': False, 'optimization_attempted': True, 'fit_days': sorted(states.trading_day.unique()),
        'inputs': list(INPUTS), 'alpha': ALPHA, 'unpenalized_intercept': True, 'coefficients': beta[:-1].T.tolist(),
        'intercepts': beta[-1].tolist(), 'observed_states': len(states), 'common_finite_targets': int(chosen.sum()),
        'common_target_episodes': len(states.loc[chosen, KEYS].drop_duplicates()),
        'positive_enter_episodes': len(states.loc[pos, KEYS].drop_duplicates()),
        'positive_enter_dates': sorted(states.loc[pos, 'trading_day'].unique()),
        'negative_enter_dates': sorted(states.loc[neg, 'trading_day'].unique()),
        'original_observed_weight_mass': float(weights.sum()), 'common_target_original_weight_mass': float(w.sum()),
        'condition_number': condition, 'maximum_normal_equation_residual': residual,
        'weighted_target_means': np.average(y[chosen], weights=w, axis=0).tolist()}


def score(states, controls, model):
    if model is None:
        q = np.full((len(states), 2), np.nan)
    else:
        x = current_inputs(states, controls)
        q = np.c_[x, np.ones(len(states))]@model
    cost = states.known_cost_drag_pct.to_numpy(float) < -HARD_STOP_PCT
    baseline = baseline_signals(states, controls)
    first = frozen.previous.history(states).first
    desires = {'F': cost & np.isfinite(q).all(axis=1) & (q[:, 0] > np.maximum(0., q[:, 1])),
        'E': cost & np.isfinite(q[:, 0]) & (q[:, 0] > 0), 'G': baseline, 'G0': baseline & first,
        'CASH': np.zeros(len(states), bool)}
    priority = {'F': q[:, 0]-np.maximum(0., q[:, 1]), 'E': q[:, 0],
        'G': controls.R328_B_p_net_5.to_numpy()-controls.R328_B_prior_net_5.to_numpy()}
    priority['G0'], priority['CASH'] = priority['G'], np.zeros(len(states))
    columns = {'R330_Q_ENTER': q[:, 0], 'R330_Q_WAIT_pi0': q[:, 1], 'R330_value_fit_supported': np.isfinite(q).all(axis=1)}
    for p in POLICIES:
        columns[f'R330_{p}_wants_ENTER'] = desires[p]
        columns[f'R330_{p}_priority'] = priority[p]
    return pd.concat([states, pd.DataFrame(columns, index=states.index)], axis=1)


def episode_replay(scored, ledger):
    rows = []
    groups = {k: g.sort_values('decision_t') for k, g in scored.groupby(KEYS, sort=False)}
    for key in ledger[KEYS].itertuples(index=False, name=None):
        group = groups.get(key)
        row = dict(zip(KEYS, key))
        row['observed_states'] = 0 if group is None else len(group)
        row['fixed_contract_positive_opportunity'] = bool(group is not None and group.enter_base_pct.gt(0).any())
        for p in POLICIES:
            chosen = None if group is None else group.loc[group[f'R330_{p}_wants_ENTER']].head(1)
            entered = chosen is not None and len(chosen) > 0
            row[f'{p}_entered'], row[f'{p}_resolved'] = entered, not entered or bool(chosen.enter_resolved.iloc[0])
            row[f'{p}_selected_index'] = int(chosen.index[0]) if entered else -1
            row[f'{p}_decision_t'] = float(chosen.decision_t.iloc[0]) if entered else np.nan
            for s in SCENARIOS:
                row[f'{p}_{s}_pct'] = float(chosen[f'enter_{s}_pct'].iloc[0]) if entered else 0.
        rows.append(row)
    return pd.DataFrame(rows)


def policy_metrics(episodes, p):
    complete = episodes[f'{p}_resolved'].to_numpy(bool)
    values = episodes[f'{p}_base_pct'].to_numpy(float)
    entered = episodes[f'{p}_entered'].to_numpy(bool)
    opportunity = episodes.fixed_contract_positive_opportunity.to_numpy(bool)
    known = values[complete]
    traded = values[complete & entered]
    return {'episodes': len(episodes), 'entries': int(entered.sum()), 'resolved_or_cash': int(complete.sum()),
        'resolution_or_cash_rate': float(complete.mean()) if len(complete) else None,
        'full_population_base_mean_pct': float(values.mean()) if complete.all() and len(values) else None,
        'full_population_stress_mean_pct': float(episodes[f'{p}_stress_pct'].mean()) if complete.all() and len(values) else None,
        'resolved_only_base_mean_pct': float(known.mean()) if len(known) else None,
        'resolved_trade_base_mean_pct': float(traded.mean()) if len(traded) else None,
        'severe_resolved_trade_rate': float((traded <= HARD_STOP_PCT).mean()) if len(traded) else None,
        'positive_resolved_trades': int((traded > 0).sum()), 'observed_positive_opportunity_episodes': int(opportunity.sum()),
        'positive_opportunity_capture_rate': float((opportunity & complete & entered & (values > 0)).sum()/opportunity.sum()) if opportunity.any() else None}


def account_replay(scored, policy):
    days, orders = {}, []
    for day, frame in scored.groupby('trading_day', sort=True):
        positions, done, queue, serial = {}, set(), [], 0
        pnl = {s: 0. for s in SCENARIOS}
        equity, peak, drawdown = 1., 1., 0.
        unresolved, capacity, same_ticker, entries, max_reserved = 0, 0, 0, 0, 0
        def release(now):
            nonlocal equity, peak, drawdown
            while queue and queue[0][0] <= now:
                _, _, key, outcome = heapq.heappop(queue)
                positions.pop(key)
                for s in SCENARIOS:
                    pnl[s] += .1*outcome[f'enter_{s}_pct']/100
                equity = 1+pnl['base']
                peak = max(peak, equity)
                drawdown = max(drawdown, (peak-equity)/peak)
        for now, batch in frame.groupby('decision_t', sort=True):
            release(now)
            candidates = batch.loc[batch[f'R330_{policy}_wants_ENTER']].sort_values([f'R330_{policy}_priority', *KEYS], ascending=[False, True, True, True])
            for index, state in candidates.iterrows():
                key = tuple(state[k] for k in KEYS)
                if key in done:
                    continue
                if any(k[1] == key[1] for k in positions):
                    same_ticker += 1
                    continue
                if len(positions) >= 2 or equity-.1*len(positions) < .1:
                    capacity += 1
                    continue
                done.add(key)
                positions[key] = True
                entries += 1
                max_reserved = max(max_reserved, len(positions))
                orders.append({**dict(zip(KEYS, key)), 'policy': policy, 'source_index': int(index),
                    'submission_t': int(now), 'priority': float(state[f'R330_{policy}_priority']),
                    'notional_initial_capital_fraction': .1, 'resolved': bool(state.enter_resolved),
                    'entry_fill_t': float(state.entry_fill_t), 'exit_fill_t': float(state.exit_fill_t),
                    **{f'{s}_pct': float(state[f'enter_{s}_pct']) for s in SCENARIOS}})
                if state.enter_resolved:
                    heapq.heappush(queue, (state.exit_fill_t, serial, key, state))
                    serial += 1
                else:
                    unresolved += 1
        release(np.inf)
        days[day] = {'entries': entries, 'unresolved_orders': unresolved, 'capacity_WAIT_decisions': capacity,
            'same_ticker_WAIT_decisions': same_ticker, 'maximum_simultaneous_reserved_slots': max_reserved,
            'complete_account': unresolved == 0, 'base_return_pct': pnl['base']*100 if not unresolved else None,
            'stress_return_pct': pnl['stress']*100 if not unresolved else None,
            'resolved_leg_only_base_return_pct': pnl['base']*100,
            'closure_ledger_drawdown_pct': drawdown*100 if not unresolved else None}
    complete = all(v['complete_account'] for v in days.values())
    summary = {'days': days, 'all_days_complete': complete,
        'aggregate_daily_reset_base_return_pct': sum(v['base_return_pct'] for v in days.values()) if complete else None,
        'maximum_closure_ledger_drawdown_pct': max((v['closure_ledger_drawdown_pct'] for v in days.values()), default=0.) if complete else None,
        'full_marked_account_drawdown_pct': None, 'daily_initial_capital_reset': True,
        'limitations': 'Next-print cost scenarios; no NBBO/size fills. Closure ledger omits intratrade marks. Unknown day prevents a complete multi-day account claim.'}
    return summary, pd.DataFrame(orders)


def paired_intervals(episodes, draws=1000):
    day_counts = episodes.trading_day.value_counts()
    weights = np.array([1/(len(day_counts)*day_counts[d]) for d in episodes.trading_day])
    result = {}
    for scheme, columns, seed in (('day', ['trading_day'], 20266450), ('ticker_day', ['trading_day', 'ticker'], 20266451)):
        groups = list(episodes.groupby(columns, sort=False).indices.values())
        ids = np.zeros(len(episodes), int)
        for j, index in enumerate(groups):
            ids[index] = j
        rng = np.random.default_rng(seed)
        values = {p: [] for p in ('E', 'G')}
        for _ in range(draws):
            counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            w = weights*counts[ids]
            for p in values:
                valid = episodes.F_resolved & episodes[f'{p}_resolved']
                chosen = valid.to_numpy() & (w > 0)
                if chosen.any():
                    delta = episodes.F_base_pct.to_numpy()[chosen]-episodes[f'{p}_base_pct'].to_numpy()[chosen]
                    values[p].append(float(np.average(delta, weights=w[chosen])))
        result[scheme] = {'clusters': len(groups), 'draws': draws, 'conditional_common_resolved_only': True,
            'comparisons': {f'F_minus_{p}': {'base_mean_ci95': np.quantile(v, [.025, .975]).tolist() if v else None,
                'valid_draws': len(v), 'common_resolved_episodes': int((episodes.F_resolved & episodes[f'{p}_resolved']).sum())} for p, v in values.items()}}
    return result


def gate(result):
    fit, m, accounts = result['fit'], result['policies'], result['accounts']
    def better(x, y):
        return x is not None and y is not None and x > y
    means = {p: m[p]['full_population_base_mean_pct'] for p in POLICIES}
    dd = {p: accounts[p]['maximum_closure_ledger_drawdown_pct'] for p in ('F', 'E', 'G')}
    checks = {'training_paired_episodes150': fit.get('common_target_episodes', 0) >= 150,
        'training_positive_enter_episodes30': fit.get('positive_enter_episodes', 0) >= 30,
        'training_both_enter_signs3_dates': all(len(fit.get(k, [])) >= 3 for k in ('positive_enter_dates', 'negative_enter_dates')),
        'strict_prefix_fit_no_outcome_inputs': result['integrity']['strict_prefix_fitting_no_outcome_inputs'],
        'immutable_current_G': result['integrity']['immutable_current_G'],
        'F_entries30': m['F']['entries'] >= 30, 'F_resolution95': m['F']['resolution_or_cash_rate'] >= .95,
        'F_all_shared_capital_days_complete': accounts['F']['all_days_complete'],
        'F_full_base_positive': better(means['F'], 0), 'F_full_base_beats_E': better(means['F'], means['E']),
        'F_full_base_beats_G': better(means['F'], means['G']),
        'F_full_stress_positive': better(m['F']['full_population_stress_mean_pct'], 0),
        'F_positive_days5': sum(better(v['F']['full_population_base_mean_pct'], 0) for v in result['per_day'].values()) >= 5,
        'F_closure_drawdown_no_worse_E_G': all(dd[p] is not None for p in dd) and all(dd['F'] <= dd[p] for p in ('E', 'G'))}
    for scheme in ('day', 'ticker_day'):
        for p in ('E', 'G'):
            ci = result['paired_intervals'][scheme]['comparisons'][f'F_minus_{p}']['base_mean_ci95']
            checks[f'{scheme}_F_minus_{p}_base_ci_positive'] = ci is not None and ci[0] > 0
    if len(checks) != 18:
        raise ValueError('registered18 gates changed')
    return checks


def parquet(frame, output, suffix):
    path = output/f'request330-{suffix}.parquet'
    temporary = path.with_suffix('.parquet.tmp')
    frame.to_parquet(temporary, index=False, compression='zstd', compression_level=19)
    meta = pq.ParquetFile(temporary).metadata
    if meta.num_rows != len(frame) or meta.num_columns != len(frame.columns):
        raise ValueError('complete readable parquet footer required')
    os.replace(temporary, path)


def run(previous, pins_path, output):
    pins = json.loads(pins_path.read_text())
    for name, sha in pins.items():
        if hashlib.sha256((previous/name).read_bytes()).hexdigest() != sha:
            raise ValueError('registered input changed: '+name)
    for n, sha in ((311, 'a6034adba43fbd02334834f68341a64951957ba6'), (315, 'c271fbab9c3567c470fc852ef80c39fe36116713'), (328, 'aa031411f1ab790c2c10d5f0a92dc9aa9225aead')):
        if (previous/f'official{n}/source-sha.txt').read_text().strip() != sha:
            raise ValueError('authenticated cached source changed')
    def read(name):
        return pd.read_parquet(previous/name)
    training = read('official321/request321-chronological-states.parquet').reset_index(drop=True)
    evaluation = read('official321/request321-states.parquet')
    ledger = read('official319/request319-episode-ledger.parquet')
    train_controls = read('official328/request328-training-states.parquet')
    eval_controls = read('official328/request328-states.parquet')
    chrono_controls = read('official328/request328-chronological-states.parquet')
    audit = json.loads((previous/'official328/request328-fit-audit.json').read_text())
    for states, control in ((training, train_controls), (training, chrono_controls), (evaluation, eval_controls)):
        frozen.verify_checkpoint(states, control[list(states)])
    if len(training) != 7806 or len(evaluation) != 19530 or len(ledger) != 828 or set(evaluation.trading_day) != set(frozen.EVAL_DAYS):
        raise ValueError('fixed original state/identity scope changed')
    train_cache = contexts(read('official315/request315-training-raw-seconds.parquet'), frozen.CROSSFIT_DAYS)
    eval_cache = contexts(read('official311/request311-raw-seconds.parquet'), frozen.EVAL_DAYS)
    verify_clocks(training, training[KEYS].drop_duplicates(), train_cache)
    verify_clocks(evaluation, ledger, eval_cache)
    output.mkdir(parents=True, exist_ok=True)
    training = targets(training, train_cache, train_controls)
    x = current_inputs(training, train_controls)
    model, fit = fit_values(training, x)
    folds, pieces = {}, []
    for day in frozen.CROSSFIT_DAYS[1:]:
        before, held = training.loc[training.trading_day.lt(day)], training.loc[training.trading_day.eq(day)]
        if before.empty:
            prefix_model, fold_fit = fit_values(before, np.empty((0, len(INPUTS))))
        else:
            transform = frozen_transform(audit['folds'][day]['fits']['S']['preprocessing'])
            if audit['folds'][day]['fits']['S']['fit_days'] != sorted(before.trading_day.unique()):
                raise ValueError('strict preceding coordinate fit scope changed')
            prefix_model, fold_fit = fit_values(before, current_inputs(before, train_controls.loc[before.index], transform))
        folds[day] = fold_fit
        # Unsupported May6 value head is scored without requiring its missing current coordinates.
        selected_controls = chrono_controls.loc[held.index]
        pieces.append(score(held, selected_controls, prefix_model))
    chrono = pd.concat(pieces).sort_index()
    train_scored = score(training, train_controls, model)
    eval_scored = score(evaluation, eval_controls, model)
    # Attach evaluation labels only after the frozen model's causal actions exist.
    eval_scored = targets(eval_scored, eval_cache, eval_controls)
    episodes = episode_replay(eval_scored, ledger)
    accounts, orders = {}, []
    for policy in POLICIES:
        accounts[policy], trace = account_replay(eval_scored, policy)
        orders.append(trace)
    result = {'request_id': REQUEST_ID, 'development_only': True, 'promotion_eligible': False,
        'market_requests': 0, 'June_HOLD_opened': False, 'final_July_August_opened': False,
        'input_sha256': pins, 'fit': fit, 'chronological_fits': folds,
        'observations': {'training': len(training), 'evaluation': len(evaluation), 'chronological': len(chrono),
            'supported_chronological_value_rows': int(chrono.R330_value_fit_supported.sum()), 'ledger_identities': len(ledger)},
        'fixed_contract': {'hold_ms': HOLD_MS, 'watch_ms': WATCH_MS, 'hard_stop_base_pct': HARD_STOP_PCT, 'maximum_fill_lag_ms': MAX_LAG_MS},
        'policies': {p: policy_metrics(episodes, p) for p in POLICIES},
        'per_day': {d: {p: policy_metrics(g, p) for p in POLICIES} for d, g in episodes.groupby('trading_day', sort=True)},
        'accounts': accounts, 'paired_intervals': paired_intervals(episodes),
        'target_support': {'training_reasons': training.enter_reason.value_counts().to_dict(), 'evaluation_reasons': eval_scored.enter_reason.value_counts().to_dict(),
            'training_common_targets': int(np.isfinite(training[['enter_base_pct', 'wait_base_pct']]).all(axis=1).sum()),
            'evaluation_common_targets': int(np.isfinite(eval_scored[['enter_base_pct', 'wait_base_pct']]).all(axis=1).sum()),
            'evaluation_cost_only_stop_rows': int(eval_scored.cost_only_stop.sum())},
        'integrity': {'registered_inputs_verified': True, 'all_original_clocks_replayed': True,
            'strict_prefix_fitting_no_outcome_inputs': True, 'immutable_current_G': True,
            'wait_target_is_fixed_causal_pi0_not_future_max': True, 'unknowns_never_cash': True},
        'limitations': 'Reused May development. Q_WAIT is value of pi0, not the learned policy or optimal WAIT; policy iteration is not established. Fixed-fit conditional intervals omit unknowns and fitting uncertainty. Next-print cost scenarios do not prove NBBO/size fills. No learned HOLD/EXIT/ADD/REDUCE/allocator or promotion.'}
    result['gate_checks'] = gate(result)
    for suffix, frame in (('training-states', train_scored), ('states', eval_scored), ('chronological-states', chrono), ('episode-decisions', episodes), ('account-orders', pd.concat(orders, ignore_index=True))):
        parquet(frame, output, suffix)
    (output/'request330.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('previous', 'pinned-inputs', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.previous, args.pinned_inputs, args.output_dir)
    print(json.dumps({'fit': result['fit'], 'policies': result['policies'], 'gate_checks': result['gate_checks']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
