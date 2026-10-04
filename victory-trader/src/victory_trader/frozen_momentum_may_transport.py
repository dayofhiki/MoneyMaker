"""R311: frozen linear models on outcome-independent sampled May HOT cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .clock_momentum_increment import BACKGROUND, CONTROL, MOMENTUM, within_day_auc
from .compact_momentum_representation import fit_linear, predict_linear
from .config import load_settings
from .feasible_upside_observability import CEILING, truth
from .frozen_entry_second_hold_exit import KEYS
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .massive_client import MassiveClient
from .pending_exit_integrity import regular_bars, session_limits
from .preentry_momentum_observability import clock_features, entry_labels, metrics, snapshots, validate_states
from .risk_reachable_hold_value import validate_fit_days
from .second_path_attention_probe import SECOND_CACHE_DIR, _second_frame

REQUEST_ID = 311
EVAL_DAYS = ('2026-05-11','2026-05-12','2026-05-13','2026-05-14','2026-05-15','2026-05-18','2026-05-19','2026-05-20')
ARMS = ('N','Q','M')
PACKAGES = {'N': BACKGROUND, 'Q': CONTROL, 'M': MOMENTUM}
POPULATION_COLUMNS = ('trading_day','ticker','t','bar_start_t','log_current_price','minutes_since_open','minutes_to_close','return_from_previous_close_pct')


def identity_hash(day: str, ticker: str, hot_t: int) -> int:
    value = f'R311|{day}|{ticker.upper()}|{hot_t}'
    return hashlib.sha256(value.encode('utf-8')).digest()[0]


def causal_population(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = source[list(POPULATION_COLUMNS)].copy()
    frame = frame.loc[frame.trading_day.astype(str).isin(EVAL_DAYS)].copy()
    frame['ticker'] = frame.ticker.astype(str).str.upper()
    frame['hot_t'] = frame.t.astype('int64')
    if set(frame.trading_day.astype(str)) != set(EVAL_DAYS) or frame.duplicated(KEYS).any():
        raise ValueError('complete unique fixed May population required')
    if not frame.t.sub(frame.bar_start_t).eq(60000).all():
        raise ValueError('HOT inputs must reference completed minutes')
    frame['sample_hash_byte'] = [identity_hash(str(r.trading_day), str(r.ticker), int(r.hot_t)) for r in frame.itertuples()]
    selected = frame.loc[frame.sample_hash_byte.lt(64)].sort_values(KEYS).reset_index(drop=True)
    audit = {'population_episodes': len(frame), 'sampled_episodes': len(selected),
             'by_day': {day: {'population': int(frame.trading_day.eq(day).sum()), 'sampled': int(selected.trading_day.eq(day).sum())} for day in EVAL_DAYS},
             'selection': "SHA256('R311|day|uppercase_ticker|HOT_ms').digest()[0]<64",
             'population_shift': 'Full validated-attention HOT sample versus training rich setup shortlist'}
    return selected, audit



def validate_training_context(train: pd.DataFrame, population_source: pd.DataFrame) -> dict:
    fields = list(BACKGROUND[:-1])
    reference = population_source[['trading_day','ticker','t',*fields]].rename(columns={'t':'hot_t'})
    first = snapshots(train,0)
    joined = first[KEYS+fields].merge(reference,on=KEYS,how='left',suffixes=('_training','_source'),indicator=True,validate='one_to_one')
    if not joined['_merge'].eq('both').all():
        raise ValueError('training identities absent from source population')
    errors = {}
    for field in fields:
        a,b = joined[field+'_training'].to_numpy(float),joined[field+'_source'].to_numpy(float)
        if not np.allclose(a,b,atol=1e-9,rtol=0,equal_nan=True):
            raise ValueError('training/source context semantics mismatch')
        finite = np.isfinite(a)&np.isfinite(b)
        errors[field] = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
    return {'training_episodes':len(first),'maximum_absolute_context_errors':errors}


def first_observation(bars: pd.DataFrame, hot_t: int, closing: int) -> int | None:
    available = bars.loc[bars.t.ge(hot_t) & bars.t.lt(hot_t+300000)]
    if available.empty:
        return None
    decision = int(available.t.min())+1000
    return decision if decision < min(hot_t+3600000, closing) else None


def freeze_models(train: pd.DataFrame, previous: dict) -> tuple[dict, dict, float]:
    validate_states(train)
    if previous.get('request_id') != 310 or not train.label_complete.all() or not np.isfinite(train[CEILING]).all():
        raise ValueError('complete original R310 training states required')
    validate_fit_days(sorted(train.trading_day.unique()), excluded=EVAL_DAYS)
    models, audits = {}, {}
    for arm, features in PACKAGES.items():
        if tuple(previous['packages'][arm]) != features:
            raise ValueError('frozen R310 input package mismatch')
        model, transform, prior, audit = fit_linear(train, features, 20264605)
        models[arm] = (model, transform, prior)
        audits[arm] = audit
    first = snapshots(train, 0)
    first_prior = float(np.average(truth(first[CEILING],5), weights=_episode_day_weights(first)))
    return models, audits, first_prior


def observe_and_label(population: pd.DataFrame, client) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not set(population.trading_day.astype(str)).issubset(EVAL_DAYS) or population.duplicated(KEYS).any():
        raise ValueError('acquisition outside fixed May identities forbidden')
    rows, raw_parts, failures = [], [], []
    for i, raw in enumerate(population.to_dict('records')):
        day, ticker, hot = str(raw['trading_day']), str(raw['ticker']), int(raw['hot_t'])
        opening, closing = session_limits(day)
        row = {**raw, 'decision_t': np.nan, 'observation_available': False, 'label_complete': False, CEILING: np.nan, 'missing_reason': None}
        try:
            bars = _second_frame(client.second_bars_range(ticker,date.fromisoformat(day),date.fromisoformat(day),adjusted=False))
            bars = regular_bars(bars, opening, closing)
        except Exception as error:
            failures.append({'trading_day':day,'ticker':ticker,'error_type':type(error).__name__})
            row['missing_reason'] = 'fetch_or_raw_validation_failure'
            rows.append(row)
            continue
        if not bars.empty:
            raw_parts.append(bars.assign(trading_day=day,ticker=ticker))
        decision = first_observation(bars,hot,closing)
        if decision is None:
            row['missing_reason'] = 'no_eligible_first_completed_regular_print'
            rows.append(row)
            continue
        row['decision_t'] = decision
        row['observation_available'] = True
        context = {(day,ticker):(bars,opening,closing)}
        try:
            features = clock_features(pd.DataFrame([row]),context).iloc[0].to_dict()
        except ValueError:
            row['observation_available'] = False
            row['missing_reason'] = 'invalid_causal_activity_inputs'
            rows.append(row)
            continue
        row.update(features)
        row.update(entry_labels(bars,decision,min(hot+3600000,closing)))
        if not row['label_complete']:
            row['missing_reason'] = 'incomplete_entry_or_terminal_fill_label'
        rows.append(row)
        if (i+1)%50 == 0:
            print(f'R311 fixed May acquisition {i+1}/{len(population)} identities',flush=True)
    raw = pd.concat(raw_parts,ignore_index=True) if raw_parts else pd.DataFrame(columns=['t','o','h','l','c','v','n','trading_day','ticker'])
    return pd.DataFrame(rows),raw,{'failures':failures,'raw_regular_rows':len(raw)}


def score_available(frame: pd.DataFrame, models: dict, first_prior: float) -> pd.DataFrame:
    result = frame.copy()
    available = result.observation_available.astype(bool)
    for arm,(model,transform,prior) in models.items():
        result[f'{arm}_p_net_5'] = np.nan
        if available.any():
            result.loc[available,f'{arm}_p_net_5'] = predict_linear(result.loc[available],model,transform,prior)
        result[f'{arm}_prior_net_5'] = prior
    result['first_training_prior'] = first_prior
    return result


def rank_report(frame: pd.DataFrame, arm: str) -> dict:
    report = metrics(frame,arm,5)
    report['within_day_auc'] = within_day_auc(frame,arm)
    return report


def transport_intervals(frame: pd.DataFrame) -> dict:
    contrasts = (('M','Q'),('Q','N'),('M','N'))
    y, w = truth(frame[CEILING],5),_episode_day_weights(frame)
    p = {a:frame[f'{a}_p_net_5'].to_numpy(float) for a in ARMS}
    output = {}
    for i,columns in enumerate((['trading_day'],['trading_day','ticker'])):
        groups = list(frame.groupby(columns,sort=False).indices.values())
        values = {f'{a}_minus_{b}':{'auc':[],'within_day_auc':[],'brier':[]} for a,b in contrasts}
        rng = np.random.default_rng(20264650+i)
        for _ in range(1000):
            indices = np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
            sample = frame.iloc[indices]
            both = len(np.unique(y[indices])) == 2
            aucs = {a:roc_auc_score(y[indices],p[a][indices],sample_weight=w[indices]) for a in ARMS} if both else {}
            within = {a:within_day_auc(sample,a,w[indices]) for a in ARMS}
            for a,b in contrasts:
                v = values[f'{a}_minus_{b}']
                if both:
                    v['auc'].append(float(aucs[a]-aucs[b]))
                if within[a] is not None:
                    v['within_day_auc'].append(float(within[a]-within[b]))
                v['brier'].append(float(np.average((y[indices]-p[a][indices])**2-(y[indices]-p[b][indices])**2,weights=w[indices])))
        output['day' if i==0 else 'ticker_day'] = {'clusters':len(groups),'comparisons':{
            name: {f'{metric}_ci95':np.quantile(v,[.025,.975]).tolist() if v else None for metric,v in report.items()} |
                  {f'{metric}_valid_draws':len(v) for metric,v in report.items()} for name,report in values.items()}}
    return output


def evaluate(frame: pd.DataFrame) -> dict:
    labeled = frame.loc[frame.observation_available & frame.label_complete & np.isfinite(frame[CEILING])].copy()
    per_day = {}
    for day in EVAL_DAYS:
        group = labeled.loc[labeled.trading_day.eq(day)]
        per_day[day] = {'sampled':int(frame.trading_day.eq(day).sum()),'labeled':len(group),
                        'positive_episodes':int(truth(group[CEILING],5).sum()),'arms':{a:rank_report(group,a) for a in ARMS}}
    report = {'sampled_episodes':len(frame),'observed_episodes':int(frame.observation_available.sum()),'labeled_episodes':len(labeled),
        'complete_coverage':len(labeled)/len(frame),'missing_reasons':frame.missing_reason.fillna('complete').value_counts().to_dict(),
        'arms':{a:rank_report(labeled,a) for a in ARMS},
        'per_day':per_day,
        'first_training_prior_brier':float(np.average((truth(labeled[CEILING],5)-labeled.first_training_prior.to_numpy())**2,weights=_episode_day_weights(labeled))) if len(labeled) else None,
        'paired_intervals':transport_intervals(labeled) if len(labeled) else {}}
    return report


def gate(report: dict) -> dict:
    m,q = report['arms']['M'],report['arms']['Q']
    evaluable = [v for v in report['per_day'].values() if v['arms']['Q']['auc'] is not None]
    ci = report.get('paired_intervals',{}).get('ticker_day',{}).get('comparisons',{}).get('M_minus_Q',{}).get('within_day_auc_ci95')
    def compare(a,b):
        return a is not None and b is not None and a>b
    return {'coverage_at_least90pct':report['complete_coverage']>=.9,
        'at_least20_positive_episodes':m.get('positive_episodes',0)>=20,
        'at_least4_positive_days':sum(v['positive_episodes']>0 for v in report['per_day'].values())>=4,
        'M_within_day_auc_beats_Q':compare(m['within_day_auc'],q['within_day_auc']),
        'M_pooled_auc_beats_Q':compare(m['auc'],q['auc']),'M_ap_beats_Q':compare(m['ap'],q['ap']),
        'M_brier_beats_Q':compare(q.get('brier'),m.get('brier')),
        'M_brier_beats_training_first_prior':compare(report['first_training_prior_brier'],m.get('brier')),
        'M_auc_improves_majority_evaluable_days':bool(evaluable) and sum(v['arms']['M']['auc']-v['arms']['Q']['auc']>1e-9 for v in evaluable)>len(evaluable)/2,
        'ticker_day_M_within_day_gain_ci_positive':ci is not None and ci[0]>0}


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ('training-states','training-summary','population','population-reference','output','states-output','raw-output','cohort-output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    train = pd.read_parquet(args.training_states)
    models,audits,first_prior = freeze_models(train,json.loads(args.training_summary.read_text()))
    candidate = pd.read_parquet(args.population,columns=list(POPULATION_COLUMNS))
    reference = pd.read_parquet(args.population_reference,columns=list(POPULATION_COLUMNS))
    context_audit = validate_training_context(train,candidate)
    population,pop_audit = causal_population(candidate)
    ref_population,_ = causal_population(reference)
    pd.testing.assert_frame_equal(population,ref_population,check_dtype=False,atol=1e-12,rtol=1e-12)
    args.cohort_output.parent.mkdir(parents=True,exist_ok=True)
    population.to_parquet(args.cohort_output,index=False,compression='zstd')
    print(f'R311 froze {len(population)} sampled identities before acquisition',flush=True)
    settings = load_settings()
    client = MassiveClient(settings.massive_api_key,cache_dir=SECOND_CACHE_DIR,request_interval_seconds=.02)
    states,raw,acquisition = observe_and_label(population,client)
    scored = score_available(states,models,first_prior)
    report = evaluate(scored)
    checks = gate(report)
    result = {'request_id':REQUEST_ID,'development_only':True,'promotion_eligible':False,'June_HOLD_opened':False,'final_July_August_opened':False,
        'entries_retuned':False,'reference_R310_run':37218791436,'population_R233_run':36251239308,'population_R163_reference_run':36001945920,
        'train_days':list(CROSSFIT_DAYS),'eval_days':list(EVAL_DAYS),'seed':20264605,'packages':{a:list(v) for a,v in PACKAGES.items()},
        'training':audits,'training_first_prior':first_prior,'training_context_alignment':context_audit,'population':pop_audit,
        'acquisition':{**acquisition,'client_stats':client.stats.to_dict()},'transport':report,'checks':checks,'research_gate_pass':all(checks.values()),
        'limitations':'Previously explored May development, broader HOT sample versus setup-conditioned training97; first canonical observation not future-label filtered, unlike inherited historical support. Labels incomplete remain reported. One model fit on May5-8 only; no eval outcome fit, threshold/policy calibration, feature/seed/date/sampling selection or profit claim.'}
    for path in (args.output,args.states_output,args.raw_output):
        path.parent.mkdir(parents=True,exist_ok=True)
    scored.to_parquet(args.states_output,index=False,compression='zstd')
    raw.to_parquet(args.raw_output,index=False,compression='zstd')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'population':pop_audit,'coverage':{k:report[k] for k in ('observed_episodes','labeled_episodes','complete_coverage','missing_reasons')},'checks':checks,'arms':report['arms']},indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
