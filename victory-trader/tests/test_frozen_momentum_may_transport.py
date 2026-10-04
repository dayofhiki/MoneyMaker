import json

import numpy as np
import pandas as pd
import pytest

from victory_trader.frozen_momentum_may_transport import (
    EVAL_DAYS,PACKAGES,causal_population,evaluate,first_observation,freeze_models,gate,
    identity_hash,observe_and_label,score_available,validate_training_context,
)
from victory_trader.pending_exit_integrity import session_limits
from victory_trader.preentry_momentum_observability import clock_features


def population():
    rows=[]
    for day in EVAL_DAYS:
        opening,_=session_limits(day)
        for i in range(30):
            rows.append({'trading_day':day,'ticker':f'X{i}','t':opening+60000,'bar_start_t':opening,
                         'log_current_price':1.,'minutes_since_open':1.,'minutes_to_close':389.,
                         'return_from_previous_close_pct':2.,'future_profit':i})
    return pd.DataFrame(rows)


def test_sampling_ignores_outcomes_row_order_and_admission_flags():
    f=population()
    before,audit=causal_population(f)
    f['future_profit']=np.arange(len(f))[::-1]
    f['economic_admission_selected']=False
    after,other=causal_population(f.sample(frac=1.,random_state=1))
    pd.testing.assert_frame_equal(before,after)
    assert audit==other
    assert before.sample_hash_byte.lt(64).all()
    assert 'future_profit' not in before and 'economic_admission_selected' not in before
    assert identity_hash('2026-05-11','x0',1)==identity_hash('2026-05-11','X0',1)


def test_population_requires_completed_minute_and_all_fixed_dates():
    f=population()
    f.loc[0,'bar_start_t']+=1
    with pytest.raises(ValueError,match='completed minutes'):
        causal_population(f)
    with pytest.raises(ValueError,match='fixed May population'):
        causal_population(population().loc[lambda f:f.trading_day.ne(EVAL_DAYS[0])])


def bars():
    opening,closing=session_limits(EVAL_DAYS[0])
    t=opening+np.arange(0,200)*1000
    f=pd.DataFrame({'t':t,'o':5.,'h':5.,'l':5.,'c':5.,'v':100.,'n':2.})
    return f,opening,closing


def test_first_snapshot_selected_without_future_label_eligibility_or_three_prints():
    f,opening,closing=bars()
    assert first_observation(f.iloc[:1],opening,closing)==opening+1000
    assert first_observation(f,opening,closing)==opening+1000
    assert first_observation(f.iloc[:0],opening,closing) is None
    assert first_observation(f.assign(t=closing-1000).iloc[:1],closing-1000,closing) is None


def test_later_path_mutation_cannot_change_initial_clock_features():
    f,opening,closing=bars()
    decision=first_observation(f,opening+60000,closing)
    source=pd.DataFrame([{'trading_day':EVAL_DAYS[0],'ticker':'X','decision_t':decision}])
    before=clock_features(source,{(EVAL_DAYS[0],'X'):(f,opening,closing)})
    f.loc[f.t.ge(decision),['o','h','l','c','v','n']]=999.
    after=clock_features(source,{(EVAL_DAYS[0],'X'):(f,opening,closing)})
    pd.testing.assert_frame_equal(before,after)


def test_incomplete_labels_preserve_first_observation_and_acquisition_scope():
    f,opening,closing=bars()
    class Client:
        def second_bars_range(self,ticker,start,end,adjusted):
            assert start==end and str(start)==EVAL_DAYS[0] and adjusted is False
            return {'results':f.iloc[:2].to_dict('records')}
    sample=pd.DataFrame([{'trading_day':EVAL_DAYS[0],'ticker':'X','hot_t':opening}])
    states,raw,audit=observe_and_label(sample,Client())
    assert states.decision_t.iloc[0]==opening+1000 and states.observation_available.iloc[0]
    assert not states.label_complete.iloc[0]
    assert states.missing_reason.iloc[0]=='incomplete_entry_or_terminal_fill_label'
    assert len(raw)==2 and not audit['failures']
    with pytest.raises(ValueError,match='acquisition outside'):
        observe_and_label(sample.assign(trading_day='2026-06-15'),Client())


def test_frozen_fit_uses_only_training_rows_and_rejects_package_changes(monkeypatch):
    train=pd.DataFrame({'trading_day':['2026-05-05','2026-05-06','2026-05-07','2026-05-08'],
                        'ticker':['A','B','C','D'],'hot_t':0,'decision_t':1000,'label_complete':True,
                        'reachable_max_base_net_pct':[0.,5.,0.,5.]})
    monkeypatch.setattr('victory_trader.frozen_momentum_may_transport.validate_states',lambda frame:None)
    seen=[]
    def fit(frame,features,seed):
        seen.append((set(frame.trading_day),features,seed))
        return None,None,.5,{}
    monkeypatch.setattr('victory_trader.frozen_momentum_may_transport.fit_linear',fit)
    previous={'request_id':310,'packages':{k:list(v) for k,v in PACKAGES.items()}}
    models,audits,prior=freeze_models(train,previous)
    assert all(days==set(train.trading_day) and seed==20264605 for days,_,seed in seen)
    assert prior==.5 and set(models)==set(PACKAGES)
    previous['packages']['M']=['future_score']
    with pytest.raises(ValueError,match='package mismatch'):
        freeze_models(train,previous)


def test_zero_observation_or_label_coverage_is_reported_without_fake_probabilities():
    f=pd.DataFrame({'trading_day':[EVAL_DAYS[0]],'ticker':['X'],'hot_t':[0],
                    'observation_available':[False],'label_complete':[False],
                    'reachable_max_base_net_pct':[np.nan],'missing_reason':['no_print']})
    scored=score_available(f,{a:(None,None,.1) for a in PACKAGES},.1)
    assert scored.M_p_net_5.isna().all()
    report=evaluate(scored)
    assert report['complete_coverage']==0 and report['missing_reasons']=={'no_print':1}
    assert set(report['per_day'])==set(EVAL_DAYS)
    assert not all(gate(report).values())
    json.dumps(report,allow_nan=False)


def test_context_semantics_must_match_original_training_identities():
    raw=population().iloc[:2].copy()
    train=raw.rename(columns={'t':'hot_t'}).assign(decision_t=raw.t.to_numpy()+1000)
    audit=validate_training_context(train,raw)
    assert audit['training_episodes']==2 and all(v==0 for v in audit['maximum_absolute_context_errors'].values())
    raw.loc[raw.index[0],'log_current_price']+=.1
    with pytest.raises(ValueError,match='context semantics'):
        validate_training_context(train,raw)
