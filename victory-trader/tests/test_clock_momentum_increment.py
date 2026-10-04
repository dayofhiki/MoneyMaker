import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from victory_trader.clock_momentum_increment import (
    BACKGROUND, CONTROL, COVERAGE, MOMENTUM, SIGNAL, increment_crossfit, within_day_auc,
)
from victory_trader.feasible_upside_observability import CEILING
from victory_trader.lagged_minute_context import CROSSFIT_DAYS
from victory_trader.preentry_momentum_observability import CLOCK_FEATURES


def pairs():
    return pd.DataFrame({'trading_day': ['2026-05-05']*2+['2026-05-06']*3,
                         'ticker': ['A','B','C','D','E'], 'hot_t': 0,
                         CEILING: [5.,0.,5.,0.,0.], 'Q_p_net_5': [.4,.3,.2,.8,.1]})


def test_within_day_auc_matches_weighted_pairs_and_is_invariant_to_day_offsets():
    f = pairs()
    w = np.array([1.,1.,2.,3.,4.])
    assert within_day_auc(f, 'Q', w) == pytest.approx(9/15)
    before = within_day_auc(f, 'Q')
    pooled = roc_auc_score(f[CEILING].ge(5), f.Q_p_net_5)
    f.loc[f.trading_day.eq('2026-05-05'), 'Q_p_net_5'] += .5
    assert within_day_auc(f, 'Q') == pytest.approx(before)
    assert roc_auc_score(f[CEILING].ge(5), f.Q_p_net_5) != pooled
    f['Q_p_net_5'] = .25
    assert within_day_auc(f, 'Q', w) == pytest.approx(.5)


def test_between_day_only_discrimination_has_no_within_day_endpoint():
    f = pairs()
    f.loc[f.trading_day.eq('2026-05-05'), CEILING] = 5.
    f.loc[f.trading_day.eq('2026-05-06'), CEILING] = 0.
    f['Q_p_net_5'] = np.where(f[CEILING].ge(5), .9, .1)
    assert roc_auc_score(f[CEILING].ge(5), f.Q_p_net_5) == 1.
    assert within_day_auc(f, 'Q') is None


def test_only_original_episode_snapshots_used_unless_explicit_bootstrap_weights():
    f = pairs()
    repeated = pd.concat([f,f],ignore_index=True)
    with pytest.raises(ValueError, match='one original snapshot'):
        within_day_auc(repeated, 'Q')
    assert within_day_auc(repeated, 'Q', np.ones(len(repeated))) == pytest.approx(within_day_auc(f,'Q',np.ones(len(f))))
    with pytest.raises(ValueError, match='weights'):
        within_day_auc(f, 'Q', np.zeros(len(f)))
    f.loc[0,'Q_p_net_5'] = np.nan
    with pytest.raises(ValueError, match='probabilities'):
        within_day_auc(f, 'Q')


def test_fixed_signal_package_excludes_background_and_coverage():
    assert [len(BACKGROUND),len(CONTROL),len(SIGNAL),len(MOMENTUM)] == [5,9,19,28]
    assert set(SIGNAL).isdisjoint(CONTROL)
    assert set(SIGNAL)|set(COVERAGE)|{'known_cost_drag_pct'} == set(CLOCK_FEATURES)
    assert len(set(MOMENTUM)) == len(MOMENTUM)


def source():
    rows=[]
    for day in CROSSFIT_DAYS:
        for episode in range(6):
            for seconds in (0,20,120):
                row={'trading_day':day,'ticker':str(episode),'hot_t':0,'decision_t':seconds*1000,
                     CEILING:10. if episode%3==0 else 0., 'W_p_net_5':.25}
                row.update({name:float(episode+seconds/120) for name in MOMENTUM})
                rows.append(row)
    return pd.DataFrame(rows)


def test_held_labels_do_not_change_own_scores_and_training_days_are_excluded(monkeypatch):
    calls=[]
    def fit(train, features, seed):
        calls.append((set(train.trading_day),tuple(features)))
        prior=.25 if features==MOMENTUM else float(train[CEILING].ge(5).mean())
        return None,None,prior,{'train_rows':len(train)}
    monkeypatch.setattr('victory_trader.clock_momentum_increment.fit_linear',fit)
    monkeypatch.setattr('victory_trader.clock_momentum_increment.predict_linear',lambda frame,model,transform,prior:np.full(len(frame),prior))
    f=source()
    before,audit_before=increment_crossfit(f,MOMENTUM)
    f.loc[f.trading_day.eq(CROSSFIT_DAYS[0]),CEILING]=1e9
    after,audit_after=increment_crossfit(f,MOMENTUM)
    held=before.trading_day.eq(CROSSFIT_DAYS[0])
    pd.testing.assert_frame_equal(before.loc[held,['N_p_net_5','Q_p_net_5','M_p_net_5','W_p_net_5']],after.loc[held,['N_p_net_5','Q_p_net_5','M_p_net_5','W_p_net_5']])
    assert audit_before[CROSSFIT_DAYS[0]]==audit_after[CROSSFIT_DAYS[0]]
    assert calls[0][0]==set(CROSSFIT_DAYS[1:])
    assert calls[0][1]==BACKGROUND and calls[1][1]==CONTROL and calls[2][1]==MOMENTUM
    for day,audit in audit_before.items():
        assert day not in audit['fit_days'] and audit['maximum_saved_W_error']==0.


def test_replay_mismatch_future_day_and_noncausal_subset_rejected(monkeypatch):
    monkeypatch.setattr('victory_trader.clock_momentum_increment.fit_linear',lambda *args:(None,None,.25,{}))
    monkeypatch.setattr('victory_trader.clock_momentum_increment.predict_linear',lambda frame,*args:np.full(len(frame),.25))
    f=source()
    f.loc[0,'W_p_net_5']=.9
    with pytest.raises(ValueError,match='score mismatch'):
        increment_crossfit(f,MOMENTUM)
    f.loc[0,'trading_day']='2026-06-15'
    with pytest.raises(ValueError,match='May5-8'):
        increment_crossfit(f,MOMENTUM)
    with pytest.raises(ValueError,match='causal subsets'):
        increment_crossfit(source(),('future_label',))
