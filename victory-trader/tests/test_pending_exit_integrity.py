import json

import numpy as np
import pandas as pd
import pytest

from victory_trader.frozen_entry_second_hold_exit import position_states
from victory_trader.pending_exit_integrity import (
    pending_replay, rebuild_targets, regular_bars, reproduce_legacy, session_limits,
    submission,
)
from victory_trader.frozen_entry_second_hold_exit import POLICIES, replay


def fixture(times=None, prices=None, cap=3600000):
    times = times or [0, 1000, 700000, 701000]
    prices = prices or [10., 10., 10.5, 10.5]
    raw = pd.DataFrame(dict(t=times,o=prices,h=prices,l=prices,c=prices,v=[100]*len(times),n=[10]*len(times)))
    e=dict(trading_day='2026-05-05',ticker='TEST',hot_t=0,entry_t=0,entry_price=10.,entry_setup_score_pct=3.,entry_value_score_pct=2.,entry_turn_score_pct=.4)
    f=position_states(raw,e,cap)
    f['predicted_hold_60s_pct']=1.
    f['predicted_hold_300s_pct']=1.
    return raw,f


def test_clock_timer_in_silence_submits_on_time_and_pays_pending_gap():
    raw,f=fixture()
    for p,t in [('fixed_300s',300000),('fixed_600s',600000),('old_10m_2pct_trailing',600000)]:
        r=pending_replay(f,raw,p,0,4000000)
        assert r['submission_t']==t
        assert r['exit_t']==700000
        assert r['execution_delay_s']==(700000-t)/1000
        assert r['resolved'] and not r['within_3s']


def test_pending_stop_cannot_be_reset_by_later_recovery_or_model_hold():
    raw,f=fixture([0,1000,10000,11000],[10.,9.7,8.,12.])
    r=pending_replay(f,raw,'dynamic_300s',0,4000000)
    assert r['submission_t']==2000 and r['exit_t']==10000
    assert r['exit_reason']=='hard_stop'
    assert r['gross_return_pct']==pytest.approx(-20)
    assert r['base_net_return_pct'] < -20


def test_missing_regular_fill_remains_censored_and_afterhours_not_used():
    raw,f=fixture([0,1000,5000,6000],[10.,10.,10.,10.])
    r=pending_replay(f,raw,'fixed_60s',0,5000)
    assert not r['resolved'] and r['exit_t'] is None
    assert r['order_state']=='RIGHT_CENSORED'
    assert np.isnan(r['base_net_return_pct'])
    assert regular_bars(raw,0,5000).t.tolist()==[0,1000]


def test_pending_fill_after_research_cap_is_counted_before_session_close():
    raw,f=fixture([0,1000,3605000,3606000],[10.,10.,9.,9.])
    r=pending_replay(f,raw,'terminal_60m',0,4000000)
    assert r['submission_t']==3600000
    assert r['exit_t']==3605000 and r['exposure_past_cap_s']==5
    assert r['resolved']


def test_future_mutation_does_not_change_submitted_timer_or_stop():
    raw,f=fixture()
    changed=raw.copy()
    changed.loc[changed.t.ge(700000),['o','c','h','l']]=100.
    _,future=fixture(prices=[10.,10.,100.,100.])
    a=pending_replay(f,raw,'fixed_300s',0,4000000)
    b=pending_replay(future,changed,'fixed_300s',0,4000000)
    assert a['submission_t']==b['submission_t']==300000
    assert a['base_net_return_pct'] != b['base_net_return_pct']


def test_rebuilt_labels_include_delays_without_changing_features():
    raw,f=fixture()
    rebuilt=rebuild_targets(f,raw,4000000)
    assert rebuilt.hold_60s_advantage_pct.iloc[:2].notna().all()
    assert rebuilt.hold_60s_advantage_pct.iloc[2:].isna().all()
    assert rebuilt.hold_60s_label_delay_s.iloc[0] > 600
    assert rebuilt.exit_now_label_delay_s.iloc[1] > 600
    assert rebuilt.entry_price.equals(f.entry_price)


def test_legacy_reproduction_and_strict_json():
    raw,f=fixture()
    original=pd.DataFrame([replay(f,p) for p in POLICIES])
    assert reproduce_legacy(f,original)['return_max_absolute_error']==0
    with pytest.raises(ValueError,match='mismatch'):
        reproduce_legacy(f,original.assign(exit_reason='wrong'))
    r=pending_replay(f,raw,'fixed_300s',0,4000000)
    # JSON document summaries must not contain NaN; row parquet may contain NaN.
    clean={k:v for k,v in r.items() if not isinstance(v,float) or np.isfinite(v)}
    json.dumps(clean,allow_nan=False)


def test_exchange_holiday_and_early_close():
    with pytest.raises(ValueError,match='no regular session'):
        session_limits('2026-06-19')
    opening,closing=session_limits('2026-11-27')
    assert closing-opening==int(3.5*3600000)


def test_timer_precedes_late_model_exit():
    _,f=fixture()
    f.loc[f.decision_t.gt(600000),'predicted_hold_300s_pct']=-10.
    assert submission(f,'fixed_300s',3600000)==(300000,'fixed_horizon')
