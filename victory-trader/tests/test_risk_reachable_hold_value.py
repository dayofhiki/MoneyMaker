import numpy as np
import pandas as pd
import pytest

from victory_trader.frozen_entry_second_hold_exit import position_states
from victory_trader.pending_exit_integrity import rebuild_targets
from victory_trader.risk_reachable_hold_value import (
    continuation_targets, mark_reachable, respect_risk_horizons, validate_fit_days, signal,
)


def fixture(times,prices):
    bars=pd.DataFrame(dict(t=times,o=prices,c=prices,h=prices,l=prices,v=[100]*len(times),n=[10]*len(times)))
    e=dict(trading_day='2026-05-05',ticker='TEST',hot_t=0,entry_t=0,entry_price=float(prices[0]),entry_setup_score_pct=3.,entry_value_score_pct=2.,entry_turn_score_pct=.5)
    states=mark_reachable(rebuild_targets(position_states(bars,e,3600000),bars,4000000))
    context={('2026-05-05','TEST',0):(bars,0,4000000)}
    return states,context


def test_first_stop_ends_reachability_and_recovery_never_reopens_it():
    states,_=fixture([0,1000,2000,3000,4000],[10.,9.7,10.,12.,13.])
    assert states.risk_reachable_hold.tolist()==[True,False,False,False,False]
    assert states.forced_stop_t.eq(2000).all()


def test_risk_horizon_pays_gap_at_stop_and_ignores_later_rebound():
    states,ctx=fixture([0,1000,10000,20000,400000],[10.,9.7,8.,12.,14.])
    changed=respect_risk_horizons(states,ctx)
    assert changed.hold_300s_advantage_pct.iloc[0] < -15
    assert changed.hold_300s_advantage_pct.iloc[1:].isna().all()
    assert changed.entry_price.equals(states.entry_price)


def test_learned_downstream_cannot_cancel_forced_pending_stop():
    states,ctx=fixture([0,1000,10000,20000,400000],[10.,9.7,8.,12.,14.])
    states['pi1']=100.
    result=continuation_targets(states,ctx,prediction_column='pi1',output_column='target')
    assert result.target.iloc[0] < -15
    assert result.target.iloc[1:].isna().all()


def test_wait_observes_next_state_then_applies_its_model_exit():
    states,ctx=fixture([0,1000,2000,3000,4000],[10.,10.1,10.2,10.3,10.4])
    states['pi1']=[10.,-1.,10.,10.,10.]
    result=continuation_targets(states,ctx,prediction_column='pi1',output_column='target')
    expected=states.exit_now_pct.iloc[1]-states.exit_now_pct.iloc[0]
    assert result.target.iloc[0]==pytest.approx(expected)


def test_cap_timer_preempts_next_observed_state():
    states,ctx=fixture([0,1000,3700000,3800000],[10.,10.,12.,12.])
    states['pi1']=10.
    result=continuation_targets(states,ctx,prediction_column='pi1',output_column='target')
    # The cap order fills at 3,700s; it never waits for a completed later state.
    assert result.target.iloc[0] > 15
    assert result.target.iloc[1]==pytest.approx(0)


def test_reference_trailing_peak_uses_position_prefix():
    states,ctx=fixture([0,1000,2000,3000,4000],[10.,12.,11.5,11.5,13.])
    result=continuation_targets(states,ctx,prediction_column=None,output_column='target')
    # Following pi0 at state index 2 recognizes the drawdown from prior peak.
    expected=states.exit_now_pct.iloc[2]-states.exit_now_pct.iloc[1]
    assert result.target.iloc[1]==pytest.approx(expected)


def test_cost_only_stop_has_no_learned_hold_rows_but_keeps_position():
    states,ctx=fixture([0,1000,2000],[.6,.6,.6])
    states['entry_price']=.6
    # Recompute observed friction for this fixture's deliberately cheap entry.
    states['observed_net_return_pct']=-3.7
    states=mark_reachable(states)
    result=respect_risk_horizons(states,ctx)
    assert len(result)==len(states)
    assert not result.risk_reachable_hold.any()
    assert result.hold_300s_advantage_pct.isna().all()


def test_nested_day_exclusion_is_enforced():
    validate_fit_days(['2026-05-05','2026-05-06'],excluded=['2026-05-07','2026-05-08'])
    with pytest.raises(ValueError,match='excluded day'):
        validate_fit_days(['2026-05-05','2026-05-07'],excluded=['2026-05-07'])
    with pytest.raises(ValueError):
        validate_fit_days(['2026-06-15'],excluded=[])


def test_rebound_mutation_changes_no_reachable_feature_or_stop_time():
    states,ctx=fixture([0,1000,10000,20000,400000],[10.,9.7,8.,12.,14.])
    mutated,ctx2=fixture([0,1000,10000,20000,400000],[10.,9.7,8.,100.,200.])
    assert states.forced_stop_t.equals(mutated.forced_stop_t)
    a=respect_risk_horizons(states,ctx)
    b=respect_risk_horizons(mutated,ctx2)
    assert a.hold_300s_advantage_pct.iloc[0]==b.hold_300s_advantage_pct.iloc[0]
    assert np.isnan(b.hold_300s_advantage_pct.iloc[-1])


def test_weighted_signal_handles_read_only_pandas_arrays():
    states,_=fixture([0,1000,2000,3000],[10.,10.1,10.2,10.3])
    states["target"]=range(len(states))
    states["prediction"]=range(len(states))
    assert signal(states,"target","prediction")["day_episode_weighted_rank_correlation"]==pytest.approx(1)
