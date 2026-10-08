import numpy as np
import pandas as pd
import pytest

from victory_trader import current_state_action_policy as exp
from test_observed_path_innovation_momentum import training_fixture


def bars(times, opens=None, closes=None):
    opens = np.ones(len(times))*10 if opens is None else np.asarray(opens, float)
    return pd.DataFrame({'t': times, 'o': opens, 'c': opens if closes is None else closes})


def test_fixed60s_next_print_costs_without_hindsight_exit_selection():
    b = bars([1000, 2000, 3000, 61000, 62000], [10, 12, 9.9, 10.5, 20])
    # The 9.9 reference completes later, and costs alone do not breach2.5% at$10.
    r = exp.enter_outcome(b, 1000, 0, 1000000)
    assert r['enter_resolved'] and not r['stop_triggered']
    assert r['exit_submission_t'] == 61000 and r['exit_reference_price'] == 10.5
    assert r['enter_light_pct'] > r['enter_base_pct'] > r['enter_stress_pct']
    assert r['enter_base_pct'] == pytest.approx(exp.net(10, 10.5))


@pytest.mark.parametrize('entry_delay', [0, 1000, 3000, 4000])
def test_entry_prompt_fill_bound_is_explicit(entry_delay):
    b = bars([1000+entry_delay, 61000+entry_delay])
    r = exp.enter_outcome(b, 1000, 0, 1000000)
    assert r['enter_resolved'] == (entry_delay <= 3000)
    if entry_delay > 3000:
        assert np.isnan(r['enter_base_pct']) and r['enter_reason'] == 'entry_missing_or_late'


def test_deadline_missing_or_late_is_unknown_never_prior_price():
    for times in ([1000, 2000, 50000], [1000, 2000, 65000]):
        r = exp.enter_outcome(bars(times), 1000, 0, 1000000)
        assert not r['enter_resolved'] and np.isnan(r['enter_base_pct'])
        assert r['exit_submission_t'] == 61000
        assert r['enter_reason'] == 'exit_missing_or_late'


def test_stop_uses_completed_close_and_absorbs_before_later_rally():
    b = bars([1000, 2000, 3000, 61000], [10, 9, 8.5, 20], [10, 9, 12, 20])
    r = exp.enter_outcome(b, 1000, 0, 1000000)
    assert r['stop_triggered'] and r['exit_submission_t'] == 3000
    assert r['exit_reference_price'] == 8.5 and r['enter_base_pct'] < 0
    changed = b.copy()
    changed.loc[3, ['o', 'c']] = 200
    assert exp.enter_outcome(changed, 1000, 0, 1000000) == r


def test_wait_target_chooses_next_causal_policy_action_not_best_return():
    f = pd.DataFrame({'trading_day': ['2026-05-06']*4, 'ticker': ['A']*4, 'hot_t': [0]*4,
        'decision_t': [1000, 2000, 3000, 4000], **{f'enter_{s}_pct': [-1., -2., 100., 3.] for s in exp.SCENARIOS}})
    out = exp.attach_wait_targets(f, np.array([False, True, False, True]))
    np.testing.assert_array_equal(out.wait_base_pct, [-2., 3., 3., 0.])
    np.testing.assert_array_equal(out.wait_selected_next_position, [1, 3, 3, -1])
    f.loc[1, 'enter_base_pct'] = np.nan
    assert np.isnan(exp.attach_wait_targets(f, np.array([False, True, False, True])).wait_base_pct.iloc[0])


def fixture():
    frame = training_fixture()
    frame.known_cost_drag_pct = 1.
    controls = frame[[*exp.KEYS, 'decision_t']].copy()
    controls['R328_B_p_net_5'], controls['R328_B_prior_net_5'] = .2, .15
    controls['R328_B_logit'] = np.log(.2/.8)
    controls['R328_gate'] = ~exp.frozen.previous.history(frame).first
    for j in range(7):
        controls[f'R328_S_design_{j}'] = frame[exp.frozen.LAG_INPUTS[j]]
    return frame, controls


def test_common_target_weighted_ridge_and_missing_targets_not_renormalized():
    frame, controls = fixture()
    x = exp.current_inputs(frame, controls)
    frame['enter_base_pct'], frame['wait_base_pct'] = np.linspace(-2, 2, len(frame)), np.linspace(-1, 1, len(frame))
    frame.loc[0, 'wait_base_pct'] = np.nan
    model, audit = exp.fit_values(frame, x)
    assert model.shape == (11, 2)
    assert audit['common_finite_targets'] == len(frame)-1
    assert audit['common_target_original_weight_mass'] == pytest.approx(2.8)
    assert audit['original_observed_weight_mass'] == pytest.approx(3.)
    assert audit['maximum_normal_equation_residual'] <= 1e-8
    z = np.c_[x[1:], np.ones(len(x)-1)]
    w = exp.original_weights(frame)[1:]
    gram = z.T@(w[:, None]*z)+np.diag([10.]*10+[0.])
    expected = np.linalg.solve(gram, z.T@(w[:, None]*frame[['enter_base_pct', 'wait_base_pct']].iloc[1:].to_numpy()))
    np.testing.assert_array_equal(model, expected)


def test_actions_compare_wait_value_and_are_label_and_future_invariant():
    frame, controls = fixture()
    model = np.zeros((11, 2))
    model[-1] = [1., 2.]
    out = exp.score(frame, controls, model)
    assert not out.R330_F_wants_ENTER.any() and out.R330_E_wants_ENTER.all()
    model[-1] = [1., .5]
    out = exp.score(frame, controls, model)
    assert out.R330_F_wants_ENTER.all()
    columns = [c for c in out if c.startswith('R330_')]
    mutated = frame.copy()
    mutated[exp.frozen.CEILING], mutated.label_complete, mutated.observation_available = 1000., False, False
    mutated['enter_base_pct'], mutated['wait_base_pct'] = 10000., -10000.
    pd.testing.assert_frame_equal(out[columns], exp.score(mutated, controls, model)[columns])
    mask = frame.decision_t.le(3000)
    future_controls = controls.copy()
    future_controls.loc[~mask, 'R328_B_logit'] = 1000
    changed = exp.score(frame, future_controls, model)
    prefix = exp.score(frame.loc[mask], controls.loc[mask], model)
    pd.testing.assert_frame_equal(out.loc[mask, columns], changed.loc[mask, columns])
    pd.testing.assert_frame_equal(out.loc[mask, columns], prefix[columns])


def test_unsupported_value_head_keeps_missing_and_cost_veto_is_causal():
    frame, controls = fixture()
    for j in range(7):
        controls[f'R328_S_design_{j}'] = np.nan
    out = exp.score(frame, controls, None)
    assert out.R330_Q_ENTER.isna().all() and not out.R330_F_wants_ENTER.any()
    assert out.R330_G_wants_ENTER.all()
    frame.known_cost_drag_pct = 2.5
    assert not exp.score(frame, controls, None).R330_G_wants_ENTER.any()


def account_fixture(rows):
    result = []
    for ticker, hot, t, priority, exit_t, resolved in rows:
        result.append({'trading_day': '2026-05-11', 'ticker': ticker, 'hot_t': hot, 'decision_t': t,
            'R330_F_wants_ENTER': True, 'R330_F_priority': priority, 'entry_fill_t': t,
            'exit_fill_t': exit_t, 'enter_resolved': resolved, **{f'enter_{s}_pct': 1. for s in exp.SCENARIOS}})
    return pd.DataFrame(result)


def test_account_simultaneous_ranking_capacity_wait_reassessment_and_release():
    f = account_fixture([('A', 0, 1000, 3, 4000, True), ('B', 0, 1000, 2, 6000, True),
        ('C', 0, 1000, 1, 7000, True), ('C', 0, 5000, 1, 8000, True)])
    summary, orders = exp.account_replay(f, 'F')
    assert orders.ticker.tolist() == ['A', 'B', 'C']
    assert orders.submission_t.tolist() == [1000, 1000, 5000]
    day = summary['days']['2026-05-11']
    assert day['capacity_WAIT_decisions'] == 1 and day['maximum_simultaneous_reserved_slots'] == 2
    assert day['base_return_pct'] == pytest.approx(.3)
    assert summary['full_marked_account_drawdown_pct'] is None


def test_account_same_ticker_unknown_blocks_slot_and_full_account_claim():
    f = account_fixture([('A', 0, 1000, 3, np.nan, False), ('A', 1, 5000, 4, 8000, True),
        ('B', 0, 5000, 2, 8000, True), ('C', 0, 5000, 1, 9000, True)])
    summary, orders = exp.account_replay(f, 'F')
    assert orders.ticker.tolist() == ['A', 'B']
    day = summary['days']['2026-05-11']
    assert day['same_ticker_WAIT_decisions'] == 1 and day['capacity_WAIT_decisions'] == 1
    assert not summary['all_days_complete'] and day['base_return_pct'] is None
    assert summary['aggregate_daily_reset_base_return_pct'] is None
    assert day['resolved_leg_only_base_return_pct'] == pytest.approx(.1)


def test_episode_censoring_preserved_and_no_observation_cash_retained():
    frame, controls = fixture()
    out = exp.score(frame, controls, None)
    out['enter_resolved'] = False
    for s in exp.SCENARIOS:
        out[f'enter_{s}_pct'] = np.nan
    ledger = pd.concat([frame[exp.KEYS].drop_duplicates(), pd.DataFrame([{'trading_day': '2026-05-07', 'ticker': 'NONE', 'hot_t': 0}])], ignore_index=True)
    ep = exp.episode_replay(out, ledger)
    assert len(ep) == 4
    assert ep.F_resolved.all() and ep.F_base_pct.eq(0).all()
    assert not ep.G_resolved.iloc[:3].any() and ep.G_base_pct.iloc[:3].isna().all()
    assert ep.G_base_pct.iloc[3] == 0 and ep.G_resolved.iloc[3]
    metrics = exp.policy_metrics(ep, 'G')
    assert metrics['full_population_base_mean_pct'] is None
    assert metrics['resolved_only_base_mean_pct'] == 0.


def test_clock_replay_uses_original_complete_bar_observations():
    opening, closing = exp.session_limits('2026-05-11')
    hot = opening+100000
    b = bars([hot, hot+1000, hot+400000])
    states = pd.DataFrame({'trading_day': ['2026-05-11']*2, 'ticker': ['A']*2, 'hot_t': [hot]*2,
        'decision_t': [hot+1000, hot+2000]})
    exp.verify_clocks(states, states[exp.KEYS].drop_duplicates(), {('2026-05-11', 'A'): b})
    with pytest.raises(AssertionError):
        exp.verify_clocks(states.iloc[:1], states[exp.KEYS].drop_duplicates(), {('2026-05-11', 'A'): b})
    assert closing > hot
