import numpy as np
import pandas as pd
import pytest

from victory_trader import observed_path_innovation_momentum as exp
from victory_trader.compact_momentum_representation import fit_transform
from test_elapsed_state_rate_momentum import fixture, initial_audit


def training_fixture():
    frame = fixture()
    frame.trading_day = frame.trading_day.replace({'2026-05-05': '2026-05-06', '2026-05-06': '2026-05-07'})
    return frame


def baseline(frame, manifest):
    current = frame.loc[manifest.current_index]
    return {'fit_days': sorted(frame.trading_day.unique()), 'preprocessing': fit_transform(current, exp.CURRENT, manifest.transform_weight.to_numpy()).audit()}


def controls(frame, unsupported=False):
    result = frame[[*exp.KEYS, 'decision_t']].copy()
    first = exp.previous.history(frame).first
    for a in ('F', 'S', 'B', 'B0'):
        p = np.full(len(frame), .2)
        if unsupported and a in ('F', 'S'):
            p[~first] = np.nan
        result[f'R326_{a}_p_net_5'] = p
        result[f'R326_{a}_prior_net_5'] = .2
    result['R326_F_retention'] = np.where(first, np.nan, .3)
    return result


def fitted():
    frame = training_fixture()
    manifest, _ = exp.transition_manifest(frame)
    bundles, audit = exp.fit_models(frame, manifest, np.full(len(frame), .2), baseline(frame, manifest))
    return frame, bundles, audit


def test_emit_before_update_and_own_last_finite_elapsed_time():
    frame = training_fixture().iloc[:4].copy()
    feature = exp.LAG_INPUTS[0]
    frame.decision_t = [1000, 2000, 5000, 10000]
    frame[feature] = [0., 2., np.nan, 8.]
    paths = exp.feature_paths(frame)
    first_ema = (1-np.exp(-1/30))*2
    assert paths['R327_EMA_innovation_'+feature].iloc[1] == 2
    assert paths['R327_ema_before_'+feature].iloc[2] == pytest.approx(first_ema)
    assert np.isnan(paths['R327_EMA_innovation_'+feature].iloc[2])
    assert paths['R327_last_finite_before_t_'+feature].iloc[3] == 2000
    assert paths['R327_history_age_s_'+feature].iloc[3] == 8
    assert paths['R327_EMA_innovation_'+feature].iloc[3] == pytest.approx(8-first_ema)
    assert paths['R327_anchor_innovation_'+feature].iloc[3] == 8
    assert paths['R327_ema_after_'+feature].iloc[3] == pytest.approx(np.exp(-8/30)*first_ema+(1-np.exp(-8/30))*8)


def test_first_available_initializes_after_emitting_missing():
    frame = training_fixture().iloc[:4].copy()
    feature = exp.LAG_INPUTS[0]
    frame[feature] = [np.nan, np.nan, 3., 5.]
    paths = exp.feature_paths(frame)
    assert paths['R327_EMA_innovation_'+feature].iloc[:3].isna().all()
    assert paths['R327_anchor_innovation_'+feature].iloc[:3].isna().all()
    assert paths['R327_EMA_innovation_'+feature].iloc[3] == 2.
    assert paths['R327_anchor_before_'+feature].iloc[3] == 3.


def test_constant_features_keep_zero_innovations_and_episode_reset():
    frame = training_fixture()
    frame.loc[:, list(exp.LAG_INPUTS)] = 2.
    paths = exp.feature_paths(frame)
    first = exp.previous.history(frame).first
    assert paths.loc[first, list(exp.EMA)].isna().all().all()
    np.testing.assert_array_equal(paths.loc[~first, list(exp.EMA)], 0.)
    np.testing.assert_array_equal(paths.loc[~first, list(exp.ANCHOR)], 0.)
    for f in exp.LAG_INPUTS:
        np.testing.assert_allclose(paths['R327_ema_after_'+f], 2., atol=1e-15, rtol=0)


def test_missing_current_preserves_history_and_all_original_clocks():
    frame = training_fixture()
    frame.loc[2, list(exp.LAG_INPUTS)] = np.nan
    frame.loc[2, 'label_complete'] = False
    paths = exp.feature_paths(frame)
    assert len(paths) == len(frame)
    assert paths.loc[2, list(exp.EMA)].isna().all()
    for f in exp.LAG_INPUTS:
        assert paths.loc[2, 'R327_ema_after_'+f] == paths.loc[1, 'R327_ema_after_'+f]
        assert paths.loc[3, 'R327_last_finite_before_t_'+f] == frame.loc[1, 'decision_t']
    assert exp.previous.target_weights(paths).sum() == pytest.approx(3)
    assert exp.previous.target_weights(paths)[2] == 0


def test_feature_paths_are_label_censor_and_future_invariant():
    frame = training_fixture()
    original = exp.feature_paths(frame)
    mutated = frame.copy()
    mutated[exp.CEILING], mutated.label_complete, mutated.observation_available = np.nan, False, False
    altered = exp.feature_paths(mutated)
    bare = exp.feature_paths(frame.drop(columns=[exp.CEILING, 'label_complete', 'observation_available']))
    cols = list(exp.path_columns())
    pd.testing.assert_frame_equal(original[cols], altered[cols])
    pd.testing.assert_frame_equal(original[cols], bare[cols])
    future = frame.copy()
    future.loc[future.decision_t.gt(2000), list(exp.LAG_INPUTS)] = 1e9
    after = exp.feature_paths(future)
    prefix = exp.feature_paths(frame.loc[frame.decision_t.le(2000)])
    pd.testing.assert_frame_equal(original.loc[prefix.index, cols], after.loc[prefix.index, cols])
    pd.testing.assert_frame_equal(original.loc[prefix.index, cols], prefix[cols])


def test_paths_are_idempotent_and_do_not_trust_supplied_history():
    frame = training_fixture()
    paths = exp.feature_paths(frame)
    changed = paths.copy()
    changed.loc[:, list(exp.EMA)] = 1e6
    pd.testing.assert_frame_equal(exp.feature_paths(changed), paths)


def test_shared_blocks_frozen_scope_and_registered_weights():
    frame, bundles, audits = fitted()
    assert bundles['F'][1] is bundles['S'][1]
    assert bundles['F'][1].current is bundles['N'][1].current
    for a in exp.NEW_ARMS:
        assert bundles[a][0] is not None and audits[a]['optimization_success']
        assert audits[a]['transform_weight_sum'] == pytest.approx(3)
        assert audits[a]['preprocessing']['processed_dimensions'] == 90
    assert audits['F']['supervision_weight_sum'] == pytest.approx(3)
    assert audits['S']['supervision_weight_sum'] == pytest.approx(2.4)
    assert audits['S']['complete_mass_including_first'] == pytest.approx(3)
    assert audits['S']['normalized_gradient_inf'] < 1e-6
    assert bundles['F'][0].coefficients.shape == (2, 90)
    manifest, _ = exp.transition_manifest(frame)
    audit = baseline(frame, manifest)
    audit['fit_days'] = ['2026-05-05']
    with pytest.raises(ValueError, match='scope'):
        exp.fit_models(frame, manifest, np.full(len(frame), .2), audit)


def test_frozen_current_transform_mutation_is_rejected():
    frame = training_fixture()
    manifest, _ = exp.transition_manifest(frame)
    audit = baseline(frame, manifest)
    audit['preprocessing']['mean'][0] += .1
    with pytest.raises(AssertionError):
        exp.frozen_current(frame, manifest, audit)


@pytest.mark.parametrize('features', [exp.EMA, exp.ANCHOR])
def test_composed_block_sequence_gradient_matches_finite_differences(features):
    frame = exp.feature_paths(training_fixture().iloc[:5])
    manifest, _ = exp.transition_manifest(frame)
    current = fit_transform(frame.loc[manifest.current_index], exp.CURRENT, manifest.transform_weight.to_numpy())
    extra = fit_transform(frame.loc[manifest.current_index], features, manifest.transform_weight.to_numpy())
    x = exp.Blocks(current, extra).apply(frame)
    theta = np.zeros(2*(x.shape[1]+1))
    theta[x.shape[1]], theta[-1] = -1., 1.
    theta[0], theta[x.shape[1]+2] = .1, -.1
    y, w = np.array([0., 0., np.nan, 1., 0.]), np.array([1., 1., 0., 1., 1.])
    args = (x, exp.previous.history(frame), np.full(len(frame), .2), y, w)
    _, grad = exp.previous.sequence_loss_gradient(theta, *args)
    for j in (0, 76, 83, 90, 91, 167, 174, 181):
        bump = np.zeros_like(theta)
        bump[j] = 1e-5
        numerical = (exp.previous.sequence_loss_gradient(theta+bump, *args)[0]-exp.previous.sequence_loss_gradient(theta-bump, *args)[0])/2e-5
        assert grad[j] == pytest.approx(numerical, abs=3e-8)


def test_inference_no_outcomes_and_future_truncation_invariance():
    frame, bundles, _ = fitted()
    original, _ = exp.infer_with_initial(frame, bundles, np.full(len(frame), .2), .2, controls(frame))
    mutated = frame.copy()
    mutated[exp.CEILING], mutated.label_complete, mutated.observation_available = np.nan, False, False
    altered, _ = exp.infer_with_initial(mutated, bundles, np.full(len(frame), .2), .2, controls(frame))
    bare, _ = exp.infer_with_initial(frame.drop(columns=[exp.CEILING, 'label_complete', 'observation_available']), bundles, np.full(len(frame), .2), .2, controls(frame))
    cols = [c for c in original if c.startswith('R327_')]
    pd.testing.assert_frame_equal(original[cols], altered[cols])
    pd.testing.assert_frame_equal(original[cols], bare[cols])
    subset = frame.loc[frame.decision_t.le(2000)]
    prefix, _ = exp.infer_with_initial(subset, bundles, np.full(len(subset), .2), .2, controls(subset))
    pd.testing.assert_frame_equal(original.loc[prefix.index, cols], prefix[cols])


def test_pinned_controls_and_common_first_are_exact_and_join_requires_clock():
    frame, bundles, _ = fitted()
    saved = controls(frame)
    result, _ = exp.infer_with_initial(frame, bundles, np.full(len(frame), .2), .2, saved)
    first = exp.previous.history(frame).first
    for a in exp.ARMS:
        np.testing.assert_array_equal(result.loc[first, f'R327_{a}_p_net_5'], .2)
    np.testing.assert_array_equal(result.R327_C_p_net_5, saved.R326_F_p_net_5)
    np.testing.assert_array_equal(result.R327_D_p_net_5, saved.R326_S_p_net_5)
    saved.loc[0, 'decision_t'] += 1
    with pytest.raises(AssertionError):
        exp.attach_controls(frame, saved)


def test_unsupported_may6_keeps_rows_first_and_all_later_missing():
    frame = training_fixture()
    bundles, audits = exp.fit_models(frame.iloc[:0], pd.DataFrame(), np.array([]), {})
    result, audit = exp.infer_with_initial(frame, bundles, np.full(len(frame), .2), .2, controls(frame, unsupported=True))
    first = exp.previous.history(frame).first
    for a in ('F', 'N', 'S', 'C', 'D'):
        np.testing.assert_array_equal(result.loc[first, f'R327_{a}_p_net_5'], .2)
        assert result.loc[~first, f'R327_{a}_p_net_5'].isna().all()
        assert audit['unscored_observations'][a] == 12
    assert all(not audits[a]['optimization_attempted'] for a in exp.NEW_ARMS)
    assert len(result) == len(frame)


def test_static_objective_includes_fixed_first_and_matching_penalty():
    frame, bundles, audit = fitted()
    paths = exp.feature_paths(frame)
    weights, first = exp.previous.target_weights(frame), exp.previous.history(frame).first
    model, transform = bundles['S']
    eta = model.decision_function(transform.apply(paths))
    labels = exp.truth(frame[exp.CEILING], 5)
    losses = np.logaddexp(0., eta)-labels*eta
    losses[first] = -np.where(labels[first], np.log(.2), np.log(.8))
    expected = (np.sum(weights*losses)+np.sum(model.coef_[0]**2)/(2*.2))/weights.sum()
    assert audit['S']['normalized_objective_including_first'] == pytest.approx(expected, abs=1e-14)


def test_invalid_features_and_may5_scope_are_rejected():
    frame = training_fixture()
    frame.loc[0, exp.LAG_INPUTS[0]] = np.inf
    with pytest.raises(ValueError, match='infinite'):
        exp.feature_paths(frame)
    old = fixture()
    manifest, _ = exp.transition_manifest(old)
    with pytest.raises(ValueError, match='out-of-time'):
        exp.fit_models(old, manifest, np.full(len(old), .2), baseline(old, manifest))


def test_canonical_initial_wrapper_reconstructs_before_using_pinned_bits():
    frame = training_fixture()
    frame.trading_day = frame.trading_day.replace({'2026-05-06': '2026-05-11', '2026-05-07': '2026-05-12'})
    bundles = {a: (None, None) for a in exp.NEW_ARMS}
    audit = initial_audit(frame)
    saved = controls(frame)
    result, record = exp.infer(frame, bundles, audit, saved)
    assert record['maximum_canonical_G_replay_error'] == 0
    np.testing.assert_array_equal(result.R327_B_p_net_5, saved.R326_B_p_net_5)
    saved.loc[0, 'R326_B_p_net_5'] = .3
    with pytest.raises(ValueError, match='replay'):
        exp.infer(frame, bundles, audit, saved)


def test_registered_gate_checks_memory_and_calibration_failures_separately():
    ms = {m: .6 for m in exp.METRICS} | {'within_episode_evaluable_episodes': 78, 'local_evaluable_episodes': 70, 'prior_brier': .2}
    arms = {a: ms.copy() for a in exp.ARMS}
    arms['F'] |= {m: .8 for m in exp.METRICS}
    arms['F']['brier'] = .1
    comparisons = {f'{a}_minus_{b}': {f'{m}_ci95': ([-.1, -.01] if m == 'brier' else [.01, .1]) for m in exp.METRICS} for a, b in exp.CONTRASTS}
    report = {'arms': arms, 'training_support': {'episodes': 218, 'cells': {c: {'episodes': 33, 'dates': list(exp.CROSSFIT_DAYS[1:])} for c in ('0_to_1', '1_to_0')}},
        'per_day': {d: {'arms': arms, 'positive_episodes': 1} for d in exp.EVAL_DAYS},
        'paired_intervals': {s: {'comparisons': {c: v.copy() for c, v in comparisons.items()}} for s in ('day', 'ticker_day')}}
    assert len(exp.gate(report)) == 36 and all(exp.gate(report).values())
    report['paired_intervals']['day']['comparisons']['F_minus_S']['local_pair_rank_ci95'] = [-.02, .1]
    checks = exp.gate(report)
    assert len(checks) == 36 and sum(checks.values()) == 35
    assert not checks['day_F_minus_S_local_pair_rank_ci_positive']


def test_feature_history_artifact_keeps_scope_and_original_loss_weights():
    frame = training_fixture()
    frame.loc[2, 'label_complete'] = False
    paths = exp.feature_paths(frame)
    table = exp.feature_history_table(paths, paths, paths)
    assert len(table) == 3*len(frame)
    for scope in ('training', 'evaluation', 'chronological'):
        rows = table.loc[table.scope.eq(scope)]
        assert rows.source_index.tolist() == list(frame.index)
        assert rows.original_complete_target_weight.sum() == pytest.approx(3)
        assert rows.loc[rows.source_index.eq(2), 'original_complete_target_weight'].iloc[0] == 0
        assert rows[list(exp.path_columns())].shape[1] == 63
