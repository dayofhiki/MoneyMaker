from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from victory_trader import prediction_only_sequence_momentum as exp
from test_elapsed_state_rate_momentum import fixture, held, initial_audit


def training_fixture():
    frame = fixture()
    frame.trading_day = frame.trading_day.replace({'2026-05-05': '2026-05-06', '2026-05-06': '2026-05-07'})
    return frame


@pytest.mark.parametrize('scale', [-25., -2., 1., 12.])
def test_sequence_gradient_multistep_censored_ragged_history(scale):
    frame = training_fixture().drop(index=[9, 14])
    clock = exp.history(frame)
    rng = np.random.default_rng(326)
    x = rng.normal(size=(len(frame), 2))
    labels = np.resize([0, 1, 0, 1], len(frame)).astype(float)
    weights = np.linspace(.1, 1., len(frame))
    weights[2] = 0.
    labels[2] = np.nan
    theta = np.array([.1, -.2, scale-.4, -.1, .3, scale+.2])
    args = (x, clock, np.full(len(frame), .2), labels, weights)
    _, gradient = exp.sequence_loss_gradient(theta, *args)
    numerical = []
    for j in range(len(theta)):
        bump = np.zeros(len(theta))
        bump[j] = 1e-5
        numerical.append((exp.sequence_loss_gradient(theta+bump, *args)[0]-exp.sequence_loss_gradient(theta-bump, *args)[0])/2e-5)
    np.testing.assert_allclose(gradient, numerical, atol=3e-8, rtol=0)


@pytest.mark.parametrize('scale', [-1000., 1000.])
def test_extreme_hazards_are_finite_without_probability_clipping(scale):
    frame = training_fixture()
    x = np.ones((len(frame), 1))
    y = np.resize([0, 1], len(frame))
    value, gradient = exp.sequence_loss_gradient(np.array([.1, scale, -.2, scale]), x, exp.history(frame), np.full(len(frame), .2), y, np.ones(len(frame)))
    assert np.isfinite(value) and np.isfinite(gradient).all()


def test_first_loss_constant_first_features_have_zero_fitted_gradient():
    frame = training_fixture()
    clock = exp.history(frame)
    x = np.arange(len(frame))[:, None].astype(float)
    y, w = np.zeros(len(frame)), np.ones(len(frame))
    initial = np.full(len(frame), .2)
    theta = np.array([.1, -2., -.1, -1.])
    original = exp.sequence_loss_gradient(theta, x, clock, initial, y, w)
    altered = x.copy()
    altered[clock.first] = 1e7
    value, gradient = exp.sequence_loss_gradient(theta, altered, clock, initial, y, w)
    assert value == original[0]
    np.testing.assert_array_equal(gradient, original[1])
    y[clock.first] = 1
    other = exp.sequence_loss_gradient(theta, x, clock, initial, y, w)
    assert other[0]-value == pytest.approx(clock.first.sum()*np.log(4)/w.sum())
    np.testing.assert_array_equal(other[1], gradient)


def test_censored_step_propagates_feature_effect_to_later_target():
    frame = training_fixture().iloc[:5]
    clock, x = exp.history(frame), np.zeros((5, 1))
    initial = np.full(5, .2)
    theta = np.array([1., -2., 0., -2.])
    y, weights = np.array([0, 0, np.nan, 0, 1]), np.array([1., 1., 0., 1., 1.])
    before = exp.sequence_loss_gradient(theta, x, clock, initial, y, weights)[0]
    x[2] = 2.
    after = exp.sequence_loss_gradient(theta, x, clock, initial, y, weights)[0]
    assert abs(after-before) > .001


def test_original_loss_weights_retain_first_and_censored_observations():
    frame = training_fixture()
    frame.loc[2, 'label_complete'] = False
    weights = exp.target_weights(frame)
    assert weights.sum() == pytest.approx(3)
    assert weights[2] == 0
    assert weights[exp.history(frame).first].sum() > 0
    assert len(exp.history(frame).previous) == len(frame)
    manifest, _ = exp.transition_manifest(frame)
    assert [1, 3] not in manifest[['previous_index', 'current_index']].values.tolist()


def fitted():
    frame = training_fixture()
    manifest, _ = exp.transition_manifest(frame)
    bundles, audits = exp.fit_models(frame, manifest, np.full(len(frame), .2))
    return frame, bundles, audits


def test_matched_transforms_and_objective_support_masses():
    frame, bundles, audits = fitted()
    assert bundles['F'][1] is bundles['M'][1] is bundles['S'][1]
    for arm in exp.NEW_ARMS:
        assert bundles[arm][0] is not None
        assert audits[arm]['transform_weight_sum'] == pytest.approx(3)
        assert audits[arm]['transform_states'] == 12
    assert audits['F']['supervision_weight_sum'] == pytest.approx(3)
    assert audits['M']['supervision_weight_sum'] == pytest.approx(6)
    assert audits['S']['complete_mass_including_first'] == pytest.approx(3)
    assert audits['S']['supervision_weight_sum'] == pytest.approx(2.4)
    assert all(audits[a]['normalized_gradient_inf'] <= 1e-6 for a in ('F', 'H', 'A', 'K', 'M'))


def test_static_penalty_matches_minimum_two_rate_penalty():
    beta = np.array([.2, -.7, 1.1])
    rate_penalty = (np.sum((beta/2)**2)+np.sum((-beta/2)**2))/(2*exp.C)
    static_penalty = np.sum(beta**2)/(2*exp.STATIC_PARAMETERS['C'])
    assert rate_penalty == pytest.approx(static_penalty)
    assert exp.STATIC_PARAMETERS['C'] == .2


def test_labels_censors_and_outcomes_absent_from_pure_inference():
    frame, bundles, _ = fitted()
    frame = held(frame)
    original, _ = exp.infer_with_initial(frame, bundles, np.full(len(frame), .2), .1)
    changed = frame.copy()
    changed[exp.CEILING], changed.label_complete, changed.observation_available = np.nan, False, False
    altered, _ = exp.infer_with_initial(changed, bundles, np.full(len(frame), .2), .1)
    bare, _ = exp.infer_with_initial(frame.drop(columns=[exp.CEILING, 'label_complete', 'observation_available']), bundles, np.full(len(frame), .2), .1)
    columns = [c for c in original if c.startswith('R326_')]
    pd.testing.assert_frame_equal(original[columns], altered[columns])
    pd.testing.assert_frame_equal(original[columns], bare[columns])


def test_future_truncation_and_features_do_not_change_present_predictions():
    frame, bundles, _ = fitted()
    initial = np.full(len(frame), .2)
    original, _ = exp.infer_with_initial(frame, bundles, initial, .1)
    changed = frame.copy()
    changed.loc[changed.decision_t.gt(2000), list(set().union(*exp.PACKAGES.values()))] = 1e9
    altered, _ = exp.infer_with_initial(changed, bundles, initial, .1)
    subset = frame.loc[frame.decision_t.le(2000)]
    prefix, _ = exp.infer_with_initial(subset, bundles, np.full(len(subset), .2), .1)
    columns = [c for c in original if c.startswith('R326_')]
    pd.testing.assert_frame_equal(original.loc[subset.index, columns], altered.loc[subset.index, columns])
    pd.testing.assert_frame_equal(original.loc[subset.index, columns], prefix[columns])


def test_no_meta_day_keeps_all_rows_first_and_missing_later():
    frame = training_fixture()
    bundles, audits = exp.fit_models(frame.iloc[:0], pd.DataFrame(), np.array([]))
    result, audit = exp.infer_with_initial(frame, bundles, np.full(len(frame), .2), .1)
    first = exp.history(frame).first
    assert len(result) == len(frame)
    for arm in exp.NEW_ARMS:
        np.testing.assert_array_equal(result[f'R326_{arm}_p_net_5'][first], .2)
        assert result[f'R326_{arm}_p_net_5'][~first].isna().all()
        assert audit['unscored_observations'][arm] == 12
        assert not audits[arm]['optimization_attempted']


def test_manifest_mutation_and_may5_training_rejected():
    frame = training_fixture()
    manifest, _ = exp.transition_manifest(frame)
    manifest.loc[0, 'current_label'] = 1
    with pytest.raises(AssertionError):
        exp.fit_models(frame, manifest, np.full(len(frame), .2))
    old = fixture()
    with pytest.raises(ValueError, match='out-of-time'):
        exp.fit_models(old, exp.transition_manifest(old)[0], np.full(len(old), .2))


def test_prefix_initial_requires_preceding_dates_and_saved_G_replay():
    frame = training_fixture()
    audit = initial_audit(frame)
    audit['fit_days'] = ['2026-05-05']
    frame['G_p_net_5'] = .2
    prior = {'chronological_training': {'folds': {d: {'G': audit} for d in frame.trading_day.unique()}}}
    p, priors, records = exp.prefix_initial(frame, prior)
    np.testing.assert_array_equal(p, .2)
    np.testing.assert_array_equal(priors, .2)
    assert all(v['maximum_saved_G_score_replay_error'] == 0 for v in records.values())
    frame.loc[0, 'G_p_net_5'] = .3
    with pytest.raises(ValueError, match='replay'):
        exp.prefix_initial(frame, prior)
    audit['fit_days'] = ['2026-05-06']
    with pytest.raises(ValueError, match='preceding'):
        exp.prefix_initial(frame, prior)


def test_optimizer_failure_saved_without_retry(monkeypatch):
    frame = training_fixture()
    manifest, _ = exp.transition_manifest(frame)
    calls = []
    def fail(fun, theta, **kwargs):
        calls.append(theta)
        return SimpleNamespace(x=theta, success=False, nit=1, message='failed')
    monkeypatch.setattr(exp, 'minimize', fail)
    model, audit = exp.solve_sequence(np.ones((len(frame), 1)), manifest, exp.history(frame), np.full(len(frame), .2), np.zeros(len(frame)), np.ones(len(frame)))
    assert model is None and not audit['optimization_success'] and len(calls) == 1


def test_boundary_transition_support_never_fabricates_rates():
    frame = training_fixture()
    frame[exp.CEILING] = 0.
    manifest, _ = exp.transition_manifest(frame)
    model, audit = exp.solve_sequence(np.ones((len(frame), 1)), manifest, exp.history(frame), np.full(len(frame), .2), np.zeros(len(frame)), np.ones(len(frame)))
    assert model is None and audit['no_support'] and not audit['optimization_attempted']


def test_sequence_predictions_match_forward_kernel_and_first_exactly():
    frame, bundles, _ = fitted()
    initial = np.linspace(.1, .3, len(frame))
    result, _ = exp.infer_with_initial(frame, bundles, initial, .1)
    clock = exp.history(frame)
    for arm in exp.NEW_ARMS:
        np.testing.assert_array_equal(result[f'R326_{arm}_p_net_5'][clock.first], initial[clock.first])
        if arm == 'S':
            continue
        model, transform = bundles[arm]
        k = exp.predecessor.kernel(model.logits(transform.apply(frame)), clock.gap)
        reference = initial.copy()
        for step in clock.steps:
            reference[step] = k['pi'][step]*k['u'][step]+reference[clock.previous[step]]*k['z'][step]
        np.testing.assert_allclose(result[f'R326_{arm}_p_net_5'], reference, atol=2e-15, rtol=0)
        assert result[f'R326_{arm}_retention'][clock.first].isna().all()
