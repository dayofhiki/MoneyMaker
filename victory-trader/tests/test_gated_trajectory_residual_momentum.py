from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from victory_trader import gated_trajectory_residual_momentum as exp
from test_observed_path_innovation_momentum import training_fixture


def fitted():
    frame = training_fixture()
    manifest, _ = exp.transition_manifest(frame)
    g = np.linspace(.11, .31, len(frame))
    bundles, audits = exp.fit_models(frame, manifest, g)
    return frame, g, bundles, audits


def test_gate_requires_all_current_and_past_features_and_is_label_free():
    frame = training_fixture()
    frame.loc[1, exp.LAG_INPUTS[0]] = np.nan
    paths = exp.old.feature_paths(frame)
    active = exp.gate_mask(paths)
    assert not active[0] and not active[1] and active[2]
    altered = frame.drop(columns=[exp.CEILING, 'label_complete', 'observation_available'])
    np.testing.assert_array_equal(exp.gate_mask(exp.old.feature_paths(altered)), active)


@pytest.mark.parametrize('arm', ('F', 'N', 'S'))
def test_numeric_only_design_and_exact_zero_inactive_rows(arm):
    frame, _, bundles, audit = fitted()
    paths = exp.old.feature_paths(frame)
    active = exp.gate_mask(paths)
    x = exp.design(paths, active, arm, bundles[arm][1])
    assert x.shape == (len(frame), 7)
    np.testing.assert_array_equal(x[~active], 0.)
    np.testing.assert_array_equal(x[active], bundles[arm][1].apply(paths.loc[active])[:, :7])
    assert audit[arm]['transform_states'] == 12
    assert audit[arm]['transform_weight_sum'] == pytest.approx(3.)
    assert audit[arm]['supervision_weight_sum'] == pytest.approx(3.)
    assert audit[arm]['active_weight_sum'] == pytest.approx(2.4)
    assert len(audit[arm]['coefficients']) == 7
    assert audit[arm]['intercept'] is False


def test_mask_only_model_has_one_penalized_coefficient():
    frame, _, bundles, audits = fitted()
    paths = exp.old.feature_paths(frame)
    active = exp.gate_mask(paths)
    x = exp.design(paths, active, 'M', None)
    np.testing.assert_array_equal(x[:, 0], active.astype(float))
    assert bundles['M'][1] is None and bundles['M'][0].shape == (1,)
    assert audits['M']['C'] == .1 and audits['M']['normalized_gradient_inf'] <= 1e-6


def test_offset_objective_gradient_original_constant_mass_and_censored_nan():
    x = np.array([[0., 0.], [1., -.3], [100., 100.], [-.2, 2.]])
    offset = np.array([-2., .1, 3., -1.])
    y, weights = np.array([1., 0., np.nan, 1.]), np.array([2., 1., 0., .5])
    beta = np.array([.3, -.1])
    value, gradient = exp.offset_loss_gradient(beta, x, offset, y, weights)
    selected = weights > 0
    eta = offset[selected]+x[selected]@beta
    expected = (np.sum(weights[selected]*(np.logaddexp(0., eta)-y[selected]*eta))+np.sum(beta**2)/(2*.1))/3.5
    assert value == pytest.approx(expected)
    for j in range(2):
        bump = np.zeros(2)
        bump[j] = 1e-5
        derivative = (exp.offset_loss_gradient(beta+bump, x, offset, y, weights)[0]-exp.offset_loss_gradient(beta-bump, x, offset, y, weights)[0])/2e-5
        assert gradient[j] == pytest.approx(derivative, abs=1e-9)
    changed_y = y.copy()
    changed_y[0] = 0.
    changed_value, changed_gradient = exp.offset_loss_gradient(beta, x, offset, changed_y, weights)
    np.testing.assert_array_equal(changed_gradient, gradient)
    assert changed_value != value


def test_extreme_offset_objective_remains_finite():
    x = np.array([[0.], [1.], [-1.]])
    value, grad = exp.offset_loss_gradient(np.array([.2]), x, np.array([-700., 700., -700.]), np.array([1., 0., 1.]), np.ones(3))
    assert np.isfinite(value) and np.isfinite(grad).all()


def test_gate0_and_zero_coefficient_predictions_copy_canonical_bits():
    frame, g, bundles, _ = fitted()
    scored, audit = exp.infer_with_initial(frame, bundles, g, .2)
    active = scored.R328_gate.to_numpy()
    for a in exp.NEW_ARMS:
        np.testing.assert_array_equal(scored.loc[~active, f'R328_{a}_p_net_5'], g[~active])
        assert audit['unscored_observations'][a] == 0
    null = {a: (np.zeros(1 if a == 'M' else 7), t) for a, (_, t) in bundles.items()}
    copied, _ = exp.infer_with_initial(frame, null, g, .2)
    for a in exp.NEW_ARMS:
        np.testing.assert_array_equal(copied[f'R328_{a}_p_net_5'], g)


def test_inference_label_censor_future_and_prefix_invariance():
    frame, g, bundles, _ = fitted()
    original, _ = exp.infer_with_initial(frame, bundles, g, .2)
    columns = [c for c in original if c.startswith(('R328_', 'R327_'))]
    mutated = frame.copy()
    mutated[exp.CEILING], mutated.label_complete, mutated.observation_available = np.nan, False, False
    changed, _ = exp.infer_with_initial(mutated, bundles, g, .2)
    bare, _ = exp.infer_with_initial(frame.drop(columns=[exp.CEILING, 'label_complete', 'observation_available']), bundles, g, .2)
    pd.testing.assert_frame_equal(original[columns], changed[columns])
    pd.testing.assert_frame_equal(original[columns], bare[columns])
    mask = frame.decision_t.le(2000).to_numpy()
    future = frame.copy()
    future.loc[~mask, list(exp.LAG_INPUTS)] = 1e8
    changed, _ = exp.infer_with_initial(future, bundles, g, .2)
    prefix, _ = exp.infer_with_initial(frame.loc[mask], bundles, g[mask], .2)
    pd.testing.assert_frame_equal(original.loc[mask, columns], changed.loc[mask, columns])
    pd.testing.assert_frame_equal(original.loc[mask, columns], prefix[columns])


def test_unsupported_fold_keeps_inactive_G_and_marks_active_missing():
    frame = training_fixture()
    frame.loc[2, exp.LAG_INPUTS[0]] = np.nan
    bundles, audits = exp.fit_models(frame.iloc[:0], pd.DataFrame(), np.array([]))
    g = np.linspace(.1, .3, len(frame))
    scored, audit = exp.infer_with_initial(frame, bundles, g, .2)
    active = scored.R328_gate.to_numpy()
    for a in exp.NEW_ARMS:
        np.testing.assert_array_equal(scored.loc[~active, f'R328_{a}_p_net_5'], g[~active])
        assert scored.loc[active, f'R328_{a}_p_net_5'].isna().all()
        assert audit['unscored_observations'][a] == active.sum()
        assert not audits[a]['optimization_attempted']
    assert len(scored) == len(frame)
    np.testing.assert_array_equal(scored.R328_B_p_net_5, g)


@pytest.mark.parametrize('p', (0., 1., np.nan, np.inf))
def test_boundary_G_is_rejected_instead_of_clipped(p):
    with pytest.raises(ValueError, match='interior'):
        exp.baseline_logits(np.array([p]), 1)


def test_single_class_active_support_is_not_a_fitted_zero_model():
    frame = training_fixture()
    frame[exp.CEILING] = 0.
    manifest, _ = exp.transition_manifest(frame)
    bundles, audits = exp.fit_models(frame, manifest, np.full(len(frame), .2))
    assert all(bundles[a][0] is None and audits[a]['no_support'] and not audits[a]['optimization_attempted'] for a in exp.NEW_ARMS)


def test_one_failed_solver_attempt_is_recorded_without_retry(monkeypatch):
    calls = []
    def failure(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(x=np.zeros(1), success=False, nit=1, message='registered failure')
    monkeypatch.setattr(exp, 'minimize', failure)
    beta, audit = exp.solve(np.array([[1.], [1.]]), np.array([-1., -1.]), np.array([0., 1.]), np.ones(2), np.ones(2, bool))
    assert beta is None and not audit['no_support'] and not audit['optimization_success']
    assert len(calls) == 1 and calls[0]['options'] == exp.OPTIONS


def test_no_gate_weight_renormalization_with_missing_and_censored_rows():
    frame, g, _, _ = fitted()
    frame.loc[1, exp.LAG_INPUTS[0]], frame.loc[2, 'label_complete'] = np.nan, False
    manifest, _ = exp.transition_manifest(frame)
    _, audits = exp.fit_models(frame, manifest, g)
    w = exp.previous.target_weights(frame)
    active = exp.gate_mask(exp.old.feature_paths(frame))
    for a in exp.NEW_ARMS:
        assert audits[a]['supervision_weight_sum'] == pytest.approx(w.sum())
        assert audits[a]['active_weight_sum'] == pytest.approx(w[active].sum())
        assert audits[a]['inactive_constant_weight_sum'] == pytest.approx(w[~active].sum())


def test_original_clock_and_all_design_rows_are_persisted():
    frame, g, bundles, _ = fitted()
    scored, _ = exp.infer_with_initial(frame, bundles, g, .2)
    histories, matrix = exp.tables(scored, scored, scored)
    assert len(histories) == len(matrix) == 3*len(frame)
    assert set(matrix.scope) == {'training', 'evaluation', 'chronological'}
    assert len([c for c in matrix if '_design_' in c]) == 22
    for scope in matrix.scope.unique():
        part = matrix.loc[matrix.scope.eq(scope)]
        np.testing.assert_array_equal(part.source_index, frame.index)
        np.testing.assert_array_equal(part.original_complete_target_weight, exp.previous.target_weights(frame))
