import numpy as np
import pandas as pd
import pytest

from victory_trader import accumulated_current_evidence_momentum as exp
from test_observed_path_innovation_momentum import training_fixture


def fixture():
    frame = training_fixture()
    manifest, _ = exp.transition_manifest(frame)
    transform = exp.fit_transform(frame.loc[manifest.current_index], exp.LAG_INPUTS, manifest.transform_weight.to_numpy())
    g = np.linspace(.11, .31, len(frame))
    frozen_models, _ = exp.frozen.fit_models(frame, manifest, g)
    controls, _ = exp.frozen.infer_with_initial(frame, frozen_models, g, .2)
    return frame, transform, g, controls


def test_update_includes_current_and_skips_missing_clocks_with_real_elapsed():
    frame, transform, _, _ = fixture()
    frame = frame.iloc[:5].copy()
    frame.decision_t = [1000, 2000, 5000, 10000, 12000]
    frame.loc[2, exp.LAG_INPUTS[0]] = np.nan
    active, current, ema, anchor, first, before, after, last = exp.memory_basis(frame, transform)
    np.testing.assert_array_equal(active, [False, True, False, True, True])
    np.testing.assert_array_equal(first, [False, True, False, False, False])
    np.testing.assert_array_equal(ema[first], current[first])
    np.testing.assert_array_equal(anchor[first], current[first])
    np.testing.assert_array_equal(after[2], after[1])
    np.testing.assert_array_equal(before[3], after[1])
    assert last[2] == 2000 and last[3] == 10000
    z = np.exp(-8/30)
    np.testing.assert_allclose(ema[3], z*ema[1]+(1-z)*current[3], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(anchor[3], current[1])
    np.testing.assert_array_equal(ema[~active], 0.)
    assert np.isnan(before[0]).all() and np.isnan(after[0]).all()


def test_episode_reset_and_label_censor_future_prefix_invariance():
    frame, transform, _, _ = fixture()
    original = exp.memory_basis(frame, transform)
    changed = frame.copy()
    changed[exp.CEILING], changed.label_complete, changed.observation_available = np.nan, False, False
    for x, y in zip(original, exp.memory_basis(changed, transform)):
        np.testing.assert_array_equal(x, y)
    bare = frame.drop(columns=[exp.CEILING, 'label_complete', 'observation_available'])
    for x, y in zip(original, exp.memory_basis(bare, transform)):
        np.testing.assert_array_equal(x, y)
    first = exp.previous.history(frame).first
    assert not original[0][first].any() and np.isnan(original[6][first]).all()
    mask = frame.decision_t.le(3000).to_numpy()
    future = frame.copy()
    future.loc[~mask, list(exp.LAG_INPUTS)] = 1e8
    for x, y, z in zip(original, exp.memory_basis(future, transform), exp.memory_basis(frame.loc[mask], transform)):
        np.testing.assert_array_equal(x[mask], y[mask])
        np.testing.assert_array_equal(x[mask], z)


def test_censored_eligible_clock_updates_evidence():
    frame, transform, _, _ = fixture()
    frame.loc[2, 'label_complete'] = False
    active, _, _, _, first, _, after, last = exp.memory_basis(frame, transform)
    assert active[2] and not first[2]
    assert last[2] == frame.decision_t.iloc[2]
    assert np.max(np.abs(after[2]-after[1])) > 0


def test_frozen_transform_exact_replay_and_scope_rejection():
    frame, transform, _, _ = fixture()
    fit = {'fit_days': sorted(frame.trading_day.unique()), 'preprocessing': transform.audit()}
    restored = exp.frozen_preprocessing(frame, fit)
    np.testing.assert_array_equal(restored.apply(frame), transform.apply(frame))
    with pytest.raises(ValueError, match='scope'):
        exp.frozen_preprocessing(frame, fit | {'fit_days': ['2026-05-06']})
    assert exp.frozen_preprocessing(frame.iloc[:0], {'no_support': True}) is None


def test_convex_gradient_uses_original_constant_mass_and_censored_weights():
    x = np.array([[0., 0.], [1., -.3], [100., 100.], [-.2, 2.]])
    offset = np.array([-2., .1, 3., -1.])
    y, w = np.array([1., 0., np.nan, 1.]), np.array([2., 1., 0., .5])
    beta = np.array([.3, -.1])
    value, gradient = exp.frozen.offset_loss_gradient(beta, x, offset, y, w)
    for j in range(2):
        bump = np.zeros(2)
        bump[j] = 1e-5
        derivative = (exp.frozen.offset_loss_gradient(beta+bump, x, offset, y, w)[0]-exp.frozen.offset_loss_gradient(beta-bump, x, offset, y, w)[0])/2e-5
        assert derivative == pytest.approx(gradient[j], abs=1e-9)
    changed, grad2 = exp.frozen.offset_loss_gradient(beta, x, offset, np.array([0., 0., np.nan, 1.]), w)
    assert value != changed
    np.testing.assert_array_equal(gradient, grad2)
    v0 = exp.frozen.offset_loss_gradient(np.zeros(2), x, offset, y, w)[0]
    vm = exp.frozen.offset_loss_gradient(-beta, x, offset, y, w)[0]
    assert v0 <= (value+vm)/2


def test_fit_two_heads_only_original_weights_and_registered_hash_guard():
    frame, transform, g, _ = fixture()
    scope = {'path_sha256': exp.path_hashes(exp.memory_basis(frame, transform))}
    models, audits = exp.fit_models(frame, g, transform, scope)
    assert set(models) == {'F', 'N'}
    for a in models:
        assert len(models[a]) == 7 and audits[a]['supervision_weight_sum'] == pytest.approx(3.)
        assert audits[a]['active_weight_sum'] == pytest.approx(2.4)
        assert audits[a]['normalized_gradient_inf'] <= 1e-6
        assert audits[a]['parameters']['starts'] == 1
        assert audits[a]['intercept'] is False
    with pytest.raises(ValueError, match='fingerprint'):
        exp.fit_models(frame, g, transform, {'path_sha256': {}})


def test_inference_keeps_immutable_controls_and_exact_gate0_and_zero_copy():
    frame, transform, g, controls = fixture()
    models = {a: np.zeros(7) for a in exp.NEW_ARMS}
    scored, audit = exp.infer(frame, models, transform, controls)
    assert audit['frozen_control_paths_exact']
    for a in exp.NEW_ARMS:
        np.testing.assert_array_equal(scored[f'R329_{a}_p_net_5'], g)
    columns = [c for c in scored if c.startswith('R329_')]
    mutated = frame.copy()
    mutated[exp.CEILING], mutated.label_complete, mutated.observation_available = np.nan, False, False
    bare = frame.drop(columns=[exp.CEILING, 'label_complete', 'observation_available'])
    for variant in (mutated, bare):
        replay, _ = exp.infer(variant, models, transform, controls)
        pd.testing.assert_frame_equal(scored[columns], replay[columns])
    for a, source in (('C', 'S'), ('M', 'M'), ('B', 'B'), ('B0', 'B0')):
        for suffix in ('p_net_5', 'logit', 'prior_net_5', 'log_p_net_5', 'log_1_minus_p'):
            np.testing.assert_array_equal(scored[f'R329_{a}_{suffix}'], controls[f'R328_{source}_{suffix}'])
    damaged = controls.copy()
    damaged.R328_gate = ~damaged.R328_gate
    with pytest.raises(ValueError, match='gate'):
        exp.infer(frame, models, transform, damaged)


def test_missing_meta_fit_is_missing_active_not_zero_model():
    frame, _, _, controls = fixture()
    models, audits = exp.fit_models(frame.iloc[:0], np.array([]), None, None)
    for a in ('S', 'M'):
        mask = controls.R328_gate
        for suffix in ('p_net_5', 'logit', 'correction', 'log_p_net_5', 'log_1_minus_p'):
            controls.loc[mask, f'R328_{a}_{suffix}'] = np.nan
    scored, _ = exp.infer(frame, models, None, controls)
    active = scored.R329_gate
    for a in exp.NEW_ARMS:
        assert scored.loc[active, f'R329_{a}_p_net_5'].isna().all()
        np.testing.assert_array_equal(scored.loc[~active, f'R329_{a}_p_net_5'], controls.loc[~active, 'R328_B_p_net_5'])
        assert not audits[a]['optimization_attempted']


def test_parquet_is_complete_atomic_and_failure_preserves_destination(tmp_path, monkeypatch):
    frame = pd.DataFrame({'x': [1., np.nan]})
    exp.parquet(frame, tmp_path, 'test')
    path = tmp_path/'request329-test.parquet'
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
    original = path.read_bytes()
    def failure(*args, **kwargs):
        raise ValueError('footer failure')
    monkeypatch.setattr(exp.pq, 'ParquetFile', failure)
    with pytest.raises(ValueError, match='footer'):
        exp.parquet(pd.DataFrame({'x': [2.]}), tmp_path, 'test')
    assert path.read_bytes() == original
