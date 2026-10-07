from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from victory_trader import elapsed_state_rate_momentum as exp
from victory_trader.compact_momentum_representation import fit_transform, independent_weights
from victory_trader.nearby_state_momentum import local_manifest


def fixture(labels=(0., 0., 6., 6., 0.)):
    rows = []
    for day, ticker in (("2026-05-05", "A"), ("2026-05-05", "B"), ("2026-05-06", "C")):
        for j, label in enumerate(labels):
            rows.append({"trading_day": day, "ticker": ticker, "hot_t": 0, "decision_t": (j+1)*1000,
                "observation_available": True, "label_complete": True, exp.CEILING: label,
                **{f: float(j) for f in set().union(*exp.PACKAGES.values())}})
    return exp.causal_states(pd.DataFrame(rows))


def held(frame):
    result = frame.copy()
    result.trading_day = result.trading_day.replace({"2026-05-05": "2026-05-11", "2026-05-06": "2026-05-12"})
    return result


def initial_audit(frame, constant=.2):
    transform = fit_transform(frame, exp.PREVIOUS_PACKAGES["G"], independent_weights(frame))
    return {"fit_days": ["2026-05-05", "2026-05-06"], "preprocessing": transform.audit(),
        "single_class_fallback": True, "prior": constant}


@pytest.mark.parametrize("gap", [0., 1e-8, 1., 30., 240., 1e6])
def test_kernel_matches_exact_two_state_matrix_exponential(gap):
    a, b = .03, .07
    k = exp.kernel(np.log([[a*exp.RATE_UNIT_S, b*exp.RATE_UNIT_S]]), np.array([gap]))
    reference = expm(np.array([[-a, a], [b, -b]])*gap)
    np.testing.assert_allclose([k["q01"][0], k["q11"][0]], reference[:, 1], atol=2e-11, rtol=0)
    assert 0 <= k["q01"][0] <= k["q11"][0] <= 1.


def test_constant_inputs_composition_zero_gap_and_stationary_limit():
    eta = np.log([[.03*exp.RATE_UNIT_S, .07*exp.RATE_UNIT_S]])
    def update(p, duration):
        k = exp.kernel(eta, np.array([duration]))
        return float(k["pi"][0]*k["u"][0]+p*k["z"][0])
    assert update(.8, 0.) == .8
    assert update(update(.8, 10.), 20.) == pytest.approx(update(.8, 30.), abs=1e-15)
    assert update(.8, 1e6) == pytest.approx(.3, abs=1e-15)


@pytest.mark.parametrize("scale", [-20., 0., 10.])
def test_analytic_joint_gradient_matches_central_differences(scale):
    rng = np.random.default_rng(325)
    x = rng.normal(size=(12, 2))
    gap = np.geomspace(1., 240., len(x))
    previous = np.tile([0, 0, 1, 1], 3)
    current = np.tile([0, 1, 0, 1], 3)
    weights = np.linspace(.1, .9, len(x))
    theta = np.array([.1, -.2, scale-.4, -.1, .3, scale+.2])
    args = (x, gap, previous, current, weights)
    _, gradient = exp.loss_gradient(theta, *args)
    numerical = []
    for j in range(len(theta)):
        bump = np.zeros(len(theta))
        bump[j] = 1e-5
        numerical.append((exp.loss_gradient(theta+bump, *args)[0]-exp.loss_gradient(theta-bump, *args)[0])/2e-5)
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=0)


def test_extreme_log_rates_have_finite_unclipped_loss_and_gradient():
    x = np.array([[-1.], [1.], [-1.], [1.]])
    args = (x, np.ones(4), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 1]), np.ones(4))
    value, gradient = exp.loss_gradient(np.array([1000., 0., -1000., 0.]), *args)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    k = exp.kernel(np.array([[-1000., 1000.], [1000., -1000.]]), np.ones(2))
    np.testing.assert_array_equal(k["q01"], [0., 1.])
    np.testing.assert_array_equal(k["q11"], [0., 1.])


def test_likelihood_uses_endpoint_transition_probabilities_and_slope_penalty():
    x = np.ones((4, 1))
    gap = np.array([1., 3., 10., 30.])
    previous, current = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    weights = np.array([.1, .2, .3, .4])
    theta = np.array([.2, -.3, -.1, .4])
    k = exp.kernel(np.c_[x, np.ones(4)]@theta.reshape(2, 2).T, gap)
    p = np.where(previous == 0, k["q01"], k["q11"])
    expected = np.average(-np.where(current == 1, np.log(p), np.log1p(-p)), weights=weights)
    expected += (.2**2+(-.1)**2)/(2*exp.C*weights.sum())
    assert exp.loss_gradient(theta, x, gap, previous, current, weights)[0] == pytest.approx(expected, abs=1e-14)


def test_training_weights_and_shared_preprocessing_preserve_original_support():
    frame = fixture()
    manifest, _ = exp.transition_manifest(frame)
    bundles, audit = exp.fit_rates(frame, manifest)
    for arm in exp.PACKAGES:
        assert bundles[arm][0] is not None and audit[arm]["optimization_success"]
        assert audit[arm]["normalized_gradient_inf"] <= 1e-6
        assert audit[arm]["transform_weight_sum"] == pytest.approx(3)
        assert audit[arm]["supervision_weight_sum"] == pytest.approx(6)
        assert audit[arm]["initialization"]["0"]["weight_sum"] == pytest.approx(3)
        assert audit[arm]["initialization"]["1"]["weight_sum"] == pytest.approx(3)
        assert bundles[arm][0].coefficients.shape == (2, 2*len(exp.PACKAGES[arm]))
    assert audit["K"]["preprocessing"]["processed_dimensions"] == 0


def test_censored_original_endpoint_is_never_skipped_in_training():
    frame = fixture().iloc[:5].copy()
    frame.loc[1, "label_complete"] = False
    manifest, _ = exp.transition_manifest(frame)
    assert manifest[["previous_index", "current_index"]].to_numpy().tolist() == [[2, 3], [3, 4]]


def test_inference_reads_no_outcomes_and_label_censor_changes_are_irrelevant():
    frame = fixture()
    manifest, _ = exp.transition_manifest(frame)
    bundles, _ = exp.fit_rates(frame, manifest)
    evaluation, audit = held(frame), initial_audit(frame)
    original, _ = exp.infer(evaluation, bundles, audit)
    evaluation[exp.CEILING], evaluation.label_complete, evaluation.observation_available = np.nan, False, False
    mutated, _ = exp.infer(evaluation, bundles, audit)
    causal, _ = exp.infer(evaluation.drop(columns=[exp.CEILING, "label_complete", "observation_available"]), bundles, audit)
    columns = [c for c in original if c.startswith("R325_")]
    pd.testing.assert_frame_equal(original[columns], mutated[columns])
    pd.testing.assert_frame_equal(original[columns], causal[columns])


def test_future_features_and_truncation_cannot_change_present_predictions():
    frame = fixture()
    manifest, _ = exp.transition_manifest(frame)
    bundles, _ = exp.fit_rates(frame, manifest)
    evaluation, audit = held(frame), initial_audit(frame)
    original, _ = exp.infer(evaluation, bundles, audit)
    future = evaluation.copy()
    future.loc[future.decision_t.gt(2000), list(set().union(*exp.PACKAGES.values()))] = 1e9
    mutated, _ = exp.infer(future, bundles, audit)
    prefix, _ = exp.infer(evaluation.loc[evaluation.decision_t.le(2000)], bundles, audit)
    columns = [c for c in original if c.startswith("R325_")]
    pd.testing.assert_frame_equal(original.loc[prefix.index, columns], mutated.loc[prefix.index, columns])
    pd.testing.assert_frame_equal(original.loc[prefix.index, columns], prefix[columns])


def test_inference_uses_elapsed_seconds_and_common_first_without_fake_gap():
    frame = fixture().iloc[:4].copy()
    manifest, _ = exp.transition_manifest(frame)
    bundles, _ = exp.fit_rates(frame, manifest)
    for arm in exp.PACKAGES:
        _, transform = bundles[arm]
        bundles[arm] = exp.RateModel(np.zeros((2, 2*len(exp.PACKAGES[arm]))), np.log([.01*30, .02*30])), transform
    evaluation = held(frame)
    evaluation.decision_t = [1000, 11000, 21000, 31000]
    audit = initial_audit(frame)
    dense, _ = exp.infer(evaluation, bundles, audit)
    sparse, _ = exp.infer(evaluation.iloc[[0, 3]], bundles, audit)
    for arm in exp.CORE_ARMS:
        assert dense[f"R325_{arm}_p_net_5"].iloc[0] == .2
    for arm in exp.PACKAGES:
        assert dense[f"R325_{arm}_q01"].isna().iloc[0]
        assert dense[f"R325_{arm}_q11"].isna().iloc[0]
        assert dense[f"R325_{arm}_p_net_5"].iloc[-1] == pytest.approx(sparse[f"R325_{arm}_p_net_5"].iloc[-1], abs=1e-15)


@pytest.mark.parametrize("labels", [(0., 0., 0.), (0., 6., 0., 6.)])
def test_boundary_support_retains_rows_and_initial_but_no_fabricated_rates(labels):
    frame = fixture(labels=labels)
    manifest, _ = exp.transition_manifest(frame)
    bundles, fits = exp.fit_rates(frame, manifest)
    result, audit = exp.infer(held(frame), bundles, initial_audit(frame))
    assert len(result) == len(frame)
    for arm in exp.PACKAGES:
        assert fits[arm]["no_support"] and not fits[arm]["optimization_attempted"]
        assert result[f"R325_{arm}_onset_rate_per_s"].isna().all()
        assert audit["unscored_observations"][arm] == len(frame)-3
        assert result.groupby(exp.KEYS)[f"R325_{arm}_p_net_5"].first().eq(.2).all()


def test_failed_optimizer_is_recorded_without_retry(monkeypatch):
    frame = fixture()
    manifest, _ = exp.transition_manifest(frame)
    calls = []
    def failed(fun, initial, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(success=False, x=initial, nit=0, message="synthetic failure")
    monkeypatch.setattr(exp, "minimize", failed)
    model, audit = exp.solve(np.empty((len(manifest), 0)), manifest)
    assert model is None and not audit["optimization_success"] and len(calls) == 1


def test_scope_and_tampered_conditional_weights_are_rejected():
    frame = fixture()
    manifest, _ = exp.transition_manifest(frame)
    manifest.conditional_weight *= 100
    with pytest.raises((ValueError, AssertionError)):
        exp.fit_rates(frame, manifest)
    frame.loc[0, "trading_day"] = "2026-06-01"
    with pytest.raises(ValueError, match="training dates"):
        exp.fit_rates(frame, manifest)
    with pytest.raises(ValueError, match="nonnegative"):
        exp.kernel(np.zeros((1, 2)), np.array([-1.]))


def test_fractional_endpoints_cannot_be_silently_cast_to_a_latent_state():
    with pytest.raises(ValueError, match="binary endpoints"):
        exp.loss_gradient(np.zeros(2), np.empty((1, 0)), np.ones(1), np.array([.5]), np.array([1.]), np.ones(1))


def test_predecessor_join_checks_identities_and_reports_same_local_weights():
    frame = held(fixture())
    for arm in exp.CORE_ARMS:
        frame[f"R325_{arm}_p_net_5"] = np.where(frame[exp.CEILING].ge(5), .8, .2)
        frame[f"R325_{arm}_prior_net_5"] = .2
    previous = frame.copy()
    previous["R324_F_p_net_5"], previous["R324_F_prior_net_5"] = frame.R325_F_p_net_5, .2
    scored = exp.attach_predecessor(frame, previous)
    pairs, _ = local_manifest(scored, scored[exp.KEYS].drop_duplicates())
    report = exp.report_metrics(scored, pairs)
    assert report["F"]["local_pair_rank"] == report["P"]["within_episode_auc"] == 1.
    intervals = exp.intervals(scored, pairs, draws=10)
    assert intervals["day"]["comparisons"]["F_minus_P"]["local_pair_rank_ci95"] == [0., 0.]
    previous.loc[0, "ticker"] = "CHANGED"
    with pytest.raises((ValueError, AssertionError)):
        exp.attach_predecessor(frame, previous)
