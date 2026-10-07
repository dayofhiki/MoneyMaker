import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from victory_trader import joint_state_timing_momentum as exp
from victory_trader.compact_momentum_representation import fit_transform, independent_weights
from victory_trader.nearby_state_momentum import local_manifest


def design():
    rng = np.random.default_rng(323)
    x = rng.normal(size=(20, 4))
    y = np.tile([0, 1], 10)
    weights = np.linspace(.1, .5, 20)
    delta = x[y == 1]-x[y == 0]
    pair_weights = np.linspace(.1, .2, 10)
    return x, y, weights, delta, pair_weights


@pytest.mark.parametrize("alpha", [0., .5])
def test_analytic_gradient_matches_central_differences(alpha):
    args = design()
    theta = np.array([.2, -.3, .1, .4, -.5])
    _, gradient = exp.loss_gradient(theta, *args, alpha)
    numerical = []
    for j in range(len(theta)):
        bump = np.zeros(len(theta))
        bump[j] = 1e-6
        numerical.append((exp.loss_gradient(theta+bump, *args, alpha)[0]-exp.loss_gradient(theta-bump, *args, alpha)[0])/2e-6)
    np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=0)


def test_state_only_boundary_matches_sklearn_regularization_and_intercept():
    x, y, weights, delta, pair_weights = design()
    model, audit = exp.solve(x, y, weights, delta, pair_weights, 0.)
    reference = LogisticRegression(C=.1, l1_ratio=0., fit_intercept=True, solver="lbfgs", max_iter=2000, tol=1e-8)
    reference.fit(x, y, sample_weight=weights)
    np.testing.assert_allclose(model.coefficients, reference.coef_[0], atol=1e-7, rtol=0)
    assert model.intercept == pytest.approx(reference.intercept_[0], abs=1e-7)
    np.testing.assert_allclose(model.predict(x), reference.predict_proba(x)[:, 1], atol=1e-7, rtol=0)
    assert audit["success"] and audit["normalized_gradient_inf"] <= 1e-6


def test_pair_loss_cancels_intercept_and_reverse_orientation_is_identical():
    x, y, weights, delta, pair_weights = design()
    theta = np.array([.2, -.3, .1, .4, -.5])
    before, gradient = exp.loss_gradient(theta, x, y, weights, delta, pair_weights, 1.)
    theta[-1] += 50
    after, changed_gradient = exp.loss_gradient(theta, x, y, weights, delta, pair_weights, 1.)
    assert before == after and gradient[-1] == changed_gradient[-1] == 0.
    z = delta@theta[:-1]
    loss = np.average(np.logaddexp(0., -z), weights=pair_weights)
    both = np.r_[z, -z]
    labels = np.r_[np.ones(len(z)), np.zeros(len(z))]
    assert loss == pytest.approx(np.average(np.logaddexp(0., both)-labels*both, weights=np.tile(pair_weights/2, 2)))


def test_pair_weight_unit_change_does_not_change_normalized_joint_loss():
    x, y, weights, delta, pair_weights = design()
    a, _ = exp.solve(x, y, weights, delta, pair_weights, .5)
    b, _ = exp.solve(x, y, weights, delta, pair_weights*100, .5)
    np.testing.assert_allclose(a.coefficients, b.coefficients, atol=1e-10, rtol=0)
    assert a.intercept == pytest.approx(b.intercept, abs=1e-10)


def test_solver_rejects_unregistered_mix_bad_weights_and_no_joint_pairs():
    x, y, weights, delta, pair_weights = design()
    with pytest.raises(ValueError, match="fixed mixture"):
        exp.solve(x, y, weights, delta, pair_weights, .25)
    weights[0] = -1
    with pytest.raises(ValueError, match="positive"):
        exp.solve(x, y, weights, delta, pair_weights, .5)
    weights[0] = 1
    with pytest.raises(ValueError, match="pair mass"):
        exp.solve(x, y, weights, delta[:0], pair_weights[:0], .5)


def fixture():
    rows = []
    for day, ticker, labels in (("2026-05-05", "A", [0., 6., 0.]),
                                ("2026-05-05", "B", [0., 0.]),
                                ("2026-05-06", "C", [6., 0.])):
        for j, label in enumerate(labels):
            rows.append({"trading_day": day, "ticker": ticker, "hot_t": 0, "decision_t": (j+1)*1000,
                "observation_available": True, "label_complete": True, exp.CEILING: label,
                **{f: float(j)+(label >= 5) for f in exp.PACKAGES["J"]}})
    return pd.DataFrame(rows)


def test_frozen_transforms_independent_mass_and_censoring_are_preserved():
    frame = fixture()
    frame.loc[2, "label_complete"] = False
    pairs, _ = exp.pair_manifest(frame)
    complete = frame.loc[exp.complete_mask(frame)]
    transform = fit_transform(complete, exp.PACKAGES["J"], independent_weights(complete))
    model, copied, audit = exp.fit(frame, pairs, exp.PACKAGES["J"], .5, transform.audit())
    assert audit["training_episodes"] == 3 and audit["mixed_training_episodes"] == 2
    assert audit["state_weight_sum"] == pytest.approx(3)
    assert audit["pair_weight_sum"] == pytest.approx(2)
    assert audit["absolute_component_mass"] == audit["relative_component_mass"] == pytest.approx(1.5)
    assert audit["effective_total_mass"] == pytest.approx(3)
    assert copied.audit() == transform.audit()
    assert np.isfinite(model.predict(copied.apply(frame))).all() and len(frame) == 7


def test_single_state_inference_is_independent_of_future_features():
    frame = fixture()
    pairs, _ = exp.pair_manifest(frame)
    transform = fit_transform(frame, exp.PACKAGES["J"], independent_weights(frame))
    model, copied, _ = exp.fit(frame, pairs, exp.PACKAGES["J"], .5, transform.audit())
    before = model.predict(copied.apply(frame.iloc[:1]))
    frame.loc[1:, list(exp.PACKAGES["J"])] = 1e9
    np.testing.assert_array_equal(before, model.predict(copied.apply(frame.iloc[:1])))


def test_scope_wrong_transform_and_cross_episode_pair_rejected():
    frame = fixture()
    pairs, _ = exp.pair_manifest(frame)
    saved = fit_transform(frame, exp.PACKAGES["J"], independent_weights(frame)).audit()
    changed = dict(saved, mean=[v+1 for v in saved["mean"]])
    with pytest.raises(ValueError, match="training scope"):
        exp.fit(frame, pairs, exp.PACKAGES["J"], .5, changed)
    invalid = pairs.copy()
    invalid.loc[0, "negative_index"] = 3
    with pytest.raises(ValueError, match="boundary"):
        exp.fit(frame, invalid, exp.PACKAGES["J"], .5, saved)
    frame.loc[0, "trading_day"] = "2026-06-01"
    with pytest.raises(ValueError, match="training dates"):
        exp.fit(frame, pairs, exp.PACKAGES["J"], .5, saved)


def test_namespaced_probability_and_local_rank_contracts_keep_saved_scores():
    frame = fixture()
    frame.trading_day = frame.trading_day.replace({"2026-05-05": "2026-05-11", "2026-05-06": "2026-05-12"})
    frame["C_p_net_5"] = .123
    for arm in exp.ARMS:
        values = np.zeros(len(frame)) if arm.endswith("0") else frame[exp.CEILING].to_numpy()
        frame[f"R323_{arm}_score"] = values
        frame[f"R323_{arm}_p_net_5"] = 1/(1+np.exp(-values))
        frame[f"R323_{arm}_prior_net_5"] = .1
    pairs, _ = local_manifest(frame, frame[exp.KEYS].drop_duplicates())
    ranked = exp.report_metrics(frame, pairs)
    assert ranked["J"]["within_episode_auc"] == ranked["J"]["local_pair_rank"] == 1.
    assert ranked["J0"]["within_episode_auc"] == ranked["J0"]["local_pair_rank"] == .5
    assert frame.C_p_net_5.eq(.123).all()
    result = exp.intervals(frame, pairs, draws=10)
    assert result["day"]["comparisons"]["J_minus_C"]["local_pair_rank_ci95"] == [0., 0.]
