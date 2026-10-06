import numpy as np
import pandas as pd
import pytest

from victory_trader import context_interaction_momentum as exp
from victory_trader.compact_momentum_representation import predict_linear


def training(day='2026-05-05'):
    rng = np.random.default_rng(18)
    n = 160
    x, z = rng.normal(size=(2, n))
    frame = pd.DataFrame({'trading_day': [day]*n, 'ticker': ['T'+str(i) for i in range(n)],
        'hot_t': [1000]*n, 'decision_t': [2000]*n, 'observation_available': True, 'label_complete': True,
        exp.CEILING: np.where(x*z > 0, 6., -3.)})
    for f in exp.MOMENTUM:
        frame[f] = 0.
    frame[exp.CONTROL[0]] = x
    frame[exp.MOMENTUM[len(exp.CONTROL)]] = z
    return frame


def test_joint_model_can_learn_conditional_signal_and_stumps_remain_additive():
    train = training()
    model, transform, prior, audit = exp.fit_shallow(train, exp.MOMENTUM, 2)
    p = predict_linear(train, model, transform, prior)
    assert np.mean((p > .5) == (train[exp.CEILING] >= 5)) > .85
    assert audit['trees']['leaf_paths']['mixed_control_signal'] > 0
    stump, _, _, stump_audit = exp.fit_shallow(train, exp.MOMENTUM, 1)
    assert stump_audit['trees']['actual_max_depth'] <= 1
    assert stump_audit['trees']['leaf_paths']['mixed_control_signal'] == 0
    assert stump.n_estimators_ == model.n_estimators_ == 100


def test_weighted_leaf_floor_counts_episode_support_not_raw_seconds():
    train = training()
    extra = pd.concat([train.iloc[:1].assign(decision_t=3000+i*1000) for i in range(100)])
    model, _, _, audit = exp.fit_shallow(pd.concat([train, extra]), exp.MOMENTUM, 2)
    assert audit['effective_weight_sum'] == pytest.approx(160)
    assert audit['trees']['minimum_leaf_weight_fraction'] >= .025-1e-10
    assert audit['trees']['minimum_leaf_distinct_episodes'] >= 4
    assert model.min_weight_fraction_leaf == .025


def test_transform_and_model_do_not_consume_held_out_values():
    train = training()
    a = exp.fit_shallow(train, exp.MOMENTUM, 2)
    held = training('2026-05-06')
    predict_linear(held, *a[:3])
    held[exp.MOMENTUM[0]] = 1e20
    held[exp.CEILING] = -1e20
    b = exp.fit_shallow(train, exp.MOMENTUM, 2)
    np.testing.assert_array_equal(predict_linear(train, *a[:3]), predict_linear(train, *b[:3]))
    assert a[3] == b[3]


def test_single_class_prior_fallback_preserves_missing_feature_handling():
    train = training()
    train[exp.CEILING] = -3.
    train[exp.MOMENTUM[-1]] = np.nan
    model, transform, prior, audit = exp.fit_shallow(train, exp.MOMENTUM, 2)
    assert model is None and prior == 0 and audit['single_class_fallback']
    assert exp.MOMENTUM[-1] in transform.all_missing
    assert np.isfinite(predict_linear(train, model, transform, prior)).all()


def test_depth_dates_censoring_and_duplicate_scope_rejected():
    train = training('2026-06-01')
    with pytest.raises(ValueError):
        exp.fit_shallow(train, exp.MOMENTUM, 2)
    with pytest.raises(ValueError):
        exp.fit_shallow(training(), exp.MOMENTUM, 3)
    train = training()
    with pytest.raises(ValueError):
        exp.score(pd.concat([train, train.iloc[:1]]), train)
    train.loc[0, 'label_complete'] = False
    with pytest.raises(ValueError):
        exp.score(train, train)


def test_new_heads_share_same_rows_weights_and_fixed_capacity(monkeypatch):
    train = training()
    captured = []
    def fit(frame, features, depth):
        captured.append((frame, features, depth))
        return None, exp.fit_transform(frame, features, exp.independent_weights(frame)), .5, {}
    monkeypatch.setattr(exp, 'fit_shallow', fit)
    held = training('2026-05-06')
    held.loc[0, 'observation_available'] = False
    scored, _ = exp.score(train, held)
    assert len(captured) == 4 and all(f is train for f, _, _ in captured)
    assert captured[0][2] == captured[1][2] == 2
    assert captured[2][2] == captured[3][2] == 1
    assert scored.loc[0, ['X_p_net_5', 'K_p_net_5', 'A_p_net_5', 'D_p_net_5']].isna().all()


def test_chronological_fit_excludes_current_and_future_days(monkeypatch):
    frames = [training(d) for d in exp.CROSSFIT_DAYS]
    first = pd.concat(frames).assign(in_broad=True)
    seen = []
    def score(train, held):
        assert train.trading_day.max() < held.trading_day.min()
        seen.append(held.trading_day.iloc[0])
        return held, {'fit_days': sorted(train.trading_day.unique())}
    monkeypatch.setattr(exp, 'score', score)
    result, folds = exp.chronological(first, pd.concat(frames))
    assert seen == list(exp.CROSSFIT_DAYS[1:]) and len(folds) == 3
    assert exp.CROSSFIT_DAYS[0] not in set(result.trading_day)


def test_score_replay_rejects_changed_label_missingness_and_score():
    saved = training().assign(C_p_net_5=.1, Q_p_net_5=.2)
    assert exp.replay_error(saved.copy(), saved) == {'C': 0., 'Q': 0.}
    for column, value in ((exp.CEILING, 100.), ('C_p_net_5', .9), ('decision_t', 9000), ('observation_available', False)):
        changed = saved.copy()
        changed.loc[0, column] = value
        with pytest.raises(AssertionError):
            exp.replay_error(changed, saved)


def test_difference_in_increments_does_not_confuse_capacity_and_signal_gain():
    values = {a: {'auc': .5} for a in exp.ARMS}
    values['X']['auc'] = .8
    values['K']['auc'] = .75
    values['C']['auc'] = .6
    values['Q']['auc'] = .55
    assert exp.contrasts(values, 'auc')['X_minus_C'] == pytest.approx(.2)
    assert exp.contrasts(values, 'auc')['joint_minus_linear_increment'] == pytest.approx(0)


def test_every_preregistered_probability_arm_uses_generic_classification_metrics():
    frame = training()
    for arm in exp.ARMS:
        frame[f'{arm}_p_net_5'] = np.where(frame[exp.CEILING] >= 5, .7, .3)
        frame[f'{arm}_prior_net_5'] = .5
    result = exp.ranks(frame, exp.ARMS)
    assert result['A']['brier'] == pytest.approx(.09)
    assert result['A']['auc'] == 1 and result['A']['positive_episodes'] > 0
    assert all(value == result['A'] for value in result.values())
