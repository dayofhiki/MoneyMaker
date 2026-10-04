import numpy as np
import pandas as pd
import pytest

from victory_trader.compact_momentum_representation import (
    COMPACT_FEATURES, fit_linear, fit_transform, independent_weights,
    predict_linear, representation_crossfit,
)
from victory_trader.feasible_upside_observability import CEILING
from victory_trader.lagged_minute_context import CROSSFIT_DAYS


def source():
    rows = []
    for day in CROSSFIT_DAYS:
        for episode in range(8):
            for seconds in (0, 20, 120):
                row = {'trading_day': day, 'ticker': str(episode), 'hot_t': 0,
                       'decision_t': seconds*1000, 'C_p_net_5': .25,
                       CEILING: 10. if episode % 3 == 0 else 0.}
                row.update({name: float(episode+seconds/120) for name in COMPACT_FEATURES})
                rows.append(row)
    return pd.DataFrame(rows)


def test_transform_fits_training_only_and_preserves_missing_information():
    train = pd.DataFrame({'x': [0., 1., 1000.], 'empty': [np.nan]*3})
    fitted = fit_transform(train, ('x', 'empty'), np.array([50., 49.9, .1]))
    assert fitted.lower[0] == 0. and fitted.upper[0] == 1. and fitted.median[0] == 0.
    assert fitted.all_missing == ('empty',)
    held = pd.DataFrame({'x': [1e20, np.nan], 'empty': [100., np.nan]})
    x = fitted.apply(held)
    assert np.isfinite(x).all() and x.shape == (2, 4)
    assert x[:, 2:].tolist() == [[0., 0.], [1., 1.]]
    assert x[0, 0] == pytest.approx(fitted.apply(pd.DataFrame({'x': [1.], 'empty': [100.]}))[0, 0])
    assert fitted.upper[0] == 1.  # Held-out extremes do not refit clipping.


def test_episode_mass_keeps_regularization_strength_when_rows_are_repeated():
    f = source().loc[lambda f: f.trading_day.eq(CROSSFIT_DAYS[0])].copy()
    model, transform, prior, audit = fit_linear(f, COMPACT_FEATURES, 1)
    repeated = pd.concat([f, f.assign(decision_t=f.decision_t+1)], ignore_index=True)
    other, other_transform, other_prior, other_audit = fit_linear(repeated, COMPACT_FEATURES, 1)
    assert independent_weights(f).sum() == pytest.approx(8.)
    assert independent_weights(repeated).sum() == pytest.approx(8.)
    assert audit['weight_sum'] == pytest.approx(other_audit['weight_sum'])
    assert prior == pytest.approx(other_prior)
    assert predict_linear(f, model, transform, prior) == pytest.approx(predict_linear(f, other, other_transform, other_prior), abs=1e-7)


def test_single_class_fallback_and_infinite_feature_rejection():
    f = source().copy()
    f[CEILING] = 0.
    model, transform, prior, audit = fit_linear(f, COMPACT_FEATURES, 1)
    assert model is None and audit['single_class_fallback'] and prior == 0.
    assert predict_linear(f, model, transform, prior) == pytest.approx(np.zeros(len(f)))
    f.loc[0, COMPACT_FEATURES[0]] = np.inf
    with pytest.raises(ValueError, match='finite'):
        fit_linear(f, COMPACT_FEATURES, 1)


def test_held_labels_cannot_change_own_model_or_transform_and_first_only_is_audited(monkeypatch):
    calls = []
    def tree(train, features, level, seed):
        calls.append(set(train.trading_day))
        return None, .25
    monkeypatch.setattr('victory_trader.compact_momentum_representation.fit_probability', tree)
    f = source()
    before, audit_before = representation_crossfit(f, COMPACT_FEATURES)
    f.loc[f.trading_day.eq(CROSSFIT_DAYS[0]), CEILING] = 1e9
    after, audit_after = representation_crossfit(f, COMPACT_FEATURES)
    held = before.trading_day.eq(CROSSFIT_DAYS[0])
    pd.testing.assert_frame_equal(before.loc[held, ['W_p_net_5', 'X_p_net_5', 'F_p_net_5']], after.loc[held, ['W_p_net_5', 'X_p_net_5', 'F_p_net_5']])
    assert audit_before[CROSSFIT_DAYS[0]] == audit_after[CROSSFIT_DAYS[0]]
    assert calls[0] == set(CROSSFIT_DAYS[1:])
    for day, audit in audit_before.items():
        assert day not in audit['fit_days']
        assert audit['models']['F']['train_rows'] == audit['models']['F']['train_episodes'] == 24
        assert audit['models']['X']['train_rows'] == 72
        assert audit['models']['X']['weight_sum'] == pytest.approx(24.)
        assert audit['models']['X']['preprocessing']['processed_dimensions'] == 20


def test_saved_score_future_day_and_missing_causal_subset_rejections(monkeypatch):
    monkeypatch.setattr('victory_trader.compact_momentum_representation.fit_probability', lambda *args: (None, .25))
    f = source()
    f.loc[0, 'C_p_net_5'] = .8
    with pytest.raises(ValueError, match='score mismatch'):
        representation_crossfit(f, COMPACT_FEATURES)
    f.loc[0, 'trading_day'] = '2026-06-15'
    with pytest.raises(ValueError, match='May5-8'):
        representation_crossfit(f, COMPACT_FEATURES)
    with pytest.raises(ValueError, match='causal subset'):
        representation_crossfit(source(), ('saved_future_score',))
