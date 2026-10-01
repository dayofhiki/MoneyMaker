from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from victory_trader.hierarchical_crack_entry_controller import RegressionModel, _episode_day_weights
from victory_trader.preentry_context_hold_value import ALL_FEATURES, PREDICTION
from victory_trader.risk_reachable_hold_value import TARGET_PI0
from victory_trader.weighted_hold_offset import crossfit_weighted_offset, weighted_offset_model


def source():
    return pd.DataFrame({
        'trading_day': ['2026-05-05'] + ['2026-05-06']*3 + ['2026-05-07'],
        'ticker': ['H', 'A', 'A', 'A', 'B'], 'hot_t': 0,
        TARGET_PI0: [100., 1., 1., 1., -3.], 'risk_reachable_hold': True,
        PREDICTION: 0.,
    })


def mock_predict(frame, model):
    return np.zeros(len(frame))+model.offset


def test_weighted_center_preserves_tree_and_historical_model(monkeypatch):
    monkeypatch.setattr('victory_trader.weighted_hold_offset._predict', mock_predict)
    train = source().iloc[1:].copy()
    model = RegressionModel(object(), ALL_FEATURES, False, 0.)
    corrected, audit = weighted_offset_model(train, model)
    assert corrected.offset == pytest.approx(-1.)
    assert np.average(train[TARGET_PI0]-corrected.offset, weights=_episode_day_weights(train)) == pytest.approx(0.)
    assert model.offset == 0.
    assert corrected.model is model.model and corrected.columns == model.columns
    assert audit['weighted_residual_after_pp'] == pytest.approx(0.)


def test_ineligible_and_missing_targets_do_not_affect_offset(monkeypatch):
    monkeypatch.setattr('victory_trader.weighted_hold_offset._predict', mock_predict)
    train = source().iloc[1:].copy()
    model = RegressionModel(object(), ALL_FEATURES, False, 0.)
    extra = train.iloc[:2].copy()
    extra[TARGET_PI0] = [1e9, np.nan]
    extra['risk_reachable_hold'] = [False, True]
    corrected, _ = weighted_offset_model(pd.concat([train, extra]), model)
    assert corrected.offset == pytest.approx(-1.)
    with pytest.raises(ValueError, match='non-log'):
        weighted_offset_model(train, replace(model, log_target=True))


def mock_fold(monkeypatch):
    captured = []
    monkeypatch.setattr('victory_trader.weighted_hold_offset.CROSSFIT_DAYS', ['2026-05-05'])
    monkeypatch.setattr('victory_trader.weighted_hold_offset._predict', mock_predict)
    def fit(frame, target, seed, features):
        captured.append((frame.trading_day.tolist(), target, seed, features))
        return RegressionModel(object(), features, False, 0.)
    monkeypatch.setattr('victory_trader.weighted_hold_offset.model_fit', fit)
    return captured


def test_held_day_labels_cannot_change_calibration(monkeypatch):
    captured = mock_fold(monkeypatch)
    before, folds = crossfit_weighted_offset(source())
    changed = source()
    changed.loc[0, TARGET_PI0] = -1e12
    after, _ = crossfit_weighted_offset(changed)
    pd.testing.assert_series_equal(before[PREDICTION], after[PREDICTION])
    assert before[PREDICTION].iloc[0] == pytest.approx(-1.)
    assert captured[0] == (['2026-05-06']*3+['2026-05-07'], TARGET_PI0, 20263975, ALL_FEATURES)
    assert folds['2026-05-05']['max_saved_prediction_error'] == 0.


def test_saved_prediction_mismatch_fails(monkeypatch):
    mock_fold(monkeypatch)
    changed = source()
    changed.loc[0, PREDICTION] = .1
    with pytest.raises(ValueError, match='prediction mismatch'):
        crossfit_weighted_offset(changed)


def test_future_training_day_fails(monkeypatch):
    mock_fold(monkeypatch)
    changed = source()
    changed.loc[4, 'trading_day'] = '2026-06-15'
    with pytest.raises(ValueError, match='excluded day'):
        crossfit_weighted_offset(changed)
