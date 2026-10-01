import numpy as np
import pandas as pd
import pytest

from victory_trader.common_cap_hold_reference import TARGET_CAP, cap_reference_targets, crossfit_cap_reference
from victory_trader.preentry_context_hold_value import ALL_FEATURES, PREDICTION
from victory_trader.risk_reachable_hold_value import TARGET_PI0, continuation_targets, liquidation


def fixture():
    frame = pd.DataFrame({'trading_day': '2026-05-05', 'ticker': 'T', 'hot_t': 0,
        'entry_t': 0, 'entry_price': 10., 'decision_t': [590000, 600000, 610000],
        'position_age_s': [590., 600., 610.], 'risk_reachable_hold': True,
        'forced_stop_t': np.nan, 'observed_net_return_pct': 0., 'drawdown_pct': 0.,
        'exit_now_pct': [0., 1., 3.]})
    bars = pd.DataFrame({'t': [590000, 600000, 610000, 625000], 'o': [10., 10.1, 10.3, 11.]})
    return frame, {('2026-05-05', 'T', 0): (bars, 0, 620000)}


def test_expiry_alone_no_longer_zeros_wait_value():
    frame, contexts = fixture()
    old = continuation_targets(frame, contexts, prediction_column=None, output_column=TARGET_PI0)
    new = cap_reference_targets(frame, contexts)
    assert old[TARGET_PI0].iloc[1:].eq(0).all()
    expected = liquidation(contexts[('2026-05-05', 'T', 0)][0], 10., 620000)
    assert new[TARGET_CAP].iloc[1] == pytest.approx(expected-1.)
    assert new[TARGET_CAP].iloc[2] == pytest.approx(expected-3.)
    assert '_fixed_trailing_teacher_action' not in new
    assert TARGET_CAP not in frame


def test_same_trailing_action_before_expiry_matches_old():
    frame, contexts = fixture()
    frame['decision_t'] = [1000, 2000, 3000]
    frame['drawdown_pct'] = [0., -2., 0.]
    old = continuation_targets(frame, contexts, prediction_column=None, output_column=TARGET_PI0)
    new = cap_reference_targets(frame, contexts)
    assert new[TARGET_CAP].iloc[0] == old[TARGET_PI0].iloc[0] == 1.


@pytest.mark.parametrize('stop_t', [610000, 620000])
def test_stop_before_or_tied_with_cap_remains_absorbing(stop_t):
    frame, contexts = fixture()
    frame.loc[2, 'decision_t'] = stop_t
    frame['forced_stop_t'] = float(stop_t)
    frame.loc[2, 'risk_reachable_hold'] = False
    frame.loc[2, 'observed_net_return_pct'] = -3.
    new = cap_reference_targets(frame, contexts)
    bars = contexts[('2026-05-05', 'T', 0)][0]
    assert new[TARGET_CAP].iloc[1] == pytest.approx(liquidation(bars, 10., stop_t)-1.)
    assert np.isnan(new[TARGET_CAP].iloc[2])
    frame.loc[2, 'drawdown_pct'] = -100.
    pd.testing.assert_series_equal(new[TARGET_CAP], cap_reference_targets(frame, contexts)[TARGET_CAP])


def test_gap_crossing_cap_submits_at_cap_not_next_state():
    frame, contexts = fixture()
    frame = frame.iloc[:2].copy()
    bars = contexts[('2026-05-05', 'T', 0)][0]
    new = cap_reference_targets(frame, contexts)
    assert new[TARGET_CAP].iloc[-1] == pytest.approx(liquidation(bars, 10., 620000)-1.)


def mock_fold(monkeypatch):
    calls = []
    monkeypatch.setattr('victory_trader.common_cap_hold_reference.CROSSFIT_DAYS', ['2026-05-05'])
    def fit(frame, target, seed, features):
        calls.append((frame.trading_day.tolist(), target, seed, features))
        return float(frame[target].mean())
    def offset(frame, model):
        assert frame[TARGET_PI0].mean() == pytest.approx(model)
        return model, {'weighted_residual_after_pp': 0.}
    monkeypatch.setattr('victory_trader.common_cap_hold_reference.model_fit', fit)
    monkeypatch.setattr('victory_trader.common_cap_hold_reference.weighted_offset_model', offset)
    monkeypatch.setattr('victory_trader.common_cap_hold_reference._predict', lambda f,m: np.full(len(f),m))
    return calls


def source():
    return pd.DataFrame({'trading_day': ['2026-05-05','2026-05-06','2026-05-07','2026-05-08'], TARGET_PI0: 1., TARGET_CAP: [9.,2.,2.,2.], PREDICTION: 1.})


def test_held_day_both_targets_cannot_change_model(monkeypatch):
    calls = mock_fold(monkeypatch)
    original = source()
    before, _ = crossfit_cap_reference(original)
    changed = source()
    changed.loc[0, [TARGET_PI0, TARGET_CAP]] = -1e12
    after, _ = crossfit_cap_reference(changed)
    pd.testing.assert_series_equal(before[PREDICTION], after[PREDICTION])
    assert before[PREDICTION].iloc[0] == 2.
    assert calls[1] == (['2026-05-06','2026-05-07','2026-05-08'], TARGET_CAP, 20263975, ALL_FEATURES)
    assert original[TARGET_PI0].eq(1.).all()


def test_saved_corrected_baseline_mismatch_fails(monkeypatch):
    mock_fold(monkeypatch)
    changed = source()
    changed.loc[0, PREDICTION] = 5.
    with pytest.raises(ValueError, match='prediction mismatch'):
        crossfit_cap_reference(changed)


def test_future_day_fails(monkeypatch):
    mock_fold(monkeypatch)
    changed = source()
    changed.loc[3, 'trading_day'] = '2026-06-15'
    with pytest.raises(ValueError, match='excluded day'):
        crossfit_cap_reference(changed)
