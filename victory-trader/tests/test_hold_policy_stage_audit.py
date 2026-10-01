import numpy as np
import pandas as pd
import pytest

from victory_trader.hold_policy_stage_audit import crossfit_one_stage, SELF_TARGET
from victory_trader.preentry_context_hold_value import ALL_FEATURES, PREDICTION
from victory_trader.risk_reachable_hold_value import TARGET_PI0, continuation_targets


def source():
    return pd.DataFrame({'trading_day': ['2026-05-05', '2026-05-06', '2026-05-07', '2026-05-08'], TARGET_PI0: [1., 1., 1., 1.], 'pi1_prediction_pct': [1., 1., 1., 1.], 'risk_reachable_hold': True})


def mock_fit(monkeypatch):
    captured = []
    monkeypatch.setattr('victory_trader.hold_policy_stage_audit.CROSSFIT_DAYS', ['2026-05-05'])
    def fit(frame, target, seed, features):
        captured.append((frame.trading_day.tolist(), target, seed, features))
        return frame[target].mean()
    monkeypatch.setattr('victory_trader.hold_policy_stage_audit.model_fit', fit)
    monkeypatch.setattr('victory_trader.hold_policy_stage_audit._predict', lambda frame, model: np.full(len(frame), model))
    return captured


def test_held_day_label_mutation_cannot_change_its_fit(monkeypatch):
    captured = mock_fit(monkeypatch)
    before, _ = crossfit_one_stage(source())
    changed = source()
    changed.loc[changed.trading_day.eq('2026-05-05'), TARGET_PI0] = 1000.
    after, _ = crossfit_one_stage(changed)
    assert before[PREDICTION].equals(after[PREDICTION])
    assert captured[0] == (['2026-05-06', '2026-05-07', '2026-05-08'], TARGET_PI0, 20263975, ALL_FEATURES)


def test_saved_prediction_disagreement_is_integrity_failure(monkeypatch):
    mock_fit(monkeypatch)
    changed = source()
    changed.loc[changed.trading_day.eq('2026-05-05'), 'pi1_prediction_pct'] = 2.
    with pytest.raises(ValueError, match='prediction mismatch'):
        crossfit_one_stage(changed)


def test_future_fit_day_is_rejected(monkeypatch):
    mock_fit(monkeypatch)
    changed = source()
    changed.loc[changed.trading_day.eq('2026-05-08'), 'trading_day'] = '2026-06-15'
    with pytest.raises(ValueError, match='excluded day'):
        crossfit_one_stage(changed)


def states():
    group = pd.DataFrame({'trading_day': '2026-05-05', 'ticker': 'TEST', 'hot_t': 0, 'entry_t': 0, 'entry_price': 10., 'decision_t': [1000, 2000, 3000], 'risk_reachable_hold': True, 'forced_stop_t': np.nan, 'observed_net_return_pct': [0., 0., 0.], 'drawdown_pct': [0., -2.1, 0.], 'exit_now_pct': [0., 1., 3.]})
    bars = pd.DataFrame({'t': [0, 1000, 2000, 3000, 4000], 'o': [10., 10., 10.1, 10.3, 10.4]})
    return group, {('2026-05-05', 'TEST', 0): (bars, 0, 4000000)}


def test_self_target_matches_reference_when_next_action_coincides():
    frame, contexts = states()
    frame[PREDICTION] = [1., -1., 1.]
    reference = continuation_targets(frame, contexts, prediction_column=None, output_column=TARGET_PI0)
    actual = continuation_targets(frame, contexts, prediction_column=PREDICTION, output_column=SELF_TARGET)
    assert actual[SELF_TARGET].iloc[0] == reference[TARGET_PI0].iloc[0] == 1.


def test_downstream_action_changes_self_target_without_mutating_prediction():
    frame, contexts = states()
    frame[PREDICTION] = [1., -1., -1.]
    earlier = continuation_targets(frame, contexts, prediction_column=PREDICTION, output_column=SELF_TARGET)
    changed = frame.copy()
    changed.loc[1, PREDICTION] = 1.
    later = continuation_targets(changed, contexts, prediction_column=PREDICTION, output_column=SELF_TARGET)
    assert later[SELF_TARGET].iloc[0] == 3.
    assert earlier[SELF_TARGET].iloc[0] == 1.
    assert later[PREDICTION].iloc[0] == earlier[PREDICTION].iloc[0]


def test_self_target_never_resurrects_forced_stop():
    frame, contexts = states()
    frame.loc[1:, 'risk_reachable_hold'] = False
    frame['forced_stop_t'] = 2000.
    frame[PREDICTION] = 100.
    result = continuation_targets(frame, contexts, prediction_column=PREDICTION, output_column=SELF_TARGET)
    assert result[SELF_TARGET].iloc[1:].isna().all()
    frame.loc[1:, PREDICTION] = -100.
    changed = continuation_targets(frame, contexts, prediction_column=PREDICTION, output_column=SELF_TARGET)
    pd.testing.assert_series_equal(result[SELF_TARGET], changed[SELF_TARGET])
