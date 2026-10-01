import numpy as np
import pandas as pd
import pytest

from victory_trader.clock_context_hold_value import CLOCK_FEATURES, CONTEXT_FEATURES, causal_features
from victory_trader.preentry_context_hold_value import calibration


def fixture(times, closes, *, entry_t=100000, decisions=None, volumes=None, opening=0):
    bars = pd.DataFrame({'t': times, 'o': closes, 'h': closes, 'l': closes, 'c': closes, 'v': volumes or [100]*len(times)})
    states = pd.DataFrame({'trading_day': '2026-05-05', 'ticker': 'TEST', 'hot_t': entry_t, 'entry_t': entry_t, 'entry_price': 10., 'decision_t': decisions or [entry_t+1000]})
    return states, {('2026-05-05', 'TEST', entry_t): (bars, opening, 4000000)}


def test_preentry_context_available_at_first_postentry_second():
    states, contexts = fixture([0, 40000, 60000, 80000, 100000], [8., 9., 10., 11., 12.])
    after = causal_features(states, contexts)
    full = causal_features(states, contexts, include_preentry=True)
    assert after.clock_return_60s_pct.isna().all()
    assert full.clock_return_60s_pct.iloc[0] == pytest.approx(100*(12/9-1))
    assert full.clock_close_location_60s.iloc[0] == 1
    assert full.clock_observed_count_60s.iloc[0] == 3
    assert full.clock_volume_ratio_20s.iloc[0] == 1


def test_missing_history_does_not_fabricate_entry_anchor():
    states, contexts = fixture([100000], [12.])
    full = causal_features(states, contexts, include_preentry=True)
    assert full[list(CLOCK_FEATURES+CONTEXT_FEATURES)].isna().all().all()


def test_exact_cutoff_sparse_history_and_zero_volume():
    states, contexts = fixture([0, 40000, 100000], [8., 10., 12.], volumes=[0, 0, 0])
    full = causal_features(states, contexts, include_preentry=True)
    assert full.clock_return_60s_pct.iloc[0] == pytest.approx(20)
    assert full.clock_observed_count_60s.iloc[0] == 1
    assert full.clock_close_location_60s.iloc[0] == .5
    assert np.isnan(full.clock_volume_ratio_20s.iloc[0])


def test_premarket_prices_are_excluded_from_history():
    states, contexts = fixture([0, 100000], [1., 12.], opening=100000)
    full = causal_features(states, contexts, include_preentry=True)
    assert full.clock_return_60s_pct.isna().all()


def test_unfinished_and_future_price_volume_mutations_preserve_earlier_features():
    states, contexts = fixture([0, 40000, 60000, 80000, 100000, 101000, 102000], [8., 9., 10., 11., 12., 13., 14.])
    original = causal_features(states, contexts, include_preentry=True)
    bars = next(iter(contexts.values()))[0]
    bars.loc[bars.t.ge(101000), ['o', 'h', 'l', 'c', 'v']] = [100., 100., 100., 100., 10000.]
    changed = causal_features(states, contexts, include_preentry=True)
    pd.testing.assert_frame_equal(original, changed)


def test_full_and_since_entry_agree_once_all_windows_are_postentry():
    times = list(range(0, 201000, 1000))
    states, contexts = fixture(times, [10+i*.001 for i in range(len(times))], decisions=[161000, 181000, 201000])
    old = causal_features(states, contexts)
    full = causal_features(states, contexts, include_preentry=True)
    names = list(CLOCK_FEATURES+CONTEXT_FEATURES)
    pd.testing.assert_frame_equal(old[names], full[names])


def test_other_ticker_history_is_not_shared():
    states, contexts = fixture([0, 40000, 100000], [8., 10., 12.])
    other = states.assign(ticker='OTHER')
    bars = next(iter(contexts.values()))[0].copy()
    bars[['o', 'h', 'l', 'c']] = 20.
    contexts[('2026-05-05', 'OTHER', 100000)] = (bars, 0, 4000000)
    full = causal_features(pd.concat([states, other], ignore_index=True), contexts, include_preentry=True)
    assert full.clock_return_60s_pct.tolist() == pytest.approx([20., 0.])


def test_calibration_distinguishes_ranking_from_wrong_action_sign():
    frame = pd.DataFrame({'trading_day': ['2026-05-05']*2, 'ticker': ['ONE', 'TWO'], 'hot_t': [0, 0], 'risk_reachable_hold': True, 'target': [1., 2.], 'predicted_next_event_advantage_pct': [-2., -1.]})
    result = calibration(frame, 'target')
    assert result['weighted_sign_accuracy'] == 0
    assert result['confusion_counts']['exit_positive_target'] == 2
    assert result['predicted_exit']['weighted_target_mean_pct'] == 1.5
