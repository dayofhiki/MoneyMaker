import numpy as np
import pandas as pd
import pytest

from victory_trader.clock_context_hold_value import (
    CLOCK_FEATURES, CONTEXT_FEATURES, causal_features, paired_intervals,
)
from victory_trader.frozen_entry_second_hold_exit import FEATURES
from victory_trader.risk_reachable_hold_value import model_fit


def fixture(times, closes, volumes=None, decision_times=None):
    bars = pd.DataFrame({"t": times, "c": closes, "v": volumes or [100]*len(times)})
    decisions = decision_times or [t+1000 for t in times]
    frame = pd.DataFrame({"decision_t": decisions, "trading_day": "2026-05-05", "ticker": "TEST", "hot_t": 0, "entry_t": 0, "entry_price": 10.})
    return frame, {("2026-05-05", "TEST", 0): (bars, 0, 4000000)}


def test_sparse_clock_count_is_not_last_twenty_observations():
    states, contexts = fixture(list(range(0, 200000, 10000)), [10.]*20)
    result = causal_features(states, contexts)
    assert result.clock_observed_count_20s.iloc[-1] == 2
    assert result.clock_observed_count_60s.iloc[-1] == 6
    assert result.clock_return_20s_pct.iloc[-1] == 0


def test_exact_boundary_anchor_and_missing_partial_history():
    states, contexts = fixture([0, 19000, 20000], [11., 12., 13.], decision_times=[1000, 20000, 21000])
    result = causal_features(states, contexts)
    assert np.isnan(result.clock_return_20s_pct.iloc[0])
    assert result.clock_return_20s_pct.iloc[1] == pytest.approx(20.)
    assert result.clock_return_20s_pct.iloc[2] == pytest.approx(100*(13/11-1))
    # Completed close ending exactly at t-20s is the return anchor, not in the count.
    assert result.clock_observed_count_20s.iloc[2] == 2


def test_volume_windows_zero_denominator_and_flat_range():
    states, contexts = fixture([0, 20000, 40000, 60000], [10.]*4, [0, 100, 200, 400])
    result = causal_features(states, contexts)
    assert result.clock_volume_ratio_20s.iloc[2] == 2
    assert result.clock_volume_ratio_20s.iloc[3] == 2
    assert result.clock_close_location_60s.iloc[3] == .5
    assert result.clock_drawdown_60s_pct.iloc[3] == 0
    contexts[('2026-05-05', 'TEST', 0)][0].loc[:, 'v'] = 0
    assert causal_features(states, contexts).clock_volume_ratio_20s.isna().all()


def test_no_interpolation_in_long_gap_and_unfinished_bar_excluded():
    states, contexts = fixture([0, 1000, 100000], [10., 12., 100.], decision_times=[1000, 2000, 100000])
    result = causal_features(states, contexts)
    # At 100s the 100s-starting bar is unfinished; known close stays 12.
    assert result.clock_return_60s_pct.iloc[-1] == 0
    assert result.clock_observed_count_60s.iloc[-1] == 0
    assert np.isnan(result.clock_close_location_60s.iloc[-1])


def test_future_price_volume_mutation_changes_no_earlier_feature():
    states, contexts = fixture(list(range(0, 81000, 1000)), [10+i*.01 for i in range(81)])
    original = causal_features(states, contexts)
    bars = contexts[('2026-05-05', 'TEST', 0)][0]
    bars.loc[bars.t.ge(60000), ['c', 'v']] = [100., 10000.]
    changed = causal_features(states, contexts)
    names = list(CLOCK_FEATURES+CONTEXT_FEATURES)
    pd.testing.assert_frame_equal(original.loc[states.decision_t.le(60000), names], changed.loc[states.decision_t.le(60000), names])
    assert not original.loc[states.decision_t.gt(60000), names].equals(changed.loc[states.decision_t.gt(60000), names])


def test_features_do_not_cross_position_or_entry_boundary():
    states, contexts = fixture([0, 1000, 60000], [100., 10., 12.])
    states['entry_t'] = 1000
    states = states.loc[states.decision_t.ge(2000)]
    result = causal_features(states, contexts)
    assert result.clock_return_60s_pct.iloc[-1] == pytest.approx(20.)
    assert result.clock_observed_count_60s.iloc[-1] == 2
    second = states.assign(ticker='OTHER', entry_price=20.)
    contexts[('2026-05-05', 'OTHER', 0)] = (pd.DataFrame({'t': [1000, 60000], 'c': [20., 20.], 'v': [1., 1.]}), 0, 4000000)
    both = causal_features(pd.concat([states, second], ignore_index=True), contexts)
    assert both.loc[both.ticker.eq('OTHER'), 'clock_return_60s_pct'].iloc[-1] == 0


def test_requested_feature_set_is_forwarded_without_changing_default(monkeypatch):
    captured = []
    monkeypatch.setattr('victory_trader.risk_reachable_hold_value._fit', lambda frame, columns, *args, **kwargs: captured.append(columns))
    frame = pd.DataFrame({'risk_reachable_hold': [True, False]})
    model_fit(frame, 'target', 1)
    model_fit(frame, 'target', 1, FEATURES+CLOCK_FEATURES)
    assert captured == [FEATURES, FEATURES+CLOCK_FEATURES]


def test_paired_cluster_intervals_keep_all_entries_and_use_matched_differences():
    frame = pd.DataFrame({'trading_day': ['2026-05-05', '2026-05-06'], 'ticker': ['ONE', 'TWO'], 'hot_t': [0, 0], 'resolved': [True, True], 'base_net_return_pct': [1., 2.]})
    result = paired_intervals(frame, frame.assign(base_net_return_pct=frame.base_net_return_pct-3))
    assert result['matched_entries'] == 2
    assert result['mean_gain_pp'] == 3
    assert result['bootstrap']['day']['gain_ci95_pp'] == [3., 3.]
