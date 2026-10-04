import numpy as np
import pandas as pd
import pytest

from victory_trader.early_momentum_weighting import sample_weights, timing, weighted_crossfit
from victory_trader.feasible_upside_observability import CEILING
from victory_trader.frozen_entry_second_hold_exit import KEYS
from victory_trader.hierarchical_crack_entry_controller import _episode_day_weights
from victory_trader.lagged_minute_context import CROSSFIT_DAYS


def episode(day, ticker, seconds):
    return pd.DataFrame({'trading_day': day, 'ticker': ticker, 'hot_t': 0,
                         'decision_t': np.array(seconds)*1000, CEILING: 0., 'C_p_net_5': .25})


def test_weights_balance_days_episodes_and_available_clock_phases():
    f = pd.concat([episode('2026-05-05', 'A', [0, 1, 2, 20, 60, 120, 121]),
                   episode('2026-05-05', 'B', [0, 20]),
                   episode('2026-05-06', 'C', [0, 1, 2])], ignore_index=True)
    phase, first = timing(f)
    assert phase.tolist()[:7] == [0, 0, 0, 1, 2, 3, 3]
    for arm in ('U', 'T', 'E'):
        w = sample_weights(f, arm)
        assert w.mean() == pytest.approx(1.)
        days = f.assign(w=w).groupby('trading_day').w.sum()
        assert days.iloc[0] == pytest.approx(days.iloc[1])
        episodes = f.assign(w=w).groupby(KEYS).w.sum()
        assert episodes.iloc[0] == pytest.approx(episodes.iloc[1])
    t = sample_weights(f, 'T')[:7]
    assert [t[phase[:7] == i].sum()/t.sum() for i in range(4)] == pytest.approx([.25]*4)
    e = sample_weights(f, 'E')[:7]
    assert e[first[:7]].sum()/e.sum() == pytest.approx(.5+.5/12)
    assert sample_weights(f, 'U') == pytest.approx(_episode_day_weights(f))


def test_label_changes_and_later_sampling_density_do_not_choose_phase_mass():
    f = episode('2026-05-05', 'A', [0, 20, 60, 120])
    original = sample_weights(f, 'T')
    f[CEILING] = [1e8, -1e8, 5, 0]
    assert sample_weights(f, 'T') == pytest.approx(original)
    dense = pd.concat([f, episode('2026-05-05', 'A', list(range(121, 181)))], ignore_index=True)
    phase, _ = timing(dense)
    for arm in ('T', 'E'):
        w = sample_weights(dense, arm)
        expected = [.25]*4 if arm == 'T' else [.625, .125, .125, .125]
        assert [w[phase == i].sum()/w.sum() for i in range(4)] == pytest.approx(expected)


def test_missing_phases_are_not_synthetic_observations():
    f = episode('2026-05-05', 'A', [0, 120, 121])
    phase, _ = timing(f)
    w = sample_weights(f, 'T')
    assert set(phase) == {0, 3}
    assert w[0]/w.sum() == pytest.approx(.5)


def test_duplicate_clocks_and_future_day_rejected():
    with pytest.raises(ValueError, match='unique'):
        sample_weights(episode('2026-05-05', 'A', [0, 0]), 'E')
    with pytest.raises(ValueError, match='May5-8'):
        weighted_crossfit(episode('2026-06-15', 'A', [0]), ())


def test_held_day_labels_cannot_change_own_scores_and_saved_replay_required(monkeypatch):
    calls = []
    def baseline(train, features, level, seed):
        calls.append(set(train.trading_day))
        return None, .25
    def weighted(train, features, arm, seed):
        return None, float(np.average(train[CEILING].ge(5), weights=sample_weights(train, arm)))
    monkeypatch.setattr('victory_trader.early_momentum_weighting.fit_probability', baseline)
    monkeypatch.setattr('victory_trader.early_momentum_weighting.fit_weighted', weighted)
    f = pd.concat([episode(day, 'A', [0, 20, 120]) for day in CROSSFIT_DAYS], ignore_index=True)
    before, audits_before = weighted_crossfit(f, ())
    f.loc[f.trading_day.eq(CROSSFIT_DAYS[0]), CEILING] = 100
    after, audits_after = weighted_crossfit(f, ())
    held = before.trading_day.eq(CROSSFIT_DAYS[0])
    pd.testing.assert_frame_equal(before.loc[held, ['T_p_net_5', 'E_p_net_5']], after.loc[held, ['T_p_net_5', 'E_p_net_5']])
    assert audits_before[CROSSFIT_DAYS[0]] == audits_after[CROSSFIT_DAYS[0]]
    for day, audit in audits_before.items():
        assert day not in audit['fit_days']
    assert calls[0] == set(CROSSFIT_DAYS[1:])
    f.loc[0, 'C_p_net_5'] = .9
    with pytest.raises(ValueError, match='score mismatch'):
        weighted_crossfit(f, ())
