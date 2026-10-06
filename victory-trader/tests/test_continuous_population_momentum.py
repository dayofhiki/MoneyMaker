import numpy as np
import pandas as pd
import pytest

from victory_trader import continuous_population_momentum as exp
from victory_trader.compact_momentum_representation import independent_weights
from victory_trader.frozen_momentum_may_transport import identity_hash
from victory_trader.preentry_momentum_observability import clock_features


def fixture(day='2026-05-05'):
    hot = int(pd.Timestamp(day+'T14:00:00Z').timestamp()*1000)
    ticker = next('T'+str(i) for i in range(1000) if identity_hash(day, 'T'+str(i), hot) < 64)
    cohort = pd.DataFrame([{'trading_day': day, 'ticker': ticker, 'hot_t': hot,
        'sample_hash_byte': identity_hash(day, ticker, hot), 'in_broad': True, 'in_selected': False,
        **dict.fromkeys(exp.CONTROL[:4], 1.)}])
    bars = pd.DataFrame({'t': hot+np.array([-200000, -1000, 0, 1000, 299000, 300000, 3600000]),
        'o': [10.]*7, 'c': [10.]*7, 'v': [1.]*7, 'n': [1.]*7})
    contexts = {(day, ticker): (bars, hot-1800000, hot+21600000)}
    return cohort, contexts


def test_completed_clock_boundary_and_no_forward_fill():
    cohort, contexts = fixture()
    manifest, ledger = exp.clock_manifest(cohort, contexts)
    assert list(manifest.decision_t-cohort.hot_t.iloc[0]) == [1000, 2000, 300000]
    assert ledger.observed_states.iloc[0] == 3
    assert exp.CEILING not in manifest


def test_clock_membership_cannot_use_future_outcomes():
    cohort, contexts = fixture()
    before, _ = exp.clock_manifest(cohort, contexts)
    cohort['target_future_return'] = 1e10
    bars = next(iter(contexts.values()))[0]
    bars.loc[bars.t.gt(cohort.hot_t.iloc[0]+300000), ['o', 'c']] = 1e9
    after, _ = exp.clock_manifest(cohort, contexts)
    pd.testing.assert_frame_equal(before, after[list(before)])


def test_empty_episode_remains_in_ledger():
    cohort, contexts = fixture()
    manifest, ledger = exp.clock_manifest(cohort, {})
    assert manifest.empty and len(ledger) == 1
    assert ledger.observed_states.iloc[0] == 0 and ledger.missing_reason.notna().all()


def test_scope_and_duplicate_clocks_rejected():
    cohort, contexts = fixture('2026-06-01')
    with pytest.raises(ValueError):
        exp.clock_manifest(cohort, contexts)
    cohort, contexts = fixture()
    key, (bars, opening, closing) = next(iter(contexts.items()))
    contexts[key] = (pd.concat([bars, bars.iloc[2:3]]).sort_values('t'), opening, closing)
    with pytest.raises(ValueError):
        exp.clock_manifest(cohort, contexts)


def test_later_prices_do_not_enter_current_clock_features():
    cohort, contexts = fixture()
    manifest, _ = exp.clock_manifest(cohort, contexts)
    first = manifest.iloc[:1]
    before = clock_features(first, contexts)
    bars = next(iter(contexts.values()))[0]
    bars.loc[bars.t.ge(first.decision_t.iloc[0]), ['o', 'c', 'v', 'n']] = 100000.
    after = clock_features(first, contexts)
    pd.testing.assert_frame_equal(before, after)


def test_terminal_missingness_is_censored_not_negative():
    cohort, contexts = fixture()
    key, (bars, opening, closing) = next(iter(contexts.items()))
    contexts[key] = (bars.loc[bars.t.lt(cohort.hot_t.iloc[0]+3600000)], opening, closing)
    manifest, _ = exp.clock_manifest(cohort, contexts)
    states = exp.label_states(manifest, contexts)
    assert not states.label_complete.any() and states[exp.CEILING].isna().all()


def test_matched_heads_have_identical_episode_support_and_direction_rows():
    cohort, contexts = fixture()
    manifest, _ = exp.clock_manifest(cohort, contexts)
    states = exp.label_states(manifest, contexts)
    first = states.iloc[:1].copy()
    rows = exp.fitting_rows(first, states)
    assert rows['C'] is rows['Q']
    assert len(rows['B']) == 1 and len(rows['H']) == 3
    changed = states.copy()
    changed.loc[0, exp.CEILING] += 1
    with pytest.raises(AssertionError):
        exp.fitting_rows(first, changed)
    changed = states.copy()
    changed['ticker'] = 'OUTSIDE'
    with pytest.raises(ValueError):
        exp.fitting_rows(first, changed)


def test_duplicate_seconds_do_not_increase_effective_episode_weight():
    frame = pd.DataFrame({'trading_day': ['2026-05-05']*4+['2026-05-06'],
        'ticker': ['A', 'B', 'B', 'B', 'C'], 'hot_t': [0]*5})
    weights = independent_weights(frame)
    totals = frame.assign(weight=weights).groupby(exp.KEYS).weight.sum()
    assert weights.sum() == pytest.approx(3)
    assert totals.loc[('2026-05-05', 'A', 0)] == pytest.approx(totals.loc[('2026-05-05', 'B', 0)])
    by_day = frame.assign(weight=weights).groupby('trading_day').weight.sum()
    assert by_day.iloc[0] == pytest.approx(by_day.iloc[1])


def test_chronological_fit_excludes_current_and_future_days(monkeypatch):
    firsts, states = [], []
    for day in exp.CROSSFIT_DAYS:
        cohort, contexts = fixture(day)
        manifest, _ = exp.clock_manifest(cohort, contexts)
        labeled = exp.label_states(manifest, contexts)
        firsts.append(labeled.iloc[:1])
        states.append(labeled)
    def score(rows, held):
        assert all(f.trading_day.max() < held.trading_day.min() for f in rows.values())
        return held, {a: {'fit_days': sorted(f.trading_day.unique())} for a, f in rows.items()}
    monkeypatch.setattr(exp, 'score_heads', score)
    scored, folds = exp.chronological(pd.concat(firsts), pd.concat(states))
    assert len(scored) == 3 and set(folds) == set(exp.CROSSFIT_DAYS[1:])


def test_invalid_training_window_rejected():
    cohort, contexts = fixture()
    manifest, _ = exp.clock_manifest(cohort, contexts)
    manifest.loc[0, 'decision_t'] = cohort.hot_t.iloc[0]+300001
    with pytest.raises(ValueError):
        exp.label_states(manifest, contexts)


def test_evaluation_guard_allows_only_previous_Q_refit():
    held = pd.DataFrame({'ticker': ['A'], 'decision_t': [1000], exp.CEILING: [5.],
        'M_p_net_5': [.1], 'Q_p_net_5': [.2], 'Q_prior_net_5': [.3]})
    saved = held.copy()
    saved['Q_p_net_5'] = .4
    exp.verify_evaluation(held, saved)
    for column in ('decision_t', exp.CEILING, 'M_p_net_5'):
        changed = saved.copy()
        changed[column] += 1
        with pytest.raises(AssertionError):
            exp.verify_evaluation(held, changed)
