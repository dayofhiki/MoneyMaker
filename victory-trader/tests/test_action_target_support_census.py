import json

import numpy as np
import pandas as pd
import pytest

from victory_trader import action_target_support_census as census
from victory_trader import current_state_action_policy as contract

DAY = '2026-05-06'
HOT = census.session_limits(DAY)[0]+60000


def identities(tickers=('A',)):
    return pd.DataFrame([{'trading_day': DAY, 'ticker': t, 'hot_t': HOT} for t in tickers])


def raw(offsets, ticker='A', prices=None):
    price = np.full(len(offsets), 2.) if prices is None else np.asarray(prices, float)
    return pd.DataFrame({'trading_day': DAY, 'ticker': ticker, 't': HOT+np.asarray(offsets, dtype=np.int64),
        'o': price, 'c': price, 'h': price*2, 'l': price/2, 'v': 1., 'n': 1.})


def labeled_fixture():
    frame = raw([0, 1000, 2000, 301000], prices=[2., 2., 2., 2.])
    clocks, _ = census.clock_manifest(identities(), frame[['trading_day', 'ticker', 't']])
    result = census.label_clocks(clocks, frame, (DAY,))
    result['enter_resolved'] = [True, False, True, True]
    result['enter_base_pct'] = [1., np.nan, -1., 2.]
    for scenario in ('light', 'stress'):
        result[f'enter_{scenario}_pct'] = result.enter_base_pct
    result['cost_compatible'] = True
    return result


def test_completed_bucket_boundaries_and_first_print_only():
    bars = raw([299000, 300000, 301000, 329000, 330000, 331000, 3598000, 3599000])
    clocks, ledger = census.clock_manifest(identities(), bars[['trading_day', 'ticker', 't']])
    assert (clocks.decision_t-HOT).tolist() == [300000, 301000, 331000, 3599000]
    assert clocks.phase.tolist() == ['EARLY', 'ADDITIONAL_LATE', 'ADDITIONAL_LATE', 'ADDITIONAL_LATE']
    assert ledger.early_clocks.iloc[0] == 1
    assert ledger.late_clocks.iloc[0] == 3


def test_empty_buckets_create_no_observation_and_identity_survives():
    clocks, ledger = census.clock_manifest(identities(('A', 'B')), raw([500000])[['trading_day', 'ticker', 't']])
    assert len(clocks) == 1
    assert len(ledger) == 2
    assert ledger.no_observation.tolist() == [False, True]
    assert ledger.raw_pair_available.tolist() == [True, False]


def test_clocks_do_not_use_future_prices_density_or_labels():
    bars = raw([0, 300000, 320000, 330000, 350000, 400000])
    original, _ = census.clock_manifest(identities(), bars[['trading_day', 'ticker', 't']])
    bars['c'], bars['o'], bars['future_outcome'] = 10000., .0001, 1000.
    modified, _ = census.clock_manifest(identities(), bars)
    pd.testing.assert_frame_equal(original, modified)
    for cut in [HOT+1000, HOT+301000, HOT+331000, HOT+351000]:
        prefix, _ = census.clock_manifest(identities(), bars.loc[bars.t+1000 <= cut])
        pd.testing.assert_frame_equal(prefix, original.loc[original.decision_t <= cut].reset_index(drop=True))


def test_session_terminal_and_duplicate_timestamps():
    closing = census.session_limits(DAY)[1]
    identity = identities().assign(hot_t=closing-2000)
    bars = raw([0, 1000, 2000]).assign(t=[closing-2000, closing-1000, closing])
    clocks, _ = census.clock_manifest(identity, bars[['trading_day', 'ticker', 't']])
    assert clocks.decision_t.tolist() == [closing-1000]
    with pytest.raises(ValueError, match='duplicate'):
        census.clock_manifest(identity, pd.concat([bars, bars]))


def test_known_cost_is_prefix_causal_and_ignores_future_labels():
    bars = raw([0, 1000, 300000, 301000, 360000], prices=[2, 2.1, 2.2, 3, 4])
    clocks, _ = census.clock_manifest(identities(), bars[['trading_day', 'ticker', 't']])
    cache = contract.contexts(bars, (DAY,))
    for state in clocks.itertuples():
        selected = clocks.loc[clocks.decision_t == state.decision_t].assign(enter_base_pct=999.)
        before = census.known_costs(selected, cache)
        changed = bars.copy()
        changed.loc[changed.t+1000 > state.decision_t, ['o', 'c']] = .001
        after = census.known_costs(selected, contract.contexts(changed, (DAY,)))
        np.testing.assert_array_equal(before, after)


def test_late_label_is_identical_unmodified_r330_contract():
    bars = raw(np.arange(300000, 366000, 1000), prices=np.full(66, 2.))
    clocks, _ = census.clock_manifest(identities(), bars[['trading_day', 'ticker', 't']])
    labeled = census.label_clocks(clocks, bars, (DAY,))
    for state in labeled.itertuples():
        expected = contract.enter_outcome(bars, state.decision_t, HOT, census.session_limits(DAY)[1])
        for name, value in expected.items():
            actual = getattr(state, name)
            if isinstance(value, float) and np.isnan(value):
                assert np.isnan(actual)
            else:
                assert actual == value
        # Intrasecond lows cannot trigger completed-close stops.
        assert not state.stop_triggered


def test_missing_or_late_entry_remains_unknown_even_when_future_wins():
    bars = raw([300000, 310000, 311000], prices=[2, 4, 4])
    clocks, _ = census.clock_manifest(identities(), bars[['trading_day', 'ticker', 't']])
    labeled = census.label_clocks(clocks, bars, (DAY,))
    assert labeled.enter_reason.iloc[0] == 'entry_missing_or_late'
    assert not labeled.enter_resolved.iloc[0]
    assert np.isnan(labeled.enter_base_pct.iloc[0])
    assert labeled.next_entry_reference_gap_s.iloc[0] == 9.


def test_causal_cost_veto_does_not_invent_cash():
    bars = raw(np.arange(300000, 366000, 1000), prices=np.full(66, .1))
    clocks, _ = census.clock_manifest(identities(), bars[['trading_day', 'ticker', 't']])
    labeled = census.label_clocks(clocks, bars, (DAY,))
    assert not labeled.cost_compatible.any()
    assert labeled.cost_only_stop.all()
    assert labeled.enter_base_pct.dropna().lt(0).all()


def test_no_available_target_reweighting_and_union_phase_mass():
    frame = labeled_fixture()
    metrics = census.support_metrics(frame, identities(), 'EARLY')
    assert metrics['original_observed_episode_mass'] == 1.
    assert metrics['complete_original_weight_mass'] == pytest.approx(2/3)
    union = census.support_metrics(frame, identities(), 'EXTENDED')
    assert union['complete_original_weight_mass'] == .75
    assert union['phase_contribution_complete_weight_mass'] == {'EARLY': .5, 'ADDITIONAL_LATE': .25}
    assert union['scenario_payoffs_pct']['base']['full_mean'] is None
    assert union['scenario_payoffs_pct']['base']['conditional_known_mean'] == pytest.approx(2/3)


def test_independent_support_not_number_of_rows_and_unobserved_retained():
    frame = labeled_fixture()
    ledger = census.identity_support(frame, identities(('A', 'B')))
    assert len(ledger) == 2
    assert ledger.EXTENDED_any_complete_compatible.tolist() == [True, False]
    assert ledger.EXTENDED_any_positive_compatible.tolist() == [True, False]
    assert ledger.EARLY_any_complete.tolist() == [True, False]
    assert ledger.EXTENDED_rows.tolist() == [4, 0]
    assert census.support_metrics(frame, identities(('A', 'B')), 'EXTENDED')['complete_compatible_episodes'] == 1


def test_paired_gain_is_preserved_with_no_old_support_loss():
    frame = labeled_fixture()
    frame.loc[frame.phase.eq('EARLY'), ['enter_resolved', 'cost_compatible']] = False
    ledger = census.identity_support(frame, identities(('A', 'B')))
    assert ledger.any_complete_compatible_change.tolist() == ['gained', 'unchanged']
    assert ledger.any_positive_compatible_change.tolist() == ['gained', 'unchanged']


def test_bootstrap_includes_no_observation_identity_and_is_deterministic():
    ledger = census.identity_support(labeled_fixture(), identities(('A', 'B')))
    result = census.clustered_support(ledger, draws=50)
    assert result == census.clustered_support(ledger, draws=50)
    assert result['day']['metrics']['EXTENDED_any_complete_compatible_rate']['ci95'] == [.5, .5]
    assert result['ticker_day']['clusters'] == 2


def test_gate_cannot_use_unfiltered_or_evaluation_positive_support():
    m = {'complete_compatible_episodes': 149, 'positive_compatible_episodes': 29,
        'positive_compatible_dates': list(census.PRIMARY_DAYS),
        'nonpositive_compatible_dates': list(census.PRIMARY_DAYS), 'compatible_row_resolution': .899}
    integrity = {str(i): True for i in range(6)}
    checks = census.gates({'arms': {'EXTENDED': m}}, integrity)
    assert len(checks) == 10
    assert not checks['training_complete_compatible_episodes150']
    assert not checks['training_positive_compatible_episodes30']
    assert not checks['training_compatible_row_resolution90']
    m.update(complete_compatible_episodes=150, positive_compatible_episodes=30, compatible_row_resolution=.9)
    assert all(census.gates({'arms': {'EXTENDED': m}}, integrity).values())


def test_authentication_fails_before_reading_or_constructing_targets(tmp_path, monkeypatch):
    (tmp_path/'raw').write_bytes(b'wrong')
    pins = tmp_path/'pins.json'
    pins.write_text(json.dumps({'raw': 'incorrect_hash'}))
    def forbidden(*args, **kwargs):
        pytest.fail('table read before authentication')
    monkeypatch.setattr(pd, 'read_parquet', forbidden)
    with pytest.raises(ValueError, match='registered input'):
        census.load_inputs(tmp_path, pins)


def test_inherited_cost_target_missingness_and_identity_changes_rejected():
    frame = labeled_fixture()
    saved = frame.loc[frame.phase.eq('EARLY')].copy()
    census.verify_r330(frame, saved)
    for field in ('decision_t', 'known_cost_drag_pct', 'enter_base_pct'):
        broken = saved.copy()
        broken.loc[broken.index[0], field] += 1
        with pytest.raises(AssertionError):
            census.verify_r330(frame, broken)
    broken = saved.copy()
    broken.loc[broken.index[1], 'enter_base_pct'] = 0.
    with pytest.raises(ValueError, match='missingness'):
        census.verify_r330(frame, broken)


def test_early_identity_cannot_disappear_in_manifest_comparison():
    clocks, _ = census.clock_manifest(identities(), raw([0, 1000, 300000])[['trading_day', 'ticker', 't']])
    inherited = clocks.loc[clocks.phase.eq('EARLY')].copy()
    census.verify_early(clocks, inherited)
    with pytest.raises(AssertionError):
        census.verify_early(clocks.iloc[1:], inherited)


def test_empty_late_arm_reports_unknown_mean_without_invented_samples():
    frame = labeled_fixture().query("phase == 'EARLY'")
    result = census.support_metrics(frame, identities(), 'ADDITIONAL_LATE')
    assert result['observed_clocks'] == 0
    assert result['compatible_row_resolution'] is None
    assert result['scenario_payoffs_pct']['base']['full_mean'] is None


def test_atomic_manifest_is_readable_and_no_temp_remains(tmp_path):
    clocks, _ = census.clock_manifest(identities(), raw([0, 300000])[['trading_day', 'ticker', 't']])
    path = tmp_path/'clocks.parquet'
    census.atomic_parquet(clocks, path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), clocks)
    assert not path.with_suffix('.parquet.tmp').exists()


def test_two_stage_synthetic_census_preserves_censoring_and_stops_on_manifest_tamper(tmp_path, monkeypatch):
    populations, caches = {}, {}
    for scope, days in (('training', census.TRAIN_DAYS), ('evaluation', census.EVAL_DAYS)):
        members, pieces = [], []
        for day in days:
            hot = census.session_limits(day)[0]+60000
            members.append({'trading_day': day, 'ticker': 'A', 'hot_t': hot})
            piece = raw(np.arange(0, 362000, 1000))
            piece['trading_day'], piece['t'] = day, hot+np.arange(0, 362000, 1000)
            pieces.append(piece)
        populations[scope], caches[scope] = pd.DataFrame(members), pd.concat(pieces, ignore_index=True)
    monkeypatch.setattr(census, 'load_inputs', lambda *args: (populations, caches, {'synthetic_only': 'not_market_data'}))
    previous, output = tmp_path/'previous', tmp_path/'output'
    (previous/'official319').mkdir(parents=True)
    (previous/'official330').mkdir()
    for scope, days in (('training', census.TRAIN_DAYS), ('evaluation', census.EVAL_DAYS)):
        clocks, _ = census.clock_manifest(populations[scope], caches[scope][['trading_day', 'ticker', 't']])
        early = clocks.loc[clocks.phase.eq('EARLY')].reset_index(drop=True)
        old_name = 'training-states' if scope == 'training' else 'clock-manifest'
        early.to_parquet(previous/f'official319/request319-{old_name}.parquet', index=False)
        inherited = census.label_clocks(early, caches[scope], days)
        if scope == 'training':
            inherited = inherited.loc[inherited.trading_day.isin(census.PRIMARY_DAYS)]
        inherited.to_parquet(previous/f'official330/request330-{"training-states" if scope == "training" else "states"}.parquet', index=False)
    registration = census.save_clocks(previous, tmp_path/'unused-pins', output)
    assert registration['stage'] == 'timestamps_only_before_prices_and_targets'
    assert not (output/'request331.json').exists()
    frozen = output/'request331-clock-registration.json'
    result = census.census(previous, tmp_path/'unused-pins', output, frozen)
    assert result['model_fits'] == 0
    assert len(result['gate_checks']) == 10
    assert result['scope_reports']['training_primary']['arms']['EXTENDED']['identities'] == 3
    assert result['scope_reports']['evaluation']['arms']['EXTENDED']['scenario_payoffs_pct']['base']['full_mean'] is None
    assert all(result['integrity'].values())
    manifest = output/'request331-training-clocks.parquet'
    altered = pd.read_parquet(manifest)
    altered.loc[altered.index[0], 'decision_t'] += 1
    altered.to_parquet(manifest, index=False)
    with pytest.raises(ValueError, match='registered clock manifest changed'):
        census.census(previous, tmp_path/'unused-pins', output, frozen)
