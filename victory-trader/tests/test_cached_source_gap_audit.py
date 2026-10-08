import copy

import numpy as np
import pandas as pd
import pytest

from victory_trader import cached_source_gap_audit as audit
from victory_trader.pending_exit_integrity import session_limits


@pytest.mark.parametrize('gap,expected', [(0, 0), (3, 0), (3.001, 1), (30, 1), (30.001, 2), (60, 2), (60.001, 3), (np.nan, 4)])
def test_fixed_bin_edges(gap, expected):
    assert audit.gap_bin(gap) == audit.BINS[expected]


def test_negative_gap_rejected():
    with pytest.raises(ValueError):
        audit.gap_bin(-1)


def test_later_record_is_not_missing_or_causal_evidence():
    assert audit.next_reference(np.array([10000]), 1000, 20000) == (10000., 9., 'late_before_boundary')
    assert audit.next_reference(np.array([10000]), 9000, 10000) == (10000., 1., 'at_or_after_boundary')
    assert audit.next_reference(np.array([10000]), 9000, 20000) == (10000., 1., 'within3s_before_boundary')


def test_missing_pair_and_end_of_retained_path_are_distinct():
    assert audit.next_reference(None, 2000, 3000)[2] == 'raw_pair_absent'
    assert audit.next_reference(np.array([1000]), 2000, 3000)[2] == 'no_later_cached_reference'
    assert audit.next_reference(np.array([1000]), np.nan, 3000)[2] == 'no_submission'


def example():
    opening, closing = session_limits('2026-05-06')
    hot = opening+60000
    clocks = pd.DataFrame([dict(trading_day='2026-05-06', ticker='XYZ', hot_t=hot, decision_t=hot+1000, phase='EARLY')])
    timing = clocks.assign(cost_compatible=True, enter_resolved=False, enter_reason='entry_missing_or_late', entry_fill_t=np.nan, exit_submission_t=np.nan, exit_fill_t=np.nan, next_entry_reference_gap_s=9., next_exit_reference_gap_s=np.nan)
    cache = {('2026-05-06', 'XYZ'): np.array([hot, hot+10000])}
    return clocks, timing, cache, closing


def test_exact_clock_and_gap_audit_preserves_missingness():
    clocks, timing, cache, _ = example()
    result = audit.clock_audit(clocks, timing, cache)
    assert len(result) == 1 and np.isnan(result.exit_fill_t.iloc[0])
    assert result.entry_reference_anatomy.iloc[0] == 'late_before_boundary'
    assert result.gap_cause.iloc[0] == 'UNIDENTIFIED_NO_INDEPENDENT_STREAM'
    assert 'enter_base_pct' not in result and 'entry_reference_price' not in result


def test_changed_gap_fails_without_relabeling():
    clocks, timing, cache, _ = example()
    timing['next_entry_reference_gap_s'] = 10.
    with pytest.raises(ValueError, match='gap changed'):
        audit.clock_audit(clocks, timing, cache)


def test_changed_clock_fails():
    clocks, timing, cache, _ = example()
    timing['decision_t'] += 1000
    with pytest.raises(AssertionError):
        audit.clock_audit(clocks, timing, cache)


def test_all_zero_row_identities_remain_unknown():
    clocks, _, _, closing = example()
    registered = clocks[audit.KEYS].assign(terminal_t=min(int(clocks.hot_t.iloc[0])+3600000, closing), early_clocks=0, late_clocks=0, extended_clocks=0, raw_pair_available=False, no_observation=True)
    ledger = audit.identity_ledger(registered, {})
    assert len(ledger) == 1 and ledger.no_observation.iloc[0]
    assert ledger.full_session_cached_seconds.iloc[0] == 0
    assert ledger.response_completeness.iloc[0] == 'UNKNOWN_NOT_RETAINED'


def test_raw_scope_validation_and_duplicates():
    opening, _ = session_limits('2026-05-06')
    raw = pd.DataFrame([dict(trading_day='2026-05-06', ticker='XYZ', t=opening)])
    assert len(audit.time_cache(raw)) == 1
    with pytest.raises(ValueError, match='unique'):
        audit.time_cache(pd.concat([raw, raw]))
    with pytest.raises(ValueError, match='regular-session'):
        audit.time_cache(raw.assign(t=opening-1000))


def test_summary_does_not_treat_late_print_as_resolved():
    clocks, timing, cache, _ = example()
    summary = audit.clock_summary(audit.clock_audit(clocks, timing, cache))
    assert summary['pooled_compatible_resolution'] == 0
    assert summary['compatible_missing_entry']['entry_reference_anatomy'] == {'late_before_boundary': 1}


def test_code_and_reported_success_cannot_become_response_proof(tmp_path):
    for number, name in ((311, 'request311.json'), (315, 'request315-acquisition.json')):
        folder = tmp_path/f'official{number}'
        folder.mkdir()
        (folder/name).write_text('{"acquisition":{"failures":[],"client_stats":{"network_requests":1}}}')
    evidence = {'scope': 'fixed packages', 'original_sources': {'311': {'code_contract': {'pagination': 'while next_url'}, 'retained_response_evidence': {k: None for k in ('request_bounds', 'adjusted', 'statuses', 'page_counts', 'terminal_next_url', 'payload_sha256')}, 'independent_full_window_streams': []}}}
    result = audit.lineage_report(tmp_path, evidence)
    assert result['response_completeness'] == 'UNKNOWN_NOT_RETAINED'
    assert result['gap_cause'] == 'UNIDENTIFIED'
    independent = copy.deepcopy(evidence)
    independent['original_sources']['311']['independent_full_window_streams'] = ['derived_seconds']
    with pytest.raises(ValueError, match='new registered reconciliation'):
        audit.lineage_report(tmp_path, independent)
