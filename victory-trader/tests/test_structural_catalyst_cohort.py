from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from victory_trader import structural_catalyst_cohort as exp
from victory_trader.feasible_upside_observability import CEILING

ACC = '0001234567-26-000001'


def header(stamp='20260511100000', accession=ACC, form='8-K'):
    return f'<SEC-HEADER>\n<ACCEPTANCE-DATETIME>{stamp}\nACCESSION NUMBER: {accession}\nCONFORMED SUBMISSION TYPE: {form}\n</SEC-HEADER>'


def test_raw_sec_clock_is_eastern_with_dst_and_identity_validation():
    assert exp.acceptance_header(header(), ACC) == pd.Timestamp('2026-05-11T14:00:00Z')
    assert exp.acceptance_header(header('20260112100000'), ACC) == pd.Timestamp('2026-01-12T15:00:00Z')
    with pytest.raises(ValueError):
        exp.acceptance_header(header(accession='0001234567-26-000002'), ACC)
    with pytest.raises(ValueError):
        exp.acceptance_header(header(form='8-K/A'), ACC)


def test_clock_lag_weekend_roll_and_held_out_boundary():
    day, clock, bucket = exp.catalyst_clock(pd.Timestamp('2026-05-11T14:00:00Z'))
    assert day == '2026-05-11' and pd.to_datetime(clock, unit='ms', utc=True) == pd.Timestamp('2026-05-11T14:03:00Z')
    assert bucket == 'intraday'
    day, clock, _ = exp.catalyst_clock(pd.Timestamp('2026-05-15T21:00:00Z'))
    assert day == '2026-05-18' and pd.to_datetime(clock, unit='ms', utc=True) == pd.Timestamp('2026-05-18T13:35:00Z')
    with pytest.raises(ValueError):
        exp.catalyst_clock(pd.Timestamp('2026-05-29T21:00:00Z'))


def test_census_contains_execution_day_without_hot_admission_and_no_foreknowledge():
    s = {'ticker': 'XYZ', 'execution_date': '2026-05-11', 'adjustment_type': 'reverse_split', 'split_from': 10, 'split_to': 1, 'id': 's'}
    client = SimpleNamespace(news=lambda *a, **k: [])
    anchors, _, _ = exp.census_anchors([s], [], client, None)
    assert set(anchors.offset) == {-1, 0, 1, 5}
    assert anchors.loc[anchors.offset.eq(0), 'trading_day'].iloc[0] == '2026-05-11'
    assert anchors.loc[anchors.offset.eq(0), 'match_price_factor'].iloc[0] == 10
    assert not anchors.causal_event_knowledge_verified.any()


def test_unknown_exact_timestamp_not_invented_from_filing_date(monkeypatch):
    monkeypatch.setattr(exp, 'selected_accession', lambda a: True)
    f = {'accession_number': ACC, 'filing_date': '2026-05-11', 'tickers': ['XYZ'], 'cik': '1234567'}
    sec = SimpleNamespace(get=lambda *a: (_ for _ in ()).throw(PermissionError('private source details')))
    anchors, ledger, _ = exp.census_anchors([], [f], None, sec)
    assert anchors.empty and ledger[0]['status'] == 'timing_unknown'
    assert ledger[0]['error_type'] == 'PermissionError'
    assert 'private source' not in str(ledger)


def test_deduplicated_filing_categories_keep_exact_clock(monkeypatch):
    monkeypatch.setattr(exp, 'selected_accession', lambda a: True)
    f = {'accession_number': ACC, 'filing_date': '2026-05-11', 'tickers': ['XYZ'], 'cik': '1234567', 'primary_category': 'earnings'}
    sec = SimpleNamespace(get=lambda *a: header())
    anchors, ledger, _ = exp.census_anchors([], [f, {**f, 'primary_category': 'financing'}], None, sec)
    assert len(anchors) == len(ledger) == 1
    assert 'earnings' in anchors.categories.iloc[0] and 'financing' in anchors.categories.iloc[0]
    assert anchors.decision_t.iloc[0] > pd.Timestamp(anchors.accepted_utc.iloc[0]).timestamp()*1000


def test_matching_uses_prior_context_and_new_share_basis(monkeypatch):
    rows = [{'T': 'XYZ', 'c': 1., 'v': 100000}, *[{'T': t, 'c': 10., 'v': 10000} for t in ('A', 'B', 'C', 'D')]]
    monkeypatch.setattr(exp, 'grouped_daily', lambda c, day: {'results': rows} if day == date(2026, 5, 8) else pytest.fail('future price request'))
    client = SimpleNamespace(ticker_details=lambda *a: {'results': {'type': 'CS', 'locale': 'us', 'market': 'stocks'}})
    anchors = pd.DataFrame([{'anchor_id': 's', 'root': 's', 'ticker': 'XYZ', 'trading_day': '2026-05-11', 'decision_t': 0, 'family': 'split_execution', 'match_price_factor': 10.}])
    manifest, ledger = exp.match_cohort(anchors, [], [], client)
    assert len(manifest) == 4 and ledger[0]['status'] == 'matched'
    assert manifest.iloc[0].prior_price == 10.
    assert list(manifest.loc[manifest.arm.eq('control'), 'ticker']) == ['A', 'B', 'C']


def test_next_print_stop_is_absorbing_and_pending_gap_not_skipped():
    bars = pd.DataFrame({'t': [0, 1000, 2000, 100000, 3600000],
                         'o': [10., 10., 9.5, 11., 15.], 'c': [10., 9.5, 9.5, 11., 15.]})
    result = exp.path_labels(bars, 0, 4000000)
    assert result['label_complete'] and result[CEILING] < 5
    assert result['label_whole_max_net_pct'] > 40
    assert result['gap60_observed']
    bars = bars.loc[bars.t.lt(3600000)]
    # Complete absorbing stop still exists; no requirement for later recovery.
    assert exp.path_labels(bars, 0, 4000000)['label_complete']


def test_missing_terminal_print_stays_censored_and_all_member_bounds():
    bars = pd.DataFrame({'t': [0, 1000], 'o': [10., 10.], 'c': [10., 10.]})
    assert not exp.path_labels(bars, 0, 4000000)['label_complete']
    frame = pd.DataFrame({'label_complete': [True, False], CEILING: [6., np.nan]})
    report = exp.prevalence_bounds(frame)
    assert report['all_member_bounds'] == [.5, 1.] and report['net5_rate'] == 1.


def test_tiny_extreme_signal_does_not_promote():
    rows = []
    for i, arm in enumerate(('event', 'control', 'control', 'control')):
        rows.append({'anchor_id': 's', 'root': 's', 'ticker': str(i), 'trading_day': '2026-05-11',
                     'family': 'split_execution', 'arm': arm, 'label_complete': True, CEILING: 100 if i == 0 else -5})
    report = exp.paired_report(pd.DataFrame(rows), 'split_execution', None)
    assert report['status'] == 'underpowered'


def test_failed_histories_remain_unknown_and_have_label_column():
    def fail(*args, **kwargs):
        raise RuntimeError('private token')
    manifest = pd.DataFrame([{'ticker': 'XYZ', 'trading_day': '2026-05-11', 'decision_t': 0}])
    labels = exp.label_manifest(manifest, SimpleNamespace(second_bars_range=fail))
    assert not labels.label_complete.any() and labels[CEILING].isna().all()
    assert 'private token' not in str(labels.to_dict())


def test_sec_feasibility_freezes_membership_before_headers_and_never_loads_prices(tmp_path, monkeypatch):
    raw = f'1234567|Example Corp|8-K|2026-05-11|edgar/data/1234567/{ACC}.txt\n'
    response = SimpleNamespace(content=raw.encode(), raise_for_status=lambda: None)
    monkeypatch.setattr(exp.requests, 'get', lambda *a, **k: response)
    monkeypatch.setattr(exp, 'selected_accession', lambda a: True)

    class Sec:
        requests = 1
        blocked = False

        def __init__(self, *args):
            pass

        def get(self, cik, accession):
            assert (tmp_path/'sec-selected-membership.json').exists()
            assert cik == '1234567' and accession == ACC
            return header()

    monkeypatch.setattr(exp, 'SecHeaders', Sec)
    result = exp.sec_feasibility(tmp_path)
    assert result['selected_accessions'] == result['exact_headers'] == 1
    assert result['coverage'] == 1 and not result['market_prices_loaded']
    assert result['stage'] == 'SEC_only_source_feasibility'
