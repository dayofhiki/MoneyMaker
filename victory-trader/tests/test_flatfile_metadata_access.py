import copy

import pytest
import requests

from victory_trader import flatfile_metadata_access as audit


def pairs():
    return [{'trading_day': f'2026-05-{day:02d}', 'stream': stream,
        'object_key': f'{prefix}/2026/05/2026-05-{day:02d}.csv.gz'}
        for day in (5, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20) for stream, prefix in audit.PREFIXES.items()]


class Response:
    def __init__(self, status=200, headers=None):
        self.status_code = status
        self.headers = {'Content-Length': '123456', 'ETag': 'digest'} if headers is None else headers

    def close(self):
        pass

    def iter_content(self, **kwargs):
        raise AssertionError('HEAD metadata must never read a body')


def collect(tmp_path, head=None, access='TEST-ACCESS', secret='TEST-SECRET'):
    return audit.collect(pairs(), tmp_path, access, secret, head or (lambda *args, **kwargs: Response()), clock=lambda: 0., sleep=lambda x: None)


def test_metadata_success_does_not_claim_market_support(tmp_path):
    report = audit.summarize(collect(tmp_path), pairs())
    assert report['http_attempts'] == 24
    assert report['market_body_bytes_read'] == 0 and report['market_rows_read'] == 0
    assert report['all_required_HEAD_access_observed']
    assert not report['GET_permission_or_record_completeness_claimed']


def test_signed_requests_have_no_credentials_in_saved_records(tmp_path):
    def head(url, **kwargs):
        assert kwargs['stream'] and kwargs['allow_redirects'] is False
        assert 'AWS4-HMAC-SHA256' in kwargs['headers']['Authorization']
        assert kwargs['headers']['X-Amz-Content-SHA256'] == audit.sha(b'')
        assert url.startswith(audit.ENDPOINT)
        return Response()
    collect(tmp_path, head)
    for path in tmp_path.iterdir():
        assert 'TEST-ACCESS' not in path.read_text() and 'TEST-SECRET' not in path.read_text()


@pytest.mark.parametrize('status,state', [(403, 'HEAD_REJECTED_403'), (404, 'NOT_FOUND_OR_UNDISCLOSED_404'), (301, 'REDIRECT_UNRESOLVED')])
def test_denial_missing_or_redirect_retains_all_pairs(tmp_path, status, state):
    report = audit.summarize(collect(tmp_path, lambda *args, **kwargs: Response(status)), pairs())
    assert not report['all_required_HEAD_access_observed']
    assert report['streams']['quotes']['state_counts'] == {state: 12}


def test_missing_credentials_never_discovers_or_requests(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError('network forbidden')
    report = audit.summarize(collect(tmp_path, forbidden, access=''), pairs())
    assert report['http_attempts'] == 0 and len(report['metadata_ledger']['rows']) == 24


def test_missing_size_is_unknown(tmp_path):
    report = audit.summarize(collect(tmp_path, lambda *args, **kwargs: Response(headers={})), pairs())
    assert report['streams']['quotes']['objects_with_unknown_size'] == 12


def test_errors_do_not_save_secret_exception_text(tmp_path):
    def fail(*args, **kwargs):
        raise requests.ConnectionError('TEST-SECRET')
    collect(tmp_path, fail)
    assert 'TEST-SECRET' not in (tmp_path/'request334-metadata.json').read_text()


def test_metadata_scope_and_body_budget_cannot_change(tmp_path):
    ledger = collect(tmp_path)
    other = copy.deepcopy(ledger)
    other['market_body_bytes_read'] = 1
    with pytest.raises(ValueError, match='resource'):
        audit.summarize(other, pairs())
    other = copy.deepcopy(ledger)
    other['rows'][0]['object_key'] += 'changed'
    with pytest.raises(ValueError, match='scope'):
        audit.summarize(other, pairs())


def test_request_spacing_is_fixed(tmp_path):
    waits = []
    audit.collect(pairs(), tmp_path, 'TEST-ACCESS', 'TEST-SECRET', lambda *args, **kwargs: Response(), clock=lambda: 0., sleep=waits.append)
    assert waits == [.5]*23


def test_metadata_cannot_be_overwritten(tmp_path):
    collect(tmp_path)
    with pytest.raises(ValueError, match='overwrite'):
        collect(tmp_path)


def test_undeclared_day_or_key_rejected_before_network(tmp_path):
    wrong = pairs()
    wrong[0]['trading_day'] = '2026-06-01'
    with pytest.raises(ValueError, match='registered'):
        audit.collect(wrong, tmp_path, 'TEST-ACCESS', 'TEST-SECRET', lambda *args, **kwargs: pytest.fail('network forbidden'))
