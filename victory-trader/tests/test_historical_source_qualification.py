import copy
import json

import pytest
import requests

from victory_trader import historical_source_qualification as source

BASE = 1778097540000
NS = BASE*1000000+123


def plan(windows=1):
    return {'full_identities': 1235, 'streams': list(source.STREAMS),
        'qualification_windows': [{'trading_day': '2026-05-06', 'ticker': 'ABVC', 'hot_t': BASE, 'scope': 'training', 'start_ms': BASE, 'end_ms': BASE+60000} for _ in range(windows)],
        'limits': {'http_attempts': 96, 'pages_per_pair': 2, 'bytes_per_page': 2097152,
            'total_response_bytes': 33554432, 'request_interval_seconds': 12.5, 'timeout_seconds': 30, 'tick_page_limit': 100}}


class Response:
    def __init__(self, payload, status=200, raw=None):
        self.status_code = status
        self.raw = raw if raw is not None else json.dumps(payload).encode()

    def iter_content(self, chunk_size):
        for position in range(0, len(self.raw), chunk_size):
            yield self.raw[position:position+chunk_size]

    def close(self):
        pass


def payload_for(url):
    ticks = '/v3/' in url
    row = {'sip_timestamp': NS, 'participant_timestamp': NS-3, 'conditions': None} if ticks else {'t': BASE, 'o': 1., 'c': 1., 'v': 100, 'n': 1}
    return {'status': 'OK', 'adjusted': False, 'results': [row], 'resultsCount': 1}


def success(url, **kwargs):
    assert kwargs['allow_redirects'] is False
    assert 'apiKey' not in kwargs['params']
    return Response(payload_for(url))


def collect(tmp_path, p=None, get=success, secret='TEST-CREDENTIAL'):
    return source.collect(p or plan(), tmp_path, secret, get=get, clock=lambda: 0., sleep=lambda seconds: None)


def test_complete_four_streams_and_exact_nanoseconds(tmp_path):
    manifest = collect(tmp_path)
    report = source.analyze(tmp_path, plan())
    assert manifest['http_attempts'] == 4
    assert report['all_required_sampled_sources_available']
    assert report['pairs'][2]['first_event_timestamp'] == NS
    assert report['pairs'][3]['missing_optional_fields']['conditions'] == 1
    assert report['new_economic_labels'] == 0


def test_auth_in_header_only_and_fixed_endpoint_windows():
    p, params = source.request_for(plan()['qualification_windows'][0], 'quotes', 100)
    assert p == '/v3/quotes/ABVC'
    assert params['timestamp.gte'] == str(BASE*1000000)
    assert params['timestamp.lt'] == str((BASE+60000)*1000000)
    p, params = source.request_for(plan()['qualification_windows'][0], 'seconds', 100)
    assert p.endswith('/'+str(BASE)+'/'+str(BASE+59999))
    assert params['adjusted'] == 'false'


@pytest.mark.parametrize('link', ['https://evil.example/v3/quotes/ABVC', 'http://api.massive.com/v3/quotes/ABVC', 'https://u:p@api.massive.com/v3/quotes/ABVC', 'https://api.massive.com:443/v3/quotes/ABVC', 'https://api.massive.com/v3/quotes/OTHER'])
def test_unsafe_pagination_rejected(link):
    with pytest.raises(ValueError, match='unsafe'):
        source.next_request(link, '/v3/quotes/ABVC')


def test_pagination_credentials_removed():
    path, params = source.next_request('https://api.massive.com/v3/quotes/ABVC?cursor=abc&apiKey=secret', '/v3/quotes/ABVC')
    assert path == '/v3/quotes/ABVC' and params == {'cursor': 'abc'}


def test_two_page_exact_replay(tmp_path):
    calls = {}
    def paged(url, **kwargs):
        calls[url] = calls.get(url, 0)+1
        x = payload_for(url)
        if calls[url] == 1:
            x['next_url'] = url+'?cursor=second'
        return Response(x)
    collect(tmp_path, get=paged)
    assert source.analyze(tmp_path, plan())['market_http_attempts'] == 8


def test_page_limit_is_partial_and_not_empty_or_complete(tmp_path):
    def endless(url, **kwargs):
        x = payload_for(url)
        x['next_url'] = url+'?cursor='+str(kwargs['params'].get('cursor', ''))+'x'
        return Response(x)
    collect(tmp_path, get=endless)
    x = source.analyze(tmp_path, plan())
    assert not x['all_required_sampled_sources_available']
    assert x['all_required_sampled_endpoint_access_observed']
    assert all(p['collection_state'] == 'PARTIAL_PAGE_LIMIT' for p in x['pairs'])


def test_repeated_pagination_does_not_loop(tmp_path):
    def repeated(url, **kwargs):
        x = payload_for(url)
        x['next_url'] = url+'?cursor=same'
        return Response(x)
    p = plan()
    p['limits']['pages_per_pair'] = 3
    collect(tmp_path, p, repeated)
    assert all(x['collection_state'] == 'REPEATED_PAGINATION' for x in source.analyze(tmp_path, p)['pairs'])


def test_denial_retains_unrequested_pairs_without_claiming_denial(tmp_path):
    def denied(url, **kwargs):
        return Response({'status': 'NOT_AUTHORIZED'}, 403)
    p = plan(2)
    collect(tmp_path, p, denied)
    x = source.analyze(tmp_path, p)
    assert x['market_http_attempts'] == 4 and len(x['pairs']) == 8
    assert x['streams']['quotes']['state_counts'] == {'DENIED_403': 1, 'SKIPPED_AFTER_DENIAL': 1}


@pytest.mark.parametrize('status,state', [(401, 'SKIPPED_AFTER_AUTH_401'), (429, 'SKIPPED_AFTER_RATE_LIMIT_429')])
def test_global_stops_do_not_retry(tmp_path, status, state):
    collect(tmp_path, get=lambda *args, **kwargs: Response({'status': 'ERROR'}, status))
    x = source.analyze(tmp_path, plan())
    assert x['market_http_attempts'] == 1
    assert [p['collection_state'] for p in x['pairs']][1:] == [state]*3


def test_missing_credentials_never_requests_or_searches(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError('network must not be called')
    collect(tmp_path, get=forbidden, secret='')
    assert source.analyze(tmp_path, plan())['market_http_attempts'] == 0


def test_secret_bearing_wire_withheld_but_sanitized_payload_retained(tmp_path):
    def echoed(url, **kwargs):
        x = payload_for(url)
        x['apiKey'] = 'TEST-CREDENTIAL'
        x['message'] = 'apiKey=TEST-CREDENTIAL'
        return Response(x)
    collect(tmp_path, get=echoed)
    assert not list(tmp_path.rglob('*.wire'))
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert b'TEST-CREDENTIAL' not in path.read_bytes()
    assert all(p['wire_body_withheld_pages'] == 1 for p in source.analyze(tmp_path, plan())['pairs'])


def test_request_budget_does_not_fabricate_completeness(tmp_path):
    p = plan()
    p['limits']['http_attempts'] = 1
    collect(tmp_path, p)
    x = source.analyze(tmp_path, p)
    assert x['market_http_attempts'] == 1 and not x['all_required_sampled_sources_available']


def test_body_budget_prefix_not_a_complete_response(tmp_path):
    p = plan()
    p['limits']['bytes_per_page'] = 10
    collect(tmp_path, p)
    x = source.analyze(tmp_path, p)
    assert x['retained_response_bytes'] == 10 and x['market_http_attempts'] == 1
    assert x['pairs'][0]['collection_state'] == 'PARTIAL_BODY_BUDGET'


def test_transport_error_text_never_exposes_secrets(tmp_path):
    def fail(*args, **kwargs):
        raise requests.ConnectionError('TEST-CREDENTIAL https://secret.example')
    collect(tmp_path, get=fail)
    manifest = (tmp_path/'request333-page-manifest.json').read_text()
    assert 'TEST-CREDENTIAL' not in manifest and 'secret.example' not in manifest


def test_empty_success_is_returned_empty_without_no_trade_inference(tmp_path):
    collect(tmp_path, get=lambda *args, **kwargs: Response({'status': 'OK', 'adjusted': False, 'resultsCount': 0}))
    x = source.analyze(tmp_path, plan())
    assert all(p['empty_returned_stream'] for p in x['pairs'])
    assert not x['full_census_support_or_gap_causes_claimed']


@pytest.mark.parametrize('row', [{'sip_timestamp': float(NS)}, {'participant_timestamp': NS}])
def test_float_or_missing_ns_rejected(row):
    with pytest.raises(ValueError, match='integer'):
        source.page_rows({'status': 'OK', 'results': [row]}, 'quotes')


def test_provider_error_not_http_success(tmp_path):
    collect(tmp_path, get=lambda *args, **kwargs: Response({'status': 'ERROR', 'results': []}))
    assert not source.analyze(tmp_path, plan())['all_required_sampled_sources_available']


def test_adjustment_response_unknown_cannot_pass():
    with pytest.raises(ValueError, match='unadjusted'):
        source.page_rows({'status': 'OK', 'results': []}, 'seconds')


def test_out_of_window_records_retained_but_not_qualified(tmp_path):
    def outside(url, **kwargs):
        x = payload_for(url)
        if '/v3/' in url:
            x['results'][0]['sip_timestamp'] = (BASE+60000)*1000000
        return Response(x)
    collect(tmp_path, get=outside)
    x = source.analyze(tmp_path, plan())
    assert x['pairs'][2]['out_of_window_rows'] == 1 and not x['all_required_sampled_sources_available']


def test_page_digest_and_manifest_order_are_authenticated(tmp_path):
    collect(tmp_path)
    path = next(tmp_path.glob('pages/*.json'))
    path.write_text('{}')
    with pytest.raises(ValueError, match='digest'):
        source.analyze(tmp_path, plan())


def test_changed_registration_rejected(tmp_path):
    collect(tmp_path)
    p = copy.deepcopy(plan())
    p['qualification_windows'][0]['end_ms'] += 1
    with pytest.raises(ValueError, match='plan changed'):
        source.analyze(tmp_path, p)


def test_source_acquisition_cannot_be_overwritten(tmp_path):
    collect(tmp_path)
    with pytest.raises(ValueError, match='overwrite'):
        collect(tmp_path)


def test_invalid_json_or_nan_is_preserved_as_failure(tmp_path):
    collect(tmp_path, get=lambda *args, **kwargs: Response({}, raw=b'{"status":"OK","results":[NaN]}'))
    report = source.analyze(tmp_path, plan())
    assert not report['all_required_sampled_sources_available']
    assert all(p['collection_state'] == 'INVALID_JSON' for p in report['pairs'])


def test_unsafe_original_link_cannot_become_safe_by_redaction(tmp_path):
    def unsafe(url, **kwargs):
        x = payload_for(url)
        x['next_url'] = url.replace('api.massive.com', 'api.massive.com:444')+'?cursor=x'
        return Response(x)
    collect(tmp_path, get=unsafe)
    assert all(p['collection_state'] == 'UNSAFE_PAGINATION' for p in source.analyze(tmp_path, plan())['pairs'])


def test_optional_pagination_value_must_have_correct_type():
    with pytest.raises(ValueError, match='pagination_schema'):
        source.page_rows({'status': 'OK', 'results': [], 'next_url': 0}, 'quotes')


def test_minimum_request_start_spacing_is_enforced(tmp_path):
    waits = []
    source.collect(plan(), tmp_path, 'TEST-CREDENTIAL', success, clock=lambda: 0., sleep=waits.append)
    assert waits == [12.5]*3
