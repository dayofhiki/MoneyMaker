"""R334: fixed metadata-only HEADs; never download or inspect market rows."""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from botocore.auth import S3SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from .action_target_support_census import EVAL_DAYS, TRAIN_DAYS
from .historical_source_qualification import canonical, save_json, sha

PREFIXES = {'trades': 'us_stocks_sip/trades_v1', 'quotes': 'us_stocks_sip/quotes_v1'}
ENDPOINT = 'https://files.massive.com/flatfiles/'


def fixed_pairs():
    return [{'trading_day': day, 'stream': stream,
        'object_key': f'{prefix}/{day[:4]}/{day[5:7]}/{day}.csv.gz'}
        for day in sorted((*TRAIN_DAYS, *EVAL_DAYS)) for stream, prefix in PREFIXES.items()]


def declared_pairs(research):
    pins = json.loads((research/'request334-inputs.json').read_text())
    for name, digest in pins.items():
        if sha((research/name).read_bytes()) != digest:
            raise ValueError('registered R333 input changed')
    plan = json.loads((research/'request333-plan.json').read_text())
    days = sorted({x['trading_day'] for x in plan['qualification_windows']})
    if days != sorted((*TRAIN_DAYS, *EVAL_DAYS)):
        raise ValueError('exact12 original dates required')
    return fixed_pairs()


def signed_headers(url, access, secret):
    request = AWSRequest(method='HEAD', url=url)
    S3SigV4Auth(Credentials(access, secret), 's3', 'us-east-1').add_auth(request)
    return dict(request.headers.items())


def collect(pairs, root, access, secret, head=None, clock=None, sleep=None):
    destination = root/'request334-metadata.json'
    if destination.exists():
        raise ValueError('refuse to overwrite metadata acquisition')
    head, clock, sleep = head or requests.head, clock or time.monotonic, sleep or time.sleep
    if pairs != fixed_pairs():
        raise ValueError('exact24 registered metadata pairs required')
    ledger = {'request_id': 334, 'method': 'HEAD', 'declared_pairs_sha256': sha(canonical(pairs)),
        'credential_configured': bool(access and secret), 'http_attempts': 0,
        'market_body_bytes_read': 0, 'market_rows_read': 0, 'rows': []}
    last = None
    for pair in pairs:
        row = {**pair, 'http_status': None, 'content_length': None,
            'etag': None, 'last_modified': None, 'error_type': None, 'request_utc': None,
            'response_utc': None, 'state': 'NOT_REQUESTED_MISSING_CREDENTIAL'}
        ledger['rows'].append(row)
        if not access or not secret:
            continue
        if last is not None:
            sleep(max(0., .5-(clock()-last)))
        last = clock()
        ledger['http_attempts'] += 1
        row['request_utc'] = datetime.now(timezone.utc).isoformat()
        response = None
        try:
            url = ENDPOINT+pair['object_key']
            response = head(url, headers=signed_headers(url, access, secret), timeout=10, stream=True, allow_redirects=False)
            status = row['http_status'] = int(response.status_code)
            row['response_utc'] = datetime.now(timezone.utc).isoformat()
            length = str(response.headers.get('Content-Length', ''))
            row['content_length'] = int(length) if length.isdigit() and len(length) <= 40 else None
            for header, field in (('ETag', 'etag'), ('Last-Modified', 'last_modified')):
                value = response.headers.get(header)
                row[field] = str(value).replace(access, '[REDACTED]').replace(secret, '[REDACTED]') if value is not None else None
            row['state'] = 'HEAD_METADATA_ACCESS_OBSERVED' if status == 200 else 'HEAD_REJECTED_403' if status == 403 else 'NOT_FOUND_OR_UNDISCLOSED_404' if status == 404 else 'REDIRECT_UNRESOLVED' if 300 <= status < 400 else 'HTTP_UNRESOLVED_'+str(status)
        except requests.RequestException as error:
            row['state'], row['error_type'] = 'TRANSPORT_ERROR', type(error).__name__
        finally:
            if response is not None:
                response.close()
        save_json(ledger, root/'request334-progress.json')
    save_json(ledger, destination)
    return ledger


def summarize(ledger, pairs):
    if pairs != fixed_pairs():
        raise ValueError('registered metadata scope changed')
    if ledger['declared_pairs_sha256'] != sha(canonical(pairs)) or [{k: row[k] for k in pair} for row, pair in zip(ledger['rows'], pairs, strict=True)] != pairs:
        raise ValueError('registered metadata scope changed')
    attempts = sum(row['request_utc'] is not None for row in ledger['rows'])
    if attempts != ledger['http_attempts'] or attempts > 24 or ledger['market_body_bytes_read'] or ledger['market_rows_read']:
        raise ValueError('metadata-only resource ledger changed')
    streams = {}
    for stream in PREFIXES:
        rows = [row for row in ledger['rows'] if row['stream'] == stream]
        observed = [row for row in rows if row['http_status'] == 200]
        states = sorted({row['state'] for row in rows})
        streams[stream] = {'declared_objects': len(rows), 'head_access_observed_objects': len(observed),
            'known_compressed_file_bytes': sum(row['content_length'] for row in observed if row['content_length'] is not None),
            'objects_with_unknown_size': sum(row['content_length'] is None for row in observed),
            'state_counts': {state: sum(row['state'] == state for row in rows) for state in states}}
    all_access = all(x['head_access_observed_objects'] == x['declared_objects'] for x in streams.values())
    return {'request_id': 334, 'metadata_audit_completed': True, 'http_attempts': attempts,
        'market_body_bytes_read': 0, 'market_rows_read': 0, 'new_economic_labels': 0, 'model_fits': 0,
        'sealed_dates_opened': False, 'promotion_eligible': False, 'R331_90pct_failure_preserved': True,
        'all_required_HEAD_access_observed': all_access,
        'GET_permission_or_record_completeness_claimed': False,
        'decision': 'PREPARE_SEPARATE_BUDGETED_DOWNLOAD_CONTRACT' if all_access else 'REQUIRED_ALTERNATIVE_SOURCE_ACCESS_UNRESOLVED',
        'streams': streams, 'metadata_ledger': ledger}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('collect', 'analyze'), required=True)
    parser.add_argument('--research', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path)
    args = parser.parse_args()
    pairs = declared_pairs(args.research)
    if args.stage == 'collect':
        ledger = collect(pairs, args.output_dir, os.environ.get('MASSIVE_S3_ACCESS_KEY', ''), os.environ.get('MASSIVE_S3_SECRET_KEY', ''))
        print(json.dumps({'request_id': 334, 'http_attempts': ledger['http_attempts'], 'market_body_bytes_read': 0}))
    else:
        path = args.metadata_dir/'request334-metadata.json'
        result = summarize(json.loads(path.read_text()), pairs)
        result['metadata_ledger_sha256'] = sha(path.read_bytes())
        save_json(result, args.output_dir/'request334.json')
        print(json.dumps({'request_id': 334, 'streams': result['streams'], 'decision': result['decision']}, indent=2))
