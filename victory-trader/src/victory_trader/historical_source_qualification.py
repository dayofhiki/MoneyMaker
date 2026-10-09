"""R333: bounded page-preserving source access check, then offline-only replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse

import pandas as pd
import requests

from .action_target_support_census import EVAL_DAYS, TRAIN_DAYS, atomic_parquet
from .pending_exit_integrity import session_limits

HOST = 'api.massive.com'
STREAMS = ('seconds', 'minutes', 'trades', 'quotes')
KEYS = ['trading_day', 'ticker', 'hot_t']
SENSITIVE = {'apikey', 'api_key', 'authorization', 'access_token', 'token', 'secret', 'password', 'access_key', 'secret_key'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def strict_json(data):
    def reject_constant(value):
        raise ValueError('invalid_JSON_constant')
    return json.loads(data, parse_constant=reject_constant)


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def sanitized(value, secret=''):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if k.lower() in SENSITIVE else sanitized(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitized(v, secret) for v in value]
    if isinstance(value, str):
        clean = value.replace(secret, '[REDACTED]') if secret else value
        if clean.startswith(('https://', 'http://')):
            try:
                parsed = urlparse(clean)
            except ValueError:
                return '[MALFORMED_URL_WITHHELD]'
            params = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() not in SENSITIVE]
            clean = parsed._replace(netloc=parsed.hostname or '', query=urlencode(params)).geturl()
        return re.sub(r'(?i)(api_?key|access_token|authorization|password|secret)([=:]\s*)([^&\s"<>]+)', r'\1\2[REDACTED]', clean)
    return value


def page_rows(payload, stream):
    if not isinstance(payload, dict) or payload.get('status') != 'OK':
        raise ValueError('provider_success_not_observed')
    if payload.get('next_url') is not None and not isinstance(payload['next_url'], str):
        raise ValueError('invalid_pagination_schema')
    rows = payload.get('results')
    if rows is None and (payload.get('resultsCount') == 0 or payload.get('count') == 0):
        rows = []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('invalid_results_schema')
    field = 't' if stream in ('seconds', 'minutes') else 'sip_timestamp'
    if any(type(row.get(field)) is not int for row in rows):
        raise ValueError('invalid_integer_event_timestamp')
    if stream in ('seconds', 'minutes') and payload.get('adjusted') is not False:
        raise ValueError('unadjusted_response_not_observed')
    return rows


def request_for(window, stream, limit):
    ticker, start, end = window['ticker'], window['start_ms'], window['end_ms']
    if stream in ('seconds', 'minutes'):
        unit = 'second' if stream == 'seconds' else 'minute'
        return f'/v2/aggs/ticker/{ticker}/range/1/{unit}/{start}/{end-1}', {'adjusted': 'false', 'sort': 'asc', 'limit': 50000}
    return f'/v3/{stream}/{ticker}', {'timestamp.gte': str(start*1000000), 'timestamp.lt': str(end*1000000), 'sort': 'timestamp', 'order': 'asc', 'limit': limit}


def next_request(link, expected_path):
    if not isinstance(link, str):
        raise ValueError('unsafe_pagination')
    parsed = urlparse(link)
    if parsed.scheme not in ('', 'https') or parsed.netloc and parsed.netloc != HOST or parsed.username or parsed.password or parsed.fragment or parsed.path != expected_path:
        raise ValueError('unsafe_pagination')
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    filtered = [(k, v) for k, v in pairs if k.lower() not in SENSITIVE]
    if len({k for k, _ in filtered}) != len(filtered):
        raise ValueError('ambiguous_pagination_parameters')
    return parsed.path, dict(filtered)


def prepare(previous, pinned, plan_path, output):
    pins, plan = (json.loads(p.read_text()) for p in (pinned, plan_path))
    for name, digest in pins.items():
        if sha((previous/name).read_bytes()) != digest:
            raise ValueError('registered identity source changed')
    if (previous/'official332/source-sha.txt').read_text().strip() != plan['source_artifact_sha']:
        raise ValueError('R332 original artifact source changed')
    frames = []
    for scope, days, count in (('training', TRAIN_DAYS, 407), ('evaluation', EVAL_DAYS, 828)):
        frame = pd.read_parquet(previous/f'official332/request332-{scope}-source-ledger.parquet', columns=KEYS)
        if len(frame) != count or set(frame.trading_day) != set(days) or frame.duplicated(KEYS).any():
            raise ValueError('original fixed identity census required')
        frames.append(frame.assign(scope=scope))
    full = pd.concat(frames).sort_values(KEYS).reset_index(drop=True)
    full['start_ms'] = [max(session_limits(r.trading_day)[0], r.hot_t-60000) for r in full.itertuples()]
    full['end_ms'] = [session_limits(d)[1] for d in full.trading_day]
    anchors = full.groupby('trading_day', sort=True).head(1).copy()
    anchors['end_ms'] = anchors.start_ms+60000
    if anchors.to_dict('records') != plan['qualification_windows'] or tuple(plan['streams']) != STREAMS:
        raise ValueError('registered qualification clocks changed')
    output.mkdir(parents=True, exist_ok=True)
    destination = output/'request333-full-window-manifest.parquet'
    atomic_parquet(full, destination)
    if sha(destination.read_bytes()) != plan['full_window_manifest_sha256']:
        raise ValueError('registered full-window bytes changed')
    return plan


class Collector:
    def __init__(self, root, limits, secret, get=None, clock=None, sleep=None):
        self.root, self.limits, self.secret = root, limits, secret
        self.get = get or requests.get
        self.clock, self.sleep = clock or time.monotonic, sleep or time.sleep
        self.attempts, self.retained_bytes, self.last_start = 0, 0, None
        (root/'pages').mkdir(parents=True, exist_ok=True)

    def fetch(self, path, params, name):
        metadata = {'request_path': path, 'request_parameters': params, 'http_status': None,
            'provider_status': None, 'request_id': None, 'adjusted': None,
            'resultsCount': None, 'queryCount': None, 'next_url': None,
            'terminal_page_observed': False, 'wire_complete': False,
            'pagination_safe': None,
            'wire_sha256': None, 'wire_file': None, 'wire_evidence_withheld': False,
            'payload_file': None, 'payload_sha256': None, 'body_bytes': 0, 'error_type': None}
        if self.attempts >= self.limits['http_attempts'] or self.retained_bytes >= self.limits['total_response_bytes']:
            return metadata, None, 'BUDGET_EXHAUSTED'
        if self.last_start is not None:
            self.sleep(max(0., self.limits['request_interval_seconds']-(self.clock()-self.last_start)))
        self.last_start = self.clock()
        self.attempts += 1
        metadata['attempt_number'] = self.attempts
        metadata['request_utc'] = datetime.now(timezone.utc).isoformat()
        response = None
        try:
            response = self.get('https://'+HOST+path, params=params,
                headers={'Authorization': 'Bearer '+self.secret}, timeout=self.limits['timeout_seconds'], stream=True, allow_redirects=False)
            metadata['http_status'] = int(response.status_code)
            bound = min(self.limits['bytes_per_page'], self.limits['total_response_bytes']-self.retained_bytes)
            body, complete = bytearray(), True
            for chunk in response.iter_content(chunk_size=4096):
                if len(body)+len(chunk) > bound:
                    body.extend(chunk[:bound-len(body)])
                    complete = False
                    break
                body.extend(chunk)
            wire = bytes(body)
            self.retained_bytes += len(wire)
            metadata.update(body_bytes=len(wire), wire_complete=complete, wire_sha256=sha(wire), response_utc=datetime.now(timezone.utc).isoformat())
            if not complete:
                # Prefix digest is explicitly not the complete response digest.
                return metadata, None, 'PARTIAL_BODY_BUDGET'
            try:
                payload = strict_json(wire)
            except (ValueError, UnicodeError):
                metadata['wire_evidence_withheld'] = True
                return metadata, None, 'INVALID_JSON'
            if isinstance(payload, dict) and payload.get('next_url'):
                try:
                    next_request(payload['next_url'], path)
                    metadata['pagination_safe'] = True
                except (ValueError, TypeError):
                    metadata['pagination_safe'] = False
            clean = sanitized(payload, self.secret)
            unsafe_wire = clean != payload or self.secret.encode() in wire
            metadata['wire_evidence_withheld'] = unsafe_wire
            if not unsafe_wire:
                metadata['wire_file'] = 'pages/'+name+'.wire'
                (self.root/metadata['wire_file']).write_bytes(wire)
            normalized = canonical(clean)
            metadata['payload_file'] = 'pages/'+name+'.json'
            metadata['payload_sha256'] = sha(normalized)
            (self.root/metadata['payload_file']).write_bytes(normalized)
            if isinstance(clean, dict):
                for field in ('status', 'request_id', 'adjusted', 'resultsCount', 'queryCount'):
                    metadata['provider_status' if field == 'status' else field] = clean.get(field)
                metadata['next_url'] = clean.get('next_url')
                metadata['terminal_page_observed'] = not bool(clean.get('next_url'))
            return metadata, clean, 'RECEIVED'
        except requests.RequestException as error:
            metadata['error_type'] = type(error).__name__
            if 'body' in locals() and body:
                self.retained_bytes += len(body)
                metadata.update(body_bytes=len(body), wire_sha256=sha(bytes(body)), wire_evidence_withheld=True)
            return metadata, None, 'TRANSPORT_ERROR'
        finally:
            if response is not None:
                response.close()


def collect(plan, root, secret, get=None, clock=None, sleep=None):
    if (root/'request333-page-manifest.json').exists():
        raise ValueError('refuse to overwrite immutable source acquisition')
    transport = Collector(root, plan['limits'], secret, get, clock, sleep)
    manifest = {'request_id': 333, 'stage': 'source_pages_before_analysis',
        'plan_sha256': sha(canonical(plan)), 'credential_configured': bool(secret),
        'http_attempts': 0, 'retained_response_bytes': 0, 'pairs': []}
    stopped, denied = None, set()
    for index, window in enumerate(plan['qualification_windows']):
        for stream in STREAMS:
            item = {'window': window, 'stream': stream, 'pages': [], 'collection_state': None}
            manifest['pairs'].append(item)
            if not secret or stopped or stream in denied:
                item['collection_state'] = 'NOT_REQUESTED_MISSING_CREDENTIAL' if not secret else 'SKIPPED_AFTER_'+stopped if stopped else 'SKIPPED_AFTER_DENIAL'
                continue
            path, params = request_for(window, stream, plan['limits']['tick_page_limit'])
            seen = set()
            for page in range(plan['limits']['pages_per_pair']):
                token = canonical([path, params])
                if token in seen:
                    item['collection_state'] = 'REPEATED_PAGINATION'
                    break
                seen.add(token)
                meta, payload, state = transport.fetch(path, params, f'{index:02d}-{stream}-{page:02d}')
                meta['page_number'] = page
                item['pages'].append(meta)
                status = meta['http_status']
                if status == 401:
                    item['collection_state'], stopped = 'DENIED_401', 'AUTH_401'
                    break
                if status == 403:
                    item['collection_state'] = 'DENIED_403'
                    denied.add(stream)
                    break
                if status == 429:
                    item['collection_state'], stopped = 'RATE_LIMITED_429', 'RATE_LIMIT_429'
                    break
                if state != 'RECEIVED':
                    item['collection_state'] = state
                    if state in ('BUDGET_EXHAUSTED', 'PARTIAL_BODY_BUDGET'):
                        stopped = 'BUDGET_LIMIT'
                    break
                if status != 200:
                    item['collection_state'] = 'HTTP_ERROR_'+str(status)
                    break
                try:
                    page_rows(payload, stream)
                except ValueError as error:
                    item['collection_state'] = str(error)
                    break
                if not payload.get('next_url'):
                    item['collection_state'] = 'COMPLETE_OBSERVED_PAGES'
                    break
                if meta['pagination_safe'] is False:
                    item['collection_state'] = 'UNSAFE_PAGINATION'
                    break
                try:
                    path, params = next_request(payload['next_url'], path)
                except ValueError:
                    item['collection_state'] = 'UNSAFE_PAGINATION'
                    break
            if item['collection_state'] is None:
                item['collection_state'] = 'PARTIAL_PAGE_LIMIT'
            manifest['http_attempts'], manifest['retained_response_bytes'] = transport.attempts, transport.retained_bytes
            save_json(manifest, root/'request333-progress.json')
    manifest['http_attempts'], manifest['retained_response_bytes'] = transport.attempts, transport.retained_bytes
    save_json(manifest, root/'request333-page-manifest.json')
    return manifest


def source_file(root, relative, digest):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path.read_bytes()) != digest:
        raise ValueError('retained source page path or digest changed')
    return path


def pair_diagnostics(root, item):
    rows = []
    valid_pages = bool(item['pages'])
    access_observed = False
    for page in item['pages']:
        if page['wire_file']:
            source_file(root, page['wire_file'], page['wire_sha256'])
        if page['payload_file']:
            payload = json.loads(source_file(root, page['payload_file'], page['payload_sha256']).read_bytes())
            if page['http_status'] != 200 or not page['wire_complete']:
                valid_pages = False
            if isinstance(payload, dict):
                expected = {k: payload.get(k) for k in ('status', 'adjusted', 'request_id', 'resultsCount', 'queryCount', 'next_url')}
                actual = {k: page['provider_status' if k == 'status' else k] for k in expected}
                if actual != expected or page['terminal_page_observed'] != (not bool(payload.get('next_url'))):
                    raise ValueError('retained response metadata changed')
            try:
                rows.extend(page_rows(payload, item['stream']))
                access_observed |= page['http_status'] == 200
            except ValueError:
                valid_pages = False
        else:
            valid_pages = False
    if item['pages'] and not item['pages'][-1]['terminal_page_observed']:
        valid_pages = False
    field = 't' if item['stream'] in ('seconds', 'minutes') else 'sip_timestamp'
    factor = 1 if field == 't' else 1000000
    stamps = [row[field] for row in rows]
    window = item['window']
    outside = sum(not window['start_ms']*factor <= stamp < window['end_ms']*factor for stamp in stamps)
    fields = ('participant_timestamp', 'sip_timestamp', 'conditions', 'correction', 'bid_price', 'ask_price', 'bid_size', 'ask_size') if factor != 1 else ('t', 'o', 'h', 'l', 'c', 'v', 'n')
    return {'window': window, 'stream': item['stream'], 'collection_state': item['collection_state'],
        'attempted_pages': sum('attempt_number' in p for p in item['pages']),
        'http_statuses': [p['http_status'] for p in item['pages']],
        'provider_statuses': [p['provider_status'] for p in item['pages']],
        'endpoint_access_observed': access_observed,
        'source_complete_in_sampled_window': item['collection_state'] == 'COMPLETE_OBSERVED_PAGES' and valid_pages and outside == 0,
        'wire_body_withheld_pages': sum(p['wire_evidence_withheld'] for p in item['pages']),
        'retained_rows': len(rows), 'empty_returned_stream': not rows and item['collection_state'] == 'COMPLETE_OBSERVED_PAGES',
        'first_event_timestamp': min(stamps) if stamps else None, 'last_event_timestamp': max(stamps) if stamps else None,
        'timestamp_unit': 'millisecond' if factor == 1 else 'nanosecond',
        'out_of_window_rows': outside, 'out_of_order_pairs': sum(a > b for a, b in zip(stamps, stamps[1:])),
        'duplicate_event_timestamps': len(stamps)-len(set(stamps)),
        'missing_optional_fields': {k: sum(row.get(k) is None for row in rows) for k in fields},
        'returned_condition_codes': sorted({code for row in rows for code in (row.get('conditions') if isinstance(row.get('conditions'), list) else []) if type(code) is int}),
        'returned_correction_rows': sum(row.get('correction') is not None for row in rows),
        'SIP_before_participant_rows': sum(type(r.get('participant_timestamp')) is int and type(r.get('sip_timestamp')) is int and r['sip_timestamp'] < r['participant_timestamp'] for r in rows)}


def analyze(root, plan):
    manifest_path = root/'request333-page-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['plan_sha256'] != sha(canonical(plan)):
        raise ValueError('registered source plan changed')
    expected = [(window, stream) for window in plan['qualification_windows'] for stream in STREAMS]
    if [(x['window'], x['stream']) for x in manifest['pairs']] != expected:
        raise ValueError('all declared source pairs must remain in order')
    for item in manifest['pairs']:
        path, params = request_for(item['window'], item['stream'], plan['limits']['tick_page_limit'])
        if len(item['pages']) > plan['limits']['pages_per_pair']:
            raise ValueError('registered page budget changed')
        for index, page in enumerate(item['pages']):
            if page['page_number'] != index or page['request_path'] != path or page['request_parameters'] != params or page['body_bytes'] > plan['limits']['bytes_per_page']:
                raise ValueError('registered request/page chain changed')
            if index+1 < len(item['pages']):
                if not page['next_url'] or page['pagination_safe'] is not True:
                    raise ValueError('unverified page-chain transition')
                path, params = next_request(page['next_url'], path)
    attempts = sum('attempt_number' in page for pair in manifest['pairs'] for page in pair['pages'])
    body_bytes = sum(page['body_bytes'] for pair in manifest['pairs'] for page in pair['pages'])
    if attempts != manifest['http_attempts'] or body_bytes != manifest['retained_response_bytes'] or attempts > plan['limits']['http_attempts'] or body_bytes > plan['limits']['total_response_bytes']:
        raise ValueError('source request/byte budget ledger changed')
    pairs = [pair_diagnostics(root, item) for item in manifest['pairs']]
    streams = {}
    for stream in STREAMS:
        selected = [p for p in pairs if p['stream'] == stream]
        states = sorted({p['collection_state'] for p in selected})
        streams[stream] = {'declared_pairs': len(selected),
            'endpoint_access_observed_pairs': sum(p['endpoint_access_observed'] for p in selected),
            'observed_complete_pairs': sum(p['source_complete_in_sampled_window'] for p in selected),
            'denied_pairs_observed': sum(p['collection_state'].startswith('DENIED_') for p in selected),
            'retained_rows': sum(p['retained_rows'] for p in selected),
            'state_counts': {state: sum(p['collection_state'] == state for p in selected) for state in states}}
    required_access = all(x['observed_complete_pairs'] == len(plan['qualification_windows']) for x in streams.values())
    endpoint_access = all(x['endpoint_access_observed_pairs'] == len(plan['qualification_windows']) for x in streams.values())
    return {'request_id': 333, 'qualification_completed': True, 'full_census_collected': False,
        'full_identity_count_retained_in_plan': plan['full_identities'], 'sampled_identities': len(plan['qualification_windows']),
        'declared_pairs': len(pairs), 'market_http_attempts': manifest['http_attempts'],
        'retained_response_bytes': manifest['retained_response_bytes'], 'credential_configured': manifest['credential_configured'],
        'page_manifest_sha256': sha(manifest_path.read_bytes()), 'plan_sha256': sha(canonical(plan)),
        'new_economic_labels': 0, 'model_fits': 0, 'sealed_dates_opened': False, 'promotion_eligible': False,
        'full_census_support_or_gap_causes_claimed': False, 'R331_90pct_failure_preserved': True,
        'all_required_sampled_sources_available': required_access,
        'all_required_sampled_endpoint_access_observed': endpoint_access,
        'decision': 'REGISTER_FULL_CENSUS_AND_ELIGIBILITY_SEPARATELY' if required_access else 'SAMPLED_ACCESS_OBSERVED_BUT_COMPLETENESS_UNRESOLVED' if endpoint_access else 'REQUIRED_SOURCE_ACCESS_OR_COMPLETENESS_UNRESOLVED',
        'streams': streams, 'pairs': pairs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('plan', 'collect', 'analyze'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--pinned-inputs', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.stage == 'plan':
        prepare(args.previous, args.pinned_inputs, args.plan, args.output_dir)
        print('R333 fixed identity/windows authenticated; no market requests')
    elif args.stage == 'collect':
        result = collect(plan, args.output_dir, os.environ.get('MASSIVE_API_KEY', ''))
        print(json.dumps({'request_id': 333, 'declared_pairs': len(result['pairs']), 'http_attempts': result['http_attempts'], 'retained_response_bytes': result['retained_response_bytes']}))
    else:
        result = analyze(args.source_dir, plan)
        save_json(result, args.output_dir/'request333.json')
        print(json.dumps({'request_id': 333, 'streams': result['streams'], 'decision': result['decision']}, indent=2))
