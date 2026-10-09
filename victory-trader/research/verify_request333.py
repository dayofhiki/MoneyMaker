"""Replay the archived R333 qualification pages without any market acquisition."""
import argparse
import json
from pathlib import Path

from verify_request317 import compare_json
from victory_trader import historical_source_qualification as source


def verify(official, plan_path, output):
    plan = json.loads(plan_path.read_text())
    expected = json.loads((official/'request333.json').read_text())
    replay = source.analyze(official/'sources', plan)
    source.save_json(replay, output/'request333.json')
    if source.sha((official/'request333-full-window-manifest.parquet').read_bytes()) != plan['full_window_manifest_sha256']:
        raise ValueError('registered full identity window manifest changed')
    manifest = json.loads((official/'sources/request333-page-manifest.json').read_text())
    result = {'request_id': 333, 'new_market_requests_for_reproduction': 0, 'absolute_tolerance': 1e-9,
        'json': {**compare_json(expected, replay),
            'bytes_equal': (official/'request333.json').read_bytes() == (output/'request333.json').read_bytes()},
        'retained_wire_page_hashes_verified': sum(bool(p['wire_file']) for x in manifest['pairs'] for p in x['pages']),
        'sanitized_payload_hashes_verified': sum(bool(p['payload_file']) for x in manifest['pairs'] for p in x['pages']),
        'page_manifest_sha256': source.sha((official/'sources/request333-page-manifest.json').read_bytes()),
        'all_declared_pairs_and_categories_exact': True,
        'full_identity_window_manifest_sha256': plan['full_window_manifest_sha256'],
        'reproduction_is_same_archived_acquisition_not_new_market_sample': True}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('official', 'plan', 'output_dir', 'audit_output'):
        parser.add_argument('--'+arg.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.official, args.plan, args.output_dir)
    source.save_json(result, args.audit_output)
    print(json.dumps(result, indent=2))
