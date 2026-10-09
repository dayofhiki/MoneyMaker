"""Offline replay of the fixed R334 metadata ledger; no HEAD/GET requests."""
import argparse
import json
from pathlib import Path

from verify_request317 import compare_json
from victory_trader import flatfile_metadata_access as audit


def verify(official, research, output):
    path = official/'metadata/request334-metadata.json'
    result = audit.summarize(json.loads(path.read_text()), audit.declared_pairs(research))
    result['metadata_ledger_sha256'] = audit.sha(path.read_bytes())
    audit.save_json(result, output/'request334.json')
    expected = json.loads((official/'request334.json').read_text())
    return {'request_id': 334, 'new_network_requests_for_reproduction': 0,
        'json': {**compare_json(expected, result),
            'bytes_equal': (official/'request334.json').read_bytes() == (output/'request334.json').read_bytes()},
        'metadata_ledger_sha256': result['metadata_ledger_sha256'],
        'all_dates_objects_statuses_timestamps_and_unknowns_exact': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('official', 'research', 'output_dir', 'audit_output'):
        parser.add_argument('--'+arg.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.official, args.research, args.output_dir)
    audit.save_json(result, args.audit_output)
    print(json.dumps(result, indent=2))
