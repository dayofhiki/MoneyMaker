"""Verify every R332 JSON field, ledger and diagnostic clock against the replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_request317 import compare_json


def verify(reference, replay):
    name = 'request332.json'
    result = {'request_id': 332, 'absolute_tolerance': 1e-9,
        'json': {**compare_json(json.loads((reference/name).read_text()), json.loads((replay/name).read_text())),
            'bytes_equal': (reference/name).read_bytes() == (replay/name).read_bytes()}, 'frames': {}}
    for scope in ('training', 'evaluation'):
        for kind in ('source-ledger', 'clock-audit'):
            name = f'request332-{scope}-{kind}.parquet'
            x, y = pd.read_parquet(reference/name), pd.read_parquet(replay/name)
            if list(x) != list(y) or not x.isna().equals(y.isna()):
                raise ValueError('columns/missingness changed: '+name)
            pd.testing.assert_frame_equal(x, y, check_dtype=False, atol=1e-9, rtol=0)
            errors = [float(np.max(np.abs(x[c].dropna().to_numpy(float)-y[c].dropna().to_numpy(float)))) for c in x.select_dtypes(include='number') if x[c].notna().any()]
            result['frames'][name] = {'rows': len(x), 'columns': len(x.columns),
                'maximum_numeric_error': max(errors, default=0.), 'exact_missingness': True,
                'exact_identities_clocks_categories_flags': True,
                'bytes_equal': (reference/name).read_bytes() == (replay/name).read_bytes()}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('reference', 'replay', 'output'):
        parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.reference, args.replay)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
