"""Compare the official fixed R317 replay with local outputs, without fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare_json(a, b, path='', audit=None):
    audit = audit if audit is not None else {'numeric_fields': 0, 'maximum_numeric_error': 0.}
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError('JSON keys changed at '+path)
        for key in a:
            compare_json(a[key], b[key], path+'/'+str(key), audit)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError('JSON length changed at '+path)
        for i, (v, w) in enumerate(zip(a, b, strict=True)):
            compare_json(v, w, path+'/'+str(i), audit)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if not isinstance(b, (int, float)) or isinstance(b, bool):
            raise ValueError('JSON numeric type changed at '+path)
        error = abs(a-b)
        if error > 1e-9 or not np.isfinite(error):
            raise ValueError('JSON number changed at '+path)
        audit['numeric_fields'] += 1
        audit['maximum_numeric_error'] = max(audit['maximum_numeric_error'], float(error))
    elif a != b:
        raise ValueError('JSON value changed at '+path)
    return audit


def verify(official: Path, local: Path) -> dict:
    a, b = official/'request317.json', local/'request317.json'
    result = {'request_id': 317, 'offline_market_requests': 0, 'absolute_tolerance': 1e-9,
              'json': compare_json(json.loads(a.read_text()), json.loads(b.read_text())),
              'json_bytes_equal': a.read_bytes() == b.read_bytes(), 'frames': {}}
    for name in ('clock-manifest', 'episode-ledger', 'training-states', 'states', 'chronological-states'):
        filename = f'request317-{name}.parquet'
        x, y = pd.read_parquet(official/filename), pd.read_parquet(local/filename)
        pd.testing.assert_frame_equal(x.where(x.notna(), None), y.where(y.notna(), None), check_dtype=False, atol=1e-9, rtol=0)
        if not x.isna().equals(y.isna()):
            raise ValueError('missingness changed in '+filename)
        errors = []
        for column in x.select_dtypes(include='number'):
            finite = x[column].notna() & y[column].notna()
            if finite.any():
                errors.append(float(np.max(np.abs(x.loc[finite, column].to_numpy(float)-y.loc[finite, column].to_numpy(float)))))
            if '_p_net_5' in column:
                xa, ya = x.loc[finite, column].to_numpy(float), y.loc[finite, column].to_numpy(float)
                # Compare complete pair ranks, including ties, not just ordering.
                if not np.array_equal(np.sign(xa[:, None]-xa), np.sign(ya[:, None]-ya)):
                    raise ValueError('probability ranks changed in '+filename)
        result['frames'][filename] = {'rows': len(x), 'columns': len(x.columns),
            'maximum_numeric_error': max(errors, default=0.), 'missingness_equal': True,
            'identities_labels_and_values_equal_within_tolerance': True, 'probability_pair_ranks_equal': True}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.official, args.local)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
