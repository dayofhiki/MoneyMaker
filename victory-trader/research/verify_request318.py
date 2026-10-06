"""Verify fixed R318 outputs without acquisition or refitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_request317 import compare_json


def verify(official: Path, local: Path) -> dict:
    a, b = official/'request318.json', local/'request318.json'
    result = {'request_id': 318, 'offline_market_requests': 0, 'absolute_tolerance': 1e-9,
        'json': compare_json(json.loads(a.read_text()), json.loads(b.read_text())),
        'json_bytes_equal': a.read_bytes() == b.read_bytes(), 'frames': {}}
    for filename in ('request318-states.parquet', 'request318-chronological-states.parquet'):
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
                if not np.array_equal(np.sign(xa[:, None]-xa), np.sign(ya[:, None]-ya)):
                    raise ValueError('probability pair ranks changed in '+filename)
        result['frames'][filename] = {'rows': len(x), 'columns': len(x.columns), 'maximum_numeric_error': max(errors, default=0.),
            'missingness_equal': True, 'identities_labels_and_values_equal_within_tolerance': True,
            'probability_pair_ranks_equal': True}
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
