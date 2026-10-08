"""Compare the complete registered R331 census; never refit or reconstruct it."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_request317 import compare_json


def verify(reference, replay):
    result = {'request_id': 331, 'absolute_tolerance': 1e-9, 'json': {}, 'frames': {}}
    for name in ('request331-clock-registration.json', 'request331.json'):
        x, y = reference/name, replay/name
        result['json'][name] = {**compare_json(json.loads(x.read_text()), json.loads(y.read_text())),
            'bytes_equal': x.read_bytes() == y.read_bytes()}
    for scope in ('training', 'evaluation'):
        for suffix in ('clocks', 'clock-ledger', 'targets', 'support-ledger'):
            name = f'request331-{scope}-{suffix}.parquet'
            x, y = pd.read_parquet(reference/name), pd.read_parquet(replay/name)
            if list(x) != list(y) or not x.isna().equals(y.isna()):
                raise ValueError('columns/missingness changed: '+name)
            pd.testing.assert_frame_equal(x, y, check_dtype=False, atol=1e-9, rtol=0)
            errors = []
            for column in x.select_dtypes(include='number'):
                valid = x[column].notna()
                if valid.any():
                    errors.append(float(np.max(np.abs(x.loc[valid, column].to_numpy(float)-y.loc[valid, column].to_numpy(float)))))
                if column.startswith('enter_') and column.endswith('_pct'):
                    if not x[column].rank(method='min', na_option='keep').equals(y[column].rank(method='min', na_option='keep')):
                        raise ValueError('payoff ties/pair ranks changed: '+name)
            result['frames'][name] = {'rows': len(x), 'columns': len(x.columns),
                'maximum_numeric_error': max(errors, default=0.), 'exact_missingness': True,
                'exact_clocks_identity_phase_flags_reasons': True, 'exact_payoff_pair_ranks': True,
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
