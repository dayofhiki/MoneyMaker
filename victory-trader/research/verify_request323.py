"""Verify fixed R323 ranks, scope and censoring without refitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_request317 import compare_json


def verify(reference: Path, replay: Path) -> dict:
    a, b = reference/"request323.json", replay/"request323.json"
    result = {"request_id": 323, "absolute_tolerance": 1e-9,
        "json": compare_json(json.loads(a.read_text()), json.loads(b.read_text())),
        "json_bytes_equal": a.read_bytes() == b.read_bytes(), "frames": {}}
    for suffix in ("states", "first-states", "episode-ledger", "pair-manifest", "pair-ledger", "local-pair-manifest", "local-pair-ledger"):
        filename = f"request323-{suffix}.parquet"
        x, y = pd.read_parquet(reference/filename), pd.read_parquet(replay/filename)
        pd.testing.assert_frame_equal(x.where(x.notna(), None), y.where(y.notna(), None), check_dtype=False, atol=1e-9, rtol=0)
        if not x.isna().equals(y.isna()):
            raise ValueError("missingness changed in "+filename)
        errors = []
        for column in x.select_dtypes(include="number"):
            finite = x[column].notna() & y[column].notna()
            if finite.any():
                errors.append(float(np.max(np.abs(x.loc[finite, column].to_numpy(float)-y.loc[finite, column].to_numpy(float)))))
            if column.endswith("_score") or "_p_net_5" in column:
                if not x[column].rank(method="min", na_option="keep").equals(y[column].rank(method="min", na_option="keep")):
                    raise ValueError("score pair ranks changed in "+filename)
        result["frames"][filename] = {"rows": len(x), "columns": len(x.columns), "maximum_numeric_error": max(errors, default=0.),
            "missingness_equal": True, "identities_labels_and_values_equal_within_tolerance": True, "score_pair_ranks_equal": True}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.reference, args.replay)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))
