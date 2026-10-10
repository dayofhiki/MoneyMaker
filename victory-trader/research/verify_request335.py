"""Compare independent offline R335 preparations; no market calls or model fits."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def numbers(value):
    if isinstance(value, dict):
        return sum(numbers(v) for v in value.values())
    if isinstance(value, list):
        return sum(numbers(v) for v in value)
    return int(type(value) in (int, float))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--reproduction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    files = {}
    for name in ("request335.json", "request335-input-ledger.json"):
        def read(root):
            path = root/name
            if not path.exists() and (root/"results"/name).exists():
                path = root/"results"/name
            return path.read_bytes() if path.exists() else gzip.decompress((root/(name+".gz")).read_bytes())
        b, c = read(a.reference), read(a.reproduction)
        if b != c:
            raise ValueError("offline_reproduction_bytes_differ")
        files[name] = {"bytes_equal": True, "sha256": hashlib.sha256(b).hexdigest(),
                       "numeric_fields": numbers(json.loads(b)), "maximum_numeric_error": 0}
    result = {"request_id": 335, "stage": "offline_input_preparation_only",
              "new_market_requests_for_reproduction": 0, "files": files,
              "same_missingness_identities_and_states": True,
              "source_completeness_or_licensing_established": False}
    with a.output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
