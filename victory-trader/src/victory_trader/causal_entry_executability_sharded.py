"""Infrastructure-only sharded wrapper for Request 165."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .causal_entry_executability import evaluate, prepare_fresh


def prepare_day(
    day: str,
    candidates_output: Path,
    scan_output: Path,
    audit_output: Path,
) -> int:
    parsed = date.fromisoformat(day)
    return prepare_fresh(
        parsed,
        parsed,
        candidates_output,
        scan_output,
        audit_output,
        only_day=day,
    )


def evaluate_dir(
    fresh_dir: Path,
    history_first_hot: Path,
    stage2_candidates: Path,
    output: Path,
) -> int:
    candidate_paths = sorted(fresh_dir.glob("*-candidates.parquet"))
    scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    audit_paths = sorted(fresh_dir.glob("*-audit.json"))
    if not candidate_paths or not scan_paths or not audit_paths:
        raise ValueError("request 165 sharded inputs are incomplete")

    candidates = pd.concat(
        [pd.read_parquet(path) for path in candidate_paths],
        ignore_index=True,
    )
    scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    audits = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in audit_paths
    ]

    merged_dir = fresh_dir / "_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = merged_dir / "fresh-candidates.parquet"
    scan_path = merged_dir / "fresh-scan.parquet"
    audit_path = merged_dir / "fresh-audit.json"
    candidates.to_parquet(candidates_path, index=False, compression="zstd")
    scan.to_parquet(scan_path, index=False, compression="zstd")
    audit_path.write_text(
        json.dumps(
            {
                "request_id": 165,
                "execution_mode": "parallel_day_shards",
                "fresh_days": sorted(
                    scan["trading_day"].astype(str).unique()
                ),
                "candidate_rows": int(len(candidates)),
                "scan_rows": int(len(scan)),
                "shards": audits,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return evaluate(
        history_first_hot,
        stage2_candidates,
        candidates_path,
        scan_path,
        audit_path,
        output,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-day")
    prep.add_argument("day")
    prep.add_argument("--candidates-output", type=Path, required=True)
    prep.add_argument("--scan-output", type=Path, required=True)
    prep.add_argument("--audit-output", type=Path, required=True)

    ev = sub.add_parser("evaluate-dir")
    ev.add_argument("--fresh-dir", type=Path, required=True)
    ev.add_argument("--history-first-hot", type=Path, required=True)
    ev.add_argument("--stage2-candidates", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare-day":
        return prepare_day(
            args.day,
            args.candidates_output,
            args.scan_output,
            args.audit_output,
        )
    return evaluate_dir(
        args.fresh_dir,
        args.history_first_hot,
        args.stage2_candidates,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
