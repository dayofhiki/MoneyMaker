"""Compact large partitioned attention replay outputs for durable artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FOCUS_STATES = {"watch", "hot", "position"}


def compact_attention_artifacts(
    scan_dir: Path,
    trace_dir: Path,
    *,
    focus_output: Path,
    runner_output: Path,
) -> tuple[int, int]:
    focus_parts: list[pd.DataFrame] = []
    runner_parts: list[pd.DataFrame] = []

    for path in sorted(trace_dir.rglob("*.parquet")):
        frame = pd.read_parquet(path)
        focus = frame.loc[frame["state"].astype(str).isin(FOCUS_STATES)].copy()
        if not focus.empty:
            focus_parts.append(focus)

    for path in sorted(scan_dir.rglob("*.parquet")):
        frame = pd.read_parquet(path)
        if "runner_cross_now" not in frame.columns:
            continue
        runners = frame.loc[
            frame["runner_cross_now"].fillna(False).astype(bool)
        ].copy()
        if not runners.empty:
            runner_parts.append(runners)

    focus = (
        pd.concat(focus_parts, ignore_index=True)
        if focus_parts
        else pd.DataFrame()
    )
    runners = (
        pd.concat(runner_parts, ignore_index=True)
        if runner_parts
        else pd.DataFrame()
    )

    focus_output.parent.mkdir(parents=True, exist_ok=True)
    runner_output.parent.mkdir(parents=True, exist_ok=True)
    focus.to_parquet(focus_output, index=False, compression="zstd")
    runners.to_parquet(runner_output, index=False, compression="zstd")
    return len(focus), len(runners)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact partitioned attention replay into research-sized artifacts."
    )
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--focus-output", type=Path, required=True)
    parser.add_argument("--runner-output", type=Path, required=True)
    args = parser.parse_args()

    focus_rows, runner_rows = compact_attention_artifacts(
        args.scan_dir,
        args.trace_dir,
        focus_output=args.focus_output,
        runner_output=args.runner_output,
    )
    print(f"focus_rows={focus_rows}")
    print(f"runner_rows={runner_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
