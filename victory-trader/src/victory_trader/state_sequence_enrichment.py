from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LAGS = (1, 2, 3, 5, 8, 13)
SEQUENCE_SOURCES = {
    "trailing_return_1m_pct": "ret1",
    "hod_distance_pct": "hod",
    "regular_vwap_distance_pct": "vwap",
    "volume_accel_1m_vs_prior20m": "volacc1",
    "bar_range_pct": "range",
    "bar_close_location": "close_loc",
}


def sequence_feature_names() -> tuple[str, ...]:
    return tuple(
        f"seq_{short}_lag{lag}"
        for _, short in SEQUENCE_SOURCES.items()
        for lag in LAGS
    )


SEQUENCE_FEATURES = sequence_feature_names()


def enrich_state_sequence(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"trading_day", "ticker", "t"} | set(SEQUENCE_SOURCES)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"state panel missing sequence columns: {sorted(missing)}"
        )

    result = frame.sort_values(
        ["trading_day", "ticker", "t"],
        kind="stable",
    ).copy()
    group_keys = [
        result["trading_day"].astype(str),
        result["ticker"].astype(str),
    ]

    for source, short in SEQUENCE_SOURCES.items():
        numeric = pd.to_numeric(result[source], errors="coerce")
        grouped = numeric.groupby(group_keys, sort=False)
        for lag in LAGS:
            result[f"seq_{short}_lag{lag}"] = grouped.shift(lag)

    return result.sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.state_sequence_enrichment"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    enriched = enrich_state_sequence(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False, compression="zstd")
    print(
        f"Enriched {len(enriched)} state rows with "
        f"{len(SEQUENCE_FEATURES)} causal lag-sequence features."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
