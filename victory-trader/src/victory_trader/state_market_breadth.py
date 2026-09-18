from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BREADTH_FEATURES = (
    "market_runner_count",
    "market_other_runner_count",
    "market_above_vwap_frac_other",
    "market_new_hod_frac_other",
    "market_reclaim_frac_other",
    "market_trailing5_positive_frac_other",
    "market_failure_frac_other",
    "market_mean_trailing5_pct_other",
    "market_mean_hod_distance_pct_other",
    "market_mean_return_from_prev_pct_other",
    "market_mean_volume_accel1_other",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean_float(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return (
        frame[column]
        .astype("boolean")
        .astype("Float64")
        .astype(float)
    )


def _leave_one_out_mean(
    values: pd.Series,
    group_sum: pd.Series,
    group_count: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    own_valid = numeric.notna().astype(float)
    denominator = group_count - own_valid
    numerator = group_sum - numeric.fillna(0.0)
    result = numerator / denominator
    return result.where(denominator > 0)


def enrich_market_breadth(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"trading_day", "t", "ticker"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"state panel missing breadth grouping columns: {sorted(missing)}"
        )

    result = frame.copy()
    keys = ["trading_day", "t"]

    above = _boolean_float(result, "above_regular_vwap")
    new_hod = _boolean_float(result, "new_hod")
    reclaim = _boolean_float(
        result, "reclaim_prior_5m_high_after_pullback"
    )
    trailing5 = _numeric(result, "trailing_return_5m_pct")
    hod_distance = _numeric(result, "hod_distance_pct")
    return_prev = _numeric(result, "return_from_previous_close_pct")
    volume_accel = _numeric(result, "volume_accel_1m_vs_prior20m")

    failure = (
        hod_distance.le(-5.0)
        & above.fillna(0.0).eq(0.0)
    ).astype(float)
    trailing5_positive = trailing5.gt(0.0).where(
        trailing5.notna()
    ).astype(float)

    temporary = pd.DataFrame(
        {
            "trading_day": result["trading_day"].astype(str),
            "t": pd.to_numeric(result["t"], errors="coerce"),
            "_above": above,
            "_new_hod": new_hod,
            "_reclaim": reclaim,
            "_trail5_pos": trailing5_positive,
            "_failure": failure,
            "_trail5": trailing5,
            "_hod_distance": hod_distance,
            "_return_prev": return_prev,
            "_volume_accel": volume_accel,
        },
        index=result.index,
    )

    grouped = temporary.groupby(keys, sort=False, dropna=False)
    result["market_runner_count"] = grouped["t"].transform("size").astype(float)
    result["market_other_runner_count"] = (
        result["market_runner_count"] - 1.0
    ).clip(lower=0.0)

    specifications = {
        "market_above_vwap_frac_other": "_above",
        "market_new_hod_frac_other": "_new_hod",
        "market_reclaim_frac_other": "_reclaim",
        "market_trailing5_positive_frac_other": "_trail5_pos",
        "market_failure_frac_other": "_failure",
        "market_mean_trailing5_pct_other": "_trail5",
        "market_mean_hod_distance_pct_other": "_hod_distance",
        "market_mean_return_from_prev_pct_other": "_return_prev",
        "market_mean_volume_accel1_other": "_volume_accel",
    }

    for output, source in specifications.items():
        values = temporary[source]
        sums = grouped[source].transform("sum")
        counts = grouped[source].transform("count").astype(float)
        result[output] = _leave_one_out_mean(values, sums, counts)

    for column in BREADTH_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.state_market_breadth"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    enriched = enrich_market_breadth(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False, compression="zstd")

    print(
        f"Enriched {len(enriched)} state rows with "
        f"{len(BREADTH_FEATURES)} leave-one-out market breadth features; "
        f"days={enriched['trading_day'].nunique()}, "
        f"median_runner_count={enriched['market_runner_count'].median():.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
