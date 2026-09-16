from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


RETURN_RE = re.compile(r"^return_(\d+)m_pct$")
RVOL_BINS = [0.0, 1.0, 2.0, 5.0, 10.0, float("inf")]
RVOL_LABELS = ["<1x", "1-2x", "2-5x", "5-10x", ">=10x"]


def load_event_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("dataset must be .parquet or .csv")


def available_horizons(frame: pd.DataFrame) -> list[int]:
    horizons: list[int] = []
    for column in frame.columns:
        match = RETURN_RE.match(column)
        if match:
            horizons.append(int(match.group(1)))
    return sorted(horizons)


def summarize_by_threshold(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe continuation after each first-crossing threshold."""
    if frame.empty:
        return pd.DataFrame()
    if "threshold_pct" not in frame.columns:
        raise ValueError("dataset missing threshold_pct")

    rows: list[dict] = []
    for threshold, group in frame.groupby("threshold_pct", sort=True):
        for horizon in available_horizons(frame):
            column = f"return_{horizon}m_pct"
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "threshold_pct": float(threshold),
                    "horizon_min": horizon,
                    "n": int(len(values)),
                    "mean_return_pct": float(values.mean()),
                    "median_return_pct": float(values.median()),
                    "positive_rate": float((values > 0).mean()),
                    "p25_return_pct": float(values.quantile(0.25)),
                    "p75_return_pct": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def summarize_excursions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"threshold_pct", "mfe_pct", "mae_pct"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing excursion columns: {sorted(missing)}")
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for threshold, group in frame.groupby("threshold_pct", sort=True):
        mfe = pd.to_numeric(group["mfe_pct"], errors="coerce").dropna()
        mae = pd.to_numeric(group["mae_pct"], errors="coerce").dropna()
        rows.append(
            {
                "threshold_pct": float(threshold),
                "n": int(len(group)),
                "mean_mfe_pct": float(mfe.mean()) if not mfe.empty else np.nan,
                "median_mfe_pct": float(mfe.median()) if not mfe.empty else np.nan,
                "mean_mae_pct": float(mae.mean()) if not mae.empty else np.nan,
                "median_mae_pct": float(mae.median()) if not mae.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_rvol(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
) -> pd.DataFrame:
    """Compare continuation across point-in-time cumulative RVOL buckets."""
    return_col = f"return_{horizon_min}m_pct"
    required = {"threshold_pct", "rvol_cumulative_20d", return_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing RVOL analysis columns: {sorted(missing)}")

    work = frame.loc[:, list(required)].copy()
    work["rvol_cumulative_20d"] = pd.to_numeric(work["rvol_cumulative_20d"], errors="coerce")
    work[return_col] = pd.to_numeric(work[return_col], errors="coerce")
    work = work.dropna(subset=["rvol_cumulative_20d", return_col])
    if work.empty:
        return pd.DataFrame()

    work["rvol_bucket"] = pd.cut(
        work["rvol_cumulative_20d"],
        bins=RVOL_BINS,
        labels=RVOL_LABELS,
        right=False,
        include_lowest=True,
    )

    rows: list[dict] = []
    for (threshold, bucket), group in work.groupby(
        ["threshold_pct", "rvol_bucket"], observed=True, sort=True
    ):
        values = group[return_col]
        rows.append(
            {
                "threshold_pct": float(threshold),
                "rvol_bucket": str(bucket),
                "horizon_min": horizon_min,
                "n": int(len(values)),
                "mean_return_pct": float(values.mean()),
                "median_return_pct": float(values.median()),
                "positive_rate": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(rows)
