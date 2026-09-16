from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


RETURN_RE = re.compile(r"^return_(\d+)m_pct$")
NET_RETURN_RE = re.compile(r"^return_(\d+)m_([^_]+)_net_return_pct$")
BARRIER_STATUS_RE = re.compile(r"^(tp[^_]+_sl[^_]+)_status$")
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


def summarize_cost_scenarios(frame: pd.DataFrame, *, horizon_min: int = 5) -> pd.DataFrame:
    """Compare gross continuation against all available execution scenarios."""
    gross_col = f"return_{horizon_min}m_pct"
    if gross_col not in frame.columns:
        raise ValueError(f"dataset missing {gross_col}")

    scenario_columns: list[tuple[str, str]] = []
    for column in frame.columns:
        match = NET_RETURN_RE.match(column)
        if match and int(match.group(1)) == horizon_min:
            scenario_columns.append((match.group(2), column))
    if not scenario_columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for threshold, group in frame.groupby("threshold_pct", sort=True):
        gross = pd.to_numeric(group[gross_col], errors="coerce").dropna()
        if not gross.empty:
            rows.append(
                {
                    "threshold_pct": float(threshold),
                    "horizon_min": horizon_min,
                    "scenario": "gross",
                    "n": int(len(gross)),
                    "mean_return_pct": float(gross.mean()),
                    "median_return_pct": float(gross.median()),
                    "positive_rate": float((gross > 0).mean()),
                }
            )
        for scenario, column in sorted(scenario_columns):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "threshold_pct": float(threshold),
                    "horizon_min": horizon_min,
                    "scenario": scenario,
                    "n": int(len(values)),
                    "mean_return_pct": float(values.mean()),
                    "median_return_pct": float(values.median()),
                    "positive_rate": float((values > 0).mean()),
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


def summarize_barriers(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    barrier_keys = []
    for column in frame.columns:
        match = BARRIER_STATUS_RE.match(column)
        if match:
            barrier_keys.append(match.group(1))
    if not barrier_keys:
        return pd.DataFrame()

    rows: list[dict] = []
    for threshold, group in frame.groupby("threshold_pct", sort=True):
        for key in sorted(barrier_keys):
            status_col = f"{key}_status"
            return_col = f"{key}_exit_return_pct"
            statuses = group[status_col].astype("string")
            resolved = statuses.isin(["take_profit", "stop_loss", "timeout"])
            returns = pd.to_numeric(group.loc[resolved, return_col], errors="coerce").dropna()
            resolved_count = int(resolved.sum())
            row = {
                "threshold_pct": float(threshold),
                "barrier": key,
                "n": int(len(group)),
                "resolved_n": resolved_count,
                "take_profit_rate": float((statuses == "take_profit").sum() / resolved_count) if resolved_count else np.nan,
                "stop_loss_rate": float((statuses == "stop_loss").sum() / resolved_count) if resolved_count else np.nan,
                "timeout_rate": float((statuses == "timeout").sum() / resolved_count) if resolved_count else np.nan,
                "ambiguous_rate": float((statuses == "ambiguous").mean()),
                "mean_exit_return_pct": float(returns.mean()) if not returns.empty else np.nan,
                "median_exit_return_pct": float(returns.median()) if not returns.empty else np.nan,
            }
            for scenario in ("light", "base", "stress"):
                net_col = f"{key}_{scenario}_net_return_pct"
                if net_col in group.columns:
                    net = pd.to_numeric(group.loc[resolved, net_col], errors="coerce").dropna()
                    row[f"mean_{scenario}_net_pct"] = float(net.mean()) if not net.empty else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_rvol(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
) -> pd.DataFrame:
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
