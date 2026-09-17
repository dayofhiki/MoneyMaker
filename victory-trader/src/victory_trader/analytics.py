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


def _basic_row(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"n": 0, "mean_return_pct": np.nan, "median_return_pct": np.nan, "positive_rate": np.nan}
    return {
        "n": int(len(clean)),
        "mean_return_pct": float(clean.mean()),
        "median_return_pct": float(clean.median()),
        "positive_rate": float((clean > 0).mean()),
    }


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
            rows.append({"threshold_pct": float(threshold), "horizon_min": horizon_min, "scenario": "gross", **_basic_row(gross)})
        for scenario, column in sorted(scenario_columns):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({"threshold_pct": float(threshold), "horizon_min": horizon_min, "scenario": scenario, **_basic_row(values)})
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


def _stop_from_key(key: str) -> float | None:
    match = re.search(r"_sl([0-9p]+)$", key)
    if not match:
        return None
    try:
        return float(match.group(1).replace("p", "."))
    except ValueError:
        return None


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
            resolved = statuses.isin(["take_profit", "stop_loss", "stop_gap", "timeout"])
            returns = pd.to_numeric(group.loc[resolved, return_col], errors="coerce").dropna()
            resolved_count = int(resolved.sum())
            row = {
                "threshold_pct": float(threshold),
                "barrier": key,
                "n": int(len(group)),
                "resolved_n": resolved_count,
                "take_profit_rate": float((statuses == "take_profit").sum() / resolved_count) if resolved_count else np.nan,
                "stop_loss_rate": float(statuses.isin(["stop_loss", "stop_gap"]).sum() / resolved_count) if resolved_count else np.nan,
                "stop_gap_rate": float((statuses == "stop_gap").mean()),
                "timeout_rate": float((statuses == "timeout").sum() / resolved_count) if resolved_count else np.nan,
                "ambiguous_rate": float((statuses == "ambiguous").mean()),
                "unresolved_missing_rate": float((statuses == "unresolved_missing").mean()),
                "entry_unavailable_rate": float((statuses == "entry_unavailable").mean()),
                "mean_exit_return_pct": float(returns.mean()) if not returns.empty else np.nan,
                "median_exit_return_pct": float(returns.median()) if not returns.empty else np.nan,
            }
            stop = _stop_from_key(key)
            if stop is not None:
                conservative = pd.to_numeric(group[return_col], errors="coerce").copy()
                conservative.loc[statuses == "ambiguous"] = -stop
                conservative = conservative.loc[~statuses.isin(["entry_unavailable", "unresolved_missing", "no_future_data"])].dropna()
                row["ambiguous_as_stop_mean_pct"] = float(conservative.mean()) if not conservative.empty else np.nan
            for scenario in ("light", "base", "stress"):
                net_col = f"{key}_{scenario}_net_return_pct"
                if net_col in group.columns:
                    net = pd.to_numeric(group.loc[resolved, net_col], errors="coerce").dropna()
                    row[f"mean_{scenario}_net_pct"] = float(net.mean()) if not net.empty else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_rvol(frame: pd.DataFrame, *, horizon_min: int = 5) -> pd.DataFrame:
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

    work["rvol_bucket"] = pd.cut(work["rvol_cumulative_20d"], bins=RVOL_BINS, labels=RVOL_LABELS, right=False, include_lowest=True)
    rows: list[dict] = []
    for (threshold, bucket), group in work.groupby(["threshold_pct", "rvol_bucket"], observed=True, sort=True):
        values = group[return_col]
        rows.append({"threshold_pct": float(threshold), "rvol_bucket": str(bucket), "horizon_min": horizon_min, **_basic_row(values)})
    return pd.DataFrame(rows)


def summarize_missingness(frame: pd.DataFrame, *, horizon_min: int = 5) -> pd.DataFrame:
    """Show whether unavailable outcomes concentrate in particular sessions/thresholds."""
    column = f"return_{horizon_min}m_pct"
    if column not in frame.columns:
        raise ValueError(f"dataset missing {column}")
    work = frame.copy()
    work["_session"] = np.select(
        [work.get("is_premarket", False), work.get("is_regular_session", False), work.get("is_after_hours", False)],
        ["premarket", "regular", "after_hours"],
        default="other",
    )
    rows: list[dict] = []
    for (threshold, session), group in work.groupby(["threshold_pct", "_session"], sort=True):
        present = pd.to_numeric(group[column], errors="coerce").notna()
        entry_present = pd.to_numeric(group["entry_price"], errors="coerce").notna() if "entry_price" in group else pd.Series(False, index=group.index)
        rows.append(
            {
                "threshold_pct": float(threshold),
                "session": str(session),
                "n": int(len(group)),
                "entry_available_rate": float(entry_present.mean()),
                "outcome_available_rate": float(present.mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_latency(frame: pd.DataFrame, *, horizon_min: int = 5, scenario: str = "base") -> pd.DataFrame:
    rows: list[dict] = []
    for delay, prefix in ((0, ""), (1, "delay1_"), (2, "delay2_")):
        column = f"{prefix}return_{horizon_min}m_{scenario}_net_return_pct"
        if column not in frame.columns:
            continue
        for threshold, group in frame.groupby("threshold_pct", sort=True):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            rows.append({"threshold_pct": float(threshold), "delay_min": delay, "scenario": scenario, **_basic_row(values)})
    return pd.DataFrame(rows)


def summarize_sessions(frame: pd.DataFrame, *, horizon_min: int = 5, scenario: str = "base") -> pd.DataFrame:
    column = f"return_{horizon_min}m_{scenario}_net_return_pct"
    if column not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    work["session"] = np.select(
        [work.get("is_premarket", False), work.get("is_regular_session", False), work.get("is_after_hours", False)],
        ["premarket", "regular", "after_hours"],
        default="other",
    )
    rows: list[dict] = []
    for (threshold, session), group in work.groupby(["threshold_pct", "session"], sort=True):
        rows.append({"threshold_pct": float(threshold), "session": str(session), **_basic_row(group[column])})
    return pd.DataFrame(rows)


def cluster_bootstrap_mean_ci(
    frame: pd.DataFrame,
    value_column: str,
    *,
    cluster_columns: tuple[str, ...] = ("trading_day", "ticker"),
    samples: int = 2000,
    seed: int = 20260917,
) -> tuple[float, float, float, int, int]:
    """Bootstrap whole clusters so nested thresholds are not treated as iid rows."""
    required = set(cluster_columns) | {value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing cluster-bootstrap columns: {sorted(missing)}")
    work = frame.loc[:, list(cluster_columns) + [value_column]].copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna(subset=[value_column])
    if work.empty:
        return np.nan, np.nan, np.nan, 0, 0

    grouped = [group[value_column].to_numpy(dtype=float) for _, group in work.groupby(list(cluster_columns), sort=False)]
    cluster_count = len(grouped)
    observed = float(work[value_column].mean())
    if cluster_count < 2 or samples <= 0:
        return observed, np.nan, np.nan, int(len(work)), cluster_count

    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        chosen = rng.integers(0, cluster_count, size=cluster_count)
        values = np.concatenate([grouped[i] for i in chosen])
        boot[index] = float(values.mean())
    low, high = np.quantile(boot, [0.025, 0.975])
    return observed, float(low), float(high), int(len(work)), cluster_count


def summarize_cluster_uncertainty(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
    samples: int = 2000,
) -> pd.DataFrame:
    column = f"return_{horizon_min}m_{scenario}_net_return_pct"
    if column not in frame.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for threshold, group in frame.groupby("threshold_pct", sort=True):
        mean, low, high, n, clusters = cluster_bootstrap_mean_ci(group, column, samples=samples)
        day_mean, day_low, day_high, _, day_clusters = cluster_bootstrap_mean_ci(
            group,
            column,
            cluster_columns=("trading_day",),
            samples=samples,
            seed=20260918,
        )
        rows.append(
            {
                "threshold_pct": float(threshold),
                "horizon_min": horizon_min,
                "scenario": scenario,
                "n": n,
                "ticker_day_clusters": clusters,
                "mean_return_pct": mean,
                "ticker_day_ci_low": low,
                "ticker_day_ci_high": high,
                "trading_day_clusters": day_clusters,
                "day_cluster_mean_pct": day_mean,
                "day_ci_low": day_low,
                "day_ci_high": day_high,
            }
        )
    return pd.DataFrame(rows)
