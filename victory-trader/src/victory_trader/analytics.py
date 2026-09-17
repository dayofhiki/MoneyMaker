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
PRICE_BINS = [0.5, 2.0, 5.0, 10.0, 20.0000001]
PRICE_LABELS = ["$0.50-2", "$2-5", "$5-10", "$10-20"]
TIME_BINS = [-0.000001, 30.0, 60.0, 120.0, 240.0, float("inf")]
TIME_LABELS = ["open-30m", "30-60m", "60-120m", "120-240m", "240m-close"]


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
        return {
            "n": 0,
            "mean_return_pct": np.nan,
            "median_return_pct": np.nan,
            "positive_rate": np.nan,
            "p05_return_pct": np.nan,
            "p95_return_pct": np.nan,
        }
    return {
        "n": len(clean),
        "mean_return_pct": float(clean.mean()),
        "median_return_pct": float(clean.median()),
        "positive_rate": float((clean > 0).mean()),
        "p05_return_pct": float(clean.quantile(0.05)),
        "p95_return_pct": float(clean.quantile(0.95)),
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
                    "n": len(values),
                    "mean_return_pct": float(values.mean()),
                    "median_return_pct": float(values.median()),
                    "positive_rate": float((values > 0).mean()),
                    "p05_return_pct": float(values.quantile(0.05)),
                    "p25_return_pct": float(values.quantile(0.25)),
                    "p75_return_pct": float(values.quantile(0.75)),
                    "p95_return_pct": float(values.quantile(0.95)),
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
            rows.append(
                {
                    "threshold_pct": float(threshold),
                    "horizon_min": horizon_min,
                    "scenario": "gross",
                    **_basic_row(gross),
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
                    **_basic_row(values),
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
                "n": len(group),
                "mean_mfe_pct": float(mfe.mean()) if not mfe.empty else np.nan,
                "median_mfe_pct": float(mfe.median()) if not mfe.empty else np.nan,
                "mean_mae_pct": float(mae.mean()) if not mae.empty else np.nan,
                "median_mae_pct": float(mae.median()) if not mae.empty else np.nan,
                "p05_mae_pct": float(mae.quantile(0.05)) if not mae.empty else np.nan,
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
    resolved_statuses = ["take_profit", "stop_loss", "stop_gap", "timeout", "session_close"]
    for threshold, group in frame.groupby("threshold_pct", sort=True):
        for key in sorted(barrier_keys):
            status_col = f"{key}_status"
            return_col = f"{key}_exit_return_pct"
            statuses = group[status_col].astype("string")
            resolved = statuses.isin(resolved_statuses)
            returns = pd.to_numeric(group.loc[resolved, return_col], errors="coerce").dropna()
            resolved_count = int(resolved.sum())
            row = {
                "threshold_pct": float(threshold),
                "barrier": key,
                "n": len(group),
                "resolved_n": resolved_count,
                "take_profit_rate": float((statuses == "take_profit").sum() / resolved_count) if resolved_count else np.nan,
                "stop_loss_rate": float(statuses.isin(["stop_loss", "stop_gap"]).sum() / resolved_count) if resolved_count else np.nan,
                "stop_gap_rate": float((statuses == "stop_gap").mean()),
                "timeout_rate": float((statuses == "timeout").sum() / resolved_count) if resolved_count else np.nan,
                "session_close_rate": float((statuses == "session_close").sum() / resolved_count) if resolved_count else np.nan,
                "ambiguous_rate": float((statuses == "ambiguous").mean()),
                "unresolved_missing_rate": float((statuses == "unresolved_missing").mean()),
                "unresolved_session_close_rate": float((statuses == "unresolved_session_close").mean()),
                "entry_unavailable_rate": float((statuses == "entry_unavailable").mean()),
                "mean_exit_return_pct": float(returns.mean()) if not returns.empty else np.nan,
                "median_exit_return_pct": float(returns.median()) if not returns.empty else np.nan,
                "p05_exit_return_pct": float(returns.quantile(0.05)) if not returns.empty else np.nan,
            }
            stop = _stop_from_key(key)
            if stop is not None:
                conservative = pd.to_numeric(group[return_col], errors="coerce").copy()
                conservative.loc[statuses == "ambiguous"] = -stop
                conservative = conservative.loc[
                    ~statuses.isin(
                        [
                            "entry_unavailable",
                            "unresolved_missing",
                            "unresolved_session_close",
                            "no_future_data",
                        ]
                    )
                ].dropna()
                row["ambiguous_as_stop_mean_pct"] = (
                    float(conservative.mean()) if not conservative.empty else np.nan
                )
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
        rows.append(
            {
                "threshold_pct": float(threshold),
                "rvol_bucket": str(bucket),
                "horizon_min": horizon_min,
                **_basic_row(group[return_col]),
            }
        )
    return pd.DataFrame(rows)


def summarize_missingness(frame: pd.DataFrame, *, horizon_min: int = 5) -> pd.DataFrame:
    column = f"return_{horizon_min}m_pct"
    if column not in frame.columns:
        raise ValueError(f"dataset missing {column}")
    work = frame.copy()
    work["_session"] = np.select(
        [
            work.get("is_premarket", False),
            work.get("is_regular_session", False),
            work.get("is_after_hours", False),
        ],
        ["premarket", "regular", "after_hours"],
        default="other",
    )
    rows: list[dict] = []
    for (threshold, session), group in work.groupby(["threshold_pct", "_session"], sort=True):
        present = pd.to_numeric(group[column], errors="coerce").notna()
        entry_present = (
            pd.to_numeric(group["entry_price"], errors="coerce").notna()
            if "entry_price" in group
            else pd.Series(False, index=group.index)
        )
        rows.append(
            {
                "threshold_pct": float(threshold),
                "session": str(session),
                "n": len(group),
                "entry_available_rate": float(entry_present.mean()),
                "outcome_available_rate": float(present.mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_latency(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
) -> pd.DataFrame:
    rows: list[dict] = []
    for delay, prefix in ((0, ""), (1, "delay1_"), (2, "delay2_")):
        column = f"{prefix}return_{horizon_min}m_{scenario}_net_return_pct"
        if column not in frame.columns:
            continue
        for threshold, group in frame.groupby("threshold_pct", sort=True):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            rows.append(
                {
                    "threshold_pct": float(threshold),
                    "delay_min": delay,
                    "scenario": scenario,
                    **_basic_row(values),
                }
            )
    return pd.DataFrame(rows)


def summarize_sessions(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
) -> pd.DataFrame:
    column = f"return_{horizon_min}m_{scenario}_net_return_pct"
    if column not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    work["session"] = np.select(
        [
            work.get("is_premarket", False),
            work.get("is_regular_session", False),
            work.get("is_after_hours", False),
        ],
        ["premarket", "regular", "after_hours"],
        default="other",
    )
    rows: list[dict] = []
    for (threshold, session), group in work.groupby(["threshold_pct", "session"], sort=True):
        rows.append(
            {
                "threshold_pct": float(threshold),
                "session": str(session),
                **_basic_row(group[column]),
            }
        )
    return pd.DataFrame(rows)


def summarize_price_buckets(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
) -> pd.DataFrame:
    column = f"return_{horizon_min}m_{scenario}_net_return_pct"
    required = {"threshold_pct", "previous_close", column}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    work = frame.loc[:, list(required)].copy()
    work["previous_close"] = pd.to_numeric(work["previous_close"], errors="coerce")
    work["price_bucket"] = pd.cut(
        work["previous_close"],
        bins=PRICE_BINS,
        labels=PRICE_LABELS,
        right=False,
        include_lowest=True,
    )
    rows: list[dict] = []
    for (threshold, bucket), group in work.groupby(
        ["threshold_pct", "price_bucket"], observed=True, sort=True
    ):
        rows.append(
            {
                "threshold_pct": float(threshold),
                "price_bucket": str(bucket),
                **_basic_row(group[column]),
            }
        )
    return pd.DataFrame(rows)


def summarize_time_buckets(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
) -> pd.DataFrame:
    column = f"return_{horizon_min}m_{scenario}_net_return_pct"
    required = {"threshold_pct", "minutes_from_regular_open", column}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    work = frame.loc[:, list(required)].copy()
    work["minutes_from_regular_open"] = pd.to_numeric(
        work["minutes_from_regular_open"], errors="coerce"
    )
    work["time_bucket"] = pd.cut(
        work["minutes_from_regular_open"],
        bins=TIME_BINS,
        labels=TIME_LABELS,
        right=False,
        include_lowest=True,
    )
    rows: list[dict] = []
    for (threshold, bucket), group in work.groupby(
        ["threshold_pct", "time_bucket"], observed=True, sort=True
    ):
        rows.append(
            {
                "threshold_pct": float(threshold),
                "time_bucket": str(bucket),
                **_basic_row(group[column]),
            }
        )
    return pd.DataFrame(rows)


def summarize_monthly_stability(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
) -> pd.DataFrame:
    column = f"return_{horizon_min}m_{scenario}_net_return_pct"
    required = {"trading_day", "threshold_pct", column}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    work = frame.loc[:, list(required)].copy()
    work["month"] = pd.to_datetime(work["trading_day"], errors="coerce").dt.to_period("M").astype("string")
    rows: list[dict] = []
    for (threshold, month), group in work.groupby(["threshold_pct", "month"], sort=True):
        rows.append(
            {
                "threshold_pct": float(threshold),
                "month": str(month),
                **_basic_row(group[column]),
            }
        )
    return pd.DataFrame(rows)


def summarize_tail_risk(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
) -> pd.DataFrame:
    column = f"return_{horizon_min}m_{scenario}_net_return_pct"
    if column not in frame.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for threshold, group in frame.groupby("threshold_pct", sort=True):
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        if values.empty:
            continue
        p05 = float(values.quantile(0.05))
        p01 = float(values.quantile(0.01))
        tail5 = values.loc[values <= p05]
        rows.append(
            {
                "threshold_pct": float(threshold),
                "n": len(values),
                "worst_return_pct": float(values.min()),
                "p01_return_pct": p01,
                "p05_return_pct": p05,
                "cvar05_return_pct": float(tail5.mean()) if not tail5.empty else np.nan,
                "best_return_pct": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_concentration(
    frame: pd.DataFrame,
    *,
    horizon_min: int = 5,
    scenario: str = "base",
) -> pd.DataFrame:
    column = f"return_{horizon_min}m_{scenario}_net_return_pct"
    required = {"trading_day", "ticker", "threshold_pct", column}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    rows: list[dict] = []
    for threshold, group in frame.groupby("threshold_pct", sort=True):
        work = group.loc[:, ["trading_day", "ticker", column]].copy()
        work[column] = pd.to_numeric(work[column], errors="coerce")
        work = work.dropna(subset=[column])
        if work.empty:
            continue
        cluster = work.groupby(["trading_day", "ticker"], sort=False)[column].sum()
        absolute = cluster.abs().sort_values(ascending=False)
        denominator = float(absolute.sum())
        rows.append(
            {
                "threshold_pct": float(threshold),
                "n": len(work),
                "ticker_day_clusters": len(cluster),
                "top1_abs_return_share": (
                    float(absolute.iloc[:1].sum() / denominator) if denominator > 0 else np.nan
                ),
                "top5_abs_return_share": (
                    float(absolute.iloc[:5].sum() / denominator) if denominator > 0 else np.nan
                ),
                "unique_tickers": int(work["ticker"].nunique()),
                "unique_days": int(work["trading_day"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_mean_ci(
    frame: pd.DataFrame,
    value_column: str,
    *,
    cluster_columns: tuple[str, ...] = ("trading_day", "ticker"),
    samples: int = 2000,
    seed: int = 20260917,
) -> tuple[float, float, float, int, int]:
    required = set(cluster_columns) | {value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing cluster-bootstrap columns: {sorted(missing)}")
    work = frame.loc[:, list(cluster_columns) + [value_column]].copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna(subset=[value_column])
    if work.empty:
        return np.nan, np.nan, np.nan, 0, 0

    grouped = [
        group[value_column].to_numpy(dtype=float)
        for _, group in work.groupby(list(cluster_columns), sort=False)
    ]
    cluster_count = len(grouped)
    observed = float(work[value_column].mean())
    if cluster_count < 2 or samples <= 0:
        return observed, np.nan, np.nan, len(work), cluster_count

    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        chosen = rng.integers(0, cluster_count, size=cluster_count)
        values = np.concatenate([grouped[i] for i in chosen])
        boot[index] = float(values.mean())
    low, high = np.quantile(boot, [0.025, 0.975])
    return observed, float(low), float(high), len(work), cluster_count


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
        mean, low, high, n, clusters = cluster_bootstrap_mean_ci(
            group,
            column,
            samples=samples,
        )
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
