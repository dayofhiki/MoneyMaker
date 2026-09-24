"""Request 187: fresh BUY horizon ceiling audit.

Pure diagnostic. Computes hindsight-best BASE exits within frozen horizons on
the already-opened Request-178 BUY population. These oracle ceilings are not
executable policy returns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS

REQUEST_ID = 187
HORIZONS = (1, 2, 3, 5, 10, 15, 30)
BOOTSTRAP_SEED = 20261087
BOOTSTRAP_SAMPLES = 10_000


def build_episode_ceilings(
    positions: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, group in positions.groupby(EPISODE_KEYS, sort=False):
        work = group.copy()
        work["_minute"] = pd.to_numeric(
            work["minutes_held"], errors="coerce"
        )
        work = work.loc[
            work["_minute"].notna() & work["_minute"].mod(1).eq(0)
        ].copy()
        work["_minute"] = work["_minute"].astype(int)
        if work.empty:
            continue
        first = work.sort_values("_minute", kind="stable").iloc[0]
        exits: list[tuple[int, float]] = []
        for _, row in work.iterrows():
            minute = int(row["_minute"])
            value = pd.to_numeric(
                pd.Series([row.get("exit_now_base_return_pct", np.nan)]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(value):
                exits.append((minute, float(value)))
            if minute == 29:
                minute30 = pd.to_numeric(
                    pd.Series(
                        [row.get("next_minute_base_return_pct", np.nan)]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(minute30):
                    exits.append((30, float(minute30)))

        record: dict[str, object] = {
            "trading_day": str(first["trading_day"]),
            "ticker": str(first["ticker"]).upper(),
            "hot_t": int(first["hot_t"]),
        }
        for horizon in HORIZONS:
            eligible = [value for minute, value in exits if minute <= horizon]
            record[f"oracle_{horizon}m_base_pct"] = (
                float(max(eligible)) if eligible else np.nan
            )
        records.append(record)
    return pd.DataFrame(records)


def _bootstrap(daily: np.ndarray, horizon: int) -> dict[str, object]:
    values = np.asarray(daily, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {
            "days": int(len(values)),
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": float(values.mean()) if len(values) else None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED + horizon)
    idx = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_SAMPLES, len(values)),
    )
    draws = values[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(values)),
        "samples": BOOTSTRAP_SAMPLES,
        "mean_pct": float(values.mean()),
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def horizon_metrics(
    frame: pd.DataFrame,
    horizon: int,
    total_episodes: int,
) -> dict[str, object]:
    column = f"oracle_{horizon}m_base_pct"
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna()
    observed = frame.loc[valid].copy()
    observed_values = values.loc[valid]
    daily = (
        pd.DataFrame(
            {
                "day": observed["trading_day"].astype(str),
                "value": observed_values.to_numpy(dtype=float),
            }
        )
        .groupby("day", sort=True)["value"]
        .mean()
    )
    positive_days = int(daily.gt(0).sum())

    by_day: dict[str, object] = {}
    for day in FRESH_DAYS:
        part = frame.loc[
            frame["trading_day"].astype(str).eq(day)
        ]
        v = pd.to_numeric(part[column], errors="coerce").dropna()
        by_day[day] = {
            "evaluable": int(len(v)),
            "mean_pct": float(v.mean()) if len(v) else None,
            "positive_rate": (
                float(v.gt(0).mean()) if len(v) else None
            ),
        }

    return {
        "evaluable_episodes": int(valid.sum()),
        "coverage": (
            float(valid.sum() / total_episodes)
            if total_episodes
            else None
        ),
        "oracle_mean_pct": (
            float(observed_values.mean()) if len(observed_values) else None
        ),
        "oracle_day_balanced_mean_pct": (
            float(daily.mean()) if len(daily) else None
        ),
        "oracle_median_pct": (
            float(observed_values.median()) if len(observed_values) else None
        ),
        "oracle_positive_rate": (
            float(observed_values.gt(0).mean())
            if len(observed_values)
            else None
        ),
        "oracle_p05_pct": (
            float(observed_values.quantile(0.05))
            if len(observed_values)
            else None
        ),
        "positive_opportunity_days": positive_days,
        "by_day": by_day,
        "day_bootstrap": _bootstrap(
            daily.to_numpy(dtype=float), horizon
        ),
    }


def evaluate(
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(paths) != len(FRESH_DAYS):
        raise ValueError("request 187 requires complete request 178 positions")
    positions = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )
    found = sorted(positions["trading_day"].astype(str).unique())
    if found != list(FRESH_DAYS):
        raise ValueError(f"request 187 expected {FRESH_DAYS}, found {found}")

    total_episodes = int(
        positions.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    ceilings = build_episode_ceilings(positions)
    metrics: dict[str, object] = {}
    previous_mean: float | None = None
    first_positive_day_balanced_horizon: int | None = None

    for horizon in HORIZONS:
        item = horizon_metrics(ceilings, horizon, total_episodes)
        mean = item["oracle_mean_pct"]
        item["incremental_mean_vs_previous_horizon_pct"] = (
            float(mean - previous_mean)
            if mean is not None and previous_mean is not None
            else None
        )
        if mean is not None:
            previous_mean = float(mean)
        day_mean = item["oracle_day_balanced_mean_pct"]
        if (
            first_positive_day_balanced_horizon is None
            and day_mean is not None
            and float(day_mean) > 0
        ):
            first_positive_day_balanced_horizon = horizon
        metrics[str(horizon)] = item

    final = metrics[str(HORIZONS[-1])]
    final_day_mean = final["oracle_day_balanced_mean_pct"]
    if final_day_mean is None:
        diagnosis = "unknown"
    elif float(final_day_mean) <= 0:
        diagnosis = "fresh_buy_population_ceiling_nonpositive"
    elif (
        first_positive_day_balanced_horizon is not None
        and first_positive_day_balanced_horizon <= 3
    ):
        diagnosis = "short_horizon_ceiling_positive_prediction_bottleneck"
    else:
        diagnosis = "multi_minute_opportunity_temporal_mismatch"

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "policy_changed": False,
        "promotion_eligible": False,
        "total_buy_episodes": total_episodes,
        "horizons_minutes": list(HORIZONS),
        "horizon_metrics": metrics,
        "first_positive_day_balanced_horizon_minutes": (
            first_positive_day_balanced_horizon
        ),
        "diagnosis": diagnosis,
        "interpretation": (
            "Hindsight executable-reference ceiling only. "
            "No oracle exit is available to a live policy."
        ),
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    ceilings.to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(args.fresh_dir, args.output, args.rows_output)


if __name__ == "__main__":
    raise SystemExit(main())
