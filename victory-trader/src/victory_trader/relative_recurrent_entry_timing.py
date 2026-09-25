"""Request 207: relative recurrent entry timing.

Request 206 showed that an absolute predicted-value > 0 boundary caused the
post-WAIT controller to wait almost forever. This diagnostic removes that
absolute boundary and asks only a local timing question:

    is ENTER now better than waiting exactly one more minute?

The same relative comparison is repeated at every observed checkpoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .recurrent_wait_entry_action_value import (
    BOOTSTRAP_SAMPLES,
    MAX_WAIT_MINUTES,
    WATCH_FEATURES,
    build_watch_states,
)
from .state_action_risk import SEVERE_LOSS_PCT
from .wait_reobserve_entry_confirmation import _day_weights, _feature_frame

REQUEST_ID = 207
MODEL_SEED = 20261109
BOOTSTRAP_SEED = 20261110
MIN_STATE_COVERAGE = 0.80
MIN_TRADES = 100
MIN_WAIT_USAGE = 0.10
MIN_IMPROVED_DAYS = 4


@dataclass(frozen=True)
class AdvantageModel:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def attach_relative_advantage(states: pd.DataFrame) -> pd.DataFrame:
    result = states.copy()
    result["next_minute_enter_3m_base_pct"] = np.nan
    result["enter_vs_wait1_advantage_pct"] = np.nan

    for _, group in result.groupby(EPISODE_KEYS, sort=False):
        by_minute = {
            int(row["minutes_since_hot"]): int(index)
            for index, row in group.iterrows()
        }
        for minute in range(1, MAX_WAIT_MINUTES):
            index = by_minute.get(minute)
            next_index = by_minute.get(minute + 1)
            if index is None or next_index is None:
                continue
            current = pd.to_numeric(
                pd.Series([result.at[index, "enter_3m_base_pct"]]), errors="coerce"
            ).iloc[0]
            later = pd.to_numeric(
                pd.Series([result.at[next_index, "enter_3m_base_pct"]]), errors="coerce"
            ).iloc[0]
            if pd.isna(current) or pd.isna(later):
                continue
            result.at[index, "next_minute_enter_3m_base_pct"] = float(later)
            result.at[index, "enter_vs_wait1_advantage_pct"] = float(current - later)
    return result


def train_model(fit: pd.DataFrame, calibration: pd.DataFrame) -> AdvantageModel:
    target = pd.to_numeric(fit["enter_vs_wait1_advantage_pct"], errors="coerce")
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 300:
        raise ValueError("request 207 needs >=300 relative-advantage fit rows")

    columns = tuple(
        column
        for column in WATCH_FEATURES
        if column in train.columns
        and pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=MODEL_SEED,
    )
    model.fit(
        _feature_frame(train, columns),
        y.clip(lower=low, upper=high),
        sample_weight=_day_weights(train),
    )

    cal_target = pd.to_numeric(
        calibration["enter_vs_wait1_advantage_pct"], errors="coerce"
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 150:
        raise ValueError("request 207 needs >=150 calibration rows")
    raw = model.predict(_feature_frame(cal, columns))
    residual = cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    offset = float(np.average(residual, weights=_day_weights(cal)))
    return AdvantageModel(
        model=model,
        columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict(frame: pd.DataFrame, fitted: AdvantageModel) -> np.ndarray:
    return fitted.model.predict(_feature_frame(frame, fitted.columns)) + fitted.offset


def _safe_spearman(actual: pd.Series, predicted: pd.Series) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = y.notna() & p.notna()
    if int(valid.sum()) < 20:
        return None
    value = y.loc[valid].corr(p.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _policy_trades(
    scored: pd.DataFrame,
    *,
    recurrent: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    counts = {
        "episodes": 0,
        "coverage_miss": 0,
        "wait_actions": 0,
        "entered": 0,
        "enter_minute_1": 0,
        "enter_minute_2": 0,
        "enter_minute_3": 0,
        "enter_minute_4": 0,
        "enter_minute_5": 0,
    }

    for _, group in scored.groupby(EPISODE_KEYS, sort=False):
        counts["episodes"] += 1
        by_minute = {
            int(row["minutes_since_hot"]): row
            for _, row in group.sort_values("minutes_since_hot", kind="stable").iterrows()
        }
        chosen = None
        if not recurrent:
            chosen = by_minute.get(1)
            if chosen is None:
                counts["coverage_miss"] += 1
                continue
        else:
            for minute in range(1, MAX_WAIT_MINUTES + 1):
                row = by_minute.get(minute)
                if row is None:
                    counts["coverage_miss"] += 1
                    break
                if minute == MAX_WAIT_MINUTES:
                    chosen = row
                    break
                advantage = pd.to_numeric(
                    pd.Series([row.get("predicted_relative_advantage_pct")]),
                    errors="coerce",
                ).iloc[0]
                if pd.isna(advantage):
                    counts["coverage_miss"] += 1
                    break
                if float(advantage) >= 0:
                    chosen = row
                    break
                counts["wait_actions"] += 1

        if chosen is None:
            continue
        realized = pd.to_numeric(
            pd.Series([chosen.get("enter_3m_base_pct")]), errors="coerce"
        ).iloc[0]
        if pd.isna(realized):
            counts["coverage_miss"] += 1
            continue
        minute = int(chosen["minutes_since_hot"])
        counts["entered"] += 1
        counts[f"enter_minute_{minute}"] += 1
        rows.append(
            {
                "trading_day": str(chosen["trading_day"]),
                "ticker": str(chosen["ticker"]),
                "hot_t": int(chosen["hot_t"]),
                "entry_minute_after_hot": minute,
                "realized_base_return_pct": float(realized),
            }
        )
    return pd.DataFrame(rows), counts


def _metrics(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty:
        return {"trades": 0}
    value = pd.to_numeric(trades["realized_base_return_pct"], errors="coerce")
    daily = (
        trades.assign(_value=value)
        .groupby(trades["trading_day"].astype(str))["_value"]
        .mean()
    )
    return {
        "trades": int(len(trades)),
        "days": int(len(daily)),
        "mean_pct": float(value.mean()),
        "day_balanced_mean_pct": float(daily.mean()),
        "positive_rate": float(value.gt(0).mean()),
        "severe_loss_rate": float(value.le(SEVERE_LOSS_PCT).mean()),
        "p05_pct": float(value.quantile(0.05)),
        "by_day": {str(k): float(v) for k, v in daily.items()},
    }


def _matched_difference(
    recurrent: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, object]:
    keys = EPISODE_KEYS
    left = recurrent.rename(
        columns={"realized_base_return_pct": "recurrent_base_pct"}
    )
    right = baseline.rename(
        columns={"realized_base_return_pct": "baseline_base_pct"}
    )
    matched = left.merge(
        right.loc[:, keys + ["baseline_base_pct"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if matched.empty:
        return {"episodes": 0}

    matched["difference_pct"] = (
        pd.to_numeric(matched["recurrent_base_pct"], errors="coerce")
        - pd.to_numeric(matched["baseline_base_pct"], errors="coerce")
    )
    daily = (
        matched.groupby(matched["trading_day"].astype(str))["difference_pct"]
        .mean()
        .dropna()
    )
    improved_days = int(daily.gt(0).sum())

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = daily.to_numpy(dtype=float)
    if len(values) >= 2:
        idx = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
        draws = values[idx].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        ci_low = float(low)
        ci_high = float(high)
    else:
        ci_low = None
        ci_high = None

    return {
        "episodes": int(len(matched)),
        "mean_difference_pct": float(matched["difference_pct"].mean()),
        "day_balanced_difference_pct": float(daily.mean()),
        "improved_days": improved_days,
        "by_day": {str(k): float(v) for k, v in daily.items()},
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "ci_low_pct": ci_low,
            "ci_high_pct": ci_high,
        },
    }


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit = attach_relative_advantage(
        build_watch_states(pd.read_parquet(fit_positions_path))
    )
    calibration = attach_relative_advantage(
        build_watch_states(pd.read_parquet(calibration_positions_path))
    )
    model = train_model(fit, calibration)

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(position_paths) != len(FRESH_DAYS):
        raise ValueError("request 207 requires complete request 178 position shards")
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths], ignore_index=True
    )
    fresh = attach_relative_advantage(build_watch_states(fresh_positions))
    if sorted(fresh["trading_day"].astype(str).unique()) != list(FRESH_DAYS):
        raise ValueError("request 207 fresh days differ from request 178")

    fresh["predicted_relative_advantage_pct"] = predict(fresh, model)
    spearman = _safe_spearman(
        fresh["enter_vs_wait1_advantage_pct"],
        fresh["predicted_relative_advantage_pct"],
    )

    total_episodes = int(
        fresh_positions.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    minute1 = int(
        fresh.loc[
            pd.to_numeric(fresh["minutes_since_hot"], errors="coerce").eq(1),
            EPISODE_KEYS,
        ].drop_duplicates().shape[0]
    )
    coverage = float(minute1 / total_episodes) if total_episodes else None

    baseline, baseline_counts = _policy_trades(fresh, recurrent=False)
    recurrent, recurrent_counts = _policy_trades(fresh, recurrent=True)
    baseline["policy"] = "always_enter_min1"
    recurrent["policy"] = "relative_recurrent_timing"

    baseline_metrics = _metrics(baseline)
    recurrent_metrics = _metrics(recurrent)
    difference = _matched_difference(recurrent, baseline)
    wait_usage = (
        float(recurrent_counts["wait_actions"] / recurrent_counts["episodes"])
        if recurrent_counts["episodes"]
        else None
    )

    diff_day = difference.get("day_balanced_difference_pct")
    diff_low = difference.get("bootstrap", {}).get("ci_low_pct")
    improved_days = int(difference.get("improved_days", 0))
    rec_severe = recurrent_metrics.get("severe_loss_rate")
    base_severe = baseline_metrics.get("severe_loss_rate")
    rec_trades = int(recurrent_metrics.get("trades", 0))

    gate = bool(
        coverage is not None
        and coverage >= MIN_STATE_COVERAGE
        and rec_trades >= MIN_TRADES
        and wait_usage is not None
        and wait_usage >= MIN_WAIT_USAGE
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
        and improved_days >= MIN_IMPROVED_DAYS
        and rec_severe is not None
        and base_severe is not None
        and float(rec_severe) <= float(base_severe)
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "question": "Can repeated relative ENTER-now vs WAIT-one-minute value improve entry timing?",
        "fresh_relative_advantage_spearman": spearman,
        "state_coverage": coverage,
        "model_features": list(model.columns),
        "model_offset": model.offset,
        "baseline": baseline_metrics,
        "recurrent": recurrent_metrics,
        "baseline_path_counts": baseline_counts,
        "recurrent_path_counts": recurrent_counts,
        "wait_usage_per_episode": wait_usage,
        "matched_recurrent_minus_baseline": difference,
        "frozen_gate": {
            "min_state_coverage": MIN_STATE_COVERAGE,
            "min_trades": MIN_TRADES,
            "min_wait_usage": MIN_WAIT_USAGE,
            "matched_day_difference_must_be_positive": True,
            "matched_bootstrap_low_must_be_positive": True,
            "min_improved_days": MIN_IMPROVED_DAYS,
            "severe_loss_no_worse_than_baseline": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.concat([baseline, recurrent], ignore_index=True).to_parquet(
        rows_output_path, index=False, compression="zstd"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument("--calibration-positions", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.fresh_dir,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
