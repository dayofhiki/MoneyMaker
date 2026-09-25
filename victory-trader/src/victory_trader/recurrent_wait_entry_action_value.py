"""Request 206: recurrent WAIT-state entry action-value diagnostic.

This development-only bridge starts after an initial WAIT. At each exact minute
1..5 after HOT, the controller compares the fitted value of ENTER now with the
fitted continuation value of WAIT one more minute and with ABSTAIN=0.

Only causal market-state columns are model inputs. Position-dependent P&L/path
columns and future execution references are labels only.
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
from .execution_costs import net_round_trip_return_pct
from .selected_hot_position_value_observability import BASE_SCENARIO
from .state_action_risk import SEVERE_LOSS_PCT
from .wait_reobserve_entry_confirmation import (
    WAIT_STATE_FEATURES,
    _day_weights,
    _feature_frame,
)

REQUEST_ID = 206
MODEL_SEED_ENTER = 20261106
MODEL_SEED_WAIT = 20261107
BOOTSTRAP_SEED = 20261108
BOOTSTRAP_SAMPLES = 10_000
MAX_WAIT_MINUTES = 5
HOLD_MINUTES = 3
MIN_STATE_COVERAGE = 0.80
MIN_TRADES = 30
MIN_WAIT_USAGE = 0.10
MIN_POSITIVE_DAYS = 4

WATCH_FEATURES = tuple(dict.fromkeys([*WAIT_STATE_FEATURES, "minutes_since_hot"]))


@dataclass(frozen=True)
class ValueModel:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def _positive(value: object) -> bool:
    return bool(pd.notna(value) and np.isfinite(float(value)) and float(value) > 0)


def _base_return(entry_open: object, exit_open: object) -> float:
    if not (_positive(entry_open) and _positive(exit_open)):
        return np.nan
    gross = (float(exit_open) / float(entry_open) - 1.0) * 100.0
    return float(net_round_trip_return_pct(float(entry_open), gross, BASE_SCENARIO))


def build_watch_states(position_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    total_episodes = int(position_rows.loc[:, EPISODE_KEYS].drop_duplicates().shape[0])

    for _, group in position_rows.groupby(EPISODE_KEYS, sort=False):
        work = group.copy()
        held = pd.to_numeric(work["minutes_held"], errors="coerce")
        work = work.loc[held.notna() & held.mod(1).eq(0)].copy()
        work["_minute"] = pd.to_numeric(work["minutes_held"], errors="coerce").astype(int)
        minute_map = {
            int(row["_minute"]): row
            for _, row in work.sort_values("_minute", kind="stable").iterrows()
        }

        for minute in range(1, MAX_WAIT_MINUTES + 1):
            row = minute_map.get(minute)
            if row is None:
                continue
            exit_row = minute_map.get(minute + HOLD_MINUTES)

            log_price = pd.to_numeric(
                pd.Series([row.get("log_current_close")]), errors="coerce"
            ).iloc[0]
            current_price = (
                float(np.exp(float(log_price)))
                if pd.notna(log_price) and np.isfinite(float(log_price))
                else np.nan
            )
            zero_cost = (
                float(-net_round_trip_return_pct(current_price, 0.0, BASE_SCENARIO))
                if _positive(current_price)
                else np.nan
            )
            entry_open = pd.to_numeric(
                pd.Series([row.get("exit_reference_open")]), errors="coerce"
            ).iloc[0]
            exit_open = (
                pd.to_numeric(
                    pd.Series([exit_row.get("exit_reference_open")]), errors="coerce"
                ).iloc[0]
                if exit_row is not None
                else np.nan
            )

            record: dict[str, object] = {
                "trading_day": str(row["trading_day"]),
                "ticker": str(row["ticker"]).upper(),
                "hot_t": int(row["hot_t"]),
                "minutes_since_hot": int(minute),
                "log_current_price": (
                    float(log_price) if pd.notna(log_price) else np.nan
                ),
                "current_price": current_price,
                "base_zero_move_cost_proxy_pct": zero_cost,
                "enter_3m_base_pct": _base_return(entry_open, exit_open),
            }
            for column in WAIT_STATE_FEATURES:
                if column in {
                    "log_current_price",
                    "base_zero_move_cost_proxy_pct",
                }:
                    continue
                record[column] = row.get(column, np.nan)
            records.append(record)

    result = pd.DataFrame(records)
    result.attrs["total_episodes"] = total_episodes
    if result.empty:
        return result

    result["oracle_state_value_pct"] = np.nan
    result["oracle_wait_value_pct"] = np.nan
    for _, group in result.groupby(EPISODE_KEYS, sort=False):
        by_minute = {
            int(row["minutes_since_hot"]): int(index)
            for index, row in group.iterrows()
        }
        next_value = 0.0
        for minute in range(MAX_WAIT_MINUTES, 0, -1):
            index = by_minute.get(minute)
            if index is None:
                next_value = 0.0
                continue
            direct = pd.to_numeric(
                pd.Series([result.at[index, "enter_3m_base_pct"]]), errors="coerce"
            ).iloc[0]
            wait_value = float(next_value) if minute < MAX_WAIT_MINUTES else 0.0
            result.at[index, "oracle_wait_value_pct"] = wait_value
            candidates = [0.0, wait_value]
            if pd.notna(direct) and np.isfinite(float(direct)):
                candidates.append(float(direct))
            state_value = float(max(candidates))
            result.at[index, "oracle_state_value_pct"] = state_value
            next_value = state_value
    return result


def _train_value_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    target_column: str,
    *,
    seed: int,
) -> ValueModel:
    target = pd.to_numeric(fit[target_column], errors="coerce")
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 300:
        raise ValueError(f"request 206 needs >=300 {target_column} fit rows")

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
        random_state=seed,
    )
    model.fit(
        _feature_frame(train, columns),
        y.clip(lower=low, upper=high),
        sample_weight=_day_weights(train),
    )

    cal_target = pd.to_numeric(calibration[target_column], errors="coerce")
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 150:
        raise ValueError(f"request 206 needs >=150 {target_column} calibration rows")
    raw = model.predict(_feature_frame(cal, columns))
    residual = cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    offset = float(np.average(residual, weights=_day_weights(cal)))
    return ValueModel(
        model=model,
        columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def _predict(frame: pd.DataFrame, fitted: ValueModel) -> np.ndarray:
    return fitted.model.predict(_feature_frame(frame, fitted.columns)) + fitted.offset


def _bootstrap(trades: pd.DataFrame) -> dict[str, float | int | None]:
    if trades.empty:
        return {
            "days": 0,
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    daily = (
        trades.assign(
            _value=pd.to_numeric(trades["realized_base_return_pct"], errors="coerce")
        )
        .groupby(trades["trading_day"].astype(str))["_value"]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(daily) < 2:
        return {
            "days": int(len(daily)),
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": float(daily.mean()) if len(daily) else None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(daily), size=(BOOTSTRAP_SAMPLES, len(daily)))
    draws = daily[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(daily)),
        "samples": BOOTSTRAP_SAMPLES,
        "mean_pct": float(daily.mean()),
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def _metrics(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty:
        return {"trades": 0}
    value = pd.to_numeric(trades["realized_base_return_pct"], errors="coerce")
    daily = (
        trades.assign(_value=value)
        .groupby(trades["trading_day"].astype(str))["_value"]
        .mean()
    )
    positive_days = int(daily.gt(0).sum())
    return {
        "trades": int(len(trades)),
        "days": int(len(daily)),
        "mean_pct": float(value.mean()),
        "day_balanced_mean_pct": float(daily.mean()),
        "median_pct": float(value.median()),
        "positive_rate": float(value.gt(0).mean()),
        "severe_loss_rate": float(value.le(SEVERE_LOSS_PCT).mean()),
        "p05_pct": float(value.quantile(0.05)),
        "positive_days": positive_days,
        "by_day": {str(k): float(v) for k, v in daily.items()},
        "bootstrap": _bootstrap(trades),
    }


def _run_policy(
    scored: pd.DataFrame,
    *,
    policy: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    trades: list[dict[str, object]] = []
    counts = {
        "episodes": 0,
        "coverage_miss": 0,
        "wait_actions": 0,
        "entered": 0,
        "abstained": 0,
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
        row1 = by_minute.get(1)
        if row1 is None:
            counts["coverage_miss"] += 1
            continue

        if policy == "always_enter_min1":
            chosen = row1
        elif policy == "one_shot_positive_enter":
            chosen = row1 if float(row1["predicted_enter_value_pct"]) > 0 else None
            if chosen is None:
                counts["abstained"] += 1
                continue
        elif policy == "recurrent_action_value":
            chosen = None
            for minute in range(1, MAX_WAIT_MINUTES + 1):
                row = by_minute.get(minute)
                if row is None:
                    counts["coverage_miss"] += 1
                    break
                q_enter = float(row["predicted_enter_value_pct"])
                q_wait = (
                    float(row["predicted_wait_value_pct"])
                    if minute < MAX_WAIT_MINUTES
                    else -np.inf
                )
                if q_enter > 0 and q_enter >= q_wait:
                    chosen = row
                    break
                if minute < MAX_WAIT_MINUTES and q_wait > 0:
                    counts["wait_actions"] += 1
                    continue
                counts["abstained"] += 1
                break
            if chosen is None:
                continue
        else:
            raise ValueError(f"unknown policy: {policy}")

        realized = pd.to_numeric(
            pd.Series([chosen.get("enter_3m_base_pct")]), errors="coerce"
        ).iloc[0]
        if pd.isna(realized):
            counts["coverage_miss"] += 1
            continue
        minute = int(chosen["minutes_since_hot"])
        counts["entered"] += 1
        counts[f"enter_minute_{minute}"] += 1
        trades.append(
            {
                "trading_day": str(chosen["trading_day"]),
                "ticker": str(chosen["ticker"]),
                "hot_t": int(chosen["hot_t"]),
                "entry_minute_after_hot": minute,
                "realized_base_return_pct": float(realized),
                "predicted_enter_value_pct": float(chosen["predicted_enter_value_pct"]),
                "predicted_wait_value_pct": float(chosen["predicted_wait_value_pct"]),
                "policy": policy,
            }
        )
    return pd.DataFrame(trades), counts


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit = build_watch_states(pd.read_parquet(fit_positions_path))
    calibration = build_watch_states(pd.read_parquet(calibration_positions_path))

    enter_model = _train_value_model(
        fit, calibration, "enter_3m_base_pct", seed=MODEL_SEED_ENTER
    )
    wait_model = _train_value_model(
        fit, calibration, "oracle_wait_value_pct", seed=MODEL_SEED_WAIT
    )

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(position_paths) != len(FRESH_DAYS):
        raise ValueError("request 206 requires complete request 178 position shards")
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths], ignore_index=True
    )
    fresh = build_watch_states(fresh_positions)
    found = sorted(fresh["trading_day"].astype(str).unique())
    if found != list(FRESH_DAYS):
        raise ValueError(f"request 206 expected {FRESH_DAYS}, found {found}")

    fresh["predicted_enter_value_pct"] = _predict(fresh, enter_model)
    fresh["predicted_wait_value_pct"] = _predict(fresh, wait_model)

    total_episodes = int(
        fresh_positions.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    minute1_episodes = int(
        fresh.loc[pd.to_numeric(fresh["minutes_since_hot"], errors="coerce").eq(1),
                  EPISODE_KEYS].drop_duplicates().shape[0]
    )
    coverage = float(minute1_episodes / total_episodes) if total_episodes else None

    policy_results: dict[str, object] = {}
    policy_trades: list[pd.DataFrame] = []
    path_counts: dict[str, object] = {}
    for policy in (
        "always_enter_min1",
        "one_shot_positive_enter",
        "recurrent_action_value",
    ):
        trades, counts = _run_policy(fresh, policy=policy)
        policy_results[policy] = _metrics(trades)
        path_counts[policy] = counts
        if len(trades):
            policy_trades.append(trades)

    recurrent = policy_results["recurrent_action_value"]
    one_shot = policy_results["one_shot_positive_enter"]
    rec_counts = path_counts["recurrent_action_value"]
    wait_usage = (
        float(rec_counts["wait_actions"] / rec_counts["episodes"])
        if rec_counts["episodes"]
        else None
    )
    rec_mean = recurrent.get("day_balanced_mean_pct")
    one_mean = one_shot.get("day_balanced_mean_pct")
    rec_severe = recurrent.get("severe_loss_rate")
    one_severe = one_shot.get("severe_loss_rate")
    rec_low = recurrent.get("bootstrap", {}).get("ci_low_pct")
    rec_positive_days = recurrent.get("positive_days", 0)
    rec_trades = int(recurrent.get("trades", 0))

    gate = bool(
        coverage is not None
        and coverage >= MIN_STATE_COVERAGE
        and rec_trades >= MIN_TRADES
        and wait_usage is not None
        and wait_usage >= MIN_WAIT_USAGE
        and rec_mean is not None
        and float(rec_mean) > 0
        and one_mean is not None
        and float(rec_mean) > float(one_mean)
        and rec_severe is not None
        and one_severe is not None
        and float(rec_severe) <= float(one_severe)
        and rec_low is not None
        and float(rec_low) > 0
        and int(rec_positive_days) >= MIN_POSITIVE_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "architecture": "post-WAIT recurrent ENTER/WAIT/ABSTAIN action-value controller",
        "max_wait_minutes": MAX_WAIT_MINUTES,
        "fixed_entry_evaluation_hold_minutes": HOLD_MINUTES,
        "state_coverage": coverage,
        "models": {
            "enter_features": list(enter_model.columns),
            "wait_features": list(wait_model.columns),
            "enter_offset": enter_model.offset,
            "wait_offset": wait_model.offset,
        },
        "policies": policy_results,
        "path_counts": path_counts,
        "wait_usage_per_episode": wait_usage,
        "frozen_gate": {
            "min_state_coverage": MIN_STATE_COVERAGE,
            "min_trades": MIN_TRADES,
            "min_wait_usage": MIN_WAIT_USAGE,
            "recurrent_day_balanced_mean_must_be_positive": True,
            "recurrent_must_beat_one_shot": True,
            "severe_loss_no_worse_than_one_shot": True,
            "bootstrap_low_must_be_positive": True,
            "min_positive_days": MIN_POSITIVE_DAYS,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    if policy_trades:
        pd.concat(policy_trades, ignore_index=True).to_parquet(
            rows_output_path, index=False, compression="zstd"
        )
    else:
        pd.DataFrame().to_parquet(rows_output_path, index=False)
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
