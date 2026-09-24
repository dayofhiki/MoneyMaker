"""Request 186: WAIT then re-observe entry confirmation.

No capital is committed at the original HOT timestamp. The model observes the
exact minute-1 causal market state and decides whether the delayed entry has
enough three-minute BASE opportunity to justify ENTER versus ABSTAIN.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .decomposed_cost_aware_entry_surplus import (
    BASE_SCENARIO,
    EPISODE_KEYS,
    FRESH_DAYS,
)
from .execution_costs import net_round_trip_return_pct
from .rich_second_position_value import RICH_SECOND_FEATURES
from .second_path_attention_probe import BASELINE_FEATURES, SECOND_FEATURES
from .three_minute_cost_cover_entry_opportunity import (
    build_three_minute_labels,
)

REQUEST_ID = 186
MODEL_SEED = 20261090
BOOTSTRAP_SEED = 20261086
BOOTSTRAP_SAMPLES = 10_000
REQUEST185_WAIT_SPEARMAN = 0.18399598640288747
MIN_WAIT_STATE_COVERAGE = 0.85
MIN_RICH_SECOND_COVERAGE = 0.90
MIN_SPEARMAN = 0.20
MIN_SPEARMAN_IMPROVEMENT = 0.03
MIN_SELECTED_RATE = 0.05
MAX_SELECTED_RATE = 0.60
MIN_SELECTED_POSITIVE_RATE = 0.60
MIN_UPLIFT_PCT = 0.50
MIN_POSITIVE_DAYS = 4

WAIT_STATE_FEATURES = tuple(
    dict.fromkeys(
        [
            *BASELINE_FEATURES,
            *SECOND_FEATURES,
            *RICH_SECOND_FEATURES,
            "log_current_price",
            "minutes_since_open",
            "minutes_to_close",
            "base_zero_move_cost_proxy_pct",
        ]
    )
)


@dataclass(frozen=True)
class WaitModel:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    day = frame["trading_day"].astype(str)
    counts = day.map(day.value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(dtype=float)
    mean = float(np.mean(weights))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("request 186 invalid equal-day weights")
    return weights / mean


def build_wait_states(position_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    keep = tuple(
        dict.fromkeys(
            [
                *BASELINE_FEATURES,
                *SECOND_FEATURES,
                *RICH_SECOND_FEATURES,
                "minutes_since_open",
                "minutes_to_close",
            ]
        )
    )

    total = int(position_rows.loc[:, EPISODE_KEYS].drop_duplicates().shape[0])
    for _, group in position_rows.groupby(EPISODE_KEYS, sort=False):
        held = pd.to_numeric(group["minutes_held"], errors="coerce")
        row1 = group.loc[held.eq(1)]
        if row1.empty:
            continue
        row = row1.sort_values("state_t", kind="stable").iloc[0]
        record: dict[str, object] = {
            "trading_day": str(row["trading_day"]),
            "ticker": str(row["ticker"]).upper(),
            "hot_t": int(row["hot_t"]),
        }
        for column in keep:
            record[column] = row.get(column, np.nan)

        log_price = pd.to_numeric(
            pd.Series([row.get("log_current_close")]), errors="coerce"
        ).iloc[0]
        current_price = (
            float(np.exp(float(log_price)))
            if pd.notna(log_price) and np.isfinite(float(log_price))
            else np.nan
        )
        record["log_current_price"] = (
            float(log_price) if pd.notna(log_price) else np.nan
        )
        record["current_price"] = current_price
        if pd.notna(current_price) and current_price > 0:
            record["base_zero_move_cost_proxy_pct"] = float(
                -net_round_trip_return_pct(
                    current_price,
                    0.0,
                    BASE_SCENARIO,
                )
            )
        else:
            record["base_zero_move_cost_proxy_pct"] = np.nan
        records.append(record)

    result = pd.DataFrame(records)
    if len(result):
        result["wait_state_available"] = True
    result.attrs["total_episodes"] = total
    return result


def attach_wait_target(
    position_rows: pd.DataFrame,
) -> pd.DataFrame:
    states = build_wait_states(position_rows)
    labels = build_three_minute_labels(position_rows).loc[
        :, EPISODE_KEYS + ["wait_3m_base_pct"]
    ]
    result = states.merge(
        labels,
        on=EPISODE_KEYS,
        how="left",
        validate="one_to_one",
    )
    return result


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric, errors="coerce"
    )


def train_wait_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> WaitModel:
    target = pd.to_numeric(fit["wait_3m_base_pct"], errors="coerce")
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 300:
        raise ValueError(
            f"request 186 needs >=300 fit targets, got {len(train)}"
        )

    columns = tuple(
        column
        for column in WAIT_STATE_FEATURES
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
        calibration["wait_3m_base_pct"], errors="coerce"
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 150:
        raise ValueError(
            f"request 186 needs >=150 calibration targets, got {len(cal)}"
        )
    raw = model.predict(_feature_frame(cal, columns))
    residual = cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    offset = float(np.average(residual, weights=_day_weights(cal)))
    return WaitModel(
        model=model,
        columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_wait(frame: pd.DataFrame, fitted: WaitModel) -> np.ndarray:
    return (
        fitted.model.predict(_feature_frame(frame, fitted.columns))
        + fitted.offset
    )


def _safe_spearman(
    actual: pd.Series,
    predicted: pd.Series,
) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = y.notna() & p.notna()
    if int(valid.sum()) < 20:
        return None
    value = y.loc[valid].corr(p.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _day_balanced(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return None
    daily = (
        pd.DataFrame(
            {
                "day": frame.loc[valid, "trading_day"].astype(str),
                "value": values.loc[valid].to_numpy(dtype=float),
            }
        )
        .groupby("day", sort=True)["value"]
        .mean()
    )
    return float(daily.mean()) if len(daily) else None


def _bootstrap(selected: pd.DataFrame) -> dict[str, float | int | None]:
    value = pd.to_numeric(
        selected["wait_3m_base_pct"], errors="coerce"
    )
    valid = value.notna()
    daily = (
        pd.DataFrame(
            {
                "day": selected.loc[valid, "trading_day"].astype(str),
                "value": value.loc[valid].to_numpy(dtype=float),
            }
        )
        .groupby("day", sort=True)["value"]
        .mean()
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
    idx = rng.integers(
        0,
        len(daily),
        size=(BOOTSTRAP_SAMPLES, len(daily)),
    )
    draws = daily[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(daily)),
        "samples": BOOTSTRAP_SAMPLES,
        "mean_pct": float(daily.mean()),
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(calibration_positions_path)
    fit = attach_wait_target(fit_positions)
    calibration = attach_wait_target(calibration_positions)
    model = train_wait_model(fit, calibration)

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(position_paths) != len(FRESH_DAYS):
        raise ValueError("request 186 requires complete request 178 positions")
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh = attach_wait_target(fresh_positions)
    found = sorted(fresh["trading_day"].astype(str).unique())
    if found != list(FRESH_DAYS):
        raise ValueError(f"request 186 expected {FRESH_DAYS}, found {found}")

    total_episodes = int(
        fresh_positions.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    wait_coverage = (
        float(len(fresh) / total_episodes) if total_episodes else None
    )
    rich_coverage = float(
        fresh.loc[:, list(RICH_SECOND_FEATURES)]
        .notna()
        .any(axis=1)
        .mean()
    ) if len(fresh) else None

    fresh["predicted_wait_state_3m_base_pct"] = predict_wait(
        fresh, model
    )
    actual = pd.to_numeric(
        fresh["wait_3m_base_pct"], errors="coerce"
    )
    predicted = pd.to_numeric(
        fresh["predicted_wait_state_3m_base_pct"], errors="coerce"
    )
    fresh["action"] = np.where(predicted.gt(0), "ENTER", "ABSTAIN")

    valid = actual.notna() & predicted.notna()
    selected = fresh.loc[valid & predicted.gt(0)].copy()
    selected_values = pd.to_numeric(
        selected["wait_3m_base_pct"], errors="coerce"
    )
    selected_rate = (
        float(len(selected) / int(valid.sum()))
        if int(valid.sum())
        else None
    )
    selected_mean = (
        float(selected_values.mean()) if len(selected_values) else None
    )
    selected_day = _day_balanced(selected, "wait_3m_base_pct")
    selected_positive = (
        float(selected_values.gt(0).mean())
        if len(selected_values)
        else None
    )

    all_values = actual.loc[valid]
    all_mean = float(all_values.mean()) if len(all_values) else None
    all_day = _day_balanced(fresh.loc[valid], "wait_3m_base_pct")
    uplift = (
        float(selected_mean - all_mean)
        if selected_mean is not None and all_mean is not None
        else None
    )
    spearman = _safe_spearman(actual, predicted)
    improvement = (
        float(spearman - REQUEST185_WAIT_SPEARMAN)
        if spearman is not None
        else None
    )
    bootstrap = _bootstrap(selected)

    current = pd.to_numeric(fresh["current_price"], errors="coerce")
    fresh["price_bin"] = pd.cut(
        current,
        bins=(-np.inf, 1.0, 2.0, 5.0, 10.0, np.inf),
        labels=("<1", "1-2", "2-5", "5-10", ">=10"),
        right=False,
    ).astype(str)

    positive_days = 0
    by_day: dict[str, object] = {}
    for day in FRESH_DAYS:
        part = selected.loc[
            selected["trading_day"].astype(str).eq(day)
        ]
        values = pd.to_numeric(
            part["wait_3m_base_pct"], errors="coerce"
        )
        mean = float(values.mean()) if len(values) else None
        good = bool(mean is not None and mean > 0)
        positive_days += int(good)
        by_day[day] = {
            "selected_rows": int(len(part)),
            "selected_mean_pct": mean,
            "selected_positive_rate": (
                float(values.gt(0).mean()) if len(values) else None
            ),
            "positive_selected_day": good,
        }

    by_price: dict[str, object] = {}
    for key, group in fresh.groupby("price_bin", sort=True, observed=True):
        part = group.loc[
            pd.to_numeric(
                group["predicted_wait_state_3m_base_pct"],
                errors="coerce",
            ).gt(0)
        ]
        values = pd.to_numeric(
            part["wait_3m_base_pct"], errors="coerce"
        )
        by_price[str(key)] = {
            "rows": int(len(group)),
            "selected_rows": int(len(part)),
            "selected_rate": (
                float(len(part) / len(group)) if len(group) else None
            ),
            "selected_mean_pct": (
                float(values.mean()) if values.notna().any() else None
            ),
        }

    gate = bool(
        wait_coverage is not None
        and wait_coverage >= MIN_WAIT_STATE_COVERAGE
        and rich_coverage is not None
        and rich_coverage >= MIN_RICH_SECOND_COVERAGE
        and spearman is not None
        and spearman >= MIN_SPEARMAN
        and improvement is not None
        and improvement >= MIN_SPEARMAN_IMPROVEMENT
        and selected_rate is not None
        and MIN_SELECTED_RATE <= selected_rate <= MAX_SELECTED_RATE
        and selected_day is not None
        and selected_day > 0
        and selected_positive is not None
        and selected_positive >= MIN_SELECTED_POSITIVE_RATE
        and uplift is not None
        and uplift >= MIN_UPLIFT_PCT
        and bootstrap["ci_low_pct"] is not None
        and float(bootstrap["ci_low_pct"]) > 0
        and positive_days >= MIN_POSITIVE_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "policy_semantics": (
            "WAIT at original HOT, re-observe minute-1 causal market state, "
            "then ENTER iff predicted 3m BASE opportunity > 0"
        ),
        "target_semantics": (
            "hindsight best BASE exit within 3 minutes after delayed entry; "
            "observability only, not executable exit"
        ),
        "wait_state_coverage": wait_coverage,
        "rich_second_coverage": rich_coverage,
        "feature_count": len(model.columns),
        "fresh_spearman": spearman,
        "request185_original_state_wait_spearman": REQUEST185_WAIT_SPEARMAN,
        "spearman_improvement": improvement,
        "selected_rows": int(len(selected)),
        "selected_rate": selected_rate,
        "selected_opportunity_mean_pct": selected_mean,
        "selected_day_balanced_opportunity_pct": selected_day,
        "selected_positive_opportunity_rate": selected_positive,
        "all_wait_opportunity_mean_pct": all_mean,
        "all_wait_day_balanced_opportunity_pct": all_day,
        "selected_minus_all_wait_uplift_pct": uplift,
        "selected_day_bootstrap": bootstrap,
        "positive_selected_days": int(positive_days),
        "by_day": by_day,
        "by_price_bin": by_price,
        "model_diagnostics": {
            "offset_pct": model.offset,
            "winsor_low_pct": model.winsor_low,
            "winsor_high_pct": model.winsor_high,
        },
        "frozen_gate": {
            "min_wait_state_coverage": MIN_WAIT_STATE_COVERAGE,
            "min_rich_second_coverage": MIN_RICH_SECOND_COVERAGE,
            "min_spearman": MIN_SPEARMAN,
            "min_spearman_improvement": MIN_SPEARMAN_IMPROVEMENT,
            "min_selected_rate": MIN_SELECTED_RATE,
            "max_selected_rate": MAX_SELECTED_RATE,
            "min_selected_positive_rate": MIN_SELECTED_POSITIVE_RATE,
            "min_uplift_pct": MIN_UPLIFT_PCT,
            "bootstrap_low_must_be_positive": True,
            "min_positive_days": MIN_POSITIVE_DAYS,
        },
        "development_bridge_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fresh.to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
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
