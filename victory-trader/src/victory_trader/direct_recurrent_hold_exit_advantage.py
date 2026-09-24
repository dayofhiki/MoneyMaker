"""Request 179: direct recurrent HOLD/EXIT expected-advantage diagnostic.

No new market dates are opened. The already-opened Request-178 fresh block is
reused only as a development diagnostic and is never promotion-eligible.

The candidate directly predicts the BASE economic advantage of HOLDing exactly
one more minute versus EXITing now, then composes that semantic zero boundary
recurrently along actually reached POSITION states.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .attention_replay import MINUTE_MS
from .fresh_consensus_hurdle_value import FRESH_DAYS
from .market_regime_position_value import (
    MARKET_REGIME_FEATURES,
    attach_market_regime,
    build_market_regime,
)
from .rich_second_position_value import RICH_SECOND_FEATURES
from .selected_hot_position_value_observability import (
    MODEL_FEATURES,
    predict_hold,
    train_hold_model,
)

REQUEST_ID = 179
MODEL_SEED = 20261079
BOOTSTRAP_SEED = 20261079
BOOTSTRAP_SAMPLES = 10_000
MAX_HOLD_MINUTES = 30
SEVERE_LOSS_PCT = -5.0

MIN_START_COVERAGE = 0.80
MIN_ADVANTAGE_SPEARMAN = 0.05
MIN_GOOD_DAYS = 3

FEATURES = tuple(
    dict.fromkeys(
        [
            *MODEL_FEATURES,
            *RICH_SECOND_FEATURES,
            *MARKET_REGIME_FEATURES,
        ]
    )
)
EPISODE_KEYS = ["trading_day", "ticker", "hot_t"]


@dataclass(frozen=True)
class AdvantageModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    day = frame["trading_day"].astype(str)
    counts = day.map(day.value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(dtype=float)
    mean = float(np.mean(weights))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("invalid equal-day weights")
    return weights / mean


def train_advantage_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> AdvantageModel:
    target = pd.to_numeric(
        fit["hold_advantage_1m_pct"],
        errors="coerce",
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError("request 179 insufficient fit action-value support")

    columns = tuple(
        column
        for column in FEATURES
        if column in train.columns
        and pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    if not columns:
        raise ValueError("request 179 has no usable features")

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
        calibration["hold_advantage_1m_pct"],
        errors="coerce",
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 500:
        raise ValueError(
            "request 179 insufficient calibration action-value support"
        )
    raw = model.predict(_feature_frame(cal, columns))
    residual = cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    offset = float(
        np.average(
            residual,
            weights=_day_weights(cal),
        )
    )
    return AdvantageModel(
        model=model,
        feature_columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_advantage(
    frame: pd.DataFrame,
    fitted: AdvantageModel,
) -> np.ndarray:
    return (
        fitted.model.predict(
            _feature_frame(frame, fitted.feature_columns)
        )
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


def _day_bootstrap_values(
    daily: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | int | None]:
    values = np.asarray(daily, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {
            "days": int(len(values)),
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": float(np.mean(values)) if len(values) else None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(seed)
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


def advantage_diagnostics(
    scored: pd.DataFrame,
) -> dict[str, object]:
    actual = pd.to_numeric(
        scored["hold_advantage_1m_pct"],
        errors="coerce",
    )
    predicted = pd.to_numeric(
        scored["predicted_hold_advantage_1m_pct"],
        errors="coerce",
    )
    valid = actual.notna() & predicted.notna()
    selected = scored.loc[valid & predicted.gt(0)].copy()
    selected_actual = pd.to_numeric(
        selected["hold_advantage_1m_pct"],
        errors="coerce",
    )

    selected_daily = (
        pd.DataFrame(
            {
                "day": selected["trading_day"].astype(str),
                "value": selected_actual,
            }
        )
        .dropna()
        .groupby("day", sort=True)["value"]
        .mean()
    )

    by_day: dict[str, object] = {}
    good_days = 0
    for day in FRESH_DAYS:
        part = scored.loc[
            scored["trading_day"].astype(str).eq(day)
        ].copy()
        y = pd.to_numeric(
            part["hold_advantage_1m_pct"],
            errors="coerce",
        )
        p = pd.to_numeric(
            part["predicted_hold_advantage_1m_pct"],
            errors="coerce",
        )
        day_valid = y.notna() & p.notna()
        corr = _safe_spearman(y.loc[day_valid], p.loc[day_valid])
        hold = part.loc[day_valid & p.gt(0)]
        hold_mean = (
            float(
                pd.to_numeric(
                    hold["hold_advantage_1m_pct"],
                    errors="coerce",
                ).mean()
            )
            if len(hold)
            else None
        )
        good = bool(
            corr is not None
            and corr > 0
            and hold_mean is not None
            and hold_mean > 0
        )
        good_days += int(good)
        by_day[day] = {
            "rows": int(len(part)),
            "evaluable_rows": int(day_valid.sum()),
            "spearman": corr,
            "predicted_hold_rows": int(len(hold)),
            "predicted_hold_rate": (
                float(len(hold) / int(day_valid.sum()))
                if int(day_valid.sum())
                else None
            ),
            "selected_realized_advantage_mean_pct": hold_mean,
            "both_positive": good,
        }

    return {
        "rows": int(len(scored)),
        "evaluable_rows": int(valid.sum()),
        "advantage_spearman": _safe_spearman(
            actual.loc[valid],
            predicted.loc[valid],
        ),
        "predicted_hold_rows": int(len(selected)),
        "predicted_hold_rate": (
            float(len(selected) / int(valid.sum()))
            if int(valid.sum())
            else None
        ),
        "selected_realized_advantage_mean_pct": (
            float(selected_actual.mean()) if len(selected) else None
        ),
        "selected_day_balanced_advantage_mean_pct": (
            float(selected_daily.mean()) if len(selected_daily) else None
        ),
        "good_days": int(good_days),
        "by_day": by_day,
    }


def _episode_key(row: pd.Series) -> tuple[str, str, int]:
    return (
        str(row["trading_day"]),
        str(row["ticker"]).upper(),
        int(row["hot_t"]),
    )


def build_trajectories(
    scored: pd.DataFrame,
    *,
    score_column: str,
    threshold: float,
    policy: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    total_episodes = int(
        scored.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    rows: list[dict[str, object]] = []

    for _, group in scored.groupby(EPISODE_KEYS, sort=False):
        work = group.copy()
        work["_minute"] = pd.to_numeric(
            work["minutes_held"],
            errors="coerce",
        )
        work = work.loc[
            work["_minute"].notna()
            & work["_minute"].mod(1).eq(0)
        ].copy()
        if work.empty:
            continue
        work["_minute"] = work["_minute"].astype(int)
        by_minute = {
            int(row["_minute"]): row
            for _, row in work.sort_values(
                "_minute", kind="stable"
            ).iterrows()
        }
        first = by_minute.get(1)
        if first is None:
            continue
        first_exit = pd.to_numeric(
            pd.Series([first.get("exit_now_base_return_pct")]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(first_exit):
            continue

        minute = 1
        hold_decisions = 0
        status = "unresolved"
        exit_reason = "unknown"
        realized = np.nan
        exit_t = np.nan

        while minute < MAX_HOLD_MINUTES:
            row = by_minute.get(minute)
            if row is None:
                status = "unresolved"
                exit_reason = "missing_reached_state"
                break

            current_exit = pd.to_numeric(
                pd.Series([row.get("exit_now_base_return_pct")]),
                errors="coerce",
            ).iloc[0]
            score = pd.to_numeric(
                pd.Series([row.get(score_column)]),
                errors="coerce",
            ).iloc[0]

            if pd.isna(current_exit):
                status = "unresolved"
                exit_reason = "missing_current_exit"
                break

            if pd.isna(score) or not np.isfinite(float(score)):
                status = "completed"
                exit_reason = "unscoreable_exit"
                realized = float(current_exit)
                exit_t = int(row["state_t"])
                break

            if float(score) <= threshold:
                status = "completed"
                exit_reason = "model_exit"
                realized = float(current_exit)
                exit_t = int(row["state_t"])
                break

            hold_decisions += 1
            next_return = pd.to_numeric(
                pd.Series([row.get("next_minute_base_return_pct")]),
                errors="coerce",
            ).iloc[0]

            if minute == MAX_HOLD_MINUTES - 1:
                if pd.notna(next_return):
                    status = "completed"
                    exit_reason = "forced_30m_cap"
                    realized = float(next_return)
                    exit_t = int(row["state_t"]) + MINUTE_MS
                else:
                    status = "unresolved"
                    exit_reason = "missing_forced_cap_exit"
                break

            next_row = by_minute.get(minute + 1)
            if next_row is None:
                if pd.notna(next_return):
                    status = "completed"
                    exit_reason = "missing_state_next_minute_exit"
                    realized = float(next_return)
                    exit_t = int(row["state_t"]) + MINUTE_MS
                else:
                    status = "unresolved"
                    exit_reason = "missing_state_and_exit"
                break
            minute += 1

        rows.append(
            {
                "trading_day": str(first["trading_day"]),
                "ticker": str(first["ticker"]).upper(),
                "hot_t": int(first["hot_t"]),
                "policy": policy,
                "status": status,
                "exit_reason": exit_reason,
                "base_net_return_pct": (
                    float(realized) if pd.notna(realized) else np.nan
                ),
                "minutes_held": (
                    float((int(exit_t) - int(first["entry_actual_t"])) / MINUTE_MS)
                    if pd.notna(exit_t)
                    else np.nan
                ),
                "hold_decisions": int(hold_decisions),
                "entry_actual_t": int(first["entry_actual_t"]),
                "exit_t": int(exit_t) if pd.notna(exit_t) else np.nan,
            }
        )

    trajectories = pd.DataFrame(rows)
    started = (
        int(trajectories.loc[:, EPISODE_KEYS].drop_duplicates().shape[0])
        if len(trajectories)
        else 0
    )
    return trajectories, {
        "total_position_episodes": total_episodes,
        "started_episodes": started,
        "start_state_coverage": (
            float(started / total_episodes)
            if total_episodes
            else None
        ),
    }


def trajectory_metrics(
    trajectories: pd.DataFrame,
    *,
    policy: str,
) -> dict[str, object]:
    if trajectories.empty:
        return {
            "policy": policy,
            "started": 0,
            "completed": 0,
        }
    completed = trajectories.loc[
        trajectories["status"].eq("completed")
    ].copy()
    values = pd.to_numeric(
        completed["base_net_return_pct"],
        errors="coerce",
    )
    valid = values.notna()
    completed = completed.loc[valid].copy()
    values = values.loc[valid]

    daily = (
        pd.DataFrame(
            {
                "day": completed["trading_day"].astype(str),
                "value": values,
            }
        )
        .groupby("day", sort=True)["value"]
        .mean()
    )
    held = pd.to_numeric(
        completed["minutes_held"],
        errors="coerce",
    )
    reasons = {
        str(key): int(value)
        for key, value in trajectories["exit_reason"]
        .value_counts(dropna=False)
        .to_dict()
        .items()
    }
    bootstrap = _day_bootstrap_values(
        daily.to_numpy(dtype=float),
        seed=BOOTSTRAP_SEED,
    )

    return {
        "policy": policy,
        "started": int(len(trajectories)),
        "completed": int(len(completed)),
        "completion_coverage": (
            float(len(completed) / len(trajectories))
            if len(trajectories)
            else None
        ),
        "days": int(completed["trading_day"].nunique()),
        "base_mean_pct": float(values.mean()) if len(values) else None,
        "base_median_pct": float(values.median()) if len(values) else None,
        "base_p05_pct": (
            float(values.quantile(0.05)) if len(values) else None
        ),
        "base_positive_rate": (
            float(values.gt(0).mean()) if len(values) else None
        ),
        "severe_loss_rate": (
            float(values.le(SEVERE_LOSS_PCT).mean())
            if len(values)
            else None
        ),
        "day_balanced_base_mean_pct": (
            float(daily.mean()) if len(daily) else None
        ),
        "mean_hold_minutes": (
            float(held.mean()) if held.notna().any() else None
        ),
        "median_hold_minutes": (
            float(held.median()) if held.notna().any() else None
        ),
        "p90_hold_minutes": (
            float(held.quantile(0.90)) if held.notna().any() else None
        ),
        "exit_reasons": reasons,
        "day_bootstrap": bootstrap,
    }


def matched_difference(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
) -> dict[str, object]:
    left = candidate.loc[
        candidate["status"].eq("completed"),
        EPISODE_KEYS + ["base_net_return_pct"],
    ].copy()
    right = comparator.loc[
        comparator["status"].eq("completed"),
        EPISODE_KEYS + ["base_net_return_pct"],
    ].copy()
    merged = left.merge(
        right,
        on=EPISODE_KEYS,
        suffixes=("_candidate", "_comparator"),
        validate="one_to_one",
    )
    if merged.empty:
        return {
            "matched_episodes": 0,
            "day_balanced_difference_pct": None,
            "bootstrap": _day_bootstrap_values(
                np.asarray([], dtype=float),
                seed=BOOTSTRAP_SEED + 1,
            ),
        }

    merged["difference_pct"] = (
        pd.to_numeric(
            merged["base_net_return_pct_candidate"],
            errors="coerce",
        )
        - pd.to_numeric(
            merged["base_net_return_pct_comparator"],
            errors="coerce",
        )
    )
    daily = (
        merged.groupby(
            merged["trading_day"].astype(str),
            sort=True,
        )["difference_pct"]
        .mean()
        .dropna()
    )
    return {
        "matched_episodes": int(len(merged)),
        "mean_difference_pct": float(merged["difference_pct"].mean()),
        "candidate_better_rate": float(
            merged["difference_pct"].gt(0).mean()
        ),
        "day_balanced_difference_pct": (
            float(daily.mean()) if len(daily) else None
        ),
        "bootstrap": _day_bootstrap_values(
            daily.to_numpy(dtype=float),
            seed=BOOTSTRAP_SEED + 1,
        ),
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    trajectories_output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    history_scan = pd.read_parquet(history_scan_path)

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError("request 179 requires complete request 178 shards")

    fresh = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    found_days = sorted(fresh["trading_day"].astype(str).unique())
    if found_days != list(FRESH_DAYS):
        raise ValueError(
            f"request 179 expected {FRESH_DAYS}, found {found_days}"
        )

    historical_days = set(
        fit["trading_day"].astype(str)
    ) | set(calibration["trading_day"].astype(str))
    historical_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(historical_days)
    ].copy()

    historical_regime = build_market_regime(historical_scan)
    fresh_regime = build_market_regime(fresh_scan)
    fit = attach_market_regime(fit, historical_regime)
    calibration = attach_market_regime(
        calibration,
        historical_regime,
    )
    fresh = attach_market_regime(fresh, fresh_regime)

    candidate_model = train_advantage_model(fit, calibration)
    comparator_model = train_hold_model(fit, calibration)

    scored = fresh.copy()
    scored["predicted_hold_advantage_1m_pct"] = predict_advantage(
        scored,
        candidate_model,
    )
    scored["baseline_hold_probability"] = predict_hold(
        scored,
        comparator_model,
    )

    diagnostics = advantage_diagnostics(scored)
    candidate, coverage = build_trajectories(
        scored,
        score_column="predicted_hold_advantage_1m_pct",
        threshold=0.0,
        policy="direct_expected_advantage",
    )
    comparator, comparator_coverage = build_trajectories(
        scored,
        score_column="baseline_hold_probability",
        threshold=0.5,
        policy="request142_hold_classifier",
    )

    candidate_metrics = trajectory_metrics(
        candidate,
        policy="direct_expected_advantage",
    )
    comparator_metrics = trajectory_metrics(
        comparator,
        policy="request142_hold_classifier",
    )
    difference = matched_difference(candidate, comparator)

    spearman = diagnostics["advantage_spearman"]
    selected_mean = diagnostics[
        "selected_realized_advantage_mean_pct"
    ]
    candidate_day = candidate_metrics.get(
        "day_balanced_base_mean_pct"
    )
    comparator_day = comparator_metrics.get(
        "day_balanced_base_mean_pct"
    )
    diff_day = difference.get("day_balanced_difference_pct")
    diff_low = difference["bootstrap"].get("ci_low_pct")

    development_gate = bool(
        coverage["start_state_coverage"] is not None
        and float(coverage["start_state_coverage"]) >= MIN_START_COVERAGE
        and spearman is not None
        and float(spearman) >= MIN_ADVANTAGE_SPEARMAN
        and selected_mean is not None
        and float(selected_mean) > 0
        and int(diagnostics["good_days"]) >= MIN_GOOD_DAYS
        and candidate_day is not None
        and float(candidate_day) > 0
        and comparator_day is not None
        and float(candidate_day) > float(comparator_day)
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "entry_policy_changed": False,
        "action_rule": (
            "HOLD iff predicted expected one-minute BASE advantage > 0"
        ),
        "feature_count": len(candidate_model.feature_columns),
        "model_diagnostics": {
            "equal_day_weighted_fit": True,
            "calibration": "equal-day-weighted offset only",
            "offset_pct": candidate_model.offset,
            "winsor_low_pct": candidate_model.winsor_low,
            "winsor_high_pct": candidate_model.winsor_high,
        },
        "advantage_diagnostics": diagnostics,
        "candidate_coverage": coverage,
        "comparator_coverage": comparator_coverage,
        "candidate": candidate_metrics,
        "comparator": comparator_metrics,
        "matched_candidate_minus_comparator": difference,
        "frozen_development_gate": {
            "min_start_coverage": MIN_START_COVERAGE,
            "min_advantage_spearman": MIN_ADVANTAGE_SPEARMAN,
            "min_good_days": MIN_GOOD_DAYS,
            "selected_advantage_mean_must_be_positive": True,
            "candidate_day_balanced_base_must_be_positive": True,
            "candidate_must_beat_comparator": True,
            "matched_bootstrap_ci_low_must_be_positive": True,
        },
        "development_gate_pass": development_gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.concat(
        [candidate, comparator],
        ignore_index=True,
    ).to_parquet(
        trajectories_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--history-scan", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectories-output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.trajectories_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
