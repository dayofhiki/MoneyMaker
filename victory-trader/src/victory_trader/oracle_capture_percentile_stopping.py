"""Request 189: oracle-capture percentile stopping.

Learns a bounded within-episode exit-quality target on historical causal
POSITION states, selects a fixed percentile threshold only on chronological
calibration, then evaluates a recurrent fresh stopping trajectory.
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
from .rich_second_position_value import RICH_SECOND_FEATURES
from .selected_hot_position_value_observability import MODEL_FEATURES

REQUEST_ID = 189
MODEL_SEED = 20261091
MAX_HOLD_MINUTES = 30
THRESHOLDS = (0.70, 0.80, 0.90)
BOOTSTRAP_SEED = 20261089
BOOTSTRAP_SAMPLES = 10_000
SEVERE_LOSS_PCT = -5.0
MIN_START_COVERAGE = 0.85
MIN_COMPLETION_COVERAGE = 0.90
MIN_FRESH_SPEARMAN = 0.10

FEATURES = tuple(dict.fromkeys([*MODEL_FEATURES, *RICH_SECOND_FEATURES]))


@dataclass(frozen=True)
class CaptureModel:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    day = frame["trading_day"].astype(str)
    counts = day.map(day.value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(dtype=float)
    mean = float(np.mean(weights))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("request 189 invalid equal-day weights")
    return weights / mean


def add_capture_target(frame: pd.DataFrame) -> pd.DataFrame:
    result_parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(EPISODE_KEYS, sort=False):
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

        exit_values: list[float] = []
        current_by_index: dict[int, float] = {}
        for idx, row in work.iterrows():
            current = pd.to_numeric(
                pd.Series([row.get("exit_now_base_return_pct")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(current):
                value = float(current)
                exit_values.append(value)
                current_by_index[int(idx)] = value

            if int(row["_minute"]) == 29:
                minute30 = pd.to_numeric(
                    pd.Series([row.get("next_minute_base_return_pct")]),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(minute30):
                    exit_values.append(float(minute30))

        if not exit_values:
            work["capture_percentile"] = np.nan
            result_parts.append(work)
            continue

        universe = np.asarray(exit_values, dtype=float)
        targets: dict[int, float] = {}
        for idx, current in current_by_index.items():
            targets[idx] = float(np.mean(universe <= current))
        work["capture_percentile"] = pd.Series(targets)
        result_parts.append(work)

    if not result_parts:
        return pd.DataFrame()
    return pd.concat(result_parts, ignore_index=True)


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric, errors="coerce"
    )


def train_capture_model(fit: pd.DataFrame) -> CaptureModel:
    target = pd.to_numeric(
        fit["capture_percentile"], errors="coerce"
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError(
            f"request 189 needs >=1000 fit targets, got {len(train)}"
        )

    columns = tuple(
        column
        for column in FEATURES
        if column in train.columns
        and pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    if not columns:
        raise ValueError("request 189 has no usable causal features")

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
        y,
        sample_weight=_day_weights(train),
    )
    return CaptureModel(model=model, columns=columns)


def predict_capture(
    frame: pd.DataFrame,
    fitted: CaptureModel,
) -> np.ndarray:
    raw = fitted.model.predict(
        _feature_frame(frame, fitted.columns)
    )
    return np.clip(raw, 0.0, 1.0)


def _episode_rows(frame: pd.DataFrame) -> dict[int, pd.Series]:
    work = frame.copy()
    work["_minute"] = pd.to_numeric(
        work["minutes_held"], errors="coerce"
    )
    work = work.loc[
        work["_minute"].notna() & work["_minute"].mod(1).eq(0)
    ].copy()
    work["_minute"] = work["_minute"].astype(int)
    return {
        int(row["_minute"]): row
        for _, row in work.sort_values(
            "_minute", kind="stable"
        ).iterrows()
    }


def build_trajectories(
    scored: pd.DataFrame,
    *,
    policy: str,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    total = int(
        scored.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    records: list[dict[str, object]] = []

    for _, group in scored.groupby(EPISODE_KEYS, sort=False):
        by_minute = _episode_rows(group)
        first = by_minute.get(1)
        if first is None:
            continue
        first_exit = pd.to_numeric(
            pd.Series([first.get("exit_now_base_return_pct")]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(first_exit):
            continue

        if policy == "minute1_exit":
            records.append(
                {
                    "trading_day": str(first["trading_day"]),
                    "ticker": str(first["ticker"]).upper(),
                    "hot_t": int(first["hot_t"]),
                    "policy": policy,
                    "status": "completed",
                    "exit_reason": "minute1_exit",
                    "base_net_return_pct": float(first_exit),
                    "minutes_held": 1.0,
                }
            )
            continue

        minute = 1
        realized = np.nan
        exit_minute = np.nan
        status = "unresolved"
        reason = "unknown"

        while minute < MAX_HOLD_MINUTES:
            row = by_minute.get(minute)
            if row is None:
                status = "unresolved"
                reason = "missing_reached_state"
                break

            current_exit = pd.to_numeric(
                pd.Series([row.get("exit_now_base_return_pct")]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(current_exit):
                status = "unresolved"
                reason = "missing_current_exit"
                break

            if policy == "hold30":
                should_exit = False
                score = np.nan
            else:
                score = pd.to_numeric(
                    pd.Series(
                        [row.get("predicted_capture_percentile")]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.isna(score) or not np.isfinite(float(score)):
                    should_exit = True
                    reason = "unscoreable_exit"
                else:
                    if threshold is None:
                        raise ValueError("threshold required")
                    should_exit = float(score) >= float(threshold)
                    if should_exit:
                        reason = "capture_percentile_exit"

            if should_exit:
                status = "completed"
                realized = float(current_exit)
                exit_minute = float(minute)
                break

            if minute == MAX_HOLD_MINUTES - 1:
                next_return = pd.to_numeric(
                    pd.Series(
                        [row.get("next_minute_base_return_pct")]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(next_return):
                    status = "completed"
                    reason = "forced_30m_cap"
                    realized = float(next_return)
                    exit_minute = float(MAX_HOLD_MINUTES)
                else:
                    status = "unresolved"
                    reason = "missing_forced_cap_exit"
                break

            next_row = by_minute.get(minute + 1)
            if next_row is None:
                next_return = pd.to_numeric(
                    pd.Series(
                        [row.get("next_minute_base_return_pct")]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(next_return):
                    status = "completed"
                    reason = "missing_state_next_minute_exit"
                    realized = float(next_return)
                    exit_minute = float(minute + 1)
                else:
                    status = "unresolved"
                    reason = "missing_state_and_exit"
                break

            minute += 1

        records.append(
            {
                "trading_day": str(first["trading_day"]),
                "ticker": str(first["ticker"]).upper(),
                "hot_t": int(first["hot_t"]),
                "policy": policy,
                "status": status,
                "exit_reason": reason,
                "base_net_return_pct": (
                    float(realized) if pd.notna(realized) else np.nan
                ),
                "minutes_held": exit_minute,
            }
        )

    trajectories = pd.DataFrame(records)
    started = int(len(trajectories))
    return trajectories, {
        "total_position_episodes": total,
        "started_episodes": started,
        "start_state_coverage": (
            float(started / total) if total else None
        ),
    }


def _bootstrap_daily(
    frame: pd.DataFrame,
    column: str,
    seed: int,
) -> dict[str, object]:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna()
    daily = (
        pd.DataFrame(
            {
                "day": frame.loc[valid, "trading_day"].astype(str),
                "value": values.loc[valid].to_numpy(dtype=float),
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
    rng = np.random.default_rng(seed)
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


def summarize(
    trajectories: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, object]:
    completed = trajectories.loc[
        trajectories["status"].eq("completed")
    ].copy()
    values = pd.to_numeric(
        completed["base_net_return_pct"], errors="coerce"
    )
    valid = values.notna()
    completed = completed.loc[valid].copy()
    values = values.loc[valid]
    held = pd.to_numeric(
        completed["minutes_held"], errors="coerce"
    )
    bootstrap = _bootstrap_daily(
        completed, "base_net_return_pct", seed
    )
    return {
        "started": int(len(trajectories)),
        "completed": int(len(completed)),
        "completion_coverage": (
            float(len(completed) / len(trajectories))
            if len(trajectories)
            else None
        ),
        "base_mean_pct": float(values.mean()) if len(values) else None,
        "day_balanced_base_mean_pct": bootstrap["mean_pct"],
        "base_median_pct": (
            float(values.median()) if len(values) else None
        ),
        "base_p05_pct": (
            float(values.quantile(0.05)) if len(values) else None
        ),
        "positive_rate": (
            float(values.gt(0).mean()) if len(values) else None
        ),
        "severe_loss_rate": (
            float(values.le(SEVERE_LOSS_PCT).mean())
            if len(values)
            else None
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
        "exit_reasons": {
            str(key): int(value)
            for key, value in trajectories["exit_reason"]
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "day_bootstrap": bootstrap,
    }


def matched_difference(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    seed: int,
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
    bootstrap = _bootstrap_daily(
        merged, "difference_pct", seed
    )
    return {
        "matched_episodes": int(len(merged)),
        "mean_difference_pct": (
            float(merged["difference_pct"].mean())
            if len(merged)
            else None
        ),
        "day_balanced_difference_pct": bootstrap["mean_pct"],
        "bootstrap": bootstrap,
    }


def _safe_spearman(
    actual: pd.Series,
    predicted: pd.Series,
) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna()
    if int(valid.sum()) < 20:
        return None
    value = a.loc[valid].corr(p.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def choose_threshold(
    calibration: pd.DataFrame,
) -> tuple[float, dict[str, object]]:
    table: dict[str, object] = {}
    candidates: list[tuple[float, float]] = []
    for index, threshold in enumerate(THRESHOLDS):
        trajectory, coverage = build_trajectories(
            calibration,
            policy="capture_percentile",
            threshold=threshold,
        )
        summary = summarize(
            trajectory,
            seed=BOOTSTRAP_SEED + 100 + index,
        )
        table[f"{threshold:.2f}"] = {
            "threshold": threshold,
            "coverage": coverage,
            "summary": summary,
        }
        completion = summary["completion_coverage"]
        day_mean = summary["day_balanced_base_mean_pct"]
        if (
            completion is not None
            and float(completion) >= MIN_COMPLETION_COVERAGE
            and day_mean is not None
        ):
            candidates.append((float(day_mean), threshold))

    if not candidates:
        raise ValueError(
            "request 189 no calibration threshold reaches completion floor"
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return float(candidates[0][1]), table


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    fresh_dir: Path,
    output_path: Path,
    trajectories_output_path: Path,
) -> int:
    fit = add_capture_target(pd.read_parquet(fit_path))
    calibration = add_capture_target(
        pd.read_parquet(calibration_path)
    )
    model = train_capture_model(fit)

    calibration["predicted_capture_percentile"] = predict_capture(
        calibration, model
    )
    threshold, calibration_table = choose_threshold(calibration)

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(position_paths) != len(FRESH_DAYS):
        raise ValueError("request 189 requires complete request 178 positions")
    fresh = add_capture_target(
        pd.concat(
            [pd.read_parquet(path) for path in position_paths],
            ignore_index=True,
        )
    )
    found = sorted(fresh["trading_day"].astype(str).unique())
    if found != list(FRESH_DAYS):
        raise ValueError(f"request 189 expected {FRESH_DAYS}, found {found}")
    fresh["predicted_capture_percentile"] = predict_capture(
        fresh, model
    )

    candidate, coverage = build_trajectories(
        fresh,
        policy="capture_percentile",
        threshold=threshold,
    )
    minute1, minute1_coverage = build_trajectories(
        fresh,
        policy="minute1_exit",
    )
    hold30, hold30_coverage = build_trajectories(
        fresh,
        policy="hold30",
    )

    candidate_summary = summarize(
        candidate, seed=BOOTSTRAP_SEED
    )
    minute1_summary = summarize(
        minute1, seed=BOOTSTRAP_SEED + 1
    )
    hold30_summary = summarize(
        hold30, seed=BOOTSTRAP_SEED + 2
    )
    diff_minute1 = matched_difference(
        candidate, minute1, seed=BOOTSTRAP_SEED + 3
    )
    diff_hold30 = matched_difference(
        candidate, hold30, seed=BOOTSTRAP_SEED + 4
    )
    spearman = _safe_spearman(
        fresh["capture_percentile"],
        fresh["predicted_capture_percentile"],
    )

    c_day = candidate_summary["day_balanced_base_mean_pct"]
    c_low = candidate_summary["day_bootstrap"]["ci_low_pct"]
    d_day = diff_minute1["day_balanced_difference_pct"]
    d_low = diff_minute1["bootstrap"]["ci_low_pct"]
    c_severe = candidate_summary["severe_loss_rate"]
    b_severe = minute1_summary["severe_loss_rate"]

    gate = bool(
        coverage["start_state_coverage"] is not None
        and float(coverage["start_state_coverage"]) >= MIN_START_COVERAGE
        and candidate_summary["completion_coverage"] is not None
        and float(candidate_summary["completion_coverage"])
        >= MIN_COMPLETION_COVERAGE
        and spearman is not None
        and float(spearman) >= MIN_FRESH_SPEARMAN
        and c_day is not None
        and float(c_day) > 0
        and c_low is not None
        and float(c_low) > 0
        and d_day is not None
        and float(d_day) > 0
        and d_low is not None
        and float(d_low) > 0
        and c_severe is not None
        and b_severe is not None
        and float(c_severe) <= float(b_severe)
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "target_semantics": (
            "within-episode percentile rank of current exact BASE exit "
            "among all executable exits through minute30"
        ),
        "feature_count": len(model.columns),
        "calibration_threshold_candidates": list(THRESHOLDS),
        "calibration_threshold_table": calibration_table,
        "selected_threshold": threshold,
        "fresh_capture_percentile_spearman": spearman,
        "candidate_coverage": coverage,
        "minute1_coverage": minute1_coverage,
        "hold30_coverage": hold30_coverage,
        "candidate": candidate_summary,
        "minute1_comparator": minute1_summary,
        "hold30_comparator": hold30_summary,
        "matched_candidate_minus_minute1": diff_minute1,
        "matched_candidate_minus_hold30": diff_hold30,
        "frozen_gate": {
            "min_start_coverage": MIN_START_COVERAGE,
            "min_completion_coverage": MIN_COMPLETION_COVERAGE,
            "min_fresh_capture_spearman": MIN_FRESH_SPEARMAN,
            "candidate_day_balanced_base_must_be_positive": True,
            "candidate_bootstrap_low_must_be_positive": True,
            "matched_minute1_difference_must_be_positive": True,
            "matched_minute1_bootstrap_low_must_be_positive": True,
            "severe_loss_no_worse_than_minute1": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.concat(
        [candidate, minute1, hold30],
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
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectories-output", type=Path, required=True
    )
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.fresh_dir,
        args.output,
        args.trajectories_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
