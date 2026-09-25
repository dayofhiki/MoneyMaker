"""Request 211: transition-aware receding 3-minute HOLD/EXIT value.

Development-only diagnostic on the already-opened Request-178 fresh block.
At every reached POSITION minute, predict the BASE economic advantage of
holding three more minutes versus exiting now. The controller still re-evaluates
every minute, so this is a receding horizon rather than a fixed three-minute hold.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .direct_recurrent_hold_exit_advantage import (
    BOOTSTRAP_SEED,
    EPISODE_KEYS,
    MIN_ADVANTAGE_SPEARMAN,
    MIN_GOOD_DAYS,
    MIN_START_COVERAGE,
    build_trajectories,
    matched_difference,
    trajectory_metrics,
)
from .fresh_consensus_hurdle_value import FRESH_DAYS
from .market_regime_position_value import (
    attach_market_regime,
    build_market_regime,
)
from .transition_recurrent_hold_exit import (
    TRANSITION_FEATURES,
    _day_weights,
    _feature_frame,
    attach_position_transitions,
    predict_transition_advantage,
    train_transition_advantage_model,
)

REQUEST_ID = 211
MODEL_SEED = 20261117
HORIZON_MINUTES = 3
TARGET_COLUMN = "hold_advantage_3m_pct"


@dataclass(frozen=True)
class HorizonAdvantageModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def attach_horizon_advantage(
    frame: pd.DataFrame,
    *,
    horizon_minutes: int = HORIZON_MINUTES,
) -> pd.DataFrame:
    result = frame.copy()
    result[TARGET_COLUMN] = np.nan

    for _, group in result.groupby(EPISODE_KEYS, sort=False):
        work = group.copy()
        work["_minute"] = pd.to_numeric(
            work["minutes_held"], errors="coerce"
        )
        work = work.loc[
            work["_minute"].notna() & work["_minute"].mod(1).eq(0)
        ].copy()
        if work.empty:
            continue
        work["_minute"] = work["_minute"].astype(int)
        exit_by_minute = {
            int(row["_minute"]): pd.to_numeric(
                pd.Series([row.get("exit_now_base_return_pct")]),
                errors="coerce",
            ).iloc[0]
            for _, row in work.iterrows()
        }
        for index, row in work.iterrows():
            minute = int(row["_minute"])
            current = exit_by_minute.get(minute, np.nan)
            future = exit_by_minute.get(
                minute + int(horizon_minutes), np.nan
            )
            if pd.notna(current) and pd.notna(future):
                result.at[index, TARGET_COLUMN] = (
                    float(future) - float(current)
                )
    return result


def train_horizon_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> HorizonAdvantageModel:
    target = pd.to_numeric(fit[TARGET_COLUMN], errors="coerce")
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError("request 211 insufficient 3m fit support")

    columns = tuple(
        column
        for column in TRANSITION_FEATURES
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
        calibration[TARGET_COLUMN], errors="coerce"
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 500:
        raise ValueError("request 211 insufficient 3m calibration support")
    raw = model.predict(_feature_frame(cal, columns))
    residual = cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    offset = float(np.average(residual, weights=_day_weights(cal)))
    return HorizonAdvantageModel(
        model=model,
        feature_columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_horizon_advantage(
    frame: pd.DataFrame,
    fitted: HorizonAdvantageModel,
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


def horizon_diagnostics(scored: pd.DataFrame) -> dict[str, object]:
    actual = pd.to_numeric(scored[TARGET_COLUMN], errors="coerce")
    predicted = pd.to_numeric(
        scored["predicted_hold_advantage_3m_pct"], errors="coerce"
    )
    valid = actual.notna() & predicted.notna()
    selected = scored.loc[valid & predicted.gt(0)].copy()
    selected_actual = pd.to_numeric(
        selected[TARGET_COLUMN], errors="coerce"
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
        y = pd.to_numeric(part[TARGET_COLUMN], errors="coerce")
        p = pd.to_numeric(
            part["predicted_hold_advantage_3m_pct"], errors="coerce"
        )
        day_valid = y.notna() & p.notna()
        corr = _safe_spearman(y.loc[day_valid], p.loc[day_valid])
        hold = part.loc[day_valid & p.gt(0)]
        hold_mean = (
            float(
                pd.to_numeric(
                    hold[TARGET_COLUMN], errors="coerce"
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
            actual.loc[valid], predicted.loc[valid]
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
        raise ValueError("request 211 requires complete request 178 shards")

    fresh = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )

    historical_days = set(fit["trading_day"].astype(str)) | set(
        calibration["trading_day"].astype(str)
    )
    historical_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(historical_days)
    ].copy()

    historical_regime = build_market_regime(historical_scan)
    fresh_regime = build_market_regime(fresh_scan)
    fit = attach_market_regime(fit, historical_regime)
    calibration = attach_market_regime(calibration, historical_regime)
    fresh = attach_market_regime(fresh, fresh_regime)

    fit = attach_position_transitions(fit)
    calibration = attach_position_transitions(calibration)
    fresh = attach_position_transitions(fresh)

    fit = attach_horizon_advantage(fit)
    calibration = attach_horizon_advantage(calibration)
    fresh = attach_horizon_advantage(fresh)

    candidate_model = train_horizon_model(fit, calibration)
    comparator_model = train_transition_advantage_model(
        fit, calibration
    )

    scored = fresh.copy()
    scored["predicted_hold_advantage_3m_pct"] = (
        predict_horizon_advantage(scored, candidate_model)
    )
    scored["predicted_hold_advantage_1m_pct"] = (
        predict_transition_advantage(scored, comparator_model)
    )

    diagnostics = horizon_diagnostics(scored)
    candidate, coverage = build_trajectories(
        scored,
        score_column="predicted_hold_advantage_3m_pct",
        threshold=0.0,
        policy="transition_receding_3m_advantage",
    )
    comparator, comparator_coverage = build_trajectories(
        scored,
        score_column="predicted_hold_advantage_1m_pct",
        threshold=0.0,
        policy="request210_transition_1m_advantage",
    )

    candidate_metrics = trajectory_metrics(
        candidate, policy="transition_receding_3m_advantage"
    )
    comparator_metrics = trajectory_metrics(
        comparator, policy="request210_transition_1m_advantage"
    )
    difference = matched_difference(candidate, comparator)

    spearman = diagnostics["advantage_spearman"]
    selected_mean = diagnostics[
        "selected_realized_advantage_mean_pct"
    ]
    candidate_day = candidate_metrics.get("day_balanced_base_mean_pct")
    comparator_day = comparator_metrics.get("day_balanced_base_mean_pct")
    diff_day = difference.get("day_balanced_difference_pct")
    diff_low = difference["bootstrap"].get("ci_low_pct")

    gate = bool(
        coverage["start_state_coverage"] is not None
        and float(coverage["start_state_coverage"]) >= MIN_START_COVERAGE
        and spearman is not None
        and float(spearman) >= MIN_ADVANTAGE_SPEARMAN
        and selected_mean is not None
        and float(selected_mean) > 0
        and int(diagnostics["good_days"]) >= MIN_GOOD_DAYS
        and candidate_day is not None
        and comparator_day is not None
        and float(candidate_day) > float(comparator_day)
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
        and float(candidate_day) > 0
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "horizon_minutes": HORIZON_MINUTES,
        "runtime_rule": (
            "each reached minute HOLD iff predicted 3m BASE advantage > 0; "
            "re-evaluate next minute"
        ),
        "representation": (
            "Request210 transition features with direct fixed-3m economic target"
        ),
        "candidate_feature_count": len(candidate_model.feature_columns),
        "horizon_advantage_diagnostics": diagnostics,
        "candidate_coverage": coverage,
        "comparator_coverage": comparator_coverage,
        "candidate": candidate_metrics,
        "comparator_request210": comparator_metrics,
        "matched_candidate_minus_request210": difference,
        "frozen_gate": {
            "min_start_coverage": MIN_START_COVERAGE,
            "min_advantage_spearman": MIN_ADVANTAGE_SPEARMAN,
            "min_good_days": MIN_GOOD_DAYS,
            "selected_advantage_mean_must_be_positive": True,
            "candidate_must_beat_request210": True,
            "matched_bootstrap_low_must_be_positive": True,
            "candidate_day_balanced_base_must_be_positive": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.concat([candidate, comparator], ignore_index=True).to_parquet(
        trajectories_output_path, index=False, compression="zstd"
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
    parser.add_argument("--trajectories-output", type=Path, required=True)
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
