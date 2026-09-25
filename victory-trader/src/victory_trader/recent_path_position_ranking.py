"""Request 217: recent-path ranking representation for recurrent HOLD/EXIT.

Development-only representation diagnostic on the already-opened Request178
dates. Request216 showed that realistic event-time replay did not recover HOLD
value ranking. Request217 therefore changes the representation and objective,
not the action threshold:

* construct causal path features from only the current and previous observed
  POSITION states;
* train a rank-oriented target: within-day percentile rank of realized
  next-executable-event HOLD advantage;
* compare the same model capacity using current-state features only versus
  current-state + recent-path features;
* keep Request216 expected-value ranking as an external comparator;
* do not run a new trading policy or open any new dates.

A pass means recent path contains transferable ordering information. It does not
mean a profitable HOLD/EXIT policy has been validated.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .direct_recurrent_hold_exit_advantage import _safe_spearman
from .event_time_position_replay import (
    EVENT_FEATURES,
    _prepare_entries,
    predict_event_advantage,
    reanchor_event_positions,
    train_event_advantage_model,
)
from .fresh_consensus_hurdle_value import FRESH_DAYS
from .market_regime_position_value import (
    attach_market_regime,
    build_market_regime,
)
from .transition_recurrent_hold_exit import attach_position_transitions

REQUEST_ID = 217
MODEL_SEED = 20261127

PATH_SOURCES = (
    "entry_to_current_close_pct",
    "drawdown_from_peak_pct",
    "recovery_from_trough_pct",
    "attention_score",
    "attention_rank",
    "log_minute_volume",
    "log_minute_transactions",
    "active_seconds_60",
    "sec_last5_return_pct",
    "sec_last10_return_pct",
    "sec_close_vs_vwap_pct",
    "sec_volume_burst_5",
    "sec_transactions_burst_5",
)
PATH_WINDOWS = (3, 5)
PATH_FEATURES = tuple(
    f"path_{source}_{suffix}"
    for source in PATH_SOURCES
    for suffix in (
        "delta1",
        "slope3_per_min",
        "slope5_per_min",
        "std3",
        "std5",
        "range3",
        "range5",
    )
)
CANDIDATE_FEATURES = tuple(dict.fromkeys([*EVENT_FEATURES, *PATH_FEATURES]))

MIN_SPEARMAN = 0.05
MIN_SPEARMAN_GAIN = 0.03
MIN_GOOD_DAYS = 3
MIN_TOP10_POSITIVE_RATE_GAIN = 0.05


@dataclass(frozen=True)
class RankModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric, errors="coerce"
    )


def _rolling_slope(
    values: np.ndarray,
    times: np.ndarray,
) -> float:
    valid = np.isfinite(values) & np.isfinite(times)
    if int(valid.sum()) < 2:
        return np.nan
    x = times[valid].astype(float)
    y = values[valid].astype(float)
    x = x - x[-1]
    denom = float(np.sum((x - x.mean()) ** 2))
    if denom <= 0:
        return np.nan
    return float(
        np.sum((x - x.mean()) * (y - y.mean())) / denom
    )


def attach_recent_path_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach causal path-shape features from current and prior observations."""
    result = frame.copy()
    for column in PATH_FEATURES:
        result[column] = np.nan

    keys = ["trading_day", "ticker", "hot_t"]
    for _, group in result.groupby(keys, sort=False):
        ordered = group.sort_values("state_t", kind="stable")
        indices = list(ordered.index)
        times = (
            pd.to_numeric(ordered["state_t"], errors="coerce")
            .to_numpy(dtype=float)
            / 60_000.0
        )

        for source in PATH_SOURCES:
            values = pd.to_numeric(
                ordered.get(source), errors="coerce"
            ).to_numpy(dtype=float)

            for position, index in enumerate(indices):
                current = values[position]
                if position >= 1 and np.isfinite(current):
                    previous = values[position - 1]
                    if np.isfinite(previous):
                        result.at[
                            index, f"path_{source}_delta1"
                        ] = float(current - previous)

                for window in PATH_WINDOWS:
                    start = max(0, position - window + 1)
                    sample = values[start : position + 1]
                    sample_times = times[start : position + 1]
                    finite = sample[np.isfinite(sample)]
                    if len(finite) >= 2:
                        result.at[
                            index, f"path_{source}_std{window}"
                        ] = float(np.std(finite, ddof=0))
                        result.at[
                            index, f"path_{source}_range{window}"
                        ] = float(np.max(finite) - np.min(finite))
                        slope = _rolling_slope(sample, sample_times)
                        result.at[
                            index,
                            f"path_{source}_slope{window}_per_min",
                        ] = slope

    return result


def attach_rank_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Within-day percentile rank of realized next-event HOLD advantage."""
    result = frame.copy()
    actual = pd.to_numeric(
        result["hold_advantage_event_pct"], errors="coerce"
    )
    result["_actual"] = actual
    result["hold_advantage_day_rank"] = (
        result.loc[actual.notna()]
        .groupby(
            result.loc[actual.notna(), "trading_day"].astype(str),
            sort=False,
        )["_actual"]
        .rank(method="average", pct=True)
    )
    return result.drop(columns=["_actual"])


def train_rank_model(
    fit: pd.DataFrame,
    feature_family: tuple[str, ...],
) -> RankModel:
    target = pd.to_numeric(
        fit["hold_advantage_day_rank"], errors="coerce"
    )
    train = fit.loc[target.notna()].copy()
    y = target.loc[target.notna()].to_numpy(dtype=float)
    if len(train) < 1000:
        raise ValueError("request 217 insufficient rank-training support")

    columns = tuple(
        column
        for column in feature_family
        if column in train.columns
        and pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    if not columns:
        raise ValueError("request 217 has no usable features")

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=MODEL_SEED,
    )
    model.fit(_feature_frame(train, columns), y)
    return RankModel(model=model, feature_columns=columns)


def predict_rank(
    frame: pd.DataFrame,
    fitted: RankModel,
) -> np.ndarray:
    return fitted.model.predict(
        _feature_frame(frame, fitted.feature_columns)
    )


def _top_fraction_by_day(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
) -> dict[str, object]:
    selected: list[pd.DataFrame] = []
    by_day: dict[str, object] = {}

    for day in FRESH_DAYS:
        part = frame.loc[
            frame["trading_day"].astype(str).eq(day)
        ].copy()
        part["_actual"] = pd.to_numeric(
            part["hold_advantage_event_pct"], errors="coerce"
        )
        part["_score"] = pd.to_numeric(
            part[score_column], errors="coerce"
        )
        valid = part.loc[
            part["_actual"].notna() & part["_score"].notna()
        ].sort_values("_score", ascending=False, kind="stable")

        count = (
            max(1, int(np.ceil(len(valid) * fraction)))
            if len(valid)
            else 0
        )
        top = valid.head(count).copy()
        if len(top):
            selected.append(top)
        by_day[day] = {
            "rows": int(len(valid)),
            "selected": int(len(top)),
            "mean_advantage_pct": (
                float(top["_actual"].mean()) if len(top) else None
            ),
            "positive_rate": (
                float(top["_actual"].gt(0).mean()) if len(top) else None
            ),
        }

    if not selected:
        return {
            "rows": 0,
            "mean_advantage_pct": None,
            "day_balanced_mean_advantage_pct": None,
            "positive_rate": None,
            "by_day": by_day,
        }

    combined = pd.concat(selected, ignore_index=True)
    daily = (
        combined.groupby(
            combined["trading_day"].astype(str), sort=True
        )["_actual"]
        .mean()
    )
    return {
        "rows": int(len(combined)),
        "mean_advantage_pct": float(combined["_actual"].mean()),
        "day_balanced_mean_advantage_pct": float(daily.mean()),
        "positive_rate": float(combined["_actual"].gt(0).mean()),
        "by_day": by_day,
    }


def _age_bucket(minutes_held: pd.Series) -> pd.Series:
    held = pd.to_numeric(minutes_held, errors="coerce")
    return pd.cut(
        held,
        bins=[-np.inf, 2.0, 5.0, 10.0, 20.0, np.inf],
        labels=["<=2", "2-5", "5-10", "10-20", ">20"],
        right=True,
    )


def ranking_diagnostics(
    frame: pd.DataFrame,
    score_column: str,
) -> dict[str, object]:
    actual = pd.to_numeric(
        frame["hold_advantage_event_pct"], errors="coerce"
    )
    score = pd.to_numeric(frame[score_column], errors="coerce")
    valid = actual.notna() & score.notna()
    work = frame.loc[valid].copy()
    work["_actual"] = actual.loc[valid]
    work["_score"] = score.loc[valid]

    overall_positive = (
        float(work["_actual"].gt(0).mean()) if len(work) else None
    )
    overall_mean = (
        float(work["_actual"].mean()) if len(work) else None
    )

    by_day: dict[str, object] = {}
    good_days = 0
    for day in FRESH_DAYS:
        part = work.loc[
            work["trading_day"].astype(str).eq(day)
        ].copy()
        corr = _safe_spearman(part["_actual"], part["_score"])
        count = max(1, int(np.ceil(len(part) * 0.10))) if len(part) else 0
        top = part.sort_values(
            "_score", ascending=False, kind="stable"
        ).head(count)
        top_mean = (
            float(top["_actual"].mean()) if len(top) else None
        )
        good = bool(
            corr is not None
            and corr > 0
            and top_mean is not None
            and top_mean > 0
        )
        good_days += int(good)
        by_day[day] = {
            "rows": int(len(part)),
            "spearman": corr,
            "top10_mean_advantage_pct": top_mean,
            "top10_positive_rate": (
                float(top["_actual"].gt(0).mean())
                if len(top)
                else None
            ),
            "both_positive": good,
        }

    age = _age_bucket(work["minutes_held"])
    age_spearman: dict[str, object] = {}
    age_values: list[float] = []
    for label in ["<=2", "2-5", "5-10", "10-20", ">20"]:
        mask = age.astype(str).eq(label)
        part = work.loc[mask]
        corr = (
            _safe_spearman(part["_actual"], part["_score"])
            if len(part) >= 20
            else None
        )
        if corr is not None:
            age_values.append(float(corr))
        age_spearman[label] = {
            "rows": int(len(part)),
            "spearman": corr,
        }

    return {
        "rows": int(len(frame)),
        "evaluable_rows": int(len(work)),
        "spearman": _safe_spearman(
            work["_actual"], work["_score"]
        ),
        "overall_mean_advantage_pct": overall_mean,
        "overall_positive_rate": overall_positive,
        "good_days": int(good_days),
        "by_day": by_day,
        "top10_by_day": _top_fraction_by_day(
            frame, score_column, 0.10
        ),
        "top20_by_day": _top_fraction_by_day(
            frame, score_column, 0.20
        ),
        "same_age_bucket": age_spearman,
        "same_age_median_spearman": (
            float(np.median(age_values)) if age_values else None
        ),
    }


def _prepare_event_frames(
    fit_positions: pd.DataFrame,
    calibration_positions: pd.DataFrame,
    fresh_positions: pd.DataFrame,
    history_scan: pd.DataFrame,
    fresh_scan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_entries, calibration_entries, fresh_entries = _prepare_entries(
        fit_positions, calibration_positions, fresh_positions
    )
    fit = reanchor_event_positions(fit_positions, fit_entries)
    calibration = reanchor_event_positions(
        calibration_positions, calibration_entries
    )
    fresh = reanchor_event_positions(fresh_positions, fresh_entries)

    historical_days = set(
        fit_positions["trading_day"].astype(str)
    ) | set(calibration_positions["trading_day"].astype(str))
    historical_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(historical_days)
    ].copy()
    historical_regime = build_market_regime(historical_scan)
    fresh_regime = build_market_regime(fresh_scan)

    fit = attach_position_transitions(
        attach_market_regime(fit, historical_regime)
    )
    calibration = attach_position_transitions(
        attach_market_regime(calibration, historical_regime)
    )
    fresh = attach_position_transitions(
        attach_market_regime(fresh, fresh_regime)
    )
    return fit, calibration, fresh


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    states_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )
    history_scan = pd.read_parquet(history_scan_path)

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError("request 217 requires complete request178 shards")

    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    found_days = sorted(
        fresh_positions["trading_day"].astype(str).unique()
    )
    if found_days != list(FRESH_DAYS):
        raise ValueError(
            f"request 217 expected {FRESH_DAYS}, found {found_days}"
        )

    fit, calibration, fresh = _prepare_event_frames(
        fit_positions,
        calibration_positions,
        fresh_positions,
        history_scan,
        fresh_scan,
    )

    fit = attach_recent_path_features(attach_rank_target(fit))
    calibration = attach_recent_path_features(
        attach_rank_target(calibration)
    )
    fresh = attach_recent_path_features(attach_rank_target(fresh))

    baseline_rank_model = train_rank_model(fit, EVENT_FEATURES)
    candidate_rank_model = train_rank_model(fit, CANDIDATE_FEATURES)
    request216_model = train_event_advantage_model(fit, calibration)

    calibration["baseline_rank_score"] = predict_rank(
        calibration, baseline_rank_model
    )
    calibration["candidate_path_rank_score"] = predict_rank(
        calibration, candidate_rank_model
    )
    fresh["baseline_rank_score"] = predict_rank(
        fresh, baseline_rank_model
    )
    fresh["candidate_path_rank_score"] = predict_rank(
        fresh, candidate_rank_model
    )
    fresh["request216_expected_advantage_score"] = (
        predict_event_advantage(fresh, request216_model)
    )

    baseline_diag = ranking_diagnostics(
        fresh, "baseline_rank_score"
    )
    candidate_diag = ranking_diagnostics(
        fresh, "candidate_path_rank_score"
    )
    request216_diag = ranking_diagnostics(
        fresh, "request216_expected_advantage_score"
    )
    calibration_candidate_diag = ranking_diagnostics(
        calibration, "candidate_path_rank_score"
    )

    candidate_s = candidate_diag.get("spearman")
    baseline_s = baseline_diag.get("spearman")
    top10 = candidate_diag["top10_by_day"]
    top20 = candidate_diag["top20_by_day"]
    overall_positive = candidate_diag.get("overall_positive_rate")
    top10_positive = top10.get("positive_rate")
    age_median = candidate_diag.get("same_age_median_spearman")

    gate = bool(
        candidate_s is not None
        and float(candidate_s) >= MIN_SPEARMAN
        and baseline_s is not None
        and float(candidate_s) - float(baseline_s)
        >= MIN_SPEARMAN_GAIN
        and int(candidate_diag["good_days"]) >= MIN_GOOD_DAYS
        and top10.get("day_balanced_mean_advantage_pct") is not None
        and float(top10["day_balanced_mean_advantage_pct"]) > 0
        and top20.get("day_balanced_mean_advantage_pct") is not None
        and float(top20["day_balanced_mean_advantage_pct"]) > 0
        and overall_positive is not None
        and top10_positive is not None
        and float(top10_positive) - float(overall_positive)
        >= MIN_TOP10_POSITIVE_RATE_GAIN
        and age_median is not None
        and float(age_median) > 0
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "hypothesis": (
            "recent causal path shape plus rank-oriented learning can "
            "recover HOLD-value ordering that Request216 could not"
        ),
        "target": (
            "within-day percentile rank of realized next-executable-event "
            "BASE HOLD advantage"
        ),
        "fit_rows": int(len(fit)),
        "calibration_rows": int(len(calibration)),
        "fresh_rows": int(len(fresh)),
        "features": {
            "baseline_current_state_count": int(
                len(baseline_rank_model.feature_columns)
            ),
            "candidate_count": int(
                len(candidate_rank_model.feature_columns)
            ),
            "path_feature_count": int(len(PATH_FEATURES)),
            "path_sources": list(PATH_SOURCES),
            "windows": list(PATH_WINDOWS),
            "future_observation_information_used": False,
        },
        "comparators": {
            "request216_expected_value": request216_diag,
            "rank_objective_current_state_only": baseline_diag,
        },
        "candidate_recent_path_rank": candidate_diag,
        "calibration_candidate_recent_path_rank": (
            calibration_candidate_diag
        ),
        "candidate_minus_baseline_spearman": (
            float(candidate_s) - float(baseline_s)
            if candidate_s is not None and baseline_s is not None
            else None
        ),
        "frozen_gate": {
            "min_candidate_spearman": MIN_SPEARMAN,
            "min_spearman_gain_vs_current_state_rank": (
                MIN_SPEARMAN_GAIN
            ),
            "min_good_days": MIN_GOOD_DAYS,
            "top10_day_balanced_advantage_positive": True,
            "top20_day_balanced_advantage_positive": True,
            "min_top10_positive_rate_gain": (
                MIN_TOP10_POSITIVE_RATE_GAIN
            ),
            "same_age_median_spearman_positive": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "calibrate an absolute HOLD/EXIT value gate using historical "
            "fit/calibration only, then replay recurrent trajectories"
        ),
        "next_boundary_if_fail": (
            "stop hand-crafted path expansion and test a learned sequence "
            "representation on chronological observed-state histories"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    states_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    fresh.to_parquet(
        states_output_path, index=False, compression="zstd"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument(
        "--calibration-positions", type=Path, required=True
    )
    parser.add_argument("--history-scan", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.states_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
