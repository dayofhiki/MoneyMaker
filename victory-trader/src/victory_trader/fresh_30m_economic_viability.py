"""Request 190: fresh 30-minute economic viability revalidation.

Tests whether the original-HOT causal entry state can identify BUY episodes
that contain a BASE-positive executable exit somewhere within 30 minutes.
Oracle labels are research targets only and are not executable policy returns.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import average_precision_score, roc_auc_score

from .decomposed_cost_aware_entry_surplus import (
    DECOMPOSED_FEATURES,
    EPISODE_KEYS,
    FRESH_DAYS,
    build_historical_states,
)
from .fresh_buy_horizon_ceiling_audit import build_episode_ceilings

REQUEST_ID = 190
CLASSIFIER_SEED = 20261092
REGRESSOR_SEED = 20261093
BOOTSTRAP_SEED = 20261090
BOOTSTRAP_SAMPLES = 10_000
MIN_ENTRY_COVERAGE = 0.90
MIN_SECOND_COVERAGE = 0.90
MIN_AUC = 0.60
MIN_SPEARMAN = 0.15
MIN_POSITIVE_SPEARMAN_DAYS = 4
MIN_SELECTED_POSITIVE_RATE = 0.60
MIN_UPLIFT_PCT = 0.50
MIN_POSITIVE_DAYS = 4


@dataclass(frozen=True)
class ViabilityModels:
    classifier: HistGradientBoostingClassifier
    regressor: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    probability_gate: float
    regression_offset: float
    winsor_low: float
    winsor_high: float


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    day = frame["trading_day"].astype(str)
    counts = day.map(day.value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(dtype=float)
    mean = float(np.mean(weights))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("request 190 invalid equal-day weights")
    return weights / mean


def build_30m_labels(position_rows: pd.DataFrame) -> pd.DataFrame:
    labels = build_episode_ceilings(position_rows).loc[
        :, EPISODE_KEYS + ["oracle_30m_base_pct"]
    ].copy()
    labels["cost_coverable"] = (
        pd.to_numeric(labels["oracle_30m_base_pct"], errors="coerce")
        .gt(0)
        .astype("Int64")
    )
    return labels


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def train_models(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> ViabilityModels:
    fit_target = pd.to_numeric(
        fit["oracle_30m_base_pct"], errors="coerce"
    )
    fit_valid = (
        fit_target.notna()
        & fit["entry_state_available"].fillna(False)
    )
    train = fit.loc[fit_valid].copy()
    y_value = fit_target.loc[fit_valid].astype(float)
    y_class = y_value.gt(0).astype(int)
    if len(train) < 300 or y_class.nunique() < 2:
        raise ValueError("request 190 insufficient fit support")

    columns = tuple(
        column
        for column in DECOMPOSED_FEATURES
        if column in train.columns
        and pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    if not columns:
        raise ValueError("request 190 has no usable causal entry features")

    weights = _day_weights(train)
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=CLASSIFIER_SEED,
    )
    classifier.fit(
        _feature_frame(train, columns),
        y_class,
        sample_weight=weights,
    )

    low, high = np.quantile(
        y_value.to_numpy(dtype=float),
        [0.005, 0.995],
    )
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=REGRESSOR_SEED,
    )
    regressor.fit(
        _feature_frame(train, columns),
        y_value.clip(lower=low, upper=high),
        sample_weight=weights,
    )

    cal_target = pd.to_numeric(
        calibration["oracle_30m_base_pct"], errors="coerce"
    )
    cal_valid = (
        cal_target.notna()
        & calibration["entry_state_available"].fillna(False)
    )
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 150:
        raise ValueError("request 190 insufficient calibration support")

    cal_x = _feature_frame(cal, columns)
    cal_probability = classifier.predict_proba(cal_x)[:, 1]
    probability_gate = float(np.quantile(cal_probability, 0.75))

    raw_value = regressor.predict(cal_x)
    residual = cal_target.loc[cal_valid].to_numpy(dtype=float) - raw_value
    regression_offset = float(
        np.average(residual, weights=_day_weights(cal))
    )
    return ViabilityModels(
        classifier=classifier,
        regressor=regressor,
        columns=columns,
        probability_gate=probability_gate,
        regression_offset=regression_offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def _safe_auc(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    y = y.loc[valid].astype(int)
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y, s.loc[valid].astype(float)))


def _safe_ap(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    y = y.loc[valid].astype(int)
    if int(y.sum()) == 0:
        return None
    return float(
        average_precision_score(
            y,
            s.loc[valid].astype(float),
        )
    )


def _safe_spearman(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    value = y.loc[valid].corr(s.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _day_balanced(
    frame: pd.DataFrame,
    column: str,
) -> float | None:
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


def _bootstrap_selected(
    selected: pd.DataFrame,
) -> dict[str, float | int | None]:
    values = pd.to_numeric(
        selected["oracle_30m_base_pct"], errors="coerce"
    )
    valid = values.notna()
    daily = (
        pd.DataFrame(
            {
                "day": selected.loc[valid, "trading_day"].astype(str),
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
    history_first_hot_path: Path,
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_entry_states_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    first_hot = pd.read_parquet(history_first_hot_path)
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(calibration_positions_path)

    fit = build_historical_states(
        first_hot,
        build_30m_labels(fit_positions),
    )
    calibration = build_historical_states(
        first_hot,
        build_30m_labels(calibration_positions),
    )
    models = train_models(fit, calibration)

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(position_paths) != len(FRESH_DAYS):
        raise ValueError("request 190 requires complete request 178 positions")
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    labels = build_30m_labels(fresh_positions)

    fresh_features = pd.read_parquet(fresh_entry_states_path)
    drop_labels = {
        "entry_open",
        "enter_gross_pct",
        "enter_drag_pct",
        "enter_now_base_pct",
        "wait_gross_pct",
        "wait_drag_pct",
        "wait_1m_base_pct",
        "predicted_enter_gross_pct",
        "predicted_enter_drag_pct",
        "predicted_enter_now_base_pct",
        "predicted_wait_gross_pct",
        "predicted_wait_drag_pct",
        "predicted_wait_1m_base_pct",
        "action",
        "policy_realized_base_pct",
        "action_value_resolved",
        "executed",
    }
    feature_columns = [
        column for column in fresh_features.columns
        if column not in drop_labels
        and column not in {"oracle_30m_base_pct", "cost_coverable"}
    ]
    fresh = fresh_features.loc[:, feature_columns].merge(
        labels,
        on=EPISODE_KEYS,
        how="left",
        validate="one_to_one",
    )
    found = sorted(fresh["trading_day"].astype(str).unique())
    if found != list(FRESH_DAYS):
        raise ValueError(f"request 190 expected {FRESH_DAYS}, found {found}")

    x = _feature_frame(fresh, models.columns)
    fresh["cost_cover_probability"] = (
        models.classifier.predict_proba(x)[:, 1]
    )
    fresh["predicted_oracle_30m_base_pct"] = (
        models.regressor.predict(x) + models.regression_offset
    )
    fresh["selected"] = (
        pd.to_numeric(
            fresh["cost_cover_probability"], errors="coerce"
        )
        .ge(models.probability_gate)
    )

    target = pd.to_numeric(
        fresh["oracle_30m_base_pct"], errors="coerce"
    )
    y = target.gt(0).astype(int)
    probability = pd.to_numeric(
        fresh["cost_cover_probability"], errors="coerce"
    )
    predicted_value = pd.to_numeric(
        fresh["predicted_oracle_30m_base_pct"], errors="coerce"
    )

    covered = fresh["entry_state_available"].fillna(False)
    second = pd.to_numeric(
        fresh.get("active_seconds_60"), errors="coerce"
    ).fillna(0).gt(0)
    entry_coverage = float(covered.mean()) if len(fresh) else None
    second_coverage = (
        float(second.loc[covered].mean()) if int(covered.sum()) else None
    )

    valid = covered & target.notna()
    selected = fresh.loc[
        valid & fresh["selected"]
    ].copy()
    selected_values = pd.to_numeric(
        selected["oracle_30m_base_pct"], errors="coerce"
    )
    all_values = target.loc[valid]
    selected_mean = (
        float(selected_values.mean()) if len(selected_values) else None
    )
    selected_day = _day_balanced(
        selected, "oracle_30m_base_pct"
    )
    selected_positive = (
        float(selected_values.gt(0).mean())
        if len(selected_values)
        else None
    )
    all_mean = float(all_values.mean()) if len(all_values) else None
    all_day = _day_balanced(
        fresh.loc[valid], "oracle_30m_base_pct"
    )
    uplift = (
        float(selected_mean - all_mean)
        if selected_mean is not None and all_mean is not None
        else None
    )
    bootstrap = _bootstrap_selected(selected)

    auc = _safe_auc(y.loc[valid], probability.loc[valid])
    ap = _safe_ap(y.loc[valid], probability.loc[valid])
    spearman = _safe_spearman(
        target.loc[valid],
        predicted_value.loc[valid],
    )

    by_day: dict[str, object] = {}
    positive_spearman_days = 0
    positive_selected_days = 0
    for day in FRESH_DAYS:
        mask = fresh["trading_day"].astype(str).eq(day) & valid
        part = fresh.loc[mask].copy()
        pt = pd.to_numeric(
            part["oracle_30m_base_pct"], errors="coerce"
        )
        py = pt.gt(0).astype(int)
        pp = pd.to_numeric(
            part["cost_cover_probability"], errors="coerce"
        )
        pv = pd.to_numeric(
            part["predicted_oracle_30m_base_pct"], errors="coerce"
        )
        day_auc = _safe_auc(py, pp)
        day_spearman = _safe_spearman(pt, pv)
        if day_spearman is not None and day_spearman > 0:
            positive_spearman_days += 1

        day_selected = part.loc[part["selected"]]
        vals = pd.to_numeric(
            day_selected["oracle_30m_base_pct"], errors="coerce"
        )
        mean = float(vals.mean()) if len(vals) else None
        if mean is not None and mean > 0:
            positive_selected_days += 1

        by_day[day] = {
            "rows": int(len(part)),
            "positive_rate": (
                float(py.mean()) if len(part) else None
            ),
            "auc": day_auc,
            "spearman": day_spearman,
            "selected_rows": int(len(day_selected)),
            "selected_mean_pct": mean,
            "selected_positive_rate": (
                float(vals.gt(0).mean()) if len(vals) else None
            ),
        }

    gate = bool(
        entry_coverage is not None
        and entry_coverage >= MIN_ENTRY_COVERAGE
        and second_coverage is not None
        and second_coverage >= MIN_SECOND_COVERAGE
        and auc is not None
        and auc >= MIN_AUC
        and spearman is not None
        and spearman >= MIN_SPEARMAN
        and positive_spearman_days >= MIN_POSITIVE_SPEARMAN_DAYS
        and selected_day is not None
        and selected_day > 0
        and selected_positive is not None
        and selected_positive >= MIN_SELECTED_POSITIVE_RATE
        and uplift is not None
        and uplift >= MIN_UPLIFT_PCT
        and bootstrap["ci_low_pct"] is not None
        and float(bootstrap["ci_low_pct"]) > 0
        and positive_selected_days >= MIN_POSITIVE_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "target_semantics": (
            "episode hindsight-best exact BASE exit through minute30; "
            "observability only"
        ),
        "feature_count": len(models.columns),
        "fresh_entry_state_coverage": entry_coverage,
        "fresh_second_feature_coverage": second_coverage,
        "cost_coverable_rate": (
            float(y.loc[valid].mean()) if int(valid.sum()) else None
        ),
        "classifier_auc": auc,
        "classifier_average_precision": ap,
        "value_spearman": spearman,
        "calibration_probability_p75_gate": models.probability_gate,
        "selected_rows": int(len(selected)),
        "selected_rate": (
            float(len(selected) / int(valid.sum()))
            if int(valid.sum())
            else None
        ),
        "selected_oracle_mean_pct": selected_mean,
        "selected_day_balanced_oracle_pct": selected_day,
        "selected_positive_rate": selected_positive,
        "all_buy_oracle_mean_pct": all_mean,
        "all_buy_day_balanced_oracle_pct": all_day,
        "selected_minus_all_buy_uplift_pct": uplift,
        "selected_day_bootstrap": bootstrap,
        "positive_spearman_days": int(positive_spearman_days),
        "positive_selected_days": int(positive_selected_days),
        "by_day": by_day,
        "model_diagnostics": {
            "regression_offset_pct": models.regression_offset,
            "winsor_low_pct": models.winsor_low,
            "winsor_high_pct": models.winsor_high,
        },
        "frozen_gate": {
            "min_entry_coverage": MIN_ENTRY_COVERAGE,
            "min_second_coverage": MIN_SECOND_COVERAGE,
            "min_auc": MIN_AUC,
            "min_value_spearman": MIN_SPEARMAN,
            "min_positive_spearman_days": MIN_POSITIVE_SPEARMAN_DAYS,
            "min_selected_positive_rate": MIN_SELECTED_POSITIVE_RATE,
            "min_selected_minus_all_buy_uplift_pct": MIN_UPLIFT_PCT,
            "bootstrap_low_must_be_positive": True,
            "min_positive_selected_days": MIN_POSITIVE_DAYS,
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
    parser.add_argument("--history-first-hot", type=Path, required=True)
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument("--calibration-positions", type=Path, required=True)
    parser.add_argument("--fresh-entry-states", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.history_first_hot,
        args.fit_positions,
        args.calibration_positions,
        args.fresh_entry_states,
        args.fresh_dir,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
