"""Request 193: pre-entry shadow-path delayed viability observability.

A HOT/BUY candidate is observed causally for minutes 1-10 without paying the
original entry cost. At each shadow state, this module asks whether entering at
the next executable open would still leave a BASE-positive executable exit
before the inherited HOT+30m research cap.

This request is observability-only. Delayed entry and future exit prices are
labels/outcomes and never model inputs.
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

from .attention_replay import MINUTE_MS
from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .future_cost_cover_state_observability import (
    FEATURES,
    _day_weights,
)
from .market_regime_position_value import (
    MARKET_REGIME_FEATURES,
    attach_market_regime,
    build_market_regime,
)
from .selected_hot_position_value_observability import (
    _base_return,
    _open_map,
)

REQUEST_ID = 193
CLASSIFIER_SEED = 20261096
REGRESSOR_SEED = 20261097
SHADOW_MIN_MINUTE = 1
SHADOW_MAX_MINUTE = 10
RESEARCH_CAP_MINUTES = 30
PROBABILITY_GATE_QUANTILE = 0.75

MIN_AUC = 0.60
MIN_VALUE_SPEARMAN = 0.15
MIN_POSITIVE_SPEARMAN_DAYS = 4
MIN_FIRST_EPISODE_RATE = 0.10
MAX_FIRST_EPISODE_RATE = 0.70
MIN_FIRST_POSITIVE_RATE = 0.60
MIN_FIRST_VALUE_UPLIFT_PCT = 0.50
MIN_POSITIVE_FIRST_DAYS = 4


@dataclass(frozen=True)
class ShadowModels:
    classifier: HistGradientBoostingClassifier
    regressor: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    probability_gate: float
    regression_offset: float
    winsor_low: float
    winsor_high: float


def build_delayed_labels(
    positions: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    """Attach delayed-entry oracle labels for causal shadow states."""

    frame = positions.copy()
    held = pd.to_numeric(frame["minutes_held"], errors="coerce")
    frame = frame.loc[
        held.notna()
        & held.ge(SHADOW_MIN_MINUTE)
        & held.le(SHADOW_MAX_MINUTE)
    ].copy()
    if frame.empty:
        return frame

    opens = _open_map(scan)
    open_groups: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for (day, ticker, timestamp), price in opens.items():
        key = (str(day), str(ticker).upper())
        open_groups.setdefault(key, []).append(
            (int(timestamp), float(price))
        )
    for key in open_groups:
        open_groups[key].sort()

    delayed_entry_prices: list[float] = []
    delayed_best: list[float] = []
    future_exit_counts: list[int] = []

    for row in frame.to_dict("records"):
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        hot_t = int(row["hot_t"])
        state_t = int(row["state_t"])
        delayed_entry = opens.get(
            (day, ticker, state_t),
            np.nan,
        )
        if (
            pd.isna(delayed_entry)
            or not np.isfinite(float(delayed_entry))
            or float(delayed_entry) <= 0
        ):
            delayed_entry_prices.append(np.nan)
            delayed_best.append(np.nan)
            future_exit_counts.append(0)
            continue

        cap_t = hot_t + RESEARCH_CAP_MINUTES * MINUTE_MS
        future_prices = [
            price
            for timestamp, price in open_groups.get(
                (day, ticker),
                [],
            )
            if state_t < int(timestamp) <= cap_t
        ]
        values = [
            _base_return(
                float(delayed_entry),
                float(exit_open),
            )
            for exit_open in future_prices
        ]
        values = [
            float(value)
            for value in values
            if pd.notna(value) and np.isfinite(float(value))
        ]

        delayed_entry_prices.append(float(delayed_entry))
        future_exit_counts.append(len(values))
        delayed_best.append(
            float(max(values))
            if values
            else np.nan
        )

    frame["delayed_entry_reference_open"] = delayed_entry_prices
    frame["delayed_future_exit_count"] = future_exit_counts
    frame["delayed_future_best_base_return_pct"] = delayed_best
    target = pd.to_numeric(
        frame["delayed_future_best_base_return_pct"],
        errors="coerce",
    )
    frame["delayed_future_cost_coverable"] = (
        target.gt(0).where(target.notna())
    )
    return frame


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def _safe_auc(actual: pd.Series, score: pd.Series) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    y = y.loc[valid].astype(int)
    if len(y) < 20 or y.nunique() < 2:
        return None
    return float(roc_auc_score(y, s.loc[valid].astype(float)))


def _safe_ap(actual: pd.Series, score: pd.Series) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    y = y.loc[valid].astype(int)
    if len(y) < 20 or int(y.sum()) == 0:
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
    value = y.loc[valid].corr(
        s.loc[valid],
        method="spearman",
    )
    return None if pd.isna(value) else float(value)


def train_models(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> ShadowModels:
    fit_target = pd.to_numeric(
        fit["delayed_future_best_base_return_pct"],
        errors="coerce",
    )
    fit_valid = fit_target.notna()
    train = fit.loc[fit_valid].copy()
    y_value = fit_target.loc[fit_valid].astype(float)
    y_class = y_value.gt(0).astype(int)
    if len(train) < 800 or y_class.nunique() < 2:
        raise ValueError("request 193 insufficient fit support")

    columns = tuple(
        column
        for column in FEATURES
        if column in train.columns
        and pd.to_numeric(
            train[column],
            errors="coerce",
        ).notna().any()
    )
    if not columns:
        raise ValueError("request 193 has no causal shadow features")

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
        calibration["delayed_future_best_base_return_pct"],
        errors="coerce",
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 400:
        raise ValueError("request 193 insufficient calibration support")

    cal_x = _feature_frame(cal, columns)
    cal_probability = classifier.predict_proba(cal_x)[:, 1]
    probability_gate = float(
        np.quantile(
            cal_probability,
            PROBABILITY_GATE_QUANTILE,
        )
    )

    raw = regressor.predict(cal_x)
    residual = (
        cal_target.loc[cal_valid].to_numpy(dtype=float)
        - raw
    )
    offset = float(
        np.average(
            residual,
            weights=_day_weights(cal),
        )
    )
    return ShadowModels(
        classifier=classifier,
        regressor=regressor,
        columns=columns,
        probability_gate=probability_gate,
        regression_offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def score_states(
    frame: pd.DataFrame,
    models: ShadowModels,
) -> pd.DataFrame:
    result = frame.copy()
    x = _feature_frame(result, models.columns)
    result["delayed_cost_cover_probability"] = (
        models.classifier.predict_proba(x)[:, 1]
    )
    result["predicted_delayed_future_best_base_pct"] = (
        models.regressor.predict(x)
        + models.regression_offset
    )
    result["shadow_qualifies"] = (
        pd.to_numeric(
            result["delayed_cost_cover_probability"],
            errors="coerce",
        ).ge(models.probability_gate)
        & pd.to_numeric(
            result["predicted_delayed_future_best_base_pct"],
            errors="coerce",
        ).gt(0)
    )
    return result


def _state_metrics(frame: pd.DataFrame) -> dict[str, object]:
    target = pd.to_numeric(
        frame["delayed_future_best_base_return_pct"],
        errors="coerce",
    )
    probability = pd.to_numeric(
        frame["delayed_cost_cover_probability"],
        errors="coerce",
    )
    prediction = pd.to_numeric(
        frame["predicted_delayed_future_best_base_pct"],
        errors="coerce",
    )
    valid = target.notna() & probability.notna() & prediction.notna()
    target = target.loc[valid]
    probability = probability.loc[valid]
    prediction = prediction.loc[valid]
    y = target.gt(0).astype(int)
    part = frame.loc[valid].copy()

    selected = part.loc[
        part["shadow_qualifies"].fillna(False)
    ].copy()
    selected_target = pd.to_numeric(
        selected["delayed_future_best_base_return_pct"],
        errors="coerce",
    )
    prevalence = float(y.mean()) if len(y) else None
    all_mean = float(target.mean()) if len(target) else None
    selected_rate = (
        float(len(selected) / len(part))
        if len(part)
        else None
    )
    selected_positive = (
        float(selected_target.gt(0).mean())
        if len(selected_target)
        else None
    )
    selected_mean = (
        float(selected_target.mean())
        if len(selected_target)
        else None
    )

    return {
        "rows": int(len(frame)),
        "evaluable_rows": int(valid.sum()),
        "delayed_cost_cover_prevalence": prevalence,
        "classifier_auc": _safe_auc(y, probability),
        "classifier_average_precision": _safe_ap(
            y,
            probability,
        ),
        "value_spearman": _safe_spearman(
            target,
            prediction,
        ),
        "selected_rows": int(len(selected)),
        "selected_rate": selected_rate,
        "selected_delayed_positive_rate": selected_positive,
        "selected_delayed_best_mean_pct": selected_mean,
        "all_shadow_delayed_best_mean_pct": all_mean,
        "selected_minus_all_value_uplift_pct": (
            float(selected_mean - all_mean)
            if selected_mean is not None
            and all_mean is not None
            else None
        ),
    }


def _first_qualifying_episode_bridge(
    scored: pd.DataFrame,
) -> dict[str, object]:
    all_episodes = int(
        scored.loc[:, EPISODE_KEYS]
        .drop_duplicates()
        .shape[0]
    )
    records: list[dict[str, object]] = []

    for _, group in scored.groupby(EPISODE_KEYS, sort=False):
        work = group.copy()
        work["_minute"] = pd.to_numeric(
            work["minutes_held"],
            errors="coerce",
        )
        work = work.sort_values("_minute", kind="stable")
        qualified = work.loc[
            work["shadow_qualifies"].fillna(False)
        ]
        if qualified.empty:
            continue
        row = qualified.iloc[0]
        records.append(
            {
                "trading_day": str(row["trading_day"]),
                "ticker": str(row["ticker"]).upper(),
                "hot_t": int(row["hot_t"]),
                "first_qualifying_minute": float(row["_minute"]),
                "realized_delayed_best_pct": float(
                    row["delayed_future_best_base_return_pct"]
                ),
                "delayed_positive": bool(
                    float(
                        row["delayed_future_best_base_return_pct"]
                    )
                    > 0
                ),
            }
        )

    selected = pd.DataFrame(records)
    selected_count = int(len(selected))
    rate = (
        float(selected_count / all_episodes)
        if all_episodes
        else None
    )
    values = (
        pd.to_numeric(
            selected["realized_delayed_best_pct"],
            errors="coerce",
        )
        if len(selected)
        else pd.Series(dtype=float)
    )
    minutes = (
        pd.to_numeric(
            selected["first_qualifying_minute"],
            errors="coerce",
        )
        if len(selected)
        else pd.Series(dtype=float)
    )

    all_state_values = pd.to_numeric(
        scored["delayed_future_best_base_return_pct"],
        errors="coerce",
    ).dropna()
    all_mean = (
        float(all_state_values.mean())
        if len(all_state_values)
        else None
    )
    selected_mean = (
        float(values.mean())
        if len(values)
        else None
    )

    by_day: dict[str, object] = {}
    positive_days = 0
    for day in FRESH_DAYS:
        if len(selected):
            part = selected.loc[
                selected["trading_day"].astype(str).eq(day)
            ].copy()
        else:
            part = selected
        day_values = (
            pd.to_numeric(
                part["realized_delayed_best_pct"],
                errors="coerce",
            )
            if len(part)
            else pd.Series(dtype=float)
        )
        mean = (
            float(day_values.mean())
            if len(day_values)
            else None
        )
        if mean is not None and mean > 0:
            positive_days += 1
        by_day[day] = {
            "qualified_episodes": int(len(part)),
            "realized_delayed_best_mean_pct": mean,
            "positive_rate": (
                float(day_values.gt(0).mean())
                if len(day_values)
                else None
            ),
        }

    return {
        "total_episodes": all_episodes,
        "qualified_episodes": selected_count,
        "qualified_episode_rate": rate,
        "first_qualifying_minute_mean": (
            float(minutes.mean())
            if len(minutes)
            else None
        ),
        "first_qualifying_minute_median": (
            float(minutes.median())
            if len(minutes)
            else None
        ),
        "first_qualifying_minute_p90": (
            float(minutes.quantile(0.90))
            if len(minutes)
            else None
        ),
        "realized_delayed_best_mean_pct": selected_mean,
        "delayed_positive_rate": (
            float(values.gt(0).mean())
            if len(values)
            else None
        ),
        "all_shadow_delayed_best_mean_pct": all_mean,
        "selected_minus_all_value_uplift_pct": (
            float(selected_mean - all_mean)
            if selected_mean is not None
            and all_mean is not None
            else None
        ),
        "positive_mean_days": int(positive_days),
        "by_day": by_day,
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_path)
    calibration_positions = pd.read_parquet(calibration_path)
    history_scan = pd.read_parquet(history_scan_path)

    position_paths = sorted(
        fresh_dir.glob("*-positions.parquet")
    )
    scan_paths = sorted(
        fresh_dir.glob("*-scan.parquet")
    )
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError(
            "request 193 requires complete request 178 shards"
        )

    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    found = sorted(
        fresh_positions["trading_day"].astype(str).unique()
    )
    if found != list(FRESH_DAYS):
        raise ValueError(
            f"request 193 expected {FRESH_DAYS}, found {found}"
        )

    historical_days = set(
        fit_positions["trading_day"].astype(str)
    ) | set(
        calibration_positions["trading_day"].astype(str)
    )
    hist_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(
            historical_days
        )
    ].copy()

    fit = build_delayed_labels(
        fit_positions,
        hist_scan,
    )
    calibration = build_delayed_labels(
        calibration_positions,
        hist_scan,
    )
    fresh = build_delayed_labels(
        fresh_positions,
        fresh_scan,
    )

    hist_regime = build_market_regime(hist_scan)
    fresh_regime = build_market_regime(fresh_scan)
    fit = attach_market_regime(fit, hist_regime)
    calibration = attach_market_regime(
        calibration,
        hist_regime,
    )
    fresh = attach_market_regime(
        fresh,
        fresh_regime,
    )

    models = train_models(fit, calibration)
    scored = score_states(fresh, models)
    overall = _state_metrics(scored)
    bridge = _first_qualifying_episode_bridge(scored)

    by_day: dict[str, object] = {}
    positive_spearman_days = 0
    for day in FRESH_DAYS:
        part = scored.loc[
            scored["trading_day"].astype(str).eq(day)
        ].copy()
        metrics = _state_metrics(part)
        value = metrics["value_spearman"]
        if value is not None and float(value) > 0:
            positive_spearman_days += 1
        by_day[day] = metrics

    context_coverage = float(
        scored.loc[:, MARKET_REGIME_FEATURES]
        .notna()
        .any(axis=1)
        .mean()
    )

    gate = bool(
        overall["classifier_auc"] is not None
        and float(overall["classifier_auc"]) >= MIN_AUC
        and overall["value_spearman"] is not None
        and float(overall["value_spearman"])
        >= MIN_VALUE_SPEARMAN
        and positive_spearman_days
        >= MIN_POSITIVE_SPEARMAN_DAYS
        and bridge["qualified_episode_rate"] is not None
        and float(bridge["qualified_episode_rate"])
        >= MIN_FIRST_EPISODE_RATE
        and float(bridge["qualified_episode_rate"])
        <= MAX_FIRST_EPISODE_RATE
        and bridge["delayed_positive_rate"] is not None
        and float(bridge["delayed_positive_rate"])
        >= MIN_FIRST_POSITIVE_RATE
        and bridge["realized_delayed_best_mean_pct"] is not None
        and float(bridge["realized_delayed_best_mean_pct"]) > 0
        and bridge["selected_minus_all_value_uplift_pct"] is not None
        and float(
            bridge["selected_minus_all_value_uplift_pct"]
        )
        >= MIN_FIRST_VALUE_UPLIFT_PCT
        and int(bridge["positive_mean_days"])
        >= MIN_POSITIVE_FIRST_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "policy_executed": False,
        "account_simulation_applicable": False,
        "shadow_window_minutes": [
            SHADOW_MIN_MINUTE,
            SHADOW_MAX_MINUTE,
        ],
        "research_cap_minutes_from_hot": RESEARCH_CAP_MINUTES,
        "feature_count": len(models.columns),
        "market_context_coverage": context_coverage,
        "calibration_probability_p75_gate": (
            models.probability_gate
        ),
        "state_level": overall,
        "first_qualifying_episode_bridge": bridge,
        "positive_value_spearman_days": int(
            positive_spearman_days
        ),
        "by_day": by_day,
        "model_diagnostics": {
            "regression_offset_pct": models.regression_offset,
            "winsor_low_pct": models.winsor_low,
            "winsor_high_pct": models.winsor_high,
        },
        "frozen_gate": {
            "min_auc": MIN_AUC,
            "min_value_spearman": MIN_VALUE_SPEARMAN,
            "min_positive_spearman_days": (
                MIN_POSITIVE_SPEARMAN_DAYS
            ),
            "min_first_episode_rate": MIN_FIRST_EPISODE_RATE,
            "max_first_episode_rate": MAX_FIRST_EPISODE_RATE,
            "min_first_positive_rate": MIN_FIRST_POSITIVE_RATE,
            "min_first_value_uplift_pct": (
                MIN_FIRST_VALUE_UPLIFT_PCT
            ),
            "min_positive_first_days": MIN_POSITIVE_FIRST_DAYS,
        },
        "development_bridge_pass": gate,
        "next_step_if_pass": (
            "request194 honest WAIT/ENTER/ABSTAIN plus recurrent "
            "post-entry management and KRW reference ledger"
        ),
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scored.to_parquet(
        scored_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--history-scan",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fresh-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--scored-output",
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
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
