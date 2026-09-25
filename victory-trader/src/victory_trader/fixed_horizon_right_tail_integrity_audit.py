"""Request 196A: fixed-horizon right-tail integrity audit.

Checks whether Request-196 winner observability survives after removing the
inherited HOT+30m remaining-horizon shortcut. Development-only; no action policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .attention_replay import MINUTE_MS
from .decomposed_cost_aware_entry_surplus import FRESH_DAYS
from .future_cost_cover_state_observability import _day_weights
from .market_regime_position_value import (
    attach_market_regime,
    build_market_regime,
)
from .selected_hot_position_value_observability import (
    _base_return,
    _open_map,
)
from .shadow_entry_repricing_decomposition import (
    FEATURES,
    add_shadow_features,
)

REQUEST_ID = "196A"
FIXED_HORIZON_MINUTES = 20
WINNER_THRESHOLD_PCT = 3.0
MODEL_SEED = 20261103
NO_CLOCK_SEED = 20261104
SHADOW_MIN_MINUTE = 1
SHADOW_MAX_MINUTE = 10
CLOCK_COLUMNS = {
    "minutes_held",
    "minutes_since_open",
    "minutes_to_close",
}


@dataclass(frozen=True)
class Fitted:
    model: HistGradientBoostingClassifier
    columns: tuple[str, ...]


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
    return float(average_precision_score(y, s.loc[valid].astype(float)))


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def add_fixed20_labels(
    positions: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    frame = positions.copy()
    held = pd.to_numeric(frame["minutes_held"], errors="coerce")
    to_close = pd.to_numeric(frame["minutes_to_close"], errors="coerce")
    frame = frame.loc[
        held.notna()
        & held.between(SHADOW_MIN_MINUTE, SHADOW_MAX_MINUTE)
        & to_close.ge(FIXED_HORIZON_MINUTES)
    ].copy()
    if frame.empty:
        return frame

    opens = _open_map(scan)
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for (day, ticker, timestamp), price in opens.items():
        grouped.setdefault(
            (str(day), str(ticker).upper()),
            [],
        ).append((int(timestamp), float(price)))
    for values in grouped.values():
        values.sort(key=lambda item: item[0])

    delayed_entries: list[float] = []
    best_values: list[float] = []
    exit_counts: list[int] = []

    for row in frame.to_dict("records"):
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        state_t = int(row["state_t"])
        entry = opens.get((day, ticker, state_t), np.nan)
        if (
            pd.isna(entry)
            or not np.isfinite(float(entry))
            or float(entry) <= 0
        ):
            delayed_entries.append(np.nan)
            best_values.append(np.nan)
            exit_counts.append(0)
            continue

        cap_t = state_t + FIXED_HORIZON_MINUTES * MINUTE_MS
        values: list[float] = []
        for timestamp, price in grouped.get((day, ticker), []):
            if not (state_t < int(timestamp) <= cap_t):
                continue
            value = _base_return(float(entry), float(price))
            if pd.notna(value) and np.isfinite(float(value)):
                values.append(float(value))

        delayed_entries.append(float(entry))
        exit_counts.append(len(values))
        best_values.append(max(values) if values else np.nan)

    frame["fixed20_entry_open"] = delayed_entries
    frame["fixed20_exit_count"] = exit_counts
    frame["fixed20_best_base_pct"] = best_values
    value = pd.to_numeric(
        frame["fixed20_best_base_pct"],
        errors="coerce",
    )
    frame["fixed20_strong_winner_3"] = (
        value.ge(WINNER_THRESHOLD_PCT).where(value.notna())
    )
    return frame


def train(
    fit: pd.DataFrame,
    *,
    exclude_clock: bool,
    seed: int,
) -> Fitted:
    target = pd.to_numeric(
        fit["fixed20_strong_winner_3"],
        errors="coerce",
    )
    valid = target.notna()
    train_frame = fit.loc[valid].copy()
    y = target.loc[valid].astype(int)
    if len(train_frame) < 700 or y.nunique() < 2:
        raise ValueError(
            "request 196A insufficient fixed-horizon fit support"
        )

    columns = tuple(
        column
        for column in FEATURES
        if column in train_frame.columns
        and (not exclude_clock or column not in CLOCK_COLUMNS)
        and pd.to_numeric(
            train_frame[column],
            errors="coerce",
        ).notna().any()
    )
    if not columns:
        raise ValueError("request 196A has no usable features")

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _feature_frame(train_frame, columns),
        y,
        sample_weight=_day_weights(train_frame),
    )
    return Fitted(model=model, columns=columns)


def predict(frame: pd.DataFrame, fitted: Fitted) -> np.ndarray:
    return fitted.model.predict_proba(
        _feature_frame(frame, fitted.columns)
    )[:, 1]


def _by_minute(frame: pd.DataFrame) -> dict[str, object]:
    rows: dict[str, object] = {}
    for minute, group in frame.groupby("minutes_held", sort=True):
        actual = pd.to_numeric(
            group["fixed20_strong_winner_3"],
            errors="coerce",
        )
        value = pd.to_numeric(
            group["fixed20_best_base_pct"],
            errors="coerce",
        )
        valid = actual.notna()
        rows[str(int(float(minute)))] = {
            "rows": int(valid.sum()),
            "winner_rate": (
                float(actual.loc[valid].mean())
                if int(valid.sum())
                else None
            ),
            "best_base_mean_pct": (
                float(value.loc[valid].mean())
                if int(valid.sum())
                else None
            ),
        }
    return rows


def metrics(frame: pd.DataFrame) -> dict[str, object]:
    actual = pd.to_numeric(
        frame["fixed20_strong_winner_3"],
        errors="coerce",
    )
    full_score = pd.to_numeric(
        frame["fixed20_full_score"],
        errors="coerce",
    )
    no_clock_score = pd.to_numeric(
        frame["fixed20_no_clock_score"],
        errors="coerce",
    )
    held = pd.to_numeric(frame["minutes_held"], errors="coerce")
    value = pd.to_numeric(
        frame["fixed20_best_base_pct"],
        errors="coerce",
    )

    valid = (
        actual.notna()
        & full_score.notna()
        & no_clock_score.notna()
        & held.notna()
        & value.notna()
    )
    part = frame.loc[valid].copy()
    actual = actual.loc[valid]
    full_score = full_score.loc[valid]
    no_clock_score = no_clock_score.loc[valid]
    held = held.loc[valid]
    value = value.loc[valid]

    by_day: dict[str, object] = {}
    for day in FRESH_DAYS:
        mask = part["trading_day"].astype(str).eq(day)
        by_day[day] = {
            "rows": int(mask.sum()),
            "full_auc": _safe_auc(
                actual.loc[mask],
                full_score.loc[mask],
            ),
            "no_clock_auc": _safe_auc(
                actual.loc[mask],
                no_clock_score.loc[mask],
            ),
        }

    cutoff = (
        float(full_score.quantile(0.75))
        if len(full_score)
        else np.nan
    )
    top_mask = full_score.ge(cutoff) if np.isfinite(cutoff) else pd.Series(
        False,
        index=full_score.index,
    )
    top_actual = actual.loc[top_mask]
    top_value = value.loc[top_mask]

    top_by_day: dict[str, object] = {}
    positive_days = 0
    for day in FRESH_DAYS:
        day_mask = (
            part.loc[top_mask, "trading_day"]
            .astype(str)
            .eq(day)
        )
        day_values = top_value.loc[day_mask.to_numpy()]
        mean_value = (
            float(day_values.mean())
            if len(day_values)
            else None
        )
        if mean_value is not None and mean_value > 0:
            positive_days += 1
        top_by_day[day] = {
            "rows": int(len(day_values)),
            "best_base_mean_pct": mean_value,
        }

    prevalence = float(actual.mean()) if len(actual) else None
    top_rate = float(top_actual.mean()) if len(top_actual) else None
    all_mean = float(value.mean()) if len(value) else None
    top_mean = float(top_value.mean()) if len(top_value) else None

    return {
        "evaluable_rows": int(len(part)),
        "prevalence": prevalence,
        "full_auc": _safe_auc(actual, full_score),
        "full_average_precision": _safe_ap(actual, full_score),
        "no_clock_auc": _safe_auc(actual, no_clock_score),
        "no_clock_average_precision": _safe_ap(
            actual,
            no_clock_score,
        ),
        "minutes_held_auc": _safe_auc(actual, held),
        "negative_minutes_held_auc": _safe_auc(actual, -held),
        "by_day": by_day,
        "by_shadow_minute": _by_minute(part),
        "top_quartile": {
            "diagnostic_only": True,
            "fresh_score_cutoff": (
                float(cutoff) if np.isfinite(cutoff) else None
            ),
            "rows": int(top_mask.sum()),
            "winner_rate": top_rate,
            "prevalence_uplift": (
                float(top_rate - prevalence)
                if top_rate is not None and prevalence is not None
                else None
            ),
            "best_base_mean_pct": top_mean,
            "all_best_base_mean_pct": all_mean,
            "value_uplift_pct": (
                float(top_mean - all_mean)
                if top_mean is not None and all_mean is not None
                else None
            ),
            "positive_mean_days": int(positive_days),
            "by_day": top_by_day,
        },
    }


def evaluate(
    fit_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_path)
    history_scan = pd.read_parquet(history_scan_path)
    fit_days = set(fit_positions["trading_day"].astype(str))
    fit_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(fit_days)
    ].copy()
    fit_regime = build_market_regime(fit_scan)
    fit = attach_market_regime(
        add_shadow_features(fit_positions),
        fit_regime,
    )
    fit = add_fixed20_labels(fit, fit_scan)

    full = train(
        fit,
        exclude_clock=False,
        seed=MODEL_SEED,
    )
    no_clock = train(
        fit,
        exclude_clock=True,
        seed=NO_CLOCK_SEED,
    )

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
            "request 196A requires complete Request-178 shards"
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
            f"request 196A expected {FRESH_DAYS}, found {found}"
        )

    fresh_regime = build_market_regime(fresh_scan)
    fresh = attach_market_regime(
        add_shadow_features(fresh_positions),
        fresh_regime,
    )
    fresh = add_fixed20_labels(fresh, fresh_scan)
    fresh["fixed20_full_score"] = predict(fresh, full)
    fresh["fixed20_no_clock_score"] = predict(
        fresh,
        no_clock,
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "integrity_audit_only": True,
        "fixed_horizon_minutes": FIXED_HORIZON_MINUTES,
        "winner_threshold_pct": WINNER_THRESHOLD_PCT,
        "training_partition": "request171_fit_only",
        "full_columns": len(full.columns),
        "no_clock_columns": len(no_clock.columns),
        "metrics": metrics(fresh),
        "promotion_eligible": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.history_scan,
        args.fresh_dir,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
