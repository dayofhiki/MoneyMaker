"""Request290: second-resolution crack potential and recurrent entry value.

This development-only experiment follows Request289's finding that:
1) the existing selector does not enrich +10% future excursions, and
2) the frozen exit often leaves before large moves mature.

Request290 changes the research target.  It does not optimize the frozen
10-minute policy.  It asks two causal questions on May5-8 only:

A. Can the first causal 2% pullback state rank the next 60 minutes of explosive
   upside when completed one-second context is added?
B. Starting at that pullback, can a model re-evaluate every five seconds and
   learn whether entering now is better than waiting five more seconds?

Future one-second paths are labels only.  Every feature at decision time t uses
only completed seconds ending at t-1 second.  The recurrent policy is evaluated
out-of-fold by trading day and is capped at 30 seconds of waiting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from .causal_minute_controller_rebuild import causal_execution_scan
from .config import load_settings
from .downstream_aligned_candidate import attach_downstream_target
from .full_hot_fixed_policy_value import attach_fixed_value
from .joint_pullback_admission import (
    FIXED_MAX_HOLD_MINUTES,
    FIXED_STOP_LOSS_PCT,
    FIXED_TRAIL_PCT,
    feature_x,
)
from .lagged_minute_context import CROSSFIT_DAYS
from .massive_client import MassiveClient
from .pullback_trailing_exit import (
    apply_exit_rule,
    build_pullback_episodes,
)
from .rich_post_hot_state import (
    CURRENT_SECOND_FEATURES,
    SECOND_DYNAMIC_FEATURES,
)
from .rich_second_position_value import rich_second_features
from .second_path_attention_probe import (
    SECOND_CACHE_DIR,
    _second_frame,
    second_path_features,
)
from .two_stage_risk_veto import (
    attach_ticker_history,
    prepare_pullback_states,
)

REQUEST_ID = 290
SECOND_MS = 1_000
HORIZON_MS = 60 * 60_000
RECHECK_STEP_S = 5
MAX_WAIT_S = 30
RECHECK_OFFSETS_S = tuple(range(0, MAX_WAIT_S + 1, RECHECK_STEP_S))
MAX_EXECUTION_LAG_S = 3
TOP_FRACTION = 0.20
EXPLOSIVE_LEVELS = (5.0, 10.0, 20.0)

MIN_CRACK_ROWS = 250
MIN_WAIT_ROWS = 1_000
MIN_SECOND_COVERAGE = 0.80

PATH_FEATURES = (
    "elapsed_since_pullback_s",
    "last_completed_close_vs_pullback_pct",
    "post_pullback_min_close_vs_pullback_pct",
    "post_pullback_max_close_vs_pullback_pct",
    "post_pullback_drawdown_from_high_pct",
    "seconds_since_post_pullback_low",
    "seconds_since_post_pullback_high",
)


@dataclass(frozen=True)
class FittedRegressor:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    offset: float
    target_transform: str


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    left = _numeric(left)
    right = _numeric(right)
    valid = left.notna() & right.notna()
    if int(valid.sum()) < 20:
        return None
    value = left.loc[valid].corr(right.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _safe_auc(target: pd.Series, score: pd.Series) -> float | None:
    target = _numeric(target)
    score = _numeric(score)
    valid = target.notna() & score.notna()
    y = target.loc[valid].astype(int)
    s = score.loc[valid].astype(float)
    if len(y) < 20 or y.nunique() != 2:
        return None
    return float(roc_auc_score(y, s))


def _usable(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    return tuple(
        name
        for name in dict.fromkeys(columns)
        if name in frame and _numeric(frame[name]).notna().any()
    )


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    day = frame.trading_day.astype(str)
    counts = day.value_counts()
    return day.map(lambda value: 1.0 / float(counts[value])).to_numpy(float)


def _fit_regressor(
    train: pd.DataFrame,
    columns: tuple[str, ...],
    target_column: str,
    seed: int,
    *,
    positive_log_target: bool,
) -> FittedRegressor:
    target = _numeric(train[target_column])
    valid = target.notna()
    fit = train.loc[valid].copy()
    y = target.loc[valid].to_numpy(float)
    if len(fit) < 80:
        raise ValueError(
            f"Request290 insufficient rows for {target_column}: {len(fit)}"
        )

    if positive_log_target:
        y_fit = np.log1p(np.maximum(y, 0.0))
        transform = "log1p_nonnegative"
    else:
        low, high = np.quantile(y, [0.01, 0.99])
        y_fit = np.clip(y, low, high)
        transform = "winsorized_raw"

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=seed,
    )
    x = feature_x(fit, columns)
    weights = _day_weights(fit)
    model.fit(x, y_fit, sample_weight=weights)

    raw_train = model.predict(x)
    if positive_log_target:
        train_pred = np.expm1(raw_train)
    else:
        train_pred = raw_train
    offset = float(np.mean(y - train_pred))
    return FittedRegressor(
        model=model,
        columns=columns,
        offset=offset,
        target_transform=transform,
    )


def _predict(frame: pd.DataFrame, fitted: FittedRegressor) -> np.ndarray:
    raw = fitted.model.predict(feature_x(frame, fitted.columns))
    if fitted.target_transform == "log1p_nonnegative":
        pred = np.expm1(raw)
    else:
        pred = raw
    return pred + fitted.offset


def _execution_reference(
    seconds: pd.DataFrame,
    decision_t: int,
    *,
    fallback_price: float | None = None,
    allow_fallback: bool = False,
) -> tuple[int, float, float] | None:
    """First second-bar open available at/after a decision timestamp.

    The returned lag is diagnostic.  For h0 only, the frozen minute-open entry
    may be supplied as a fallback so the experiment stays anchored to the
    existing causal entry contract.
    """
    if not seconds.empty:
        times = seconds["t"].to_numpy(dtype=np.int64, copy=False)
        index = int(np.searchsorted(times, int(decision_t), side="left"))
        if index < len(seconds):
            row = seconds.iloc[index]
            t = int(row["t"])
            lag_s = float((t - int(decision_t)) / SECOND_MS)
            price = pd.to_numeric(
                pd.Series([row.get("o")]),
                errors="coerce",
            ).iloc[0]
            if (
                pd.notna(price)
                and float(price) > 0
                and lag_s <= MAX_EXECUTION_LAG_S
            ):
                return t, float(price), lag_s
    if allow_fallback and fallback_price is not None and fallback_price > 0:
        return int(decision_t), float(fallback_price), 0.0
    return None


def _future_excursion(
    seconds: pd.DataFrame,
    entry_t: int,
    entry_price: float,
) -> dict[str, float | bool | int | None]:
    deadline = int(entry_t) + HORIZON_MS
    if seconds.empty or entry_price <= 0:
        return {
            "complete": False,
            "mfe_close_pct": None,
            "mfe_high_pct": None,
            "mae_low_pct": None,
            "future_rows": 0,
        }

    after_deadline = seconds.loc[seconds.t.ge(deadline)]
    complete = bool(len(after_deadline))
    future = seconds.loc[
        seconds.t.ge(int(entry_t)) & seconds.t.le(deadline)
    ].copy()
    if not complete or future.empty:
        return {
            "complete": complete,
            "mfe_close_pct": None,
            "mfe_high_pct": None,
            "mae_low_pct": None,
            "future_rows": int(len(future)),
        }

    close = _numeric(future["c"])
    high = _numeric(future["h"])
    low = _numeric(future["l"])
    return {
        "complete": True,
        "mfe_close_pct": float(
            max(0.0, (float(close.max()) / entry_price - 1.0) * 100.0)
        ),
        "mfe_high_pct": float(
            max(0.0, (float(high.max()) / entry_price - 1.0) * 100.0)
        ),
        "mae_low_pct": float(
            min(0.0, (float(low.min()) / entry_price - 1.0) * 100.0)
        ),
        "future_rows": int(len(future)),
    }


def _completed_second_features(
    seconds: pd.DataFrame,
    decision_t: int,
) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(second_path_features(seconds, int(decision_t)))
    features.update(rich_second_features(seconds, int(decision_t)))
    return {
        f"current_{name}": features.get(name, np.nan)
        for name in SECOND_DYNAMIC_FEATURES
    }


def _path_features(
    seconds: pd.DataFrame,
    pullback_t: int,
    decision_t: int,
    pullback_price: float,
) -> dict[str, float]:
    completed = seconds.loc[
        seconds.t.ge(int(pullback_t))
        & seconds.t.le(int(decision_t) - SECOND_MS)
    ].copy()
    result = {
        "elapsed_since_pullback_s": float(
            (int(decision_t) - int(pullback_t)) / SECOND_MS
        ),
        "last_completed_close_vs_pullback_pct": np.nan,
        "post_pullback_min_close_vs_pullback_pct": np.nan,
        "post_pullback_max_close_vs_pullback_pct": np.nan,
        "post_pullback_drawdown_from_high_pct": np.nan,
        "seconds_since_post_pullback_low": np.nan,
        "seconds_since_post_pullback_high": np.nan,
    }
    if completed.empty or pullback_price <= 0:
        return result

    close = _numeric(completed["c"]).dropna()
    if close.empty:
        return result

    last_close = float(close.iloc[-1])
    min_index = int(close.idxmin())
    max_index = int(close.idxmax())
    min_close = float(close.loc[min_index])
    max_close = float(close.loc[max_index])
    min_t = int(completed.loc[min_index, "t"])
    max_t = int(completed.loc[max_index, "t"])

    result.update(
        {
            "last_completed_close_vs_pullback_pct": float(
                (last_close / pullback_price - 1.0) * 100.0
            ),
            "post_pullback_min_close_vs_pullback_pct": float(
                (min_close / pullback_price - 1.0) * 100.0
            ),
            "post_pullback_max_close_vs_pullback_pct": float(
                (max_close / pullback_price - 1.0) * 100.0
            ),
            "post_pullback_drawdown_from_high_pct": float(
                (last_close / max_close - 1.0) * 100.0
                if max_close > 0
                else np.nan
            ),
            "seconds_since_post_pullback_low": float(
                max(int(decision_t) - SECOND_MS - min_t, 0) / SECOND_MS
            ),
            "seconds_since_post_pullback_high": float(
                max(int(decision_t) - SECOND_MS - max_t, 0) / SECOND_MS
            ),
        }
    )
    return result


def build_second_dataset(
    base_states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR / "request290",
        request_interval_seconds=0.02,
    )

    h0_records: list[dict] = []
    sequential_records: list[dict] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}

    for raw in base_states.to_dict("records"):
        day = str(raw["trading_day"])
        ticker = str(raw["ticker"]).upper()
        key = (day, ticker)
        if key not in cache:
            payload = client.second_bars_range(
                ticker,
                date.fromisoformat(day),
                date.fromisoformat(day),
                adjusted=False,
            )
            cache[key] = _second_frame(payload)
        seconds = cache[key]

        pullback_t = int(raw["state_t"])
        pullback_price = float(raw["entry_price"])

        h0 = dict(raw)
        h0.update(_completed_second_features(seconds, pullback_t))
        h0.update(_path_features(
            seconds,
            pullback_t,
            pullback_t,
            pullback_price,
        ))
        h0_target = _future_excursion(
            seconds,
            pullback_t,
            pullback_price,
        )
        h0["crack_complete_60m"] = bool(h0_target["complete"])
        h0["crack_mfe_close_60m_pct"] = h0_target["mfe_close_pct"]
        h0["crack_mfe_high_60m_pct"] = h0_target["mfe_high_pct"]
        h0["crack_mae_low_60m_pct"] = h0_target["mae_low_pct"]
        h0_records.append(h0)

        episode_rows: list[dict] = []
        for offset_s in RECHECK_OFFSETS_S:
            decision_t = pullback_t + int(offset_s) * SECOND_MS
            execution = _execution_reference(
                seconds,
                decision_t,
                fallback_price=pullback_price,
                allow_fallback=(offset_s == 0),
            )
            if execution is None:
                continue
            fill_t, fill_price, execution_lag_s = execution
            target = _future_excursion(seconds, fill_t, fill_price)

            record = {
                key_name: value
                for key_name, value in raw.items()
            }
            record["decision_t"] = int(decision_t)
            record["recheck_offset_s"] = int(offset_s)
            record["execution_t"] = int(fill_t)
            record["execution_price"] = float(fill_price)
            record["execution_lag_s"] = float(execution_lag_s)
            record.update(_completed_second_features(seconds, decision_t))
            record.update(_path_features(
                seconds,
                pullback_t,
                decision_t,
                pullback_price,
            ))
            record["enter_now_complete_60m"] = bool(target["complete"])
            record["enter_now_mfe_close_60m_pct"] = target["mfe_close_pct"]
            record["enter_now_mfe_high_60m_pct"] = target["mfe_high_pct"]
            record["enter_now_mae_low_60m_pct"] = target["mae_low_pct"]
            episode_rows.append(record)

        episode = pd.DataFrame(episode_rows)
        if episode.empty:
            continue
        episode = episode.sort_values("recheck_offset_s")
        next_value = _numeric(
            episode["enter_now_mfe_close_60m_pct"]
        ).shift(-1)
        current_value = _numeric(
            episode["enter_now_mfe_close_60m_pct"]
        )
        next_offset = _numeric(episode["recheck_offset_s"]).shift(-1)
        expected_next = _numeric(episode["recheck_offset_s"]) + RECHECK_STEP_S
        valid_next = next_offset.eq(expected_next)
        episode["wait5_mfe_close_60m_pct"] = next_value.where(valid_next)
        episode["wait5_advantage_pct"] = (
            episode["wait5_mfe_close_60m_pct"] - current_value
        )
        sequential_records.extend(episode.to_dict("records"))

    h0_frame = pd.DataFrame(h0_records)
    sequential = pd.DataFrame(sequential_records)
    second_cols = [
        name
        for name in CURRENT_SECOND_FEATURES
        if name in h0_frame
    ]
    coverage = (
        float(h0_frame.loc[:, second_cols].notna().any(axis=1).mean())
        if second_cols
        else 0.0
    )
    audit = {
        "ticker_days": int(len(cache)),
        "h0_rows": int(len(h0_frame)),
        "sequential_rows": int(len(sequential)),
        "h0_any_second_feature_coverage": coverage,
        "api_stats": client.stats.to_dict(),
        "execution_lag": {
            "sequential_rows_with_reference": int(len(sequential)),
            "mean_seconds": (
                float(_numeric(sequential.execution_lag_s).mean())
                if len(sequential)
                else None
            ),
            "p95_seconds": (
                float(_numeric(sequential.execution_lag_s).quantile(0.95))
                if len(sequential)
                else None
            ),
        },
    }
    return h0_frame, sequential, audit


def crossfit_crack(
    h0: pd.DataFrame,
    base_columns: tuple[str, ...],
    rich_columns: tuple[str, ...],
) -> pd.DataFrame:
    parts = []
    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train = h0.loc[
            ~h0.trading_day.astype(str).eq(held_day)
            & h0.crack_complete_60m.astype(bool)
            & _numeric(h0.crack_mfe_close_60m_pct).notna()
        ].copy()
        held = h0.loc[
            h0.trading_day.astype(str).eq(held_day)
            & h0.crack_complete_60m.astype(bool)
            & _numeric(h0.crack_mfe_close_60m_pct).notna()
        ].copy()

        base = _fit_regressor(
            train,
            base_columns,
            "crack_mfe_close_60m_pct",
            20262900 + fold_index * 10,
            positive_log_target=True,
        )
        rich = _fit_regressor(
            train,
            rich_columns,
            "crack_mfe_close_60m_pct",
            20262901 + fold_index * 10,
            positive_log_target=True,
        )
        held["base_crack_score_pct"] = _predict(held, base)
        held["rich_crack_score_pct"] = _predict(held, rich)
        held["held_out_day"] = held_day
        parts.append(held)
    return pd.concat(parts, ignore_index=True)


def _ranking_report(
    oof: pd.DataFrame,
    score_column: str,
) -> dict:
    target = _numeric(oof.crack_mfe_close_60m_pct)
    score = _numeric(oof[score_column])
    valid = target.notna() & score.notna()
    work = oof.loc[valid].copy()
    target = _numeric(work.crack_mfe_close_60m_pct)
    score = _numeric(work[score_column])

    selected_parts = []
    for _, part in work.groupby(
        work.trading_day.astype(str),
        sort=True,
    ):
        cutoff = float(
            _numeric(part[score_column]).quantile(1.0 - TOP_FRACTION)
        )
        selected_parts.append(
            part.loc[_numeric(part[score_column]).ge(cutoff)].copy()
        )
    selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else work.iloc[0:0].copy()
    )

    result = {
        "rows": int(len(work)),
        "spearman": _safe_spearman(target, score),
        "population_mean_mfe_close_pct": (
            float(target.mean()) if len(target) else None
        ),
        "top20_rows": int(len(selected)),
        "top20_mean_mfe_close_pct": (
            float(_numeric(selected.crack_mfe_close_60m_pct).mean())
            if len(selected)
            else None
        ),
        "top20_median_mfe_close_pct": (
            float(_numeric(selected.crack_mfe_close_60m_pct).median())
            if len(selected)
            else None
        ),
        "levels": {},
    }
    for threshold in EXPLOSIVE_LEVELS:
        tag = int(threshold)
        population_rate = (
            float(target.ge(threshold).mean())
            if len(target)
            else None
        )
        selected_target = _numeric(
            selected.crack_mfe_close_60m_pct
        )
        selected_rate = (
            float(selected_target.ge(threshold).mean())
            if len(selected_target)
            else None
        )
        result["levels"][f"plus{tag}"] = {
            "population_rate": population_rate,
            "top20_rate": selected_rate,
            "rate_ratio": (
                float(selected_rate / population_rate)
                if population_rate is not None
                and selected_rate is not None
                and population_rate > 0
                else None
            ),
            "auc": _safe_auc(
                target.ge(threshold).astype(int),
                score,
            ),
        }
    return result


def crossfit_wait(
    sequential: pd.DataFrame,
    base_columns: tuple[str, ...],
    rich_columns: tuple[str, ...],
) -> pd.DataFrame:
    trainable = sequential.loc[
        _numeric(sequential.wait5_advantage_pct).notna()
    ].copy()
    parts = []
    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train = trainable.loc[
            ~trainable.trading_day.astype(str).eq(held_day)
        ].copy()
        held = sequential.loc[
            sequential.trading_day.astype(str).eq(held_day)
        ].copy()

        base = _fit_regressor(
            train,
            base_columns,
            "wait5_advantage_pct",
            20263000 + fold_index * 10,
            positive_log_target=False,
        )
        rich = _fit_regressor(
            train,
            rich_columns,
            "wait5_advantage_pct",
            20263001 + fold_index * 10,
            positive_log_target=False,
        )
        held["base_predicted_wait5_advantage_pct"] = _predict(
            held,
            base,
        )
        held["rich_predicted_wait5_advantage_pct"] = _predict(
            held,
            rich,
        )
        held["held_out_day"] = held_day
        parts.append(held)
    return pd.concat(parts, ignore_index=True)


def _wait_prediction_report(
    oof: pd.DataFrame,
    score_column: str,
) -> dict:
    target = _numeric(oof.wait5_advantage_pct)
    score = _numeric(oof[score_column])
    valid = target.notna() & score.notna()
    actual = target.loc[valid]
    predicted = score.loc[valid]
    sign = actual.gt(0).astype(int)
    predicted_wait = predicted.gt(0)

    return {
        "rows": int(valid.sum()),
        "spearman": _safe_spearman(actual, predicted),
        "wait_beneficial_rate": float(sign.mean()) if len(sign) else None,
        "sign_auc": _safe_auc(sign, predicted),
        "predicted_wait_rate": (
            float(predicted_wait.mean()) if len(predicted_wait) else None
        ),
        "actual_mean_advantage_when_predicted_wait_pct": (
            float(actual.loc[predicted_wait].mean())
            if int(predicted_wait.sum())
            else None
        ),
        "actual_mean_advantage_when_predicted_enter_pct": (
            float(actual.loc[~predicted_wait].mean())
            if int((~predicted_wait).sum())
            else None
        ),
    }


def _policy_decisions(
    oof: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    rows = []
    keys = ["trading_day", "ticker", "hot_t"]
    for _, episode in oof.groupby(keys, sort=False):
        ordered = episode.sort_values("recheck_offset_s")
        chosen = None
        for raw in ordered.to_dict("records"):
            offset = int(raw["recheck_offset_s"])
            score = raw.get(score_column)
            can_wait = (
                offset < MAX_WAIT_S
                and pd.notna(raw.get("wait5_advantage_pct"))
                and pd.notna(score)
            )
            if can_wait and float(score) > 0:
                continue
            chosen = raw
            break
        if chosen is None and len(ordered):
            chosen = ordered.iloc[-1].to_dict()
        if chosen is None:
            continue

        h0 = ordered.loc[ordered.recheck_offset_s.eq(0)]
        if h0.empty:
            continue
        h0 = h0.iloc[0]
        h0_price = float(h0.execution_price)
        chosen_price = float(chosen["execution_price"])
        rows.append(
            {
                "trading_day": str(chosen["trading_day"]),
                "ticker": str(chosen["ticker"]).upper(),
                "hot_t": int(chosen["hot_t"]),
                "chosen_offset_s": int(chosen["recheck_offset_s"]),
                "chosen_price": chosen_price,
                "h0_price": h0_price,
                "entry_price_improvement_pct": float(
                    (h0_price / chosen_price - 1.0) * 100.0
                ),
                "chosen_mfe_close_60m_pct": chosen[
                    "enter_now_mfe_close_60m_pct"
                ],
                "h0_mfe_close_60m_pct": h0[
                    "enter_now_mfe_close_60m_pct"
                ],
                "chosen_mae_low_60m_pct": chosen[
                    "enter_now_mae_low_60m_pct"
                ],
            }
        )
    return pd.DataFrame(rows)


def _policy_report(decisions: pd.DataFrame) -> dict:
    if decisions.empty:
        return {"rows": 0}
    chosen_mfe = _numeric(decisions.chosen_mfe_close_60m_pct)
    h0_mfe = _numeric(decisions.h0_mfe_close_60m_pct)
    improvement = _numeric(decisions.entry_price_improvement_pct)
    valid = chosen_mfe.notna() & h0_mfe.notna()
    result = {
        "rows": int(len(decisions)),
        "complete_mfe_rows": int(valid.sum()),
        "waited_rate": float(
            _numeric(decisions.chosen_offset_s).gt(0).mean()
        ),
        "mean_chosen_offset_s": float(
            _numeric(decisions.chosen_offset_s).mean()
        ),
        "median_chosen_offset_s": float(
            _numeric(decisions.chosen_offset_s).median()
        ),
        "mean_entry_price_improvement_pct": float(improvement.mean()),
        "median_entry_price_improvement_pct": float(improvement.median()),
        "mean_h0_mfe_close_60m_pct": (
            float(h0_mfe.loc[valid].mean()) if int(valid.sum()) else None
        ),
        "mean_chosen_mfe_close_60m_pct": (
            float(chosen_mfe.loc[valid].mean()) if int(valid.sum()) else None
        ),
        "mean_mfe_gain_vs_h0_pp": (
            float((chosen_mfe.loc[valid] - h0_mfe.loc[valid]).mean())
            if int(valid.sum())
            else None
        ),
        "levels": {},
    }
    for threshold in EXPLOSIVE_LEVELS:
        tag = int(threshold)
        result["levels"][f"plus{tag}"] = {
            "h0_rate": (
                float(h0_mfe.loc[valid].ge(threshold).mean())
                if int(valid.sum())
                else None
            ),
            "chosen_rate": (
                float(chosen_mfe.loc[valid].ge(threshold).mean())
                if int(valid.sum())
                else None
            ),
        }
    return result


def _fixed_wait_report(sequential: pd.DataFrame) -> dict:
    report = {}
    h0 = sequential.loc[
        sequential.recheck_offset_s.eq(0)
    ].loc[:, [
        "trading_day",
        "ticker",
        "hot_t",
        "execution_price",
        "enter_now_mfe_close_60m_pct",
    ]].rename(
        columns={
            "execution_price": "h0_price",
            "enter_now_mfe_close_60m_pct": "h0_mfe",
        }
    )
    for offset in (5, 15, 30):
        current = sequential.loc[
            sequential.recheck_offset_s.eq(offset)
        ].copy()
        merged = current.merge(
            h0,
            on=["trading_day", "ticker", "hot_t"],
            how="inner",
            validate="one_to_one",
        )
        price = _numeric(merged.execution_price)
        h0_price = _numeric(merged.h0_price)
        mfe = _numeric(merged.enter_now_mfe_close_60m_pct)
        h0_mfe = _numeric(merged.h0_mfe)
        valid = mfe.notna() & h0_mfe.notna()
        report[str(offset)] = {
            "rows": int(len(merged)),
            "mean_entry_price_improvement_pct": (
                float(((h0_price / price - 1.0) * 100.0).mean())
                if len(merged)
                else None
            ),
            "median_entry_price_improvement_pct": (
                float(((h0_price / price - 1.0) * 100.0).median())
                if len(merged)
                else None
            ),
            "mean_mfe_gain_vs_h0_pp": (
                float((mfe.loc[valid] - h0_mfe.loc[valid]).mean())
                if int(valid.sum())
                else None
            ),
            "plus10_rate": (
                float(mfe.loc[valid].ge(10.0).mean())
                if int(valid.sum())
                else None
            ),
        }
    return report


def evaluate(
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    h0_output: Path,
    sequential_output: Path,
) -> int:
    raw_scan = pd.read_parquet(scan_path)
    raw_scan = raw_scan.loc[
        raw_scan.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    causal_scan = causal_execution_scan(raw_scan)

    first_hot = pd.read_parquet(first_hot_path).drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    first_hot = attach_fixed_value(first_hot, causal_scan)

    episodes = build_pullback_episodes(first_hot, causal_scan)
    frozen_policy = apply_exit_rule(
        episodes,
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name="request290_reference_only",
    )
    labeled = attach_downstream_target(first_hot, frozen_policy)
    candidates, ticker_extra = attach_ticker_history(
        labeled,
        raw_scan,
    )
    base_states, base_columns = prepare_pullback_states(
        candidates,
        ticker_extra,
        raw_scan,
        causal_scan,
        frozen_policy,
    )

    h0, sequential, second_audit = build_second_dataset(base_states)
    second_columns = _usable(
        h0,
        list(CURRENT_SECOND_FEATURES),
    )
    crack_rich_columns = _usable(
        h0,
        [*base_columns, *second_columns],
    )

    crack_oof = crossfit_crack(
        h0,
        base_columns,
        crack_rich_columns,
    )
    crack_base = _ranking_report(
        crack_oof,
        "base_crack_score_pct",
    )
    crack_rich = _ranking_report(
        crack_oof,
        "rich_crack_score_pct",
    )

    wait_base_columns = _usable(
        sequential,
        [*base_columns, "elapsed_since_pullback_s"],
    )
    wait_rich_columns = _usable(
        sequential,
        [
            *base_columns,
            *second_columns,
            *PATH_FEATURES,
        ],
    )
    wait_oof = crossfit_wait(
        sequential,
        wait_base_columns,
        wait_rich_columns,
    )
    wait_base = _wait_prediction_report(
        wait_oof,
        "base_predicted_wait5_advantage_pct",
    )
    wait_rich = _wait_prediction_report(
        wait_oof,
        "rich_predicted_wait5_advantage_pct",
    )

    base_decisions = _policy_decisions(
        wait_oof,
        "base_predicted_wait5_advantage_pct",
    )
    rich_decisions = _policy_decisions(
        wait_oof,
        "rich_predicted_wait5_advantage_pct",
    )
    base_policy = _policy_report(base_decisions)
    rich_policy = _policy_report(rich_decisions)
    fixed_waits = _fixed_wait_report(wait_oof)

    crack_spearman_gain = (
        crack_rich["spearman"] - crack_base["spearman"]
        if crack_rich["spearman"] is not None
        and crack_base["spearman"] is not None
        else None
    )
    plus10_ratio = crack_rich["levels"]["plus10"]["rate_ratio"]
    wait_spearman_gain = (
        wait_rich["spearman"] - wait_base["spearman"]
        if wait_rich["spearman"] is not None
        and wait_base["spearman"] is not None
        else None
    )

    checks = {
        "crack_support": crack_rich["rows"] >= MIN_CRACK_ROWS,
        "wait_support": wait_rich["rows"] >= MIN_WAIT_ROWS,
        "second_coverage": (
            second_audit["h0_any_second_feature_coverage"]
            >= MIN_SECOND_COVERAGE
        ),
        "rich_crack_spearman_positive": (
            crack_rich["spearman"] is not None
            and crack_rich["spearman"] > 0
        ),
        "rich_crack_improves_baseline": (
            crack_spearman_gain is not None
            and crack_spearman_gain > 0
        ),
        "rich_plus10_enrichment": (
            plus10_ratio is not None and plus10_ratio > 1.0
        ),
        "rich_wait_spearman_positive": (
            wait_rich["spearman"] is not None
            and wait_rich["spearman"] > 0
        ),
        "rich_wait_improves_baseline": (
            wait_spearman_gain is not None
            and wait_spearman_gain > 0
        ),
        "recurrent_entry_price_nonworse": (
            rich_policy.get("mean_entry_price_improvement_pct") is not None
            and rich_policy["mean_entry_price_improvement_pct"] >= 0
        ),
        "recurrent_mfe_nonworse": (
            rich_policy.get("mean_mfe_gain_vs_h0_pp") is not None
            and rich_policy["mean_mfe_gain_vs_h0_pp"] >= 0
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "research_target": (
            "explosive 60-minute upside plus recurrent 5-second ENTER/WAIT "
            "value, not frozen-policy trade return"
        ),
        "causal_contract": (
            "features at decision_t use only completed one-second aggregates "
            "ending at decision_t-1s; future one-second paths are labels only"
        ),
        "reference_pullback": "first causal 2% pullback, diagnostic anchor only",
        "second_audit": second_audit,
        "feature_counts": {
            "pullback_baseline": len(base_columns),
            "completed_second_extra": len(second_columns),
            "crack_rich_total": len(crack_rich_columns),
            "wait_base": len(wait_base_columns),
            "wait_rich_total": len(wait_rich_columns),
        },
        "crack_potential": {
            "target": "max future one-second close return over 60 minutes",
            "baseline": crack_base,
            "second_enriched": crack_rich,
            "second_minus_baseline_spearman": crack_spearman_gain,
        },
        "wait5_value": {
            "target": (
                "60m MFE if forced to enter 5s later minus 60m MFE if "
                "entering at the current recheck"
            ),
            "baseline": wait_base,
            "second_enriched": wait_rich,
            "second_minus_baseline_spearman": wait_spearman_gain,
        },
        "recurrent_entry_policy": {
            "rule": (
                "starting at h0, every 5s WAIT iff OOF predicted wait5 "
                "advantage > 0; otherwise ENTER; force entry by 30s"
            ),
            "baseline_policy": base_policy,
            "second_enriched_policy": rich_policy,
            "fixed_wait_diagnostics": fixed_waits,
        },
        "checks": checks,
        "research_gate_pass": bool(all(checks.values())),
        "next_boundary": (
            "if crack enrichment works, combine the crack score with the "
            "second-resolution recurrent entry controller and then learn "
            "continuation/HOLD value on the same second-resolution state"
            if (
                checks["rich_plus10_enrichment"]
                and checks["recurrent_entry_price_nonworse"]
                and checks["recurrent_mfe_nonworse"]
            )
            else (
                "separate the failed component: if crack ranking fails, "
                "redesign the explosive-upside representation; if WAIT value "
                "fails, enrich the sub-minute action state before changing "
                "risk or exit rules"
            )
        ),
        "interpretation": (
            "Second aggregates are historical aggregate bars, not tick trades "
            "or NBBO. MFE is an opportunity label, not a realizable profit "
            "guarantee. No per-episode best wait duration is used by the "
            "learned policy."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    h0_output.parent.mkdir(parents=True, exist_ok=True)
    sequential_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    crack_oof.to_parquet(h0_output, index=False)
    wait_oof.to_parquet(sequential_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h0-output", type=Path, required=True)
    parser.add_argument("--sequential-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
        args.h0_output,
        args.sequential_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
