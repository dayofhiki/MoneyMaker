"""Request291: hierarchical crack setup selection and event-driven entry.

Request290 showed that a first-2%-pullback anchor is the wrong place to ask
both "is this a crack setup?" and "is this the best entry?".  Request291
separates the hierarchy:

1. Setup selector: score the full first-HOT candidate population with broad
   pre-HOT / multi-minute context against 60-minute future minute-open MFE.
2. Entry controller: only for out-of-fold crack-selected candidates, inspect
   every observed active one-second aggregate after HOT and repeatedly choose
   ENTER versus preserving the option to WAIT.

The 2% pullback is retained only as a comparison baseline.  No post-decision
second is a feature.  Future paths are labels only.
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
from .downstream_aligned_candidate import candidate_columns
from .full_hot_fixed_policy_value import xframe
from .lagged_minute_context import CROSSFIT_DAYS
from .massive_client import MassiveClient
from .prehot_context_ablation import usable_context_columns
from .pullback_trailing_exit import build_pullback_episodes
from .rich_post_hot_state import SECOND_DYNAMIC_FEATURES
from .rich_second_position_value import rich_second_features
from .second_path_attention_probe import (
    SECOND_CACHE_DIR,
    _second_frame,
    second_path_features,
)
from .two_stage_risk_veto import attach_ticker_history

REQUEST_ID = 291
MINUTE_MS = 60_000
SECOND_MS = 1_000
CRACK_HORIZON_MS = 60 * MINUTE_MS
ENTRY_OBSERVE_MS = 5 * MINUTE_MS
MAX_ACTIVE_STATES = 300
SETUP_SELECTION_FRACTION = 0.20
EXPLOSIVE_LEVELS = (5.0, 10.0, 20.0)

EVENT_FEATURES = (
    "elapsed_from_hot_s",
    "active_event_index",
    "last_close_vs_hot_pct",
    "running_low_vs_hot_pct",
    "running_high_vs_hot_pct",
    "drawdown_from_active_high_pct",
    "bounce_from_active_low_pct",
    "seconds_since_active_high",
    "seconds_since_active_low",
    "last_active_return_pct",
    "inter_event_gap_s",
)


@dataclass(frozen=True)
class RegressionModel:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    log_target: bool
    offset: float


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_spearman(a: pd.Series, b: pd.Series) -> float | None:
    a = _numeric(a)
    b = _numeric(b)
    valid = a.notna() & b.notna()
    if int(valid.sum()) < 20:
        return None
    value = a.loc[valid].corr(b.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _safe_auc(target: pd.Series, score: pd.Series) -> float | None:
    y = _numeric(target)
    s = _numeric(score)
    valid = y.notna() & s.notna()
    y = y.loc[valid].astype(int)
    s = s.loc[valid].astype(float)
    if len(y) < 20 or y.nunique() != 2:
        return None
    return float(roc_auc_score(y, s))


def _usable(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        name
        for name in dict.fromkeys(columns)
        if name in frame and _numeric(frame[name]).notna().any()
    )


def _episode_day_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = (
        frame.trading_day.astype(str)
        + "|"
        + frame.ticker.astype(str)
        + "|"
        + frame.hot_t.astype(str)
    )
    state_count = keys.map(keys.value_counts()).astype(float)
    episode_day = frame.loc[:, ["trading_day", "ticker", "hot_t"]].drop_duplicates()
    episode_counts = episode_day.trading_day.astype(str).value_counts()
    day_episode_count = frame.trading_day.astype(str).map(episode_counts).astype(float)
    weights = 1.0 / (state_count.to_numpy(float) * day_episode_count.to_numpy(float))
    return weights / float(np.mean(weights))


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    day = frame.trading_day.astype(str)
    counts = day.map(day.value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(float)
    return weights / float(np.mean(weights))


def _fit(
    train: pd.DataFrame,
    columns: tuple[str, ...],
    target_column: str,
    seed: int,
    *,
    log_target: bool,
    episode_weighting: bool = False,
) -> RegressionModel:
    target = _numeric(train[target_column])
    valid = target.notna()
    fit = train.loc[valid].copy()
    y = target.loc[valid].to_numpy(float)
    if len(fit) < 100:
        raise ValueError(
            f"Request291 insufficient {target_column} support: {len(fit)}"
        )
    if log_target:
        transformed = np.log1p(np.maximum(y, 0.0))
    else:
        low, high = np.quantile(y, [0.005, 0.995])
        transformed = np.clip(y, low, high)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=(
            75 if episode_weighting else 40
        ),
        l2_regularization=3.0,
        early_stopping=False,
        random_state=seed,
    )
    weights = (
        _episode_day_weights(fit)
        if episode_weighting
        else _day_weights(fit)
    )
    model.fit(xframe(fit, columns), transformed, sample_weight=weights)
    raw = model.predict(xframe(fit, columns))
    prediction = np.expm1(raw) if log_target else raw
    offset = float(np.mean(y - prediction))
    return RegressionModel(model, columns, log_target, offset)


def _predict(frame: pd.DataFrame, fitted: RegressionModel) -> np.ndarray:
    raw = fitted.model.predict(xframe(frame, fitted.columns))
    prediction = np.expm1(raw) if fitted.log_target else raw
    return prediction + fitted.offset


def _minute_paths(
    causal_scan: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    scan = causal_scan.copy()
    scan["ticker"] = scan.ticker.astype(str).str.upper()
    scan["trading_day"] = scan.trading_day.astype(str)
    return {
        (str(day), str(ticker).upper()): part.sort_values("t").reset_index(drop=True)
        for (day, ticker), part in scan.groupby(
            ["trading_day", "ticker"], sort=False
        )
    }


def _minute_crack_target(
    path: pd.DataFrame,
    hot_t: int,
) -> dict[str, float | bool | None]:
    if path.empty:
        return {"complete": False, "reference_price": None, "mfe_pct": None}
    at_hot = path.loc[path.t.eq(int(hot_t))]
    if at_hot.empty:
        return {"complete": False, "reference_price": None, "mfe_pct": None}
    price = float(at_hot.iloc[0].o)
    deadline = int(hot_t) + CRACK_HORIZON_MS
    complete = bool(path.t.ge(deadline).any())
    future = path.loc[
        path.t.ge(int(hot_t)) & path.t.le(deadline)
    ]
    if not complete or future.empty or price <= 0:
        return {"complete": complete, "reference_price": price, "mfe_pct": None}
    maximum = float(_numeric(future.o).max())
    return {
        "complete": True,
        "reference_price": price,
        "mfe_pct": float(max(0.0, (maximum / price - 1.0) * 100.0)),
    }


def attach_crack_targets(
    candidates: pd.DataFrame,
    causal_scan: pd.DataFrame,
) -> pd.DataFrame:
    lookup = _minute_paths(causal_scan)
    rows = []
    for raw in candidates.to_dict("records"):
        day = str(raw["trading_day"])
        ticker = str(raw["ticker"]).upper()
        hot_t = int(raw["t"])
        target = _minute_crack_target(
            lookup.get((day, ticker), pd.DataFrame(columns=["t", "o"])),
            hot_t,
        )
        item = dict(raw)
        item["hot_t"] = hot_t
        item["hot_reference_price"] = target["reference_price"]
        item["crack_complete_60m"] = bool(target["complete"])
        item["crack_mfe_open_60m_pct"] = target["mfe_pct"]
        rows.append(item)
    return pd.DataFrame(rows)


def crossfit_setup(
    frame: pd.DataFrame,
    base_columns: tuple[str, ...],
    rich_columns: tuple[str, ...],
) -> pd.DataFrame:
    parts = []
    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train = frame.loc[
            ~frame.trading_day.astype(str).eq(held_day)
            & frame.crack_complete_60m.astype(bool)
            & _numeric(frame.crack_mfe_open_60m_pct).notna()
        ].copy()
        held = frame.loc[
            frame.trading_day.astype(str).eq(held_day)
            & frame.crack_complete_60m.astype(bool)
            & _numeric(frame.crack_mfe_open_60m_pct).notna()
        ].copy()

        base = _fit(
            train,
            base_columns,
            "crack_mfe_open_60m_pct",
            20263100 + fold_index * 10,
            log_target=True,
        )
        rich = _fit(
            train,
            rich_columns,
            "crack_mfe_open_60m_pct",
            20263101 + fold_index * 10,
            log_target=True,
        )
        train_score = _predict(train, rich)
        threshold = float(
            np.quantile(train_score, 1.0 - SETUP_SELECTION_FRACTION)
        )
        held["base_setup_score_pct"] = _predict(held, base)
        held["rich_setup_score_pct"] = _predict(held, rich)
        held["setup_threshold"] = threshold
        held["selected_crack"] = held.rich_setup_score_pct.ge(threshold)
        held["held_out_day"] = held_day
        parts.append(held)
    return pd.concat(parts, ignore_index=True)


def setup_report(
    oof: pd.DataFrame,
    score_column: str,
    *,
    selection_column: str | None = None,
) -> dict:
    target = _numeric(oof.crack_mfe_open_60m_pct)
    score = _numeric(oof[score_column])
    valid = target.notna() & score.notna()
    work = oof.loc[valid].copy()
    target = _numeric(work.crack_mfe_open_60m_pct)
    score = _numeric(work[score_column])

    if selection_column is None:
        pieces = []
        for day, part in work.groupby(work.trading_day.astype(str), sort=True):
            train = work.loc[~work.trading_day.astype(str).eq(day)]
            cutoff = float(
                _numeric(train[score_column]).quantile(
                    1.0 - SETUP_SELECTION_FRACTION
                )
            )
            pieces.append(
                part.loc[_numeric(part[score_column]).ge(cutoff)]
            )
        selected = (
            pd.concat(pieces, ignore_index=True)
            if pieces
            else work.iloc[0:0].copy()
        )
    else:
        selected = work.loc[
            work[selection_column].fillna(False).astype(bool)
        ].copy()

    report = {
        "rows": int(len(work)),
        "spearman": _safe_spearman(target, score),
        "population_mean_mfe_pct": float(target.mean()),
        "selected_rows": int(len(selected)),
        "selected_rate": float(len(selected) / len(work)) if len(work) else None,
        "selected_mean_mfe_pct": (
            float(_numeric(selected.crack_mfe_open_60m_pct).mean())
            if len(selected)
            else None
        ),
        "selected_median_mfe_pct": (
            float(_numeric(selected.crack_mfe_open_60m_pct).median())
            if len(selected)
            else None
        ),
        "levels": {},
    }
    for threshold in EXPLOSIVE_LEVELS:
        tag = int(threshold)
        population_rate = float(target.ge(threshold).mean())
        selected_target = _numeric(selected.crack_mfe_open_60m_pct)
        selected_rate = (
            float(selected_target.ge(threshold).mean())
            if len(selected_target)
            else None
        )
        report["levels"][f"plus{tag}"] = {
            "population_rate": population_rate,
            "selected_rate": selected_rate,
            "rate_ratio": (
                float(selected_rate / population_rate)
                if selected_rate is not None and population_rate > 0
                else None
            ),
            "auc": _safe_auc(target.ge(threshold).astype(int), score),
        }
    return report


def _second_features(
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


def _event_features(
    active: pd.DataFrame,
    index: int,
    *,
    hot_t: int,
    hot_price: float,
) -> dict[str, float]:
    observed = active.iloc[: index + 1]
    current = observed.iloc[-1]
    close = _numeric(observed.c).dropna()
    result = {name: np.nan for name in EVENT_FEATURES}
    if close.empty or hot_price <= 0:
        return result

    current_close = float(close.iloc[-1])
    low_index = int(close.idxmin())
    high_index = int(close.idxmax())
    low_close = float(close.loc[low_index])
    high_close = float(close.loc[high_index])
    low_t = int(active.loc[low_index, "t"])
    high_t = int(active.loc[high_index, "t"])

    if len(close) >= 2:
        previous_close = float(close.iloc[-2])
        last_return = (
            (current_close / previous_close - 1.0) * 100.0
            if previous_close > 0
            else np.nan
        )
    else:
        last_return = np.nan

    if index > 0:
        gap_s = float(
            (int(current.t) - int(active.iloc[index - 1].t)) / SECOND_MS
        )
    else:
        gap_s = np.nan

    return {
        "elapsed_from_hot_s": float(
            (int(current.t) + SECOND_MS - int(hot_t)) / SECOND_MS
        ),
        "active_event_index": float(index),
        "last_close_vs_hot_pct": float(
            (current_close / hot_price - 1.0) * 100.0
        ),
        "running_low_vs_hot_pct": float(
            (low_close / hot_price - 1.0) * 100.0
        ),
        "running_high_vs_hot_pct": float(
            (high_close / hot_price - 1.0) * 100.0
        ),
        "drawdown_from_active_high_pct": float(
            (current_close / high_close - 1.0) * 100.0
            if high_close > 0
            else np.nan
        ),
        "bounce_from_active_low_pct": float(
            (current_close / low_close - 1.0) * 100.0
            if low_close > 0
            else np.nan
        ),
        "seconds_since_active_high": float(
            max(int(current.t) - high_t, 0) / SECOND_MS
        ),
        "seconds_since_active_low": float(
            max(int(current.t) - low_t, 0) / SECOND_MS
        ),
        "last_active_return_pct": last_return,
        "inter_event_gap_s": gap_s,
    }


def _entry_value(
    seconds: pd.DataFrame,
    execution_index: int,
) -> tuple[bool, float | None, float | None]:
    if execution_index >= len(seconds):
        return False, None, None
    row = seconds.iloc[execution_index]
    entry_t = int(row.t)
    entry_price = float(row.o)
    if entry_price <= 0:
        return False, None, None
    deadline = entry_t + CRACK_HORIZON_MS
    if not bool(seconds.t.ge(deadline).any()):
        return False, None, None
    future = seconds.iloc[execution_index:]
    future = future.loc[future.t.le(deadline)]
    if future.empty:
        return False, None, None
    maximum = float(_numeric(future.c).max())
    minimum = float(_numeric(future.l).min())
    mfe = float(max(0.0, (maximum / entry_price - 1.0) * 100.0))
    mae = float(min(0.0, (minimum / entry_price - 1.0) * 100.0))
    return True, mfe, mae


def _attach_wait_option(episode: pd.DataFrame) -> pd.DataFrame:
    result = episode.sort_values("decision_t").copy()
    value = _numeric(result.entry_value_60m_pct)
    future_best = (
        value.iloc[::-1].cummax().iloc[::-1].shift(-1)
    )
    result["future_best_entry_value_60m_pct"] = future_best
    result["wait_option_advantage_pct"] = future_best - value
    return result


def build_event_states(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR / "request291",
        request_interval_seconds=0.02,
    )
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    episodes: list[pd.DataFrame] = []

    static_exclude = {
        "crack_mfe_open_60m_pct",
        "crack_complete_60m",
    }

    for raw in selected.to_dict("records"):
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
        hot_t = int(raw["hot_t"])
        hot_price = float(raw["hot_reference_price"])

        active = seconds.loc[
            seconds.t.ge(hot_t)
            & seconds.t.lt(hot_t + ENTRY_OBSERVE_MS)
        ].copy()
        if len(active) < 3:
            continue
        active = active.head(MAX_ACTIVE_STATES).reset_index(drop=True)

        rows = []
        for index in range(len(active) - 1):
            current = active.iloc[index]
            next_row = active.iloc[index + 1]
            decision_t = int(current.t) + SECOND_MS
            execution_t = int(next_row.t)
            execution_price = float(next_row.o)
            global_execution_index = int(
                np.searchsorted(
                    seconds.t.to_numpy(dtype=np.int64, copy=False),
                    execution_t,
                    side="left",
                )
            )
            complete, entry_value, entry_mae = _entry_value(
                seconds,
                global_execution_index,
            )
            if not complete:
                continue

            item = {
                key_name: value
                for key_name, value in raw.items()
                if key_name not in static_exclude
            }
            item["decision_t"] = decision_t
            item["execution_t"] = execution_t
            item["execution_price"] = execution_price
            item["execution_gap_s"] = float(
                (execution_t - decision_t) / SECOND_MS
            )
            item["entry_value_60m_pct"] = entry_value
            item["entry_mae_60m_pct"] = entry_mae
            item.update(_second_features(seconds, decision_t))
            item.update(
                _event_features(
                    active,
                    index,
                    hot_t=hot_t,
                    hot_price=hot_price,
                )
            )
            rows.append(item)

        if rows:
            episode = _attach_wait_option(pd.DataFrame(rows))
            episodes.append(episode)

    states = (
        pd.concat(episodes, ignore_index=True)
        if episodes
        else pd.DataFrame()
    )
    second_columns = [
        name
        for name in states.columns
        if name.startswith("current_")
    ]
    audit = {
        "selected_candidate_rows": int(len(selected)),
        "ticker_days": int(len(cache)),
        "event_state_rows": int(len(states)),
        "event_episodes": (
            int(
                states.loc[:, ["trading_day", "ticker", "hot_t"]]
                .drop_duplicates()
                .shape[0]
            )
            if len(states)
            else 0
        ),
        "second_feature_coverage": (
            float(states.loc[:, second_columns].notna().any(axis=1).mean())
            if len(states) and second_columns
            else 0.0
        ),
        "mean_execution_gap_s": (
            float(_numeric(states.execution_gap_s).mean())
            if len(states)
            else None
        ),
        "p95_execution_gap_s": (
            float(_numeric(states.execution_gap_s).quantile(0.95))
            if len(states)
            else None
        ),
        "api_stats": client.stats.to_dict(),
    }
    return states, audit


def crossfit_entry(
    states: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    trainable = states.loc[
        _numeric(states.wait_option_advantage_pct).notna()
    ].copy()
    parts = []
    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train = trainable.loc[
            ~trainable.trading_day.astype(str).eq(held_day)
        ].copy()
        held = states.loc[
            states.trading_day.astype(str).eq(held_day)
        ].copy()
        fitted = _fit(
            train,
            feature_columns,
            "wait_option_advantage_pct",
            20263200 + fold_index * 10,
            log_target=False,
            episode_weighting=True,
        )
        held["predicted_wait_option_advantage_pct"] = _predict(
            held,
            fitted,
        )
        held["entry_held_out_day"] = held_day
        parts.append(held)
    return pd.concat(parts, ignore_index=True)


def entry_signal_report(oof: pd.DataFrame) -> dict:
    target = _numeric(oof.wait_option_advantage_pct)
    score = _numeric(oof.predicted_wait_option_advantage_pct)
    valid = target.notna() & score.notna()
    actual = target.loc[valid]
    predicted = score.loc[valid]
    beneficial = actual.gt(0).astype(int)
    predicted_wait = predicted.gt(0)
    return {
        "labeled_rows": int(valid.sum()),
        "wait_beneficial_rate": float(beneficial.mean()),
        "spearman": _safe_spearman(actual, predicted),
        "sign_auc": _safe_auc(beneficial, predicted),
        "predicted_wait_rate": float(predicted_wait.mean()),
        "actual_advantage_when_predicted_wait_pct": (
            float(actual.loc[predicted_wait].mean())
            if int(predicted_wait.sum())
            else None
        ),
        "actual_advantage_when_predicted_enter_pct": (
            float(actual.loc[~predicted_wait].mean())
            if int((~predicted_wait).sum())
            else None
        ),
    }


def recurrent_policy(oof: pd.DataFrame) -> pd.DataFrame:
    decisions = []
    for _, episode in oof.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        ordered = episode.sort_values("decision_t")
        if ordered.empty:
            continue
        chosen = None
        for raw in ordered.to_dict("records"):
            score = raw.get("predicted_wait_option_advantage_pct")
            has_future = pd.notna(raw.get("wait_option_advantage_pct"))
            if has_future and pd.notna(score) and float(score) > 0:
                continue
            chosen = raw
            break
        if chosen is None:
            chosen = ordered.iloc[-1].to_dict()
        immediate = ordered.iloc[0]
        immediate_price = float(immediate.execution_price)
        chosen_price = float(chosen["execution_price"])
        decisions.append(
            {
                "trading_day": str(chosen["trading_day"]),
                "ticker": str(chosen["ticker"]).upper(),
                "hot_t": int(chosen["hot_t"]),
                "immediate_decision_t": int(immediate.decision_t),
                "chosen_decision_t": int(chosen["decision_t"]),
                "delay_s": float(
                    (int(chosen["decision_t"]) - int(immediate.decision_t))
                    / SECOND_MS
                ),
                "immediate_price": immediate_price,
                "chosen_price": chosen_price,
                "entry_price_improvement_pct": float(
                    (immediate_price / chosen_price - 1.0) * 100.0
                ),
                "immediate_mfe_60m_pct": immediate.entry_value_60m_pct,
                "chosen_mfe_60m_pct": chosen["entry_value_60m_pct"],
                "chosen_mae_60m_pct": chosen["entry_mae_60m_pct"],
            }
        )
    return pd.DataFrame(decisions)


def policy_report(decisions: pd.DataFrame) -> dict:
    if decisions.empty:
        return {"episodes": 0}
    immediate = _numeric(decisions.immediate_mfe_60m_pct)
    chosen = _numeric(decisions.chosen_mfe_60m_pct)
    improvement = _numeric(decisions.entry_price_improvement_pct)
    valid = immediate.notna() & chosen.notna()
    report = {
        "episodes": int(len(decisions)),
        "waited_rate": float(_numeric(decisions.delay_s).gt(0).mean()),
        "mean_delay_s": float(_numeric(decisions.delay_s).mean()),
        "median_delay_s": float(_numeric(decisions.delay_s).median()),
        "mean_entry_price_improvement_pct": float(improvement.mean()),
        "median_entry_price_improvement_pct": float(improvement.median()),
        "mean_immediate_mfe_pct": float(immediate.loc[valid].mean()),
        "mean_chosen_mfe_pct": float(chosen.loc[valid].mean()),
        "mean_mfe_gain_vs_immediate_pp": float(
            (chosen.loc[valid] - immediate.loc[valid]).mean()
        ),
        "levels": {},
    }
    for threshold in EXPLOSIVE_LEVELS:
        tag = int(threshold)
        report["levels"][f"plus{tag}"] = {
            "immediate_rate": float(
                immediate.loc[valid].ge(threshold).mean()
            ),
            "chosen_rate": float(
                chosen.loc[valid].ge(threshold).mean()
            ),
        }
    return report


def fixed_pullback_comparison(
    first_hot: pd.DataFrame,
    causal_scan: pd.DataFrame,
    selected: pd.DataFrame,
    decisions: pd.DataFrame,
) -> dict:
    episodes = build_pullback_episodes(first_hot, causal_scan)
    chosen_keys = selected.loc[
        selected.selected_crack.fillna(False).astype(bool),
        ["trading_day", "ticker", "hot_t"],
    ].copy()
    chosen_keys["ticker"] = chosen_keys.ticker.astype(str).str.upper()
    episodes["ticker"] = episodes.ticker.astype(str).str.upper()
    fixed = episodes.merge(
        chosen_keys,
        on=["trading_day", "ticker", "hot_t"],
        how="inner",
        validate="one_to_one",
    )
    fixed = fixed.loc[fixed.entered.astype(bool)].copy()
    baseline = decisions.loc[:, [
        "trading_day",
        "ticker",
        "hot_t",
        "immediate_price",
    ]]
    fixed = fixed.merge(
        baseline,
        on=["trading_day", "ticker", "hot_t"],
        how="inner",
        validate="one_to_one",
    )
    if fixed.empty:
        return {"entered_rows": 0}
    improvement = (
        _numeric(fixed.immediate_price)
        / _numeric(fixed.entry_price)
        - 1.0
    ) * 100.0
    return {
        "entered_rows": int(len(fixed)),
        "selected_candidate_coverage": float(
            len(fixed) / len(chosen_keys)
        ) if len(chosen_keys) else 0.0,
        "mean_entry_price_improvement_pct": float(improvement.mean()),
        "median_entry_price_improvement_pct": float(improvement.median()),
    }


def evaluate(
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    setup_output: Path,
    entry_output: Path,
) -> int:
    raw_scan = pd.read_parquet(scan_path)
    raw_scan = raw_scan.loc[
        raw_scan.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    causal_scan = causal_execution_scan(raw_scan)

    first_hot = pd.read_parquet(first_hot_path)
    first_hot = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    first_hot["ticker"] = first_hot.ticker.astype(str).str.upper()

    candidates, ticker_extra = attach_ticker_history(first_hot, raw_scan)
    ticker_extra = usable_context_columns(candidates, ticker_extra)
    base_columns = candidate_columns(candidates)
    rich_columns = _usable(
        candidates,
        [*base_columns, *ticker_extra],
    )
    labeled = attach_crack_targets(candidates, causal_scan)

    setup_oof = crossfit_setup(
        labeled,
        base_columns,
        rich_columns,
    )
    base_setup = setup_report(
        setup_oof,
        "base_setup_score_pct",
    )
    rich_setup = setup_report(
        setup_oof,
        "rich_setup_score_pct",
        selection_column="selected_crack",
    )

    selected = setup_oof.loc[
        setup_oof.selected_crack.fillna(False).astype(bool)
    ].copy()
    states, second_audit = build_event_states(selected)
    second_columns = _usable(
        states,
        [
            name
            for name in states.columns
            if name.startswith("current_")
        ],
    )
    entry_columns = _usable(
        states,
        [
            *rich_columns,
            "rich_setup_score_pct",
            *second_columns,
            *EVENT_FEATURES,
        ],
    )
    entry_oof = crossfit_entry(states, entry_columns)
    entry_signal = entry_signal_report(entry_oof)
    decisions = recurrent_policy(entry_oof)
    policy = policy_report(decisions)
    fixed_2pct = fixed_pullback_comparison(
        first_hot,
        causal_scan,
        setup_oof,
        decisions,
    )

    plus10_ratio = rich_setup["levels"]["plus10"]["rate_ratio"]
    base_plus10_ratio = base_setup["levels"]["plus10"]["rate_ratio"]
    checks = {
        "setup_support": rich_setup["rows"] >= 1000,
        "setup_spearman_positive": (
            rich_setup["spearman"] is not None
            and rich_setup["spearman"] > 0
        ),
        "setup_plus10_enrichment": (
            plus10_ratio is not None and plus10_ratio > 1.0
        ),
        "setup_plus10_improves_base": (
            plus10_ratio is not None
            and base_plus10_ratio is not None
            and plus10_ratio > base_plus10_ratio
        ),
        "event_state_support": second_audit["event_state_rows"] >= 5000,
        "second_feature_coverage": second_audit["second_feature_coverage"] >= 0.80,
        "entry_wait_auc": (
            entry_signal["sign_auc"] is not None
            and entry_signal["sign_auc"] > 0.52
        ),
        "entry_price_improves": (
            policy.get("mean_entry_price_improvement_pct") is not None
            and policy["mean_entry_price_improvement_pct"] > 0
        ),
        "entry_mfe_nonlower": (
            policy.get("mean_mfe_gain_vs_immediate_pp") is not None
            and policy["mean_mfe_gain_vs_immediate_pp"] >= 0
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "architecture": {
            "setup": (
                "full first-HOT population, broad pre-HOT plus multi-minute "
                "ticker context, 60m future causal minute-open MFE target"
            ),
            "entry": (
                "OOF crack-selected candidates only; decision after each "
                "completed active one-second aggregate; execution reference "
                "is the next active second-bar open"
            ),
            "fixed_2pct_pullback": "comparison baseline only",
        },
        "feature_counts": {
            "setup_base": len(base_columns),
            "setup_rich": len(rich_columns),
            "entry": len(entry_columns),
            "entry_second": len(second_columns),
        },
        "setup_selector": {
            "baseline": base_setup,
            "multi_minute_rich": rich_setup,
        },
        "event_entry": {
            "audit": second_audit,
            "signal": entry_signal,
            "policy": policy,
            "fixed_2pct_comparison": fixed_2pct,
        },
        "checks": checks,
        "research_gate_pass": bool(all(checks.values())),
        "next_boundary": (
            "if both setup enrichment and entry timing work, freeze the "
            "hierarchy and build second-resolution HOLD/EXIT continuation; "
            "otherwise isolate the failed stage before adding exit logic"
        ),
        "interpretation": (
            "Future MFE and future-best entry option are research labels only. "
            "The live policy never sees future prices. Historical one-second "
            "aggregates are trade aggregates, not NBBO quotes or guaranteed fills."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    setup_output.parent.mkdir(parents=True, exist_ok=True)
    entry_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    setup_oof.to_parquet(setup_output, index=False)
    entry_oof.to_parquet(entry_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setup-output", type=Path, required=True)
    parser.add_argument("--entry-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
        args.setup_output,
        args.entry_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
