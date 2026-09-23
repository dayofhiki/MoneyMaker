"""Request 142: selected-HOT position value observability.

Conditional on request 140B passing. Builds causal post-entry position states
from already-opened dates and tests short-horizon continuation plus residual
multi-minute option-value observability. This module does not execute a policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import MINUTE_MS
from .config import load_settings, require_flatfile_credentials
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS, modeled_buy_fill, modeled_sell_fill
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hot_economic_opportunity import CAL_DAYS, EVAL_DAYS, FIT_DAYS, _fit_models
from .massive_client import MassiveClient
from .second_path_attention_probe import (
    BASELINE_FEATURES,
    SECOND_CACHE_DIR,
    SECOND_FEATURES,
    _annotate_scan,
    _second_frame,
    second_path_features,
)

REQUEST_ID = 142
MAX_HOLD_MINUTES = 30
BASE_SCENARIO = next(item for item in DEFAULT_EXECUTION_SCENARIOS if item.name == "base")
PATH_FEATURES = (
    "minutes_held",
    "log_entry_price",
    "log_current_open",
    "entry_to_current_open_pct",
    "entry_to_current_close_pct",
    "running_max_return_pct",
    "running_min_return_pct",
    "drawdown_from_peak_pct",
    "recovery_from_trough_pct",
    "minutes_since_open",
    "minutes_to_close",
)
MODEL_FEATURES = tuple(dict.fromkeys([*BASELINE_FEATURES, *SECOND_FEATURES, *PATH_FEATURES]))
HOLD_SEED = 20261052
OPTION_SEED = 20261053


@dataclass(frozen=True)
class HoldModel:
    model: HistGradientBoostingClassifier
    platt: LogisticRegression


@dataclass(frozen=True)
class OptionModel:
    model: HistGradientBoostingRegressor
    offset: float
    winsor_low: float
    winsor_high: float


def _positive(value: object) -> bool:
    return bool(pd.notna(value) and np.isfinite(float(value)) and float(value) > 0)


def _base_return(entry_open: float, exit_open: float) -> float:
    if not (_positive(entry_open) and _positive(exit_open)):
        return np.nan
    buy = modeled_buy_fill(float(entry_open), BASE_SCENARIO)
    sell = modeled_sell_fill(float(exit_open), BASE_SCENARIO)
    sell *= 1.0 - BASE_SCENARIO.sell_fee_bps / 10_000.0
    return (sell / buy - 1.0) * 100.0


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reindex(columns=MODEL_FEATURES).apply(pd.to_numeric, errors="coerce")


def _open_map(scan: pd.DataFrame) -> dict[tuple[str, str, int], float]:
    result: dict[tuple[str, str, int], float] = {}
    for row in scan.itertuples(index=False):
        opening = pd.to_numeric(pd.Series([getattr(row, "o")]), errors="coerce").iloc[0]
        if not _positive(opening):
            continue
        actual_open_t = int(row.t) - MINUTE_MS
        result[(str(row.trading_day), str(row.ticker).upper(), actual_open_t)] = float(opening)
    return result


def _position_scan_features(scan: pd.DataFrame) -> pd.DataFrame:
    annotated = _annotate_scan(scan)
    annotated["attention_rank"] = annotated.groupby(
        ["trading_day", "t"], sort=False
    )["attention_score"].rank(method="first", ascending=False)
    return annotated


def _minute_lookup(scan: pd.DataFrame) -> dict[tuple[str, str, int], dict[str, object]]:
    annotated = _position_scan_features(scan)
    return {
        (str(row.trading_day), str(row.ticker).upper(), int(row.t)): row._asdict()
        for row in annotated.itertuples(index=False)
    }


def _ticker_completed_rows(
    scan: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    annotated = _position_scan_features(scan)
    result: dict[tuple[str, str], pd.DataFrame] = {
    for (day, ticker), group in annotated.groupby(["trading_day", "ticker"], sort=False):
        result[(str(day), str(ticker).upper())] = group.sort_values("t", kind="stable")
    return result


def _session_clock(timestamp_ms: int) -> tuple[float, float]:
    local = pd.Timestamp(timestamp_ms, unit="ms", tz="UTC").tz_convert("America/New_York")
    minute = local.hour * 60 + local.minute
    since = float(minute - (9 * 60 + 30))
    return since, float(390 - since)


def build_position_rows(
    anchors: pd.DataFrame,
    scan: pd.DataFrame,
    second_client: MassiveClient,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build causal minute-by-minute post-entry states for first-HOT anchors."""

    opens = _open_map(scan)
    open_groups: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for (day, ticker, timestamp), price in opens.items():
        open_groups.setdefault((day, ticker), []).append((timestamp, price))
    for key in open_groups:
        open_groups[key].sort()

    minute_rows = _minute_lookup(scan)
    completed = _ticker_completed_rows(scan)
    second_cache: dict[tuple[str, str], pd.DataFrame] = {}
    records: list[dict[str, object]] = []
    requested_ticker_days = 0
    nonempty_second_days = 0

    for anchor in anchors.sort_values(["trading_day", "t", "ticker"], kind="stable").to_dict("records"):
        day = str(anchor["trading_day"])
        ticker = str(anchor["ticker"]).upper()
        hot_t = int(anchor["t"])
        entry_actual_t = hot_t
        entry_open = opens.get((day, ticker, entry_actual_t), np.nan)
        if not _positive(entry_open):
            continue

        key = (day, ticker)
        if key not in second_cache:
            requested_ticker_days += 1
            payload = second_client.second_bars_range(
                ticker,
                date.fromisoformat(day),
                date.fromisoformat(day),
                adjusted=False,
            )
            second_cache[key] = _second_frame(payload)
            if not second_cache[key].empty:
                nonempty_second_days += 1
        seconds = second_cache[key]
        ticker_minutes = completed.get(key)
        if ticker_minutes is None or ticker_minutes.empty:
            continue

        cap_t = entry_actual_t + MAX_HOLD_MINUTES * MINUTE_MS
        future_open_pairs = [
            (timestamp, price)
            for timestamp, price in open_groups.get(key, [])
            if entry_actual_t < timestamp <= cap_t
        ]

        for held in range(1, MAX_HOLD_MINUTES):
            state_t = entry_actual_t + held * MINUTE_MS
            state = minute_rows.get((day, ticker, state_t))
            current_open = opens.get((day, ticker, state_t), np.nan)
            if state is None or not _positive(current_open):
                continue

            completed_since_entry = ticker_minutes.loc[
                ticker_minutes["t"].gt(entry_actual_t)
                & ticker_minutes["t"].le(state_t)
            ]
            if completed_since_entry.empty:
                continue

            high = pd.to_numeric(completed_since_entry["h"], errors="coerce")
            low = pd.to_numeric(completed_since_entry["l"], errors="coerce")
            current_close = pd.to_numeric(
                pd.Series([state.get("c")]), errors="coerce"
            ).iloc[0]
            running_high = float(high.max()) if high.notna().any() else np.nan
            running_low = float(low.min()) if low.notna().any() else np.nan

            exit_now = _base_return(float(entry_open), float(current_open))
            next_open = opens.get((day, ticker, state_t + MINUTE_MS), np.nan)
            next_exit = _base_return(float(entry_open), float(next_open))
            hold_advantage = (
                float(next_exit - exit_now)
                if pd.notna(next_exit) and pd.notna(exit_now)
                else np.nan
            )

            later = [
                _base_return(float(entry_open), float(price))
                for timestamp, price in future_open_pairs
                if timestamp > state_t
            ]
            later = [value for value in later if pd.notna(value)]
            best_future = max(later) if later else np.nan
            remaining = (
                float(best_future - exit_now)
                if pd.notna(best_future) and pd.notna(exit_now)
                else np.nan
            )

            record = {
                "trading_day": day,
                "ticker": ticker,
                "hot_t": hot_t,
                "entry_actual_t": entry_actual_t,
                "entry_open": float(entry_open),
                "state_t": state_t,
                "minutes_held": float(held),
                "current_open": float(current_open),
                "exit_now_base_return_pct": exit_now,
                "next_minute_base_return_pct": next_exit,
                "hold_advantage_1m_pct": hold_advantage,
                "best_future_base_return_pct": best_future,
                "remaining_option_value_pct": remaining,
            }
            for column in BASELINE_FEATURES:
                record[column] = state.get(column)

            record.update(second_path_features(seconds, state_t))
            record["log_entry_price"] = float(np.log(entry_open))
            record["log_current_open"] = float(np.log(current_open))
            record["entry_to_current_open_pct"] = (
                float(current_open / entry_open - 1.0) * 100.0
            )
            record["entry_to_current_close_pct"] = (
                float(current_close / entry_open - 1.0) * 100.0
                if _positive(current_close)
                else np.nan
            )
            record["running_max_return_pct"] = (
                float(running_high / entry_open - 1.0) * 100.0
                if _positive(running_high)
                else np.nan
            )
            record["running_min_return_pct"] = (
                float(running_low / entry_open - 1.0) * 100.0
                if _positive(running_low)
                else np.nan
            )
            record["drawdown_from_peak_pct"] = (
                float(current_open / running_high - 1.0) * 100.0
                if _positive(running_high)
                else np.nan
            )
            record["recovery_from_trough_pct"] = (
                float(current_open / running_low - 1.0) * 100.0
                if _positive(running_low)
                else np.nan
            )
            since_open, to_close = _session_clock(state_t)
            record["minutes_since_open"] = since_open
            record["minutes_to_close"] = to_close
            records.append(record)

    frame = pd.DataFrame(records)
    second_coverage = (
        float(
            pd.to_numeric(frame.get("active_seconds_60"), errors="coerce")
            .fillna(0)
            .gt(0)
            .mean()
        )
        if len(frame)
        else None
    )
    return frame, {
        "position_rows": int(len(frame)),
        "position_ticker_day_requests": requested_ticker_days,
        "nonempty_second_ticker_days": nonempty_second_days,
        "second_feature_row_coverage": second_coverage,
    }


def fit_minute_baselines(frame: pd.DataFrame) -> dict[int, float]:
    target = pd.to_numeric(frame["remaining_option_value_pct"], errors="coerce")
    held = pd.to_numeric(frame["minutes_held"], errors="coerce")
    valid = target.notna() & held.notna()
    grouped = pd.DataFrame(
        {"minute": held.loc[valid].astype(int), "target": target.loc[valid]}
    ).groupby("minute")["target"].mean()
    return {int(index): float(value) for index, value in grouped.items()}


def apply_excess_target(frame: pd.DataFrame, baselines: dict[int, float]) -> pd.DataFrame:
    result = frame.copy()
    held = pd.to_numeric(result["minutes_held"], errors="coerce")
    result["fit_minute_baseline_pct"] = held.map(
        lambda value: baselines.get(int(value), np.nan) if pd.notna(value) else np.nan
    )
    result["excess_remaining_option_value_pct"] = (
        pd.to_numeric(result["remaining_option_value_pct"], errors="coerce")
        - pd.to_numeric(result["fit_minute_baseline_pct"], errors="coerce")
    )
    return result


def train_hold_model(fit: pd.DataFrame, calibration: pd.DataFrame) -> HoldModel:
    fit = fit.loc[pd.to_numeric(fit["hold_advantage_1m_pct"], errors="coerce").notna()].copy()
    calibration = calibration.loc[
        pd.to_numeric(calibration["hold_advantage_1m_pct"], errors="coerce").notna()
    ].copy()
    y_fit = pd.to_numeric(fit["hold_advantage_1m_pct"], errors="coerce").gt(0).astype(int)
    y_cal = pd.to_numeric(calibration["hold_advantage_1m_pct"], errors="coerce").gt(0).astype(int)
    if len(fit) < 1000 or len(calibration) < 500 or y_fit.nunique() < 2 or y_cal.nunique() < 2:
        raise ValueError("insufficient one-minute continuation support")

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=HOLD_SEED,
    )
    model.fit(_feature_frame(fit), y_fit)

    raw = np.clip(model.predict_proba(_feature_frame(calibration))[:, 1], 1e-6, 1 - 1e-6)
    logits = np.log(raw / (1 - raw)).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=HOLD_SEED,
    )
    platt.fit(logits, y_cal)
    return HoldModel(model=model, platt=platt)


def predict_hold(frame: pd.DataFrame, fitted: HoldModel) -> np.ndarray:
    raw = np.clip(fitted.model.predict_proba(_feature_frame(frame))[:, 1], 1e-6, 1 - 1e-6)
    logits = np.log(raw / (1 - raw)).reshape(-1, 1)
    return fitted.platt.predict_proba(logits)[:, 1]


def train_option_model(fit: pd.DataFrame, calibration: pd.DataFrame) -> OptionModel:
    target = pd.to_numeric(fit["excess_remaining_option_value_pct"], errors="coerce")
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError("insufficient remaining-option support")

    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=OPTION_SEED,
    )
    model.fit(_feature_frame(train), y.clip(lower=low, upper=high))

    cal_target = pd.to_numeric(
        calibration["excess_remaining_option_value_pct"], errors="coerce"
    )
    cal_valid = cal_target.notna()
    if int(cal_valid.sum()) < 500:
        raise ValueError("insufficient remaining-option calibration support")
    prediction = model.predict(_feature_frame(calibration.loc[cal_valid]))
    offset = float(cal_target.loc[cal_valid].mean() - float(np.mean(prediction)))
    return OptionModel(
        model=model,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_option(frame: pd.DataFrame, fitted: OptionModel) -> np.ndarray:
    return fitted.model.predict(_feature_frame(frame)) + fitted.offset


def _safe_auc(actual: pd.Series, score: pd.Series) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    y = y.loc[valid].astype(int)
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y, s.loc[valid].astype(float)))


def _day_balanced(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return None
    daily = pd.DataFrame(
        {"day": frame.loc[valid, "trading_day"].astype(str), "value": values.loc[valid]}
    ).groupby("day")["value"].mean()
    return float(daily.mean()) if len(daily) else None


def evaluate_observability(frame: pd.DataFrame, *, label: str) -> dict[str, object]:
    hold_target = pd.to_numeric(frame["hold_advantage_1m_pct"], errors="coerce")
    hold_score = pd.to_numeric(frame["hold_probability"], errors="coerce")
    hold_valid = hold_target.notna() & hold_score.notna()
    hold_y = hold_target.loc[hold_valid].gt(0).astype(int)
    auc = _safe_auc(hold_y, hold_score.loc[hold_valid])

    semantic_hold = frame.loc[
        hold_valid & hold_score.gt(0.5)
    ].copy()
    hold_mean = (
        float(pd.to_numeric(semantic_hold["hold_advantage_1m_pct"], errors="coerce").mean())
        if len(semantic_hold)
        else None
    )
    hold_day_mean = _day_balanced(semantic_hold, "hold_advantage_1m_pct")

    option_target = pd.to_numeric(
        frame["excess_remaining_option_value_pct"], errors="coerce"
    )
    option_score = pd.to_numeric(frame["predicted_excess_option_value_pct"], errors="coerce")
    option_valid = option_target.notna() & option_score.notna()
    global_spearman = (
        float(option_target.loc[option_valid].corr(option_score.loc[option_valid], method="spearman"))
        if int(option_valid.sum()) >= 20
        else None
    )

    minute_corrs: list[float] = []
    for minute, group in frame.loc[option_valid].groupby("minutes_held", sort=True):
        if len(group) < 20:
            continue
        actual = pd.to_numeric(group["excess_remaining_option_value_pct"], errors="coerce")
        predicted = pd.to_numeric(group["predicted_excess_option_value_pct"], errors="coerce")
        value = actual.corr(predicted, method="spearman")
        if pd.notna(value):
            minute_corrs.append(float(value))

    selected_option = frame.loc[
        option_valid & option_score.gt(0)
    ].copy()
    selected_excess_mean = (
        float(pd.to_numeric(
            selected_option["excess_remaining_option_value_pct"], errors="coerce"
        ).mean())
        if len(selected_option)
        else None
    )
    selected_excess_day_mean = _day_balanced(
        selected_option, "excess_remaining_option_value_pct"
    )

    by_day: dict[str, object] = {}
    daily_support_pass = 0
    for day, day_frame in frame.groupby(frame["trading_day"].astype(str), sort=True):
        day_hold_target = pd.to_numeric(day_frame["hold_advantage_1m_pct"], errors="coerce")
        day_hold_score = pd.to_numeric(day_frame["hold_probability"], errors="coerce")
        day_hold_valid = day_hold_target.notna() & day_hold_score.notna()
        day_hold_y = day_hold_target.loc[day_hold_valid].gt(0).astype(int)
        day_auc = _safe_auc(day_hold_y, day_hold_score.loc[day_hold_valid])
        day_semantic = day_frame.loc[day_hold_valid & day_hold_score.gt(0.5)]
        day_hold_mean = (
            float(pd.to_numeric(day_semantic["hold_advantage_1m_pct"], errors="coerce").mean())
            if len(day_semantic)
            else None
        )

        day_option_target = pd.to_numeric(
            day_frame["excess_remaining_option_value_pct"], errors="coerce"
        )
        day_option_score = pd.to_numeric(
            day_frame["predicted_excess_option_value_pct"], errors="coerce"
        )
        day_option_valid = day_option_target.notna() & day_option_score.notna()
        day_spearman = (
            float(day_option_target.loc[day_option_valid].corr(
                day_option_score.loc[day_option_valid], method="spearman"
            ))
            if int(day_option_valid.sum()) >= 20
            else None
        )
        day_option_selected = day_frame.loc[day_option_valid & day_option_score.gt(0)]
        day_excess = (
            float(pd.to_numeric(
                day_option_selected["excess_remaining_option_value_pct"], errors="coerce"
            ).mean())
            if len(day_option_selected)
            else None
        )

        supported = (
            day_auc is not None
            and day_auc > 0.5
            and day_hold_mean is not None
            and day_hold_mean > 0
            and day_spearman is not None
            and day_spearman > 0
            and day_excess is not None
            and day_excess > 0
        )
        daily_support_pass += int(supported)
        by_day[str(day)] = {
            "rows": int(len(day_frame)),
            "hold_auc": day_auc,
            "semantic_hold_rows": int(len(day_semantic)),
            "semantic_hold_incremental_mean_pct": day_hold_mean,
            "remaining_value_spearman": day_spearman,
            "predicted_excess_positive_rows": int(len(day_option_selected)),
            "predicted_excess_positive_realized_mean_pct": day_excess,
            "all_signs_positive": bool(supported),
        }

    decision_coverage = float(
        pd.to_numeric(frame["exit_now_base_return_pct"], errors="coerce").notna().mean()
    ) if len(frame) else None

    result = {
        "label": label,
        "rows": int(len(frame)),
        "decision_state_coverage": decision_coverage,
        "hold_auc": auc,
        "semantic_hold_rows": int(len(semantic_hold)),
        "semantic_hold_incremental_mean_pct": hold_mean,
        "semantic_hold_day_balanced_incremental_mean_pct": hold_day_mean,
        "remaining_value_spearman": global_spearman,
        "same_minute_spearman_median": (
            float(np.median(minute_corrs)) if minute_corrs else None
        ),
        "same_minute_spearman_positive_fraction": (
            float(np.mean(np.asarray(minute_corrs) > 0)) if minute_corrs else None
        ),
        "predicted_excess_positive_rows": int(len(selected_option)),
        "predicted_excess_positive_realized_mean_pct": selected_excess_mean,
        "predicted_excess_positive_day_balanced_mean_pct": selected_excess_day_mean,
        "daily_all_signs_positive_days": int(daily_support_pass),
        "by_day": by_day,
    }
    result["bridge_pass"] = bool(
        decision_coverage is not None
        and decision_coverage >= 0.90
        and auc is not None
        and auc > 0.50
        and hold_mean is not None
        and hold_mean > 0
        and hold_day_mean is not None
        and hold_day_mean > 0
        and global_spearman is not None
        and global_spearman > 0
        and result["same_minute_spearman_median"] is not None
        and float(result["same_minute_spearman_median"]) > 0
        and selected_excess_day_mean is not None
        and selected_excess_day_mean > 0
        and daily_support_pass >= 4
    )
    return result


def run_probe(
    opportunity_path: Path,
    opportunity_summary_path: Path,
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    second_client: MassiveClient,
) -> tuple[pd.DataFrame, dict[str, object]]:
    opportunity = pd.read_parquet(opportunity_path)
    summary140 = json.loads(
        opportunity_summary_path.read_text(encoding="utf-8")
    )
    if not bool(summary140["evaluation"]["promotion_gate_pass"]):
        raise ValueError("request 142 requires request 140B to pass")

    classifier, _regressor, _offset, threshold, _low, _high = _fit_models(opportunity)
    opportunity = opportunity.copy()
    opportunity["entry_opportunity_probability"] = classifier.predict_proba(
        opportunity[summary140["entry_features"]].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    opportunity["entry_selected_140b"] = (
        opportunity["entry_opportunity_probability"] >= threshold
    )

    days = FIT_DAYS + CAL_DAYS + EVAL_DAYS
    scans: list[pd.DataFrame] = []
    for day_text in days:
        scan, _ = build_flatfile_scan_day(
            store, scan_client, date.fromisoformat(day_text)
        )
        scans.append(scan)
    scan = pd.concat(scans, ignore_index=True)

    path_rows, path_audit = build_position_rows(
        opportunity.loc[:, ["trading_day", "ticker", "t"]],
        scan,
        second_client,
    )
    anchor_flags = opportunity.loc[
        :, ["trading_day", "ticker", "t", "entry_selected_140b"]
    ].rename(columns={"t": "hot_t"})
    path_rows = path_rows.merge(
        anchor_flags,
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="many_to_one",
    )

    fit = path_rows.loc[
        path_rows["trading_day"].astype(str).isin(FIT_DAYS)
    ].copy()
    calibration = path_rows.loc[
        path_rows["trading_day"].astype(str).isin(CAL_DAYS)
    ].copy()
    evaluation = path_rows.loc[
        path_rows["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(calibration, baselines)
    evaluation = apply_excess_target(evaluation, baselines)

    hold_model = train_hold_model(fit, calibration)
    option_model = train_option_model(fit, calibration)
    evaluation["hold_probability"] = predict_hold(evaluation, hold_model)
    evaluation["predicted_excess_option_value_pct"] = predict_option(
        evaluation, option_model
    )

    all_eval = evaluate_observability(evaluation, label="all_first_hot")
    selected_eval = evaluate_observability(
        evaluation.loc[
            evaluation["entry_selected_140b"].fillna(False).astype(bool)
        ].copy(),
        label="request140b_selected",
    )

    final = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "fit_days": FIT_DAYS,
        "calibration_days": CAL_DAYS,
        "evaluation_days": EVAL_DAYS,
        "model_features": MODEL_FEATURES,
        "path_audit": path_audit,
        "option_winsor_low_pct": option_model.winsor_low,
        "option_winsor_high_pct": option_model.winsor_high,
        "option_calibration_offset_pct": option_model.offset,
        "all_first_hot": all_eval,
        "request140b_selected": selected_eval,
        "promotion_gate_pass": bool(selected_eval["bridge_pass"]),
    }
    return evaluation, final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opportunity", type=Path, required=True)
    parser.add_argument("--opportunity-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key), FLATFILE_CACHE_DIR
    )
    scan_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=0.0,
    )
    second_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR,
        request_interval_seconds=0.0,
    )
    output, summary = run_probe(
        args.opportunity,
        args.opportunity_summary,
        store,
        scan_client,
        second_client,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
