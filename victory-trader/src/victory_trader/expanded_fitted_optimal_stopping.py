from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .expanded_opportunity_shared_q import (
    predict_opportunity,
    train_opportunity_model,
)
from .expanded_path_aware_continuation import (
    KEYS,
    MAX_HOLD_MINUTES,
    MINUTE_MS,
    build_continuation_rows,
)
from .expanded_path_transition_continuation import (
    predict_transition_continuation,
    train_transition_model,
    transition_feature_frame,
)
from .expanded_recurrent_path_policy import (
    SEVERE_LOSS_PCT,
    _add_returns,
    _exact_open,
    _first_open_at_or_after,
    _group_state,
    _normalize_state,
    _positive,
    account_replay,
    build_always_hold_comparator,
    build_recurrent_trajectories,
)
from .expanded_remaining_option_observability import _base_return
from .expanded_supply_hurdle_ev import load_fold

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261046


@dataclass(frozen=True)
class MinuteStoppingModel:
    minute: int
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]


def _month_of(frame: pd.DataFrame) -> pd.Series:
    return frame["trading_day"].astype(str).str.slice(0, 7)


def _attach_exit_now_base(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    entry = pd.to_numeric(result["entry_open"], errors="coerce")
    current = pd.to_numeric(result["exit_open"], errors="coerce")
    result["exit_now_base_return_pct"] = [
        _base_return(float(e), float(x))
        if _positive(e) and _positive(x)
        else np.nan
        for e, x in zip(entry, current, strict=True)
    ]
    return result


def _state_groups_by_month(
    states: dict[str, pd.DataFrame],
) -> dict[str, dict[tuple[str, str], pd.DataFrame]]:
    return {
        month: _group_state(_normalize_state(state))
        for month, state in states.items()
    }


def _first_later_base_return(
    row: pd.Series,
    groups: dict[str, dict[tuple[str, str], pd.DataFrame]],
    timestamp: int,
) -> float:
    month = str(row["trading_day"])[:7]
    group = groups.get(month, {}).get(
        (str(row["trading_day"]), str(row["ticker"]))
    )
    _t, price = _first_open_at_or_after(group, int(timestamp))
    entry = pd.to_numeric(
        pd.Series([row.get("entry_open")]),
        errors="coerce",
    ).iloc[0]
    if not (_positive(entry) and _positive(price)):
        return np.nan
    return _base_return(float(entry), float(price))


def _forced_30m_base_return(
    row: pd.Series,
    groups: dict[str, dict[tuple[str, str], pd.DataFrame]],
) -> float:
    entry_t = pd.to_numeric(
        pd.Series([row.get("entry_reference_t")]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(entry_t):
        return np.nan
    target = int(entry_t) + MAX_HOLD_MINUTES * MINUTE_MS
    month = str(row["trading_day"])[:7]
    group = groups.get(month, {}).get(
        (str(row["trading_day"]), str(row["ticker"]))
    )
    exact = _exact_open(group, target)
    entry = pd.to_numeric(
        pd.Series([row.get("entry_open")]),
        errors="coerce",
    ).iloc[0]
    if _positive(entry) and _positive(exact):
        return _base_return(float(entry), float(exact))
    return _first_later_base_return(row, groups, target)


def _policy_key(row: pd.Series, minute: int) -> tuple[str, str, int]:
    return (
        str(row["trading_day"]),
        str(row["ticker"]),
        int(minute),
    )


def train_fitted_stopping(
    fit_rows: pd.DataFrame,
    states: dict[str, pd.DataFrame],
) -> tuple[
    dict[int, MinuteStoppingModel],
    pd.DataFrame,
    pd.Series,
]:
    """Fit finite-horizon stopping values backwards using fit rows only."""
    frame = _attach_exit_now_base(fit_rows).reset_index(drop=True)
    features = transition_feature_frame(frame)
    groups = _state_groups_by_month(states)

    held = pd.to_numeric(frame["minutes_held"], errors="coerce")
    row_by_key: dict[tuple[str, str, int], int] = {}
    for idx, row in frame.iterrows():
        minute = held.iloc[idx]
        if pd.notna(minute) and float(minute).is_integer():
            row_by_key[_policy_key(row, int(minute))] = int(idx)

    policy_value = pd.Series(np.nan, index=frame.index, dtype=float)
    models: dict[int, MinuteStoppingModel] = {}
    diagnostics: list[dict[str, object]] = []

    for minute in range(MAX_HOLD_MINUTES - 1, 0, -1):
        minute_idx = frame.index[held.eq(minute)]
        if not len(minute_idx):
            continue

        hold_values = pd.Series(np.nan, index=minute_idx, dtype=float)
        for idx in minute_idx:
            row = frame.loc[idx]
            if minute == MAX_HOLD_MINUTES - 1:
                hold_values.loc[idx] = _forced_30m_base_return(row, groups)
                continue

            next_idx = row_by_key.get(
                _policy_key(row, minute + 1)
            )
            if next_idx is not None and pd.notna(policy_value.loc[next_idx]):
                hold_values.loc[idx] = float(policy_value.loc[next_idx])
                continue

            expected_next_decision_open = pd.to_numeric(
                pd.Series([row.get("hold_reference_t")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(expected_next_decision_open):
                hold_values.loc[idx] = _first_later_base_return(
                    row,
                    groups,
                    int(expected_next_decision_open),
                )

        exit_now = pd.to_numeric(
            frame.loc[minute_idx, "exit_now_base_return_pct"],
            errors="coerce",
        )
        target = hold_values - exit_now
        valid_target = target.notna() & exit_now.notna()
        train_idx = minute_idx[valid_target.to_numpy()]

        if len(train_idx) < 150:
            raise ValueError(
                f"minute {minute} has only {len(train_idx)} fit targets"
            )

        y = target.loc[train_idx].astype(float)
        low, high = np.quantile(y.to_numpy(), [0.005, 0.995])
        y_fit = y.clip(lower=low, upper=high)

        x_minute = features.loc[train_idx]
        columns = tuple(
            column for column in x_minute
            if x_minute[column].notna().any()
        )
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=75,
            l2_regularization=2.0,
            random_state=20261046 + minute,
        )
        model.fit(x_minute.loc[:, columns], y_fit)
        fitted = MinuteStoppingModel(minute, model, columns)
        models[minute] = fitted

        current_valid = minute_idx[exit_now.notna().to_numpy()]
        predictions = pd.Series(
            model.predict(
                features.loc[current_valid].reindex(columns=columns)
            ),
            index=current_valid,
            dtype=float,
        )

        for idx in current_valid:
            predicted = float(predictions.loc[idx])
            if predicted > 0:
                value = hold_values.loc[idx]
                policy_value.loc[idx] = (
                    float(value) if pd.notna(value) else np.nan
                )
            else:
                policy_value.loc[idx] = float(
                    frame.loc[idx, "exit_now_base_return_pct"]
                )

        invalid_exit_idx = minute_idx[
            pd.to_numeric(
                frame.loc[minute_idx, "exit_now_base_return_pct"],
                errors="coerce",
            ).isna().to_numpy()
        ]
        for idx in invalid_exit_idx:
            row = frame.loc[idx]
            timestamp = pd.to_numeric(
                pd.Series([row.get("exit_reference_t")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(timestamp):
                policy_value.loc[idx] = _first_later_base_return(
                    row,
                    groups,
                    int(timestamp),
                )

        pred_train = predictions.reindex(train_idx)
        diagnostics.append(
            {
                "minute": minute,
                "candidate_rows": int(len(minute_idx)),
                "target_rows": int(len(train_idx)),
                "target_mean_pct": float(y.mean()),
                "winsor_low_pct": float(low),
                "winsor_high_pct": float(high),
                "prediction_mean_pct": float(pred_train.mean()),
                "predicted_hold_rate": float(pred_train.gt(0).mean()),
                "fit_policy_value_mean_pct": float(
                    pd.to_numeric(
                        policy_value.loc[minute_idx],
                        errors="coerce",
                    ).mean()
                ),
            }
        )

    missing = sorted(set(range(1, MAX_HOLD_MINUTES)) - set(models))
    if missing:
        raise ValueError(f"missing fitted stopping minutes: {missing}")

    return (
        models,
        pd.DataFrame(diagnostics).sort_values("minute"),
        policy_value,
    )


def score_stopping_rows(
    rows: pd.DataFrame,
    models: dict[int, MinuteStoppingModel],
) -> pd.DataFrame:
    frame = _attach_exit_now_base(rows).reset_index(drop=True)
    features = transition_feature_frame(frame)
    held = pd.to_numeric(frame["minutes_held"], errors="coerce")
    prediction = pd.Series(np.nan, index=frame.index, dtype=float)

    for minute, fitted in models.items():
        idx = frame.index[held.eq(minute)]
        if not len(idx):
            continue
        prediction.loc[idx] = fitted.model.predict(
            features.loc[idx].reindex(columns=fitted.feature_columns)
        )

    frame["predicted_stopping_advantage_pct"] = prediction
    return frame


def build_optimal_trajectories(
    evaluation_anchors: pd.DataFrame,
    scored_rows: pd.DataFrame,
    state: pd.DataFrame,
    month: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors = evaluation_anchors.copy()
    anchors["trading_day"] = anchors["trading_day"].astype(str)
    anchors["ticker"] = anchors["ticker"].astype(str)
    anchors["t"] = pd.to_numeric(anchors["t"], errors="coerce")
    anchors = anchors.loc[
        pd.to_numeric(
            anchors["opportunity_probability"],
            errors="coerce",
        ).gt(0.5)
    ].copy()
    if anchors.duplicated(KEYS).any():
        raise ValueError("evaluation anchors must be unique per ticker-day")

    state = _normalize_state(state)
    state_groups = _group_state(state)

    rows = scored_rows.copy()
    rows["trading_day"] = rows["trading_day"].astype(str)
    rows["ticker"] = rows["ticker"].astype(str)
    rows["t"] = pd.to_numeric(rows["t"], errors="coerce")
    row_groups = {
        (str(day), str(ticker)): group.sort_values(
            "minutes_held",
            kind="stable",
        ).reset_index(drop=True)
        for (day, ticker), group in rows.groupby(KEYS, sort=False)
    }

    trajectories: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []

    for anchor in anchors.sort_values(
        ["trading_day", "t", "ticker"],
        kind="stable",
    ).to_dict("records"):
        day = str(anchor["trading_day"])
        ticker = str(anchor["ticker"])
        key = (day, ticker)
        anchor_t = int(anchor["t"])
        entry_t = anchor_t + MINUTE_MS
        state_group = state_groups.get(key)
        entry_price = _exact_open(state_group, entry_t)

        record: dict[str, object] = {
            "month": month,
            "trading_day": day,
            "ticker": ticker,
            "anchor_t": anchor_t,
            "opportunity_probability": float(
                anchor["opportunity_probability"]
            ),
            "entry_t": entry_t,
            "entry_price": entry_price,
            "status": "missing_entry",
            "exit_reason": "missing_exact_entry_open",
            "exit_t": np.nan,
            "exit_price": np.nan,
            "holding_minutes": np.nan,
            "hold_decisions": 0,
            "decision_count": 0,
        }
        if not _positive(entry_price):
            _add_returns(record)
            trajectories.append(record)
            continue

        group = row_groups.get(key, pd.DataFrame())
        by_minute: dict[int, dict[str, object]] = {}
        if not group.empty:
            for row in group.to_dict("records"):
                held = pd.to_numeric(
                    pd.Series([row.get("minutes_held")]),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(held) and float(held).is_integer():
                    by_minute[int(held)] = row

        exit_t = np.nan
        exit_price = np.nan
        exit_reason = "unresolved"
        hold_decisions = 0
        decision_count = 0

        for minute in range(1, MAX_HOLD_MINUTES):
            expected_row_t = anchor_t + minute * MINUTE_MS
            decision_open_t = expected_row_t + MINUTE_MS
            row = by_minute.get(minute)

            if row is None or int(float(row["t"])) != expected_row_t:
                later_t, later_open = _first_open_at_or_after(
                    state_group,
                    decision_open_t,
                )
                decisions.append(
                    {
                        "month": month,
                        "trading_day": day,
                        "ticker": ticker,
                        "minutes_held": minute,
                        "decision_t": decision_open_t,
                        "predicted_advantage_pct": np.nan,
                        "action": "EXIT_PENDING",
                        "reason": "missing_exact_decision_state",
                    }
                )
                if _positive(later_open):
                    exit_t = later_t
                    exit_price = later_open
                    exit_reason = (
                        "gap_exit_exact_open"
                        if int(later_t) == decision_open_t
                        else "gap_exit_delayed_open"
                    )
                else:
                    exit_reason = "unresolved_after_missing_state"
                break

            current_open = pd.to_numeric(
                pd.Series([row.get("exit_open")]),
                errors="coerce",
            ).iloc[0]
            predicted = pd.to_numeric(
                pd.Series(
                    [row.get("predicted_stopping_advantage_pct")]
                ),
                errors="coerce",
            ).iloc[0]

            if not _positive(current_open):
                later_t, later_open = _first_open_at_or_after(
                    state_group,
                    decision_open_t,
                )
                decisions.append(
                    {
                        "month": month,
                        "trading_day": day,
                        "ticker": ticker,
                        "minutes_held": minute,
                        "decision_t": decision_open_t,
                        "predicted_advantage_pct": predicted,
                        "action": "EXIT_PENDING",
                        "reason": "missing_exact_decision_open",
                    }
                )
                if _positive(later_open):
                    exit_t = later_t
                    exit_price = later_open
                    exit_reason = "gap_exit_delayed_open"
                else:
                    exit_reason = "unresolved_after_missing_open"
                break

            decision_count += 1
            if not pd.notna(predicted) or not np.isfinite(float(predicted)):
                exit_t = decision_open_t
                exit_price = float(current_open)
                exit_reason = "model_score_missing_exit"
                decisions.append(
                    {
                        "month": month,
                        "trading_day": day,
                        "ticker": ticker,
                        "minutes_held": minute,
                        "decision_t": decision_open_t,
                        "predicted_advantage_pct": predicted,
                        "action": "EXIT",
                        "reason": exit_reason,
                    }
                )
                break

            if float(predicted) <= 0:
                exit_t = decision_open_t
                exit_price = float(current_open)
                exit_reason = "fitted_value_exit"
                decisions.append(
                    {
                        "month": month,
                        "trading_day": day,
                        "ticker": ticker,
                        "minutes_held": minute,
                        "decision_t": decision_open_t,
                        "predicted_advantage_pct": float(predicted),
                        "action": "EXIT",
                        "reason": exit_reason,
                    }
                )
                break

            hold_decisions += 1
            decisions.append(
                {
                    "month": month,
                    "trading_day": day,
                    "ticker": ticker,
                    "minutes_held": minute,
                    "decision_t": decision_open_t,
                    "predicted_advantage_pct": float(predicted),
                    "action": "HOLD",
                    "reason": (
                        "fitted_value_hold"
                        if minute < MAX_HOLD_MINUTES - 1
                        else "fitted_value_hold_to_forced_cap"
                    ),
                }
            )

            if minute == MAX_HOLD_MINUTES - 1:
                forced_t = entry_t + MAX_HOLD_MINUTES * MINUTE_MS
                exact_forced = _exact_open(state_group, forced_t)
                if _positive(exact_forced):
                    exit_t = forced_t
                    exit_price = exact_forced
                    exit_reason = "forced_30m_exact_open"
                else:
                    later_t, later_open = _first_open_at_or_after(
                        state_group,
                        forced_t,
                    )
                    if _positive(later_open):
                        exit_t = later_t
                        exit_price = later_open
                        exit_reason = "forced_30m_delayed_open"
                    else:
                        exit_reason = "unresolved_after_forced_cap"
                break

        record.update(
            status="completed" if _positive(exit_price) else "unresolved",
            exit_reason=exit_reason,
            exit_t=exit_t,
            exit_price=exit_price,
            holding_minutes=(
                (float(exit_t) - entry_t) / MINUTE_MS
                if _positive(exit_t)
                else np.nan
            ),
            hold_decisions=hold_decisions,
            decision_count=decision_count,
        )
        _add_returns(record)
        trajectories.append(record)

    return pd.DataFrame(trajectories), pd.DataFrame(decisions)


def _daily_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    daily = pd.DataFrame(
        {
            "day": frame["trading_day"].astype(str),
            "value": values,
        }
    ).dropna().groupby("day")["value"].mean()
    return float(daily.mean()) if len(daily) else np.nan


def _bootstrap_daily(
    frame: pd.DataFrame,
    column: str,
) -> tuple[int, float, float, float]:
    values = pd.to_numeric(frame[column], errors="coerce")
    daily = pd.DataFrame(
        {
            "day": frame["trading_day"].astype(str),
            "value": values,
        }
    ).dropna().groupby("day")["value"].mean()
    if len(daily) < 2:
        return (
            int(len(daily)),
            float(daily.mean()) if len(daily) else np.nan,
            np.nan,
            np.nan,
        )
    arr = daily.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = arr[
        rng.integers(
            0,
            len(arr),
            size=(BOOTSTRAP_SAMPLES, len(arr)),
        )
    ].mean(axis=1)
    return (
        int(len(arr)),
        float(arr.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def summarize_policy(
    trajectories: pd.DataFrame,
    policy: str,
    month: str,
) -> pd.DataFrame:
    valid = trajectories.loc[
        pd.to_numeric(
            trajectories["entry_price"],
            errors="coerce",
        ).gt(0)
    ].copy()
    completed = trajectories.loc[
        trajectories["status"].eq("completed")
    ].copy()
    row: dict[str, object] = {
        "month": month,
        "policy": policy,
        "attempts": int(len(trajectories)),
        "valid_entries": int(len(valid)),
        "valid_entry_coverage": (
            float(len(valid) / len(trajectories))
            if len(trajectories)
            else np.nan
        ),
        "completed": int(len(completed)),
        "unresolved": int(valid["status"].eq("unresolved").sum()),
        "completion_coverage": (
            float(len(completed) / len(valid))
            if len(valid)
            else np.nan
        ),
    }
    for column in (
        "gross_return_pct",
        "light_net_return_pct",
        "base_net_return_pct",
        "stress_net_return_pct",
    ):
        values = pd.to_numeric(
            completed[column],
            errors="coerce",
        )
        row[f"{column}_mean"] = (
            float(values.mean()) if len(values) else np.nan
        )

    base = pd.to_numeric(
        completed["base_net_return_pct"],
        errors="coerce",
    ).dropna()
    row["day_balanced_base_mean_pct"] = _daily_mean(
        completed,
        "base_net_return_pct",
    )
    row["base_median_pct"] = (
        float(base.median()) if len(base) else np.nan
    )
    row["base_p05_pct"] = (
        float(base.quantile(0.05)) if len(base) else np.nan
    )
    row["base_positive_rate"] = (
        float(base.gt(0).mean()) if len(base) else np.nan
    )
    row["base_severe_loss_rate"] = (
        float(base.le(SEVERE_LOSS_PCT).mean())
        if len(base)
        else np.nan
    )

    holds = pd.to_numeric(
        completed["holding_minutes"],
        errors="coerce",
    ).dropna()
    for name, value in (
        ("holding_mean_min", holds.mean()),
        ("holding_median_min", holds.median()),
        ("holding_p10_min", holds.quantile(0.10)),
        ("holding_p90_min", holds.quantile(0.90)),
        ("holding_max_min", holds.max()),
    ):
        row[name] = float(value) if len(holds) else np.nan
    row["cap_reach_rate"] = (
        float(holds.ge(MAX_HOLD_MINUTES).mean())
        if len(holds)
        else np.nan
    )
    row["mean_hold_decisions"] = (
        float(
            pd.to_numeric(
                completed["hold_decisions"],
                errors="coerce",
            ).mean()
        )
        if len(completed)
        else np.nan
    )
    return pd.DataFrame([row])


def matched_difference(
    primary: pd.DataFrame,
    comparator: pd.DataFrame,
    comparator_name: str,
    month: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["trading_day", "ticker", "anchor_t"]
    left = primary.loc[
        primary["status"].eq("completed"),
        keys + ["base_net_return_pct"],
    ].rename(
        columns={"base_net_return_pct": "primary_base_pct"}
    )
    right = comparator.loc[
        comparator["status"].eq("completed"),
        keys + ["base_net_return_pct"],
    ].rename(
        columns={"base_net_return_pct": "comparator_base_pct"}
    )
    matched = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    matched["base_difference_pct"] = (
        pd.to_numeric(matched["primary_base_pct"], errors="coerce")
        - pd.to_numeric(
            matched["comparator_base_pct"],
            errors="coerce",
        )
    )
    days, mean, low, high = _bootstrap_daily(
        matched,
        "base_difference_pct",
    )
    summary = pd.DataFrame(
        [
            {
                "month": month,
                "comparator": comparator_name,
                "matched_trades": int(len(matched)),
                "base_difference_mean_pct": float(
                    pd.to_numeric(
                        matched["base_difference_pct"],
                        errors="coerce",
                    ).mean()
                )
                if len(matched)
                else np.nan,
                "day_balanced_difference_pct": mean,
                "bootstrap_days": days,
                "ci_low_pct": low,
                "ci_high_pct": high,
            }
        ]
    )
    return matched, summary


def run_fold(
    anchor_paths: dict[str, Path],
    state_paths: dict[str, Path],
    evaluation_month: str,
):
    fit, calibration, evaluation, provenance = load_fold(
        anchor_paths,
        evaluation_month,
    )
    opportunity = train_opportunity_model(fit, calibration)

    partitions = []
    for name, frame in (
        ("fit", fit),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        item = frame.copy()
        item["_partition"] = name
        item["opportunity_probability"] = predict_opportunity(
            item,
            opportunity,
        )
        partitions.append(item)
    anchors = pd.concat(partitions, ignore_index=True)

    states: dict[str, pd.DataFrame] = {}
    sequence_parts = []
    for month, group in anchors.groupby(_month_of(anchors), sort=True):
        if month not in state_paths:
            raise ValueError(f"missing state panel for {month}")
        state = pd.read_parquet(state_paths[month])
        states[month] = state
        sequence_parts.append(build_continuation_rows(state, group))

    sequence = pd.concat(sequence_parts, ignore_index=True)
    fit_rows = sequence.loc[
        sequence["_partition"].eq("fit")
    ].reset_index(drop=True)
    cal_rows = sequence.loc[
        sequence["_partition"].eq("calibration")
    ].reset_index(drop=True)
    eval_rows = sequence.loc[
        sequence["_partition"].eq("evaluation")
    ].reset_index(drop=True)

    models, fit_diagnostics, _fit_values = train_fitted_stopping(
        fit_rows,
        states,
    )
    cal_scored = score_stopping_rows(cal_rows, models)
    eval_scored = score_stopping_rows(eval_rows, models)

    eval_anchors = anchors.loc[
        anchors["_partition"].eq("evaluation")
    ].reset_index(drop=True)
    eval_state = states[evaluation_month]

    primary, decisions = build_optimal_trajectories(
        eval_anchors,
        eval_scored,
        eval_state,
        evaluation_month,
    )

    transition = train_transition_model(fit_rows, cal_rows)
    v38_scored = eval_rows.copy()
    v38_scored["hold_probability"] = (
        predict_transition_continuation(v38_scored, transition)
    )
    v38, _v38_decisions = build_recurrent_trajectories(
        eval_anchors,
        v38_scored,
        eval_state,
        evaluation_month,
    )
    hold30 = build_always_hold_comparator(
        primary,
        eval_state,
        evaluation_month,
    )

    primary_summary = summarize_policy(
        primary,
        "fitted_optimal_stopping_v40",
        evaluation_month,
    )
    v38_summary = summarize_policy(
        v38,
        "recurrent_v38",
        evaluation_month,
    )
    hold_summary = summarize_policy(
        hold30,
        "always_hold_30m",
        evaluation_month,
    )
    summaries = pd.concat(
        [primary_summary, v38_summary, hold_summary],
        ignore_index=True,
    )

    _, v38_difference = matched_difference(
        primary,
        v38,
        "recurrent_v38",
        evaluation_month,
    )
    _, hold_difference = matched_difference(
        primary,
        hold30,
        "always_hold_30m",
        evaluation_month,
    )
    differences = pd.concat(
        [v38_difference, hold_difference],
        ignore_index=True,
    )

    completed = primary.loc[
        primary["status"].eq("completed")
    ]
    days, mean, low, high = _bootstrap_daily(
        completed,
        "base_net_return_pct",
    )
    bootstrap = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "days": days,
                "day_balanced_base_mean_pct": mean,
                "ci_low_pct": low,
                "ci_high_pct": high,
            }
        ]
    )

    cal_anchors = anchors.loc[
        anchors["_partition"].eq("calibration")
    ].reset_index(drop=True)
    calibration_months = sorted(set(_month_of(cal_anchors)))
    cal_parts = []
    for month in calibration_months:
        month_anchors = cal_anchors.loc[
            _month_of(cal_anchors).eq(month)
        ].reset_index(drop=True)
        month_rows = cal_scored.loc[
            _month_of(cal_scored).eq(month)
        ].reset_index(drop=True)
        trajectories, _ = build_optimal_trajectories(
            month_anchors,
            month_rows,
            states[month],
            month,
        )
        cal_parts.append(
            summarize_policy(
                trajectories,
                "fitted_optimal_stopping_v40",
                month,
            )
        )
    calibration_summary = (
        pd.concat(cal_parts, ignore_index=True)
        if cal_parts
        else pd.DataFrame()
    )

    accounts = pd.concat(
        [
            account_replay(
                primary,
                eval_state,
                "fitted_optimal_stopping_v40",
                evaluation_month,
            ),
            account_replay(
                v38,
                eval_state,
                "recurrent_v38",
                evaluation_month,
            ),
            account_replay(
                hold30,
                eval_state,
                "always_hold_30m",
                evaluation_month,
            ),
        ],
        ignore_index=True,
    )

    p = primary_summary.iloc[0]
    c = v38_summary.iloc[0]
    diff = v38_difference.iloc[0]
    boot = bootstrap.iloc[0]
    base_account = accounts.loc[
        accounts["policy"].eq("fitted_optimal_stopping_v40")
        & accounts["scenario"].eq("base")
    ]
    if len(base_account) != 1:
        raise AssertionError("expected one primary BASE account row")
    account = base_account.iloc[0]

    success = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "at_least_100_completed": bool(p["completed"] >= 100),
                "valid_entry_coverage_at_least_95pct": bool(
                    p["valid_entry_coverage"] >= 0.95
                ),
                "completion_at_least_90pct": bool(
                    p["completion_coverage"] >= 0.90
                ),
                "base_mean_positive": bool(
                    p["base_net_return_pct_mean"] > 0
                ),
                "day_base_mean_positive": bool(
                    p["day_balanced_base_mean_pct"] > 0
                ),
                "base_bootstrap_low_positive": bool(
                    boot["ci_low_pct"] > 0
                ),
                "beats_v38_day_mean": bool(
                    p["day_balanced_base_mean_pct"]
                    > c["day_balanced_base_mean_pct"]
                ),
                "v38_difference_positive_robust": bool(
                    diff["day_balanced_difference_pct"] > 0
                    and diff["ci_low_pct"] > 0
                ),
                "stress_not_worse_than_v38": bool(
                    p["stress_net_return_pct_mean"]
                    >= c["stress_net_return_pct_mean"]
                ),
                "base_account_positive_no_unresolved": bool(
                    account["marked_return_pct"] > 0
                    and int(account["unresolved"]) == 0
                ),
            }
        ]
    )
    success["all_frozen_checks_pass"] = success.iloc[:, 1:].all(axis=1)

    exit_reasons = (
        primary.groupby(
            ["month", "exit_reason"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    return (
        primary,
        decisions,
        summaries,
        differences,
        bootstrap,
        exit_reasons,
        calibration_summary,
        fit_diagnostics,
        accounts,
        provenance,
        success,
    )


def render_report(outputs: tuple[pd.DataFrame, ...]) -> str:
    (
        _primary,
        _decisions,
        summaries,
        differences,
        bootstrap,
        exit_reasons,
        calibration_summary,
        fit_diagnostics,
        accounts,
        provenance,
        success,
    ) = outputs
    lines = [
        "=== MoneyMaker Fitted Optimal Stopping v4.0 ===",
        "fit-only backward value recursion; HOLD iff predicted downstream BASE advantage > 0",
        "frozen v3.3 entry gate and v3.7 causal path-transition state",
        "hard maximum hold=30m; full LIGHT/BASE/STRESS round-trip evaluation",
        "April 2026+ sealed",
    ]
    for title, frame in (
        ("Policy summaries", summaries),
        ("Matched differences", differences),
        ("Primary day bootstrap", bootstrap),
        ("Primary exit reasons", exit_reasons),
        ("Calibration trajectory diagnostics", calibration_summary),
        ("Fit minute diagnostics", fit_diagnostics),
        ("Account replay", accounts),
        ("Provenance", provenance),
        ("Frozen checks", success),
    ):
        lines.extend(("", f"=== {title} ===", frame.to_string(index=False)))
    return "\n".join(lines)


def _parse(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-month", required=True)
    parser.add_argument("--anchor", action="append", type=_parse, required=True)
    parser.add_argument("--state", action="append", type=_parse, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trajectories-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--summaries-csv", type=Path, required=True)
    parser.add_argument("--differences-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-csv", type=Path, required=True)
    parser.add_argument("--exit-reasons-csv", type=Path, required=True)
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--fit-diagnostics-csv", type=Path, required=True)
    parser.add_argument("--accounts-csv", type=Path, required=True)
    parser.add_argument("--provenance-csv", type=Path, required=True)
    parser.add_argument("--success-csv", type=Path, required=True)
    args = parser.parse_args()

    outputs = run_fold(
        dict(args.anchor),
        dict(args.state),
        args.evaluation_month,
    )
    report = render_report(outputs)
    print(report, flush=True)

    paths = (
        args.trajectories_csv,
        args.decisions_csv,
        args.summaries_csv,
        args.differences_csv,
        args.bootstrap_csv,
        args.exit_reasons_csv,
        args.calibration_csv,
        args.fit_diagnostics_csv,
        args.accounts_csv,
        args.provenance_csv,
        args.success_csv,
    )
    for path, frame in zip(paths, outputs, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
