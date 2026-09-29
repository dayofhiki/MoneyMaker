"""Request292: fixed-horizon fitted stopping for crack entry.

Request291 established two useful signals but its direct WAIT teacher searched
the whole future five-minute window and gave delayed entries a fresh 60-minute
horizon.  That made WAIT structurally too attractive.

Request292 keeps the OOF crack-selected setup population frozen and changes only
entry stopping:
- every candidate entry is evaluated to the same HOT+60m terminal timestamp;
- direct ENTER utility rewards remaining upside and penalizes adverse excursion;
- WAIT value is bootstrapped from a fitted next-state continuation model rather
  than the hindsight best future entry;
- ENTER / WAIT / SKIP are all available;
- action margins are chosen on a nested calibration day, then models are refit
  on all outer-training days before scoring the held-out day.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .causal_minute_controller_rebuild import causal_execution_scan
from .config import load_settings
from .downstream_aligned_candidate import candidate_columns
from .hierarchical_crack_entry_controller import (
    CRACK_HORIZON_MS,
    ENTRY_OBSERVE_MS,
    EVENT_FEATURES,
    EXPLOSIVE_LEVELS,
    MAX_ACTIVE_STATES,
    MINUTE_MS,
    SECOND_MS,
    _event_features,
    _fit,
    _numeric,
    _predict,
    _second_features,
    _usable,
)
from .lagged_minute_context import CROSSFIT_DAYS
from .massive_client import MassiveClient
from .prehot_context_ablation import usable_context_columns
from .pullback_trailing_exit import build_pullback_episodes
from .second_path_attention_probe import SECOND_CACHE_DIR
from .two_stage_risk_veto import attach_ticker_history

REQUEST_ID = 292
BELL_MAN_ITERATIONS = 3
MIN_CAL_ENTRY_RATE = 0.20
ACTION_MARGINS = (0.0, 0.10, 0.25, 0.50, 1.00)
ENTER_FLOORS = (-0.25, 0.0, 0.25, 0.50, 1.00)


@dataclass(frozen=True)
class QModels:
    enter_model: object
    wait_model: object
    action_margin: float
    enter_floor: float
    calibration_day: str
    calibration_objective_pct: float
    calibration_entry_rate: float


def _fixed_terminal_value(
    seconds: pd.DataFrame,
    *,
    execution_t: int,
    execution_price: float,
    terminal_t: int,
) -> dict[str, float | bool | None]:
    """Evaluate an entry against one shared HOT-anchored terminal."""
    if seconds.empty or execution_price <= 0:
        return {
            "complete": False,
            "max_return_pct": None,
            "mfe_pct": None,
            "mae_pct": None,
            "utility_pct": None,
        }
    if execution_t >= terminal_t or not bool(seconds.t.ge(int(terminal_t)).any()):
        return {
            "complete": False,
            "max_return_pct": None,
            "mfe_pct": None,
            "mae_pct": None,
            "utility_pct": None,
        }
    future = seconds.loc[
        seconds.t.ge(int(execution_t)) & seconds.t.le(int(terminal_t))
    ].copy()
    if future.empty:
        return {
            "complete": False,
            "max_return_pct": None,
            "mfe_pct": None,
            "mae_pct": None,
            "utility_pct": None,
        }

    maximum = float(_numeric(future.c).max())
    minimum = float(_numeric(future.l).min())
    max_return = float((maximum / execution_price - 1.0) * 100.0)
    mfe = float(max(0.0, max_return))
    mae = float(min(0.0, (minimum / execution_price - 1.0) * 100.0))
    # Simple asymmetric access utility: remaining upside minus the adverse
    # excursion required to access it. A setup with no upside can be negative,
    # which gives SKIP an economically meaningful zero-value baseline.
    utility = float(max_return + mae)
    return {
        "complete": True,
        "max_return_pct": max_return,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "utility_pct": utility,
    }


def _next_state_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(
        ["trading_day", "ticker", "hot_t", "decision_t"],
        kind="stable",
    ).copy()
    grouped = result.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    )
    result["next_entry_utility_fixed_pct"] = grouped[
        "entry_utility_fixed_pct"
    ].shift(-1)
    result["one_step_wait_advantage_pct"] = (
        result["next_entry_utility_fixed_pct"]
        - result["entry_utility_fixed_pct"]
    )
    return result


def _fixed_pullback_map(
    first_hot: pd.DataFrame,
    causal_scan: pd.DataFrame,
) -> dict[tuple[str, str, int], dict[str, float | int]]:
    episodes = build_pullback_episodes(first_hot, causal_scan)
    out: dict[tuple[str, str, int], dict[str, float | int]] = {}
    for raw in episodes.loc[episodes.entered.astype(bool)].to_dict("records"):
        out[
            (
                str(raw["trading_day"]),
                str(raw["ticker"]).upper(),
                int(raw["hot_t"]),
            )
        ] = {
            "entry_t": int(raw["entry_t"]),
            "entry_price": float(raw["entry_price"]),
        }
    return out


def build_fixed_horizon_states(
    selected: pd.DataFrame,
    first_hot: pd.DataFrame,
    causal_scan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR / "request292",
        request_interval_seconds=0.02,
    )
    fixed_map = _fixed_pullback_map(first_hot, causal_scan)
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    episodes: list[pd.DataFrame] = []
    fixed_rows: list[dict] = []

    for raw in selected.to_dict("records"):
        day = str(raw["trading_day"])
        ticker = str(raw["ticker"]).upper()
        hot_t = int(raw["hot_t"])
        terminal_t = hot_t + CRACK_HORIZON_MS
        key = (day, ticker)
        if key not in cache:
            payload = client.second_bars_range(
                ticker,
                date.fromisoformat(day),
                date.fromisoformat(day),
                adjusted=False,
            )
            from .second_path_attention_probe import _second_frame
            cache[key] = _second_frame(payload)
        seconds = cache[key]
        hot_price = float(raw["hot_reference_price"])

        active = seconds.loc[
            seconds.t.ge(hot_t)
            & seconds.t.lt(hot_t + ENTRY_OBSERVE_MS)
        ].copy()
        if len(active) < 3:
            continue
        active = active.head(MAX_ACTIVE_STATES).reset_index(drop=True)

        rows: list[dict] = []
        for index in range(len(active) - 1):
            current = active.iloc[index]
            next_row = active.iloc[index + 1]
            decision_t = int(current.t) + SECOND_MS
            execution_t = int(next_row.t)
            execution_price = float(next_row.o)
            label = _fixed_terminal_value(
                seconds,
                execution_t=execution_t,
                execution_price=execution_price,
                terminal_t=terminal_t,
            )
            if not bool(label["complete"]):
                continue

            item = dict(raw)
            item["decision_t"] = decision_t
            item["execution_t"] = execution_t
            item["execution_price"] = execution_price
            item["execution_gap_s"] = float(
                (execution_t - decision_t) / SECOND_MS
            )
            item["entry_max_return_fixed_pct"] = label["max_return_pct"]
            item["entry_mfe_fixed_pct"] = label["mfe_pct"]
            item["entry_mae_fixed_pct"] = label["mae_pct"]
            item["entry_utility_fixed_pct"] = label["utility_pct"]
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
            episodes.append(_next_state_columns(pd.DataFrame(rows)))

        fixed = fixed_map.get((day, ticker, hot_t))
        if fixed is not None:
            label = _fixed_terminal_value(
                seconds,
                execution_t=int(fixed["entry_t"]),
                execution_price=float(fixed["entry_price"]),
                terminal_t=terminal_t,
            )
            if bool(label["complete"]):
                fixed_rows.append(
                    {
                        "trading_day": day,
                        "ticker": ticker,
                        "hot_t": hot_t,
                        "entry_t": int(fixed["entry_t"]),
                        "entry_price": float(fixed["entry_price"]),
                        "entry_max_return_fixed_pct": label["max_return_pct"],
                        "entry_mfe_fixed_pct": label["mfe_pct"],
                        "entry_mae_fixed_pct": label["mae_pct"],
                        "entry_utility_fixed_pct": label["utility_pct"],
                    }
                )

    states = (
        pd.concat(episodes, ignore_index=True)
        if episodes
        else pd.DataFrame()
    )
    fixed = pd.DataFrame(fixed_rows)
    second_columns = [
        name for name in states.columns if name.startswith("current_")
    ]
    audit = {
        "selected_candidate_rows": int(len(selected)),
        "event_episodes": (
            int(
                states.loc[:, ["trading_day", "ticker", "hot_t"]]
                .drop_duplicates()
                .shape[0]
            )
            if len(states)
            else 0
        ),
        "event_state_rows": int(len(states)),
        "fixed_2pct_rows": int(len(fixed)),
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
    return states, fixed, audit


def _attach_bootstrap_target(
    frame: pd.DataFrame,
    state_values: pd.Series,
) -> pd.DataFrame:
    result = frame.copy()
    result["_state_value"] = state_values.to_numpy(float)
    grouped = result.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    )
    result["_bootstrap_wait_target"] = grouped["_state_value"].shift(-1)
    result["_bootstrap_wait_target"] = (
        _numeric(result["_bootstrap_wait_target"]).fillna(0.0)
    )
    return result


def _fit_bellman_models(
    train: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
):
    enter_model = _fit(
        train,
        columns,
        "entry_utility_fixed_pct",
        seed,
        log_target=False,
        episode_weighting=True,
    )
    predicted_enter = pd.Series(
        _predict(train, enter_model),
        index=train.index,
        dtype=float,
    )
    state_value = pd.Series(
        np.maximum(0.0, predicted_enter.to_numpy(float)),
        index=train.index,
        dtype=float,
    )
    wait_model = None
    for iteration in range(BELL_MAN_ITERATIONS):
        target_frame = _attach_bootstrap_target(train, state_value)
        wait_model = _fit(
            target_frame,
            columns,
            "_bootstrap_wait_target",
            seed + 10 + iteration,
            log_target=False,
            episode_weighting=True,
        )
        predicted_wait = pd.Series(
            _predict(train, wait_model),
            index=train.index,
            dtype=float,
        )
        state_value = pd.Series(
            np.maximum.reduce(
                [
                    np.zeros(len(train), dtype=float),
                    predicted_enter.to_numpy(float),
                    predicted_wait.to_numpy(float),
                ]
            ),
            index=train.index,
            dtype=float,
        )
    if wait_model is None:
        raise ValueError("Request292 failed to fit wait model")
    return enter_model, wait_model


def _score_q(
    frame: pd.DataFrame,
    enter_model,
    wait_model,
) -> pd.DataFrame:
    result = frame.copy()
    result["predicted_enter_utility_pct"] = _predict(
        result,
        enter_model,
    )
    result["predicted_wait_value_pct"] = _predict(
        result,
        wait_model,
    )
    result["predicted_wait_minus_enter_pct"] = (
        result.predicted_wait_value_pct
        - result.predicted_enter_utility_pct
    )
    return result


def _run_policy(
    scored: pd.DataFrame,
    *,
    action_margin: float,
    enter_floor: float,
) -> pd.DataFrame:
    rows = []
    keys = ["trading_day", "ticker", "hot_t"]
    for _, episode in scored.groupby(keys, sort=False):
        ordered = episode.sort_values("decision_t", kind="stable")
        if ordered.empty:
            continue
        records = ordered.to_dict("records")
        immediate = records[0]
        action = "SKIP"
        chosen = None
        wait_actions = 0
        for position, raw in enumerate(records):
            q_enter = float(raw["predicted_enter_utility_pct"])
            q_wait = float(raw["predicted_wait_value_pct"])
            can_wait = position < len(records) - 1
            if (
                can_wait
                and q_wait > max(0.0, q_enter) + float(action_margin)
            ):
                wait_actions += 1
                continue
            if q_enter > float(enter_floor):
                action = "ENTER"
                chosen = raw
            else:
                action = "SKIP"
            break

        if chosen is None and action != "SKIP":
            chosen = records[-1]

        if chosen is None:
            rows.append(
                {
                    "trading_day": str(immediate["trading_day"]),
                    "ticker": str(immediate["ticker"]).upper(),
                    "hot_t": int(immediate["hot_t"]),
                    "action": "SKIP",
                    "wait_actions": int(wait_actions),
                    "delay_s": np.nan,
                    "immediate_price": float(immediate["execution_price"]),
                    "chosen_price": np.nan,
                    "entry_price_improvement_pct": np.nan,
                    "realized_utility_pct": 0.0,
                    "chosen_mfe_fixed_pct": 0.0,
                    "chosen_mae_fixed_pct": 0.0,
                    "immediate_utility_pct": float(
                        immediate["entry_utility_fixed_pct"]
                    ),
                    "immediate_mfe_fixed_pct": float(
                        immediate["entry_mfe_fixed_pct"]
                    ),
                    "immediate_mae_fixed_pct": float(
                        immediate["entry_mae_fixed_pct"]
                    ),
                }
            )
            continue

        rows.append(
            {
                "trading_day": str(chosen["trading_day"]),
                "ticker": str(chosen["ticker"]).upper(),
                "hot_t": int(chosen["hot_t"]),
                "action": "ENTER",
                "wait_actions": int(wait_actions),
                "delay_s": float(
                    (int(chosen["decision_t"]) - int(immediate["decision_t"]))
                    / SECOND_MS
                ),
                "immediate_price": float(immediate["execution_price"]),
                "chosen_price": float(chosen["execution_price"]),
                "entry_price_improvement_pct": float(
                    (
                        float(immediate["execution_price"])
                        / float(chosen["execution_price"])
                        - 1.0
                    )
                    * 100.0
                ),
                "realized_utility_pct": float(
                    chosen["entry_utility_fixed_pct"]
                ),
                "chosen_mfe_fixed_pct": float(
                    chosen["entry_mfe_fixed_pct"]
                ),
                "chosen_mae_fixed_pct": float(
                    chosen["entry_mae_fixed_pct"]
                ),
                "immediate_utility_pct": float(
                    immediate["entry_utility_fixed_pct"]
                ),
                "immediate_mfe_fixed_pct": float(
                    immediate["entry_mfe_fixed_pct"]
                ),
                "immediate_mae_fixed_pct": float(
                    immediate["entry_mae_fixed_pct"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _decision_objective(decisions: pd.DataFrame) -> tuple[float, float, float]:
    if decisions.empty:
        return -np.inf, 0.0, np.inf
    utility = _numeric(decisions.realized_utility_pct)
    entry_rate = float(decisions.action.eq("ENTER").mean())
    entered = decisions.loc[decisions.action.eq("ENTER")]
    delay = (
        float(_numeric(entered.delay_s).mean())
        if len(entered)
        else np.inf
    )
    return float(utility.mean()), entry_rate, delay


def _calibrate_actions(scored: pd.DataFrame) -> tuple[float, float, dict]:
    candidates = []
    for margin in ACTION_MARGINS:
        for floor in ENTER_FLOORS:
            decisions = _run_policy(
                scored,
                action_margin=margin,
                enter_floor=floor,
            )
            objective, entry_rate, delay = _decision_objective(decisions)
            eligible = entry_rate >= MIN_CAL_ENTRY_RATE
            candidates.append(
                {
                    "action_margin": float(margin),
                    "enter_floor": float(floor),
                    "objective_pct": objective,
                    "entry_rate": entry_rate,
                    "mean_delay_s": delay,
                    "eligible": bool(eligible),
                }
            )
    eligible = [row for row in candidates if row["eligible"]]
    pool = eligible if eligible else candidates
    best = max(
        pool,
        key=lambda row: (
            row["objective_pct"],
            -row["mean_delay_s"],
            row["entry_rate"],
        ),
    )
    return (
        float(best["action_margin"]),
        float(best["enter_floor"]),
        {
            "selected": best,
            "grid": candidates,
        },
    )


def crossfit_fitted_stopping(
    states: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scored_parts = []
    decision_parts = []
    folds = {}

    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train_days = [
            day for day in CROSSFIT_DAYS if day != held_day
        ]
        calibration_day = train_days[-1]
        inner_fit_days = train_days[:-1]

        inner_fit = states.loc[
            states.trading_day.astype(str).isin(inner_fit_days)
        ].copy()
        calibration = states.loc[
            states.trading_day.astype(str).eq(calibration_day)
        ].copy()
        outer_train = states.loc[
            states.trading_day.astype(str).isin(train_days)
        ].copy()
        held = states.loc[
            states.trading_day.astype(str).eq(held_day)
        ].copy()

        inner_enter, inner_wait = _fit_bellman_models(
            inner_fit,
            feature_columns,
            20263300 + fold_index * 100,
        )
        cal_scored = _score_q(
            calibration,
            inner_enter,
            inner_wait,
        )
        margin, floor, calibration_report = _calibrate_actions(
            cal_scored
        )

        enter_model, wait_model = _fit_bellman_models(
            outer_train,
            feature_columns,
            20263350 + fold_index * 100,
        )
        held_scored = _score_q(
            held,
            enter_model,
            wait_model,
        )
        held_scored["entry_held_out_day"] = held_day
        held_scored["action_margin"] = margin
        held_scored["enter_floor"] = floor
        scored_parts.append(held_scored)

        decisions = _run_policy(
            held_scored,
            action_margin=margin,
            enter_floor=floor,
        )
        decisions["held_out_day"] = held_day
        decision_parts.append(decisions)

        folds[held_day] = {
            "inner_fit_days": inner_fit_days,
            "calibration_day": calibration_day,
            "outer_train_days": train_days,
            "action_margin": margin,
            "enter_floor": floor,
            "calibration": calibration_report["selected"],
        }

    return (
        pd.concat(scored_parts, ignore_index=True),
        pd.concat(decision_parts, ignore_index=True),
        folds,
    )


def matched_setup_report(setup_oof: pd.DataFrame) -> dict:
    rows = []
    for day, part in setup_oof.groupby(
        setup_oof.trading_day.astype(str),
        sort=True,
    ):
        count = int(part.selected_crack.fillna(False).astype(bool).sum())
        if count <= 0:
            continue
        rich = part.nlargest(count, "rich_setup_score_pct")
        base = part.nlargest(count, "base_setup_score_pct")
        for name, chosen in (("base", base), ("rich", rich)):
            target = _numeric(chosen.crack_mfe_open_60m_pct)
            rows.append(
                {
                    "day": str(day),
                    "model": name,
                    "rows": int(len(chosen)),
                    "mean_mfe_pct": float(target.mean()),
                    "plus5_rate": float(target.ge(5.0).mean()),
                    "plus10_rate": float(target.ge(10.0).mean()),
                    "plus20_rate": float(target.ge(20.0).mean()),
                }
            )
    table = pd.DataFrame(rows)
    out = {}
    for name in ("base", "rich"):
        part = table.loc[table.model.eq(name)]
        out[name] = {
            "rows": int(part.rows.sum()) if len(part) else 0,
            "day_balanced_mean_mfe_pct": (
                float(part.mean_mfe_pct.mean()) if len(part) else None
            ),
            "day_balanced_plus5_rate": (
                float(part.plus5_rate.mean()) if len(part) else None
            ),
            "day_balanced_plus10_rate": (
                float(part.plus10_rate.mean()) if len(part) else None
            ),
            "day_balanced_plus20_rate": (
                float(part.plus20_rate.mean()) if len(part) else None
            ),
        }
    return out


def policy_report(
    decisions: pd.DataFrame,
    fixed: pd.DataFrame,
) -> dict:
    if decisions.empty:
        return {"episodes": 0}

    entered = decisions.loc[decisions.action.eq("ENTER")].copy()
    utility = _numeric(decisions.realized_utility_pct)
    immediate_utility = _numeric(decisions.immediate_utility_pct)
    result = {
        "episodes": int(len(decisions)),
        "entered": int(len(entered)),
        "entry_rate": float(len(entered) / len(decisions)),
        "skipped": int(decisions.action.eq("SKIP").sum()),
        "mean_decision_utility_pct": float(utility.mean()),
        "mean_immediate_utility_pct": float(immediate_utility.mean()),
        "utility_gain_vs_immediate_pp": float(
            (utility - immediate_utility).mean()
        ),
        "mean_delay_s_entered": (
            float(_numeric(entered.delay_s).mean())
            if len(entered)
            else None
        ),
        "median_delay_s_entered": (
            float(_numeric(entered.delay_s).median())
            if len(entered)
            else None
        ),
        "mean_entry_price_improvement_pct": (
            float(_numeric(entered.entry_price_improvement_pct).mean())
            if len(entered)
            else None
        ),
        "mean_chosen_mfe_pct": (
            float(_numeric(entered.chosen_mfe_fixed_pct).mean())
            if len(entered)
            else None
        ),
        "mean_immediate_mfe_same_entered_pct": (
            float(_numeric(entered.immediate_mfe_fixed_pct).mean())
            if len(entered)
            else None
        ),
        "mfe_gain_same_entered_pp": (
            float(
                (
                    _numeric(entered.chosen_mfe_fixed_pct)
                    - _numeric(entered.immediate_mfe_fixed_pct)
                ).mean()
            )
            if len(entered)
            else None
        ),
        "mean_chosen_mae_pct": (
            float(_numeric(entered.chosen_mae_fixed_pct).mean())
            if len(entered)
            else None
        ),
        "levels": {},
    }
    for threshold in EXPLOSIVE_LEVELS:
        tag = int(threshold)
        result["levels"][f"plus{tag}"] = {
            "chosen_rate": (
                float(
                    _numeric(entered.chosen_mfe_fixed_pct)
                    .ge(threshold)
                    .mean()
                )
                if len(entered)
                else None
            ),
            "immediate_same_entered_rate": (
                float(
                    _numeric(entered.immediate_mfe_fixed_pct)
                    .ge(threshold)
                    .mean()
                )
                if len(entered)
                else None
            ),
        }

    if not fixed.empty:
        fixed_keys = ["trading_day", "ticker", "hot_t"]
        comparable = decisions.loc[:, fixed_keys].merge(
            fixed,
            on=fixed_keys,
            how="inner",
            validate="one_to_one",
        )
        selected_count = len(decisions)
        result["fixed_2pct"] = {
            "available_rows": int(len(comparable)),
            "coverage": float(
                len(comparable) / selected_count
            ) if selected_count else 0.0,
            "cash_adjusted_mean_utility_pct": float(
                _numeric(comparable.entry_utility_fixed_pct).sum()
                / selected_count
            ) if selected_count else None,
            "available_mean_utility_pct": (
                float(_numeric(comparable.entry_utility_fixed_pct).mean())
                if len(comparable)
                else None
            ),
            "available_mean_mfe_pct": (
                float(_numeric(comparable.entry_mfe_fixed_pct).mean())
                if len(comparable)
                else None
            ),
        }
    else:
        result["fixed_2pct"] = {"available_rows": 0, "coverage": 0.0}

    return result


def entry_diagnostic(scored: pd.DataFrame) -> dict:
    target = _numeric(scored.one_step_wait_advantage_pct)
    score = _numeric(scored.predicted_wait_minus_enter_pct)
    valid = target.notna() & score.notna()
    if not int(valid.sum()):
        return {"rows": 0}
    actual = target.loc[valid]
    predicted = score.loc[valid]
    from .hierarchical_crack_entry_controller import _safe_auc, _safe_spearman
    return {
        "rows": int(valid.sum()),
        "one_step_wait_beneficial_rate": float(actual.gt(0).mean()),
        "predicted_diff_spearman": _safe_spearman(actual, predicted),
        "predicted_diff_auc": _safe_auc(
            actual.gt(0).astype(int),
            predicted,
        ),
    }


def evaluate(
    setup_path: Path,
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    state_output: Path,
    decision_output: Path,
) -> int:
    setup_oof = pd.read_parquet(setup_path)
    setup_oof["ticker"] = setup_oof.ticker.astype(str).str.upper()
    selected = setup_oof.loc[
        setup_oof.selected_crack.fillna(False).astype(bool)
    ].copy()

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

    candidates, ticker_extra = attach_ticker_history(
        first_hot,
        raw_scan,
    )
    ticker_extra = usable_context_columns(candidates, ticker_extra)
    base_columns = candidate_columns(candidates)
    rich_columns = _usable(
        candidates,
        [*base_columns, *ticker_extra],
    )

    # setup_oof already carries the static candidate context used in Request291.
    states, fixed, audit = build_fixed_horizon_states(
        selected,
        first_hot,
        causal_scan,
    )
    second_columns = _usable(
        states,
        [
            name for name in states.columns
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

    scored, decisions, folds = crossfit_fitted_stopping(
        states,
        entry_columns,
    )
    diagnostic = entry_diagnostic(scored)
    policy = policy_report(decisions, fixed)
    matched_setup = matched_setup_report(setup_oof)

    fixed_cash = policy.get("fixed_2pct", {}).get(
        "cash_adjusted_mean_utility_pct"
    )
    checks = {
        "matched_setup_plus10_rich_not_worse": (
            matched_setup["rich"]["day_balanced_plus10_rate"]
            is not None
            and matched_setup["base"]["day_balanced_plus10_rate"]
            is not None
            and matched_setup["rich"]["day_balanced_plus10_rate"]
            >= matched_setup["base"]["day_balanced_plus10_rate"]
        ),
        "event_state_support": audit["event_state_rows"] >= 4000,
        "second_feature_coverage": audit["second_feature_coverage"] >= 0.80,
        "entry_rate_nontrivial": (
            0.20 <= policy["entry_rate"] <= 0.95
        ),
        "utility_beats_immediate": (
            policy["utility_gain_vs_immediate_pp"] > 0
        ),
        "entry_price_improves": (
            policy["mean_entry_price_improvement_pct"] is not None
            and policy["mean_entry_price_improvement_pct"] > 0
        ),
        "mfe_nonlower_on_entered": (
            policy["mfe_gain_same_entered_pp"] is not None
            and policy["mfe_gain_same_entered_pp"] >= 0
        ),
        "beats_fixed_2pct_cash_adjusted_utility": (
            fixed_cash is None
            or policy["mean_decision_utility_pct"] > fixed_cash
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "setup_frozen_from_request291": True,
        "matched_rate_setup": matched_setup,
        "fixed_horizon": {
            "terminal": "HOT_t + 60 minutes for every entry candidate",
            "entry_utility": (
                "max future second-close return to fixed terminal + "
                "adverse excursion (negative); cash/SKIP = 0"
            ),
        },
        "event_state_audit": audit,
        "feature_counts": {
            "setup_base": len(base_columns),
            "setup_rich": len(rich_columns),
            "entry_total": len(entry_columns),
            "entry_second": len(second_columns),
        },
        "fitted_stopping": {
            "bellman_iterations": BELL_MAN_ITERATIONS,
            "outer_folds": folds,
            "diagnostic": diagnostic,
            "policy": policy,
        },
        "checks": checks,
        "research_gate_pass": bool(all(checks.values())),
        "next_boundary": (
            "if gate passes, freeze setup+entry and begin second-resolution "
            "HOLD/EXIT continuation; otherwise diagnose entry utility and "
            "action calibration only, without reopening crack selection"
        ),
        "interpretation": (
            "All action features are causal. Future prices are labels only. "
            "WAIT value is fitted from the next-state continuation estimate, "
            "not a direct hindsight search for the best future entry. "
            "Historical one-second aggregates are trade aggregates, not NBBO."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_output.parent.mkdir(parents=True, exist_ok=True)
    decision_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    scored.to_parquet(state_output, index=False)
    decisions.to_parquet(decision_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-oof", type=Path, required=True)
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-output", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.setup_oof,
        args.first_hot,
        args.opportunity_scan,
        args.output,
        args.state_output,
        args.decision_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
