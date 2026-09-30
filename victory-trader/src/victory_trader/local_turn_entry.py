"""Request294: local-turn-aware direct entry.

Request293 showed that current fixed-horizon entry value is learnable, but a
plain first-threshold crossing usually enters immediately and does not identify
the local turn.

Request294 keeps the crack setup and fixed-horizon utility frozen. It adds
strictly causal event-sequence features that describe falling pressure,
stabilization, bounce and micro re-acceleration. A separate short-horizon turn
model is used only as an entry gate, while a high direct-value override allows
runaway setups to be entered without demanding a pullback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .direct_attractive_entry import (
    MIN_ENTRY_RATE,
    MAX_ENTRY_RATE,
)
from .downstream_aligned_candidate import candidate_columns
from .hierarchical_crack_entry_controller import (
    EVENT_FEATURES,
    _fit,
    _numeric,
    _predict,
    _safe_auc,
    _safe_spearman,
    _usable,
)
from .lagged_minute_context import CROSSFIT_DAYS
from .prehot_context_ablation import usable_context_columns
from .two_stage_risk_veto import attach_ticker_history

REQUEST_ID = 294
TURN_HORIZON_MS = 15_000
UTILITY_QUANTILES = (0.20, 0.35, 0.50, 0.65)
TURN_QUANTILES = (0.20, 0.40, 0.60)
OVERRIDE_QUANTILES = (0.80, 0.90)
MIN_CAL_ENTRY_RATE = 0.35
MAX_CAL_ENTRY_RATE = 0.95

TURN_FEATURES = (
    "turn_return_sum_3",
    "turn_return_sum_5",
    "turn_positive_frac_3",
    "turn_positive_frac_5",
    "turn_negative_streak",
    "turn_positive_streak",
    "turn_close_slope_3",
    "turn_close_slope_5",
    "turn_return_accel_3",
    "turn_bounce_recent_low_5",
    "turn_drawdown_recent_high_5",
    "turn_reclaim_prev3_high",
    "turn_gap_ratio_3",
    "turn_bounce_x_volume_burst5",
    "turn_accel_x_signed_volume",
    "turn_low_age_x_bounce",
)


def _streak(values: list[float], positive: bool) -> float:
    count = 0
    for value in reversed(values):
        if not np.isfinite(value):
            break
        condition = value > 0 if positive else value < 0
        if not condition:
            break
        count += 1
    return float(count)


def _last(values: list[float], count: int) -> np.ndarray:
    arr = np.asarray(values[-count:], dtype=float)
    return arr[np.isfinite(arr)]


def attach_local_turn_features(states: pd.DataFrame) -> pd.DataFrame:
    result = states.sort_values(
        ["trading_day", "ticker", "hot_t", "decision_t"],
        kind="stable",
    ).copy()
    for name in TURN_FEATURES:
        result[name] = np.nan
    result["future_return_15s_pct"] = np.nan

    keys = ["trading_day", "ticker", "hot_t"]
    for _, group in result.groupby(keys, sort=False):
        indices = group.index.tolist()
        returns: list[float] = []
        closes: list[float] = []
        gaps: list[float] = []

        execution_t = _numeric(group.execution_t).to_numpy(dtype=float)
        execution_price = _numeric(group.execution_price).to_numpy(dtype=float)

        for pos, index in enumerate(indices):
            row = result.loc[index]
            ret = float(row.get("last_active_return_pct", np.nan))
            close = float(row.get("last_close_vs_hot_pct", np.nan))
            gap = float(row.get("inter_event_gap_s", np.nan))
            returns.append(ret)
            closes.append(close)
            gaps.append(gap)

            r3 = _last(returns, 3)
            r5 = _last(returns, 5)
            c3 = _last(closes, 3)
            c5 = _last(closes, 5)
            previous3 = _last(returns[:-1], 3)
            gap3 = _last(gaps[:-1], 3)

            result.at[index, "turn_return_sum_3"] = (
                float(r3.sum()) if len(r3) else np.nan
            )
            result.at[index, "turn_return_sum_5"] = (
                float(r5.sum()) if len(r5) else np.nan
            )
            result.at[index, "turn_positive_frac_3"] = (
                float(np.mean(r3 > 0)) if len(r3) else np.nan
            )
            result.at[index, "turn_positive_frac_5"] = (
                float(np.mean(r5 > 0)) if len(r5) else np.nan
            )
            result.at[index, "turn_negative_streak"] = _streak(
                returns, positive=False
            )
            result.at[index, "turn_positive_streak"] = _streak(
                returns, positive=True
            )
            result.at[index, "turn_close_slope_3"] = (
                float(c3[-1] - c3[0]) if len(c3) >= 2 else np.nan
            )
            result.at[index, "turn_close_slope_5"] = (
                float(c5[-1] - c5[0]) if len(c5) >= 2 else np.nan
            )
            previous_mean = (
                float(previous3.mean()) if len(previous3) else np.nan
            )
            result.at[index, "turn_return_accel_3"] = (
                float(ret - previous_mean)
                if np.isfinite(ret) and np.isfinite(previous_mean)
                else np.nan
            )
            result.at[index, "turn_bounce_recent_low_5"] = (
                float(c5[-1] - c5.min()) if len(c5) else np.nan
            )
            result.at[index, "turn_drawdown_recent_high_5"] = (
                float(c5[-1] - c5.max()) if len(c5) else np.nan
            )
            previous_close = _last(closes[:-1], 3)
            result.at[index, "turn_reclaim_prev3_high"] = (
                float(close - previous_close.max())
                if np.isfinite(close) and len(previous_close)
                else np.nan
            )
            previous_gap = (
                float(gap3.mean()) if len(gap3) else np.nan
            )
            result.at[index, "turn_gap_ratio_3"] = (
                float(gap / previous_gap)
                if np.isfinite(gap)
                and np.isfinite(previous_gap)
                and previous_gap > 0
                else np.nan
            )

            bounce = float(row.get("bounce_from_active_low_pct", np.nan))
            volume_burst = float(
                row.get("current_sec_volume_burst_5", np.nan)
            )
            sec_accel = float(
                row.get("current_sec_accel_5_pct", np.nan)
            )
            signed_volume = float(
                row.get(
                    "current_sec_signed_volume_imbalance_60",
                    np.nan,
                )
            )
            low_age = float(row.get("seconds_since_active_low", np.nan))
            result.at[index, "turn_bounce_x_volume_burst5"] = (
                bounce * volume_burst
                if np.isfinite(bounce) and np.isfinite(volume_burst)
                else np.nan
            )
            result.at[index, "turn_accel_x_signed_volume"] = (
                sec_accel * signed_volume
                if np.isfinite(sec_accel) and np.isfinite(signed_volume)
                else np.nan
            )
            result.at[index, "turn_low_age_x_bounce"] = (
                low_age * bounce
                if np.isfinite(low_age) and np.isfinite(bounce)
                else np.nan
            )

            current_t = execution_t[pos]
            current_price = execution_price[pos]
            if not np.isfinite(current_t) or not np.isfinite(current_price):
                continue
            target_t = current_t + TURN_HORIZON_MS
            future_positions = np.flatnonzero(execution_t >= target_t)
            future_positions = future_positions[future_positions > pos]
            if not len(future_positions):
                continue
            future_pos = int(future_positions[0])
            future_price = execution_price[future_pos]
            if current_price > 0 and np.isfinite(future_price):
                result.at[index, "future_return_15s_pct"] = float(
                    (future_price / current_price - 1.0) * 100.0
                )

    return result


def _policy(
    scored: pd.DataFrame,
    *,
    utility_threshold: float,
    turn_threshold: float,
    override_threshold: float,
) -> pd.DataFrame:
    rows = []
    keys = ["trading_day", "ticker", "hot_t"]
    for _, episode in scored.groupby(keys, sort=False):
        ordered = episode.sort_values("decision_t", kind="stable")
        if ordered.empty:
            continue
        immediate = ordered.iloc[0]
        chosen = None
        path = None
        for _, row in ordered.iterrows():
            q_value = float(row.predicted_enter_utility_pct)
            q_turn = float(row.predicted_turn_15s_pct)
            if q_value >= override_threshold:
                chosen = row
                path = "STRONG_VALUE"
                break
            if q_value >= utility_threshold and q_turn >= turn_threshold:
                chosen = row
                path = "TURN_CONFIRMED"
                break

        if chosen is None:
            rows.append(
                {
                    "trading_day": str(immediate.trading_day),
                    "ticker": str(immediate.ticker).upper(),
                    "hot_t": int(immediate.hot_t),
                    "action": "SKIP",
                    "entry_path": "SKIP",
                    "delay_s": np.nan,
                    "immediate_price": float(immediate.execution_price),
                    "chosen_price": np.nan,
                    "entry_price_improvement_pct": np.nan,
                    "realized_utility_pct": 0.0,
                    "chosen_mfe_fixed_pct": 0.0,
                    "chosen_mae_fixed_pct": 0.0,
                    "immediate_utility_pct": float(
                        immediate.entry_utility_fixed_pct
                    ),
                    "immediate_mfe_fixed_pct": float(
                        immediate.entry_mfe_fixed_pct
                    ),
                    "immediate_mae_fixed_pct": float(
                        immediate.entry_mae_fixed_pct
                    ),
                }
            )
            continue

        rows.append(
            {
                "trading_day": str(chosen.trading_day),
                "ticker": str(chosen.ticker).upper(),
                "hot_t": int(chosen.hot_t),
                "action": "ENTER",
                "entry_path": str(path),
                "delay_s": float(
                    (
                        int(chosen.decision_t)
                        - int(immediate.decision_t)
                    )
                    / 1000.0
                ),
                "immediate_price": float(immediate.execution_price),
                "chosen_price": float(chosen.execution_price),
                "entry_price_improvement_pct": float(
                    (
                        float(immediate.execution_price)
                        / float(chosen.execution_price)
                        - 1.0
                    )
                    * 100.0
                ),
                "realized_utility_pct": float(
                    chosen.entry_utility_fixed_pct
                ),
                "chosen_mfe_fixed_pct": float(
                    chosen.entry_mfe_fixed_pct
                ),
                "chosen_mae_fixed_pct": float(
                    chosen.entry_mae_fixed_pct
                ),
                "immediate_utility_pct": float(
                    immediate.entry_utility_fixed_pct
                ),
                "immediate_mfe_fixed_pct": float(
                    immediate.entry_mfe_fixed_pct
                ),
                "immediate_mae_fixed_pct": float(
                    immediate.entry_mae_fixed_pct
                ),
            }
        )
    return pd.DataFrame(rows)


def _objective(decisions: pd.DataFrame) -> tuple[float, float, float]:
    if decisions.empty:
        return -np.inf, 0.0, np.inf
    utility = float(_numeric(decisions.realized_utility_pct).mean())
    entry_rate = float(decisions.action.eq("ENTER").mean())
    entered = decisions.loc[decisions.action.eq("ENTER")]
    delay = (
        float(_numeric(entered.delay_s).mean())
        if len(entered)
        else np.inf
    )
    return utility, entry_rate, delay


def _threshold_grid(
    fit: pd.DataFrame,
) -> tuple[list[float], list[float], list[float]]:
    value = _numeric(fit.predicted_enter_utility_pct).dropna()
    turn = _numeric(fit.predicted_turn_15s_pct).dropna()
    value_thresholds = [
        float(value.quantile(q)) for q in UTILITY_QUANTILES
    ]
    turn_thresholds = [-np.inf]
    turn_thresholds.extend(
        float(turn.quantile(q)) for q in TURN_QUANTILES
    )
    override_thresholds = [np.inf]
    override_thresholds.extend(
        float(value.quantile(q)) for q in OVERRIDE_QUANTILES
    )
    return value_thresholds, turn_thresholds, override_thresholds


def _calibrate(
    fit_scored: pd.DataFrame,
    calibration_scored: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    value_thresholds, turn_thresholds, override_thresholds = (
        _threshold_grid(fit_scored)
    )
    rows = []
    for value_threshold in value_thresholds:
        for turn_threshold in turn_thresholds:
            for override_threshold in override_thresholds:
                if np.isfinite(override_threshold):
                    override_threshold = max(
                        float(override_threshold),
                        float(value_threshold),
                    )
                decisions = _policy(
                    calibration_scored,
                    utility_threshold=float(value_threshold),
                    turn_threshold=float(turn_threshold),
                    override_threshold=float(override_threshold),
                )
                utility, entry_rate, delay = _objective(decisions)
                rows.append(
                    {
                        "utility_threshold": float(value_threshold),
                        "turn_threshold": (
                            None
                            if not np.isfinite(turn_threshold)
                            else float(turn_threshold)
                        ),
                        "override_threshold": (
                            None
                            if not np.isfinite(override_threshold)
                            else float(override_threshold)
                        ),
                        "mean_decision_utility_pct": utility,
                        "entry_rate": entry_rate,
                        "mean_delay_s": delay,
                        "eligible": bool(
                            MIN_CAL_ENTRY_RATE
                            <= entry_rate
                            <= MAX_CAL_ENTRY_RATE
                        ),
                    }
                )
    eligible = [row for row in rows if row["eligible"]]
    pool = eligible if eligible else rows
    best = max(
        pool,
        key=lambda row: (
            row["mean_decision_utility_pct"],
            -row["mean_delay_s"],
            row["entry_rate"],
        ),
    )
    return best, rows


def _fit_models(
    frame: pd.DataFrame,
    entry_columns: tuple[str, ...],
    turn_columns: tuple[str, ...],
    seed: int,
):
    entry = _fit(
        frame,
        entry_columns,
        "entry_utility_fixed_pct",
        seed,
        log_target=False,
        episode_weighting=True,
    )
    turn_train = frame.loc[
        _numeric(frame.future_return_15s_pct).notna()
    ].copy()
    turn = _fit(
        turn_train,
        turn_columns,
        "future_return_15s_pct",
        seed + 1,
        log_target=False,
        episode_weighting=True,
    )
    return entry, turn


def _score(
    frame: pd.DataFrame,
    entry_model,
    turn_model,
) -> pd.DataFrame:
    result = frame.copy()
    result["predicted_enter_utility_pct"] = _predict(
        result, entry_model
    )
    result["predicted_turn_15s_pct"] = _predict(
        result, turn_model
    )
    return result


def crossfit(
    states: pd.DataFrame,
    entry_columns: tuple[str, ...],
    turn_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scored_parts = []
    decision_parts = []
    folds = {}

    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train_days = [
            day for day in CROSSFIT_DAYS if day != held_day
        ]
        calibration_day = train_days[-1]
        inner_days = train_days[:-1]

        inner = states.loc[
            states.trading_day.astype(str).isin(inner_days)
        ].copy()
        calibration = states.loc[
            states.trading_day.astype(str).eq(calibration_day)
        ].copy()
        outer = states.loc[
            states.trading_day.astype(str).isin(train_days)
        ].copy()
        held = states.loc[
            states.trading_day.astype(str).eq(held_day)
        ].copy()

        inner_entry, inner_turn = _fit_models(
            inner,
            entry_columns,
            turn_columns,
            20263500 + fold_index * 50,
        )
        inner_scored = _score(inner, inner_entry, inner_turn)
        cal_scored = _score(
            calibration,
            inner_entry,
            inner_turn,
        )
        selected, grid = _calibrate(
            inner_scored,
            cal_scored,
        )

        entry_model, turn_model = _fit_models(
            outer,
            entry_columns,
            turn_columns,
            20263525 + fold_index * 50,
        )
        held_scored = _score(
            held,
            entry_model,
            turn_model,
        )
        held_scored["held_out_day"] = held_day
        scored_parts.append(held_scored)

        turn_threshold = (
            -np.inf
            if selected["turn_threshold"] is None
            else float(selected["turn_threshold"])
        )
        override_threshold = (
            np.inf
            if selected["override_threshold"] is None
            else float(selected["override_threshold"])
        )
        decisions = _policy(
            held_scored,
            utility_threshold=float(
                selected["utility_threshold"]
            ),
            turn_threshold=turn_threshold,
            override_threshold=override_threshold,
        )
        decisions["held_out_day"] = held_day
        decision_parts.append(decisions)

        folds[held_day] = {
            "inner_fit_days": inner_days,
            "calibration_day": calibration_day,
            "outer_train_days": train_days,
            "selected": selected,
            "grid_size": len(grid),
        }

    return (
        pd.concat(scored_parts, ignore_index=True),
        pd.concat(decision_parts, ignore_index=True),
        folds,
    )


def signal_report(scored: pd.DataFrame) -> dict:
    utility = _numeric(scored.entry_utility_fixed_pct)
    utility_score = _numeric(scored.predicted_enter_utility_pct)
    future_turn = _numeric(scored.future_return_15s_pct)
    turn_score = _numeric(scored.predicted_turn_15s_pct)

    uv = utility.notna() & utility_score.notna()
    tv = future_turn.notna() & turn_score.notna()

    return {
        "entry_utility": {
            "rows": int(uv.sum()),
            "spearman": _safe_spearman(
                utility.loc[uv],
                utility_score.loc[uv],
            ),
            "positive_auc": _safe_auc(
                utility.loc[uv].gt(0).astype(int),
                utility_score.loc[uv],
            ),
        },
        "local_turn_15s": {
            "rows": int(tv.sum()),
            "mean_future_return_pct": (
                float(future_turn.loc[tv].mean())
                if int(tv.sum())
                else None
            ),
            "spearman": _safe_spearman(
                future_turn.loc[tv],
                turn_score.loc[tv],
            ),
            "positive_auc": _safe_auc(
                future_turn.loc[tv].gt(0).astype(int),
                turn_score.loc[tv],
            ),
        },
    }


def policy_report(
    decisions: pd.DataFrame,
    request293: dict,
) -> dict:
    entered = decisions.loc[decisions.action.eq("ENTER")].copy()
    utility = _numeric(decisions.realized_utility_pct)
    immediate = _numeric(decisions.immediate_utility_pct)
    result = {
        "episodes": int(len(decisions)),
        "entered": int(len(entered)),
        "entry_rate": float(len(entered) / len(decisions)),
        "skipped": int(decisions.action.eq("SKIP").sum()),
        "strong_value_entries": int(
            entered.entry_path.eq("STRONG_VALUE").sum()
        ),
        "turn_confirmed_entries": int(
            entered.entry_path.eq("TURN_CONFIRMED").sum()
        ),
        "mean_decision_utility_pct": float(utility.mean()),
        "mean_immediate_utility_pct": float(immediate.mean()),
        "utility_gain_vs_immediate_pp": float(
            (utility - immediate).mean()
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
            float(
                _numeric(
                    entered.entry_price_improvement_pct
                ).mean()
            )
            if len(entered)
            else None
        ),
        "mean_chosen_mfe_pct": (
            float(_numeric(entered.chosen_mfe_fixed_pct).mean())
            if len(entered)
            else None
        ),
        "mean_immediate_mfe_same_entered_pct": (
            float(
                _numeric(
                    entered.immediate_mfe_fixed_pct
                ).mean()
            )
            if len(entered)
            else None
        ),
        "mfe_gain_same_entered_pp": (
            float(
                (
                    _numeric(entered.chosen_mfe_fixed_pct)
                    - _numeric(
                        entered.immediate_mfe_fixed_pct
                    )
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
    for threshold in (5.0, 10.0, 20.0):
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
                    _numeric(
                        entered.immediate_mfe_fixed_pct
                    )
                    .ge(threshold)
                    .mean()
                )
                if len(entered)
                else None
            ),
        }

    old = request293.get("policy_metrics", {})
    result["request293"] = {
        "mean_decision_utility_pct": old.get(
            "mean_decision_utility_pct"
        ),
        "entry_rate": old.get("entry_rate"),
        "mean_entry_price_improvement_pct": old.get(
            "mean_entry_price_improvement_pct"
        ),
        "mfe_gain_same_entered_pp": old.get(
            "mfe_gain_same_entered_pp"
        ),
    }
    result["fixed_2pct"] = old.get("fixed_2pct", {})
    return result


def evaluate(
    state_path: Path,
    first_hot_path: Path,
    scan_path: Path,
    request293_path: Path,
    output_path: Path,
    scored_output: Path,
    decisions_output: Path,
) -> int:
    raw_states = pd.read_parquet(state_path)
    states = attach_local_turn_features(raw_states)
    request293 = json.loads(
        request293_path.read_text(encoding="utf-8")
    )

    first_hot = pd.read_parquet(first_hot_path)
    first_hot = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    raw_scan = pd.read_parquet(scan_path)
    raw_scan = raw_scan.loc[
        raw_scan.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()

    candidates, ticker_extra = attach_ticker_history(
        first_hot,
        raw_scan,
    )
    ticker_extra = usable_context_columns(
        candidates,
        ticker_extra,
    )
    base_columns = candidate_columns(candidates)
    rich_columns = _usable(
        states,
        [*base_columns, *ticker_extra],
    )
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
            *TURN_FEATURES,
        ],
    )
    turn_columns = _usable(
        states,
        [
            "rich_setup_score_pct",
            *second_columns,
            *EVENT_FEATURES,
            *TURN_FEATURES,
        ],
    )

    scored, decisions, folds = crossfit(
        states,
        entry_columns,
        turn_columns,
    )
    signals = signal_report(scored)
    policy = policy_report(decisions, request293)
    fixed_cash = policy.get("fixed_2pct", {}).get(
        "cash_adjusted_mean_utility_pct"
    )

    checks = {
        "entry_signal": (
            signals["entry_utility"]["spearman"] is not None
            and signals["entry_utility"]["spearman"] > 0.20
        ),
        "local_turn_signal": (
            signals["local_turn_15s"]["spearman"] is not None
            and signals["local_turn_15s"]["spearman"] > 0.05
            and signals["local_turn_15s"]["positive_auc"] is not None
            and signals["local_turn_15s"]["positive_auc"] > 0.55
        ),
        "entry_rate": (
            MIN_ENTRY_RATE
            <= policy["entry_rate"]
            <= MAX_ENTRY_RATE
        ),
        "utility_beats_request293": (
            policy["mean_decision_utility_pct"]
            > policy["request293"]["mean_decision_utility_pct"]
        ),
        "utility_beats_immediate": (
            policy["utility_gain_vs_immediate_pp"] > 0
        ),
        "price_improves": (
            policy["mean_entry_price_improvement_pct"] is not None
            and policy["mean_entry_price_improvement_pct"] > 0
        ),
        "mfe_nonlower": (
            policy["mfe_gain_same_entered_pp"] is not None
            and policy["mfe_gain_same_entered_pp"] >= 0
        ),
        "beats_fixed_2pct": (
            fixed_cash is None
            or policy["mean_decision_utility_pct"] > fixed_cash
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "setup_frozen_from_request291": True,
        "fixed_horizon_labels_from_request292": True,
        "local_turn_horizon_seconds": 15,
        "turn_features": list(TURN_FEATURES),
        "feature_counts": {
            "entry_total": len(entry_columns),
            "turn_total": len(turn_columns),
        },
        "outer_folds": folds,
        "signals": signals,
        "policy_metrics": policy,
        "checks": checks,
        "research_gate_pass": bool(all(checks.values())),
        "next_boundary": (
            "if local-turn gating beats immediate, freeze setup+entry and "
            "move to HOLD/EXIT. If local-turn signal exists but policy still "
            "lags immediate, test episode-specific value/risk calibration "
            "without inventing another fixed pullback."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_output.parent.mkdir(parents=True, exist_ok=True)
    decisions_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    scored.to_parquet(scored_output, index=False)
    decisions.to_parquet(decisions_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-oof", type=Path, required=True)
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--request293-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.state_oof,
        args.first_hot,
        args.opportunity_scan,
        args.request293_json,
        args.output,
        args.scored_output,
        args.decisions_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
