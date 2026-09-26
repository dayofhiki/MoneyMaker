"""Request 227: multi-event pre-entry continuation controller.

Request226 showed that entry-time features weakly rank first-stress cushion, but
a positive-first-stress admission rule rejects almost every trade because most
Request208 entries are immediately cost-negative at the next observation.

Request227 replaces the fixed three-minute / one-step entry objective.

At each causal WATCH checkpoint (minutes 1..5 after HOT), a hypothetical entry
is labeled by the equal-weight average BASE return at the 1st, 2nd, 3rd and 5th
actually observed post-entry events. This is a fixed multi-event continuation
target, not a hindsight best price.

A second target compares ENTER now with the option to WAIT exactly one more
WATCH minute and still retain ABSTAIN=0:

    enter_vs_wait = current_multi_event_value - max(0, next_watch_value)

Two models are trained on fit and calibrated on calibration only.

Runtime policy:
* ENTER when predicted multi-event value > 0 and predicted ENTER-vs-WAIT
  advantage >= 0;
* otherwise WAIT to the next WATCH checkpoint;
* at minute 5, ENTER only if predicted multi-event value > 0, else ABSTAIN.

Economic comparison uses the same fixed three-minute BASE outcome after each
chosen entry so that only entry timing/admission changes relative to Request208.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .pullback_turn_transition_entry import (
    MODEL_FEATURES,
    attach_transition_features,
    predict as predict_request208,
    train_model as train_request208,
)
from .recurrent_wait_entry_action_value import (
    MAX_WAIT_MINUTES,
    _base_return,
    build_watch_states,
)
from .state_action_risk import SEVERE_LOSS_PCT
from .wait_reobserve_entry_confirmation import _day_weights, _feature_frame

REQUEST_ID = 227
VALUE_SEED = 20261227
ADVANTAGE_SEED = 20261228
BOOTSTRAP_SEED = 20261227
BOOTSTRAP_SAMPLES = 10_000

EVENT_HORIZONS = (1, 2, 3, 5)
MIN_STATE_COVERAGE = 0.80
MIN_VALUE_SPEARMAN = 0.10
MIN_ADVANTAGE_SPEARMAN = 0.05
MIN_EXECUTION_RATE = 0.10
MAX_EXECUTION_RATE = 0.80
MIN_TRADES = 50
MIN_IMPROVED_DAYS = 4

FEATURES = tuple(
    dict.fromkeys([*MODEL_FEATURES, "minutes_since_hot"])
)


@dataclass(frozen=True)
class Head:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def attach_multi_event_entry_labels(
    watch_states: pd.DataFrame,
    position_rows: pd.DataFrame,
) -> pd.DataFrame:
    result = watch_states.copy()
    for horizon in EVENT_HORIZONS:
        result[f"entry_event{horizon}_base_pct"] = np.nan
    result["entry_multi_event_value_pct"] = np.nan
    result["next_watch_multi_event_value_pct"] = np.nan
    result["enter_vs_wait_multi_event_advantage_pct"] = np.nan

    row_map: dict[tuple[str, str, int, int], int] = {}
    for index, row in result.iterrows():
        row_map[
            (
                str(row["trading_day"]),
                str(row["ticker"]).upper(),
                int(row["hot_t"]),
                int(row["minutes_since_hot"]),
            )
        ] = int(index)

    for keys, group in position_rows.groupby(EPISODE_KEYS, sort=False):
        work = group.copy()
        work["_minute"] = pd.to_numeric(
            work["minutes_held"], errors="coerce"
        )
        work["_state_t"] = pd.to_numeric(
            work["state_t"], errors="coerce"
        )
        work["_open"] = pd.to_numeric(
            work["exit_reference_open"], errors="coerce"
        )
        work = work.loc[
            work["_minute"].notna()
            & work["_state_t"].notna()
            & work["_open"].gt(0)
        ].copy()
        work = work.sort_values(
            ["_state_t", "_minute"], kind="stable"
        ).drop_duplicates("_state_t", keep="first")
        if work.empty:
            continue

        for minute in range(1, MAX_WAIT_MINUTES + 1):
            key = (
                str(keys[0]),
                str(keys[1]).upper(),
                int(keys[2]),
                minute,
            )
            target_index = row_map.get(key)
            if target_index is None:
                continue

            candidates = work.loc[
                (work["_minute"] - float(minute)).abs() <= 1e-9
            ]
            if candidates.empty:
                continue
            entry = candidates.iloc[0]
            entry_t = int(entry["_state_t"])
            entry_open = float(entry["_open"])

            future = work.loc[
                work["_state_t"].gt(entry_t)
            ].sort_values("_state_t", kind="stable")
            if len(future) < max(EVENT_HORIZONS):
                continue

            values: list[float] = []
            valid = True
            for horizon in EVENT_HORIZONS:
                exit_open = float(
                    future.iloc[horizon - 1]["_open"]
                )
                value = _base_return(entry_open, exit_open)
                if pd.isna(value) or not np.isfinite(float(value)):
                    valid = False
                    break
                result.at[
                    target_index,
                    f"entry_event{horizon}_base_pct",
                ] = float(value)
                values.append(float(value))

            if valid and len(values) == len(EVENT_HORIZONS):
                result.at[
                    target_index,
                    "entry_multi_event_value_pct",
                ] = float(np.mean(values))

    for _, group in result.groupby(EPISODE_KEYS, sort=False):
        by_minute = {
            int(row["minutes_since_hot"]): int(index)
            for index, row in group.iterrows()
        }
        for minute in range(1, MAX_WAIT_MINUTES + 1):
            index = by_minute.get(minute)
            if index is None:
                continue
            current = pd.to_numeric(
                pd.Series(
                    [result.at[index, "entry_multi_event_value_pct"]]
                ),
                errors="coerce",
            ).iloc[0]
            if pd.isna(current):
                continue

            if minute < MAX_WAIT_MINUTES:
                next_index = by_minute.get(minute + 1)
                if next_index is None:
                    continue
                next_value = pd.to_numeric(
                    pd.Series(
                        [
                            result.at[
                                next_index,
                                "entry_multi_event_value_pct",
                            ]
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.isna(next_value):
                    continue
                wait_value = max(0.0, float(next_value))
                result.at[
                    index,
                    "next_watch_multi_event_value_pct",
                ] = float(next_value)
            else:
                wait_value = 0.0
                result.at[
                    index,
                    "next_watch_multi_event_value_pct",
                ] = 0.0

            result.at[
                index,
                "enter_vs_wait_multi_event_advantage_pct",
            ] = float(current - wait_value)

    return result


def _usable_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in FEATURES
        if column in frame.columns
        and pd.to_numeric(
            frame[column], errors="coerce"
        ).notna().any()
    )


def train_head(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    target_column: str,
    seed: int,
) -> Head:
    target = pd.to_numeric(
        fit[target_column], errors="coerce"
    )
    train = fit.loc[target.notna()].copy()
    y = target.loc[target.notna()].to_numpy(dtype=float)
    if len(train) < 500:
        raise ValueError(
            f"request 227 insufficient fit rows for {target_column}: "
            f"{len(train)}"
        )

    columns = _usable_columns(train)
    low, high = np.quantile(y, [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _feature_frame(train, columns),
        np.clip(y, low, high),
        sample_weight=_day_weights(train),
    )

    cal_target = pd.to_numeric(
        calibration[target_column], errors="coerce"
    )
    cal = calibration.loc[cal_target.notna()].copy()
    actual = cal_target.loc[cal_target.notna()].to_numpy(dtype=float)
    if len(cal) < 250:
        raise ValueError(
            f"request 227 insufficient calibration rows for "
            f"{target_column}: {len(cal)}"
        )
    raw = model.predict(_feature_frame(cal, columns))
    residual = actual - raw
    offset = float(
        np.average(residual, weights=_day_weights(cal))
    )
    return Head(
        model=model,
        columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_head(frame: pd.DataFrame, fitted: Head) -> np.ndarray:
    return (
        fitted.model.predict(
            _feature_frame(frame, fitted.columns)
        )
        + fitted.offset
    )


def _safe_spearman(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = a.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    value = a.loc[valid].corr(s.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def target_diagnostics(
    frame: pd.DataFrame,
    target_column: str,
    score_column: str,
) -> dict[str, object]:
    target = pd.to_numeric(
        frame[target_column], errors="coerce"
    )
    score = pd.to_numeric(frame[score_column], errors="coerce")
    valid = target.notna() & score.notna()
    work = frame.loc[valid].copy()
    work["_actual"] = target.loc[valid]
    work["_score"] = score.loc[valid]

    by_day: dict[str, object] = {}
    positive_days = 0
    for day, part in work.groupby(
        work["trading_day"].astype(str), sort=True
    ):
        corr = _safe_spearman(part["_actual"], part["_score"])
        positive_days += int(corr is not None and corr > 0)
        by_day[str(day)] = {
            "rows": int(len(part)),
            "spearman": corr,
            "actual_mean_pct": float(part["_actual"].mean()),
        }

    return {
        "rows": int(len(frame)),
        "evaluable_rows": int(len(work)),
        "coverage": (
            float(len(work) / len(frame)) if len(frame) else None
        ),
        "spearman": _safe_spearman(
            work["_actual"], work["_score"]
        ),
        "positive_spearman_days": int(positive_days),
        "actual_mean_pct": (
            float(work["_actual"].mean()) if len(work) else None
        ),
        "by_day": by_day,
    }


def recurrent_policy(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    counts = {
        "episodes": 0,
        "entered": 0,
        "abstained": 0,
        "coverage_miss": 0,
        "wait_actions": 0,
    }

    for keys, group in scored.groupby(EPISODE_KEYS, sort=False):
        counts["episodes"] += 1
        by_minute = {
            int(row["minutes_since_hot"]): row
            for _, row in group.sort_values(
                "minutes_since_hot", kind="stable"
            ).iterrows()
        }
        chosen = None
        failed = False

        for minute in range(1, MAX_WAIT_MINUTES + 1):
            row = by_minute.get(minute)
            if row is None:
                counts["coverage_miss"] += 1
                failed = True
                break

            value = pd.to_numeric(
                pd.Series(
                    [row.get("predicted_multi_event_value_pct")]
                ),
                errors="coerce",
            ).iloc[0]
            advantage = pd.to_numeric(
                pd.Series(
                    [
                        row.get(
                            "predicted_enter_vs_wait_advantage_pct"
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]
            if pd.isna(value) or pd.isna(advantage):
                counts["coverage_miss"] += 1
                failed = True
                break

            if minute == MAX_WAIT_MINUTES:
                if float(value) > 0:
                    chosen = row
                else:
                    counts["abstained"] += 1
                break

            if float(value) > 0 and float(advantage) >= 0:
                chosen = row
                break

            counts["wait_actions"] += 1

        if failed or chosen is None:
            continue

        realized = pd.to_numeric(
            pd.Series([chosen.get("enter_3m_base_pct")]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(realized):
            counts["coverage_miss"] += 1
            continue

        counts["entered"] += 1
        rows.append(
            {
                "trading_day": str(keys[0]),
                "ticker": str(keys[1]).upper(),
                "hot_t": int(keys[2]),
                "entry_minute_after_hot": int(
                    chosen["minutes_since_hot"]
                ),
                "realized_base_return_pct": float(realized),
                "predicted_multi_event_value_pct": float(
                    chosen["predicted_multi_event_value_pct"]
                ),
                "predicted_enter_vs_wait_advantage_pct": float(
                    chosen[
                        "predicted_enter_vs_wait_advantage_pct"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows), counts


def request208_policy(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in scored.groupby(EPISODE_KEYS, sort=False):
        by_minute = {
            int(row["minutes_since_hot"]): row
            for _, row in group.sort_values(
                "minutes_since_hot", kind="stable"
            ).iterrows()
        }
        chosen = None
        for minute in range(1, MAX_WAIT_MINUTES + 1):
            row = by_minute.get(minute)
            if row is None:
                break
            if minute == MAX_WAIT_MINUTES:
                chosen = row
                break
            score = pd.to_numeric(
                pd.Series(
                    [row.get("predicted_request208_advantage_pct")]
                ),
                errors="coerce",
            ).iloc[0]
            if pd.isna(score):
                break
            if float(score) >= 0:
                chosen = row
                break

        if chosen is None:
            continue
        realized = pd.to_numeric(
            pd.Series([chosen.get("enter_3m_base_pct")]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(realized):
            continue
        rows.append(
            {
                "trading_day": str(keys[0]),
                "ticker": str(keys[1]).upper(),
                "hot_t": int(keys[2]),
                "entry_minute_after_hot": int(
                    chosen["minutes_since_hot"]
                ),
                "realized_base_return_pct": float(realized),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_daily(
    values: np.ndarray,
    seed: int,
) -> dict[str, object]:
    daily = np.asarray(values, dtype=float)
    daily = daily[np.isfinite(daily)]
    if len(daily) < 2:
        return {
            "days": int(len(daily)),
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": float(daily.mean()) if len(daily) else None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(seed)
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


def portfolio_rows(
    position_rows: pd.DataFrame,
    trades: pd.DataFrame,
    policy: str,
) -> pd.DataFrame:
    universe = position_rows.loc[
        :, EPISODE_KEYS
    ].drop_duplicates().copy()
    universe["ticker"] = universe["ticker"].astype(str).str.upper()
    if trades.empty:
        universe["realized_base_return_pct"] = 0.0
        universe["executed"] = False
    else:
        trade = trades.loc[
            :, EPISODE_KEYS + ["realized_base_return_pct"]
        ].copy()
        trade["ticker"] = trade["ticker"].astype(str).str.upper()
        universe = universe.merge(
            trade,
            on=EPISODE_KEYS,
            how="left",
            validate="one_to_one",
        )
        universe["executed"] = universe[
            "realized_base_return_pct"
        ].notna()
        universe["realized_base_return_pct"] = pd.to_numeric(
            universe["realized_base_return_pct"], errors="coerce"
        ).fillna(0.0)
    universe["policy"] = policy
    return universe


def portfolio_metrics(
    rows: pd.DataFrame,
) -> dict[str, object]:
    value = pd.to_numeric(
        rows["realized_base_return_pct"], errors="coerce"
    )
    work = rows.loc[value.notna()].copy()
    work["_value"] = value.loc[value.notna()]
    daily = work.groupby(
        work["trading_day"].astype(str), sort=True
    )["_value"].mean()
    executed = work.loc[work["executed"].astype(bool)]
    executed_values = pd.to_numeric(
        executed["_value"], errors="coerce"
    )
    return {
        "episodes": int(len(work)),
        "executed": int(work["executed"].sum()),
        "execution_rate": float(work["executed"].mean()),
        "portfolio_mean_pct": float(work["_value"].mean()),
        "portfolio_day_balanced_mean_pct": float(daily.mean()),
        "portfolio_positive_rate": float(
            work["_value"].gt(0).mean()
        ),
        "trade_mean_pct": (
            float(executed_values.mean())
            if len(executed)
            else None
        ),
        "trade_positive_rate": (
            float(executed_values.gt(0).mean())
            if len(executed)
            else None
        ),
        "trade_severe_loss_rate": (
            float(executed_values.le(SEVERE_LOSS_PCT).mean())
            if len(executed)
            else None
        ),
        "trade_p05_pct": (
            float(executed_values.quantile(0.05))
            if len(executed)
            else None
        ),
        "by_day": {
            str(k): float(v) for k, v in daily.items()
        },
    }


def matched_portfolio_difference(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, object]:
    left = candidate.loc[
        :, EPISODE_KEYS + ["realized_base_return_pct"]
    ].rename(
        columns={
            "realized_base_return_pct": "candidate_return_pct"
        }
    )
    right = baseline.loc[
        :, EPISODE_KEYS + ["realized_base_return_pct"]
    ].rename(
        columns={
            "realized_base_return_pct": "baseline_return_pct"
        }
    )
    matched = left.merge(
        right,
        on=EPISODE_KEYS,
        validate="one_to_one",
    )
    matched["difference_pct"] = (
        matched["candidate_return_pct"]
        - matched["baseline_return_pct"]
    )
    daily = matched.groupby(
        matched["trading_day"].astype(str), sort=True
    )["difference_pct"].mean()
    return {
        "episodes": int(len(matched)),
        "mean_difference_pct": float(
            matched["difference_pct"].mean()
        ),
        "candidate_better_rate": float(
            matched["difference_pct"].gt(0).mean()
        ),
        "day_balanced_difference_pct": float(daily.mean()),
        "improved_days": int((daily > 0).sum()),
        "bootstrap": _bootstrap_daily(
            daily.to_numpy(dtype=float),
            BOOTSTRAP_SEED,
        ),
    }


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )
    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(position_paths) != len(FRESH_DAYS):
        raise ValueError(
            "request 227 requires complete Request178 position shards"
        )
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )

    fit_watch = attach_transition_features(
        build_watch_states(fit_positions)
    )
    cal_watch = attach_transition_features(
        build_watch_states(calibration_positions)
    )
    fresh_watch = attach_transition_features(
        build_watch_states(fresh_positions)
    )

    fit_watch = attach_multi_event_entry_labels(
        fit_watch, fit_positions
    )
    cal_watch = attach_multi_event_entry_labels(
        cal_watch, calibration_positions
    )
    fresh_watch = attach_multi_event_entry_labels(
        fresh_watch, fresh_positions
    )

    value_head = train_head(
        fit_watch,
        cal_watch,
        target_column="entry_multi_event_value_pct",
        seed=VALUE_SEED,
    )
    advantage_head = train_head(
        fit_watch,
        cal_watch,
        target_column=(
            "enter_vs_wait_multi_event_advantage_pct"
        ),
        seed=ADVANTAGE_SEED,
    )

    request208_model = train_request208(
        attach_relative_advantage(fit_watch.copy()),
        attach_relative_advantage(cal_watch.copy()),
    )

    for frame in (cal_watch, fresh_watch):
        frame["predicted_multi_event_value_pct"] = (
            predict_head(frame, value_head)
        )
        frame["predicted_enter_vs_wait_advantage_pct"] = (
            predict_head(frame, advantage_head)
        )
        request208_scored = attach_relative_advantage(frame.copy())
        frame["predicted_request208_advantage_pct"] = (
            predict_request208(
                request208_scored,
                request208_model,
            )
        )

    value_diag = target_diagnostics(
        fresh_watch,
        "entry_multi_event_value_pct",
        "predicted_multi_event_value_pct",
    )
    advantage_diag = target_diagnostics(
        fresh_watch,
        "enter_vs_wait_multi_event_advantage_pct",
        "predicted_enter_vs_wait_advantage_pct",
    )

    candidate_trades, candidate_counts = recurrent_policy(
        fresh_watch
    )
    baseline_trades = request208_policy(fresh_watch)

    candidate_portfolio = portfolio_rows(
        fresh_positions,
        candidate_trades,
        "request227_multi_event_entry",
    )
    baseline_portfolio = portfolio_rows(
        fresh_positions,
        baseline_trades,
        "request208_relative_entry",
    )

    candidate_metrics = portfolio_metrics(
        candidate_portfolio
    )
    baseline_metrics = portfolio_metrics(
        baseline_portfolio
    )
    difference = matched_portfolio_difference(
        candidate_portfolio,
        baseline_portfolio,
    )

    total_episodes = int(
        fresh_positions.loc[
            :, EPISODE_KEYS
        ].drop_duplicates().shape[0]
    )
    minute1 = int(
        fresh_watch.loc[
            pd.to_numeric(
                fresh_watch["minutes_since_hot"], errors="coerce"
            ).eq(1),
            EPISODE_KEYS,
        ].drop_duplicates().shape[0]
    )
    state_coverage = (
        float(minute1 / total_episodes)
        if total_episodes
        else None
    )

    value_s = value_diag.get("spearman")
    advantage_s = advantage_diag.get("spearman")
    execution_rate = candidate_metrics.get("execution_rate")
    candidate_day = candidate_metrics.get(
        "portfolio_day_balanced_mean_pct"
    )
    baseline_day = baseline_metrics.get(
        "portfolio_day_balanced_mean_pct"
    )
    diff_day = difference.get("day_balanced_difference_pct")
    diff_low = difference.get("bootstrap", {}).get("ci_low_pct")
    candidate_severe = candidate_metrics.get(
        "trade_severe_loss_rate"
    )
    baseline_severe = baseline_metrics.get(
        "trade_severe_loss_rate"
    )

    gate = bool(
        state_coverage is not None
        and state_coverage >= MIN_STATE_COVERAGE
        and value_s is not None
        and float(value_s) >= MIN_VALUE_SPEARMAN
        and advantage_s is not None
        and float(advantage_s) >= MIN_ADVANTAGE_SPEARMAN
        and execution_rate is not None
        and MIN_EXECUTION_RATE
        <= float(execution_rate)
        <= MAX_EXECUTION_RATE
        and int(candidate_metrics.get("executed", 0))
        >= MIN_TRADES
        and candidate_day is not None
        and baseline_day is not None
        and float(candidate_day) > 0
        and float(candidate_day) > float(baseline_day)
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
        and int(difference.get("improved_days", 0))
        >= MIN_IMPROVED_DAYS
        and candidate_severe is not None
        and baseline_severe is not None
        and float(candidate_severe) <= float(baseline_severe)
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "target": {
            "multi_event_horizons": list(EVENT_HORIZONS),
            "multi_event_value": (
                "equal-weight mean BASE return at observed events "
                "1, 2, 3 and 5 after hypothetical entry"
            ),
            "enter_vs_wait": (
                "current multi-event value minus max(0, next WATCH "
                "minute multi-event value)"
            ),
            "hindsight_best_price_used": False,
            "future_information_used_as_input": False,
        },
        "policy": {
            "minutes": [1, 2, 3, 4, 5],
            "enter_condition": (
                "predicted multi-event value > 0 and predicted "
                "ENTER-vs-WAIT advantage >= 0"
            ),
            "minute5": (
                "ENTER iff predicted multi-event value > 0, "
                "otherwise ABSTAIN"
            ),
            "threshold_tuning_on_fresh": False,
        },
        "model": {
            "family": "HistGradientBoostingRegressor",
            "feature_count_value": len(value_head.columns),
            "feature_count_advantage": len(
                advantage_head.columns
            ),
            "value_seed": VALUE_SEED,
            "advantage_seed": ADVANTAGE_SEED,
        },
        "state_coverage": state_coverage,
        "fresh_value_diagnostics": value_diag,
        "fresh_advantage_diagnostics": advantage_diag,
        "candidate_path_counts": candidate_counts,
        "fresh_policy": {
            "request208": baseline_metrics,
            "request227": candidate_metrics,
            "request227_minus_request208": difference,
        },
        "frozen_gate": {
            "min_state_coverage": MIN_STATE_COVERAGE,
            "min_value_spearman": MIN_VALUE_SPEARMAN,
            "min_advantage_spearman": MIN_ADVANTAGE_SPEARMAN,
            "min_execution_rate": MIN_EXECUTION_RATE,
            "max_execution_rate": MAX_EXECUTION_RATE,
            "min_trades": MIN_TRADES,
            "candidate_day_balanced_mean_must_be_positive": True,
            "candidate_must_beat_request208": True,
            "matched_bootstrap_ci_low_must_be_positive": True,
            "min_improved_days": MIN_IMPROVED_DAYS,
            "severe_loss_no_worse_than_request208": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze the multi-event entry controller, integrate it with "
            "the best causal POSITION management nodes, and replay full "
            "trajectories before opening untouched dates."
        ),
        "next_boundary_if_fail": (
            "multi-event continuation is still not transferable enough "
            "from the current WATCH state; revisit candidate admission "
            "before HOT/WATCH transition rather than adding more "
            "post-entry management complexity."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.concat(
        [baseline_portfolio, candidate_portfolio],
        ignore_index=True,
    ).to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument(
        "--calibration-positions", type=Path, required=True
    )
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.fresh_dir,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
