"""Request 205: recurrent watch-entry plus recurrent HOLD/EXIT.

Development-only. Train on Request-171 FIT, evaluate Request-171 chronological
CALIBRATION, and keep Request-178 sealed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .causal_second_execution import SecondBar
from .causal_second_risk_compatible_entry import (
    BASE_SCENARIO,
    LATENCY_MS,
    SecondStore,
    _bars_from_seconds,
    _summarize,
    prepare_states,
)
from .execution_costs import modeled_sell_fill
from .future_cost_cover_state_observability import (
    FEATURES as POSITION_FEATURES,
    _day_weights,
    _safe_spearman,
)
from .market_regime_position_value import (
    attach_market_regime,
    build_market_regime,
)
from .recurrent_watch_entry_controller import (
    RULE_NAME,
    add_action_targets,
    fit_controller,
    run_controller,
    score_controller,
)
from .selected_hot_position_value_observability import (
    build_position_rows,
)

REQUEST_ID = 205
STOP_PCT = 3.0
MAX_HOLD_MS = 30 * 60_000
MODEL_SEED = 20261221
BOOTSTRAP_SEED = 20261222
BOOTSTRAP_SAMPLES = 10_000
MIN_TRADES = 30
MIN_POSITION_START_COVERAGE = 0.90
MIN_COMPLETION_COVERAGE = 0.90
MIN_POSITIVE_DAYS = 6
MIN_MATCHED_UPLIFT_PCT = 0.75


@dataclass(frozen=True)
class FutureBestModel:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    low: float
    high: float


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def train_future_best_model(
    fit: pd.DataFrame,
) -> FutureBestModel:
    target = pd.to_numeric(
        fit["best_future_base_return_pct"],
        errors="coerce",
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError(
            f"request 205 insufficient position fit support: {len(train)}"
        )
    columns = tuple(
        column
        for column in POSITION_FEATURES
        if column in train.columns
        and pd.to_numeric(
            train[column],
            errors="coerce",
        ).notna().any()
    )
    if not columns:
        raise ValueError("request 205 has no usable position features")
    low, high = np.quantile(
        y.to_numpy(dtype=float),
        [0.005, 0.995],
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=MODEL_SEED,
    )
    model.fit(
        _feature_frame(train, columns),
        y.clip(lower=low, upper=high),
        sample_weight=_day_weights(train),
    )
    return FutureBestModel(
        model=model,
        columns=columns,
        low=float(low),
        high=float(high),
    )


def predict_future_best(
    frame: pd.DataFrame,
    fitted: FutureBestModel,
) -> np.ndarray:
    return fitted.model.predict(
        _feature_frame(frame, fitted.columns)
    )


def _sell_return_pct(
    entry_modeled_price: float,
    exit_reference_price: float,
) -> float:
    if not (
        np.isfinite(entry_modeled_price)
        and entry_modeled_price > 0
        and np.isfinite(exit_reference_price)
        and exit_reference_price > 0
    ):
        return np.nan
    proceeds = modeled_sell_fill(
        float(exit_reference_price),
        BASE_SCENARIO,
    )
    proceeds *= (
        1.0
        - BASE_SCENARIO.sell_fee_bps / 10_000.0
    )
    return float(
        (proceeds / float(entry_modeled_price) - 1.0)
        * 100.0
    )


def add_exact_entry_marks(
    positions: pd.DataFrame,
    entries: pd.DataFrame,
) -> pd.DataFrame:
    info = entries.loc[
        :,
        [
            "trading_day",
            "ticker",
            "hot_t",
            "state_t",
            "entry_fill_t",
            "entry_reference_price",
            "entry_modeled_price",
        ],
    ].copy()
    info = info.rename(
        columns={
            "hot_t": "origin_hot_t",
            "state_t": "hot_t",
            "entry_fill_t": "exact_entry_fill_t",
            "entry_reference_price": "exact_entry_reference_price",
            "entry_modeled_price": "exact_entry_modeled_price",
        }
    )
    info["trading_day"] = info["trading_day"].astype(str)
    info["ticker"] = info["ticker"].astype(str).str.upper()
    info["hot_t"] = pd.to_numeric(
        info["hot_t"],
        errors="raise",
    ).astype("int64")
    if info.duplicated(
        ["trading_day", "ticker", "hot_t"]
    ).any():
        raise ValueError("request 205 duplicate dynamic entry key")

    result = positions.copy()
    result["trading_day"] = result[
        "trading_day"
    ].astype(str)
    result["ticker"] = result[
        "ticker"
    ].astype(str).str.upper()
    result = result.merge(
        info,
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="many_to_one",
    )
    entry_modeled = pd.to_numeric(
        result["exact_entry_modeled_price"],
        errors="coerce",
    )
    current_close = np.exp(
        pd.to_numeric(
            result["log_current_close"],
            errors="coerce",
        )
    )
    result["causal_mark_base_return_pct"] = [
        _sell_return_pct(
            float(entry),
            float(close),
        )
        if pd.notna(entry)
        and pd.notna(close)
        else np.nan
        for entry, close in zip(
            entry_modeled,
            current_close,
            strict=True,
        )
    ]
    return result


def score_position_states(
    positions: pd.DataFrame,
    fitted: FutureBestModel,
) -> pd.DataFrame:
    result = positions.copy()
    result["predicted_best_future_base_return_pct"] = (
        predict_future_best(
            result,
            fitted,
        )
    )
    mark = pd.to_numeric(
        result["causal_mark_base_return_pct"],
        errors="coerce",
    )
    future = pd.to_numeric(
        result[
            "predicted_best_future_base_return_pct"
        ],
        errors="coerce",
    )
    result["position_action"] = np.where(
        future.notna()
        & mark.notna()
        & future.gt(mark),
        "HOLD",
        "EXIT",
    )
    return result


def _first_stop_trigger(
    bars: list[SecondBar],
    *,
    entry_fill_t: int,
    entry_modeled_price: float,
) -> int | None:
    stop_level = float(entry_modeled_price) * (
        1.0 - STOP_PCT / 100.0
    )
    for bar in bars:
        if bar.halted or bar.t < int(entry_fill_t):
            continue
        if bar.low <= stop_level:
            return int(bar.t + 1_000)
    return None


def _first_fill_after(
    bars: list[SecondBar],
    trigger_t: int,
) -> SecondBar | None:
    eligible_t = int(trigger_t) + LATENCY_MS
    for bar in bars:
        if bar.halted:
            continue
        if int(bar.t) >= eligible_t:
            return bar
    return None


def _trajectory_for_entry(
    entry: pd.Series,
    decisions: pd.DataFrame,
    store: SecondStore,
) -> dict[str, object]:
    day = str(entry["trading_day"])
    ticker = str(entry["ticker"]).upper()
    origin_hot_t = int(entry["hot_t"])
    entry_decision_t = int(entry["state_t"])
    entry_fill_t = pd.to_numeric(
        pd.Series([entry.get("entry_fill_t")]),
        errors="coerce",
    ).iloc[0]
    entry_modeled = pd.to_numeric(
        pd.Series([entry.get("entry_modeled_price")]),
        errors="coerce",
    ).iloc[0]
    if (
        pd.isna(entry_fill_t)
        or pd.isna(entry_modeled)
        or not np.isfinite(float(entry_modeled))
        or float(entry_modeled) <= 0
    ):
        return {
            "trading_day": day,
            "ticker": ticker,
            "origin_hot_t": origin_hot_t,
            "entry_decision_t": entry_decision_t,
            "status": "unresolved",
            "exit_reason": "missing_exact_entry",
            "base_net_return_pct": np.nan,
            "minutes_held": np.nan,
        }

    entry_fill_t = int(entry_fill_t)
    entry_modeled = float(entry_modeled)
    seconds = store.seconds(day, ticker)
    path = seconds.loc[
        pd.to_numeric(
            seconds["t"],
            errors="coerce",
        ).ge(entry_fill_t)
    ].copy()
    bars = _bars_from_seconds(
        path,
        ticker=ticker,
        intervals=store.halts(day),
    )
    stop_trigger = _first_stop_trigger(
        bars,
        entry_fill_t=entry_fill_t,
        entry_modeled_price=entry_modeled,
    )

    group = decisions.sort_values(
        "state_t",
        kind="stable",
    )
    exit_states = group.loc[
        group["position_action"].astype(str).eq("EXIT")
    ]
    discretionary_trigger = (
        int(exit_states.iloc[0]["state_t"])
        if len(exit_states)
        else None
    )
    cap_trigger = int(
        entry_fill_t + MAX_HOLD_MS
    )

    triggers: list[tuple[int, int, str]] = []
    if stop_trigger is not None:
        triggers.append(
            (int(stop_trigger), 0, "hard_stop")
        )
    if discretionary_trigger is not None:
        triggers.append(
            (
                int(discretionary_trigger),
                1,
                "model_exit",
            )
        )
    triggers.append(
        (cap_trigger, 2, "time_cap")
    )
    trigger_t, _, reason = min(
        triggers,
        key=lambda item: (item[0], item[1]),
    )
    fill = _first_fill_after(
        bars,
        trigger_t,
    )
    if fill is None:
        return {
            "trading_day": day,
            "ticker": ticker,
            "origin_hot_t": origin_hot_t,
            "entry_decision_t": entry_decision_t,
            "status": "unresolved",
            "exit_reason": f"{reason}_unfilled",
            "base_net_return_pct": np.nan,
            "minutes_held": np.nan,
        }

    realized = _sell_return_pct(
        entry_modeled,
        float(fill.o),
    )
    minutes_held = (
        int(fill.t) - entry_fill_t
    ) / 60_000.0
    return {
        "trading_day": day,
        "ticker": ticker,
        "origin_hot_t": origin_hot_t,
        "entry_decision_t": entry_decision_t,
        "status": "completed",
        "exit_reason": reason,
        "base_net_return_pct": realized,
        "minutes_held": float(minutes_held),
        "entry_modeled_price": entry_modeled,
        "exit_reference_price": float(fill.o),
        "exit_fill_t": int(fill.t),
    }


def build_trajectories(
    entries: pd.DataFrame,
    scored_positions: pd.DataFrame,
    store: SecondStore,
) -> pd.DataFrame:
    groups = {
        (
            str(day),
            str(ticker).upper(),
            int(entry_t),
        ): group.copy()
        for (day, ticker, entry_t), group in (
            scored_positions.groupby(
                ["trading_day", "ticker", "hot_t"],
                sort=False,
            )
        )
    }
    rows: list[dict[str, object]] = []
    for _, entry in entries.iterrows():
        key = (
            str(entry["trading_day"]),
            str(entry["ticker"]).upper(),
            int(entry["state_t"]),
        )
        decisions = groups.get(
            key,
            scored_positions.iloc[0:0].copy(),
        )
        rows.append(
            _trajectory_for_entry(
                entry,
                decisions,
                store,
            )
        )
    return pd.DataFrame(rows)


def _daily_bootstrap(
    frame: pd.DataFrame,
) -> dict[str, object]:
    completed = frame.loc[
        frame["status"].astype(str).eq("completed")
    ].copy()
    values = pd.to_numeric(
        completed["base_net_return_pct"],
        errors="coerce",
    )
    valid = values.notna()
    work = completed.loc[valid].copy()
    work["_value"] = values.loc[valid]
    daily = (
        work.groupby(
            work["trading_day"].astype(str),
            sort=True,
        )["_value"]
        .mean()
        .dropna()
    )
    if daily.empty:
        return {
            "days": 0,
            "mean_pct": None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    vals = daily.to_numpy(dtype=float)
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )
    samples = rng.choice(
        vals,
        size=(BOOTSTRAP_SAMPLES, len(vals)),
        replace=True,
    )
    means = samples.mean(axis=1)
    return {
        "days": int(len(vals)),
        "mean_pct": float(vals.mean()),
        "ci_low_pct": float(
            np.quantile(means, 0.025)
        ),
        "ci_high_pct": float(
            np.quantile(means, 0.975)
        ),
    }


def summarize_trajectories(
    frame: pd.DataFrame,
) -> dict[str, object]:
    total = int(len(frame))
    completed = frame.loc[
        frame["status"].astype(str).eq("completed")
    ].copy()
    values = pd.to_numeric(
        completed["base_net_return_pct"],
        errors="coerce",
    )
    valid = values.notna()
    completed = completed.loc[valid].copy()
    values = values.loc[valid]
    completed["_value"] = values
    wins = values.loc[values.gt(0)]
    losses = values.loc[values.lt(0)]
    daily = (
        completed.groupby(
            completed["trading_day"].astype(str),
            sort=True,
        )["_value"]
        .mean()
    )
    reasons = (
        frame["exit_reason"]
        .fillna("missing")
        .astype(str)
        .value_counts()
        .sort_index()
    )
    return {
        "episodes": total,
        "completed": int(len(completed)),
        "completed_coverage": (
            float(len(completed) / total)
            if total
            else None
        ),
        "mean_net_return_pct": (
            float(values.mean())
            if len(values)
            else None
        ),
        "day_balanced_net_return_pct": (
            float(daily.mean())
            if len(daily)
            else None
        ),
        "positive_rate": (
            float(values.gt(0).mean())
            if len(values)
            else None
        ),
        "mean_winner_pct": (
            float(wins.mean())
            if len(wins)
            else None
        ),
        "mean_loser_pct": (
            float(losses.mean())
            if len(losses)
            else None
        ),
        "payoff_ratio": (
            float(
                wins.mean()
                / abs(losses.mean())
            )
            if len(wins)
            and len(losses)
            and float(losses.mean()) != 0
            else None
        ),
        "severe_loss_rate": (
            float(values.le(-5.0).mean())
            if len(values)
            else None
        ),
        "positive_days": int(
            daily.gt(0).sum()
        ),
        "by_day_mean_pct": {
            str(k): float(v)
            for k, v in daily.items()
        },
        "mean_holding_minutes": (
            float(
                pd.to_numeric(
                    completed["minutes_held"],
                    errors="coerce",
                ).mean()
            )
            if len(completed)
            else None
        ),
        "exit_reason_counts": {
            str(k): int(v)
            for k, v in reasons.items()
        },
        "daily_bootstrap": _daily_bootstrap(
            frame
        ),
    }


def matched_improvement(
    frozen: pd.DataFrame,
    dynamic: pd.DataFrame,
) -> dict[str, object]:
    left = frozen.loc[
        :,
        [
            "trading_day",
            "ticker",
            "hot_t",
            "replay_status",
            "policy_net_return_pct",
        ],
    ].rename(
        columns={
            "replay_status": "frozen_status",
            "policy_net_return_pct": "frozen_return",
        }
    )
    right = dynamic.loc[
        :,
        [
            "trading_day",
            "ticker",
            "origin_hot_t",
            "status",
            "base_net_return_pct",
        ],
    ].rename(
        columns={
            "origin_hot_t": "hot_t",
            "status": "dynamic_status",
            "base_net_return_pct": "dynamic_return",
        }
    )
    merged = left.merge(
        right,
        on=["trading_day", "ticker", "hot_t"],
        how="inner",
        validate="one_to_one",
    )
    frozen_value = pd.to_numeric(
        merged["frozen_return"],
        errors="coerce",
    )
    dynamic_value = pd.to_numeric(
        merged["dynamic_return"],
        errors="coerce",
    )
    valid = (
        merged["frozen_status"].astype(str).eq("closed")
        & merged["dynamic_status"].astype(str).eq("completed")
        & frozen_value.notna()
        & dynamic_value.notna()
    )
    work = merged.loc[valid].copy()
    if work.empty:
        return {
            "matched_closed": 0,
            "mean_improvement_pct": None,
            "day_balanced_improvement_pct": None,
        }
    work["_delta"] = (
        dynamic_value.loc[valid].to_numpy(dtype=float)
        - frozen_value.loc[valid].to_numpy(dtype=float)
    )
    daily = work.groupby(
        work["trading_day"].astype(str),
        sort=True,
    )["_delta"].mean()
    return {
        "matched_closed": int(len(work)),
        "mean_improvement_pct": float(
            work["_delta"].mean()
        ),
        "day_balanced_improvement_pct": float(
            daily.mean()
        ),
        "by_day_improvement_pct": {
            str(k): float(v)
            for k, v in daily.items()
        },
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(
        fit_path
    )
    cal_positions = pd.read_parquet(
        calibration_path
    )
    scan = pd.read_parquet(
        history_scan_path
    )

    fit_days = set(
        fit_positions["trading_day"].astype(str)
    )
    cal_days = set(
        cal_positions["trading_day"].astype(str)
    )
    fit_scan = scan.loc[
        scan["trading_day"].astype(str).isin(
            fit_days
        )
    ].copy()
    cal_scan = scan.loc[
        scan["trading_day"].astype(str).isin(
            cal_days
        )
    ].copy()

    store = SecondStore()
    _, fit_by_rule = prepare_states(
        fit_positions,
        fit_scan,
        store,
    )
    _, cal_by_rule = prepare_states(
        cal_positions,
        cal_scan,
        store,
    )
    fit_entry = add_action_targets(
        fit_by_rule[RULE_NAME]
    )
    cal_entry = add_action_targets(
        cal_by_rule[RULE_NAME]
    )
    entry_now, entry_wait = fit_controller(
        fit_entry,
        include_watch_path=False,
    )
    cal_entry = score_controller(
        cal_entry,
        entry_now,
        entry_wait,
        prefix="entry",
    )
    selected_entries, entry_audit = (
        run_controller(
            cal_entry,
            prefix="entry",
        )
    )
    frozen_summary = _summarize(
        selected_entries
    )

    fit_regime = build_market_regime(
        fit_scan
    )
    fit_position_train = (
        attach_market_regime(
            fit_positions,
            fit_regime,
        )
    )
    future_model = train_future_best_model(
        fit_position_train
    )

    anchors = selected_entries.loc[
        :,
        ["trading_day", "ticker", "state_t"],
    ].rename(
        columns={"state_t": "t"}
    )
    eval_positions, position_audit = (
        build_position_rows(
            anchors,
            cal_scan,
            store.client,
        )
    )
    cal_regime = build_market_regime(
        cal_scan
    )
    eval_positions = attach_market_regime(
        eval_positions,
        cal_regime,
    )
    eval_positions = add_exact_entry_marks(
        eval_positions,
        selected_entries,
    )
    eval_positions = score_position_states(
        eval_positions,
        future_model,
    )

    actual_future = pd.to_numeric(
        eval_positions[
            "best_future_base_return_pct"
        ],
        errors="coerce",
    )
    predicted_future = pd.to_numeric(
        eval_positions[
            "predicted_best_future_base_return_pct"
        ],
        errors="coerce",
    )
    value_spearman = _safe_spearman(
        actual_future,
        predicted_future,
    )

    trajectories = build_trajectories(
        selected_entries,
        eval_positions,
        store,
    )
    dynamic_summary = (
        summarize_trajectories(
            trajectories
        )
    )
    matched = matched_improvement(
        selected_entries,
        trajectories,
    )

    started_keys = (
        eval_positions.loc[
            :,
            ["trading_day", "ticker", "hot_t"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    position_start_coverage = (
        float(
            started_keys
            / len(selected_entries)
        )
        if len(selected_entries)
        else None
    )

    dynamic_day = dynamic_summary[
        "day_balanced_net_return_pct"
    ]
    dynamic_ci_low = dynamic_summary[
        "daily_bootstrap"
    ]["ci_low_pct"]
    frozen_severe = frozen_summary[
        "severe_loss_rate"
    ]
    dynamic_severe = dynamic_summary[
        "severe_loss_rate"
    ]
    matched_day = matched[
        "day_balanced_improvement_pct"
    ]

    checks = {
        "at_least_30_entries": bool(
            len(selected_entries) >= MIN_TRADES
        ),
        "position_start_coverage_at_least_90pct": bool(
            position_start_coverage is not None
            and position_start_coverage
            >= MIN_POSITION_START_COVERAGE
        ),
        "completed_coverage_at_least_90pct": bool(
            dynamic_summary[
                "completed_coverage"
            ]
            is not None
            and float(
                dynamic_summary[
                    "completed_coverage"
                ]
            )
            >= MIN_COMPLETION_COVERAGE
        ),
        "positive_day_balanced_return": bool(
            dynamic_day is not None
            and float(dynamic_day) > 0.0
        ),
        "at_least_6_positive_days": bool(
            int(
                dynamic_summary[
                    "positive_days"
                ]
            )
            >= MIN_POSITIVE_DAYS
        ),
        "bootstrap_lower_above_zero": bool(
            dynamic_ci_low is not None
            and float(dynamic_ci_low) > 0.0
        ),
        "matched_uplift_at_least_075pp": bool(
            matched_day is not None
            and float(matched_day)
            >= MIN_MATCHED_UPLIFT_PCT
        ),
        "severe_loss_not_worse": bool(
            frozen_severe is not None
            and dynamic_severe is not None
            and float(dynamic_severe)
            <= float(frozen_severe)
        ),
    }
    checks["all_pass"] = all(
        checks.values()
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "fresh_evaluated": False,
        "promotion_eligible": False,
        "entry_controller": {
            "audit": entry_audit,
            "frozen_runner_summary": (
                frozen_summary
            ),
        },
        "position_value_model": {
            "calibration_position_state_spearman": (
                value_spearman
            ),
            "feature_count": int(
                len(future_model.columns)
            ),
        },
        "position_construction": {
            **position_audit,
            "selected_entries": int(
                len(selected_entries)
            ),
            "position_started_entries": int(
                started_keys
            ),
            "position_start_coverage": (
                position_start_coverage
            ),
        },
        "recurrent_hold_exit": {
            "summary": dynamic_summary,
            "matched_vs_frozen_runner": matched,
        },
        "development_checks": checks,
        "development_signal_pass": bool(
            checks["all_pass"]
        ),
        "halt_feed_by_day": dict(
            sorted(store.halt_status.items())
        ),
        "second_client_stats": (
            store.client.stats.to_dict()
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit",
        type=Path,
        required=True,
    )
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
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
