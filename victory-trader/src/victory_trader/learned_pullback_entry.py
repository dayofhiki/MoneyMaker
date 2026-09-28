"""Request254: learned causal pullback ENTER/WAIT controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from .attention_replay import MINUTE_MS
from .causal_pullback_entry_audit import (
    ENTRY_WINDOW_MINUTES,
    fixed_value_after_entry,
    pick_entry,
)
from .extended_history_economic_opportunity import EXTENDED_CAL_DAYS
from .full_hot_fixed_policy_value import (
    attach_fixed_value,
    event_lookup,
    usable_columns,
    xframe,
)

REQUEST_ID = 254
CANDIDATE_MODEL_DAYS = ["2026-04-30", "2026-05-01", "2026-05-04"]
CONTROLLER_TRAIN_DAYS = [
    "2026-05-05",
    "2026-05-06",
    "2026-05-07",
    "2026-05-08",
]
TEST_DAYS = EXTENDED_CAL_DAYS
CANDIDATE_FRACTION = 0.05
MIN_CONTROLLER_TRAIN_STATES = 250
MIN_TARGET_ROWS = 200
MIN_TEST_STATES = 250
MIN_ENTER_SPEARMAN = 0.10
MIN_ADVANTAGE_SPEARMAN = 0.03
MIN_LEARNED_ENTRIES = 20
MIN_ENTRY_RATE = 0.10
MAX_ENTRY_RATE = 0.70
MIN_RESOLUTION_OR_CASH = 0.90
MIN_POSITIVE_MEAN_DAYS = 5
MAX_SEVERE_RATE = 0.15
MIN_GAIN_VS_IMMEDIATE = 0.50
MIN_GAIN_VS_FIXED_PULLBACK = 0.20

PATH_FEATURES = (
    "candidate_probability",
    "elapsed_minutes",
    "event_index",
    "current_return_from_first_pct",
    "running_high_return_from_first_pct",
    "running_low_return_from_first_pct",
    "drawdown_from_running_high_pct",
    "recovery_from_running_low_pct",
    "recovery_fraction_of_range",
    "last_event_return_pct",
    "two_event_return_pct",
    "path_range_pct",
    "path_efficiency",
    "events_since_high",
    "events_since_low",
)


def fit_candidate_model(
    first_hot: pd.DataFrame,
) -> tuple[HistGradientBoostingClassifier, tuple[str, ...]]:
    fit = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CANDIDATE_MODEL_DAYS)
    ].copy()
    target = pd.to_numeric(
        fit.fixed_first_watch_value_pct,
        errors="coerce",
    )
    fit = fit.loc[target.notna()].copy()
    target = pd.to_numeric(
        fit.fixed_first_watch_value_pct,
        errors="coerce",
    )
    columns = usable_columns(fit)
    y = target.gt(0).astype(int)
    if len(fit) < 500 or y.nunique() != 2:
        raise ValueError(
            "Request254 candidate fit lacks required support/classes"
        )
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=20261355,
    )
    model.fit(xframe(fit, columns), y)
    return model, columns


def score_candidates(
    first_hot: pd.DataFrame,
    model: HistGradientBoostingClassifier,
    columns: tuple[str, ...],
) -> tuple[pd.DataFrame, float]:
    scored = first_hot.copy()
    scored["candidate_probability"] = model.predict_proba(
        xframe(scored, columns)
    )[:, 1]
    threshold_source = scored.loc[
        scored.trading_day.astype(str).isin(CONTROLLER_TRAIN_DAYS),
        "candidate_probability",
    ]
    if len(threshold_source) < 500:
        raise ValueError("Request254 candidate threshold support too small")
    threshold = float(
        np.quantile(
            threshold_source.to_numpy(float),
            1.0 - CANDIDATE_FRACTION,
        )
    )
    scored["candidate_selected"] = scored.candidate_probability.ge(
        threshold
    )
    return scored, threshold


def _pct(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    if denominator <= 0:
        return np.nan
    return float((numerator / denominator - 1.0) * 100.0)


def build_episode_states(
    selected: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    lookup = event_lookup(scan)
    static_columns = usable_columns(selected)
    records: list[dict[str, object]] = []

    for row in selected.itertuples(index=False):
        day = str(row.trading_day)
        ticker = str(row.ticker).upper()
        hot_t = int(row.t)
        episode = lookup.get((day, ticker), [])
        limit_t = hot_t + ENTRY_WINDOW_MINUTES * MINUTE_MS
        events = [
            (int(t), float(price))
            for t, price in episode
            if hot_t < int(t) <= limit_t and float(price) > 0
        ]
        if not events:
            continue

        first_price = float(events[0][1])
        running_high = first_price
        running_low = first_price
        high_index = 0
        low_index = 0
        path_abs_move = 0.0
        prices: list[float] = []

        base = {
            column: getattr(row, column)
            for column in static_columns
            if hasattr(row, column)
        }
        candidate_probability = float(row.candidate_probability)

        provisional: list[dict[str, object]] = []
        for event_index, (state_t, price) in enumerate(events):
            prices.append(price)
            if event_index:
                path_abs_move += abs(
                    _pct(price, prices[event_index - 1])
                )
            if price >= running_high:
                running_high = price
                high_index = event_index
            if price <= running_low:
                running_low = price
                low_index = event_index

            current_return = _pct(price, first_price)
            high_return = _pct(running_high, first_price)
            low_return = _pct(running_low, first_price)
            drawdown = _pct(price, running_high)
            recovery = _pct(price, running_low)
            path_range = _pct(running_high, running_low)
            range_den = max(running_high - running_low, 1e-12)
            recovery_fraction = float(
                (price - running_low) / range_den
            )
            last_return = (
                _pct(price, prices[event_index - 1])
                if event_index >= 1
                else 0.0
            )
            two_return = (
                _pct(price, prices[event_index - 2])
                if event_index >= 2
                else last_return
            )
            displacement = abs(current_return)
            efficiency = (
                float(displacement / path_abs_move)
                if path_abs_move > 1e-12
                else 1.0
            )

            enter_value = fixed_value_after_entry(
                episode,
                hot_t,
                (state_t, price),
            )
            state = {
                **base,
                "trading_day": day,
                "ticker": ticker,
                "hot_t": hot_t,
                "state_t": state_t,
                "state_price": price,
                "candidate_probability": candidate_probability,
                "elapsed_minutes": float(
                    (state_t - hot_t) / MINUTE_MS
                ),
                "event_index": int(event_index),
                "current_return_from_first_pct": current_return,
                "running_high_return_from_first_pct": high_return,
                "running_low_return_from_first_pct": low_return,
                "drawdown_from_running_high_pct": drawdown,
                "recovery_from_running_low_pct": recovery,
                "recovery_fraction_of_range": recovery_fraction,
                "last_event_return_pct": last_return,
                "two_event_return_pct": two_return,
                "path_range_pct": path_range,
                "path_efficiency": efficiency,
                "events_since_high": int(event_index - high_index),
                "events_since_low": int(event_index - low_index),
                "enter_value_pct": enter_value,
            }
            provisional.append(state)

        for index, state in enumerate(provisional):
            enter_value = pd.to_numeric(
                pd.Series([state["enter_value_pct"]]),
                errors="coerce",
            ).iloc[0]
            if index + 1 < len(provisional):
                wait_value = pd.to_numeric(
                    pd.Series(
                        [provisional[index + 1]["enter_value_pct"]]
                    ),
                    errors="coerce",
                ).iloc[0]
            else:
                wait_value = 0.0
            state["wait_one_state_value_pct"] = (
                float(wait_value) if pd.notna(wait_value) else np.nan
            )
            state["enter_minus_wait_pct"] = (
                float(enter_value - wait_value)
                if pd.notna(enter_value) and pd.notna(wait_value)
                else np.nan
            )
            records.append(state)

    return pd.DataFrame(records)


def controller_columns(states: pd.DataFrame) -> tuple[str, ...]:
    static = usable_columns(states)
    return tuple(
        column
        for column in dict.fromkeys([*static, *PATH_FEATURES])
        if column in states
        and pd.to_numeric(states[column], errors="coerce").notna().any()
    )


def controller_x(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return (
        frame.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def _weights(frame: pd.DataFrame) -> np.ndarray:
    episode_key = (
        frame.trading_day.astype(str)
        + "|"
        + frame.ticker.astype(str)
        + "|"
        + frame.hot_t.astype(str)
    )
    per_episode = episode_key.map(episode_key.value_counts()).astype(float)
    per_day = frame.trading_day.astype(str).map(
        frame.trading_day.astype(str).value_counts()
    ).astype(float)
    weights = 1.0 / per_episode.to_numpy(float)
    weights *= 1.0 / per_day.to_numpy(float)
    return weights / float(np.mean(weights))


def fit_regressor(
    frame: pd.DataFrame,
    target_column: str,
    columns: tuple[str, ...],
    seed: int,
) -> tuple[HistGradientBoostingRegressor, dict]:
    target = pd.to_numeric(frame[target_column], errors="coerce")
    train = frame.loc[target.notna()].copy()
    y = pd.to_numeric(
        train[target_column],
        errors="coerce",
    ).to_numpy(float)
    if len(train) < MIN_TARGET_ROWS:
        raise ValueError(
            f"Request254 {target_column} finite target support only {len(train)}"
        )
    low, high = np.quantile(y, [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(
        controller_x(train, columns),
        np.clip(y, low, high),
        sample_weight=_weights(train),
    )
    return model, {
        "rows": int(len(train)),
        "mean_pct": float(np.mean(y)),
        "positive_rate": float(np.mean(y > 0)),
        "winsor": [float(low), float(high)],
    }


def safe_spearman(actual: pd.Series, predicted: np.ndarray) -> float | None:
    actual = pd.to_numeric(actual, errors="coerce")
    predicted_series = pd.Series(predicted, index=actual.index)
    valid = actual.notna() & predicted_series.notna()
    if int(valid.sum()) < 20:
        return None
    value = actual.loc[valid].corr(
        predicted_series.loc[valid],
        method="spearman",
    )
    return None if pd.isna(value) else float(value)


def make_learned_policy(
    states: pd.DataFrame,
    enter_model: HistGradientBoostingRegressor,
    advantage_model: HistGradientBoostingRegressor,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    work = states.copy()
    work["pred_enter_value_pct"] = enter_model.predict(
        controller_x(work, columns)
    )
    work["pred_enter_minus_wait_pct"] = advantage_model.predict(
        controller_x(work, columns)
    )
    rows = []
    for key, group in work.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        ordered = group.sort_values("state_t")
        chosen = None
        for position, (_, row) in enumerate(ordered.iterrows()):
            last_state = position == len(ordered) - 1
            enter_good = float(row.pred_enter_value_pct) > 0
            advantage_good = (
                float(row.pred_enter_minus_wait_pct) >= 0
            )
            if enter_good and (advantage_good or last_state):
                chosen = row
                break
        record = {
            "trading_day": str(key[0]),
            "ticker": str(key[1]).upper(),
            "hot_t": int(key[2]),
            "entered": chosen is not None,
            "entry_t": (
                int(chosen.state_t) if chosen is not None else None
            ),
            "entry_price": (
                float(chosen.state_price) if chosen is not None else None
            ),
            "value_pct": (
                float(chosen.enter_value_pct)
                if chosen is not None
                and pd.notna(chosen.enter_value_pct)
                else np.nan
            ),
            "decision_states": int(len(ordered)),
        }
        rows.append(record)
    return pd.DataFrame(rows)


def baseline_policy(
    selected: pd.DataFrame,
    scan: pd.DataFrame,
    rule: str,
) -> pd.DataFrame:
    lookup = event_lookup(scan)
    rows = []
    for row in selected.itertuples(index=False):
        day = str(row.trading_day)
        ticker = str(row.ticker).upper()
        hot_t = int(row.t)
        episode = lookup.get((day, ticker), [])
        limit_t = hot_t + ENTRY_WINDOW_MINUTES * MINUTE_MS
        events = [
            (int(t), float(price))
            for t, price in episode
            if hot_t < int(t) <= limit_t and float(price) > 0
        ]
        entry = pick_entry(events, rule)
        value = fixed_value_after_entry(
            episode,
            hot_t,
            entry,
        )
        rows.append(
            {
                "trading_day": day,
                "ticker": ticker,
                "hot_t": hot_t,
                "entered": entry is not None,
                "entry_t": entry[0] if entry is not None else None,
                "entry_price": (
                    entry[1] if entry is not None else None
                ),
                "value_pct": value,
            }
        )
    return pd.DataFrame(rows)


def policy_metrics(
    rows: pd.DataFrame,
    candidate_count: int,
) -> dict:
    work = rows.copy()
    entered = work.entered.astype(bool)
    value = pd.to_numeric(work.value_pct, errors="coerce")
    unresolved = entered & value.isna()
    work["economic_return_pct"] = np.where(
        ~entered,
        0.0,
        value,
    )
    resolved = work.loc[~unresolved].copy()
    returns = pd.to_numeric(
        resolved.economic_return_pct,
        errors="coerce",
    )
    daily = resolved.groupby(
        resolved.trading_day.astype(str),
        sort=True,
    ).economic_return_pct.mean()
    return {
        "candidate_episodes": int(candidate_count),
        "entries": int(entered.sum()),
        "entry_rate": (
            float(entered.mean()) if len(work) else 0.0
        ),
        "cash_episodes": int((~entered).sum()),
        "unresolved_entries": int(unresolved.sum()),
        "resolution_or_cash_rate": (
            float((~unresolved).mean()) if len(work) else 0.0
        ),
        "mean_pct": float(returns.mean()) if len(returns) else None,
        "day_balanced_mean_pct": (
            float(daily.mean()) if len(daily) else None
        ),
        "positive_rate": (
            float(returns.gt(0).mean()) if len(returns) else None
        ),
        "severe_loss_rate_le_minus2": (
            float(returns.le(-2).mean()) if len(returns) else None
        ),
        "positive_mean_days": int((daily > 0).sum()),
        "by_day": {
            str(day): {
                "episodes": int(len(part)),
                "mean_pct": float(part.economic_return_pct.mean()),
            }
            for day, part in resolved.groupby(
                resolved.trading_day.astype(str),
                sort=True,
            )
        },
    }


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, scan)

    candidate_model, candidate_columns = fit_candidate_model(first_hot)
    scored, candidate_threshold = score_candidates(
        first_hot,
        candidate_model,
        candidate_columns,
    )

    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(CONTROLLER_TRAIN_DAYS)
        & scored.candidate_selected
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_selected
    ].copy()

    train_states = build_episode_states(train_selected, scan)
    test_states = build_episode_states(test_selected, scan)
    columns = controller_columns(train_states)

    enter_model, enter_support = fit_regressor(
        train_states,
        "enter_value_pct",
        columns,
        seed=20261356,
    )
    advantage_model, advantage_support = fit_regressor(
        train_states,
        "enter_minus_wait_pct",
        columns,
        seed=20261357,
    )

    test_enter_pred = enter_model.predict(
        controller_x(test_states, columns)
    )
    test_advantage_pred = advantage_model.predict(
        controller_x(test_states, columns)
    )
    enter_spearman = safe_spearman(
        test_states.enter_value_pct,
        test_enter_pred,
    )
    advantage_spearman = safe_spearman(
        test_states.enter_minus_wait_pct,
        test_advantage_pred,
    )

    learned_rows = make_learned_policy(
        test_states,
        enter_model,
        advantage_model,
        columns,
    )
    immediate_rows = baseline_policy(
        test_selected,
        scan,
        "immediate",
    )
    fixed_pullback_rows = baseline_policy(
        test_selected,
        scan,
        "pullback_2",
    )

    candidate_count = int(len(test_selected))
    learned_metrics = policy_metrics(
        learned_rows,
        candidate_count,
    )
    immediate_metrics = policy_metrics(
        immediate_rows,
        candidate_count,
    )
    fixed_pullback_metrics = policy_metrics(
        fixed_pullback_rows,
        candidate_count,
    )

    learned_mean = learned_metrics["mean_pct"]
    immediate_mean = immediate_metrics["mean_pct"]
    fixed_pullback_mean = fixed_pullback_metrics["mean_pct"]
    gain_vs_immediate = (
        learned_mean - immediate_mean
        if learned_mean is not None and immediate_mean is not None
        else None
    )
    gain_vs_fixed_pullback = (
        learned_mean - fixed_pullback_mean
        if learned_mean is not None and fixed_pullback_mean is not None
        else None
    )

    checks = {
        "controller_train_states": (
            len(train_states) >= MIN_CONTROLLER_TRAIN_STATES
        ),
        "test_states": len(test_states) >= MIN_TEST_STATES,
        "enter_value_spearman": (
            enter_spearman is not None
            and enter_spearman >= MIN_ENTER_SPEARMAN
        ),
        "advantage_spearman": (
            advantage_spearman is not None
            and advantage_spearman >= MIN_ADVANTAGE_SPEARMAN
        ),
        "learned_entries": (
            learned_metrics["entries"] >= MIN_LEARNED_ENTRIES
        ),
        "entry_rate": (
            MIN_ENTRY_RATE
            <= learned_metrics["entry_rate"]
            <= MAX_ENTRY_RATE
        ),
        "resolution_or_cash_rate": (
            learned_metrics["resolution_or_cash_rate"]
            >= MIN_RESOLUTION_OR_CASH
        ),
        "learned_mean_positive": (
            learned_mean is not None and learned_mean > 0
        ),
        "day_balanced_mean_positive": (
            learned_metrics["day_balanced_mean_pct"] is not None
            and learned_metrics["day_balanced_mean_pct"] > 0
        ),
        "positive_mean_days": (
            learned_metrics["positive_mean_days"]
            >= MIN_POSITIVE_MEAN_DAYS
        ),
        "severe_loss_rate": (
            learned_metrics["severe_loss_rate_le_minus2"] is not None
            and learned_metrics["severe_loss_rate_le_minus2"]
            <= MAX_SEVERE_RATE
        ),
        "gain_vs_immediate": (
            gain_vs_immediate is not None
            and gain_vs_immediate >= MIN_GAIN_VS_IMMEDIATE
        ),
        "gain_vs_fixed_pullback": (
            gain_vs_fixed_pullback is not None
            and gain_vs_fixed_pullback >= MIN_GAIN_VS_FIXED_PULLBACK
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "candidate_model_days": CANDIDATE_MODEL_DAYS,
        "controller_train_days": CONTROLLER_TRAIN_DAYS,
        "development_test_days": list(TEST_DAYS),
        "candidate_fraction": CANDIDATE_FRACTION,
        "candidate_threshold": candidate_threshold,
        "selected_support": {
            "controller_train_episodes": int(len(train_selected)),
            "development_test_episodes": candidate_count,
            "controller_train_states": int(len(train_states)),
            "development_test_states": int(len(test_states)),
        },
        "controller_feature_count": len(columns),
        "controller_features": list(columns),
        "training_support": {
            "enter_value": enter_support,
            "enter_minus_wait": advantage_support,
        },
        "development_signal": {
            "enter_value_spearman": enter_spearman,
            "enter_minus_wait_spearman": advantage_spearman,
        },
        "development_policies": {
            "immediate": immediate_metrics,
            "fixed_2pct_pullback": fixed_pullback_metrics,
            "learned_enter_wait": learned_metrics,
        },
        "mean_gain_vs_immediate_pct": gain_vs_immediate,
        "mean_gain_vs_fixed_2pct_pullback_pct": gain_vs_fixed_pullback,
        "checks": checks,
        "economic_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "Development-only recurrent entry timing. May11-20 was already "
            "seen in prior requests, so this is not fresh validation. "
            "Minute aggregate opens are research references, not brokerage fills."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
