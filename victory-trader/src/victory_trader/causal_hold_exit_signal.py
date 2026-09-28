"""Request267: causal HOLD-vs-EXIT observability after corrected entry."""

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
from sklearn.metrics import roc_auc_score

from .causal_minute_controller_rebuild import causal_execution_scan
from .full_hot_fixed_policy_value import (
    attach_fixed_value,
    event_lookup,
    usable_columns,
)
from .learned_pullback_entry import (
    CONTROLLER_TRAIN_DAYS,
    TEST_DAYS,
    _weights,
    controller_x,
)
from .recurrent_wait_entry_action_value import _base_return
from .rich_post_hot_state import (
    CURRENT_MINUTE_FEATURES,
    CURRENT_SECOND_FEATURES,
    DELTA_MINUTE_FEATURES,
    DELTA_SECOND_FEATURES,
    attach_dynamic_deltas,
    attach_minute_features,
    attach_second_features,
    score_and_select_candidates,
    usable,
)

REQUEST_ID = 267
POSITION_CAP_MINUTES = 30
MIN_SPEARMAN = 0.10
MIN_AUC = 0.60
MIN_SPEARMAN_GAIN = 0.05
MIN_AUC_GAIN = 0.03

POSITION_FEATURES = (
    "candidate_probability",
    "minutes_since_entry",
    "events_since_entry",
    "current_return_from_entry_pct",
    "running_high_return_from_entry_pct",
    "running_low_return_from_entry_pct",
    "drawdown_from_running_high_pct",
    "recovery_from_running_low_pct",
    "last_event_return_pct",
    "two_event_return_pct",
    "path_range_pct",
    "path_efficiency",
    "events_since_high",
    "events_since_low",
)


def build_position_states(
    selected: pd.DataFrame,
    causal_scan: pd.DataFrame,
) -> pd.DataFrame:
    lookup = event_lookup(causal_scan)
    static_columns = usable_columns(selected)
    records = []

    for row in selected.itertuples(index=False):
        day = str(row.trading_day)
        ticker = str(row.ticker).upper()
        hot_t = int(row.t)
        cap_t = hot_t + POSITION_CAP_MINUTES * 60_000
        events = [
            (int(t), float(price))
            for t, price in lookup.get((day, ticker), [])
            if hot_t < int(t) <= cap_t and float(price) > 0
        ]
        if len(events) < 2:
            continue
        entry_t, entry_price = events[0]
        path = events[1:]
        returns = np.asarray(
            [_base_return(entry_price, price) for _, price in path],
            dtype=float,
        )
        base = {
            column: getattr(row, column)
            for column in static_columns
            if hasattr(row, column)
        }
        prices = [entry_price]
        running_high = entry_price
        running_low = entry_price
        high_index = 0
        low_index = 0
        abs_move = 0.0

        for i, ((state_t, price), exit_now) in enumerate(
            zip(path, returns),
            start=1,
        ):
            prices.append(price)
            last_move = (
                (price / prices[-2] - 1.0) * 100.0
                if prices[-2] > 0
                else np.nan
            )
            if np.isfinite(last_move):
                abs_move += abs(last_move)
            if price >= running_high:
                running_high = price
                high_index = i
            if price <= running_low:
                running_low = price
                low_index = i

            later = returns[i:]
            later = later[np.isfinite(later)]
            best_later = (
                float(np.max(later)) if len(later) else float(exit_now)
            )
            hold_advantage = float(best_later - exit_now)
            current_return = (
                (price / entry_price - 1.0) * 100.0
            )
            two_move = (
                (price / prices[-3] - 1.0) * 100.0
                if len(prices) >= 3 and prices[-3] > 0
                else last_move
            )
            high_return = (
                (running_high / entry_price - 1.0) * 100.0
            )
            low_return = (
                (running_low / entry_price - 1.0) * 100.0
            )
            drawdown = (
                (price / running_high - 1.0) * 100.0
            )
            recovery = (
                (price / running_low - 1.0) * 100.0
                if running_low > 0
                else np.nan
            )
            path_range = (
                (running_high / running_low - 1.0) * 100.0
                if running_low > 0
                else np.nan
            )
            efficiency = (
                abs(current_return) / abs_move
                if abs_move > 1e-12
                else 1.0
            )
            records.append(
                {
                    **base,
                    "trading_day": day,
                    "ticker": ticker,
                    "hot_t": hot_t,
                    "entry_t": entry_t,
                    "entry_price": entry_price,
                    "state_t": state_t,
                    "state_price": price,
                    "candidate_probability": float(
                        row.candidate_probability
                    ),
                    "minutes_since_entry": float(
                        (state_t - entry_t) / 60_000
                    ),
                    "events_since_entry": i,
                    "current_return_from_entry_pct": current_return,
                    "running_high_return_from_entry_pct": high_return,
                    "running_low_return_from_entry_pct": low_return,
                    "drawdown_from_running_high_pct": drawdown,
                    "recovery_from_running_low_pct": recovery,
                    "last_event_return_pct": last_move,
                    "two_event_return_pct": two_move,
                    "path_range_pct": path_range,
                    "path_efficiency": efficiency,
                    "events_since_high": i - high_index,
                    "events_since_low": i - low_index,
                    "exit_now_pct": float(exit_now),
                    "hold_advantage_pct": hold_advantage,
                    "hold_optimal": hold_advantage > 0,
                }
            )
    return pd.DataFrame(records)


def path_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    static = usable_columns(frame)
    return usable(
        frame,
        [*static, *POSITION_FEATURES],
    )


def fit_models(
    train: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
):
    target = pd.to_numeric(
        train.hold_advantage_pct,
        errors="coerce",
    )
    valid = target.notna()
    fit = train.loc[valid].copy()
    y = pd.to_numeric(
        fit.hold_advantage_pct,
        errors="coerce",
    ).to_numpy(float)
    label = y > 0
    if len(fit) < 500 or len(np.unique(label)) != 2:
        raise ValueError("Request267 position support/classes insufficient")
    low, high = np.quantile(y, [0.005, 0.995])
    weights = _weights(fit)
    reg = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )
    cls = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed + 1,
    )
    x = controller_x(fit, columns)
    reg.fit(x, np.clip(y, low, high), sample_weight=weights)
    cls.fit(x, label.astype(int), sample_weight=weights)
    return reg, cls


def signal_report(
    test: pd.DataFrame,
    reg,
    cls,
    columns: tuple[str, ...],
) -> dict:
    target = pd.to_numeric(
        test.hold_advantage_pct,
        errors="coerce",
    )
    valid = target.notna()
    work = test.loc[valid].copy()
    target = pd.to_numeric(
        work.hold_advantage_pct,
        errors="coerce",
    )
    x = controller_x(work, columns)
    pred = reg.predict(x)
    prob = cls.predict_proba(x)[:, 1]
    spearman = pd.Series(pred).corr(
        pd.Series(target.to_numpy(float)),
        method="spearman",
    )
    y = target.gt(0).astype(int)
    auc = (
        float(roc_auc_score(y, prob))
        if y.nunique() == 2
        else None
    )
    return {
        "rows": int(len(work)),
        "feature_count": len(columns),
        "target_mean_pct": float(target.mean()),
        "hold_optimal_rate": float(y.mean()),
        "hold_advantage_spearman": (
            None if pd.isna(spearman) else float(spearman)
        ),
        "hold_auc": auc,
    }


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    causal_scan = causal_execution_scan(scan)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, causal_scan)
    scored, train_threshold, _ = score_and_select_candidates(
        first_hot
    )

    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(CONTROLLER_TRAIN_DAYS)
        & scored.candidate_probability.ge(train_threshold)
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_probability.ge(train_threshold)
    ].copy()

    train_states = build_position_states(
        train_selected,
        causal_scan,
    )
    test_states = build_position_states(
        test_selected,
        causal_scan,
    )
    combined = pd.concat(
        [
            train_states.assign(_split="train"),
            test_states.assign(_split="test"),
        ],
        ignore_index=True,
    )
    combined = attach_minute_features(combined, scan)
    combined, second_audit = attach_second_features(combined)
    combined = attach_dynamic_deltas(combined)
    train_states = combined.loc[
        combined._split.eq("train")
    ].drop(columns="_split")
    test_states = combined.loc[
        combined._split.eq("test")
    ].drop(columns="_split")

    path = path_columns(train_states)
    minute = usable(
        train_states,
        [*path, *CURRENT_MINUTE_FEATURES, *DELTA_MINUTE_FEATURES],
    )
    rich = usable(
        train_states,
        [*minute, *CURRENT_SECOND_FEATURES, *DELTA_SECOND_FEATURES],
    )

    models = {
        "path": fit_models(train_states, path, 20261392),
        "minute": fit_models(train_states, minute, 20261394),
        "rich": fit_models(train_states, rich, 20261396),
    }
    reports = {
        name: signal_report(
            test_states,
            fitted[0],
            fitted[1],
            {"path": path, "minute": minute, "rich": rich}[name],
        )
        for name, fitted in models.items()
    }
    base = reports["path"]
    rich_report = reports["rich"]
    spearman_gain = (
        rich_report["hold_advantage_spearman"]
        - base["hold_advantage_spearman"]
        if rich_report["hold_advantage_spearman"] is not None
        and base["hold_advantage_spearman"] is not None
        else None
    )
    auc_gain = (
        rich_report["hold_auc"] - base["hold_auc"]
        if rich_report["hold_auc"] is not None
        and base["hold_auc"] is not None
        else None
    )
    checks = {
        "rich_spearman": (
            rich_report["hold_advantage_spearman"] is not None
            and rich_report["hold_advantage_spearman"] >= MIN_SPEARMAN
        ),
        "rich_auc": (
            rich_report["hold_auc"] is not None
            and rich_report["hold_auc"] >= MIN_AUC
        ),
        "spearman_gain": (
            spearman_gain is not None
            and spearman_gain >= MIN_SPEARMAN_GAIN
        ),
        "auc_gain": (
            auc_gain is not None
            and auc_gain >= MIN_AUC_GAIN
        ),
    }
    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "causal_reference_contract": True,
        "candidate_threshold_top20": train_threshold,
        "support": {
            "train_candidate_episodes": int(len(train_selected)),
            "test_candidate_episodes": int(len(test_selected)),
            "train_position_states": int(len(train_states)),
            "test_position_states": int(len(test_states)),
        },
        "second_data_audit": second_audit,
        "representations": reports,
        "rich_minus_path": {
            "hold_advantage_spearman": spearman_gain,
            "hold_auc": auc_gain,
        },
        "checks": checks,
        "hold_exit_signal_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "Future-best exit is used only to construct a training/diagnostic "
            "optimal-stopping label. Model features remain causal current-state "
            "information only."
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
