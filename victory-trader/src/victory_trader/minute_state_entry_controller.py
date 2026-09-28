"""Request258: recurrent entry controller using Request257 minute state."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from .learned_pullback_entry import (
    baseline_policy,
    build_episode_states,
    controller_columns,
    controller_x,
    fit_candidate_model,
    policy_metrics,
)
from .rich_post_hot_state import (
    CURRENT_MINUTE_FEATURES,
    DELTA_MINUTE_FEATURES,
    TEST_CANDIDATE_FRACTION,
    TRAIN_CANDIDATE_FRACTION,
    attach_dynamic_deltas,
    attach_minute_features,
    pullback_rows,
    score_and_select_candidates,
    usable,
)

REQUEST_ID = 258
MODEL_FIT_DAYS = ["2026-05-05", "2026-05-06", "2026-05-07"]
MODEL_CALIBRATION_DAY = "2026-05-08"
TEST_DAYS = [
    "2026-05-11",
    "2026-05-12",
    "2026-05-13",
    "2026-05-14",
    "2026-05-15",
    "2026-05-18",
    "2026-05-19",
    "2026-05-20",
]
PULLBACK_TRIGGER_PCT = 2.0
POSITIVE_PROBABILITY_THRESHOLD = 0.50
SEVERE_PROBABILITY_THRESHOLD = 0.50
MIN_FIT_PULLBACK_ROWS = 100
MIN_CAL_PULLBACK_ROWS = 25
MIN_TEST_PULLBACK_ROWS = 100
MIN_ENTRIES = 20
MIN_ENTRY_RATE = 0.10
MAX_ENTRY_RATE = 0.45
MIN_RESOLUTION_OR_CASH = 0.95
MIN_POSITIVE_MEAN_DAYS = 5
MAX_SEVERE_RATE = 0.08
MIN_GAIN_VS_FIXED_PULLBACK = 0.10


@dataclass(frozen=True)
class MinuteStateModels:
    value: HistGradientBoostingRegressor
    positive: HistGradientBoostingClassifier
    severe: HistGradientBoostingClassifier
    columns: tuple[str, ...]
    value_offset: float


def _day_episode_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = (
        frame.trading_day.astype(str)
        + "|"
        + frame.ticker.astype(str)
        + "|"
        + frame.hot_t.astype(str)
    )
    sizes = keys.map(keys.value_counts()).astype(float)
    episodes = (
        frame[["trading_day", "ticker", "hot_t"]]
        .drop_duplicates()
        .groupby("trading_day")
        .size()
    )
    ep_per_day = frame.trading_day.astype(str).map(episodes).astype(float)
    weights = 1.0 / sizes.to_numpy(float) / ep_per_day.to_numpy(float)
    return weights / float(np.mean(weights))


def minute_state_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    baseline = controller_columns(frame)
    return usable(
        frame,
        [
            *baseline,
            *CURRENT_MINUTE_FEATURES,
            *DELTA_MINUTE_FEATURES,
        ],
    )


def fit_models(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[MinuteStateModels, dict]:
    if len(fit) < MIN_FIT_PULLBACK_ROWS:
        raise ValueError(f"Request258 fit pullback support only {len(fit)}")
    if len(calibration) < MIN_CAL_PULLBACK_ROWS:
        raise ValueError(
            f"Request258 calibration pullback support only {len(calibration)}"
        )

    target = pd.to_numeric(fit.enter_value_pct, errors="coerce")
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].to_numpy(float)
    positive_y = y > 0
    severe_y = y <= -2.0
    if len(np.unique(positive_y)) != 2 or len(np.unique(severe_y)) != 2:
        raise ValueError("Request258 fit needs both positive/severe classes")

    weights = _day_episode_weights(train)
    low, high = np.quantile(y, [0.005, 0.995])
    value = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261370,
    )
    positive = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261371,
    )
    severe = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261372,
    )
    x = controller_x(train, columns)
    value.fit(x, np.clip(y, low, high), sample_weight=weights)
    positive.fit(x, positive_y, sample_weight=weights)
    severe.fit(x, severe_y, sample_weight=weights)

    cal_target = pd.to_numeric(
        calibration.enter_value_pct,
        errors="coerce",
    )
    cal = calibration.loc[cal_target.notna()].copy()
    cal_target = pd.to_numeric(cal.enter_value_pct, errors="coerce")
    raw = value.predict(controller_x(cal, columns))
    offset = float(cal_target.mean() - np.mean(raw))

    support = {
        "fit_rows": int(len(train)),
        "fit_mean_pct": float(np.mean(y)),
        "fit_positive_rate": float(np.mean(positive_y)),
        "fit_severe_rate": float(np.mean(severe_y)),
        "calibration_rows": int(len(cal)),
        "calibration_mean_pct": float(cal_target.mean()),
        "value_offset_pct": offset,
        "winsor": [float(low), float(high)],
    }
    return MinuteStateModels(
        value=value,
        positive=positive,
        severe=severe,
        columns=columns,
        value_offset=offset,
    ), support


def pullback_mask(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(
        frame.drawdown_from_running_high_pct,
        errors="coerce",
    ).le(-PULLBACK_TRIGGER_PCT)


def make_policy(
    states: pd.DataFrame,
    selected: pd.DataFrame,
    fitted: MinuteStateModels,
) -> pd.DataFrame:
    work = states.copy()
    if len(work):
        x = controller_x(work, fitted.columns)
        work["predicted_value_pct"] = (
            fitted.value.predict(x) + fitted.value_offset
        )
        work["positive_probability"] = fitted.positive.predict_proba(x)[:, 1]
        work["severe_probability"] = fitted.severe.predict_proba(x)[:, 1]
    else:
        for name in (
            "predicted_value_pct",
            "positive_probability",
            "severe_probability",
        ):
            work[name] = pd.Series(dtype=float)

    groups = {
        (str(k[0]), str(k[1]).upper(), int(k[2])): g.sort_values("state_t")
        for k, g in work.groupby(
            ["trading_day", "ticker", "hot_t"],
            sort=False,
        )
    }
    rows = []
    for row in selected.itertuples(index=False):
        key = (
            str(row.trading_day),
            str(row.ticker).upper(),
            int(row.t),
        )
        group = groups.get(key)
        chosen = None
        if group is not None:
            for _, state in group.loc[pullback_mask(group)].iterrows():
                if (
                    float(state.predicted_value_pct) > 0
                    and float(state.positive_probability)
                    >= POSITIVE_PROBABILITY_THRESHOLD
                    and float(state.severe_probability)
                    < SEVERE_PROBABILITY_THRESHOLD
                ):
                    chosen = state
                    break
        rows.append(
            {
                "trading_day": key[0],
                "ticker": key[1],
                "hot_t": key[2],
                "entered": chosen is not None,
                "entry_t": int(chosen.state_t) if chosen is not None else None,
                "entry_price": (
                    float(chosen.state_price) if chosen is not None else None
                ),
                "value_pct": (
                    float(chosen.enter_value_pct)
                    if chosen is not None
                    and pd.notna(chosen.enter_value_pct)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def signal_report(
    test: pd.DataFrame,
    fitted: MinuteStateModels,
) -> dict:
    if test.empty:
        return {"rows": 0}
    x = controller_x(test, fitted.columns)
    target = pd.to_numeric(test.enter_value_pct, errors="coerce")
    predicted = fitted.value.predict(x) + fitted.value_offset
    positive_probability = fitted.positive.predict_proba(x)[:, 1]
    severe_probability = fitted.severe.predict_proba(x)[:, 1]
    spearman = pd.Series(predicted).corr(
        pd.Series(target.to_numpy(float)),
        method="spearman",
    )

    from sklearn.metrics import roc_auc_score

    def auc(y, score):
        y = np.asarray(y, bool)
        return (
            float(roc_auc_score(y, score))
            if len(y) >= 20 and len(np.unique(y)) == 2
            else None
        )

    return {
        "rows": int(len(test)),
        "value_spearman": None if pd.isna(spearman) else float(spearman),
        "positive_auc": auc(target.gt(0), positive_probability),
        "severe_auc": auc(target.le(-2), severe_probability),
        "predicted_positive_value_rate": float(np.mean(predicted > 0)),
        "actual_mean_pct": float(target.mean()),
    }


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
) -> int:
    from .full_hot_fixed_policy_value import attach_fixed_value

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

    scored, train_threshold, test_threshold = score_and_select_candidates(
        first_hot
    )
    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(
            [*MODEL_FIT_DAYS, MODEL_CALIBRATION_DAY]
        )
        & scored.candidate_probability.ge(train_threshold)
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_probability.ge(test_threshold)
    ].copy()

    train_states = build_episode_states(train_selected, scan)
    test_states = build_episode_states(test_selected, scan)
    combined = pd.concat(
        [
            train_states.assign(_split="train"),
            test_states.assign(_split="test"),
        ],
        ignore_index=True,
    )
    combined = attach_minute_features(combined, scan)
    combined = attach_dynamic_deltas(combined)
    train_states = combined.loc[combined._split.eq("train")].drop(
        columns="_split"
    )
    test_states = combined.loc[combined._split.eq("test")].drop(
        columns="_split"
    )

    train_pullback = pullback_rows(train_states)
    test_pullback = pullback_rows(test_states)
    fit_pullback = train_pullback.loc[
        train_pullback.trading_day.astype(str).isin(MODEL_FIT_DAYS)
    ].copy()
    cal_pullback = train_pullback.loc[
        train_pullback.trading_day.astype(str).eq(MODEL_CALIBRATION_DAY)
    ].copy()

    columns = minute_state_columns(fit_pullback)
    fitted, model_support = fit_models(
        fit_pullback,
        cal_pullback,
        columns,
    )
    signal = signal_report(test_pullback, fitted)

    policy_rows = make_policy(test_states, test_selected, fitted)
    immediate_rows = baseline_policy(test_selected, scan, "immediate")
    fixed_rows = baseline_policy(test_selected, scan, "pullback_2")

    candidate_count = int(len(test_selected))
    policy = policy_metrics(policy_rows, candidate_count)
    immediate = policy_metrics(immediate_rows, candidate_count)
    fixed = policy_metrics(fixed_rows, candidate_count)
    policy_mean = policy["mean_pct"]
    fixed_mean = fixed["mean_pct"]
    gain = (
        policy_mean - fixed_mean
        if policy_mean is not None and fixed_mean is not None
        else None
    )

    checks = {
        "fit_pullback_rows": len(fit_pullback) >= MIN_FIT_PULLBACK_ROWS,
        "calibration_pullback_rows": (
            len(cal_pullback) >= MIN_CAL_PULLBACK_ROWS
        ),
        "test_pullback_rows": len(test_pullback) >= MIN_TEST_PULLBACK_ROWS,
        "entries": policy["entries"] >= MIN_ENTRIES,
        "entry_rate": MIN_ENTRY_RATE <= policy["entry_rate"] <= MAX_ENTRY_RATE,
        "resolution_or_cash_rate": (
            policy["resolution_or_cash_rate"] >= MIN_RESOLUTION_OR_CASH
        ),
        "mean_positive": policy_mean is not None and policy_mean > 0,
        "day_balanced_mean_positive": (
            policy["day_balanced_mean_pct"] is not None
            and policy["day_balanced_mean_pct"] > 0
        ),
        "positive_mean_days": (
            policy["positive_mean_days"] >= MIN_POSITIVE_MEAN_DAYS
        ),
        "severe_loss_rate": (
            policy["severe_loss_rate_le_minus2"] is not None
            and policy["severe_loss_rate_le_minus2"] <= MAX_SEVERE_RATE
        ),
        "gain_vs_fixed_pullback": (
            gain is not None and gain >= MIN_GAIN_VS_FIXED_PULLBACK
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "training_candidate_fraction": TRAIN_CANDIDATE_FRACTION,
        "test_candidate_fraction": TEST_CANDIDATE_FRACTION,
        "train_candidate_threshold": train_threshold,
        "test_candidate_threshold": test_threshold,
        "pullback_trigger_pct": PULLBACK_TRIGGER_PCT,
        "probability_thresholds": {
            "positive": POSITIVE_PROBABILITY_THRESHOLD,
            "severe": SEVERE_PROBABILITY_THRESHOLD,
        },
        "support": {
            "train_candidate_episodes": int(len(train_selected)),
            "test_candidate_episodes": candidate_count,
            "fit_pullback_rows": int(len(fit_pullback)),
            "calibration_pullback_rows": int(len(cal_pullback)),
            "test_pullback_rows": int(len(test_pullback)),
        },
        "model_support": model_support,
        "development_signal": signal,
        "development_policies": {
            "immediate": immediate,
            "fixed_2pct_pullback": fixed,
            "minute_state_controller": policy,
        },
        "mean_gain_vs_fixed_2pct_pullback_pct": gain,
        "checks": checks,
        "economic_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "Development-only. State uses completed minute aggregates only. "
            "May11-20 has been seen in prior requests and is not fresh validation."
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
    p = argparse.ArgumentParser()
    p.add_argument("--first-hot", type=Path, required=True)
    p.add_argument("--opportunity-scan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    return evaluate(a.first_hot, a.opportunity_scan, a.output)


if __name__ == "__main__":
    raise SystemExit(main())
