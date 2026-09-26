"""Request 228: WATCH option admission + relative multi-event timing.

Request227 exposed an asymmetry:
* the absolute multi-event value head transferred weakly and collapsed the
  runtime policy to almost no trades;
* the ENTER-vs-WAIT relative timing head transferred materially better.

Request228 separates two jobs.

1) At the first causal WATCH checkpoint, estimate whether the episode contains
   any economically positive multi-event entry opportunity somewhere in the
   fixed minute-1..5 WATCH window. This is an admission / monitoring-value
   target, not an executable entry oracle.

2) Only for admitted episodes, reuse a relative timing model trained on the
   Request227 ENTER-vs-WAIT multi-event advantage. ENTER when the predicted
   advantage is >= 0; otherwise WAIT. At minute 5, negative advantage means
   ABSTAIN because the target compares the final entry with cash=0.

All runtime features are causal. Future outcomes are labels only. Economic
evaluation still uses the fixed three-minute BASE outcome after the selected
entry, with cash zero for abstained episodes, so the comparison isolates
admission + timing rather than downstream position management.
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .multi_event_entry_continuation import (
    ADVANTAGE_SEED,
    BOOTSTRAP_SAMPLES,
    FEATURES as TIMING_FEATURES,
    MAX_WAIT_MINUTES,
    attach_multi_event_entry_labels,
    matched_portfolio_difference,
    portfolio_metrics,
    portfolio_rows,
    predict_head,
    request208_policy,
    train_head,
)
from .pullback_turn_transition_entry import (
    MODEL_FEATURES,
    attach_transition_features,
    predict as predict_request208,
    train_model as train_request208,
)
from .recurrent_wait_entry_action_value import build_watch_states
from .relative_recurrent_entry_timing import attach_relative_advantage
from .wait_reobserve_entry_confirmation import _day_weights, _feature_frame

REQUEST_ID = 228
OPTION_REGRESSION_SEED = 20261229
OPTION_CLASSIFIER_SEED = 20261230
BOOTSTRAP_SEED = 20261231

ADMISSION_FEATURES = MODEL_FEATURES

MIN_STATE_COVERAGE = 0.80
MIN_OPTION_SPEARMAN = 0.08
MIN_OPTION_AUC = 0.60
MIN_ADMISSION_RATE = 0.10
MAX_ADMISSION_RATE = 0.70
MIN_ADMITTED_ACTUAL_POSITIVE_RATE = 0.55
MIN_TIMING_ADVANTAGE_SPEARMAN = 0.05
MIN_EXECUTION_RATE = 0.10
MAX_EXECUTION_RATE = 0.80
MIN_TRADES = 30
MIN_IMPROVED_DAYS = 4


@dataclass(frozen=True)
class OptionAdmissionModel:
    regressor: HistGradientBoostingRegressor
    classifier: HistGradientBoostingClassifier
    platt: LogisticRegression
    columns: tuple[str, ...]
    regression_offset: float
    winsor_low: float
    winsor_high: float


def attach_episode_watch_option(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["episode_watch_option_value_pct"] = np.nan

    for _, group in result.groupby(EPISODE_KEYS, sort=False):
        value = pd.to_numeric(
            group["entry_multi_event_value_pct"],
            errors="coerce",
        ).dropna()
        if value.empty:
            continue
        option_value = float(value.max())
        result.loc[
            group.index,
            "episode_watch_option_value_pct",
        ] = option_value

    return result


def minute1(frame: pd.DataFrame) -> pd.DataFrame:
    minute = pd.to_numeric(
        frame["minutes_since_hot"], errors="coerce"
    )
    return frame.loc[minute.eq(1)].copy()


def _usable_admission_columns(
    frame: pd.DataFrame,
) -> tuple[str, ...]:
    return tuple(
        column
        for column in ADMISSION_FEATURES
        if column in frame.columns
        and pd.to_numeric(
            frame[column], errors="coerce"
        ).notna().any()
    )


def train_option_admission(
    fit_first: pd.DataFrame,
    calibration_first: pd.DataFrame,
) -> OptionAdmissionModel:
    target = pd.to_numeric(
        fit_first["episode_watch_option_value_pct"],
        errors="coerce",
    )
    valid = target.notna()
    train = fit_first.loc[valid].copy()
    y = target.loc[valid].to_numpy(dtype=float)
    if len(train) < 300:
        raise ValueError(
            f"request 228 insufficient fit admission rows: {len(train)}"
        )

    columns = _usable_admission_columns(train)
    if not columns:
        raise ValueError(
            "request 228 has no usable admission features"
        )

    low, high = np.quantile(y, [0.005, 0.995])
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=OPTION_REGRESSION_SEED,
    )
    regressor.fit(
        _feature_frame(train, columns),
        np.clip(y, low, high),
        sample_weight=_day_weights(train),
    )

    label = (y > 0).astype(int)
    if np.unique(label).size < 2:
        raise ValueError(
            "request 228 admission target has one class in fit"
        )
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=OPTION_CLASSIFIER_SEED,
    )
    classifier.fit(
        _feature_frame(train, columns),
        label,
        sample_weight=_day_weights(train),
    )

    cal_target = pd.to_numeric(
        calibration_first["episode_watch_option_value_pct"],
        errors="coerce",
    )
    cal = calibration_first.loc[cal_target.notna()].copy()
    cal_y = cal_target.loc[cal_target.notna()].to_numpy(
        dtype=float
    )
    if len(cal) < 150:
        raise ValueError(
            f"request 228 insufficient calibration admission rows: "
            f"{len(cal)}"
        )

    x_cal = _feature_frame(cal, columns)
    reg_raw = regressor.predict(x_cal)
    regression_offset = float(
        np.average(
            cal_y - reg_raw,
            weights=_day_weights(cal),
        )
    )

    cal_label = (cal_y > 0).astype(int)
    if np.unique(cal_label).size < 2:
        raise ValueError(
            "request 228 admission target has one class in calibration"
        )
    raw_p = np.clip(
        classifier.predict_proba(x_cal)[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(
        raw_p / (1.0 - raw_p)
    ).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=OPTION_CLASSIFIER_SEED,
    )
    platt.fit(logits, cal_label)

    return OptionAdmissionModel(
        regressor=regressor,
        classifier=classifier,
        platt=platt,
        columns=columns,
        regression_offset=regression_offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def score_option_admission(
    frame: pd.DataFrame,
    fitted: OptionAdmissionModel,
) -> pd.DataFrame:
    result = frame.copy()
    x = _feature_frame(result, fitted.columns)

    result["predicted_watch_option_value_pct"] = (
        fitted.regressor.predict(x)
        + fitted.regression_offset
    )

    raw_p = np.clip(
        fitted.classifier.predict_proba(x)[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(
        raw_p / (1.0 - raw_p)
    ).reshape(-1, 1)
    result["predicted_watch_option_positive_probability"] = (
        fitted.platt.predict_proba(logits)[:, 1]
    )
    result["watch_admitted"] = pd.to_numeric(
        result["predicted_watch_option_value_pct"],
        errors="coerce",
    ).gt(0)
    return result


def _safe_spearman(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = a.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    value = a.loc[valid].corr(
        s.loc[valid],
        method="spearman",
    )
    return None if pd.isna(value) else float(value)


def _safe_auc(
    actual: pd.Series,
    probability: pd.Series,
) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(probability, errors="coerce")
    valid = a.notna() & p.notna()
    if int(valid.sum()) < 20:
        return None
    label = a.loc[valid].gt(0).astype(int)
    if label.nunique() < 2:
        return None
    return float(
        roc_auc_score(label, p.loc[valid].astype(float))
    )


def admission_diagnostics(
    first: pd.DataFrame,
) -> dict[str, object]:
    actual = pd.to_numeric(
        first["episode_watch_option_value_pct"],
        errors="coerce",
    )
    ev = pd.to_numeric(
        first["predicted_watch_option_value_pct"],
        errors="coerce",
    )
    probability = pd.to_numeric(
        first["predicted_watch_option_positive_probability"],
        errors="coerce",
    )
    admitted_mask = first["watch_admitted"].fillna(False).astype(bool)
    valid = actual.notna() & ev.notna() & probability.notna()
    work = first.loc[valid].copy()
    work["_actual"] = actual.loc[valid]
    work["_ev"] = ev.loc[valid]
    work["_p"] = probability.loc[valid]
    work["_admit"] = admitted_mask.loc[valid]
    admitted = work.loc[work["_admit"]]

    by_day: dict[str, object] = {}
    good_days = 0
    for day, part in work.groupby(
        work["trading_day"].astype(str),
        sort=True,
    ):
        selected = part.loc[part["_admit"]]
        selected_mean = (
            float(selected["_actual"].mean())
            if len(selected)
            else None
        )
        selected_positive = (
            float(selected["_actual"].gt(0).mean())
            if len(selected)
            else None
        )
        corr = _safe_spearman(
            part["_actual"],
            part["_ev"],
        )
        good = bool(
            corr is not None
            and corr > 0
            and selected_mean is not None
            and selected_mean > 0
        )
        good_days += int(good)
        by_day[str(day)] = {
            "rows": int(len(part)),
            "spearman": corr,
            "admission_rate": float(
                part["_admit"].mean()
            ),
            "admitted_actual_mean_pct": selected_mean,
            "admitted_actual_positive_rate": selected_positive,
        }

    return {
        "rows": int(len(first)),
        "evaluable_rows": int(len(work)),
        "coverage": (
            float(len(work) / len(first))
            if len(first)
            else None
        ),
        "spearman": _safe_spearman(
            work["_actual"],
            work["_ev"],
        ),
        "positive_auc": _safe_auc(
            work["_actual"],
            work["_p"],
        ),
        "population_actual_mean_pct": (
            float(work["_actual"].mean())
            if len(work)
            else None
        ),
        "population_actual_positive_rate": (
            float(work["_actual"].gt(0).mean())
            if len(work)
            else None
        ),
        "admitted_rows": int(work["_admit"].sum()),
        "admission_rate": (
            float(work["_admit"].mean())
            if len(work)
            else None
        ),
        "admitted_actual_mean_pct": (
            float(admitted["_actual"].mean())
            if len(admitted)
            else None
        ),
        "admitted_actual_positive_rate": (
            float(admitted["_actual"].gt(0).mean())
            if len(admitted)
            else None
        ),
        "good_days": int(good_days),
        "by_day": by_day,
    }


def timing_diagnostics(
    frame: pd.DataFrame,
    admitted_keys: set[tuple[str, str, int]],
) -> dict[str, object]:
    if not admitted_keys:
        return {
            "rows": 0,
            "spearman": None,
            "positive_spearman_days": 0,
        }
    mask = [
        (
            str(row.trading_day),
            str(row.ticker).upper(),
            int(row.hot_t),
        )
        in admitted_keys
        for row in frame.itertuples(index=False)
    ]
    work = frame.loc[mask].copy()
    actual = pd.to_numeric(
        work["enter_vs_wait_multi_event_advantage_pct"],
        errors="coerce",
    )
    score = pd.to_numeric(
        work["predicted_enter_vs_wait_advantage_pct"],
        errors="coerce",
    )
    valid = actual.notna() & score.notna()
    work = work.loc[valid].copy()
    work["_actual"] = actual.loc[valid]
    work["_score"] = score.loc[valid]

    by_day: dict[str, float | None] = {}
    positive_days = 0
    for day, part in work.groupby(
        work["trading_day"].astype(str),
        sort=True,
    ):
        corr = _safe_spearman(
            part["_actual"],
            part["_score"],
        )
        positive_days += int(
            corr is not None and corr > 0
        )
        by_day[str(day)] = corr

    return {
        "rows": int(len(work)),
        "spearman": _safe_spearman(
            work["_actual"],
            work["_score"],
        ),
        "positive_spearman_days": int(positive_days),
        "by_day": by_day,
    }


def admitted_relative_policy(
    scored: pd.DataFrame,
    admitted_keys: set[tuple[str, str, int]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    records: list[dict[str, object]] = []
    counts = {
        "episodes": int(
            scored.loc[:, EPISODE_KEYS]
            .drop_duplicates()
            .shape[0]
        ),
        "admitted_episodes": len(admitted_keys),
        "entered": 0,
        "abstained_after_admission": 0,
        "rejected_at_admission": 0,
        "coverage_miss": 0,
        "wait_actions": 0,
    }

    all_keys = {
        (
            str(row.trading_day),
            str(row.ticker).upper(),
            int(row.hot_t),
        )
        for row in scored.loc[
            :, EPISODE_KEYS
        ].drop_duplicates().itertuples(index=False)
    }
    counts["rejected_at_admission"] = int(
        len(all_keys - admitted_keys)
    )

    for keys, group in scored.groupby(
        EPISODE_KEYS,
        sort=False,
    ):
        key = (
            str(keys[0]),
            str(keys[1]).upper(),
            int(keys[2]),
        )
        if key not in admitted_keys:
            continue

        by_minute = {
            int(row["minutes_since_hot"]): row
            for _, row in group.sort_values(
                "minutes_since_hot",
                kind="stable",
            ).iterrows()
        }
        chosen = None
        failed = False

        for minute in range(
            1,
            MAX_WAIT_MINUTES + 1,
        ):
            row = by_minute.get(minute)
            if row is None:
                counts["coverage_miss"] += 1
                failed = True
                break

            score = pd.to_numeric(
                pd.Series(
                    [
                        row.get(
                            "predicted_enter_vs_wait_advantage_pct"
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]
            if pd.isna(score):
                counts["coverage_miss"] += 1
                failed = True
                break

            if float(score) >= 0:
                chosen = row
                break

            counts["wait_actions"] += 1

        if failed:
            continue
        if chosen is None:
            counts["abstained_after_admission"] += 1
            continue

        realized = pd.to_numeric(
            pd.Series(
                [chosen.get("enter_3m_base_pct")]
            ),
            errors="coerce",
        ).iloc[0]
        if pd.isna(realized):
            counts["coverage_miss"] += 1
            continue

        counts["entered"] += 1
        records.append(
            {
                "trading_day": str(keys[0]),
                "ticker": str(keys[1]).upper(),
                "hot_t": int(keys[2]),
                "entry_minute_after_hot": int(
                    chosen["minutes_since_hot"]
                ),
                "realized_base_return_pct": float(realized),
                "predicted_enter_vs_wait_advantage_pct": float(
                    chosen[
                        "predicted_enter_vs_wait_advantage_pct"
                    ]
                ),
            }
        )

    return pd.DataFrame(records), counts


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(
        fit_positions_path
    )
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )

    paths = sorted(
        fresh_dir.glob("*-positions.parquet")
    )
    if len(paths) != len(FRESH_DAYS):
        raise ValueError(
            "request 228 requires complete Request178 position shards"
        )
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )
    found_days = sorted(
        fresh_positions["trading_day"]
        .astype(str)
        .unique()
    )
    if found_days != list(FRESH_DAYS):
        raise ValueError(
            f"request 228 expected {FRESH_DAYS}, "
            f"found {found_days}"
        )

    fit = attach_episode_watch_option(
        attach_multi_event_entry_labels(
            attach_transition_features(
                build_watch_states(fit_positions)
            ),
            fit_positions,
        )
    )
    calibration = attach_episode_watch_option(
        attach_multi_event_entry_labels(
            attach_transition_features(
                build_watch_states(calibration_positions)
            ),
            calibration_positions,
        )
    )
    fresh = attach_episode_watch_option(
        attach_multi_event_entry_labels(
            attach_transition_features(
                build_watch_states(fresh_positions)
            ),
            fresh_positions,
        )
    )

    admission_model = train_option_admission(
        minute1(fit),
        minute1(calibration),
    )
    timing_head = train_head(
        fit,
        calibration,
        target_column=(
            "enter_vs_wait_multi_event_advantage_pct"
        ),
        seed=ADVANTAGE_SEED,
    )

    fresh[
        "predicted_enter_vs_wait_advantage_pct"
    ] = predict_head(
        fresh,
        timing_head,
    )
    fresh_first = score_option_admission(
        minute1(fresh),
        admission_model,
    )

    admitted_keys = {
        (
            str(row.trading_day),
            str(row.ticker).upper(),
            int(row.hot_t),
        )
        for row in fresh_first.loc[
            fresh_first["watch_admitted"]
            .fillna(False)
            .astype(bool)
        ].itertuples(index=False)
    }

    admission_diag = admission_diagnostics(
        fresh_first
    )
    timing_diag = timing_diagnostics(
        fresh,
        admitted_keys,
    )

    candidate_trades, candidate_counts = (
        admitted_relative_policy(
            fresh,
            admitted_keys,
        )
    )

    request208_fit = attach_relative_advantage(
        attach_transition_features(
            build_watch_states(fit_positions)
        )
    )
    request208_cal = attach_relative_advantage(
        attach_transition_features(
            build_watch_states(calibration_positions)
        )
    )
    request208_model = train_request208(
        request208_fit,
        request208_cal,
    )
    request208_fresh = attach_relative_advantage(
        fresh.copy()
    )
    request208_fresh[
        "predicted_request208_advantage_pct"
    ] = predict_request208(
        request208_fresh,
        request208_model,
    )
    # request208_policy expects this exact score name.
    request208_fresh[
        "predicted_request208_advantage_pct"
    ] = pd.to_numeric(
        request208_fresh[
            "predicted_request208_advantage_pct"
        ],
        errors="coerce",
    )
    request208_fresh[
        "predicted_request208_advantage_pct"
    ] = request208_fresh[
        "predicted_request208_advantage_pct"
    ]
    baseline_ready = request208_fresh.rename(
        columns={
            "predicted_request208_advantage_pct": (
                "predicted_request208_advantage_pct"
            )
        }
    )

    # Reproduce Request208 timing explicitly because the Request227 helper
    # expects the same score name used in that module.
    baseline_records: list[dict[str, object]] = []
    for keys, group in baseline_ready.groupby(
        EPISODE_KEYS,
        sort=False,
    ):
        by_minute = {
            int(row["minutes_since_hot"]): row
            for _, row in group.sort_values(
                "minutes_since_hot",
                kind="stable",
            ).iterrows()
        }
        chosen = None
        for minute in range(
            1,
            MAX_WAIT_MINUTES + 1,
        ):
            row = by_minute.get(minute)
            if row is None:
                break
            if minute == MAX_WAIT_MINUTES:
                chosen = row
                break
            score = pd.to_numeric(
                pd.Series(
                    [
                        row.get(
                            "predicted_request208_advantage_pct"
                        )
                    ]
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
            pd.Series(
                [chosen.get("enter_3m_base_pct")]
            ),
            errors="coerce",
        ).iloc[0]
        if pd.isna(realized):
            continue
        baseline_records.append(
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
    baseline_trades = pd.DataFrame(
        baseline_records
    )

    candidate_portfolio = portfolio_rows(
        fresh_positions,
        candidate_trades,
        "request228_watch_option_admission",
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
    state_coverage = (
        float(len(fresh_first) / total_episodes)
        if total_episodes
        else None
    )

    option_s = admission_diag.get("spearman")
    option_auc = admission_diag.get(
        "positive_auc"
    )
    admission_rate = admission_diag.get(
        "admission_rate"
    )
    admitted_positive = admission_diag.get(
        "admitted_actual_positive_rate"
    )
    timing_s = timing_diag.get("spearman")
    execution_rate = candidate_metrics.get(
        "execution_rate"
    )
    candidate_day = candidate_metrics.get(
        "portfolio_day_balanced_mean_pct"
    )
    baseline_day = baseline_metrics.get(
        "portfolio_day_balanced_mean_pct"
    )
    candidate_trade_mean = candidate_metrics.get(
        "trade_mean_pct"
    )
    diff_day = difference.get(
        "day_balanced_difference_pct"
    )
    diff_low = difference.get(
        "bootstrap", {}
    ).get("ci_low_pct")
    candidate_severe = candidate_metrics.get(
        "trade_severe_loss_rate"
    )
    baseline_severe = baseline_metrics.get(
        "trade_severe_loss_rate"
    )

    gate = bool(
        state_coverage is not None
        and state_coverage >= MIN_STATE_COVERAGE
        and option_s is not None
        and float(option_s) >= MIN_OPTION_SPEARMAN
        and option_auc is not None
        and float(option_auc) >= MIN_OPTION_AUC
        and admission_rate is not None
        and MIN_ADMISSION_RATE
        <= float(admission_rate)
        <= MAX_ADMISSION_RATE
        and admitted_positive is not None
        and float(admitted_positive)
        >= MIN_ADMITTED_ACTUAL_POSITIVE_RATE
        and timing_s is not None
        and float(timing_s)
        >= MIN_TIMING_ADVANTAGE_SPEARMAN
        and execution_rate is not None
        and MIN_EXECUTION_RATE
        <= float(execution_rate)
        <= MAX_EXECUTION_RATE
        and int(candidate_metrics.get("executed", 0))
        >= MIN_TRADES
        and candidate_trade_mean is not None
        and float(candidate_trade_mean) > 0
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
        and float(candidate_severe)
        <= float(baseline_severe)
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "architecture": (
            "minute1 WATCH option admission -> "
            "Request227 relative ENTER-vs-WAIT timing"
        ),
        "admission_target": {
            "definition": (
                "maximum fixed multi-event continuation value across "
                "WATCH minutes 1..5"
            ),
            "role": (
                "monitoring/admission option value only; not used as "
                "the executable entry price oracle"
            ),
            "future_information_used_as_input": False,
        },
        "runtime_policy": {
            "admit_episode_if": (
                "calibrated predicted WATCH option value > 0"
            ),
            "within_admitted_episode": (
                "ENTER at first minute whose predicted Request227 "
                "ENTER-vs-WAIT advantage >= 0; otherwise keep WAITing"
            ),
            "minute5_negative_advantage": "ABSTAIN",
            "fresh_threshold_tuning": False,
        },
        "state_coverage": state_coverage,
        "fresh_admission": admission_diag,
        "fresh_relative_timing_within_admitted": (
            timing_diag
        ),
        "candidate_path_counts": candidate_counts,
        "fresh_policy": {
            "request208": baseline_metrics,
            "request228": candidate_metrics,
            "request228_minus_request208": difference,
        },
        "frozen_gate": {
            "min_state_coverage": MIN_STATE_COVERAGE,
            "min_option_spearman": MIN_OPTION_SPEARMAN,
            "min_option_auc": MIN_OPTION_AUC,
            "min_admission_rate": MIN_ADMISSION_RATE,
            "max_admission_rate": MAX_ADMISSION_RATE,
            "min_admitted_actual_positive_rate": (
                MIN_ADMITTED_ACTUAL_POSITIVE_RATE
            ),
            "min_timing_advantage_spearman": (
                MIN_TIMING_ADVANTAGE_SPEARMAN
            ),
            "min_execution_rate": MIN_EXECUTION_RATE,
            "max_execution_rate": MAX_EXECUTION_RATE,
            "min_trades": MIN_TRADES,
            "trade_mean_must_be_positive": True,
            "portfolio_day_balanced_mean_must_be_positive": True,
            "must_beat_request208": True,
            "matched_bootstrap_ci_low_must_be_positive": True,
            "min_improved_days": MIN_IMPROVED_DAYS,
            "severe_loss_no_worse_than_request208": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze WATCH admission + relative timing, integrate with "
            "the least-damaging causal POSITION controller, and replay "
            "complete trajectories before untouched validation."
        ),
        "next_boundary_if_fail": (
            "the first WATCH checkpoint cannot reliably identify "
            "economically tradable episodes; move admission upstream "
            "to HOT promotion using market-wide/attention context and "
            "train directly on downstream tradability."
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    rows_output_path.parent.mkdir(
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
    pd.concat(
        [
            baseline_portfolio,
            candidate_portfolio,
        ],
        ignore_index=True,
    ).to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
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
        "--fit-positions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration-positions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fresh-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        required=True,
    )
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
