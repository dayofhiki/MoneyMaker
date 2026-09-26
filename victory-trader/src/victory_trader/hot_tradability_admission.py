"""Request 229: selected-HOT downstream-tradability admission.

Requests 226-228 show that downstream rescue is not the main bottleneck.
Request229 moves the admission decision to the HOT anchor itself.

Population:
* the same historical/fresh BUY-anchor episodes already selected by the frozen
  upstream system;
* no new dates and no expansion to previously rejected HOTs yet.

Target:
* build the Request227 multi-event entry value at WATCH minutes 1..5;
* define a persistent tradability label as the best mean across any two
  consecutive WATCH checkpoints;
* this rewards episodes that provide a usable entry window rather than a
  one-checkpoint hindsight spike.

Inputs:
* only causal market-wide + ticker context available at HOT time;
* minute OHLCV/transactions, attention, one-minute deltas, cross-sectional
  ranks, market breadth and session clock;
* no post-HOT WATCH state is an admission input.

Runtime:
HOT admission -> Request227 relative ENTER-vs-WAIT timing.
The weak Request227 absolute value head is not used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
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
from .market_calendar import regular_session_bounds
from .multi_event_entry_continuation import (
    ADVANTAGE_SEED,
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
    attach_transition_features,
    predict as predict_request208,
    train_model as train_request208,
)
from .recurrent_wait_entry_action_value import build_watch_states
from .relative_recurrent_entry_timing import attach_relative_advantage
from .second_path_attention_probe import (
    BASELINE_FEATURES,
    _annotate_scan,
)
from .wait_reobserve_entry_confirmation import _day_weights, _feature_frame
from .watch_option_admission import (
    admitted_relative_policy,
    timing_diagnostics,
)

REQUEST_ID = 229
REGRESSION_SEED = 20261231
CLASSIFIER_SEED = 20270101

HOT_CONTEXT_FEATURES = tuple(
    dict.fromkeys(
        [
            *BASELINE_FEATURES,
            "log_current_price",
            "minutes_since_open",
            "minutes_to_close",
            "attention_rank_pct",
            "return_rank_pct",
            "volume_rank_pct",
            "transactions_rank_pct",
            "market_symbol_count",
            "market_positive_share",
            "market_gt5_share",
            "market_gt10_share",
            "market_return_median_pct",
            "market_log_volume_median",
            "market_log_transactions_median",
        ]
    )
)

MIN_HOT_COVERAGE = 0.90
MIN_TRADABILITY_SPEARMAN = 0.08
MIN_TRADABILITY_AUC = 0.60
MIN_ADMISSION_RATE = 0.10
MAX_ADMISSION_RATE = 0.70
MIN_ADMITTED_POSITIVE_RATE = 0.55
MIN_TIMING_SPEARMAN = 0.05
MIN_EXECUTION_RATE = 0.10
MAX_EXECUTION_RATE = 0.80
MIN_TRADES = 30
MIN_IMPROVED_DAYS = 4


@dataclass(frozen=True)
class HotAdmissionModel:
    regressor: HistGradientBoostingRegressor
    classifier: HistGradientBoostingClassifier
    platt: LogisticRegression
    columns: tuple[str, ...]
    regression_offset: float
    winsor_low: float
    winsor_high: float


def attach_persistent_tradability(
    watch: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for keys, group in watch.groupby(EPISODE_KEYS, sort=False):
        values = {
            int(row["minutes_since_hot"]): float(value)
            for _, row in group.iterrows()
            if pd.notna(
                value := pd.to_numeric(
                    pd.Series(
                        [row.get("entry_multi_event_value_pct")]
                    ),
                    errors="coerce",
                ).iloc[0]
            )
            and np.isfinite(float(value))
        }

        pairs: list[float] = []
        for minute in range(1, MAX_WAIT_MINUTES):
            if minute in values and minute + 1 in values:
                pairs.append(
                    float(
                        (
                            values[minute]
                            + values[minute + 1]
                        )
                        / 2.0
                    )
                )

        option_max = (
            float(max(values.values()))
            if values
            else np.nan
        )
        persistent = (
            float(max(pairs))
            if pairs
            else np.nan
        )
        records.append(
            {
                "trading_day": str(keys[0]),
                "ticker": str(keys[1]).upper(),
                "hot_t": int(keys[2]),
                "episode_option_max_pct": option_max,
                "episode_persistent_tradability_pct": persistent,
                "persistent_pair_count": int(len(pairs)),
            }
        )
    return pd.DataFrame(records)


def annotate_hot_market_context(
    scan: pd.DataFrame,
) -> pd.DataFrame:
    frame = _annotate_scan(scan).copy()
    frame["trading_day"] = frame["trading_day"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["t"] = pd.to_numeric(
        frame["t"],
        errors="coerce",
    ).astype("Int64")

    group_keys = [
        frame["trading_day"],
        frame["t"],
    ]
    groups = frame.groupby(
        ["trading_day", "t"],
        sort=False,
    )

    attention = pd.to_numeric(
        frame["attention_score"],
        errors="coerce",
    )
    ret = pd.to_numeric(
        frame["return_from_previous_close_pct"],
        errors="coerce",
    )
    volume = pd.to_numeric(
        frame["log_minute_volume"],
        errors="coerce",
    )
    transactions = pd.to_numeric(
        frame["log_minute_transactions"],
        errors="coerce",
    )

    count = groups["ticker"].transform("count").astype(float)
    frame["market_symbol_count"] = count
    frame["attention_rank"] = attention.groupby(
        group_keys,
        sort=False,
    ).rank(method="first", ascending=False)
    frame["attention_rank_pct"] = (
        frame["attention_rank"] / count
    )
    frame["return_rank_pct"] = ret.groupby(
        group_keys,
        sort=False,
    ).rank(method="average", ascending=False) / count
    frame["volume_rank_pct"] = volume.groupby(
        group_keys,
        sort=False,
    ).rank(method="average", ascending=False) / count
    frame["transactions_rank_pct"] = transactions.groupby(
        group_keys,
        sort=False,
    ).rank(method="average", ascending=False) / count

    frame["_positive"] = ret.gt(0).astype(float)
    frame["_gt5"] = ret.ge(5).astype(float)
    frame["_gt10"] = ret.ge(10).astype(float)
    frame["market_positive_share"] = groups[
        "_positive"
    ].transform("mean")
    frame["market_gt5_share"] = groups[
        "_gt5"
    ].transform("mean")
    frame["market_gt10_share"] = groups[
        "_gt10"
    ].transform("mean")
    frame["market_return_median_pct"] = groups[
        "return_from_previous_close_pct"
    ].transform("median")
    frame["market_log_volume_median"] = groups[
        "log_minute_volume"
    ].transform("median")
    frame["market_log_transactions_median"] = groups[
        "log_minute_transactions"
    ].transform("median")

    price = pd.to_numeric(
        frame["c"],
        errors="coerce",
    )
    frame["log_current_price"] = np.log(
        price.where(price.gt(0))
    )

    since: dict[str, tuple[int, int]] = {}
    for day_text in frame["trading_day"].unique():
        bounds = regular_session_bounds(
            date.fromisoformat(str(day_text))
        )
        if bounds is None:
            continue
        since[str(day_text)] = (
            int(bounds[0].timestamp() * 1000),
            int(bounds[1].timestamp() * 1000),
        )

    open_values: list[float] = []
    close_values: list[float] = []
    for row in frame.loc[
        :, ["trading_day", "t"]
    ].itertuples(index=False):
        bounds = since.get(str(row.trading_day))
        if bounds is None or pd.isna(row.t):
            open_values.append(np.nan)
            close_values.append(np.nan)
            continue
        timestamp = int(row.t)
        open_values.append(
            (timestamp - bounds[0]) / 60_000.0
        )
        close_values.append(
            (bounds[1] - timestamp) / 60_000.0
        )
    frame["minutes_since_open"] = open_values
    frame["minutes_to_close"] = close_values

    return frame.drop(
        columns=["_positive", "_gt5", "_gt10"],
        errors="ignore",
    )


def hot_anchor_features(
    positions: pd.DataFrame,
    scan: pd.DataFrame,
    tradability: pd.DataFrame,
) -> pd.DataFrame:
    anchors = positions.loc[
        :, EPISODE_KEYS
    ].drop_duplicates().copy()
    anchors["trading_day"] = anchors[
        "trading_day"
    ].astype(str)
    anchors["ticker"] = anchors[
        "ticker"
    ].astype(str).str.upper()

    days = set(
        anchors["trading_day"].astype(str)
    )
    scan_days = scan.loc[
        scan["trading_day"].astype(str).isin(days)
    ].copy()
    context = annotate_hot_market_context(scan_days)

    hot = anchors.rename(
        columns={"hot_t": "t"}
    ).merge(
        context,
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
    )
    hot = hot.rename(
        columns={"t": "hot_t"}
    )
    hot = hot.merge(
        tradability,
        on=EPISODE_KEYS,
        how="left",
        validate="one_to_one",
    )
    return hot


def _usable_columns(
    frame: pd.DataFrame,
) -> tuple[str, ...]:
    return tuple(
        column
        for column in HOT_CONTEXT_FEATURES
        if column in frame.columns
        and pd.to_numeric(
            frame[column],
            errors="coerce",
        ).notna().any()
    )


def train_hot_admission(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> HotAdmissionModel:
    target = pd.to_numeric(
        fit["episode_persistent_tradability_pct"],
        errors="coerce",
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].to_numpy(dtype=float)
    if len(train) < 250:
        raise ValueError(
            f"request 229 insufficient fit HOT labels: {len(train)}"
        )

    columns = _usable_columns(train)
    if not columns:
        raise ValueError(
            "request 229 has no usable HOT features"
        )

    low, high = np.quantile(y, [0.005, 0.995])
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=REGRESSION_SEED,
    )
    regressor.fit(
        _feature_frame(train, columns),
        np.clip(y, low, high),
        sample_weight=_day_weights(train),
    )

    label = (y > 0).astype(int)
    if np.unique(label).size < 2:
        raise ValueError(
            "request 229 fit HOT target has one class"
        )
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=CLASSIFIER_SEED,
    )
    classifier.fit(
        _feature_frame(train, columns),
        label,
        sample_weight=_day_weights(train),
    )

    cal_target = pd.to_numeric(
        calibration[
            "episode_persistent_tradability_pct"
        ],
        errors="coerce",
    )
    cal = calibration.loc[
        cal_target.notna()
    ].copy()
    cal_y = cal_target.loc[
        cal_target.notna()
    ].to_numpy(dtype=float)
    if len(cal) < 120:
        raise ValueError(
            f"request 229 insufficient calibration HOT labels: "
            f"{len(cal)}"
        )

    x_cal = _feature_frame(cal, columns)
    raw_value = regressor.predict(x_cal)
    regression_offset = float(
        np.average(
            cal_y - raw_value,
            weights=_day_weights(cal),
        )
    )

    cal_label = (cal_y > 0).astype(int)
    if np.unique(cal_label).size < 2:
        raise ValueError(
            "request 229 calibration HOT target has one class"
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
        random_state=CLASSIFIER_SEED,
    )
    platt.fit(logits, cal_label)

    return HotAdmissionModel(
        regressor=regressor,
        classifier=classifier,
        platt=platt,
        columns=columns,
        regression_offset=regression_offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def score_hot_admission(
    frame: pd.DataFrame,
    fitted: HotAdmissionModel,
) -> pd.DataFrame:
    result = frame.copy()
    x = _feature_frame(
        result,
        fitted.columns,
    )
    result[
        "predicted_persistent_tradability_pct"
    ] = (
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
    result[
        "predicted_tradability_positive_probability"
    ] = fitted.platt.predict_proba(logits)[:, 1]
    result["hot_admitted"] = pd.to_numeric(
        result[
            "predicted_persistent_tradability_pct"
        ],
        errors="coerce",
    ).gt(0)
    return result


def _safe_spearman(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    a = pd.to_numeric(
        actual,
        errors="coerce",
    )
    s = pd.to_numeric(
        score,
        errors="coerce",
    )
    valid = a.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    value = a.loc[valid].corr(
        s.loc[valid],
        method="spearman",
    )
    return (
        None
        if pd.isna(value)
        else float(value)
    )


def _safe_auc(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    a = pd.to_numeric(
        actual,
        errors="coerce",
    )
    s = pd.to_numeric(
        score,
        errors="coerce",
    )
    valid = a.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    label = a.loc[valid].gt(0).astype(int)
    if label.nunique() < 2:
        return None
    return float(
        roc_auc_score(
            label,
            s.loc[valid].astype(float),
        )
    )


def admission_diagnostics(
    frame: pd.DataFrame,
) -> dict[str, object]:
    actual = pd.to_numeric(
        frame[
            "episode_persistent_tradability_pct"
        ],
        errors="coerce",
    )
    score = pd.to_numeric(
        frame[
            "predicted_persistent_tradability_pct"
        ],
        errors="coerce",
    )
    probability = pd.to_numeric(
        frame[
            "predicted_tradability_positive_probability"
        ],
        errors="coerce",
    )
    valid = (
        actual.notna()
        & score.notna()
        & probability.notna()
    )
    work = frame.loc[valid].copy()
    work["_actual"] = actual.loc[valid]
    work["_score"] = score.loc[valid]
    work["_probability"] = probability.loc[valid]
    work["_admit"] = (
        work["hot_admitted"]
        .fillna(False)
        .astype(bool)
    )
    admitted = work.loc[work["_admit"]]

    by_day: dict[str, object] = {}
    good_days = 0
    for day, part in work.groupby(
        work["trading_day"].astype(str),
        sort=True,
    ):
        selected = part.loc[
            part["_admit"]
        ]
        corr = _safe_spearman(
            part["_actual"],
            part["_score"],
        )
        selected_mean = (
            float(
                selected["_actual"].mean()
            )
            if len(selected)
            else None
        )
        positive_rate = (
            float(
                selected["_actual"]
                .gt(0)
                .mean()
            )
            if len(selected)
            else None
        )
        good_days += int(
            corr is not None
            and corr > 0
            and selected_mean is not None
            and selected_mean > 0
        )
        by_day[str(day)] = {
            "rows": int(len(part)),
            "spearman": corr,
            "admission_rate": float(
                part["_admit"].mean()
            ),
            "admitted_actual_mean_pct": (
                selected_mean
            ),
            "admitted_actual_positive_rate": (
                positive_rate
            ),
        }

    return {
        "rows": int(len(frame)),
        "evaluable_rows": int(len(work)),
        "coverage": (
            float(len(work) / len(frame))
            if len(frame)
            else None
        ),
        "spearman": _safe_spearman(
            work["_actual"],
            work["_score"],
        ),
        "positive_auc": _safe_auc(
            work["_actual"],
            work["_probability"],
        ),
        "population_actual_mean_pct": (
            float(work["_actual"].mean())
            if len(work)
            else None
        ),
        "population_actual_positive_rate": (
            float(
                work["_actual"]
                .gt(0)
                .mean()
            )
            if len(work)
            else None
        ),
        "admitted_rows": int(
            work["_admit"].sum()
        ),
        "admission_rate": (
            float(work["_admit"].mean())
            if len(work)
            else None
        ),
        "admitted_actual_mean_pct": (
            float(
                admitted["_actual"].mean()
            )
            if len(admitted)
            else None
        ),
        "admitted_actual_positive_rate": (
            float(
                admitted["_actual"]
                .gt(0)
                .mean()
            )
            if len(admitted)
            else None
        ),
        "good_days": int(good_days),
        "by_day": by_day,
    }


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    history_scan_path: Path,
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
    history_scan = pd.read_parquet(
        history_scan_path
    )

    position_paths = sorted(
        fresh_dir.glob("*-positions.parquet")
    )
    scan_paths = sorted(
        fresh_dir.glob("*-scan.parquet")
    )
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError(
            "request 229 requires complete Request178 "
            "position and scan shards"
        )
    fresh_positions = pd.concat(
        [
            pd.read_parquet(path)
            for path in position_paths
        ],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [
            pd.read_parquet(path)
            for path in scan_paths
        ],
        ignore_index=True,
    )

    fit_watch = attach_multi_event_entry_labels(
        attach_transition_features(
            build_watch_states(fit_positions)
        ),
        fit_positions,
    )
    cal_watch = attach_multi_event_entry_labels(
        attach_transition_features(
            build_watch_states(
                calibration_positions
            )
        ),
        calibration_positions,
    )
    fresh_watch = attach_multi_event_entry_labels(
        attach_transition_features(
            build_watch_states(
                fresh_positions
            )
        ),
        fresh_positions,
    )

    fit_target = attach_persistent_tradability(
        fit_watch
    )
    cal_target = attach_persistent_tradability(
        cal_watch
    )
    fresh_target = attach_persistent_tradability(
        fresh_watch
    )

    fit_hot = hot_anchor_features(
        fit_positions,
        history_scan,
        fit_target,
    )
    cal_hot = hot_anchor_features(
        calibration_positions,
        history_scan,
        cal_target,
    )
    fresh_hot = hot_anchor_features(
        fresh_positions,
        fresh_scan,
        fresh_target,
    )

    admission_model = train_hot_admission(
        fit_hot,
        cal_hot,
    )
    timing_head = train_head(
        fit_watch,
        cal_watch,
        target_column=(
            "enter_vs_wait_multi_event_advantage_pct"
        ),
        seed=ADVANTAGE_SEED,
    )

    fresh_hot = score_hot_admission(
        fresh_hot,
        admission_model,
    )
    fresh_watch[
        "predicted_enter_vs_wait_advantage_pct"
    ] = predict_head(
        fresh_watch,
        timing_head,
    )

    admitted_keys = {
        (
            str(row.trading_day),
            str(row.ticker).upper(),
            int(row.hot_t),
        )
        for row in fresh_hot.loc[
            fresh_hot[
                "hot_admitted"
            ].fillna(False).astype(bool)
        ].itertuples(index=False)
    }

    admission_diag = admission_diagnostics(
        fresh_hot
    )
    timing_diag = timing_diagnostics(
        fresh_watch,
        admitted_keys,
    )
    candidate_trades, candidate_counts = (
        admitted_relative_policy(
            fresh_watch,
            admitted_keys,
        )
    )

    request208_fit = attach_relative_advantage(
        attach_transition_features(
            build_watch_states(
                fit_positions
            )
        )
    )
    request208_cal = attach_relative_advantage(
        attach_transition_features(
            build_watch_states(
                calibration_positions
            )
        )
    )
    request208_model = train_request208(
        request208_fit,
        request208_cal,
    )
    request208_fresh = attach_relative_advantage(
        fresh_watch.copy()
    )
    request208_fresh[
        "predicted_request208_advantage_pct"
    ] = predict_request208(
        request208_fresh,
        request208_model,
    )
    baseline_trades = request208_policy(
        request208_fresh
    )

    candidate_portfolio = portfolio_rows(
        fresh_positions,
        candidate_trades,
        "request229_hot_tradability_admission",
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
    hot_context_rows = int(
        fresh_hot.loc[
            pd.to_numeric(
                fresh_hot[
                    "attention_score"
                ],
                errors="coerce",
            ).notna(),
            EPISODE_KEYS,
        ].drop_duplicates().shape[0]
    )
    hot_coverage = (
        float(
            hot_context_rows
            / total_episodes
        )
        if total_episodes
        else None
    )

    spearman = admission_diag.get(
        "spearman"
    )
    auc = admission_diag.get(
        "positive_auc"
    )
    admission_rate = admission_diag.get(
        "admission_rate"
    )
    admitted_positive = admission_diag.get(
        "admitted_actual_positive_rate"
    )
    timing_s = timing_diag.get(
        "spearman"
    )
    execution_rate = candidate_metrics.get(
        "execution_rate"
    )
    trade_mean = candidate_metrics.get(
        "trade_mean_pct"
    )
    candidate_day = candidate_metrics.get(
        "portfolio_day_balanced_mean_pct"
    )
    baseline_day = baseline_metrics.get(
        "portfolio_day_balanced_mean_pct"
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
        hot_coverage is not None
        and hot_coverage >= MIN_HOT_COVERAGE
        and spearman is not None
        and float(spearman)
        >= MIN_TRADABILITY_SPEARMAN
        and auc is not None
        and float(auc)
        >= MIN_TRADABILITY_AUC
        and admission_rate is not None
        and MIN_ADMISSION_RATE
        <= float(admission_rate)
        <= MAX_ADMISSION_RATE
        and admitted_positive is not None
        and float(admitted_positive)
        >= MIN_ADMITTED_POSITIVE_RATE
        and timing_s is not None
        and float(timing_s)
        >= MIN_TIMING_SPEARMAN
        and execution_rate is not None
        and MIN_EXECUTION_RATE
        <= float(execution_rate)
        <= MAX_EXECUTION_RATE
        and int(
            candidate_metrics.get(
                "executed", 0
            )
        )
        >= MIN_TRADES
        and trade_mean is not None
        and float(trade_mean) > 0
        and candidate_day is not None
        and baseline_day is not None
        and float(candidate_day) > 0
        and float(candidate_day)
        > float(baseline_day)
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
        and int(
            difference.get(
                "improved_days", 0
            )
        )
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
        "population": (
            "frozen upstream BUY-anchor episodes only; "
            "does not yet reopen rejected HOT promotions"
        ),
        "architecture": (
            "HOT market-context tradability admission -> "
            "Request227 relative ENTER-vs-WAIT timing"
        ),
        "tradability_target": {
            "definition": (
                "best mean Request227 multi-event value across "
                "two consecutive WATCH checkpoints"
            ),
            "purpose": (
                "reward persistent usable entry windows rather than "
                "single-checkpoint hindsight spikes"
            ),
            "future_information_used_as_input": False,
        },
        "hot_features": list(
            admission_model.columns
        ),
        "hot_context_coverage": hot_coverage,
        "fresh_admission": admission_diag,
        "fresh_relative_timing_within_admitted": (
            timing_diag
        ),
        "candidate_path_counts": candidate_counts,
        "fresh_policy": {
            "request208": baseline_metrics,
            "request229": candidate_metrics,
            "request229_minus_request208": difference,
        },
        "frozen_gate": {
            "min_hot_context_coverage": MIN_HOT_COVERAGE,
            "min_tradability_spearman": (
                MIN_TRADABILITY_SPEARMAN
            ),
            "min_tradability_auc": (
                MIN_TRADABILITY_AUC
            ),
            "min_admission_rate": MIN_ADMISSION_RATE,
            "max_admission_rate": MAX_ADMISSION_RATE,
            "min_admitted_positive_rate": (
                MIN_ADMITTED_POSITIVE_RATE
            ),
            "min_timing_spearman": (
                MIN_TIMING_SPEARMAN
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
            "freeze selected-HOT tradability admission, then "
            "rebuild the full HOT promotion population and test the "
            "same target before old BUY gating."
        ),
        "next_boundary_if_fail": (
            "minute-level HOT context is still insufficient; test "
            "whether the missing information is second-scale HOT "
            "microstructure or whether upstream runner selection and "
            "downstream tradability are fundamentally different tasks."
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
        "--history-scan",
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
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
