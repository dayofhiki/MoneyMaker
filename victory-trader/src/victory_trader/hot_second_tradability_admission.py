"""Request 230: HOT tradability admission with causal one-second microstructure.

Request229 showed that minute-level HOT context can rank downstream tradability,
but admitted episodes still have negative average economic value and the
Request227 timing signal collapses within that admitted set.

Request230 changes only the HOT admission information set.

Baseline:
    Request229 minute-level HOT + market-wide context.

Candidate:
    The same baseline context plus two causal one-second feature families,
    computed strictly from completed seconds before HOT:
    * recent 10/30-second path, activity, burst, drawdown/runup;
    * richer 5/15-second momentum, VWAP, imbalance, efficiency,
      high/low age, volatility and mean-trade-size proxies.

The downstream tradability target, Request227 relative ENTER-vs-WAIT timing,
Request208 comparator, dates, costs, and semantic zero admission boundary are
frozen. No fresh threshold tuning is allowed.
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

from .config import load_settings
from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .hot_tradability_admission import (
    CLASSIFIER_SEED,
    HOT_CONTEXT_FEATURES,
    REGRESSION_SEED,
    admission_diagnostics,
    attach_persistent_tradability,
    hot_anchor_features,
)
from .massive_client import MassiveClient
from .multi_event_entry_continuation import (
    ADVANTAGE_SEED,
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
from .rich_second_position_value import (
    RICH_SECOND_FEATURES,
    rich_second_features,
)
from .second_path_attention_probe import (
    SECOND_CACHE_DIR,
    SECOND_FEATURES,
    _second_frame,
    second_path_features,
)
from .wait_reobserve_entry_confirmation import _day_weights, _feature_frame
from .watch_option_admission import (
    admitted_relative_policy,
    timing_diagnostics,
)

REQUEST_ID = 230
SECOND_REGRESSION_SEED = REGRESSION_SEED + 230
SECOND_CLASSIFIER_SEED = CLASSIFIER_SEED + 230

HOT_SECOND_FEATURES = tuple(
    dict.fromkeys([*SECOND_FEATURES, *RICH_SECOND_FEATURES])
)
HOT_AUGMENTED_FEATURES = tuple(
    dict.fromkeys([*HOT_CONTEXT_FEATURES, *HOT_SECOND_FEATURES])
)

MIN_SECOND_COVERAGE = 0.85
MIN_SPEARMAN_GAIN_VS_229 = 0.03
MIN_AUC_GAIN_VS_229 = 0.01
MIN_CANDIDATE_SPEARMAN = 0.15
MIN_CANDIDATE_AUC = 0.62
MIN_ADMISSION_RATE = 0.10
MAX_ADMISSION_RATE = 0.70
MIN_ADMITTED_MEAN_PCT = 0.0
MIN_ADMITTED_POSITIVE_RATE = 0.50
MIN_TIMING_SPEARMAN = 0.05
MIN_EXECUTION_RATE = 0.10
MAX_EXECUTION_RATE = 0.80
MIN_TRADES = 30
MIN_IMPROVED_DAYS = 4
BOOTSTRAP_CI_LOW_MIN = 0.0


@dataclass(frozen=True)
class AdmissionModel:
    regressor: HistGradientBoostingRegressor
    classifier: HistGradientBoostingClassifier
    platt: LogisticRegression
    columns: tuple[str, ...]
    regression_offset: float
    winsor_low: float
    winsor_high: float


def enrich_hot_seconds(
    frame: pd.DataFrame,
    client: MassiveClient,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach only completed-second information available strictly before HOT."""
    result = frame.copy()
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    records: list[dict[str, float]] = []

    for row in result.loc[
        :, ["trading_day", "ticker", "hot_t"]
    ].to_dict("records"):
        day_text = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        key = (day_text, ticker)
        if key not in cache:
            payload = client.second_bars_range(
                ticker,
                date.fromisoformat(day_text),
                date.fromisoformat(day_text),
                adjusted=False,
            )
            cache[key] = _second_frame(payload)

        state_t = int(row["hot_t"])
        seconds = cache[key]
        features = {
            **second_path_features(seconds, state_t),
            **rich_second_features(seconds, state_t),
        }
        records.append(features)

    second = pd.DataFrame(records, index=result.index)
    for feature in HOT_SECOND_FEATURES:
        result[feature] = pd.to_numeric(
            second.get(feature),
            errors="coerce",
        )

    coverage_by_feature = {
        feature: float(result[feature].notna().mean())
        for feature in HOT_SECOND_FEATURES
    }
    any_coverage = float(
        result.loc[:, HOT_SECOND_FEATURES]
        .notna()
        .any(axis=1)
        .mean()
    )
    rich_coverage = float(
        result.loc[:, RICH_SECOND_FEATURES]
        .notna()
        .any(axis=1)
        .mean()
    )
    path_coverage = float(
        result.loc[:, SECOND_FEATURES]
        .notna()
        .any(axis=1)
        .mean()
    )

    return result, {
        "rows": int(len(result)),
        "ticker_days": int(len(cache)),
        "any_second_coverage": any_coverage,
        "path_second_coverage": path_coverage,
        "rich_second_coverage": rich_coverage,
        "feature_coverage": coverage_by_feature,
    }


def _usable_columns(
    frame: pd.DataFrame,
    feature_family: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        column
        for column in feature_family
        if column in frame.columns
        and pd.to_numeric(
            frame[column], errors="coerce"
        ).notna().any()
    )


def train_admission(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_family: tuple[str, ...],
    *,
    regression_seed: int,
    classifier_seed: int,
) -> AdmissionModel:
    target = pd.to_numeric(
        fit["episode_persistent_tradability_pct"],
        errors="coerce",
    )
    train = fit.loc[target.notna()].copy()
    y = target.loc[target.notna()].to_numpy(dtype=float)
    if len(train) < 250:
        raise ValueError(
            f"request 230 insufficient fit HOT labels: {len(train)}"
        )

    columns = _usable_columns(train, feature_family)
    if not columns:
        raise ValueError("request 230 has no usable HOT features")

    low, high = np.quantile(y, [0.005, 0.995])
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=regression_seed,
    )
    regressor.fit(
        _feature_frame(train, columns),
        np.clip(y, low, high),
        sample_weight=_day_weights(train),
    )

    label = (y > 0).astype(int)
    if np.unique(label).size < 2:
        raise ValueError("request 230 fit target has one class")
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=classifier_seed,
    )
    classifier.fit(
        _feature_frame(train, columns),
        label,
        sample_weight=_day_weights(train),
    )

    cal_target = pd.to_numeric(
        calibration["episode_persistent_tradability_pct"],
        errors="coerce",
    )
    cal = calibration.loc[cal_target.notna()].copy()
    cal_y = cal_target.loc[cal_target.notna()].to_numpy(dtype=float)
    if len(cal) < 120:
        raise ValueError(
            f"request 230 insufficient calibration HOT labels: {len(cal)}"
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
            "request 230 calibration target has one class"
        )
    raw_p = np.clip(
        classifier.predict_proba(x_cal)[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw_p / (1.0 - raw_p)).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=classifier_seed,
    )
    platt.fit(logits, cal_label)

    return AdmissionModel(
        regressor=regressor,
        classifier=classifier,
        platt=platt,
        columns=columns,
        regression_offset=regression_offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def score_admission(
    frame: pd.DataFrame,
    fitted: AdmissionModel,
    prefix: str,
) -> pd.DataFrame:
    result = frame.copy()
    x = _feature_frame(result, fitted.columns)

    value_column = f"{prefix}_predicted_tradability_pct"
    probability_column = f"{prefix}_positive_probability"
    admitted_column = f"{prefix}_admitted"

    result[value_column] = (
        fitted.regressor.predict(x)
        + fitted.regression_offset
    )
    raw_p = np.clip(
        fitted.classifier.predict_proba(x)[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw_p / (1.0 - raw_p)).reshape(-1, 1)
    result[probability_column] = (
        fitted.platt.predict_proba(logits)[:, 1]
    )
    result[admitted_column] = pd.to_numeric(
        result[value_column],
        errors="coerce",
    ).gt(0)
    return result


def admission_diag_for(
    frame: pd.DataFrame,
    prefix: str,
) -> dict[str, object]:
    renamed = frame.copy()
    renamed["predicted_persistent_tradability_pct"] = (
        renamed[f"{prefix}_predicted_tradability_pct"]
    )
    renamed[
        "predicted_tradability_positive_probability"
    ] = renamed[f"{prefix}_positive_probability"]
    renamed["hot_admitted"] = renamed[
        f"{prefix}_admitted"
    ]
    return admission_diagnostics(renamed)


def admitted_key_set(
    frame: pd.DataFrame,
    prefix: str,
) -> set[tuple[str, str, int]]:
    admitted = frame.loc[
        frame[f"{prefix}_admitted"]
        .fillna(False)
        .astype(bool)
    ]
    return {
        (
            str(row.trading_day),
            str(row.ticker).upper(),
            int(row.hot_t),
        )
        for row in admitted.itertuples(index=False)
    }


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )
    history_scan = pd.read_parquet(history_scan_path)

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
            "request 230 requires complete Request178 shards"
        )

    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
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
            build_watch_states(calibration_positions)
        ),
        calibration_positions,
    )
    fresh_watch = attach_multi_event_entry_labels(
        attach_transition_features(
            build_watch_states(fresh_positions)
        ),
        fresh_positions,
    )

    fit_target = attach_persistent_tradability(fit_watch)
    cal_target = attach_persistent_tradability(cal_watch)
    fresh_target = attach_persistent_tradability(fresh_watch)

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

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR,
        request_interval_seconds=0.0,
    )
    fit_hot, fit_second_audit = enrich_hot_seconds(
        fit_hot, client
    )
    cal_hot, cal_second_audit = enrich_hot_seconds(
        cal_hot, client
    )
    fresh_hot, fresh_second_audit = enrich_hot_seconds(
        fresh_hot, client
    )

    baseline_model = train_admission(
        fit_hot,
        cal_hot,
        HOT_CONTEXT_FEATURES,
        regression_seed=REGRESSION_SEED,
        classifier_seed=CLASSIFIER_SEED,
    )
    candidate_model = train_admission(
        fit_hot,
        cal_hot,
        HOT_AUGMENTED_FEATURES,
        regression_seed=SECOND_REGRESSION_SEED,
        classifier_seed=SECOND_CLASSIFIER_SEED,
    )

    cal_hot = score_admission(
        cal_hot, baseline_model, "baseline"
    )
    cal_hot = score_admission(
        cal_hot, candidate_model, "second"
    )
    fresh_hot = score_admission(
        fresh_hot, baseline_model, "baseline"
    )
    fresh_hot = score_admission(
        fresh_hot, candidate_model, "second"
    )

    calibration_baseline = admission_diag_for(
        cal_hot, "baseline"
    )
    calibration_candidate = admission_diag_for(
        cal_hot, "second"
    )
    fresh_baseline = admission_diag_for(
        fresh_hot, "baseline"
    )
    fresh_candidate = admission_diag_for(
        fresh_hot, "second"
    )

    timing_head = train_head(
        fit_watch,
        cal_watch,
        target_column=(
            "enter_vs_wait_multi_event_advantage_pct"
        ),
        seed=ADVANTAGE_SEED,
    )
    fresh_watch[
        "predicted_enter_vs_wait_advantage_pct"
    ] = predict_head(fresh_watch, timing_head)

    baseline_keys = admitted_key_set(
        fresh_hot, "baseline"
    )
    candidate_keys = admitted_key_set(
        fresh_hot, "second"
    )

    baseline_timing = timing_diagnostics(
        fresh_watch, baseline_keys
    )
    candidate_timing = timing_diagnostics(
        fresh_watch, candidate_keys
    )
    baseline_trades_229, baseline_counts_229 = (
        admitted_relative_policy(
            fresh_watch, baseline_keys
        )
    )
    candidate_trades, candidate_counts = (
        admitted_relative_policy(
            fresh_watch, candidate_keys
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
        request208_fit, request208_cal
    )
    request208_fresh = attach_relative_advantage(
        fresh_watch.copy()
    )
    request208_fresh[
        "predicted_request208_advantage_pct"
    ] = predict_request208(
        request208_fresh, request208_model
    )
    request208_trades = request208_policy(
        request208_fresh
    )

    candidate_portfolio = portfolio_rows(
        fresh_positions,
        candidate_trades,
        "request230_hot_second_admission",
    )
    baseline229_portfolio = portfolio_rows(
        fresh_positions,
        baseline_trades_229,
        "request229_minute_hot_admission",
    )
    request208_portfolio = portfolio_rows(
        fresh_positions,
        request208_trades,
        "request208_relative_entry",
    )

    candidate_metrics = portfolio_metrics(
        candidate_portfolio
    )
    baseline229_metrics = portfolio_metrics(
        baseline229_portfolio
    )
    request208_metrics = portfolio_metrics(
        request208_portfolio
    )
    candidate_minus_229 = matched_portfolio_difference(
        candidate_portfolio,
        baseline229_portfolio,
    )
    candidate_minus_208 = matched_portfolio_difference(
        candidate_portfolio,
        request208_portfolio,
    )

    baseline_s = fresh_baseline.get("spearman")
    candidate_s = fresh_candidate.get("spearman")
    baseline_auc = fresh_baseline.get("positive_auc")
    candidate_auc = fresh_candidate.get("positive_auc")
    spearman_gain = (
        float(candidate_s) - float(baseline_s)
        if candidate_s is not None and baseline_s is not None
        else None
    )
    auc_gain = (
        float(candidate_auc) - float(baseline_auc)
        if candidate_auc is not None and baseline_auc is not None
        else None
    )

    second_coverage = fresh_second_audit[
        "any_second_coverage"
    ]
    admission_rate = fresh_candidate.get("admission_rate")
    admitted_mean = fresh_candidate.get(
        "admitted_actual_mean_pct"
    )
    admitted_positive = fresh_candidate.get(
        "admitted_actual_positive_rate"
    )
    timing_s = candidate_timing.get("spearman")
    execution_rate = candidate_metrics.get("execution_rate")
    trades = candidate_metrics.get("executed", 0)
    trade_mean = candidate_metrics.get("trade_mean_pct")
    candidate_day = candidate_metrics.get(
        "portfolio_day_balanced_mean_pct"
    )
    diff208 = candidate_minus_208.get(
        "day_balanced_difference_pct"
    )
    diff208_low = candidate_minus_208.get(
        "bootstrap", {}
    ).get("ci_low_pct")
    severe = candidate_metrics.get(
        "trade_severe_loss_rate"
    )
    severe208 = request208_metrics.get(
        "trade_severe_loss_rate"
    )

    gate = bool(
        second_coverage >= MIN_SECOND_COVERAGE
        and candidate_s is not None
        and float(candidate_s) >= MIN_CANDIDATE_SPEARMAN
        and spearman_gain is not None
        and float(spearman_gain)
        >= MIN_SPEARMAN_GAIN_VS_229
        and candidate_auc is not None
        and float(candidate_auc) >= MIN_CANDIDATE_AUC
        and auc_gain is not None
        and float(auc_gain) >= MIN_AUC_GAIN_VS_229
        and admission_rate is not None
        and MIN_ADMISSION_RATE
        <= float(admission_rate)
        <= MAX_ADMISSION_RATE
        and admitted_mean is not None
        and float(admitted_mean) > MIN_ADMITTED_MEAN_PCT
        and admitted_positive is not None
        and float(admitted_positive)
        >= MIN_ADMITTED_POSITIVE_RATE
        and timing_s is not None
        and float(timing_s) >= MIN_TIMING_SPEARMAN
        and execution_rate is not None
        and MIN_EXECUTION_RATE
        <= float(execution_rate)
        <= MAX_EXECUTION_RATE
        and int(trades) >= MIN_TRADES
        and trade_mean is not None
        and float(trade_mean) > 0
        and candidate_day is not None
        and float(candidate_day) > 0
        and diff208 is not None
        and float(diff208) > 0
        and diff208_low is not None
        and float(diff208_low) > BOOTSTRAP_CI_LOW_MIN
        and int(
            candidate_minus_208.get("improved_days", 0)
        ) >= MIN_IMPROVED_DAYS
        and severe is not None
        and severe208 is not None
        and float(severe) <= float(severe208)
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "population": (
            "same frozen upstream BUY-anchor episodes as Request229"
        ),
        "architecture": (
            "HOT minute+market context plus causal completed-second "
            "microstructure admission -> frozen Request227 relative "
            "ENTER-vs-WAIT timing"
        ),
        "tradability_target": {
            "definition": (
                "same Request229 best mean multi-event value across "
                "two consecutive WATCH checkpoints"
            ),
            "future_information_used_as_input": False,
        },
        "second_features": list(HOT_SECOND_FEATURES),
        "feature_counts": {
            "request229_baseline": len(
                baseline_model.columns
            ),
            "request230_candidate": len(
                candidate_model.columns
            ),
        },
        "second_feature_audit": {
            "fit": fit_second_audit,
            "calibration": cal_second_audit,
            "fresh": fresh_second_audit,
            "client_stats": client.stats.to_dict(),
        },
        "calibration_admission": {
            "request229_minute": calibration_baseline,
            "request230_second": calibration_candidate,
        },
        "fresh_admission": {
            "request229_minute": fresh_baseline,
            "request230_second": fresh_candidate,
            "spearman_gain": spearman_gain,
            "auc_gain": auc_gain,
        },
        "fresh_relative_timing": {
            "request229_admitted": baseline_timing,
            "request230_admitted": candidate_timing,
        },
        "candidate_path_counts": candidate_counts,
        "request229_path_counts": baseline_counts_229,
        "fresh_policy": {
            "request208": request208_metrics,
            "request229": baseline229_metrics,
            "request230": candidate_metrics,
            "request230_minus_request229": candidate_minus_229,
            "request230_minus_request208": candidate_minus_208,
        },
        "frozen_gate": {
            "min_second_feature_coverage": MIN_SECOND_COVERAGE,
            "min_candidate_spearman": MIN_CANDIDATE_SPEARMAN,
            "min_spearman_gain_vs_request229": (
                MIN_SPEARMAN_GAIN_VS_229
            ),
            "min_candidate_auc": MIN_CANDIDATE_AUC,
            "min_auc_gain_vs_request229": MIN_AUC_GAIN_VS_229,
            "min_admission_rate": MIN_ADMISSION_RATE,
            "max_admission_rate": MAX_ADMISSION_RATE,
            "admitted_mean_must_be_positive": True,
            "min_admitted_positive_rate": (
                MIN_ADMITTED_POSITIVE_RATE
            ),
            "min_timing_spearman": MIN_TIMING_SPEARMAN,
            "min_execution_rate": MIN_EXECUTION_RATE,
            "max_execution_rate": MAX_EXECUTION_RATE,
            "min_trades": MIN_TRADES,
            "trade_mean_must_be_positive": True,
            "portfolio_day_balanced_mean_must_be_positive": True,
            "must_beat_request208": True,
            "matched_vs_request208_bootstrap_ci_low_positive": True,
            "min_improved_days_vs_request208": MIN_IMPROVED_DAYS,
            "severe_loss_no_worse_than_request208": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze HOT second-scale tradability admission and rebuild "
            "the full upstream HOT promotion population before old BUY "
            "gating, then replay recurrent entry/position management."
        ),
        "next_boundary_if_fail": (
            "HOT second-scale microstructure does not close the gap; "
            "treat runner discovery and tradability as separate latent "
            "tasks and learn tradability earlier from the full HOT "
            "promotion population rather than the already BUY-selected "
            "subset."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fresh_hot.to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-positions", type=Path, required=True
    )
    parser.add_argument(
        "--calibration-positions", type=Path, required=True
    )
    parser.add_argument(
        "--history-scan", type=Path, required=True
    )
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
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
