"""Request 217: phase-aware pullback ranking diagnostic.

Request215/216 show that exact-clock accounting was not the main bottleneck.
This request changes the state representation and learning objective rather than
tuning an EV threshold.

The model predicts the probability that an executable 3-minute event-time entry
covers BASE round-trip costs.  The candidate representation explicitly encodes
pullback phase and turn quality from information known at the current state.
Selection uses a probability threshold frozen from the 90th percentile of the
calibration score distribution, never an evaluation-tuned threshold.

A same-capacity baseline classifier without the explicit phase features is
trained in parallel.  This request is diagnostic only; overlapping watch states
are reduced to the first threshold crossing per episode before trade metrics are
reported.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .event_time_executable_entry import (
    attach_event_time_execution,
    prepare,
)
from .pullback_turn_transition_entry import MODEL_FEATURES as BASE_FEATURES
from .recurrent_wait_entry_action_value import BOOTSTRAP_SAMPLES

REQUEST_ID = 217
MODEL_SEED = 20261127
BOOTSTRAP_SEED = 20261128
CALIBRATION_SELECTION_QUANTILE = 0.90
MIN_AUC = 0.63
MIN_AUC_UPLIFT = 0.03
MIN_RESOLVED_SELECTION_COVERAGE = 0.90
MIN_SELECTED_TRADES = 20
MIN_SELECTED_POSITIVE_RATE = 0.35
MIN_POSITIVE_DAYS = 4

PHASE_FEATURES = (
    "phase_pullback_depth_pct",
    "phase_rebound_pct",
    "phase_recovery_fraction",
    "phase_turn_5",
    "phase_turn_10",
    "phase_turn_15",
    "phase_decelerating_selloff_5",
    "phase_decelerating_selloff_15",
    "phase_short_momentum_agreement",
    "phase_recent_low_10s",
    "phase_recent_low_20s",
    "phase_rebound_after_recent_low",
    "phase_recent_high_10s",
    "phase_vwap_reclaim",
    "phase_vwap_reclaim_strength",
    "phase_positive_signed_flow",
    "phase_flow_agreement",
    "phase_clean_turn",
    "phase_volume_confirmation",
    "phase_transactions_confirmation",
    "phase_attention_resilience",
    "phase_turn_with_attention",
)
PHASE_MODEL_FEATURES = tuple(
    dict.fromkeys([*BASE_FEATURES, *PHASE_FEATURES])
)


@dataclass(frozen=True)
class ProbabilityModel:
    model: HistGradientBoostingClassifier
    platt: LogisticRegression
    feature_columns: tuple[str, ...]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    den = denominator.where(denominator.abs().gt(1e-9))
    result = numerator / den
    return result.replace([np.inf, -np.inf], np.nan)


def attach_phase_features(states: pd.DataFrame) -> pd.DataFrame:
    """Encode causal pullback/turn phase from current and past observations."""
    result = states.copy()

    drawdown = _numeric(result, "watch_drawdown_from_peak_pct")
    rebound = _numeric(result, "watch_rebound_from_trough_pct")
    pullback_depth = (-drawdown).clip(lower=0)
    rebound_pos = rebound.clip(lower=0)
    result["phase_pullback_depth_pct"] = pullback_depth
    result["phase_rebound_pct"] = rebound_pos
    result["phase_recovery_fraction"] = (
        _safe_divide(rebound_pos, pullback_depth + 1e-6)
        .clip(lower=0, upper=3)
    )

    last5 = _numeric(result, "sec_last5_return_pct")
    prev5 = _numeric(result, "sec_prev5_return_pct")
    last10 = _numeric(result, "sec_last10_return_pct")
    prev10 = _numeric(result, "sec_prev10_return_pct")
    last15 = _numeric(result, "sec_last15_return_pct")
    prev15 = _numeric(result, "sec_prev15_return_pct")

    def turn(previous: pd.Series, current: pd.Series) -> pd.Series:
        valid = previous.notna() & current.notna()
        return (
            valid & previous.lt(0) & current.gt(0)
        ).astype(float).where(valid)

    result["phase_turn_5"] = turn(prev5, last5)
    result["phase_turn_10"] = turn(prev10, last10)
    result["phase_turn_15"] = turn(prev15, last15)

    valid5 = prev5.notna() & last5.notna()
    valid15 = prev15.notna() & last15.notna()
    result["phase_decelerating_selloff_5"] = (
        valid5 & prev5.lt(0) & last5.gt(prev5)
    ).astype(float).where(valid5)
    result["phase_decelerating_selloff_15"] = (
        valid15 & prev15.lt(0) & last15.gt(prev15)
    ).astype(float).where(valid15)

    momentum_flags = pd.concat(
        [last5.gt(0), last10.gt(0), last15.gt(0)],
        axis=1,
    )
    momentum_valid = pd.concat(
        [last5.notna(), last10.notna(), last15.notna()],
        axis=1,
    )
    result["phase_short_momentum_agreement"] = (
        momentum_flags.sum(axis=1)
        / momentum_valid.sum(axis=1).replace(0, np.nan)
    )

    since_low = _numeric(result, "sec_seconds_since_low")
    since_high = _numeric(result, "sec_seconds_since_high")
    result["phase_recent_low_10s"] = (
        since_low.notna() & since_low.le(10)
    ).astype(float).where(since_low.notna())
    result["phase_recent_low_20s"] = (
        since_low.notna() & since_low.le(20)
    ).astype(float).where(since_low.notna())
    result["phase_recent_high_10s"] = (
        since_high.notna() & since_high.le(10)
    ).astype(float).where(since_high.notna())
    result["phase_rebound_after_recent_low"] = (
        since_low.notna()
        & last5.notna()
        & since_low.le(20)
        & last5.gt(0)
    ).astype(float).where(since_low.notna() & last5.notna())

    close_vwap = _numeric(result, "sec_close_vs_vwap_pct")
    vwap_change = _numeric(result, "watch_close_vs_vwap_change_1m")
    reclaim_valid = close_vwap.notna() & vwap_change.notna()
    result["phase_vwap_reclaim"] = (
        reclaim_valid & close_vwap.gt(0) & vwap_change.gt(0)
    ).astype(float).where(reclaim_valid)
    result["phase_vwap_reclaim_strength"] = (
        close_vwap.clip(lower=0) * vwap_change.clip(lower=0)
    )

    signed_volume = _numeric(result, "sec_signed_volume_imbalance_60")
    signed_tx = _numeric(result, "sec_signed_transactions_imbalance_60")
    flow_valid = signed_volume.notna() | signed_tx.notna()
    result["phase_positive_signed_flow"] = (
        pd.concat([signed_volume, signed_tx], axis=1)
        .mean(axis=1, skipna=True)
        .where(flow_valid)
    )
    flow_agree_valid = signed_volume.notna() & signed_tx.notna()
    result["phase_flow_agreement"] = (
        flow_agree_valid & signed_volume.gt(0) & signed_tx.gt(0)
    ).astype(float).where(flow_agree_valid)

    efficiency = _numeric(result, "sec_return_efficiency_60")
    flip_rate = _numeric(result, "sec_sign_flip_rate_60")
    turn5 = _numeric(result, "phase_turn_5").fillna(0)
    clean_components_valid = efficiency.notna() & flip_rate.notna()
    result["phase_clean_turn"] = (
        turn5
        * efficiency.clip(lower=0)
        * (1.0 - flip_rate.clip(lower=0, upper=1))
    ).where(clean_components_valid)

    volume_burst = _numeric(result, "sec_volume_burst_5")
    tx_burst = _numeric(result, "sec_transactions_burst_5")
    positive_short = last5.clip(lower=0)
    result["phase_volume_confirmation"] = (
        positive_short * np.log1p(volume_burst.clip(lower=0))
    )
    result["phase_transactions_confirmation"] = (
        positive_short * np.log1p(tx_burst.clip(lower=0))
    )

    attention_drawdown = _numeric(
        result, "watch_attention_drawdown_from_peak"
    )
    attention_change = _numeric(result, "watch_attention_change_1m")
    result["phase_attention_resilience"] = (
        attention_change.fillna(0)
        - attention_drawdown.abs().fillna(0)
    ).where(attention_change.notna() | attention_drawdown.notna())
    result["phase_turn_with_attention"] = (
        turn5 * attention_change.clip(lower=0)
    ).where(attention_change.notna())

    return result


def attach_state_timestamp(
    states: pd.DataFrame,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the current executable state timestamp for cross-sectional audit."""
    records: list[dict[str, object]] = []
    for _, row in positions.iterrows():
        minute = pd.to_numeric(
            pd.Series([row.get("minutes_held")]), errors="coerce"
        ).iloc[0]
        state_t = pd.to_numeric(
            pd.Series([row.get("state_t")]), errors="coerce"
        ).iloc[0]
        if (
            pd.notna(minute)
            and np.isfinite(float(minute))
            and abs(float(minute) - round(float(minute))) <= 1e-9
            and 1 <= int(round(float(minute))) <= 5
            and pd.notna(state_t)
            and np.isfinite(float(state_t))
        ):
            records.append(
                {
                    "trading_day": str(row["trading_day"]),
                    "ticker": str(row["ticker"]).upper(),
                    "hot_t": int(row["hot_t"]),
                    "minutes_since_hot": int(round(float(minute))),
                    "state_t": int(float(state_t)),
                }
            )
    timestamp = pd.DataFrame(records)
    if timestamp.empty:
        result = states.copy()
        result["state_t"] = np.nan
        return result
    timestamp = timestamp.drop_duplicates(
        EPISODE_KEYS + ["minutes_since_hot"],
        keep="first",
    )
    return states.merge(
        timestamp,
        on=EPISODE_KEYS + ["minutes_since_hot"],
        how="left",
        validate="one_to_one",
    )


def build_states(positions: pd.DataFrame) -> pd.DataFrame:
    base = prepare(positions)
    base = attach_event_time_execution(base, positions)
    base = attach_state_timestamp(base, positions)
    return attach_phase_features(base)


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric, errors="coerce"
    )


def _day_class_weights(
    frame: pd.DataFrame,
    target: pd.Series,
) -> np.ndarray:
    temp = pd.DataFrame(
        {
            "day": frame["trading_day"].astype(str).to_numpy(),
            "class": target.astype(int).to_numpy(),
        },
        index=frame.index,
    )
    counts = temp.groupby(["day", "class"]).size()
    weights = np.array(
        [
            1.0 / float(counts.loc[(day, int(cls))])
            for day, cls in zip(
                temp["day"], temp["class"], strict=True
            )
        ],
        dtype=float,
    )
    mean = float(np.mean(weights))
    return weights / mean if mean > 0 else weights


def train_probability_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    seed: int,
) -> ProbabilityModel:
    fit_target = pd.to_numeric(
        fit["enter_3m_event_base_pct"], errors="coerce"
    )
    cal_target = pd.to_numeric(
        calibration["enter_3m_event_base_pct"], errors="coerce"
    )
    fit_valid = fit_target.notna()
    cal_valid = cal_target.notna()
    train = fit.loc[fit_valid].copy()
    cal = calibration.loc[cal_valid].copy()
    y_fit = fit_target.loc[fit_valid].gt(0).astype(int)
    y_cal = cal_target.loc[cal_valid].gt(0).astype(int)
    if (
        len(train) < 500
        or len(cal) < 250
        or y_fit.nunique() < 2
        or y_cal.nunique() < 2
    ):
        raise ValueError("request 217 insufficient positive/negative support")

    columns = tuple(
        column
        for column in feature_columns
        if column in train.columns
        and pd.to_numeric(
            train[column], errors="coerce"
        ).notna().any()
    )
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _feature_frame(train, columns),
        y_fit,
        sample_weight=_day_class_weights(train, y_fit),
    )

    raw = np.clip(
        model.predict_proba(_feature_frame(cal, columns))[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    platt.fit(
        logits,
        y_cal,
        sample_weight=_day_class_weights(cal, y_cal),
    )
    return ProbabilityModel(
        model=model,
        platt=platt,
        feature_columns=columns,
    )


def predict_probability(
    frame: pd.DataFrame,
    fitted: ProbabilityModel,
) -> np.ndarray:
    raw = np.clip(
        fitted.model.predict_proba(
            _feature_frame(frame, fitted.feature_columns)
        )[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    return fitted.platt.predict_proba(logits)[:, 1]


def _auc(
    frame: pd.DataFrame,
    probability: pd.Series,
) -> float | None:
    target = pd.to_numeric(
        frame["enter_3m_event_base_pct"], errors="coerce"
    )
    valid = target.notna() & probability.notna()
    y = target.loc[valid].gt(0).astype(int)
    if len(y) < 20 or y.nunique() < 2:
        return None
    return float(
        roc_auc_score(
            y,
            probability.loc[valid].to_numpy(dtype=float),
        )
    )


def _bootstrap_daily(
    trades: pd.DataFrame,
) -> dict[str, object]:
    if trades.empty:
        return {
            "days": 0,
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    daily = (
        trades.assign(
            _value=pd.to_numeric(
                trades["realized_base_return_pct"],
                errors="coerce",
            )
        )
        .groupby(trades["trading_day"].astype(str))["_value"]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(daily) < 2:
        return {
            "days": int(len(daily)),
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": float(daily.mean()) if len(daily) else None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    index = rng.integers(
        0, len(daily), size=(BOOTSTRAP_SAMPLES, len(daily))
    )
    samples = daily[index].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "days": int(len(daily)),
        "samples": BOOTSTRAP_SAMPLES,
        "mean_pct": float(daily.mean()),
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def _trade_metrics(
    selected: pd.DataFrame,
) -> dict[str, object]:
    if selected.empty:
        return {"attempts": 0, "resolved_trades": 0}
    value = pd.to_numeric(
        selected["enter_3m_event_base_pct"], errors="coerce"
    )
    resolved = selected.loc[value.notna()].copy()
    resolved["realized_base_return_pct"] = value.loc[
        value.notna()
    ].to_numpy(dtype=float)
    coverage = float(len(resolved) / len(selected))
    if resolved.empty:
        return {
            "attempts": int(len(selected)),
            "resolved_trades": 0,
            "resolved_coverage": coverage,
        }
    daily = (
        resolved.groupby(
            resolved["trading_day"].astype(str)
        )["realized_base_return_pct"].mean()
    )
    return {
        "attempts": int(len(selected)),
        "resolved_trades": int(len(resolved)),
        "resolved_coverage": coverage,
        "mean_pct": float(
            resolved["realized_base_return_pct"].mean()
        ),
        "day_balanced_mean_pct": float(daily.mean()),
        "positive_rate": float(
            resolved["realized_base_return_pct"].gt(0).mean()
        ),
        "positive_days": int(daily.gt(0).sum()),
        "p05_pct": float(
            resolved["realized_base_return_pct"].quantile(0.05)
        ),
        "by_day": {
            str(day): float(value)
            for day, value in daily.items()
        },
        "bootstrap": _bootstrap_daily(resolved),
    }


def first_crossing(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    threshold: float,
) -> pd.DataFrame:
    chosen: list[pd.Series] = []
    for _, group in frame.groupby(EPISODE_KEYS, sort=False):
        ordered = group.sort_values(
            "minutes_since_hot", kind="stable"
        )
        passing = ordered.loc[
            pd.to_numeric(
                ordered[probability_column], errors="coerce"
            ).ge(threshold)
        ]
        if not passing.empty:
            chosen.append(passing.iloc[0])
    if not chosen:
        return frame.iloc[0:0].copy()
    return pd.DataFrame(chosen).reset_index(drop=True)


def cross_sectional_leaders(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    work = frame.copy()
    state_t = pd.to_numeric(work["state_t"], errors="coerce")
    work = work.loc[state_t.notna()].copy()
    work["_clock_minute"] = (
        pd.to_numeric(work["state_t"], errors="coerce") // 60_000
    ).astype("int64")
    probability = pd.to_numeric(
        work[probability_column], errors="coerce"
    )
    passing = work.loc[probability.ge(threshold)].copy()
    if passing.empty:
        return passing, {
            "clock_buckets": 0,
            "multi_candidate_buckets": 0,
            "max_candidates_in_bucket": 0,
        }

    bucket_sizes = passing.groupby(
        ["trading_day", "_clock_minute"]
    ).size()
    leaders = (
        passing.sort_values(
            probability_column,
            ascending=False,
            kind="stable",
        )
        .groupby(
            ["trading_day", "_clock_minute"],
            sort=False,
            as_index=False,
        )
        .head(1)
        .sort_values("state_t", kind="stable")
    )
    leaders = (
        leaders.groupby(EPISODE_KEYS, sort=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return leaders, {
        "clock_buckets": int(len(bucket_sizes)),
        "multi_candidate_buckets": int(
            bucket_sizes.gt(1).sum()
        ),
        "max_candidates_in_bucket": int(bucket_sizes.max()),
    }


def model_diagnostics(
    frame: pd.DataFrame,
    probability_column: str,
) -> dict[str, object]:
    probability = pd.to_numeric(
        frame[probability_column], errors="coerce"
    )
    target = pd.to_numeric(
        frame["enter_3m_event_base_pct"], errors="coerce"
    )
    valid = target.notna() & probability.notna()
    by_day: dict[str, object] = {}
    for day, group in frame.loc[valid].groupby(
        frame.loc[valid, "trading_day"].astype(str)
    ):
        p = pd.to_numeric(
            group[probability_column], errors="coerce"
        )
        y = pd.to_numeric(
            group["enter_3m_event_base_pct"], errors="coerce"
        )
        by_day[str(day)] = {
            "rows": int(len(group)),
            "positive_rate": float(y.gt(0).mean()),
            "auc": _auc(group, p),
        }
    return {
        "evaluable_states": int(valid.sum()),
        "realized_positive_rate": float(
            target.loc[valid].gt(0).mean()
        ) if int(valid.sum()) else None,
        "auc": _auc(frame, probability),
        "probability_mean": float(
            probability.loc[valid].mean()
        ) if int(valid.sum()) else None,
        "probability_range": [
            float(probability.loc[valid].min()),
            float(probability.loc[valid].max()),
        ] if int(valid.sum()) else [None, None],
        "by_day": by_day,
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    fresh_dir: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_path)
    calibration_positions = pd.read_parquet(calibration_path)
    fresh_paths = sorted(
        fresh_dir.glob("*-positions.parquet")
    )
    if len(fresh_paths) != len(FRESH_DAYS):
        raise ValueError(
            "request 217 requires complete Request178 position shards"
        )
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in fresh_paths],
        ignore_index=True,
    )

    fit = build_states(fit_positions)
    calibration = build_states(calibration_positions)
    fresh = build_states(fresh_positions)
    found_days = sorted(
        fresh["trading_day"].astype(str).unique()
    )
    if found_days != list(FRESH_DAYS):
        raise ValueError(
            f"request 217 expected {FRESH_DAYS}, found {found_days}"
        )

    baseline_model = train_probability_model(
        fit,
        calibration,
        feature_columns=BASE_FEATURES,
        seed=MODEL_SEED,
    )
    phase_model = train_probability_model(
        fit,
        calibration,
        feature_columns=PHASE_MODEL_FEATURES,
        seed=MODEL_SEED + 1,
    )

    cal_baseline = predict_probability(
        calibration, baseline_model
    )
    cal_phase = predict_probability(
        calibration, phase_model
    )
    baseline_threshold = float(
        np.quantile(
            cal_baseline[
                pd.to_numeric(
                    calibration[
                        "enter_3m_event_base_pct"
                    ],
                    errors="coerce",
                ).notna()
            ],
            CALIBRATION_SELECTION_QUANTILE,
        )
    )
    phase_threshold = float(
        np.quantile(
            cal_phase[
                pd.to_numeric(
                    calibration[
                        "enter_3m_event_base_pct"
                    ],
                    errors="coerce",
                ).notna()
            ],
            CALIBRATION_SELECTION_QUANTILE,
        )
    )

    scored = fresh.copy()
    scored["baseline_positive_probability"] = (
        predict_probability(scored, baseline_model)
    )
    scored["phase_positive_probability"] = (
        predict_probability(scored, phase_model)
    )

    baseline_selected = first_crossing(
        scored,
        probability_column="baseline_positive_probability",
        threshold=baseline_threshold,
    )
    phase_selected = first_crossing(
        scored,
        probability_column="phase_positive_probability",
        threshold=phase_threshold,
    )
    leaders, leader_context = cross_sectional_leaders(
        scored,
        probability_column="phase_positive_probability",
        threshold=phase_threshold,
    )

    baseline_diag = model_diagnostics(
        scored, "baseline_positive_probability"
    )
    phase_diag = model_diagnostics(
        scored, "phase_positive_probability"
    )
    baseline_metrics = _trade_metrics(baseline_selected)
    phase_metrics = _trade_metrics(phase_selected)
    leader_metrics = _trade_metrics(leaders)

    baseline_auc = baseline_diag.get("auc")
    phase_auc = phase_diag.get("auc")
    phase_mean = phase_metrics.get("mean_pct")
    phase_positive_rate = phase_metrics.get("positive_rate")
    phase_positive_days = phase_metrics.get("positive_days")
    resolved_coverage = phase_metrics.get(
        "resolved_coverage"
    )
    resolved_trades = int(
        phase_metrics.get("resolved_trades", 0)
    )

    gate = bool(
        baseline_auc is not None
        and phase_auc is not None
        and float(phase_auc) >= MIN_AUC
        and float(phase_auc) - float(baseline_auc)
        >= MIN_AUC_UPLIFT
        and resolved_coverage is not None
        and float(resolved_coverage)
        >= MIN_RESOLVED_SELECTION_COVERAGE
        and resolved_trades >= MIN_SELECTED_TRADES
        and phase_mean is not None
        and float(phase_mean) > 0
        and phase_positive_rate is not None
        and float(phase_positive_rate)
        >= MIN_SELECTED_POSITIVE_RATE
        and phase_positive_days is not None
        and int(phase_positive_days) >= MIN_POSITIVE_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "objective": (
            "probability executable 3m event-time entry "
            "covers BASE round-trip costs"
        ),
        "selection_rule": (
            "first recurrent state crossing the calibration "
            "90th-percentile probability threshold"
        ),
        "calibration_selection_quantile": (
            CALIBRATION_SELECTION_QUANTILE
        ),
        "baseline": {
            "threshold": baseline_threshold,
            "diagnostics": baseline_diag,
            "first_crossing": baseline_metrics,
        },
        "phase_model": {
            "threshold": phase_threshold,
            "diagnostics": phase_diag,
            "first_crossing": phase_metrics,
            "cross_sectional_leader": {
                "context": leader_context,
                "metrics": leader_metrics,
            },
            "feature_count": int(
                len(phase_model.feature_columns)
            ),
            "explicit_phase_features": list(PHASE_FEATURES),
        },
        "request215_reference": {
            "positive_probability_auc": 0.5938870012096557,
            "predicted_ev_spearman": 0.06460355879404042,
            "all_policy_entries": 0,
            "realized_positive_state_rate": 0.18387096774193548,
        },
        "frozen_gate": {
            "min_auc": MIN_AUC,
            "min_auc_uplift_vs_same_capacity_baseline": MIN_AUC_UPLIFT,
            "min_resolved_selection_coverage": (
                MIN_RESOLVED_SELECTION_COVERAGE
            ),
            "min_selected_trades": MIN_SELECTED_TRADES,
            "selected_mean_base_must_be_positive": True,
            "min_selected_positive_rate": (
                MIN_SELECTED_POSITIVE_RATE
            ),
            "min_positive_days": MIN_POSITIVE_DAYS,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    scored_output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scored.to_parquet(
        scored_output_path,
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
        "--calibration-positions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fresh-dir", type=Path, required=True
    )
    parser.add_argument(
        "--output", type=Path, required=True
    )
    parser.add_argument(
        "--scored-output", type=Path, required=True
    )
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.fresh_dir,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
