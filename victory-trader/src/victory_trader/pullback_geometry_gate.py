"""Request 224: pullback-geometry audit and causal eligibility gate.

Request221 found that pullback onset contains transferable information.
Request222 showed that a fixed three-event wait is harmful. Request223 improved
that by re-evaluating every observed pullback state, but the ungated adaptive
controller was still slightly negative on the fresh development block.

Request224 asks a narrower question:

    Which pullbacks are worth handing to the adaptive controller at all?

The gate is decided once, at pullback onset, using only causal onset geometry
and microstructure. If a segment is ineligible, EXIT at onset. If eligible,
hand it to the frozen Request223 entry-conditioned adaptive controller.

The eligibility model is fit on historical fit rows. A small frozen set of
score quantiles is evaluated on calibration only to choose the gate threshold.
The Request178 fresh block is evaluated once after that choice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .adaptive_pullback_redecision import (
    BOOTSTRAP_SAMPLES,
    CANDIDATE_FEATURES as REQUEST223_FEATURES,
    build_pullback_wait_states,
    policy_metrics,
    predict_value,
    simulate_adaptive_policy,
    train_value_model,
)
from .entry_conditioned_rich_second import attach_entry_conditioned_features
from .transition_conditioned_option_value import (
    _prepare_historical,
    attach_transition_phase,
)

REQUEST_ID = 224
MODEL_SEED = 20261224
BOOTSTRAP_SEED = 20261224

THRESHOLD_QUANTILES = (0.40, 0.50, 0.60, 0.70, 0.80)
MIN_CAL_ELIGIBLE_RATE = 0.15
MAX_CAL_ELIGIBLE_RATE = 0.60

MIN_FRESH_ELIGIBLE_RATE = 0.10
MAX_FRESH_ELIGIBLE_RATE = 0.70
MIN_FRESH_ELIGIBILITY_SPEARMAN = 0.03
MIN_GOOD_DAYS = 3

DERIVED_GEOMETRY_FEATURES = (
    "geometry_onset_drop_pct",
    "geometry_previous_rise_pct",
    "geometry_reversal_ratio",
    "geometry_abs_accel_pct",
    "geometry_entry_cushion_pct",
)

GEOMETRY_SOURCE_FEATURES = (
    "position_return_change_1m_pct",
    "previous_observed_return_change_pct",
    "position_return_accel_1m_pct",
    "entry_to_current_close_pct",
    "drawdown_from_peak_pct",
    "recovery_from_trough_pct",
    "minutes_held",
    "position_drawdown_change_1m_pct",
    "position_recovery_change_1m_pct",
    "position_attention_change_1m",
    "position_rank_change_1m",
    "position_log_volume_change_1m",
    "position_log_transactions_change_1m",
    "position_active_seconds_change_1m",
    "position_sec_last5_return_change_1m",
    "position_sec_last10_return_change_1m",
    "position_close_vs_vwap_change_1m",
    "position_volume_burst5_change_1m",
    "position_transactions_burst5_change_1m",
    "sec_last5_return_pct",
    "sec_prev5_return_pct",
    "sec_accel_5_pct",
    "sec_last15_return_pct",
    "sec_prev15_return_pct",
    "sec_accel_15_pct",
    "sec_volume_burst_5",
    "sec_transactions_burst_5",
    "sec_signed_volume_imbalance_60",
    "sec_signed_transactions_imbalance_60",
    "sec_return_efficiency_60",
    "sec_sign_flip_rate_60",
    "sec_seconds_since_high",
    "sec_seconds_since_low",
    "sec_close_location_60",
    "sec_close_vs_vwap_pct",
    "sec_volatility_ratio_10_to_prev50",
    "sec_mean_trade_size_ratio_10_to_prev50",
    "entry_delta_sec_last5_return_pct",
    "entry_delta_sec_accel_5_pct",
    "entry_delta_sec_volume_burst_5",
    "entry_delta_sec_transactions_burst_5",
    "entry_delta_sec_signed_volume_imbalance_60",
    "entry_delta_sec_return_efficiency_60",
    "entry_delta_sec_close_location_60",
    "entry_delta_sec_close_vs_vwap_pct",
)

GEOMETRY_FEATURES = tuple(
    dict.fromkeys(
        [*DERIVED_GEOMETRY_FEATURES, *GEOMETRY_SOURCE_FEATURES]
    )
)


@dataclass(frozen=True)
class EligibilityModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    winsor_low: float
    winsor_high: float


def attach_geometry_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    current = pd.to_numeric(
        result.get("position_return_change_1m_pct"),
        errors="coerce",
    )
    previous = pd.to_numeric(
        result.get("previous_observed_return_change_pct"),
        errors="coerce",
    )
    accel = pd.to_numeric(
        result.get("position_return_accel_1m_pct"),
        errors="coerce",
    )
    cushion = pd.to_numeric(
        result.get("entry_to_current_close_pct"),
        errors="coerce",
    )

    result["geometry_onset_drop_pct"] = (-current).clip(lower=0.0)
    result["geometry_previous_rise_pct"] = previous.clip(lower=0.0)
    denominator = previous.abs().clip(lower=0.05)
    ratio = (-current).clip(lower=0.0) / denominator
    result["geometry_reversal_ratio"] = ratio.clip(upper=20.0)
    result["geometry_abs_accel_pct"] = accel.abs()
    result["geometry_entry_cushion_pct"] = cushion
    return result


def onset_rows(states: pd.DataFrame) -> pd.DataFrame:
    onset = states.loc[
        pd.to_numeric(
            states["pullback_age_events"], errors="coerce"
        ).eq(0)
    ].copy()
    onset = attach_geometry_features(onset)
    onset["segment_terminal_advantage_pct"] = pd.to_numeric(
        onset["adaptive_wait_value_pct"], errors="coerce"
    )
    return onset


def _usable_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in GEOMETRY_FEATURES
        if column in frame.columns
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    )


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric, errors="coerce"
    )


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    day = frame["trading_day"].astype(str)
    counts = day.map(day.value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(dtype=float)
    return weights / float(np.mean(weights))


def train_eligibility_model(fit_onset: pd.DataFrame) -> EligibilityModel:
    target = pd.to_numeric(
        fit_onset["segment_terminal_advantage_pct"],
        errors="coerce",
    )
    valid = target.notna()
    train = fit_onset.loc[valid].copy()
    y = target.loc[valid].to_numpy(dtype=float)
    if len(train) < 1000:
        raise ValueError(
            f"request 224 insufficient fit onset support: {len(train)}"
        )

    columns = _usable_columns(train)
    if not columns:
        raise ValueError("request 224 has no usable geometry features")

    low, high = np.quantile(y, [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=MODEL_SEED,
    )
    model.fit(
        _feature_frame(train, columns),
        np.clip(y, low, high),
        sample_weight=_day_weights(train),
    )
    return EligibilityModel(
        model=model,
        feature_columns=columns,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_eligibility(
    frame: pd.DataFrame,
    fitted: EligibilityModel,
) -> np.ndarray:
    return fitted.model.predict(
        _feature_frame(frame, fitted.feature_columns)
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


def _bootstrap_daily(
    daily: np.ndarray,
    seed: int,
) -> dict[str, object]:
    values = np.asarray(daily, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {
            "days": int(len(values)),
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": float(values.mean()) if len(values) else None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_SAMPLES, len(values)),
    )
    draws = values[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(values)),
        "samples": BOOTSTRAP_SAMPLES,
        "mean_pct": float(values.mean()),
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def gated_policy_rows(
    onset: pd.DataFrame,
    adaptive_rows: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    scores = onset.loc[
        :,
        [
            "pullback_segment_id",
            "trading_day",
            "ticker",
            "hot_t",
            "segment_terminal_advantage_pct",
            "eligibility_score",
        ],
    ].copy()
    merged = adaptive_rows.merge(
        scores,
        on=[
            "pullback_segment_id",
            "trading_day",
            "ticker",
            "hot_t",
        ],
        how="inner",
        validate="one_to_one",
    )
    merged["eligible"] = pd.to_numeric(
        merged["eligibility_score"], errors="coerce"
    ).ge(float(threshold))
    merged["ungated_uplift_pct"] = pd.to_numeric(
        merged["local_uplift_pct"], errors="coerce"
    )
    merged["local_uplift_pct"] = np.where(
        merged["eligible"],
        merged["ungated_uplift_pct"],
        0.0,
    )
    merged["policy"] = "geometry_gated_request223"
    return merged


def gate_metrics(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {"segments": 0}
    value = pd.to_numeric(rows["local_uplift_pct"], errors="coerce")
    valid = value.notna()
    work = rows.loc[valid].copy()
    work["_value"] = value.loc[valid]
    work["_eligible"] = work["eligible"].astype(bool)
    terminal = pd.to_numeric(
        work["segment_terminal_advantage_pct"],
        errors="coerce",
    )
    score = pd.to_numeric(work["eligibility_score"], errors="coerce")

    daily = work.groupby(
        work["trading_day"].astype(str), sort=True
    )["_value"].mean()
    good_days = int((daily > 0).sum())

    eligible = work.loc[work["_eligible"]].copy()
    eligible_terminal = pd.to_numeric(
        eligible["segment_terminal_advantage_pct"],
        errors="coerce",
    )

    return {
        "segments": int(len(work)),
        "eligible_segments": int(work["_eligible"].sum()),
        "eligible_rate": float(work["_eligible"].mean()),
        "eligibility_spearman": _safe_spearman(terminal, score),
        "eligible_terminal_advantage_mean_pct": (
            float(eligible_terminal.mean())
            if len(eligible)
            else None
        ),
        "mean_uplift_pct": float(work["_value"].mean()),
        "median_uplift_pct": float(work["_value"].median()),
        "p05_uplift_pct": float(work["_value"].quantile(0.05)),
        "positive_rate": float(work["_value"].gt(0).mean()),
        "day_balanced_uplift_pct": float(daily.mean()),
        "day_bootstrap": _bootstrap_daily(
            daily.to_numpy(dtype=float), BOOTSTRAP_SEED
        ),
        "good_days": good_days,
        "by_day": {
            str(day): {
                "segments": int(len(part)),
                "eligible_rate": float(part["_eligible"].mean()),
                "mean_uplift_pct": float(part["_value"].mean()),
            }
            for day, part in work.groupby(
                work["trading_day"].astype(str), sort=True
            )
        },
    }


def matched_gate_difference(
    gated: pd.DataFrame,
) -> dict[str, object]:
    work = gated.copy()
    work["difference_pct"] = (
        pd.to_numeric(work["local_uplift_pct"], errors="coerce")
        - pd.to_numeric(work["ungated_uplift_pct"], errors="coerce")
    )
    work = work.loc[work["difference_pct"].notna()].copy()
    if work.empty:
        return {"matched_segments": 0}
    daily = work.groupby(
        work["trading_day"].astype(str), sort=True
    )["difference_pct"].mean()
    return {
        "matched_segments": int(len(work)),
        "mean_difference_pct": float(work["difference_pct"].mean()),
        "gated_better_rate": float(work["difference_pct"].gt(0).mean()),
        "day_balanced_difference_pct": float(daily.mean()),
        "bootstrap": _bootstrap_daily(
            daily.to_numpy(dtype=float), BOOTSTRAP_SEED + 1
        ),
    }


def select_threshold(
    calibration_onset: pd.DataFrame,
    adaptive_rows: pd.DataFrame,
) -> tuple[float, dict[str, object]]:
    scores = pd.to_numeric(
        calibration_onset["eligibility_score"], errors="coerce"
    ).dropna()
    if len(scores) < 500:
        raise ValueError("request 224 insufficient calibration gate scores")

    candidates: list[dict[str, object]] = []
    for quantile in THRESHOLD_QUANTILES:
        threshold = float(scores.quantile(quantile))
        rows = gated_policy_rows(
            calibration_onset, adaptive_rows, threshold
        )
        metrics = gate_metrics(rows)
        eligible_rate = metrics.get("eligible_rate")
        valid_rate = bool(
            eligible_rate is not None
            and MIN_CAL_ELIGIBLE_RATE
            <= float(eligible_rate)
            <= MAX_CAL_ELIGIBLE_RATE
        )
        candidates.append(
            {
                "quantile": float(quantile),
                "threshold": threshold,
                "valid_eligible_rate": valid_rate,
                "metrics": metrics,
            }
        )

    valid = [
        item for item in candidates if item["valid_eligible_rate"]
    ]
    pool = valid if valid else candidates
    chosen = max(
        pool,
        key=lambda item: (
            float(
                item["metrics"].get(
                    "day_balanced_uplift_pct", -np.inf
                )
                if item["metrics"].get(
                    "day_balanced_uplift_pct"
                )
                is not None
                else -np.inf
            ),
            float(item["quantile"]),
        ),
    )
    return float(chosen["threshold"]), {
        "candidates": candidates,
        "selected_quantile": chosen["quantile"],
        "selected_threshold": chosen["threshold"],
        "selected_metrics": chosen["metrics"],
    }


def geometry_audit(
    frame: pd.DataFrame,
    *,
    max_features: int = 16,
) -> dict[str, object]:
    target = pd.to_numeric(
        frame["segment_terminal_advantage_pct"], errors="coerce"
    )
    adaptive = pd.to_numeric(
        frame.get("adaptive_policy_uplift_pct"), errors="coerce"
    )
    records: list[tuple[str, float, dict[str, object]]] = []

    for column in GEOMETRY_FEATURES:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna() & target.notna()
        if int(valid.sum()) < 100 or values.loc[valid].nunique() < 8:
            continue

        corr = values.loc[valid].corr(
            target.loc[valid], method="spearman"
        )
        corr_value = 0.0 if pd.isna(corr) else float(corr)
        try:
            bins = pd.qcut(
                values.loc[valid],
                q=4,
                duplicates="drop",
            )
        except ValueError:
            continue

        temp = pd.DataFrame(
            {
                "bin": bins.astype(str),
                "target": target.loc[valid],
                "adaptive": adaptive.loc[valid],
            }
        )
        grouped = []
        for name, part in temp.groupby("bin", sort=False):
            grouped.append(
                {
                    "bin": str(name),
                    "rows": int(len(part)),
                    "terminal_advantage_mean_pct": float(
                        part["target"].mean()
                    ),
                    "adaptive_policy_uplift_mean_pct": (
                        float(part["adaptive"].mean())
                        if part["adaptive"].notna().any()
                        else None
                    ),
                }
            )
        records.append(
            (
                column,
                abs(corr_value),
                {
                    "spearman_terminal_advantage": corr_value,
                    "quartiles": grouped,
                },
            )
        )

    records.sort(key=lambda item: item[1], reverse=True)
    return {
        name: payload
        for name, _, payload in records[:max_features]
    }


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    history_scan_path: Path,
    fresh_states_path: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )
    history_scan = pd.read_parquet(history_scan_path)
    fresh_states = pd.read_parquet(fresh_states_path).copy()

    fit_full, calibration_full = _prepare_historical(
        fit_positions,
        calibration_positions,
        history_scan,
    )
    fit_full = attach_entry_conditioned_features(
        attach_transition_phase(fit_full)
    )
    calibration_full = attach_entry_conditioned_features(
        attach_transition_phase(calibration_full)
    )

    fit_states, fit_support = build_pullback_wait_states(fit_full)
    cal_states, cal_support = build_pullback_wait_states(calibration_full)

    request223_model = train_value_model(
        fit_states,
        cal_states,
        REQUEST223_FEATURES,
    )
    cal_states["entry_conditioned_score"] = predict_value(
        cal_states, request223_model
    )

    if "entry_conditioned_score" not in fresh_states.columns:
        raise ValueError(
            "request 224 requires Request223 scored fresh pullback states"
        )

    fit_onset = onset_rows(fit_states)
    cal_onset = onset_rows(cal_states)
    fresh_onset = onset_rows(fresh_states)

    eligibility_model = train_eligibility_model(fit_onset)
    cal_onset["eligibility_score"] = predict_eligibility(
        cal_onset, eligibility_model
    )
    fresh_onset["eligibility_score"] = predict_eligibility(
        fresh_onset, eligibility_model
    )

    cal_adaptive = simulate_adaptive_policy(
        cal_states,
        "entry_conditioned_score",
        policy="request223_entry_conditioned",
    )
    fresh_adaptive = simulate_adaptive_policy(
        fresh_states,
        "entry_conditioned_score",
        policy="request223_entry_conditioned",
    )

    threshold, threshold_audit = select_threshold(
        cal_onset,
        cal_adaptive,
    )

    cal_gated_rows = gated_policy_rows(
        cal_onset, cal_adaptive, threshold
    )
    fresh_gated_rows = gated_policy_rows(
        fresh_onset, fresh_adaptive, threshold
    )

    cal_gated = gate_metrics(cal_gated_rows)
    fresh_gated = gate_metrics(fresh_gated_rows)
    fresh_ungated = policy_metrics(fresh_adaptive)
    difference = matched_gate_difference(fresh_gated_rows)

    cal_policy_map = cal_adaptive.set_index(
        "pullback_segment_id"
    )["local_uplift_pct"]
    cal_onset["adaptive_policy_uplift_pct"] = (
        cal_onset["pullback_segment_id"].map(cal_policy_map)
    )
    calibration_geometry_audit = geometry_audit(cal_onset)

    eligible_rate = fresh_gated.get("eligible_rate")
    spearman = fresh_gated.get("eligibility_spearman")
    day_uplift = fresh_gated.get("day_balanced_uplift_pct")
    day_ci_low = fresh_gated.get("day_bootstrap", {}).get(
        "ci_low_pct"
    )
    diff_day = difference.get("day_balanced_difference_pct")
    diff_ci_low = difference.get("bootstrap", {}).get("ci_low_pct")
    eligible_terminal = fresh_gated.get(
        "eligible_terminal_advantage_mean_pct"
    )

    gate = bool(
        eligible_rate is not None
        and MIN_FRESH_ELIGIBLE_RATE
        <= float(eligible_rate)
        <= MAX_FRESH_ELIGIBLE_RATE
        and spearman is not None
        and float(spearman) >= MIN_FRESH_ELIGIBILITY_SPEARMAN
        and eligible_terminal is not None
        and float(eligible_terminal) > 0
        and day_uplift is not None
        and float(day_uplift) > 0
        and day_ci_low is not None
        and float(day_ci_low) > 0
        and int(fresh_gated.get("good_days", 0)) >= MIN_GOOD_DAYS
        and diff_day is not None
        and float(diff_day) > 0
        and diff_ci_low is not None
        and float(diff_ci_low) > 0
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "hypothesis": (
            "Request223 is diluted by structurally bad pullbacks; a "
            "causal onset-geometry gate can restrict the adaptive "
            "controller to pullbacks with transferable continuation value."
        ),
        "policy": {
            "ineligible": "EXIT at pullback onset",
            "eligible": (
                "hand segment to frozen Request223 entry-conditioned "
                "adaptive pullback controller"
            ),
            "gate_decided_once_at_pullback_onset": True,
            "future_information_used_as_input": False,
        },
        "target": (
            "pullback terminal BASE advantage at first causal "
            "reacceleration/30m cap versus EXIT at onset"
        ),
        "feature_count": len(eligibility_model.feature_columns),
        "features": list(eligibility_model.feature_columns),
        "support": {
            "fit": fit_support,
            "calibration": cal_support,
            "fresh_states_rows": int(len(fresh_states)),
            "fit_onset_rows": int(len(fit_onset)),
            "calibration_onset_rows": int(len(cal_onset)),
            "fresh_onset_rows": int(len(fresh_onset)),
        },
        "calibration_threshold_selection": threshold_audit,
        "calibration_geometry_audit": calibration_geometry_audit,
        "calibration_gated": cal_gated,
        "fresh": {
            "gated_request223": fresh_gated,
            "ungated_request223": fresh_ungated,
            "gated_minus_ungated": difference,
        },
        "frozen_gate": {
            "min_fresh_eligible_rate": MIN_FRESH_ELIGIBLE_RATE,
            "max_fresh_eligible_rate": MAX_FRESH_ELIGIBLE_RATE,
            "min_fresh_eligibility_spearman": (
                MIN_FRESH_ELIGIBILITY_SPEARMAN
            ),
            "eligible_terminal_advantage_positive": True,
            "gated_day_balanced_uplift_positive": True,
            "gated_bootstrap_ci_low_positive": True,
            "min_good_days": MIN_GOOD_DAYS,
            "gated_must_beat_ungated_request223": True,
            "matched_difference_bootstrap_ci_low_positive": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze pullback eligibility and adaptive management, then "
            "build the reacceleration continuation/exit handoff and replay "
            "full Request208-entry trajectories."
        ),
        "next_boundary_if_fail": (
            "the remaining pullback edge is not separable by onset "
            "geometry alone; move the gate one observation into the "
            "pullback and test a causal early-shape eligibility decision."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fresh_gated_rows.to_parquet(
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
    parser.add_argument("--history-scan", type=Path, required=True)
    parser.add_argument("--fresh-states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.history_scan,
        args.fresh_states,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
