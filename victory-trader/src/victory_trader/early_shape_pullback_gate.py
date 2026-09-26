"""Request 225: one-observation early-shape pullback eligibility gate.

Request224 showed that pullback-onset geometry does not transfer. Request225
pays for exactly one additional observed event before deciding whether the
pullback deserves continued management.

Policy semantics:
* pullback onset -> force one observed-event probe;
* if reacceleration/cap is reached on that event, hand off/resolve naturally;
* otherwise, at the first post-onset observed state, use only causal early-shape
  information to decide eligibility;
* ineligible -> EXIT at that first post-onset state;
* eligible -> hand the remaining segment to the frozen Request223 adaptive
  controller.

The first probe event is not free: all local P&L is measured from the original
pullback-onset EXIT value.
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
    predict_value,
    train_value_model,
)
from .entry_conditioned_rich_second import attach_entry_conditioned_features
from .transition_conditioned_option_value import (
    _prepare_historical,
    attach_transition_phase,
)

REQUEST_ID = 225
MODEL_SEED = 20261225
BOOTSTRAP_SEED = 20261225

THRESHOLD_QUANTILES = (0.40, 0.50, 0.60, 0.70, 0.80)
MIN_CAL_ELIGIBLE_RATE = 0.15
MAX_CAL_ELIGIBLE_RATE = 0.60

MIN_FRESH_ELIGIBLE_RATE = 0.10
MAX_FRESH_ELIGIBLE_RATE = 0.70
MIN_FRESH_SPEARMAN = 0.03
MIN_GOOD_DAYS = 3

EARLY_SHAPE_SOURCES = (
    "entry_to_current_close_pct",
    "drawdown_from_peak_pct",
    "recovery_from_trough_pct",
    "attention_score",
    "attention_rank",
    "log_minute_volume",
    "log_minute_transactions",
    "active_seconds_60",
    "sec_last5_return_pct",
    "sec_last15_return_pct",
    "sec_close_vs_vwap_pct",
    "sec_volume_burst_5",
    "sec_transactions_burst_5",
    "sec_signed_volume_imbalance_60",
    "sec_signed_transactions_imbalance_60",
    "sec_return_efficiency_60",
    "sec_sign_flip_rate_60",
    "sec_close_location_60",
)

EARLY_SHAPE_FEATURES = tuple(
    dict.fromkeys(
        [
            *REQUEST223_FEATURES,
            "early_shape_slope_pct_per_min",
            "early_shape_recovery_fraction",
            *[
                f"onset_{source}"
                for source in EARLY_SHAPE_SOURCES
            ],
            *[
                f"early_delta_{source}"
                for source in EARLY_SHAPE_SOURCES
            ],
        ]
    )
)


@dataclass(frozen=True)
class EarlyShapeModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    winsor_low: float
    winsor_high: float


def build_decision_rows(states: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for segment_id, group in states.groupby(
        "pullback_segment_id", sort=False
    ):
        ordered = group.sort_values("pullback_age_events", kind="stable")
        onset = ordered.loc[
            pd.to_numeric(
                ordered["pullback_age_events"], errors="coerce"
            ).eq(0)
        ]
        decision = ordered.loc[
            pd.to_numeric(
                ordered["pullback_age_events"], errors="coerce"
            ).eq(1)
        ]
        if onset.empty or decision.empty:
            continue

        onset_row = onset.iloc[0]
        row = decision.iloc[0].to_dict()
        row["pullback_segment_id"] = str(segment_id)

        elapsed = pd.to_numeric(
            pd.Series([row.get("pullback_elapsed_min")]),
            errors="coerce",
        ).iloc[0]
        ret = pd.to_numeric(
            pd.Series([row.get("pullback_return_from_onset_pct")]),
            errors="coerce",
        ).iloc[0]
        worst = pd.to_numeric(
            pd.Series([row.get("pullback_worst_from_onset_pct")]),
            errors="coerce",
        ).iloc[0]
        recovery = pd.to_numeric(
            pd.Series([row.get("pullback_recovery_from_worst_pct")]),
            errors="coerce",
        ).iloc[0]

        row["early_shape_slope_pct_per_min"] = (
            float(ret) / max(float(elapsed), 1e-6)
            if pd.notna(ret) and pd.notna(elapsed)
            else np.nan
        )
        row["early_shape_recovery_fraction"] = (
            float(recovery) / max(abs(float(worst)), 0.05)
            if pd.notna(recovery) and pd.notna(worst)
            else np.nan
        )

        for source in EARLY_SHAPE_SOURCES:
            onset_value = pd.to_numeric(
                pd.Series([onset_row.get(source)]),
                errors="coerce",
            ).iloc[0]
            current_value = pd.to_numeric(
                pd.Series([row.get(source)]),
                errors="coerce",
            ).iloc[0]
            row[f"onset_{source}"] = (
                float(onset_value) if pd.notna(onset_value) else np.nan
            )
            row[f"early_delta_{source}"] = (
                float(current_value - onset_value)
                if pd.notna(current_value) and pd.notna(onset_value)
                else np.nan
            )

        row["early_continue_value_pct"] = pd.to_numeric(
            pd.Series([row.get("adaptive_wait_value_pct")]),
            errors="coerce",
        ).iloc[0]
        records.append(row)

    return pd.DataFrame(records)


def _usable_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in EARLY_SHAPE_FEATURES
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


def train_early_shape_model(
    fit_decisions: pd.DataFrame,
) -> EarlyShapeModel:
    target = pd.to_numeric(
        fit_decisions["early_continue_value_pct"], errors="coerce"
    )
    valid = target.notna()
    train = fit_decisions.loc[valid].copy()
    y = target.loc[valid].to_numpy(dtype=float)
    if len(train) < 700:
        raise ValueError(
            f"request 225 insufficient fit decision rows: {len(train)}"
        )

    columns = _usable_columns(train)
    low, high = np.quantile(y, [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=MODEL_SEED,
    )
    model.fit(
        _feature_frame(train, columns),
        np.clip(y, low, high),
        sample_weight=_day_weights(train),
    )
    return EarlyShapeModel(
        model=model,
        feature_columns=columns,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_early_shape(
    frame: pd.DataFrame,
    fitted: EarlyShapeModel,
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


def simulate_ungated_request223(
    states: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for segment_id, group in states.groupby(
        "pullback_segment_id", sort=False
    ):
        work = group.sort_values("pullback_age_events", kind="stable")
        first = work.iloc[0]
        onset_exit = float(first["pullback_onset_exit_pct"])
        terminal_exit = float(first["pullback_terminal_exit_pct"])
        exit_return = terminal_exit
        reason = str(first["pullback_terminal_reason"])

        for _, row in work.iterrows():
            score = pd.to_numeric(
                pd.Series([row.get(score_column)]),
                errors="coerce",
            ).iloc[0]
            current_exit = pd.to_numeric(
                pd.Series([row.get("exit_now_base_return_pct")]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(score) or pd.isna(current_exit):
                exit_return = (
                    float(current_exit)
                    if pd.notna(current_exit)
                    else onset_exit
                )
                reason = "unscoreable_exit"
                break
            if float(score) <= 0:
                exit_return = float(current_exit)
                reason = "model_exit"
                break

        records.append(
            {
                "pullback_segment_id": str(segment_id),
                "trading_day": str(first["trading_day"]),
                "ticker": str(first["ticker"]).upper(),
                "hot_t": int(first["hot_t"]),
                "local_uplift_pct": float(exit_return - onset_exit),
                "exit_reason": reason,
            }
        )
    return pd.DataFrame(records)


def simulate_probe_gated_policy(
    states: pd.DataFrame,
    decisions: pd.DataFrame,
    threshold: float,
    score_column: str,
) -> pd.DataFrame:
    decision_map = decisions.set_index("pullback_segment_id")
    records: list[dict[str, object]] = []

    for segment_id, group in states.groupby(
        "pullback_segment_id", sort=False
    ):
        work = group.sort_values("pullback_age_events", kind="stable")
        first = work.iloc[0]
        onset_exit = float(first["pullback_onset_exit_pct"])
        terminal_exit = float(first["pullback_terminal_exit_pct"])

        post = work.loc[
            pd.to_numeric(
                work["pullback_age_events"], errors="coerce"
            ).ge(1)
        ].copy()

        if post.empty:
            # The very next observed event is already the terminal handoff.
            records.append(
                {
                    "pullback_segment_id": str(segment_id),
                    "trading_day": str(first["trading_day"]),
                    "ticker": str(first["ticker"]).upper(),
                    "hot_t": int(first["hot_t"]),
                    "eligible": None,
                    "eligibility_score": np.nan,
                    "decision_available": False,
                    "local_uplift_pct": float(
                        terminal_exit - onset_exit
                    ),
                    "exit_reason": "probe_reached_terminal",
                }
            )
            continue

        if str(segment_id) not in decision_map.index:
            raise ValueError(
                "request 225 missing first-post-onset decision row"
            )
        decision = decision_map.loc[str(segment_id)]
        eligibility_score = float(decision["eligibility_score"])
        eligible = eligibility_score >= float(threshold)

        first_post = post.iloc[0]
        first_post_exit = pd.to_numeric(
            pd.Series([first_post.get("exit_now_base_return_pct")]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(first_post_exit):
            exit_return = onset_exit
            reason = "probe_unscoreable_exit"
        elif not eligible:
            exit_return = float(first_post_exit)
            reason = "early_shape_reject"
        else:
            exit_return = terminal_exit
            reason = str(first["pullback_terminal_reason"])
            for _, row in post.iterrows():
                score = pd.to_numeric(
                    pd.Series([row.get(score_column)]),
                    errors="coerce",
                ).iloc[0]
                current_exit = pd.to_numeric(
                    pd.Series([row.get("exit_now_base_return_pct")]),
                    errors="coerce",
                ).iloc[0]
                if pd.isna(score) or pd.isna(current_exit):
                    exit_return = (
                        float(current_exit)
                        if pd.notna(current_exit)
                        else float(first_post_exit)
                    )
                    reason = "unscoreable_exit"
                    break
                if float(score) <= 0:
                    exit_return = float(current_exit)
                    reason = "request223_model_exit"
                    break

        records.append(
            {
                "pullback_segment_id": str(segment_id),
                "trading_day": str(first["trading_day"]),
                "ticker": str(first["ticker"]).upper(),
                "hot_t": int(first["hot_t"]),
                "eligible": bool(eligible),
                "eligibility_score": eligibility_score,
                "decision_available": True,
                "local_uplift_pct": float(exit_return - onset_exit),
                "exit_reason": reason,
            }
        )
    return pd.DataFrame(records)


def policy_metrics(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {"segments": 0}
    values = pd.to_numeric(rows["local_uplift_pct"], errors="coerce")
    work = rows.loc[values.notna()].copy()
    work["_value"] = values.loc[values.notna()]
    daily = work.groupby(
        work["trading_day"].astype(str), sort=True
    )["_value"].mean()
    decision = work.loc[work["decision_available"].astype(bool)]
    eligible = (
        decision["eligible"].astype(bool)
        if len(decision)
        else pd.Series(dtype=bool)
    )
    return {
        "segments": int(len(work)),
        "decision_segments": int(len(decision)),
        "probe_terminal_segments": int(
            (~work["decision_available"].astype(bool)).sum()
        ),
        "eligible_segments": int(eligible.sum()) if len(decision) else 0,
        "eligible_rate_among_decisions": (
            float(eligible.mean()) if len(decision) else None
        ),
        "mean_uplift_pct": float(work["_value"].mean()),
        "median_uplift_pct": float(work["_value"].median()),
        "p05_uplift_pct": float(work["_value"].quantile(0.05)),
        "positive_rate": float(work["_value"].gt(0).mean()),
        "day_balanced_uplift_pct": float(daily.mean()),
        "day_bootstrap": _bootstrap_daily(
            daily.to_numpy(dtype=float), BOOTSTRAP_SEED
        ),
        "good_days": int((daily > 0).sum()),
        "by_day": {
            str(day): {
                "segments": int(len(part)),
                "mean_uplift_pct": float(part["_value"].mean()),
            }
            for day, part in work.groupby(
                work["trading_day"].astype(str), sort=True
            )
        },
    }


def matched_difference(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
) -> dict[str, object]:
    left = candidate.loc[
        :, ["pullback_segment_id", "trading_day", "local_uplift_pct"]
    ]
    right = comparator.loc[
        :, ["pullback_segment_id", "local_uplift_pct"]
    ]
    merged = left.merge(
        right,
        on="pullback_segment_id",
        suffixes=("_candidate", "_comparator"),
        validate="one_to_one",
    )
    merged["difference_pct"] = (
        pd.to_numeric(
            merged["local_uplift_pct_candidate"], errors="coerce"
        )
        - pd.to_numeric(
            merged["local_uplift_pct_comparator"], errors="coerce"
        )
    )
    merged = merged.loc[merged["difference_pct"].notna()].copy()
    daily = merged.groupby(
        merged["trading_day"].astype(str), sort=True
    )["difference_pct"].mean()
    return {
        "matched_segments": int(len(merged)),
        "mean_difference_pct": float(merged["difference_pct"].mean()),
        "candidate_better_rate": float(
            merged["difference_pct"].gt(0).mean()
        ),
        "day_balanced_difference_pct": float(daily.mean()),
        "bootstrap": _bootstrap_daily(
            daily.to_numpy(dtype=float), BOOTSTRAP_SEED + 1
        ),
    }


def gate_signal_metrics(decisions: pd.DataFrame) -> dict[str, object]:
    actual = pd.to_numeric(
        decisions["early_continue_value_pct"], errors="coerce"
    )
    score = pd.to_numeric(
        decisions["eligibility_score"], errors="coerce"
    )
    return {
        "rows": int(len(decisions)),
        "spearman": _safe_spearman(actual, score),
        "actual_mean_pct": float(actual.mean()),
    }


def select_threshold(
    cal_states: pd.DataFrame,
    cal_decisions: pd.DataFrame,
) -> tuple[float, dict[str, object]]:
    scores = pd.to_numeric(
        cal_decisions["eligibility_score"], errors="coerce"
    ).dropna()
    candidates: list[dict[str, object]] = []

    for quantile in THRESHOLD_QUANTILES:
        threshold = float(scores.quantile(quantile))
        rows = simulate_probe_gated_policy(
            cal_states,
            cal_decisions,
            threshold,
            "entry_conditioned_score",
        )
        metrics = policy_metrics(rows)
        rate = metrics.get("eligible_rate_among_decisions")
        valid_rate = bool(
            rate is not None
            and MIN_CAL_ELIGIBLE_RATE
            <= float(rate)
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

    valid = [x for x in candidates if x["valid_eligible_rate"]]
    pool = valid if valid else candidates
    chosen = max(
        pool,
        key=lambda x: (
            float(
                x["metrics"].get("day_balanced_uplift_pct")
                if x["metrics"].get("day_balanced_uplift_pct")
                is not None
                else -np.inf
            ),
            float(x["quantile"]),
        ),
    )
    return float(chosen["threshold"]), {
        "candidates": candidates,
        "selected_quantile": chosen["quantile"],
        "selected_threshold": chosen["threshold"],
        "selected_metrics": chosen["metrics"],
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

    fit_full, cal_full = _prepare_historical(
        fit_positions,
        calibration_positions,
        history_scan,
    )
    fit_full = attach_entry_conditioned_features(
        attach_transition_phase(fit_full)
    )
    cal_full = attach_entry_conditioned_features(
        attach_transition_phase(cal_full)
    )
    fit_states, fit_support = build_pullback_wait_states(fit_full)
    cal_states, cal_support = build_pullback_wait_states(cal_full)

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
            "request 225 requires Request223 scored fresh states"
        )

    fit_decisions = build_decision_rows(fit_states)
    cal_decisions = build_decision_rows(cal_states)
    fresh_decisions = build_decision_rows(fresh_states)

    model = train_early_shape_model(fit_decisions)
    cal_decisions["eligibility_score"] = predict_early_shape(
        cal_decisions, model
    )
    fresh_decisions["eligibility_score"] = predict_early_shape(
        fresh_decisions, model
    )

    threshold, threshold_audit = select_threshold(
        cal_states, cal_decisions
    )

    fresh_candidate_rows = simulate_probe_gated_policy(
        fresh_states,
        fresh_decisions,
        threshold,
        "entry_conditioned_score",
    )
    fresh_ungated_rows = simulate_ungated_request223(
        fresh_states,
        "entry_conditioned_score",
    )

    fresh_candidate = policy_metrics(fresh_candidate_rows)
    fresh_ungated = policy_metrics(
        fresh_ungated_rows.assign(
            decision_available=True,
            eligible=True,
        )
    )
    difference = matched_difference(
        fresh_candidate_rows, fresh_ungated_rows
    )
    fresh_signal = gate_signal_metrics(fresh_decisions)

    eligible_rate = fresh_candidate.get(
        "eligible_rate_among_decisions"
    )
    spearman = fresh_signal.get("spearman")
    day_uplift = fresh_candidate.get("day_balanced_uplift_pct")
    day_low = fresh_candidate.get("day_bootstrap", {}).get(
        "ci_low_pct"
    )
    diff_day = difference.get("day_balanced_difference_pct")
    diff_low = difference.get("bootstrap", {}).get("ci_low_pct")

    gate = bool(
        eligible_rate is not None
        and MIN_FRESH_ELIGIBLE_RATE
        <= float(eligible_rate)
        <= MAX_FRESH_ELIGIBLE_RATE
        and spearman is not None
        and float(spearman) >= MIN_FRESH_SPEARMAN
        and day_uplift is not None
        and float(day_uplift) > 0
        and day_low is not None
        and float(day_low) > 0
        and int(fresh_candidate.get("good_days", 0))
        >= MIN_GOOD_DAYS
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "policy": {
            "probe": "always observe exactly one event after pullback onset",
            "immediate_terminal": (
                "if that event is reacceleration/cap, handoff/resolve "
                "without an eligibility decision"
            ),
            "ineligible_after_probe": "EXIT at first post-onset state",
            "eligible_after_probe": (
                "hand remaining pullback to frozen Request223 adaptive "
                "entry-conditioned controller"
            ),
            "probe_cost_included_from_original_onset": True,
            "future_information_used_as_input": False,
        },
        "eligibility_target": (
            "terminal continuation value versus EXIT at first post-onset "
            "decision state"
        ),
        "feature_count": len(model.feature_columns),
        "support": {
            "fit": fit_support,
            "calibration": cal_support,
            "fit_decision_rows": int(len(fit_decisions)),
            "calibration_decision_rows": int(len(cal_decisions)),
            "fresh_state_rows": int(len(fresh_states)),
            "fresh_decision_rows": int(len(fresh_decisions)),
        },
        "calibration_threshold_selection": threshold_audit,
        "fresh": {
            "gate_signal": fresh_signal,
            "probe_gated_request223": fresh_candidate,
            "ungated_request223": fresh_ungated,
            "probe_gated_minus_ungated": difference,
        },
        "frozen_gate": {
            "min_fresh_eligible_rate": MIN_FRESH_ELIGIBLE_RATE,
            "max_fresh_eligible_rate": MAX_FRESH_ELIGIBLE_RATE,
            "min_fresh_spearman": MIN_FRESH_SPEARMAN,
            "policy_day_balanced_uplift_positive": True,
            "policy_bootstrap_ci_low_positive": True,
            "min_good_days": MIN_GOOD_DAYS,
            "must_beat_ungated_request223": True,
            "matched_difference_bootstrap_ci_low_positive": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze the one-event probe eligibility node, build the "
            "reacceleration continuation/exit handoff, then replay full "
            "Request208-entry trajectories."
        ),
        "next_boundary_if_fail": (
            "the pullback branch is not yet economically predictive; "
            "stop local gate refinements and move upstream to improve "
            "entry timing so fewer weak positions reach pullback management."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fresh_candidate_rows.to_parquet(
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
