"""Request 221: transition-conditioned remaining-option ranking.

Request220 showed that entry-conditioned microstructure helped on some days and
hurt on others. Request221 tests whether this is because the same observable
change has different meanings in different causal transition contexts.

No new dates and no policy replay. The Request219 five-event remaining-option
target is frozen. The Request220 entry-conditioned information set is frozen.
Only the learning structure changes:

* classify each observed POSITION state by the sign pattern of the current and
  previous observed return change;
* train one global Request220 model;
* train phase specialists on fit data only;
* choose, phase by phase, whether to use the specialist or the global model
  using calibration data only;
* evaluate the frozen mixture once on the already-open Request178 block.

This is a representation/value diagnostic, not a deployable HOLD/EXIT rule.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from .entry_conditioned_rich_second import (
    ENTRY_CONDITIONED_FEATURES,
    attach_entry_conditioned_features,
)
from .event_time_position_replay import (
    _prepare_entries,
    reanchor_event_positions,
)
from .market_regime_position_value import (
    attach_market_regime,
    build_market_regime,
)
from .learned_sequence_position_ranking import (
    _fit_scaler,
    _transform,
)
from .multi_event_option_value import (
    attach_option_rank_target,
    attach_option_value_target,
    option_diagnostics,
)
from .transition_recurrent_hold_exit import attach_position_transitions

REQUEST_ID = 221
MODEL_SEED = 20261129

PHASES = (
    "reacceleration",
    "pullback_onset",
    "continuing_strength",
    "continuing_weakness",
    "flat_or_sparse",
)

MIN_PHASE_FIT_ROWS = 400
MIN_PHASE_CAL_ROWS = 150
MIN_CAL_PHASE_SPEARMAN_GAIN = 0.015
MIN_SPECIALIST_PHASES = 1

MIN_FRESH_SPEARMAN = 0.05
MIN_FRESH_GAIN = 0.02
MIN_GOOD_DAYS = 4
MIN_TOP10_LIFT_PCT = 0.45
MIN_TOP10_POSITIVE_RATE_LIFT = 0.03
MIN_SAME_AGE_MEDIAN_SPEARMAN = 0.03
MIN_SELECTED_PHASE_POSITIVE_TRANSFER_FRACTION = 0.50


@dataclass(frozen=True)
class PhaseRankModel:
    model: MLPRegressor
    feature_columns: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray


def attach_transition_phase(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a deterministic causal phase from current/previous observed moves."""
    result = frame.copy()
    current = pd.to_numeric(
        result.get("position_return_change_1m_pct"),
        errors="coerce",
    )
    accel = pd.to_numeric(
        result.get("position_return_accel_1m_pct"),
        errors="coerce",
    )
    previous = current - accel

    phase = pd.Series("flat_or_sparse", index=result.index, dtype="object")
    valid = current.notna() & previous.notna()

    phase.loc[valid & current.gt(0) & previous.le(0)] = "reacceleration"
    phase.loc[valid & current.lt(0) & previous.ge(0)] = "pullback_onset"
    phase.loc[valid & current.gt(0) & previous.gt(0)] = "continuing_strength"
    phase.loc[valid & current.lt(0) & previous.lt(0)] = "continuing_weakness"

    result["transition_phase"] = phase
    result["previous_observed_return_change_pct"] = previous
    return result


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


def train_phase_rank_model(
    fit: pd.DataFrame,
    *,
    min_rows: int,
) -> PhaseRankModel:
    target = pd.to_numeric(
        fit["option_value_day_rank"], errors="coerce"
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].to_numpy(dtype=float)
    if len(train) < min_rows:
        raise ValueError(
            f"request 221 insufficient phase support: {len(train)} < {min_rows}"
        )

    columns = _usable_columns(train, ENTRY_CONDITIONED_FEATURES)
    if not columns:
        raise ValueError("request 221 has no usable phase features")

    center, scale = _fit_scaler(train, columns)
    x = _transform(train, columns, center, scale)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=250,
        shuffle=True,
        random_state=MODEL_SEED,
        tol=1e-4,
        n_iter_no_change=20,
        early_stopping=False,
    )
    model.fit(x, y)
    return PhaseRankModel(
        model=model,
        feature_columns=columns,
        center=center,
        scale=scale,
    )


def predict_phase_rank(
    frame: pd.DataFrame,
    fitted: PhaseRankModel,
) -> np.ndarray:
    x = _transform(
        frame,
        fitted.feature_columns,
        fitted.center,
        fitted.scale,
    )
    return fitted.model.predict(x)


def _phase_diagnostics(
    frame: pd.DataFrame,
    score_column: str,
) -> dict[str, object]:
    actual = pd.to_numeric(
        frame["option_value_5event_pct"], errors="coerce"
    )
    score = pd.to_numeric(frame[score_column], errors="coerce")
    valid = actual.notna() & score.notna()
    work = frame.loc[valid].copy()
    work["_actual"] = actual.loc[valid]
    work["_score"] = score.loc[valid]

    if len(work) < 20:
        return {
            "rows": int(len(work)),
            "spearman": None,
            "top20_mean_lift_pct": None,
            "top20_positive_rate_lift": None,
        }

    corr = work["_actual"].corr(work["_score"], method="spearman")
    corr_value = None if pd.isna(corr) else float(corr)

    work = work.sort_values(
        "_score", ascending=False, kind="stable"
    )
    count = max(1, int(np.ceil(len(work) * 0.20)))
    top = work.head(count)
    overall_mean = float(work["_actual"].mean())
    top_mean = float(top["_actual"].mean())
    overall_positive = float(work["_actual"].gt(0).mean())
    top_positive = float(top["_actual"].gt(0).mean())

    return {
        "rows": int(len(work)),
        "spearman": corr_value,
        "top20_mean_lift_pct": float(top_mean - overall_mean),
        "top20_positive_rate_lift": float(
            top_positive - overall_positive
        ),
    }


def _prepare_historical(
    fit_positions: pd.DataFrame,
    calibration_positions: pd.DataFrame,
    history_scan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_entries, calibration_entries, _ = _prepare_entries(
        fit_positions,
        calibration_positions,
        calibration_positions,
    )
    fit = reanchor_event_positions(fit_positions, fit_entries)
    calibration = reanchor_event_positions(
        calibration_positions,
        calibration_entries,
    )

    historical_days = set(
        fit_positions["trading_day"].astype(str)
    ) | set(calibration_positions["trading_day"].astype(str))
    scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(historical_days)
    ].copy()
    regime = build_market_regime(scan)

    fit = attach_position_transitions(
        attach_market_regime(fit, regime)
    )
    calibration = attach_position_transitions(
        attach_market_regime(calibration, regime)
    )
    return fit, calibration


def _score_specialist_mix(
    frame: pd.DataFrame,
    global_model: PhaseRankModel,
    specialists: dict[str, PhaseRankModel],
    selected_phases: set[str],
) -> pd.DataFrame:
    result = frame.copy()
    result["global_entry_conditioned_score"] = predict_phase_rank(
        result, global_model
    )
    result["phase_conditioned_score"] = result[
        "global_entry_conditioned_score"
    ].astype(float)

    for phase in sorted(selected_phases):
        model = specialists.get(phase)
        if model is None:
            continue
        mask = result["transition_phase"].astype(str).eq(phase)
        if not mask.any():
            continue
        result.loc[mask, "phase_conditioned_score"] = predict_phase_rank(
            result.loc[mask], model
        )
    return result


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    history_scan_path: Path,
    fresh_states_path: Path,
    output_path: Path,
    states_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )
    history_scan = pd.read_parquet(history_scan_path)
    fresh = pd.read_parquet(fresh_states_path).copy()

    fit, calibration = _prepare_historical(
        fit_positions,
        calibration_positions,
        history_scan,
    )

    fit = attach_entry_conditioned_features(
        attach_transition_phase(
            attach_option_rank_target(
                attach_option_value_target(fit)
            )
        )
    )
    calibration = attach_entry_conditioned_features(
        attach_transition_phase(
            attach_option_rank_target(
                attach_option_value_target(calibration)
            )
        )
    )

    # Request220 fresh states already contain the frozen target and
    # entry-conditioned rich-second features. Only add the phase label.
    fresh = attach_transition_phase(fresh)

    global_model = train_phase_rank_model(
        fit,
        min_rows=1000,
    )

    calibration["global_entry_conditioned_score"] = (
        predict_phase_rank(calibration, global_model)
    )
    fresh["global_entry_conditioned_score"] = predict_phase_rank(
        fresh, global_model
    )

    specialists: dict[str, PhaseRankModel] = {}
    calibration_phase_results: dict[str, object] = {}
    selected_phases: set[str] = set()

    for phase in PHASES:
        fit_phase = fit.loc[
            fit["transition_phase"].astype(str).eq(phase)
        ].copy()
        cal_phase = calibration.loc[
            calibration["transition_phase"].astype(str).eq(phase)
        ].copy()

        fit_target_rows = int(
            pd.to_numeric(
                fit_phase["option_value_day_rank"],
                errors="coerce",
            ).notna().sum()
        )
        cal_target_rows = int(
            pd.to_numeric(
                cal_phase["option_value_day_rank"],
                errors="coerce",
            ).notna().sum()
        )

        global_diag = _phase_diagnostics(
            cal_phase,
            "global_entry_conditioned_score",
        )
        record: dict[str, object] = {
            "fit_target_rows": fit_target_rows,
            "calibration_target_rows": cal_target_rows,
            "global": global_diag,
            "specialist_trained": False,
            "specialist_selected": False,
        }

        if (
            fit_target_rows >= MIN_PHASE_FIT_ROWS
            and cal_target_rows >= MIN_PHASE_CAL_ROWS
        ):
            model = train_phase_rank_model(
                fit_phase,
                min_rows=MIN_PHASE_FIT_ROWS,
            )
            specialists[phase] = model
            cal_phase["specialist_score"] = predict_phase_rank(
                cal_phase, model
            )
            specialist_diag = _phase_diagnostics(
                cal_phase,
                "specialist_score",
            )
            global_s = global_diag.get("spearman")
            specialist_s = specialist_diag.get("spearman")
            gain = (
                float(specialist_s) - float(global_s)
                if specialist_s is not None and global_s is not None
                else None
            )
            specialist_lift = specialist_diag.get(
                "top20_mean_lift_pct"
            )
            global_lift = global_diag.get("top20_mean_lift_pct")
            selected = bool(
                gain is not None
                and gain >= MIN_CAL_PHASE_SPEARMAN_GAIN
                and specialist_lift is not None
                and global_lift is not None
                and float(specialist_lift) >= float(global_lift)
            )
            if selected:
                selected_phases.add(phase)
            record.update(
                {
                    "specialist_trained": True,
                    "specialist_selected": selected,
                    "specialist": specialist_diag,
                    "specialist_minus_global_spearman": gain,
                }
            )

        calibration_phase_results[phase] = record

    calibration_scored = _score_specialist_mix(
        calibration,
        global_model,
        specialists,
        selected_phases,
    )
    fresh_scored = _score_specialist_mix(
        fresh,
        global_model,
        specialists,
        selected_phases,
    )

    calibration_global_diag = option_diagnostics(
        calibration_scored,
        "global_entry_conditioned_score",
    )
    calibration_candidate_diag = option_diagnostics(
        calibration_scored,
        "phase_conditioned_score",
    )
    fresh_global_diag = option_diagnostics(
        fresh_scored,
        "global_entry_conditioned_score",
    )
    fresh_candidate_diag = option_diagnostics(
        fresh_scored,
        "phase_conditioned_score",
    )

    fresh_phase_results: dict[str, object] = {}
    positive_transfer = 0
    selected_with_eval = 0
    for phase in PHASES:
        part = fresh_scored.loc[
            fresh_scored["transition_phase"].astype(str).eq(phase)
        ].copy()
        global_diag = _phase_diagnostics(
            part,
            "global_entry_conditioned_score",
        )
        candidate_diag = _phase_diagnostics(
            part,
            "phase_conditioned_score",
        )
        global_s = global_diag.get("spearman")
        candidate_s = candidate_diag.get("spearman")
        gain = (
            float(candidate_s) - float(global_s)
            if candidate_s is not None and global_s is not None
            else None
        )
        selected = phase in selected_phases
        if selected and gain is not None:
            selected_with_eval += 1
            positive_transfer += int(gain > 0)
        fresh_phase_results[phase] = {
            "rows": int(len(part)),
            "selected_specialist": selected,
            "global": global_diag,
            "candidate": candidate_diag,
            "candidate_minus_global_spearman": gain,
        }

    global_s = fresh_global_diag.get("spearman")
    candidate_s = fresh_candidate_diag.get("spearman")
    fresh_gain = (
        float(candidate_s) - float(global_s)
        if candidate_s is not None and global_s is not None
        else None
    )
    top10 = fresh_candidate_diag["top10_by_day"]
    positive_transfer_fraction = (
        float(positive_transfer / selected_with_eval)
        if selected_with_eval
        else None
    )

    gate = bool(
        len(selected_phases) >= MIN_SPECIALIST_PHASES
        and candidate_s is not None
        and float(candidate_s) >= MIN_FRESH_SPEARMAN
        and fresh_gain is not None
        and float(fresh_gain) >= MIN_FRESH_GAIN
        and int(fresh_candidate_diag["good_days"]) >= MIN_GOOD_DAYS
        and top10.get("day_balanced_mean_lift_pct") is not None
        and float(top10["day_balanced_mean_lift_pct"])
        >= MIN_TOP10_LIFT_PCT
        and top10.get("day_balanced_positive_rate_lift")
        is not None
        and float(top10["day_balanced_positive_rate_lift"])
        >= MIN_TOP10_POSITIVE_RATE_LIFT
        and fresh_candidate_diag.get("same_age_median_spearman")
        is not None
        and float(
            fresh_candidate_diag["same_age_median_spearman"]
        ) >= MIN_SAME_AGE_MEDIAN_SPEARMAN
        and positive_transfer_fraction is not None
        and positive_transfer_fraction
        >= MIN_SELECTED_PHASE_POSITIVE_TRANSFER_FRACTION
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "target_changed_from_request219": False,
        "information_set_changed_from_request220": False,
        "hypothesis": (
            "Request220 microstructure signals transfer poorly because "
            "their meaning differs by causal transition context; "
            "phase-specific specialists should recover that information."
        ),
        "phase_definition": {
            "reacceleration": "current move > 0, previous move <= 0",
            "pullback_onset": "current move < 0, previous move >= 0",
            "continuing_strength": "current move > 0, previous move > 0",
            "continuing_weakness": "current move < 0, previous move < 0",
            "flat_or_sparse": "otherwise or insufficient causal history",
            "future_information_used": False,
        },
        "model": {
            "global_family": "MLPRegressor",
            "specialist_family": "MLPRegressor",
            "hidden_layers": [64, 32],
            "seed": MODEL_SEED,
            "feature_family": "Request220 entry-conditioned state",
        },
        "support": {
            "fit_rows": int(len(fit)),
            "calibration_rows": int(len(calibration)),
            "fresh_rows": int(len(fresh)),
        },
        "calibration_phase_selection": calibration_phase_results,
        "selected_specialist_phases": sorted(selected_phases),
        "calibration_global": calibration_global_diag,
        "calibration_candidate": calibration_candidate_diag,
        "fresh_global": fresh_global_diag,
        "fresh_candidate": fresh_candidate_diag,
        "fresh_candidate_minus_global_spearman": fresh_gain,
        "fresh_phase_transfer": fresh_phase_results,
        "selected_phase_positive_transfer_fraction": (
            positive_transfer_fraction
        ),
        "frozen_gate": {
            "min_calibration_phase_spearman_gain": (
                MIN_CAL_PHASE_SPEARMAN_GAIN
            ),
            "min_selected_specialist_phases": MIN_SPECIALIST_PHASES,
            "min_fresh_spearman": MIN_FRESH_SPEARMAN,
            "min_fresh_gain_vs_global": MIN_FRESH_GAIN,
            "min_good_days": MIN_GOOD_DAYS,
            "min_top10_day_balanced_lift_pct": MIN_TOP10_LIFT_PCT,
            "min_top10_positive_rate_lift": (
                MIN_TOP10_POSITIVE_RATE_LIFT
            ),
            "min_same_age_median_spearman": (
                MIN_SAME_AGE_MEDIAN_SPEARMAN
            ),
            "min_selected_phase_positive_transfer_fraction": (
                MIN_SELECTED_PHASE_POSITIVE_TRANSFER_FRACTION
            ),
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze transition conditioning, replace hindsight option "
            "upper bound with a causal continuation-value recursion, "
            "then replay recurrent HOLD/EXIT decisions."
        ),
        "next_boundary_if_fail": (
            "do not further split phases by hand; inspect whether the "
            "predictive pocket is specifically tied to pullback/reversal "
            "events and train an event-triggered controller rather than "
            "scoring every POSITION observation."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    states_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fresh_scored.to_parquet(
        states_output_path,
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
    parser.add_argument("--states-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.history_scan,
        args.fresh_states,
        args.output,
        args.states_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
