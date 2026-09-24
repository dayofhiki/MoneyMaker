"""Request 167: exact path-transition HOLD signal on Request-165 BUY paths.

No new market dates. Reuses Request-166 position rows and changes only the
short-horizon HOLD classifier information set by adding the previously
supported v3.5/v3.7 causal position-path state plus exact
1/2/3/5/8/13-minute deltas. Remaining-option model, labels, model capacity,
Platt calibration and semantic P(HOLD)>0.5 boundary remain frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .selected_hot_position_value_observability import (
    MODEL_FEATURES,
    apply_excess_target,
    evaluate_observability,
    fit_minute_baselines,
    predict_hold,
    predict_option,
    train_hold_model,
    train_option_model,
)

REQUEST_ID = 167
MINUTE_MS = 60_000
PATH_TRANSITION_LAGS = (1, 2, 3, 5, 8, 13)
RANDOM_SEED = 20261043

RICH_PATH_FEATURES = (
    "path_entry_return_pct",
    "path_mfe_pct",
    "path_mae_pct",
    "path_drawdown_from_high_pct",
    "path_rebound_from_low_pct",
    "path_range_pct",
    "path_minutes_since_high",
    "path_minutes_since_low",
    "path_observed_fraction",
    "path_efficiency",
)


class TransitionHoldModel:
    def __init__(
        self,
        model: HistGradientBoostingClassifier,
        feature_columns: tuple[str, ...],
        platt: LogisticRegression,
    ) -> None:
        self.model = model
        self.feature_columns = feature_columns
        self.platt = platt


def _episode_keys() -> list[str]:
    return ["trading_day", "ticker", "hot_t"]


def add_rich_path_state(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result = result.sort_values(
        _episode_keys() + ["state_t"],
        kind="stable",
    ).reset_index(drop=True)

    result["path_entry_return_pct"] = pd.to_numeric(
        result["entry_to_current_close_pct"], errors="coerce"
    )
    result["path_mfe_pct"] = pd.to_numeric(
        result["running_max_return_pct"], errors="coerce"
    )
    result["path_mae_pct"] = pd.to_numeric(
        result["running_min_return_pct"], errors="coerce"
    )
    result["path_drawdown_from_high_pct"] = pd.to_numeric(
        result["drawdown_from_peak_pct"], errors="coerce"
    )
    result["path_rebound_from_low_pct"] = pd.to_numeric(
        result["recovery_from_trough_pct"], errors="coerce"
    )
    result["path_range_pct"] = (
        result["path_mfe_pct"] - result["path_mae_pct"]
    )

    minutes_since_high = pd.Series(np.nan, index=result.index, dtype=float)
    minutes_since_low = pd.Series(np.nan, index=result.index, dtype=float)
    observed_fraction = pd.Series(np.nan, index=result.index, dtype=float)
    efficiency = pd.Series(np.nan, index=result.index, dtype=float)

    for _, group in result.groupby(_episode_keys(), sort=False):
        idx = group.index
        held = pd.to_numeric(group["minutes_held"], errors="coerce")
        mfe = pd.to_numeric(group["path_mfe_pct"], errors="coerce")
        mae = pd.to_numeric(group["path_mae_pct"], errors="coerce")
        ret = pd.to_numeric(group["path_entry_return_pct"], errors="coerce")
        state_t = pd.to_numeric(group["state_t"], errors="coerce")

        prev_mfe = mfe.shift(1)
        prev_mae = mae.shift(1)
        high_event = prev_mfe.isna() | mfe.gt(prev_mfe)
        low_event = prev_mae.isna() | mae.lt(prev_mae)
        last_high_minute = held.where(high_event).ffill()
        last_low_minute = held.where(low_event).ffill()
        minutes_since_high.loc[idx] = (
            held - last_high_minute
        ).to_numpy()
        minutes_since_low.loc[idx] = (
            held - last_low_minute
        ).to_numpy()

        observed_count = np.arange(1, len(group) + 1, dtype=float)
        observed_fraction.loc[idx] = np.divide(
            observed_count,
            held.to_numpy(dtype=float),
            out=np.full(len(group), np.nan),
            where=held.to_numpy(dtype=float) > 0,
        )

        prev_t = state_t.shift(1)
        exact_prev = state_t.sub(prev_t).eq(MINUTE_MS)
        one_minute_move = ret.sub(ret.shift(1)).abs().where(exact_prev, 0.0)
        accumulated = one_minute_move.fillna(0.0).cumsum()
        eff = ret / accumulated.where(accumulated.gt(0))
        efficiency.loc[idx] = eff.to_numpy()

    result["path_minutes_since_high"] = minutes_since_high
    result["path_minutes_since_low"] = minutes_since_low
    result["path_observed_fraction"] = observed_fraction
    result["path_efficiency"] = efficiency
    return result


def add_exact_path_transitions(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_rich_path_state(frame)
    keys = _episode_keys()
    if result.duplicated(keys + ["state_t"]).any():
        raise ValueError(
            "request 167 path rows must be unique by episode/state_t"
        )

    for lag in PATH_TRANSITION_LAGS:
        left = result.loc[:, keys + ["state_t"]].copy()
        left["_row_order"] = np.arange(len(left))
        right = result.loc[
            :, keys + ["state_t"] + list(RICH_PATH_FEATURES)
        ].copy()
        right["state_t"] = (
            pd.to_numeric(right["state_t"], errors="coerce")
            + lag * MINUTE_MS
        )
        right["_exact_lag_present"] = True
        right = right.rename(
            columns={
                column: f"_lag_{column}"
                for column in RICH_PATH_FEATURES
            }
        )
        merged = left.merge(
            right,
            on=keys + ["state_t"],
            how="left",
            sort=False,
            validate="one_to_one",
        ).sort_values("_row_order", kind="stable")
        exact = (
            merged["_exact_lag_present"]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )
        for column in RICH_PATH_FEATURES:
            current = pd.to_numeric(result[column], errors="coerce")
            lagged = pd.to_numeric(
                merged[f"_lag_{column}"], errors="coerce"
            )
            result[
                f"path_transition_{lag}m_{column}"
            ] = (current.to_numpy() - lagged.to_numpy())
            result.loc[
                ~exact,
                f"path_transition_{lag}m_{column}",
            ] = np.nan
    return result


def _transition_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    transition = [
        column
        for column in frame.columns
        if column.startswith("path_transition_")
    ]
    return tuple(
        dict.fromkeys(
            [
                *MODEL_FEATURES,
                *RICH_PATH_FEATURES,
                *transition,
            ]
        )
    )


def _transition_feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def train_transition_hold(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> TransitionHoldModel:
    fit = add_exact_path_transitions(fit)
    calibration = add_exact_path_transitions(calibration)

    fit_target = pd.to_numeric(
        fit["hold_advantage_1m_pct"], errors="coerce"
    )
    cal_target = pd.to_numeric(
        calibration["hold_advantage_1m_pct"], errors="coerce"
    )
    fit = fit.loc[fit_target.notna()].copy()
    calibration = calibration.loc[cal_target.notna()].copy()
    y_fit = pd.to_numeric(
        fit["hold_advantage_1m_pct"], errors="coerce"
    ).gt(0).astype(int)
    y_cal = pd.to_numeric(
        calibration["hold_advantage_1m_pct"], errors="coerce"
    ).gt(0).astype(int)
    if (
        len(fit) < 1000
        or len(calibration) < 500
        or y_fit.nunique() < 2
        or y_cal.nunique() < 2
    ):
        raise ValueError("request 167 has insufficient HOLD support")

    columns = _transition_columns(fit)
    columns = tuple(
        column
        for column in columns
        if pd.to_numeric(
            fit[column], errors="coerce"
        ).notna().any()
    )
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=RANDOM_SEED,
    )
    model.fit(_transition_feature_frame(fit, columns), y_fit)

    raw = np.clip(
        model.predict_proba(
            _transition_feature_frame(calibration, columns)
        )[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw / (1 - raw)).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    platt.fit(logits, y_cal)
    return TransitionHoldModel(model, columns, platt)


def predict_transition_hold(
    frame: pd.DataFrame,
    fitted: TransitionHoldModel,
) -> np.ndarray:
    enriched = add_exact_path_transitions(frame)
    raw = np.clip(
        fitted.model.predict_proba(
            _transition_feature_frame(
                enriched,
                fitted.feature_columns,
            )
        )[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw / (1 - raw)).reshape(-1, 1)
    return fitted.platt.predict_proba(logits)[:, 1]


def lag_coverage(frame: pd.DataFrame) -> dict[str, float]:
    enriched = add_exact_path_transitions(frame)
    result: dict[str, float] = {}
    for lag in PATH_TRANSITION_LAGS:
        cols = [
            f"path_transition_{lag}m_{column}"
            for column in RICH_PATH_FEATURES
        ]
        result[str(lag)] = float(
            enriched[cols].notna().any(axis=1).mean()
        )
    return result


def run(
    fit_path: Path,
    calibration_path: Path,
    evaluation_path: Path,
    evaluation_audit_path: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    evaluation = pd.read_parquet(evaluation_path)

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(calibration, baselines)
    evaluation = apply_excess_target(evaluation, baselines)

    baseline_hold = train_hold_model(fit, calibration)
    transition_hold = train_transition_hold(fit, calibration)
    option_model = train_option_model(fit, calibration)

    baseline_eval = evaluation.copy()
    baseline_eval["hold_probability"] = predict_hold(
        baseline_eval,
        baseline_hold,
    )
    baseline_eval["predicted_excess_option_value_pct"] = predict_option(
        baseline_eval,
        option_model,
    )
    baseline_metrics = evaluate_observability(
        baseline_eval,
        label="request166_baseline",
    )

    transition_eval = evaluation.copy()
    transition_eval["hold_probability"] = predict_transition_hold(
        transition_eval,
        transition_hold,
    )
    transition_eval[
        "predicted_excess_option_value_pct"
    ] = predict_option(
        transition_eval,
        option_model,
    )
    transition_metrics = evaluate_observability(
        transition_eval,
        label="path_transition_hold",
    )

    eval_audit = json.loads(
        evaluation_audit_path.read_text(encoding="utf-8")
    )
    anchor_coverage = eval_audit["anchor_path_coverage"]

    summary = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "only_hold_information_set_changed": True,
        "path_transition_lags_minutes": list(PATH_TRANSITION_LAGS),
        "rich_path_features": list(RICH_PATH_FEATURES),
        "transition_feature_count": len(
            transition_hold.feature_columns
        ),
        "evaluation_lag_coverage": lag_coverage(evaluation),
        "evaluation_anchor_path_coverage": anchor_coverage,
        "request166_baseline": baseline_metrics,
        "transition_hold": transition_metrics,
        "controller_signal_pass": bool(
            transition_metrics["bridge_pass"]
        ),
        "full_position_bridge_pass": bool(
            transition_metrics["bridge_pass"]
            and anchor_coverage is not None
            and float(anchor_coverage) >= 0.90
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    transition_eval.to_parquet(
        scored_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--evaluation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-output", type=Path, required=True)
    args = parser.parse_args()
    return run(
        args.fit,
        args.calibration,
        args.evaluation,
        args.evaluation_audit,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
