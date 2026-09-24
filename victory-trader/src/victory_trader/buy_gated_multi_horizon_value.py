"""Request 169: direct multi-horizon continuation-value diagnostic.

No new dates. Reuses Request-166 BUY-gated POSITION rows. Instead of asking
whether exactly one more minute beats EXIT now, fit separate causal regressors
for the incremental BASE value of exiting exactly 2, 5, or 10 minutes later.

The horizons are prediction lookaheads, not holding commitments. A future
recurrent policy, if justified, must re-score every reached minute.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .selected_hot_position_value_observability import (
    MODEL_FEATURES,
)

REQUEST_ID = 169
MINUTE_MS = 60_000
VALUE_HORIZONS = (2, 5, 10)
BOOTSTRAP_SEED = 20261069
BOOTSTRAP_SAMPLES = 10_000

MIN_CHOSEN_TARGET_COVERAGE = 0.70
MIN_CHOSEN_SPEARMAN = 0.05
MIN_SELECTED_RATE = 0.10
MIN_GOOD_DAYS = 4


@dataclass(frozen=True)
class HorizonValueModel:
    horizon: int
    model: HistGradientBoostingRegressor
    offset: float
    winsor_low: float
    winsor_high: float


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reindex(columns=MODEL_FEATURES).apply(
        pd.to_numeric,
        errors="coerce",
    )


def add_future_advantage_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    keys = ["trading_day", "ticker", "hot_t"]
    if result.duplicated(keys + ["state_t"]).any():
        raise ValueError("request 169 requires unique episode/state rows")

    current_exit = pd.to_numeric(
        result["exit_now_base_return_pct"],
        errors="coerce",
    )
    for horizon in VALUE_HORIZONS:
        future = result.loc[
            :, keys + ["state_t", "exit_now_base_return_pct"]
        ].copy()
        future["state_t"] = (
            pd.to_numeric(future["state_t"], errors="raise").astype("int64")
            - horizon * MINUTE_MS
        )
        future = future.rename(
            columns={
                "exit_now_base_return_pct": (
                    f"_future_exit_{horizon}m"
                )
            }
        )
        merged = result.loc[:, keys + ["state_t"]].merge(
            future,
            on=keys + ["state_t"],
            how="left",
            sort=False,
            validate="one_to_one",
        )
        future_exit = pd.to_numeric(
            merged[f"_future_exit_{horizon}m"],
            errors="coerce",
        )
        result[
            f"hold_advantage_{horizon}m_pct"
        ] = future_exit.to_numpy() - current_exit.to_numpy()
    return result


def train_horizon_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    horizon: int,
) -> HorizonValueModel:
    target_column = f"hold_advantage_{horizon}m_pct"
    target = pd.to_numeric(fit[target_column], errors="coerce")
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError(
            f"request 169 {horizon}m fit support {len(train)} < 1000"
        )

    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=BOOTSTRAP_SEED + horizon,
    )
    model.fit(
        _feature_frame(train),
        y.clip(lower=low, upper=high),
    )

    cal_target = pd.to_numeric(
        calibration[target_column],
        errors="coerce",
    )
    cal_valid = cal_target.notna()
    if int(cal_valid.sum()) < 500:
        raise ValueError(
            f"request 169 {horizon}m calibration support "
            f"{int(cal_valid.sum())} < 500"
        )
    prediction = model.predict(
        _feature_frame(calibration.loc[cal_valid])
    )
    offset = float(
        cal_target.loc[cal_valid].mean() - float(np.mean(prediction))
    )
    return HorizonValueModel(
        horizon=horizon,
        model=model,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_horizon(
    frame: pd.DataFrame,
    fitted: HorizonValueModel,
) -> np.ndarray:
    return fitted.model.predict(_feature_frame(frame)) + fitted.offset


def _safe_spearman(a: pd.Series, b: pd.Series) -> float | None:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    valid = aa.notna() & bb.notna()
    if int(valid.sum()) < 20:
        return None
    value = aa.loc[valid].corr(bb.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _day_balanced(
    frame: pd.DataFrame,
    value_column: str,
) -> float | None:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    tmp = frame.loc[values.notna(), ["trading_day"]].copy()
    tmp["_value"] = values.loc[values.notna()].to_numpy()
    if tmp.empty:
        return None
    return float(
        tmp.groupby(tmp["trading_day"].astype(str), sort=True)["_value"]
        .mean()
        .mean()
    )


def _bootstrap_days(
    frame: pd.DataFrame,
    value_column: str,
) -> dict[str, float | int | None]:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    tmp = frame.loc[values.notna(), ["trading_day"]].copy()
    tmp["_value"] = values.loc[values.notna()].to_numpy()
    day_means = (
        tmp.groupby(tmp["trading_day"].astype(str), sort=True)["_value"]
        .mean()
        .to_numpy(dtype=float)
    )
    if len(day_means) < 2:
        return {
            "days": int(len(day_means)),
            "samples": BOOTSTRAP_SAMPLES,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(
        0,
        len(day_means),
        size=(BOOTSTRAP_SAMPLES, len(day_means)),
    )
    draws = day_means[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(day_means)),
        "samples": BOOTSTRAP_SAMPLES,
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def horizon_metrics(
    frame: pd.DataFrame,
    horizon: int,
) -> dict[str, object]:
    target_column = f"hold_advantage_{horizon}m_pct"
    score_column = f"predicted_hold_advantage_{horizon}m_pct"
    target = pd.to_numeric(frame[target_column], errors="coerce")
    score = pd.to_numeric(frame[score_column], errors="coerce")
    valid = target.notna() & score.notna()
    selected = frame.loc[valid & score.gt(0)].copy()
    return {
        "horizon_minutes": horizon,
        "rows": int(len(frame)),
        "target_rows": int(valid.sum()),
        "target_coverage": float(valid.mean()) if len(frame) else None,
        "spearman": _safe_spearman(
            target.loc[valid],
            score.loc[valid],
        ),
        "predicted_positive_rows": int(len(selected)),
        "predicted_positive_rate": (
            float(len(selected) / int(valid.sum()))
            if int(valid.sum())
            else None
        ),
        "predicted_positive_realized_mean_pct": (
            float(
                pd.to_numeric(
                    selected[target_column], errors="coerce"
                ).mean()
            )
            if len(selected)
            else None
        ),
        "predicted_positive_day_balanced_mean_pct": _day_balanced(
            selected,
            target_column,
        ),
    }


def chosen_horizon_metrics(frame: pd.DataFrame) -> dict[str, object]:
    predictions = np.column_stack(
        [
            pd.to_numeric(
                frame[f"predicted_hold_advantage_{h}m_pct"],
                errors="coerce",
            ).to_numpy(dtype=float)
            for h in VALUE_HORIZONS
        ]
    )
    choice_index = np.argmax(predictions, axis=1)
    horizons = np.asarray(VALUE_HORIZONS, dtype=int)
    chosen_horizon = horizons[choice_index]
    chosen_prediction = predictions[
        np.arange(len(frame)),
        choice_index,
    ]

    target_matrix = np.column_stack(
        [
            pd.to_numeric(
                frame[f"hold_advantage_{h}m_pct"],
                errors="coerce",
            ).to_numpy(dtype=float)
            for h in VALUE_HORIZONS
        ]
    )
    chosen_target = target_matrix[
        np.arange(len(frame)),
        choice_index,
    ]

    scored = frame.copy()
    scored["chosen_horizon_minutes"] = chosen_horizon
    scored["chosen_predicted_advantage_pct"] = chosen_prediction
    scored["chosen_realized_advantage_pct"] = chosen_target

    valid = (
        np.isfinite(chosen_prediction)
        & np.isfinite(chosen_target)
    )
    selected_mask = valid & (chosen_prediction > 0)
    selected = scored.loc[selected_mask].copy()

    by_day: dict[str, object] = {}
    good_days = 0
    positive_spearman_days = 0
    positive_value_days = 0
    for day, part in scored.groupby(
        scored["trading_day"].astype(str),
        sort=True,
    ):
        pv = pd.to_numeric(
            part["chosen_predicted_advantage_pct"],
            errors="coerce",
        )
        tv = pd.to_numeric(
            part["chosen_realized_advantage_pct"],
            errors="coerce",
        )
        day_valid = pv.notna() & tv.notna()
        corr = _safe_spearman(
            tv.loc[day_valid],
            pv.loc[day_valid],
        )
        day_selected = part.loc[day_valid & pv.gt(0)].copy()
        mean_value = (
            float(
                pd.to_numeric(
                    day_selected["chosen_realized_advantage_pct"],
                    errors="coerce",
                ).mean()
            )
            if len(day_selected)
            else None
        )
        if corr is not None and corr > 0:
            positive_spearman_days += 1
        if mean_value is not None and mean_value > 0:
            positive_value_days += 1
        both = bool(
            corr is not None
            and corr > 0
            and mean_value is not None
            and mean_value > 0
        )
        good_days += int(both)
        by_day[str(day)] = {
            "rows": int(len(part)),
            "evaluable_rows": int(day_valid.sum()),
            "spearman": corr,
            "selected_rows": int(len(day_selected)),
            "selected_realized_advantage_mean_pct": mean_value,
        }

    bootstrap = _bootstrap_days(
        selected,
        "chosen_realized_advantage_pct",
    )
    selected_rate = (
        float(len(selected) / int(valid.sum()))
        if int(valid.sum())
        else None
    )
    metrics = {
        "rows": int(len(scored)),
        "evaluable_rows": int(valid.sum()),
        "evaluable_coverage": float(valid.mean()) if len(scored) else None,
        "spearman": _safe_spearman(
            pd.Series(chosen_target[valid]),
            pd.Series(chosen_prediction[valid]),
        ),
        "selected_rows": int(len(selected)),
        "selected_rate": selected_rate,
        "selected_realized_advantage_mean_pct": (
            float(
                pd.to_numeric(
                    selected["chosen_realized_advantage_pct"],
                    errors="coerce",
                ).mean()
            )
            if len(selected)
            else None
        ),
        "selected_day_balanced_advantage_mean_pct": _day_balanced(
            selected,
            "chosen_realized_advantage_pct",
        ),
        "selected_day_bootstrap": bootstrap,
        "positive_spearman_days": int(positive_spearman_days),
        "positive_value_days": int(positive_value_days),
        "both_positive_days": int(good_days),
        "chosen_horizon_counts": {
            str(int(k)): int(v)
            for k, v in pd.Series(chosen_horizon).value_counts().to_dict().items()
        },
        "by_day": by_day,
    }
    metrics["signal_bridge_pass"] = bool(
        metrics["evaluable_coverage"] is not None
        and float(metrics["evaluable_coverage"])
        >= MIN_CHOSEN_TARGET_COVERAGE
        and metrics["spearman"] is not None
        and float(metrics["spearman"]) >= MIN_CHOSEN_SPEARMAN
        and selected_rate is not None
        and selected_rate >= MIN_SELECTED_RATE
        and metrics[
            "selected_day_balanced_advantage_mean_pct"
        ]
        is not None
        and float(
            metrics[
                "selected_day_balanced_advantage_mean_pct"
            ]
        )
        > 0
        and bootstrap["ci_low_pct"] is not None
        and float(bootstrap["ci_low_pct"]) > 0
        and good_days >= MIN_GOOD_DAYS
    )
    return metrics


def run(
    fit_path: Path,
    calibration_path: Path,
    evaluation_path: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit = add_future_advantage_targets(pd.read_parquet(fit_path))
    calibration = add_future_advantage_targets(
        pd.read_parquet(calibration_path)
    )
    evaluation = add_future_advantage_targets(
        pd.read_parquet(evaluation_path)
    )

    models = {
        h: train_horizon_model(fit, calibration, h)
        for h in VALUE_HORIZONS
    }
    scored = evaluation.copy()
    for h, fitted in models.items():
        scored[
            f"predicted_hold_advantage_{h}m_pct"
        ] = predict_horizon(scored, fitted)

    per_horizon = {
        str(h): horizon_metrics(scored, h)
        for h in VALUE_HORIZONS
    }
    chosen = chosen_horizon_metrics(scored)

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "position_features_changed": False,
        "one_minute_hold_classifier_used": False,
        "value_horizons_minutes": list(VALUE_HORIZONS),
        "semantics": (
            "Prediction horizons only. A future recurrent controller must "
            "re-evaluate after each reached minute and is not committed to "
            "the chosen diagnostic horizon."
        ),
        "model_diagnostics": {
            str(h): {
                "offset": models[h].offset,
                "winsor_low_pct": models[h].winsor_low,
                "winsor_high_pct": models[h].winsor_high,
            }
            for h in VALUE_HORIZONS
        },
        "per_horizon": per_horizon,
        "chosen_horizon": chosen,
        "frozen_gate": {
            "min_chosen_target_coverage": MIN_CHOSEN_TARGET_COVERAGE,
            "min_chosen_spearman": MIN_CHOSEN_SPEARMAN,
            "min_selected_rate": MIN_SELECTED_RATE,
            "min_good_days": MIN_GOOD_DAYS,
            "bootstrap_ci_low_must_be_positive": True,
        },
        "promotion_gate_pass": bool(chosen["signal_bridge_pass"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_output_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-output", type=Path, required=True)
    args = parser.parse_args()
    return run(
        args.fit,
        args.calibration,
        args.evaluation,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
