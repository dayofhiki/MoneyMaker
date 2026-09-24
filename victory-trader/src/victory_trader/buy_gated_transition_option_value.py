"""Request 168: transition-enhanced remaining-option observability.

No new dates. The failed one-minute HOLD classifier is not modified further.
Instead, apply the previously supported causal path-transition information to
the multi-minute excess remaining-option target that survived Requests 166-167.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .buy_gated_path_transition_observability import (
    PATH_TRANSITION_LAGS,
    RICH_PATH_FEATURES,
    _transition_columns,
    _transition_feature_frame,
    add_exact_path_transitions,
)
from .selected_hot_position_value_observability import (
    OPTION_SEED,
    apply_excess_target,
    fit_minute_baselines,
    predict_option,
    train_option_model,
)

REQUEST_ID = 168
BOOTSTRAP_SEED = 20261068
BOOTSTRAP_SAMPLES = 10_000


class TransitionOptionModel:
    def __init__(
        self,
        model: HistGradientBoostingRegressor,
        feature_columns: tuple[str, ...],
        offset: float,
        winsor_low: float,
        winsor_high: float,
    ) -> None:
        self.model = model
        self.feature_columns = feature_columns
        self.offset = offset
        self.winsor_low = winsor_low
        self.winsor_high = winsor_high


def train_transition_option(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> TransitionOptionModel:
    fit_enriched = add_exact_path_transitions(fit)
    cal_enriched = add_exact_path_transitions(calibration)

    target = pd.to_numeric(
        fit_enriched["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    valid = target.notna()
    train = fit_enriched.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 1000:
        raise ValueError("request 168 has insufficient option-value support")

    columns = _transition_columns(train)
    columns = tuple(
        column
        for column in columns
        if pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=OPTION_SEED,
    )
    model.fit(
        _transition_feature_frame(train, columns),
        y.clip(lower=low, upper=high),
    )

    cal_target = pd.to_numeric(
        cal_enriched["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    cal_valid = cal_target.notna()
    if int(cal_valid.sum()) < 500:
        raise ValueError("request 168 has insufficient calibration support")
    prediction = model.predict(
        _transition_feature_frame(
            cal_enriched.loc[cal_valid],
            columns,
        )
    )
    offset = float(
        cal_target.loc[cal_valid].mean() - float(np.mean(prediction))
    )
    return TransitionOptionModel(
        model=model,
        feature_columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_transition_option(
    frame: pd.DataFrame,
    fitted: TransitionOptionModel,
) -> np.ndarray:
    enriched = add_exact_path_transitions(frame)
    return (
        fitted.model.predict(
            _transition_feature_frame(
                enriched,
                fitted.feature_columns,
            )
        )
        + fitted.offset
    )


def _safe_spearman(a: pd.Series, b: pd.Series) -> float | None:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    valid = aa.notna() & bb.notna()
    if int(valid.sum()) < 20:
        return None
    value = aa.loc[valid].corr(bb.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _day_balanced(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    tmp = frame.loc[values.notna(), ["trading_day"]].copy()
    tmp["_value"] = values.loc[values.notna()].to_numpy()
    if tmp.empty:
        return None
    return float(
        tmp.groupby(tmp["trading_day"].astype(str), sort=True)["_value"]
        .mean()
        .mean()
    )


def _day_cluster_bootstrap(
    selected: pd.DataFrame,
) -> dict[str, float | int | None]:
    values = pd.to_numeric(
        selected["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    frame = selected.loc[values.notna(), ["trading_day"]].copy()
    frame["_value"] = values.loc[values.notna()].to_numpy()
    day_means = (
        frame.groupby(frame["trading_day"].astype(str), sort=True)["_value"]
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
    indices = rng.integers(
        0,
        len(day_means),
        size=(BOOTSTRAP_SAMPLES, len(day_means)),
    )
    draws = day_means[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(day_means)),
        "samples": BOOTSTRAP_SAMPLES,
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def evaluate_option_signal(
    frame: pd.DataFrame,
    score_column: str,
    *,
    label: str,
) -> dict[str, object]:
    target = pd.to_numeric(
        frame["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    score = pd.to_numeric(frame[score_column], errors="coerce")
    valid = target.notna() & score.notna()
    eval_frame = frame.loc[valid].copy()

    global_spearman = _safe_spearman(
        target.loc[valid],
        score.loc[valid],
    )

    minute_corrs: list[float] = []
    for _, group in eval_frame.groupby("minutes_held", sort=True):
        if len(group) < 20:
            continue
        corr = _safe_spearman(
            group["excess_remaining_option_value_pct"],
            group[score_column],
        )
        if corr is not None:
            minute_corrs.append(corr)

    selected = eval_frame.loc[
        pd.to_numeric(eval_frame[score_column], errors="coerce").gt(0)
    ].copy()
    selected_mean = (
        float(
            pd.to_numeric(
                selected["excess_remaining_option_value_pct"],
                errors="coerce",
            ).mean()
        )
        if len(selected)
        else None
    )
    selected_day_mean = _day_balanced(
        selected,
        "excess_remaining_option_value_pct",
    )
    bootstrap = _day_cluster_bootstrap(selected)

    by_day: dict[str, object] = {}
    good_days = 0
    for day, part in eval_frame.groupby(
        eval_frame["trading_day"].astype(str),
        sort=True,
    ):
        corr = _safe_spearman(
            part["excess_remaining_option_value_pct"],
            part[score_column],
        )
        day_selected = part.loc[
            pd.to_numeric(part[score_column], errors="coerce").gt(0)
        ]
        day_excess = (
            float(
                pd.to_numeric(
                    day_selected["excess_remaining_option_value_pct"],
                    errors="coerce",
                ).mean()
            )
            if len(day_selected)
            else None
        )
        supported = bool(
            corr is not None
            and corr > 0
            and day_excess is not None
            and day_excess > 0
        )
        good_days += int(supported)
        by_day[str(day)] = {
            "rows": int(len(part)),
            "spearman": corr,
            "selected_rows": int(len(day_selected)),
            "selected_realized_excess_mean_pct": day_excess,
            "both_positive": supported,
        }

    result = {
        "label": label,
        "rows": int(len(eval_frame)),
        "global_spearman": global_spearman,
        "same_minute_spearman_median": (
            float(np.median(minute_corrs)) if minute_corrs else None
        ),
        "same_minute_spearman_positive_fraction": (
            float(np.mean(np.asarray(minute_corrs) > 0))
            if minute_corrs
            else None
        ),
        "selected_rows": int(len(selected)),
        "selected_rate": (
            float(len(selected) / len(eval_frame))
            if len(eval_frame)
            else None
        ),
        "selected_realized_excess_mean_pct": selected_mean,
        "selected_day_balanced_excess_mean_pct": selected_day_mean,
        "selected_day_bootstrap": bootstrap,
        "positive_spearman_and_excess_days": int(good_days),
        "by_day": by_day,
    }
    result["signal_bridge_pass"] = bool(
        global_spearman is not None
        and global_spearman > 0
        and result["same_minute_spearman_median"] is not None
        and float(result["same_minute_spearman_median"]) > 0
        and selected_day_mean is not None
        and selected_day_mean > 0
        and bootstrap["ci_low_pct"] is not None
        and float(bootstrap["ci_low_pct"]) > 0
        and good_days >= 4
    )
    return result


def run(
    fit_path: Path,
    calibration_path: Path,
    evaluation_path: Path,
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

    baseline = train_option_model(fit, calibration)
    transition = train_transition_option(fit, calibration)

    scored = evaluation.copy()
    scored["baseline_option_score"] = predict_option(
        scored,
        baseline,
    )
    scored["transition_option_score"] = predict_transition_option(
        scored,
        transition,
    )

    baseline_metrics = evaluate_option_signal(
        scored,
        "baseline_option_score",
        label="request166_baseline_option",
    )
    transition_metrics = evaluate_option_signal(
        scored,
        "transition_option_score",
        label="transition_option",
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "one_minute_hold_model_changed": False,
        "remaining_option_target_changed": False,
        "path_transition_lags_minutes": list(PATH_TRANSITION_LAGS),
        "rich_path_features": list(RICH_PATH_FEATURES),
        "transition_feature_count": len(
            transition.feature_columns
        ),
        "transition_winsor_low_pct": transition.winsor_low,
        "transition_winsor_high_pct": transition.winsor_high,
        "transition_calibration_offset_pct": transition.offset,
        "baseline": baseline_metrics,
        "transition": transition_metrics,
        "promotion_gate_pass": bool(
            transition_metrics["signal_bridge_pass"]
        ),
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
