"""Request 176: hurdle expected remaining-value model.

No new dates. Reuse Request-171 rich-second POSITION phases plus Request-173
causal market regime. Replace the failed P(value>0)>0.5 semantic boundary with
a preregistered three-head expected-value decomposition:

EV = P(value>0) * E[value | value>0]
   + (1-P(value>0)) * E[value | value<=0]

Probability is chronologically Platt-calibrated. Positive and non-positive
magnitude heads are independently fit on the fit partition and receive only a
chronological class-conditional mean-bias offset on calibration.

EV>0 is the only semantic selection boundary. No June threshold tuning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .market_regime_position_value import (
    MARKET_REGIME_FEATURES,
    attach_market_regime,
    build_market_regime,
)
from .market_regime_positive_value import (
    PositiveValueClassifier,
    _feature_frame,
    predict_probability,
    train_classifier,
)
from .selected_hot_position_value_observability import (
    apply_excess_target,
    fit_minute_baselines,
)

REQUEST_ID = 176
RANDOM_SEED = 20261076
BOOTSTRAP_SAMPLES = 10_000

MIN_CONTEXT_COVERAGE = 0.95
MIN_EV_SPEARMAN = 0.08
MIN_SELECTED_RATE = 0.10
MIN_SELECTED_REALIZED_MEAN_PCT = 0.20
MIN_GOOD_DAYS = 4


@dataclass(frozen=True)
class MagnitudeHead:
    model: HistGradientBoostingRegressor
    offset: float
    winsor_low: float
    winsor_high: float
    positive_class: bool


def _train_magnitude_head(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    positive_class: bool,
    seed: int,
) -> MagnitudeHead:
    fit_target = pd.to_numeric(
        fit["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    cal_target = pd.to_numeric(
        calibration["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    if positive_class:
        fit_mask = fit_target.gt(0)
        cal_mask = cal_target.gt(0)
    else:
        fit_mask = fit_target.le(0)
        cal_mask = cal_target.le(0)

    train = fit.loc[fit_mask].copy()
    y = fit_target.loc[fit_mask].astype(float)
    cal = calibration.loc[cal_mask].copy()
    y_cal = cal_target.loc[cal_mask].astype(float)

    if len(train) < 500 or len(cal) < 200:
        raise ValueError(
            "request 176 has insufficient class-conditional magnitude support"
        )

    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _feature_frame(train),
        y.clip(lower=low, upper=high),
    )
    raw = model.predict(_feature_frame(cal))
    offset = float(y_cal.mean() - float(np.mean(raw)))
    return MagnitudeHead(
        model=model,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
        positive_class=positive_class,
    )


def _predict_magnitude(
    frame: pd.DataFrame,
    head: MagnitudeHead,
) -> np.ndarray:
    pred = head.model.predict(_feature_frame(frame)) + head.offset
    if head.positive_class:
        return np.maximum(pred, 0.0)
    return np.minimum(pred, 0.0)


def _safe_spearman(
    target: pd.Series,
    score: pd.Series | np.ndarray,
) -> float | None:
    y = pd.to_numeric(target, errors="coerce")
    s = pd.to_numeric(pd.Series(score, index=target.index), errors="coerce")
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    value = y.loc[valid].corr(s.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _bootstrap_day_means(
    selected: pd.DataFrame,
) -> dict[str, float | int | None]:
    target = pd.to_numeric(
        selected["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    temp = selected.loc[target.notna(), ["trading_day"]].copy()
    temp["value"] = target.loc[target.notna()].to_numpy()
    daily = (
        temp.groupby(temp["trading_day"].astype(str), sort=True)["value"]
        .mean()
        .to_numpy(dtype=float)
    )
    if len(daily) < 2:
        return {
            "days": int(len(daily)),
            "samples": BOOTSTRAP_SAMPLES,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.integers(
        0,
        len(daily),
        size=(BOOTSTRAP_SAMPLES, len(daily)),
    )
    draws = daily[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(daily)),
        "samples": BOOTSTRAP_SAMPLES,
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def _score_frame(
    frame: pd.DataFrame,
    classifier: PositiveValueClassifier,
    positive_head: MagnitudeHead,
    nonpositive_head: MagnitudeHead,
) -> pd.DataFrame:
    scored = frame.copy()
    p = predict_probability(scored, classifier)
    win = _predict_magnitude(scored, positive_head)
    loss = _predict_magnitude(scored, nonpositive_head)
    scored["positive_value_probability"] = p
    scored["predicted_positive_magnitude_pct"] = win
    scored["predicted_nonpositive_magnitude_pct"] = loss
    scored["hurdle_expected_value_pct"] = p * win + (1.0 - p) * loss
    return scored


def _evaluate(
    frame: pd.DataFrame,
) -> dict[str, object]:
    target = pd.to_numeric(
        frame["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    ev = pd.to_numeric(
        frame["hurdle_expected_value_pct"],
        errors="coerce",
    )
    valid = target.notna() & ev.notna()
    part = frame.loc[valid].copy()
    target_v = target.loc[valid]
    ev_v = ev.loc[valid]

    selected = part.loc[
        pd.to_numeric(
            part["hurdle_expected_value_pct"],
            errors="coerce",
        ).gt(0)
    ].copy()
    selected_target = pd.to_numeric(
        selected["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    selected_rate = (
        float(len(selected) / len(part)) if len(part) else None
    )
    selected_mean = (
        float(selected_target.mean()) if len(selected) else None
    )
    selected_positive_rate = (
        float(selected_target.gt(0).mean()) if len(selected) else None
    )
    daily_selected = (
        pd.DataFrame(
            {
                "day": selected["trading_day"].astype(str),
                "value": selected_target,
            }
        )
        .dropna()
        .groupby("day", sort=True)["value"]
        .mean()
    )
    day_balanced = (
        float(daily_selected.mean()) if len(daily_selected) else None
    )
    bootstrap = _bootstrap_day_means(selected)

    by_day: dict[str, object] = {}
    good_days = 0
    for day in sorted(part["trading_day"].astype(str).unique()):
        day_part = part.loc[
            part["trading_day"].astype(str).eq(day)
        ].copy()
        day_target = pd.to_numeric(
            day_part["excess_remaining_option_value_pct"],
            errors="coerce",
        )
        day_ev = pd.to_numeric(
            day_part["hurdle_expected_value_pct"],
            errors="coerce",
        )
        corr = _safe_spearman(day_target, day_ev)
        day_selected = day_part.loc[day_ev.gt(0)].copy()
        vals = pd.to_numeric(
            day_selected["excess_remaining_option_value_pct"],
            errors="coerce",
        )
        mean = float(vals.mean()) if len(day_selected) else None
        positive_rate = (
            float(vals.gt(0).mean()) if len(day_selected) else None
        )
        good = bool(
            corr is not None
            and corr > 0
            and mean is not None
            and mean > 0
        )
        good_days += int(good)
        by_day[day] = {
            "rows": int(len(day_part)),
            "spearman": corr,
            "selected_rows": int(len(day_selected)),
            "selected_rate": (
                float(len(day_selected) / len(day_part))
                if len(day_part)
                else None
            ),
            "selected_positive_rate": positive_rate,
            "selected_realized_excess_mean_pct": mean,
            "both_positive": good,
        }

    return {
        "rows": int(len(part)),
        "ev_spearman": _safe_spearman(target_v, ev_v),
        "predicted_ev_mean_pct": float(ev_v.mean()),
        "selected_rows": int(len(selected)),
        "selected_rate": selected_rate,
        "selected_positive_rate": selected_positive_rate,
        "selected_realized_excess_mean_pct": selected_mean,
        "selected_day_balanced_excess_mean_pct": day_balanced,
        "selected_day_bootstrap": bootstrap,
        "good_days": int(good_days),
        "by_day": by_day,
    }


def run(
    fit_path: Path,
    calibration_path: Path,
    evaluation_path: Path,
    scan_path: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    evaluation = pd.read_parquet(evaluation_path)
    scan = pd.read_parquet(scan_path)

    wanted = set(
        fit["trading_day"].astype(str).unique()
    ) | set(calibration["trading_day"].astype(str).unique()) | set(
        evaluation["trading_day"].astype(str).unique()
    )
    scan = scan.loc[
        scan["trading_day"].astype(str).isin(wanted)
    ].copy()
    regime = build_market_regime(scan)
    fit = attach_market_regime(fit, regime)
    calibration = attach_market_regime(calibration, regime)
    evaluation = attach_market_regime(evaluation, regime)

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(calibration, baselines)
    evaluation = apply_excess_target(evaluation, baselines)

    classifier = train_classifier(fit, calibration)
    positive_head = _train_magnitude_head(
        fit,
        calibration,
        positive_class=True,
        seed=RANDOM_SEED + 1,
    )
    nonpositive_head = _train_magnitude_head(
        fit,
        calibration,
        positive_class=False,
        seed=RANDOM_SEED + 2,
    )

    scored = _score_frame(
        evaluation,
        classifier,
        positive_head,
        nonpositive_head,
    )
    metrics = _evaluate(scored)

    context_coverage = float(
        evaluation.loc[:, MARKET_REGIME_FEATURES]
        .notna()
        .any(axis=1)
        .mean()
    )
    gate = bool(
        context_coverage >= MIN_CONTEXT_COVERAGE
        and metrics["ev_spearman"] is not None
        and float(metrics["ev_spearman"]) >= MIN_EV_SPEARMAN
        and metrics["selected_rate"] is not None
        and float(metrics["selected_rate"]) >= MIN_SELECTED_RATE
        and metrics["selected_realized_excess_mean_pct"] is not None
        and float(metrics["selected_realized_excess_mean_pct"])
        >= MIN_SELECTED_REALIZED_MEAN_PCT
        and metrics["selected_day_balanced_excess_mean_pct"] is not None
        and float(metrics["selected_day_balanced_excess_mean_pct"]) > 0
        and metrics["selected_day_bootstrap"]["ci_low_pct"] is not None
        and float(metrics["selected_day_bootstrap"]["ci_low_pct"]) > 0
        and metrics["good_days"] >= MIN_GOOD_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "target": "excess_remaining_option_value_pct",
        "policy_boundary": "hurdle_expected_value_pct > 0",
        "market_context_coverage": context_coverage,
        "metrics": metrics,
        "head_diagnostics": {
            "positive": {
                "offset": positive_head.offset,
                "winsor_low_pct": positive_head.winsor_low,
                "winsor_high_pct": positive_head.winsor_high,
            },
            "nonpositive": {
                "offset": nonpositive_head.offset,
                "winsor_low_pct": nonpositive_head.winsor_low,
                "winsor_high_pct": nonpositive_head.winsor_high,
            },
        },
        "frozen_gate": {
            "min_context_coverage": MIN_CONTEXT_COVERAGE,
            "min_ev_spearman": MIN_EV_SPEARMAN,
            "min_selected_rate": MIN_SELECTED_RATE,
            "min_selected_realized_mean_pct": (
                MIN_SELECTED_REALIZED_MEAN_PCT
            ),
            "min_good_days": MIN_GOOD_DAYS,
            "bootstrap_ci_low_must_be_positive": True,
        },
        "promotion_gate_pass": gate,
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
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-output", type=Path, required=True)
    args = parser.parse_args()
    return run(
        args.fit,
        args.calibration,
        args.evaluation,
        args.scan,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
