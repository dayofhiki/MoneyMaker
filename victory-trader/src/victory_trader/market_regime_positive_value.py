"""Request 174: direct multi-minute positive-value classifier.

No new dates. Reuse Request-171 rich-second POSITION phases and Request-163
market-wide scan. Build the same Request-173 causal market context, then
classify whether excess remaining option value is positive.

This is not the failed one-minute HOLD target. It asks whether a position state
still has above-baseline multi-minute option value.
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

from .market_regime_position_value import (
    MARKET_REGIME_FEATURES,
    attach_market_regime,
    build_market_regime,
)
from .rich_second_position_value import RICH_SECOND_FEATURES
from .selected_hot_position_value_observability import (
    MODEL_FEATURES,
    apply_excess_target,
    fit_minute_baselines,
)

REQUEST_ID = 174
RANDOM_SEED = 20261074

MIN_CONTEXT_COVERAGE = 0.95
MIN_GLOBAL_AUC = 0.55
MIN_SAME_MINUTE_AUC = 0.53
MIN_SELECTED_RATE = 0.10
MIN_SELECTED_POSITIVE_RATE = 0.58
MIN_GOOD_DAYS = 4
BOOTSTRAP_SAMPLES = 10_000


@dataclass(frozen=True)
class PositiveValueClassifier:
    model: HistGradientBoostingClassifier
    platt: LogisticRegression
    feature_columns: tuple[str, ...]


def _columns() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                *MODEL_FEATURES,
                *RICH_SECOND_FEATURES,
                *MARKET_REGIME_FEATURES,
            ]
        )
    )


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reindex(columns=_columns()).apply(
        pd.to_numeric,
        errors="coerce",
    )


def train_classifier(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> PositiveValueClassifier:
    fit_target = pd.to_numeric(
        fit["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    cal_target = pd.to_numeric(
        calibration["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    fit_valid = fit_target.notna()
    cal_valid = cal_target.notna()
    train = fit.loc[fit_valid].copy()
    cal = calibration.loc[cal_valid].copy()
    y_fit = fit_target.loc[fit_valid].gt(0).astype(int)
    y_cal = cal_target.loc[cal_valid].gt(0).astype(int)
    if (
        len(train) < 1000
        or len(cal) < 500
        or y_fit.nunique() < 2
        or y_cal.nunique() < 2
    ):
        raise ValueError("request 174 insufficient classifier support")

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=RANDOM_SEED,
    )
    model.fit(_feature_frame(train), y_fit)

    raw = np.clip(
        model.predict_proba(_feature_frame(cal))[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    platt.fit(logits, y_cal)
    return PositiveValueClassifier(
        model=model,
        platt=platt,
        feature_columns=_columns(),
    )


def predict_probability(
    frame: pd.DataFrame,
    fitted: PositiveValueClassifier,
) -> np.ndarray:
    raw = np.clip(
        fitted.model.predict_proba(_feature_frame(frame))[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    return fitted.platt.predict_proba(logits)[:, 1]


def _safe_auc(y: pd.Series, score: pd.Series) -> float | None:
    yy = pd.to_numeric(y, errors="coerce")
    ss = pd.to_numeric(score, errors="coerce")
    valid = yy.notna() & ss.notna()
    yy = yy.loc[valid].astype(int)
    if yy.nunique() < 2 or len(yy) < 20:
        return None
    return float(roc_auc_score(yy, ss.loc[valid]))


def _bootstrap_day_means(
    selected: pd.DataFrame,
) -> dict[str, float | int | None]:
    values = pd.to_numeric(
        selected["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    temp = selected.loc[values.notna(), ["trading_day"]].copy()
    temp["value"] = values.loc[values.notna()].to_numpy()
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

    wanted_days = set(
        fit["trading_day"].astype(str).unique().tolist()
        + calibration["trading_day"].astype(str).unique().tolist()
        + evaluation["trading_day"].astype(str).unique().tolist()
    )
    scan = scan.loc[
        scan["trading_day"].astype(str).isin(wanted_days)
    ].copy()
    regime = build_market_regime(scan)
    fit = attach_market_regime(fit, regime)
    calibration = attach_market_regime(calibration, regime)
    evaluation = attach_market_regime(evaluation, regime)

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(calibration, baselines)
    evaluation = apply_excess_target(evaluation, baselines)

    fitted = train_classifier(fit, calibration)
    scored = evaluation.copy()
    scored["positive_value_probability"] = predict_probability(
        scored, fitted
    )

    target = pd.to_numeric(
        scored["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    probability = pd.to_numeric(
        scored["positive_value_probability"],
        errors="coerce",
    )
    valid = target.notna() & probability.notna()
    y = target.loc[valid].gt(0).astype(int)
    global_auc = _safe_auc(y, probability.loc[valid])
    base_positive_rate = float(y.mean()) if len(y) else None

    minute_aucs: list[float] = []
    for _, group in scored.loc[valid].groupby(
        "minutes_held",
        sort=True,
    ):
        gt = pd.to_numeric(
            group["excess_remaining_option_value_pct"],
            errors="coerce",
        )
        gp = pd.to_numeric(
            group["positive_value_probability"],
            errors="coerce",
        )
        auc = _safe_auc(gt.gt(0).astype(int), gp)
        if auc is not None:
            minute_aucs.append(auc)

    selected = scored.loc[
        valid & probability.gt(0.5)
    ].copy()
    selected_target = pd.to_numeric(
        selected["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    selected_rate = (
        float(len(selected) / int(valid.sum()))
        if int(valid.sum())
        else None
    )
    selected_positive_rate = (
        float(selected_target.gt(0).mean()) if len(selected) else None
    )
    selected_mean = (
        float(selected_target.mean()) if len(selected) else None
    )
    daily = (
        pd.DataFrame(
            {
                "day": selected["trading_day"].astype(str),
                "value": selected_target,
            }
        )
        .dropna()
        .groupby("day", sort=True)["value"]
        .agg(["size", "mean"])
    )
    day_balanced = (
        float(daily["mean"].mean()) if len(daily) else None
    )
    bootstrap = _bootstrap_day_means(selected)

    by_day: dict[str, object] = {}
    good_days = 0
    for day in sorted(scored["trading_day"].astype(str).unique()):
        part = scored.loc[
            scored["trading_day"].astype(str).eq(day)
        ].copy()
        pt = pd.to_numeric(
            part["excess_remaining_option_value_pct"],
            errors="coerce",
        )
        pp = pd.to_numeric(
            part["positive_value_probability"],
            errors="coerce",
        )
        pv = pt.notna() & pp.notna()
        day_auc = _safe_auc(
            pt.loc[pv].gt(0).astype(int),
            pp.loc[pv],
        )
        psel = part.loc[pv & pp.gt(0.5)].copy()
        vals = pd.to_numeric(
            psel["excess_remaining_option_value_pct"],
            errors="coerce",
        )
        mean = float(vals.mean()) if len(psel) else None
        pos = float(vals.gt(0).mean()) if len(psel) else None
        good = bool(
            day_auc is not None
            and day_auc > 0.5
            and mean is not None
            and mean > 0
        )
        good_days += int(good)
        by_day[day] = {
            "rows": int(len(part)),
            "evaluable_rows": int(pv.sum()),
            "auc": day_auc,
            "selected_rows": int(len(psel)),
            "selected_rate": (
                float(len(psel) / int(pv.sum()))
                if int(pv.sum())
                else None
            ),
            "selected_positive_rate": pos,
            "selected_realized_excess_mean_pct": mean,
            "both_positive": good,
        }

    context_coverage = float(
        evaluation.loc[:, MARKET_REGIME_FEATURES]
        .notna()
        .any(axis=1)
        .mean()
    )
    same_minute_median = (
        float(np.median(minute_aucs)) if minute_aucs else None
    )

    gate = bool(
        context_coverage >= MIN_CONTEXT_COVERAGE
        and global_auc is not None
        and global_auc >= MIN_GLOBAL_AUC
        and same_minute_median is not None
        and same_minute_median >= MIN_SAME_MINUTE_AUC
        and selected_rate is not None
        and selected_rate >= MIN_SELECTED_RATE
        and selected_positive_rate is not None
        and selected_positive_rate >= MIN_SELECTED_POSITIVE_RATE
        and selected_mean is not None
        and selected_mean > 0
        and day_balanced is not None
        and day_balanced > 0
        and bootstrap["ci_low_pct"] is not None
        and float(bootstrap["ci_low_pct"]) > 0
        and good_days >= MIN_GOOD_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "target": "excess_remaining_option_value_pct > 0",
        "semantic_boundary": "P(positive value) > 0.5",
        "market_context_coverage": context_coverage,
        "global_auc": global_auc,
        "same_minute_auc_median": same_minute_median,
        "same_minute_auc_positive_fraction": (
            float(np.mean(np.asarray(minute_aucs) > 0.5))
            if minute_aucs
            else None
        ),
        "unconditional_positive_rate": base_positive_rate,
        "selected_rows": int(len(selected)),
        "selected_rate": selected_rate,
        "selected_positive_rate": selected_positive_rate,
        "selected_realized_excess_mean_pct": selected_mean,
        "selected_day_balanced_excess_mean_pct": day_balanced,
        "selected_day_bootstrap": bootstrap,
        "good_days": int(good_days),
        "by_day": by_day,
        "feature_count": len(fitted.feature_columns),
        "frozen_gate": {
            "min_context_coverage": MIN_CONTEXT_COVERAGE,
            "min_global_auc": MIN_GLOBAL_AUC,
            "min_same_minute_auc": MIN_SAME_MINUTE_AUC,
            "min_selected_rate": MIN_SELECTED_RATE,
            "min_selected_positive_rate": MIN_SELECTED_POSITIVE_RATE,
            "min_good_days": MIN_GOOD_DAYS,
            "bootstrap_ci_low_must_be_positive": True,
        },
        "positive_value_classifier_pass": gate,
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
