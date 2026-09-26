"""Request239: non-consecutive multi-opportunity tradability.

The Request232 target requires the best mean across two consecutive WATCH
states. That may be too rigid for a trader who can wait through a weak minute
and still exploit multiple distinct entry opportunities.

Request239 keeps the same first-HOT causal representation and the same model
families, but changes only the supervised tradability target:

* use the five event-time WATCH values already constructed by Request232;
* require at least two evaluable WATCH states;
* take the two highest WATCH multi-event values, regardless of adjacency;
* average those two values.

A single hindsight spike is still insufficient, but the two opportunities no
longer need to be consecutive. June 8-12 is already-opened development data, so
this request is diagnostic and cannot promote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score

from .extended_history_economic_opportunity import (
    EXTENDED_CAL_DAYS,
    EXTENDED_FIT_DAYS,
)
from .fresh_validated_attention_economic_opportunity import FRESH_EVAL_DAYS
from .hot_economic_opportunity import ENTRY_FEATURES

REQUEST_ID = 239
SOURCE_RUN = 36251239308
WATCH_COLUMNS = [
    f"watch_event_{i}_multi_event_value_pct"
    for i in range(1, 6)
]
MIN_FIT_ROWS = 1500
MIN_CAL_ROWS = 1500

MIN_COVERAGE = 0.80
MIN_SPEARMAN = 0.25
MIN_AUC = 0.65
MIN_SELECTED_POSITIVE_RATE = 0.40
MIN_POSITIVE_SELECTED_MEAN_DAYS = 3

MODEL_SEED_REG = 20261309
MODEL_SEED_CLS = 20261310


def attach_target(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(WATCH_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"request239 rows missing watch columns: {sorted(missing)}"
        )
    result = frame.copy()
    result["nonconsecutive_top2_tradability_pct"] = np.nan
    result["nonconsecutive_evaluable"] = False

    values = result[WATCH_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    for idx, row in values.iterrows():
        valid = row.dropna().to_numpy(dtype=float)
        if len(valid) < 2:
            continue
        top2 = np.sort(valid)[-2:]
        result.at[
            idx,
            "nonconsecutive_top2_tradability_pct",
        ] = float(np.mean(top2))
        result.at[idx, "nonconsecutive_evaluable"] = True
    return result


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)


def _usable_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in ENTRY_FEATURES
        if column in frame.columns
        and pd.to_numeric(
            frame[column], errors="coerce"
        ).notna().any()
    )


def _safe_spearman(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = a.notna() & s.notna()
    if int(valid.sum()) < 3:
        return None
    value = a.loc[valid].corr(
        s.loc[valid], method="spearman"
    )
    return None if pd.isna(value) else float(value)


def _safe_auc(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = a.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    y = a.loc[valid].gt(0).astype(int)
    if y.nunique() < 2:
        return None
    return float(
        roc_auc_score(
            y,
            s.loc[valid].astype(float),
        )
    )


def train(frame: pd.DataFrame):
    day = frame["trading_day"].astype(str)
    fit = frame.loc[
        day.isin(EXTENDED_FIT_DAYS)
    ].copy()
    cal = frame.loc[
        day.isin(EXTENDED_CAL_DAYS)
    ].copy()
    target_col = "nonconsecutive_top2_tradability_pct"

    fit = fit.loc[
        pd.to_numeric(
            fit[target_col], errors="coerce"
        ).notna()
    ].copy()
    cal = cal.loc[
        pd.to_numeric(
            cal[target_col], errors="coerce"
        ).notna()
    ].copy()

    if len(fit) < MIN_FIT_ROWS:
        raise ValueError(
            f"request239 fit needs >= {MIN_FIT_ROWS}, got {len(fit)}"
        )
    if len(cal) < MIN_CAL_ROWS:
        raise ValueError(
            f"request239 cal needs >= {MIN_CAL_ROWS}, got {len(cal)}"
        )

    columns = _usable_columns(fit)
    x_fit = _feature_frame(fit, columns)
    y_fit = pd.to_numeric(
        fit[target_col], errors="raise"
    ).to_numpy(dtype=float)

    low, high = np.quantile(
        y_fit, [0.005, 0.995]
    )
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=MODEL_SEED_REG,
    )
    regressor.fit(
        x_fit,
        np.clip(y_fit, low, high),
    )

    y_class = (y_fit > 0).astype(int)
    if np.unique(y_class).size < 2:
        raise ValueError(
            "request239 fit target needs both classes"
        )
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=MODEL_SEED_CLS,
    )
    classifier.fit(x_fit, y_class)

    x_cal = _feature_frame(cal, columns)
    cal_target = pd.to_numeric(
        cal[target_col], errors="raise"
    ).to_numpy(dtype=float)
    raw = regressor.predict(x_cal)
    offset = float(np.mean(cal_target - raw))
    probability = classifier.predict_proba(
        x_cal
    )[:, 1]
    probability_gate = float(
        np.quantile(probability, 0.75)
    )
    return (
        regressor,
        classifier,
        columns,
        offset,
        probability_gate,
        len(fit),
        len(cal),
    )


def score(
    frame: pd.DataFrame,
    fitted,
) -> pd.DataFrame:
    (
        regressor,
        classifier,
        columns,
        offset,
        probability_gate,
        _,
        _,
    ) = fitted
    result = frame.copy()
    x = _feature_frame(result, columns)
    result["request239_predicted_value_pct"] = (
        regressor.predict(x) + offset
    )
    result["request239_positive_probability"] = (
        classifier.predict_proba(x)[:, 1]
    )
    result["request239_selected"] = result[
        "request239_positive_probability"
    ].ge(probability_gate)
    return result


def _selection_metrics(
    frame: pd.DataFrame,
    day: str,
) -> dict[str, object]:
    part = frame.loc[
        frame["trading_day"].astype(str).eq(day)
    ].copy()
    target = pd.to_numeric(
        part["nonconsecutive_top2_tradability_pct"],
        errors="coerce",
    )
    part = part.loc[target.notna()].copy()
    target = pd.to_numeric(
        part["nonconsecutive_top2_tradability_pct"],
        errors="coerce",
    )
    selected = part.loc[
        part["request239_selected"]
        .fillna(False)
        .astype(bool)
    ].copy()
    sy = pd.to_numeric(
        selected["nonconsecutive_top2_tradability_pct"],
        errors="coerce",
    )
    return {
        "evaluable_rows": int(len(part)),
        "selected_rows": int(len(selected)),
        "selected_rate": (
            float(len(selected) / len(part))
            if len(part) else None
        ),
        "population_mean_pct": (
            float(target.mean())
            if len(target) else None
        ),
        "selected_mean_pct": (
            float(sy.mean()) if len(sy) else None
        ),
        "selected_positive_rate": (
            float(sy.gt(0).mean())
            if len(sy) else None
        ),
    }


def evaluate(
    scored_hot_path: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    source = pd.read_parquet(scored_hot_path)
    labeled = attach_target(source)
    fitted = train(labeled)
    scored = score(labeled, fitted)

    day = scored["trading_day"].astype(str)
    fresh = scored.loc[
        day.isin(FRESH_EVAL_DAYS)
    ].copy()
    target = pd.to_numeric(
        fresh["nonconsecutive_top2_tradability_pct"],
        errors="coerce",
    )
    valid = target.notna()
    evaluable = fresh.loc[valid].copy()
    target = pd.to_numeric(
        evaluable["nonconsecutive_top2_tradability_pct"],
        errors="coerce",
    )

    candidate_spearman = _safe_spearman(
        target,
        evaluable["request239_predicted_value_pct"],
    )
    candidate_auc = _safe_auc(
        target,
        evaluable["request239_positive_probability"],
    )
    runner_spearman = _safe_spearman(
        target,
        evaluable["second_rerank_probability"],
    )
    runner_auc = _safe_auc(
        target,
        evaluable["second_rerank_probability"],
    )

    selected = evaluable.loc[
        evaluable["request239_selected"]
        .fillna(False)
        .astype(bool)
    ].copy()
    selected_target = pd.to_numeric(
        selected["nonconsecutive_top2_tradability_pct"],
        errors="coerce",
    )

    by_day = {
        d: _selection_metrics(scored, d)
        for d in FRESH_EVAL_DAYS
    }
    positive_selected_days = sum(
        1
        for metrics in by_day.values()
        if metrics["selected_mean_pct"] is not None
        and metrics["selected_mean_pct"] > 0
    )

    coverage = (
        float(valid.mean()) if len(fresh) else None
    )
    selected_mean = (
        float(selected_target.mean())
        if len(selected_target) else None
    )
    selected_positive = (
        float(selected_target.gt(0).mean())
        if len(selected_target) else None
    )
    selected_rate = (
        float(len(selected) / len(evaluable))
        if len(evaluable) else None
    )

    gate = bool(
        coverage is not None
        and coverage >= MIN_COVERAGE
        and candidate_spearman is not None
        and candidate_spearman >= MIN_SPEARMAN
        and candidate_auc is not None
        and candidate_auc >= MIN_AUC
        and selected_mean is not None
        and selected_mean > 0
        and selected_positive is not None
        and selected_positive
        >= MIN_SELECTED_POSITIVE_RATE
        and positive_selected_days
        >= MIN_POSITIVE_SELECTED_MEAN_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "source_run": SOURCE_RUN,
        "opens_new_dates": False,
        "development_only": True,
        "promotion_eligible": False,
        "target": {
            "name": "nonconsecutive_top2_tradability",
            "definition": (
                "mean of the two highest Request232 WATCH multi-event "
                "values among the first five observed WATCH states"
            ),
            "requires_two_opportunities": True,
            "requires_consecutive_opportunities": False,
            "single_spike_sufficient": False,
        },
        "support": {
            "fit_labeled_rows": int(fitted[5]),
            "calibration_labeled_rows": int(fitted[6]),
            "fresh_rows": int(len(fresh)),
            "fresh_evaluable_rows": int(len(evaluable)),
            "fresh_coverage": coverage,
        },
        "fresh_development": {
            "population_mean_pct": (
                float(target.mean())
                if len(target) else None
            ),
            "population_positive_rate": (
                float(target.gt(0).mean())
                if len(target) else None
            ),
            "candidate_spearman": candidate_spearman,
            "candidate_auc": candidate_auc,
            "runner_spearman": runner_spearman,
            "runner_auc": runner_auc,
            "selected_rows": int(len(selected)),
            "selected_rate": selected_rate,
            "selected_mean_pct": selected_mean,
            "selected_positive_rate": selected_positive,
            "positive_selected_mean_days": (
                positive_selected_days
            ),
            "by_day": by_day,
        },
        "development_gate": {
            "min_coverage": MIN_COVERAGE,
            "min_spearman": MIN_SPEARMAN,
            "min_auc": MIN_AUC,
            "selected_mean_must_be_positive": True,
            "min_selected_positive_rate": (
                MIN_SELECTED_POSITIVE_RATE
            ),
            "min_positive_selected_mean_days": (
                MIN_POSITIVE_SELECTED_MEAN_DAYS
            ),
            "pass": gate,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze non-consecutive optionality target and validate on "
            "a genuinely new holdout block before timing/controller work."
        ),
        "next_boundary_if_fail": (
            "the adjacency constraint was not the core issue; redesign "
            "tradability around risk-controlled upside versus downside "
            "survivability rather than more admission calibration."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scored.to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-hot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.scored_hot,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
