"""Request 235: market-conditioned economic calibration.

Request234 showed that a single global monotonic zero boundary is unstable.
Request235 keeps the Request232 tradability ranking frozen and adds only causal
market-regime context derived from first-HOT episodes that occurred strictly
before the current timestamp on the same trading day.

A low-capacity gradient-boosted regressor maps the frozen Request232 scores plus
that prior-HOT market context to expected persistent tradability. The two frozen
Request232 score dimensions are constrained monotonic-positive. Calibration is
checked leave-one-day-out across the eight already-opened calibration days.
Fresh June outcomes are opened only if that out-of-day economic gate passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .economic_full_hot_admission import _selection_metrics
from .event_time_full_hot_tradability import MIN_TARGET_COVERAGE
from .extended_history_economic_opportunity import EXTENDED_CAL_DAYS
from .fresh_validated_attention_economic_opportunity import FRESH_EVAL_DAYS

REQUEST_ID = 235
SOURCE_RUN = 36251239308

MIN_SELECTED_RATE = 0.03
MAX_SELECTED_RATE = 0.25
MIN_SELECTED_ROWS = 40
MIN_SELECTED_POSITIVE_RATE = 0.35
MIN_CAL_POSITIVE_MEAN_DAYS = 5
MIN_FRESH_POSITIVE_MEAN_DAYS = 4
MIN_MEAN_GAIN_VS_REQUEST232_PCT = 0.25
MIN_FIT_ROWS = 900

CONTEXT_SOURCE_COLUMNS = [
    "return_from_previous_close_pct",
    "attention_score",
    "second_rerank_probability",
    "minute_body_return_pct",
    "sec_last10_return_pct",
    "sec_realized_vol_pct",
]

MODEL_FEATURES = [
    "predicted_persistent_tradability_pct",
    "tradability_positive_probability",
    "minutes_since_open",
    "prior_hot_count_log",
    "prior_hot_mean_return_pct",
    "prior_hot_mean_attention_score",
    "prior_hot_mean_second_probability",
    "prior_hot_mean_minute_body_pct",
    "prior_hot_mean_sec_last10_return_pct",
    "prior_hot_mean_sec_realized_vol_pct",
]

MONOTONIC = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def add_causal_market_context(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trading_day",
        "ticker",
        "t",
        "minutes_since_open",
        "predicted_persistent_tradability_pct",
        "tradability_positive_probability",
        *CONTEXT_SOURCE_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"request235 frozen rows missing columns: {sorted(missing)}"
        )

    result = frame.copy()
    mapping = {
        "return_from_previous_close_pct": "prior_hot_mean_return_pct",
        "attention_score": "prior_hot_mean_attention_score",
        "second_rerank_probability": "prior_hot_mean_second_probability",
        "minute_body_return_pct": "prior_hot_mean_minute_body_pct",
        "sec_last10_return_pct": "prior_hot_mean_sec_last10_return_pct",
        "sec_realized_vol_pct": "prior_hot_mean_sec_realized_vol_pct",
    }
    result["prior_hot_count_log"] = np.nan
    for output in mapping.values():
        result[output] = np.nan

    for day, indices in result.groupby("trading_day", sort=False).groups.items():
        part = result.loc[indices].sort_values(
            ["t", "ticker"], kind="stable"
        )
        count = 0
        sums = {column: 0.0 for column in CONTEXT_SOURCE_COLUMNS}
        valid_counts = {column: 0 for column in CONTEXT_SOURCE_COLUMNS}

        for _, same_t in part.groupby("t", sort=True):
            prior_count_log = float(np.log1p(count))
            prior_means = {}
            for source, output in mapping.items():
                n = valid_counts[source]
                prior_means[output] = (
                    sums[source] / n if n > 0 else np.nan
                )

            for idx in same_t.index:
                result.at[idx, "prior_hot_count_log"] = prior_count_log
                for output, value in prior_means.items():
                    result.at[idx, output] = value

            count += len(same_t)
            for source in CONTEXT_SOURCE_COLUMNS:
                values = pd.to_numeric(
                    same_t[source], errors="coerce"
                ).dropna()
                if len(values):
                    sums[source] += float(values.sum())
                    valid_counts[source] += int(len(values))

    return result


def _fit_model(frame: pd.DataFrame) -> HistGradientBoostingRegressor:
    target = pd.to_numeric(
        frame["persistent_tradability_pct"],
        errors="coerce",
    )
    valid = target.notna()
    work = frame.loc[valid].copy()
    target = pd.to_numeric(
        work["persistent_tradability_pct"],
        errors="coerce",
    )
    if len(work) < MIN_FIT_ROWS:
        raise ValueError(
            f"request235 calibration fit needs >= {MIN_FIT_ROWS}, "
            f"got {len(work)}"
        )

    day = work["trading_day"].astype(str)
    counts = day.value_counts()
    weights = day.map(
        lambda value: 1.0 / float(counts.loc[value])
    ).to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=100,
        l2_regularization=5.0,
        monotonic_cst=MONOTONIC,
        random_state=20261305,
    )
    x = work.reindex(columns=MODEL_FEATURES).apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    model.fit(
        x,
        target.to_numpy(dtype=float),
        sample_weight=weights,
    )
    return model


def _predict(
    frame: pd.DataFrame,
    model: HistGradientBoostingRegressor,
) -> np.ndarray:
    x = frame.reindex(columns=MODEL_FEATURES).apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    return model.predict(x)


def cross_fitted_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["market_calibrated_expected_value_pct"] = np.nan
    for held_day in EXTENDED_CAL_DAYS:
        train_days = [d for d in EXTENDED_CAL_DAYS if d != held_day]
        train = result.loc[
            result["trading_day"].astype(str).isin(train_days)
        ].copy()
        held = result.loc[
            result["trading_day"].astype(str).eq(held_day)
        ].copy()
        model = _fit_model(train)
        result.loc[
            held.index,
            "market_calibrated_expected_value_pct",
        ] = _predict(held, model)
    return result


def add_selection(
    frame: pd.DataFrame,
    value_column: str,
    output_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    value = pd.to_numeric(result[value_column], errors="coerce")
    result[output_column] = (
        result["tradability_selected"].fillna(False).astype(bool)
        & value.gt(0.0)
    )
    return result


def economic_gate(
    candidate: dict[str, object],
    baseline: dict[str, object],
    required_positive_days: int,
) -> tuple[bool, dict[str, object]]:
    mean = candidate["selected_mean_pct"]
    day_mean = candidate["day_balanced_selected_mean_pct"]
    rate = candidate["selected_rate"]
    positive_rate = candidate["selected_positive_rate"]
    rows = int(candidate["selected_rows"])
    baseline_mean = baseline["selected_mean_pct"]
    gain = (
        float(mean) - float(baseline_mean)
        if mean is not None and baseline_mean is not None
        else None
    )
    checks = {
        "min_selected_rows": rows >= MIN_SELECTED_ROWS,
        "selected_rate_in_range": (
            rate is not None
            and MIN_SELECTED_RATE <= float(rate) <= MAX_SELECTED_RATE
        ),
        "selected_mean_positive": mean is not None and float(mean) > 0,
        "day_balanced_mean_positive": (
            day_mean is not None and float(day_mean) > 0
        ),
        "selected_positive_rate": (
            positive_rate is not None
            and float(positive_rate) >= MIN_SELECTED_POSITIVE_RATE
        ),
        "positive_mean_days": (
            int(candidate["positive_selected_mean_days"])
            >= required_positive_days
        ),
        "mean_gain_vs_request232": (
            gain is not None
            and gain >= MIN_MEAN_GAIN_VS_REQUEST232_PCT
        ),
    }
    return bool(all(checks.values())), {
        "checks": checks,
        "selected_mean_gain_vs_request232_pct": gain,
    }


def evaluate(
    scored_hot_path: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    scored_hot = pd.read_parquet(scored_hot_path)
    scored_hot = add_causal_market_context(scored_hot)
    scored_hot = cross_fitted_predictions(scored_hot)
    scored_hot = add_selection(
        scored_hot,
        "market_calibrated_expected_value_pct",
        "request235_oof_selected",
    )

    cal_baseline = _selection_metrics(
        scored_hot, EXTENDED_CAL_DAYS, "tradability_selected"
    )
    cal_candidate = _selection_metrics(
        scored_hot, EXTENDED_CAL_DAYS, "request235_oof_selected"
    )
    cal_gate, cal_detail = economic_gate(
        cal_candidate,
        cal_baseline,
        MIN_CAL_POSITIVE_MEAN_DAYS,
    )

    fresh_baseline = None
    fresh_candidate = None
    fresh_gate = False
    fresh_detail = None

    if cal_gate:
        cal = scored_hot.loc[
            scored_hot["trading_day"].astype(str).isin(EXTENDED_CAL_DAYS)
        ].copy()
        final_model = _fit_model(cal)
        fresh_mask = scored_hot["trading_day"].astype(str).isin(
            FRESH_EVAL_DAYS
        )
        scored_hot.loc[
            fresh_mask,
            "market_calibrated_expected_value_pct",
        ] = _predict(scored_hot.loc[fresh_mask], final_model)
        scored_hot = add_selection(
            scored_hot,
            "market_calibrated_expected_value_pct",
            "request235_fresh_selected",
        )
        fresh_baseline = _selection_metrics(
            scored_hot, FRESH_EVAL_DAYS, "tradability_selected"
        )
        fresh_candidate = _selection_metrics(
            scored_hot, FRESH_EVAL_DAYS, "request235_fresh_selected"
        )
        fresh_gate, fresh_detail = economic_gate(
            fresh_candidate,
            fresh_baseline,
            MIN_FRESH_POSITIVE_MEAN_DAYS,
        )

    fresh_coverage = (
        fresh_candidate["coverage"]
        if fresh_candidate is not None else None
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "source_run": SOURCE_RUN,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "architecture": (
            "frozen Request232 tradability ranking plus low-capacity "
            "market-conditioned economic calibration"
        ),
        "market_context": {
            "strictly_prior_same_day_first_hot_only": True,
            "same_timestamp_rows_excluded_from_each_other": True,
            "features": MODEL_FEATURES,
            "score_monotonic_constraints": {
                "predicted_persistent_tradability_pct": 1,
                "tradability_positive_probability": 1,
            },
            "fresh_used_for_fit_or_selection": False,
        },
        "calibration_oof": {
            "request232_baseline": cal_baseline,
            "candidate": cal_candidate,
            "gate": cal_detail,
            "economic_gate_pass": cal_gate,
        },
        "fresh_evaluated": bool(cal_gate),
        "fresh": (
            {
                "request232_baseline": fresh_baseline,
                "candidate": fresh_candidate,
                "gate": fresh_detail,
                "economic_gate_pass": fresh_gate,
            }
            if cal_gate else None
        ),
        "coverage_promotion_ready": bool(
            fresh_coverage is not None
            and float(fresh_coverage) >= MIN_TARGET_COVERAGE
        ),
        "development_gate_pass": bool(cal_gate and fresh_gate),
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze market-conditioned economic admission and train "
            "a new causal ENTER-vs-WAIT timing controller only inside "
            "admitted episodes; repair target coverage separately."
        ),
        "next_boundary_if_fail_calibration": (
            "even causal market conditioning cannot produce a stable "
            "positive admission boundary from the current Request232 "
            "representation; revisit the tradability target/representation "
            "instead of adding more admission layers."
        ),
        "next_boundary_if_fail_fresh": (
            "market-conditioned calibration is development-stable but "
            "not June-stable; inspect regime transport before timing."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scored_hot.to_parquet(
        rows_output_path, index=False, compression="zstd"
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
