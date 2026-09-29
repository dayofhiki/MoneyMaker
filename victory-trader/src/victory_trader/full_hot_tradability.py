"""Request 231: full first-HOT tradability learnability.

Requests 228-230 learned downstream tradability only inside an already
BUY-selected population. That conditioning improved relative ranking a little
but produced negative admitted economics and destroyed the downstream timing
signal.

Request231 removes that sample-selection bottleneck. It reconstructs the frozen
Request163 validated-attention first-HOT population and learns a separate
tradability head for every first-HOT episode.

The label is intentionally different from runner discovery:

* create hypothetical WATCH entry checkpoints at minutes 1..5 after first HOT;
* at each checkpoint, compute BASE net return at exact +1/+2/+3/+5 minute
  executable references;
* average those four returns to obtain a multi-event entry value;
* define persistent tradability as the best mean value across any two
  consecutive WATCH checkpoints.

Thus a single hindsight spike is not enough. Inputs remain the causal first-HOT
features already frozen in Request163. Future prices are labels only.

Development split reuses Request163 already-opened dates:
fit 2026-04-30..2026-05-08, calibration 2026-05-11..2026-05-20,
fresh evaluation 2026-06-08..2026-06-12.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score

from .attention_replay import MINUTE_MS
from .extended_history_economic_opportunity import (
    EXTENDED_CAL_DAYS,
    EXTENDED_FIT_DAYS,
)
from .fresh_validated_attention_economic_opportunity import FRESH_EVAL_DAYS
from .hierarchical_attention_runtime import add_stage2_scores, fit_stage2
from .hot_economic_opportunity import (
    ENTRY_FEATURES,
    _first_hot_feature_rows,
    _safe_spearman,
)
from .recurrent_wait_entry_action_value import _base_return

REQUEST_ID = 231
WATCH_MINUTES = (1, 2, 3, 4, 5)
EVENT_HORIZONS = (1, 2, 3, 5)
MODEL_SEED_REG = 20261231
MODEL_SEED_CLS = 20261301
MIN_FIT_ROWS = 1500
MIN_CAL_ROWS = 1500

MIN_TARGET_COVERAGE = 0.90
MIN_VALUE_SPEARMAN = 0.10
MIN_POSITIVE_AUC = 0.60
MIN_SPEARMAN_GAIN_VS_RUNNER = 0.03
MIN_AUC_GAIN_VS_RUNNER = 0.03
MIN_SELECTED_RATE = 0.10
MAX_SELECTED_RATE = 0.40
MIN_SELECTED_POSITIVE_RATE_GAIN = 0.10
MIN_POSITIVE_SPEARMAN_DAYS = 4
MIN_AUC_ABOVE_RANDOM_DAYS = 4
MIN_POSITIVE_SELECTED_MEAN_DAYS = 4


@dataclass(frozen=True)
class TradabilityModel:
    regressor: HistGradientBoostingRegressor
    classifier: HistGradientBoostingClassifier
    columns: tuple[str, ...]
    offset: float
    probability_gate: float
    winsor_low: float
    winsor_high: float


def _price_lookup(
    scan: pd.DataFrame,
) -> dict[tuple[str, str, int], float]:
    required = {"trading_day", "ticker", "t", "o"}
    missing = required - set(scan.columns)
    if missing:
        raise ValueError(
            f"request 231 scan missing columns: {sorted(missing)}"
        )
    result: dict[tuple[str, str, int], float] = {}
    work = scan.loc[
        :, ["trading_day", "ticker", "t", "o"]
    ].copy()
    work["o"] = pd.to_numeric(work["o"], errors="coerce")
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work = work.loc[
        work["o"].gt(0) & work["t"].notna()
    ]
    for row in work.itertuples(index=False):
        result[
            (
                str(row.trading_day),
                str(row.ticker).upper(),
                int(row.t),
            )
        ] = float(row.o)
    return result


def attach_full_hot_tradability(
    first_hot: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    """Attach persistent multi-event tradability to full first-HOT rows.

    Exact-minute missing references remain unresolved. There is no cash-zero
    imputation and no use of the best future price.
    """
    result = first_hot.copy()
    lookup = _price_lookup(scan)

    for minute in WATCH_MINUTES:
        result[
            f"watch_{minute}m_multi_event_value_pct"
        ] = np.nan
    result["persistent_tradability_pct"] = np.nan
    result["persistent_pair_start_minute"] = np.nan
    result["tradability_evaluable"] = False

    for index, row in result.iterrows():
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        hot_t = int(row["t"])

        values: dict[int, float] = {}
        for minute in WATCH_MINUTES:
            entry_t = hot_t + minute * MINUTE_MS
            entry_open = lookup.get(
                (day, ticker, entry_t)
            )
            if entry_open is None:
                continue

            horizon_values: list[float] = []
            complete = True
            for horizon in EVENT_HORIZONS:
                exit_t = (
                    hot_t
                    + (minute + horizon) * MINUTE_MS
                )
                exit_open = lookup.get(
                    (day, ticker, exit_t)
                )
                if exit_open is None:
                    complete = False
                    break
                value = _base_return(
                    entry_open,
                    exit_open,
                )
                if not np.isfinite(value):
                    complete = False
                    break
                horizon_values.append(float(value))

            if not complete:
                continue
            multi = float(np.mean(horizon_values))
            values[minute] = multi
            result.at[
                index,
                f"watch_{minute}m_multi_event_value_pct",
            ] = multi

        pairs: list[tuple[int, float]] = []
        for minute in WATCH_MINUTES[:-1]:
            if (
                minute in values
                and minute + 1 in values
            ):
                pairs.append(
                    (
                        minute,
                        float(
                            (
                                values[minute]
                                + values[minute + 1]
                            )
                            / 2.0
                        ),
                    )
                )

        if not pairs:
            continue
        best_minute, best_value = max(
            pairs,
            key=lambda item: item[1],
        )
        result.at[
            index,
            "persistent_tradability_pct",
        ] = float(best_value)
        result.at[
            index,
            "persistent_pair_start_minute",
        ] = int(best_minute)
        result.at[
            index,
            "tradability_evaluable",
        ] = True

    return result


def _usable_columns(
    frame: pd.DataFrame,
) -> tuple[str, ...]:
    return tuple(
        column
        for column in ENTRY_FEATURES
        if column in frame.columns
        and pd.to_numeric(
            frame[column],
            errors="coerce",
        ).notna().any()
    )


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(
        columns=columns
    ).apply(
        pd.to_numeric,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def train_tradability(
    frame: pd.DataFrame,
) -> TradabilityModel:
    day = frame["trading_day"].astype(str)
    fit = frame.loc[
        day.isin(EXTENDED_FIT_DAYS)
    ].copy()
    calibration = frame.loc[
        day.isin(EXTENDED_CAL_DAYS)
    ].copy()

    fit_target = pd.to_numeric(
        fit["persistent_tradability_pct"],
        errors="coerce",
    )
    cal_target = pd.to_numeric(
        calibration["persistent_tradability_pct"],
        errors="coerce",
    )
    fit = fit.loc[fit_target.notna()].copy()
    calibration = calibration.loc[
        cal_target.notna()
    ].copy()
    fit_target = pd.to_numeric(
        fit["persistent_tradability_pct"],
        errors="coerce",
    )
    cal_target = pd.to_numeric(
        calibration["persistent_tradability_pct"],
        errors="coerce",
    )

    if len(fit) < MIN_FIT_ROWS:
        raise ValueError(
            f"request 231 fit needs >= {MIN_FIT_ROWS} labels, "
            f"got {len(fit)}"
        )
    if len(calibration) < MIN_CAL_ROWS:
        raise ValueError(
            f"request 231 calibration needs >= {MIN_CAL_ROWS} labels, "
            f"got {len(calibration)}"
        )

    columns = _usable_columns(fit)
    x_fit = _feature_frame(fit, columns)
    y_fit = fit_target.to_numpy(dtype=float)
    low, high = np.quantile(
        y_fit,
        [0.005, 0.995],
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
            "request 231 fit target has one class"
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
    classifier.fit(
        x_fit,
        y_class,
    )

    x_cal = _feature_frame(
        calibration,
        columns,
    )
    raw_value = regressor.predict(x_cal)
    offset = float(
        np.mean(
            cal_target.to_numpy(dtype=float)
            - raw_value
        )
    )
    cal_probability = classifier.predict_proba(
        x_cal
    )[:, 1]
    probability_gate = float(
        np.quantile(
            cal_probability,
            0.75,
        )
    )

    return TradabilityModel(
        regressor=regressor,
        classifier=classifier,
        columns=columns,
        offset=offset,
        probability_gate=probability_gate,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def score_tradability(
    frame: pd.DataFrame,
    fitted: TradabilityModel,
) -> pd.DataFrame:
    result = frame.copy()
    x = _feature_frame(
        result,
        fitted.columns,
    )
    result[
        "predicted_persistent_tradability_pct"
    ] = (
        fitted.regressor.predict(x)
        + fitted.offset
    )
    result[
        "tradability_positive_probability"
    ] = fitted.classifier.predict_proba(
        x
    )[:, 1]
    result["tradability_selected"] = result[
        "tradability_positive_probability"
    ].ge(fitted.probability_gate)
    return result


def _safe_auc(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    a = pd.to_numeric(
        actual,
        errors="coerce",
    )
    s = pd.to_numeric(
        score,
        errors="coerce",
    )
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


def evaluate_fresh(
    frame: pd.DataFrame,
    fitted: TradabilityModel,
) -> dict[str, object]:
    fresh = frame.loc[
        frame["trading_day"]
        .astype(str)
        .isin(FRESH_EVAL_DAYS)
    ].copy()
    target = pd.to_numeric(
        fresh["persistent_tradability_pct"],
        errors="coerce",
    )
    valid = target.notna()
    evaluable = fresh.loc[valid].copy()
    target = pd.to_numeric(
        evaluable["persistent_tradability_pct"],
        errors="coerce",
    )

    coverage = (
        float(valid.mean())
        if len(fresh)
        else None
    )
    candidate_spearman = _safe_spearman(
        target,
        evaluable[
            "predicted_persistent_tradability_pct"
        ],
    )
    candidate_auc = _safe_auc(
        target,
        evaluable[
            "tradability_positive_probability"
        ],
    )

    runner_score_column = (
        "second_rerank_probability"
        if "second_rerank_probability"
        in evaluable.columns
        else "minute_rerank_probability"
    )
    runner_score = pd.to_numeric(
        evaluable[runner_score_column],
        errors="coerce",
    )
    runner_spearman = _safe_spearman(
        target,
        runner_score,
    )
    runner_auc = _safe_auc(
        target,
        runner_score,
    )

    selected = evaluable.loc[
        evaluable[
            "tradability_selected"
        ].fillna(False).astype(bool)
    ].copy()
    selected_target = pd.to_numeric(
        selected["persistent_tradability_pct"],
        errors="coerce",
    )

    population_mean = (
        float(target.mean())
        if len(target)
        else None
    )
    population_positive = (
        float(target.gt(0).mean())
        if len(target)
        else None
    )
    selected_mean = (
        float(selected_target.mean())
        if len(selected)
        else None
    )
    selected_positive = (
        float(selected_target.gt(0).mean())
        if len(selected)
        else None
    )
    selected_rate = (
        float(len(selected) / len(evaluable))
        if len(evaluable)
        else None
    )

    by_day: dict[str, object] = {}
    positive_spearman_days = 0
    auc_above_random_days = 0
    positive_selected_mean_days = 0

    for day in FRESH_EVAL_DAYS:
        part = evaluable.loc[
            evaluable["trading_day"]
            .astype(str)
            .eq(day)
        ].copy()
        day_target = pd.to_numeric(
            part["persistent_tradability_pct"],
            errors="coerce",
        )
        day_s = _safe_spearman(
            day_target,
            part[
                "predicted_persistent_tradability_pct"
            ],
        )
        day_auc = _safe_auc(
            day_target,
            part[
                "tradability_positive_probability"
            ],
        )
        chosen = part.loc[
            part["tradability_selected"]
            .fillna(False)
            .astype(bool)
        ]
        chosen_target = pd.to_numeric(
            chosen[
                "persistent_tradability_pct"
            ],
            errors="coerce",
        )
        chosen_mean = (
            float(chosen_target.mean())
            if len(chosen)
            else None
        )

        positive_spearman_days += int(
            day_s is not None and day_s > 0
        )
        auc_above_random_days += int(
            day_auc is not None and day_auc > 0.5
        )
        positive_selected_mean_days += int(
            chosen_mean is not None
            and chosen_mean > 0
        )

        source = fresh.loc[
            fresh["trading_day"]
            .astype(str)
            .eq(day)
        ]
        source_target = pd.to_numeric(
            source[
                "persistent_tradability_pct"
            ],
            errors="coerce",
        )
        by_day[day] = {
            "first_hot_rows": int(len(source)),
            "evaluable_rows": int(len(part)),
            "coverage": (
                float(source_target.notna().mean())
                if len(source)
                else None
            ),
            "population_mean_pct": (
                float(day_target.mean())
                if len(part)
                else None
            ),
            "population_positive_rate": (
                float(day_target.gt(0).mean())
                if len(part)
                else None
            ),
            "spearman": day_s,
            "auc": day_auc,
            "selected_rows": int(len(chosen)),
            "selected_rate": (
                float(len(chosen) / len(part))
                if len(part)
                else None
            ),
            "selected_mean_pct": chosen_mean,
            "selected_positive_rate": (
                float(chosen_target.gt(0).mean())
                if len(chosen)
                else None
            ),
        }

    spearman_gain = (
        float(candidate_spearman)
        - float(runner_spearman)
        if candidate_spearman is not None
        and runner_spearman is not None
        else None
    )
    auc_gain = (
        float(candidate_auc)
        - float(runner_auc)
        if candidate_auc is not None
        and runner_auc is not None
        else None
    )
    positive_rate_gain = (
        float(selected_positive)
        - float(population_positive)
        if selected_positive is not None
        and population_positive is not None
        else None
    )

    gate = bool(
        coverage is not None
        and coverage >= MIN_TARGET_COVERAGE
        and candidate_spearman is not None
        and candidate_spearman >= MIN_VALUE_SPEARMAN
        and candidate_auc is not None
        and candidate_auc >= MIN_POSITIVE_AUC
        and spearman_gain is not None
        and spearman_gain
        >= MIN_SPEARMAN_GAIN_VS_RUNNER
        and auc_gain is not None
        and auc_gain >= MIN_AUC_GAIN_VS_RUNNER
        and selected_rate is not None
        and MIN_SELECTED_RATE
        <= selected_rate
        <= MAX_SELECTED_RATE
        and selected_mean is not None
        and selected_mean > 0
        and positive_rate_gain is not None
        and positive_rate_gain
        >= MIN_SELECTED_POSITIVE_RATE_GAIN
        and positive_spearman_days
        >= MIN_POSITIVE_SPEARMAN_DAYS
        and auc_above_random_days
        >= MIN_AUC_ABOVE_RANDOM_DAYS
        and positive_selected_mean_days
        >= MIN_POSITIVE_SELECTED_MEAN_DAYS
    )

    return {
        "rows": int(len(fresh)),
        "evaluable_rows": int(len(evaluable)),
        "coverage": coverage,
        "runner_score_column": runner_score_column,
        "runner_spearman": runner_spearman,
        "runner_auc": runner_auc,
        "candidate_spearman": candidate_spearman,
        "candidate_auc": candidate_auc,
        "spearman_gain_vs_runner": spearman_gain,
        "auc_gain_vs_runner": auc_gain,
        "population_mean_pct": population_mean,
        "population_positive_rate": population_positive,
        "selected_rows": int(len(selected)),
        "selected_rate": selected_rate,
        "selected_mean_pct": selected_mean,
        "selected_positive_rate": selected_positive,
        "selected_positive_rate_gain": positive_rate_gain,
        "positive_spearman_days": positive_spearman_days,
        "auc_above_random_days": auc_above_random_days,
        "positive_selected_mean_days": (
            positive_selected_mean_days
        ),
        "by_day": by_day,
        "development_gate_pass": gate,
    }


def evaluate(
    stage2_candidates_path: Path,
    opportunity_candidates_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    stage2_candidates = pd.read_parquet(
        stage2_candidates_path
    )
    opportunity_candidates = pd.read_parquet(
        opportunity_candidates_path
    )
    opportunity_scan = pd.read_parquet(
        opportunity_scan_path
    )

    minute_model, second_model = fit_stage2(
        stage2_candidates
    )
    scored = add_stage2_scores(
        opportunity_candidates,
        minute_model,
        second_model,
    )
    first_hot, runtime_audit = _first_hot_feature_rows(
        scored,
        opportunity_scan,
    )
    first_hot = attach_full_hot_tradability(
        first_hot,
        opportunity_scan,
    )

    model = train_tradability(first_hot)
    scored_hot = score_tradability(
        first_hot,
        model,
    )
    fresh = evaluate_fresh(
        scored_hot,
        model,
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request163_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "population": (
            "full validated-attention first-HOT population before "
            "downstream BUY selection"
        ),
        "architecture": (
            "runner discovery remains frozen; separate first-HOT "
            "tradability head predicts persistent usable entry windows"
        ),
        "target": {
            "watch_minutes": list(WATCH_MINUTES),
            "event_horizons": list(EVENT_HORIZONS),
            "checkpoint_value": (
                "equal-weight BASE net return at exact +1/+2/+3/+5 "
                "minute references after hypothetical WATCH entry"
            ),
            "persistent_value": (
                "best mean checkpoint value across any two consecutive "
                "WATCH entry minutes"
            ),
            "single_hindsight_spike_sufficient": False,
            "missing_exact_minutes_zero_filled": False,
            "future_information_used_as_input": False,
        },
        "development_split": {
            "fit_days": EXTENDED_FIT_DAYS,
            "calibration_days": EXTENDED_CAL_DAYS,
            "fresh_eval_days": FRESH_EVAL_DAYS,
        },
        "features": list(model.columns),
        "feature_count": len(model.columns),
        "calibration_probability_gate": (
            model.probability_gate
        ),
        "runtime_audit": runtime_audit,
        "support": {
            "first_hot_rows": int(len(scored_hot)),
            "fit_labeled_rows": int(
                scored_hot.loc[
                    scored_hot["trading_day"]
                    .astype(str)
                    .isin(EXTENDED_FIT_DAYS),
                    "persistent_tradability_pct",
                ].notna().sum()
            ),
            "calibration_labeled_rows": int(
                scored_hot.loc[
                    scored_hot["trading_day"]
                    .astype(str)
                    .isin(EXTENDED_CAL_DAYS),
                    "persistent_tradability_pct",
                ].notna().sum()
            ),
        },
        "fresh": fresh,
        "frozen_gate": {
            "min_target_coverage": MIN_TARGET_COVERAGE,
            "min_value_spearman": MIN_VALUE_SPEARMAN,
            "min_positive_auc": MIN_POSITIVE_AUC,
            "min_spearman_gain_vs_runner": (
                MIN_SPEARMAN_GAIN_VS_RUNNER
            ),
            "min_auc_gain_vs_runner": (
                MIN_AUC_GAIN_VS_RUNNER
            ),
            "min_selected_rate": MIN_SELECTED_RATE,
            "max_selected_rate": MAX_SELECTED_RATE,
            "selected_mean_must_be_positive": True,
            "min_selected_positive_rate_gain": (
                MIN_SELECTED_POSITIVE_RATE_GAIN
            ),
            "min_positive_spearman_days": (
                MIN_POSITIVE_SPEARMAN_DAYS
            ),
            "min_auc_above_random_days": (
                MIN_AUC_ABOVE_RANDOM_DAYS
            ),
            "min_positive_selected_mean_days": (
                MIN_POSITIVE_SELECTED_MEAN_DAYS
            ),
        },
        "development_gate_pass": bool(
            fresh["development_gate_pass"]
        ),
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze the full-HOT tradability head; train a new "
            "ENTER-vs-WAIT timing model only inside tradability-selected "
            "episodes, then replay the complete recurrent path."
        ),
        "next_boundary_if_fail": (
            "full first-HOT causal features still cannot separate "
            "persistent tradability; runner discovery and tradability "
            "need a different shared representation or additional raw "
            "market information rather than more downstream gating."
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    rows_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    scored_hot.to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2-candidates",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--opportunity-candidates",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--opportunity-scan",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.stage2_candidates,
        args.opportunity_candidates,
        args.opportunity_scan,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
