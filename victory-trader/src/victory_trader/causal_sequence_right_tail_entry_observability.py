"""Request 196: causal sequence right-tail entry observability.

Tests whether exact prior-five-minute causal state sequences improve ranking of
shadow entries that retain a large cost-adjusted winner within the inherited
30-minute research cap. Fresh top-quartile selection is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .attention_replay import MINUTE_MS
from .decomposed_cost_aware_entry_surplus import FRESH_DAYS
from .future_cost_cover_state_observability import _day_weights
from .market_regime_position_value import (
    attach_market_regime,
    build_market_regime,
)
from .shadow_entry_repricing_decomposition import (
    EPISODE_KEYS,
    FEATURES as BASE_FEATURES,
    add_shadow_features,
    build_labels,
)

REQUEST_ID = 196
PRIMARY_THRESHOLD_PCT = 3.0
SECONDARY_THRESHOLD_PCT = 5.0
BASELINE_SEED = 20261100
SEQUENCE_SEED = 20261101
SECONDARY_SEED = 20261102
LAGS = (1, 2, 3, 4, 5)

SEQUENCE_SOURCE_FEATURES = (
    "attention_score",
    "attention_rank",
    "minute_body_return_pct",
    "minute_range_pct",
    "log_minute_volume",
    "log_minute_transactions",
    "entry_to_current_close_pct",
    "running_max_return_pct",
    "running_min_return_pct",
    "drawdown_from_peak_pct",
    "recovery_from_trough_pct",
    "sec_last5_return_pct",
    "sec_last10_return_pct",
    "sec_last15_return_pct",
    "sec_accel_5_pct",
    "sec_accel_10_pct",
    "sec_accel_15_pct",
    "sec_realized_vol_pct",
    "sec_return_efficiency_60",
    "sec_sign_flip_rate_60",
    "sec_close_location_60",
    "sec_close_vs_vwap_pct",
    "sec_volume_burst_10",
    "sec_transactions_burst_10",
)

MIN_SEQUENCE_AUC = 0.60
MIN_AUC_IMPROVEMENT = 0.03
MIN_POSITIVE_AUC_DAYS = 4
MIN_TOP_PREVALENCE_UPLIFT = 0.10
MIN_TOP_VALUE_UPLIFT_PCT = 0.50
MIN_POSITIVE_TOP_DAYS = 4


@dataclass(frozen=True)
class ClassifierModel:
    model: HistGradientBoostingClassifier
    columns: tuple[str, ...]


def add_exact_sequence_lags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["state_t"] = pd.to_numeric(
        result["state_t"],
        errors="coerce",
    )
    if result["state_t"].isna().any():
        raise ValueError(
            "request 196 requires finite causal state timestamps"
        )
    result = result.sort_values(
        EPISODE_KEYS + ["state_t"],
        kind="stable",
    ).copy()

    available_sources = [
        column
        for column in SEQUENCE_SOURCE_FEATURES
        if column in result.columns
    ]
    index_columns = [*EPISODE_KEYS, "state_t"]
    if result.duplicated(index_columns).any():
        raise ValueError(
            "request 196 requires unique episode/state timestamps"
        )

    source_index = pd.MultiIndex.from_frame(
        result[index_columns]
    )
    source_table = result.loc[
        :, available_sources
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )
    source_table.index = source_index

    blocks: list[pd.DataFrame] = [result]
    for lag in LAGS:
        target_keys = result[index_columns].copy()
        target_keys["state_t"] = (
            target_keys["state_t"]
            - lag * MINUTE_MS
        )
        target_index = pd.MultiIndex.from_frame(
            target_keys
        )
        lagged = source_table.reindex(
            target_index
        ).copy()
        lagged.index = result.index
        lagged.columns = [
            f"{column}_lag{lag}m"
            for column in lagged.columns
        ]
        blocks.append(lagged)

    return pd.concat(
        blocks,
        axis=1,
    ).sort_index()

def sequence_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    lagged = tuple(
        f"{column}_lag{lag}m"
        for lag in LAGS
        for column in SEQUENCE_SOURCE_FEATURES
        if f"{column}_lag{lag}m" in frame.columns
    )
    return tuple(
        dict.fromkeys([*BASE_FEATURES, *lagged])
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
    )


def _safe_auc(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    y = y.loc[valid].astype(int)
    if len(y) < 20 or y.nunique() < 2:
        return None
    return float(
        roc_auc_score(
            y,
            s.loc[valid].astype(float),
        )
    )


def _safe_ap(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    y = y.loc[valid].astype(int)
    if len(y) < 20 or int(y.sum()) == 0:
        return None
    return float(
        average_precision_score(
            y,
            s.loc[valid].astype(float),
        )
    )


def train_classifier(
    historical: pd.DataFrame,
    *,
    target_column: str,
    columns: tuple[str, ...],
    seed: int,
) -> ClassifierModel:
    target = pd.to_numeric(
        historical[target_column],
        errors="coerce",
    )
    valid = target.notna()
    train = historical.loc[valid].copy()
    y = target.loc[valid].astype(int)
    if len(train) < 800 or y.nunique() < 2:
        raise ValueError(
            f"request 196 insufficient support for {target_column}"
        )

    usable = tuple(
        column
        for column in columns
        if column in train.columns
        and pd.to_numeric(
            train[column],
            errors="coerce",
        ).notna().any()
    )
    if not usable:
        raise ValueError(
            "request 196 has no usable causal features"
        )

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _feature_frame(train, usable),
        y,
        sample_weight=_day_weights(train),
    )
    return ClassifierModel(
        model=model,
        columns=usable,
    )


def predict(
    frame: pd.DataFrame,
    fitted: ClassifierModel,
) -> np.ndarray:
    return fitted.model.predict_proba(
        _feature_frame(
            frame,
            fitted.columns,
        )
    )[:, 1]


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    value = pd.to_numeric(
        result["remaining_best_base_pct"],
        errors="coerce",
    )
    result["strong_winner_3"] = (
        value.ge(PRIMARY_THRESHOLD_PCT)
        .where(value.notna())
    )
    result["strong_winner_5"] = (
        value.ge(SECONDARY_THRESHOLD_PCT)
        .where(value.notna())
    )
    return result


def evaluate_scores(
    scored: pd.DataFrame,
) -> dict[str, object]:
    actual = pd.to_numeric(
        scored["strong_winner_3"],
        errors="coerce",
    )
    baseline = pd.to_numeric(
        scored["baseline_winner3_score"],
        errors="coerce",
    )
    sequence = pd.to_numeric(
        scored["sequence_winner3_score"],
        errors="coerce",
    )
    secondary_actual = pd.to_numeric(
        scored["strong_winner_5"],
        errors="coerce",
    )
    secondary_score = pd.to_numeric(
        scored["sequence_winner5_score"],
        errors="coerce",
    )
    value = pd.to_numeric(
        scored["remaining_best_base_pct"],
        errors="coerce",
    )

    valid = (
        actual.notna()
        & baseline.notna()
        & sequence.notna()
        & value.notna()
    )
    part = scored.loc[valid].copy()
    y = actual.loc[valid].astype(int)
    base_score = baseline.loc[valid]
    seq_score = sequence.loc[valid]
    values = value.loc[valid]

    baseline_auc = _safe_auc(y, base_score)
    sequence_auc = _safe_auc(y, seq_score)
    auc_improvement = (
        float(sequence_auc - baseline_auc)
        if sequence_auc is not None
        and baseline_auc is not None
        else None
    )

    by_day: dict[str, object] = {}
    positive_auc_days = 0
    for day in FRESH_DAYS:
        mask = (
            part["trading_day"]
            .astype(str)
            .eq(day)
        )
        day_part = part.loc[mask].copy()
        day_auc = _safe_auc(
            day_part["strong_winner_3"],
            day_part["sequence_winner3_score"],
        )
        if day_auc is not None and day_auc > 0.50:
            positive_auc_days += 1
        by_day[day] = {
            "rows": int(len(day_part)),
            "sequence_auc": day_auc,
        }

    cutoff = (
        float(seq_score.quantile(0.75))
        if len(seq_score)
        else np.nan
    )
    top_mask = seq_score.ge(cutoff)
    top = part.loc[top_mask].copy()
    top_values = pd.to_numeric(
        top["remaining_best_base_pct"],
        errors="coerce",
    )
    top_y = pd.to_numeric(
        top["strong_winner_3"],
        errors="coerce",
    )

    prevalence = (
        float(y.mean()) if len(y) else None
    )
    top_prevalence = (
        float(top_y.mean())
        if len(top_y)
        else None
    )
    all_mean = (
        float(values.mean())
        if len(values)
        else None
    )
    top_mean = (
        float(top_values.mean())
        if len(top_values)
        else None
    )

    top_by_day: dict[str, object] = {}
    positive_top_days = 0
    for day in FRESH_DAYS:
        day_top = top.loc[
            top["trading_day"]
            .astype(str)
            .eq(day)
        ]
        day_values = pd.to_numeric(
            day_top["remaining_best_base_pct"],
            errors="coerce",
        )
        mean = (
            float(day_values.mean())
            if len(day_values)
            else None
        )
        if mean is not None and mean > 0:
            positive_top_days += 1
        top_by_day[day] = {
            "rows": int(len(day_top)),
            "actual_base_mean_pct": mean,
            "strong_winner_3_rate": (
                float(
                    pd.to_numeric(
                        day_top["strong_winner_3"],
                        errors="coerce",
                    ).mean()
                )
                if len(day_top)
                else None
            ),
        }

    secondary_valid = (
        secondary_actual.notna()
        & secondary_score.notna()
    )

    return {
        "evaluable_rows": int(valid.sum()),
        "strong_winner_3_prevalence": prevalence,
        "baseline_auc": baseline_auc,
        "baseline_average_precision": _safe_ap(
            y,
            base_score,
        ),
        "sequence_auc": sequence_auc,
        "sequence_average_precision": _safe_ap(
            y,
            seq_score,
        ),
        "sequence_auc_improvement": auc_improvement,
        "positive_sequence_auc_days": int(
            positive_auc_days
        ),
        "by_day": by_day,
        "top_quartile": {
            "diagnostic_only": True,
            "fresh_score_cutoff": (
                cutoff
                if np.isfinite(cutoff)
                else None
            ),
            "rows": int(len(top)),
            "rate": (
                float(len(top) / len(part))
                if len(part)
                else None
            ),
            "strong_winner_3_rate": top_prevalence,
            "prevalence_uplift": (
                float(
                    top_prevalence
                    - prevalence
                )
                if top_prevalence is not None
                and prevalence is not None
                else None
            ),
            "actual_base_mean_pct": top_mean,
            "all_shadow_actual_base_mean_pct": all_mean,
            "value_uplift_pct": (
                float(top_mean - all_mean)
                if top_mean is not None
                and all_mean is not None
                else None
            ),
            "positive_mean_days": int(
                positive_top_days
            ),
            "by_day": top_by_day,
        },
        "secondary_strong_winner_5": {
            "prevalence": (
                float(
                    secondary_actual.loc[
                        secondary_valid
                    ].mean()
                )
                if int(
                    secondary_valid.sum()
                )
                else None
            ),
            "auc": _safe_auc(
                secondary_actual.loc[
                    secondary_valid
                ],
                secondary_score.loc[
                    secondary_valid
                ],
            ),
            "average_precision": _safe_ap(
                secondary_actual.loc[
                    secondary_valid
                ],
                secondary_score.loc[
                    secondary_valid
                ],
            ),
        },
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(
        fit_path
    )
    calibration_positions = pd.read_parquet(
        calibration_path
    )
    history_scan = pd.read_parquet(
        history_scan_path
    )

    historical_days = set(
        fit_positions["trading_day"]
        .astype(str)
    ) | set(
        calibration_positions[
            "trading_day"
        ].astype(str)
    )
    hist_scan = history_scan.loc[
        history_scan["trading_day"]
        .astype(str)
        .isin(historical_days)
    ].copy()
    hist_regime = build_market_regime(
        hist_scan
    )

    def prepare_positions(
        positions: pd.DataFrame,
        regime: pd.DataFrame,
        scan: pd.DataFrame,
    ) -> pd.DataFrame:
        prepared = attach_market_regime(
            add_shadow_features(
                positions
            ),
            regime,
        )
        prepared = add_exact_sequence_lags(
            prepared
        )
        return add_targets(
            build_labels(
                prepared,
                scan,
            )
        )

    fit = prepare_positions(
        fit_positions,
        hist_regime,
        hist_scan,
    )
    calibration = prepare_positions(
        calibration_positions,
        hist_regime,
        hist_scan,
    )
    historical = pd.concat(
        [fit, calibration],
        ignore_index=True,
    )

    seq_cols = sequence_columns(
        historical
    )
    baseline = train_classifier(
        historical,
        target_column="strong_winner_3",
        columns=BASE_FEATURES,
        seed=BASELINE_SEED,
    )
    sequence = train_classifier(
        historical,
        target_column="strong_winner_3",
        columns=seq_cols,
        seed=SEQUENCE_SEED,
    )
    sequence5 = train_classifier(
        historical,
        target_column="strong_winner_5",
        columns=seq_cols,
        seed=SECONDARY_SEED,
    )

    position_paths = sorted(
        fresh_dir.glob(
            "*-positions.parquet"
        )
    )
    scan_paths = sorted(
        fresh_dir.glob("*-scan.parquet")
    )
    if (
        len(position_paths)
        != len(FRESH_DAYS)
        or len(scan_paths)
        != len(FRESH_DAYS)
    ):
        raise ValueError(
            "request 196 requires complete request 178 shards"
        )

    fresh_positions = pd.concat(
        [
            pd.read_parquet(path)
            for path in position_paths
        ],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [
            pd.read_parquet(path)
            for path in scan_paths
        ],
        ignore_index=True,
    )
    found = sorted(
        fresh_positions[
            "trading_day"
        ].astype(str).unique()
    )
    if found != list(FRESH_DAYS):
        raise ValueError(
            f"request 196 expected {FRESH_DAYS}, found {found}"
        )
    fresh_regime = build_market_regime(
        fresh_scan
    )
    fresh = prepare_positions(
        fresh_positions,
        fresh_regime,
        fresh_scan,
    )

    fresh["baseline_winner3_score"] = predict(
        fresh,
        baseline,
    )
    fresh["sequence_winner3_score"] = predict(
        fresh,
        sequence,
    )
    fresh["sequence_winner5_score"] = predict(
        fresh,
        sequence5,
    )

    metrics = evaluate_scores(fresh)
    top = metrics["top_quartile"]

    gate = bool(
        metrics["sequence_auc"] is not None
        and float(
            metrics["sequence_auc"]
        )
        >= MIN_SEQUENCE_AUC
        and metrics[
            "sequence_auc_improvement"
        ]
        is not None
        and float(
            metrics[
                "sequence_auc_improvement"
            ]
        )
        >= MIN_AUC_IMPROVEMENT
        and int(
            metrics[
                "positive_sequence_auc_days"
            ]
        )
        >= MIN_POSITIVE_AUC_DAYS
        and top[
            "prevalence_uplift"
        ]
        is not None
        and float(
            top["prevalence_uplift"]
        )
        >= MIN_TOP_PREVALENCE_UPLIFT
        and top["value_uplift_pct"]
        is not None
        and float(
            top["value_uplift_pct"]
        )
        >= MIN_TOP_VALUE_UPLIFT_PCT
        and int(
            top["positive_mean_days"]
        )
        >= MIN_POSITIVE_TOP_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "observability_only": True,
        "primary_threshold_pct": PRIMARY_THRESHOLD_PCT,
        "secondary_threshold_pct": SECONDARY_THRESHOLD_PCT,
        "sequence_lags_minutes": list(LAGS),
        "metrics": metrics,
        "model": {
            "baseline_seed": BASELINE_SEED,
            "sequence_seed": SEQUENCE_SEED,
            "secondary_seed": SECONDARY_SEED,
            "baseline_columns": len(
                baseline.columns
            ),
            "sequence_columns": len(
                sequence.columns
            ),
        },
        "frozen_gate": {
            "min_sequence_auc": MIN_SEQUENCE_AUC,
            "min_auc_improvement": MIN_AUC_IMPROVEMENT,
            "min_positive_auc_days": MIN_POSITIVE_AUC_DAYS,
            "min_top_prevalence_uplift": MIN_TOP_PREVALENCE_UPLIFT,
            "min_top_value_uplift_pct": MIN_TOP_VALUE_UPLIFT_PCT,
            "min_positive_top_days": MIN_POSITIVE_TOP_DAYS,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    scored_output_path.parent.mkdir(
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
    fresh.to_parquet(
        scored_output_path,
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
        "--fit",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--history-scan",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fresh-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--scored-output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
