from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .state_action_risk import SEVERE_LOSS_PCT
from .state_action_value import _sample_training, _target_column
from .state_entry_ranker import (
    EPISODE_KEYS,
    MIN_EPISODE_LABELS,
    episode_percentile_target,
)
from .state_multi_source_value import (
    EXTERNAL_FEATURES,
    _coverage_row,
    multi_source_action_feature_frame,
)
from .state_rank_turn import day_cluster_bootstrap
from .state_sequence_enrichment import SEQUENCE_FEATURES
from .state_value_model import (
    BOOLEAN_FEATURES,
    LOG_FEATURES,
    RAW_FEATURES,
    _eligible,
)


HORIZON = 15
VIABILITY_THRESHOLD = 0.50
RANK_GATE_QUANTILE = 0.75
MIN_CALIBRATION_ANCHORS = 100
MIN_CALIBRATION_RANK_ROWS = 15
MIN_MONTH_TRADES = 15
BOOTSTRAP_SAMPLES = 10_000
POLICIES = (
    "earliest_eligible_15m_cap1",
    "earliest_viable_15m_cap1",
    "viability_rank_15m_cap1",
)

MODEL_INPUT_COLUMNS = tuple(
    dict.fromkeys(
        [
            *EPISODE_KEYS,
            "t",
            "c",
            "previous_close",
            "entry_price",
            "active_minute_fraction_15m",
            *RAW_FEATURES,
            *BOOLEAN_FEATURES,
            *LOG_FEATURES,
            *SEQUENCE_FEATURES,
            *EXTERNAL_FEATURES,
            f"buy_return_{HORIZON}m_pct",
            f"buy_return_{HORIZON}m_base_net_return_pct",
            f"buy_return_{HORIZON}m_stress_net_return_pct",
            "eight_k_query_complete",
            "short_interest_query_complete",
            "short_ratio_latest_prior",
            "short_interest_latest",
        ]
    )
)


@dataclass(frozen=True)
class ViabilityModel:
    model: HistGradientBoostingClassifier
    feature_columns: tuple[str, ...]
    platt: LogisticRegression


@dataclass(frozen=True)
class RankModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]


def read_model_panel(path: str | Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    columns = [column for column in MODEL_INPUT_COLUMNS if column in available]
    return pd.read_parquet(path, columns=columns)


def _first_eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[_eligible(frame)].copy()
    if eligible.empty:
        return eligible
    return (
        eligible.sort_values(EPISODE_KEYS + ["t"], kind="stable")
        .groupby(EPISODE_KEYS, sort=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _scoreable_days(
    dataset_paths: dict[str, Path],
    evaluation_month: str,
) -> list[str]:
    days: set[str] = set()
    for label in sorted(dataset_paths):
        if label >= evaluation_month:
            continue
        minimal = pd.read_parquet(
            dataset_paths[label],
            columns=["trading_day", "c", "active_minute_fraction_15m"],
        )
        eligible = minimal.loc[_eligible(minimal), "trading_day"].astype(str)
        days.update(eligible.unique().tolist())
        del minimal, eligible
        gc.collect()
    result = sorted(days)
    if len(result) < 10:
        raise ValueError("fewer than 10 strictly-prior scoreable days")
    return result


def _fit_calibration_days(
    dataset_paths: dict[str, Path],
    evaluation_month: str,
) -> tuple[set[str], set[str]]:
    days = _scoreable_days(dataset_paths, evaluation_month)
    cut = max(1, int(len(days) * 0.80))
    cut = min(cut, len(days) - 1)
    return set(days[:cut]), set(days[cut:])


def _rank_fit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[_eligible(frame)].copy()
    target_col = _target_column(HORIZON)
    labeled = eligible.loc[
        pd.to_numeric(eligible[target_col], errors="coerce").notna()
    ].copy()
    if labeled.empty:
        return labeled
    labeled["_episode_rank_target"] = episode_percentile_target(
        labeled,
        HORIZON,
    )
    labeled = labeled.loc[labeled["_episode_rank_target"].notna()].copy()
    return _sample_training(labeled).reset_index(drop=True)


def load_fold(
    dataset_paths: dict[str, Path],
    evaluation_month: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if evaluation_month not in dataset_paths:
        raise ValueError(f"missing evaluation month: {evaluation_month}")

    fit_days, calibration_days = _fit_calibration_days(
        dataset_paths,
        evaluation_month,
    )

    rank_fit_parts: list[pd.DataFrame] = []
    rank_cal_parts: list[pd.DataFrame] = []
    viability_fit_parts: list[pd.DataFrame] = []
    viability_cal_parts: list[pd.DataFrame] = []

    prior_months = [
        label for label in sorted(dataset_paths) if label < evaluation_month
    ]
    if not prior_months:
        raise ValueError("no strictly-prior training months")

    for label in prior_months:
        panel = read_model_panel(dataset_paths[label])
        day = panel["trading_day"].astype(str)

        fit_panel = panel.loc[day.isin(fit_days)].copy()
        if not fit_panel.empty:
            rank_fit_parts.append(_rank_fit_rows(fit_panel))
            viability_fit_parts.append(_first_eligible_rows(fit_panel))

        cal_panel = panel.loc[day.isin(calibration_days)].copy()
        if not cal_panel.empty:
            rank_cal_parts.append(
                cal_panel.loc[_eligible(cal_panel)].copy().reset_index(drop=True)
            )
            viability_cal_parts.append(_first_eligible_rows(cal_panel))

        del panel, day, fit_panel, cal_panel
        gc.collect()

    rank_fit = pd.concat(rank_fit_parts, ignore_index=True)
    rank_cal = pd.concat(rank_cal_parts, ignore_index=True)
    viability_fit = pd.concat(viability_fit_parts, ignore_index=True)
    viability_cal = pd.concat(viability_cal_parts, ignore_index=True)
    evaluation = read_model_panel(dataset_paths[evaluation_month])

    provenance = pd.DataFrame(
        [
            {
                "evaluation_month": evaluation_month,
                "fit_min_day": min(fit_days),
                "fit_max_day": max(fit_days),
                "calibration_min_day": min(calibration_days),
                "calibration_max_day": max(calibration_days),
                "evaluation_min_day": str(
                    evaluation["trading_day"].astype(str).min()
                ),
                "evaluation_max_day": str(
                    evaluation["trading_day"].astype(str).max()
                ),
                "fit_days": len(fit_days),
                "calibration_days": len(calibration_days),
                "training_months": ",".join(prior_months),
            }
        ]
    )
    if (
        str(provenance.iloc[0]["calibration_max_day"])
        >= str(provenance.iloc[0]["evaluation_min_day"])
    ):
        raise ValueError("training/calibration overlaps evaluation")

    return (
        rank_fit,
        rank_cal,
        viability_fit,
        viability_cal,
        evaluation,
        provenance,
    )


def train_viability_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> ViabilityModel:
    target_col = _target_column(HORIZON)
    y_fit_numeric = pd.to_numeric(fit[target_col], errors="coerce")
    fit_valid = y_fit_numeric.notna()
    fit_labeled = fit.loc[fit_valid].copy()
    y_fit = y_fit_numeric.loc[fit_valid].gt(0.0).astype(int)

    if fit_labeled.empty or y_fit.nunique() < 2:
        raise ValueError("viability fit set lacks both classes")

    x_fit_full = multi_source_action_feature_frame(fit_labeled)
    columns = [
        column for column in x_fit_full.columns
        if x_fit_full[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable viability features")

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=20261001,
    )
    model.fit(x_fit_full.loc[:, columns], y_fit)

    y_cal_numeric = pd.to_numeric(calibration[target_col], errors="coerce")
    cal_valid = y_cal_numeric.notna()
    cal = calibration.loc[cal_valid].copy()
    y_cal = y_cal_numeric.loc[cal_valid].gt(0.0).astype(int)
    if len(cal) < MIN_CALIBRATION_ANCHORS:
        raise ValueError(
            f"fewer than {MIN_CALIBRATION_ANCHORS} labeled calibration anchors"
        )
    if y_cal.nunique() < 2:
        raise ValueError("viability calibration set lacks both classes")

    raw = model.predict_proba(
        multi_source_action_feature_frame(cal).reindex(columns=columns)
    )[:, 1]
    raw = np.clip(raw.astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)

    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=20261002,
    )
    platt.fit(logits, y_cal)

    return ViabilityModel(
        model=model,
        feature_columns=tuple(columns),
        platt=platt,
    )


def predict_viability(
    frame: pd.DataFrame,
    fitted: ViabilityModel,
) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    raw = fitted.model.predict_proba(
        multi_source_action_feature_frame(frame).reindex(
            columns=fitted.feature_columns
        )
    )[:, 1]
    raw = np.clip(raw.astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    return fitted.platt.predict_proba(logits)[:, 1].astype(float)


def train_rank_model(fit: pd.DataFrame) -> RankModel:
    target = pd.to_numeric(fit["_episode_rank_target"], errors="coerce")
    valid = target.notna()
    labeled = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if labeled.empty:
        raise ValueError("no rank training labels")

    features = multi_source_action_feature_frame(labeled)
    columns = [
        column for column in features.columns if features[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable rank features")

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=2.0,
        random_state=20261035,
    )
    model.fit(features.loc[:, columns], y)
    return RankModel(model=model, feature_columns=tuple(columns))


def predict_rank(frame: pd.DataFrame, fitted: RankModel) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    features = multi_source_action_feature_frame(frame).reindex(
        columns=fitted.feature_columns
    )
    return fitted.model.predict(features).astype(float)


def fit_rank_gate(calibration: pd.DataFrame, fitted: RankModel) -> float:
    prediction = predict_rank(calibration, fitted)
    finite = prediction[np.isfinite(prediction)]
    if len(finite) < MIN_CALIBRATION_RANK_ROWS:
        raise ValueError("fewer than 15 finite calibration rank scores")
    return float(np.quantile(finite, RANK_GATE_QUANTILE))


def _evaluation_reason(row: pd.Series) -> tuple[str, dict[str, bool]]:
    entry = pd.to_numeric(
        pd.Series([row.get("entry_price", np.nan)]), errors="coerce"
    ).iloc[0]
    gross = pd.to_numeric(
        pd.Series([row.get(f"buy_return_{HORIZON}m_pct", np.nan)]),
        errors="coerce",
    ).iloc[0]
    base = pd.to_numeric(
        pd.Series([row.get(_target_column(HORIZON), np.nan)]),
        errors="coerce",
    ).iloc[0]
    stress = pd.to_numeric(
        pd.Series(
            [row.get(f"buy_return_{HORIZON}m_stress_net_return_pct", np.nan)]
        ),
        errors="coerce",
    ).iloc[0]

    flags = {
        "entry_missing": bool(pd.isna(entry)),
        "entry_invalid": bool(
            pd.notna(entry) and (not np.isfinite(entry) or float(entry) <= 0)
        ),
        "gross_label_missing": bool(pd.isna(gross)),
        "base_label_missing": bool(pd.isna(base)),
        "stress_label_missing": bool(pd.isna(stress)),
        "return_nonfinite": bool(
            any(
                pd.notna(value) and not np.isfinite(value)
                for value in (gross, base, stress)
            )
        ),
    }

    if flags["entry_missing"]:
        reason = "entry_missing"
    elif flags["entry_invalid"]:
        reason = "entry_invalid"
    elif flags["gross_label_missing"]:
        reason = "gross_label_missing"
    elif flags["base_label_missing"] or flags["stress_label_missing"]:
        reason = "scenario_label_missing"
    elif flags["return_nonfinite"]:
        reason = "return_nonfinite"
    else:
        reason = "evaluated"
    return reason, flags


def _attempt_row(
    row: pd.Series,
    *,
    policy: str,
    viability_probability: float,
    rank_score: float,
) -> tuple[dict[str, object], dict[str, object] | None]:
    reason, flags = _evaluation_reason(row)
    attempt = {
        "trading_day": row.get("trading_day"),
        "ticker": row.get("ticker"),
        "decision_t": row.get("t"),
        "action_horizon_min": HORIZON,
        "entry_price": row.get("entry_price", np.nan),
        "selected_viability_probability": viability_probability,
        "selected_predicted_rank_score": rank_score,
        "evaluation_reason": reason,
        **flags,
    }
    if reason != "evaluated":
        return attempt, None

    trade = row.to_dict()
    trade["policy"] = policy
    trade["action_horizon_min"] = HORIZON
    trade["selected_viability_probability"] = viability_probability
    trade["selected_predicted_rank_score"] = rank_score
    trade["realized_gross_return_pct"] = float(
        row[f"buy_return_{HORIZON}m_pct"]
    )
    trade["realized_base_net_return_pct"] = float(
        row[_target_column(HORIZON)]
    )
    trade["realized_stress_net_return_pct"] = float(
        row[f"buy_return_{HORIZON}m_stress_net_return_pct"]
    )
    return attempt, trade


def select_policies(
    evaluation: pd.DataFrame,
    viability_model: ViabilityModel,
    rank_model: RankModel,
    rank_gate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = evaluation.loc[_eligible(evaluation)].copy()
    scored = scored.sort_values(EPISODE_KEYS + ["t"], kind="stable")
    anchors = (
        scored.groupby(EPISODE_KEYS, sort=False, as_index=False)
        .head(1)
        .copy()
    )
    anchors["predicted_viability_probability"] = predict_viability(
        anchors,
        viability_model,
    )
    viable_keys = set(
        map(
            tuple,
            anchors.loc[
                anchors["predicted_viability_probability"].ge(
                    VIABILITY_THRESHOLD
                ),
                EPISODE_KEYS,
            ].astype(str).to_numpy(),
        )
    )

    scored["predicted_episode_rank_15m"] = predict_rank(scored, rank_model)
    viability_by_key = {
        (str(row["trading_day"]), str(row["ticker"])): float(
            row["predicted_viability_probability"]
        )
        for _, row in anchors.iterrows()
    }

    attempts: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []

    anchor_map = {
        (str(row["trading_day"]), str(row["ticker"])): row
        for _, row in anchors.iterrows()
    }

    for key, anchor in anchor_map.items():
        p = viability_by_key[key]

        attempt, trade = _attempt_row(
            anchor,
            policy="earliest_eligible_15m_cap1",
            viability_probability=p,
            rank_score=float(anchor.get("predicted_episode_rank_15m", np.nan)),
        )
        attempts.append(
            {"policy": "earliest_eligible_15m_cap1", **attempt}
        )
        if trade is not None:
            trades.append(trade)

        if key not in viable_keys:
            continue

        attempt, trade = _attempt_row(
            anchor,
            policy="earliest_viable_15m_cap1",
            viability_probability=p,
            rank_score=float(anchor.get("predicted_episode_rank_15m", np.nan)),
        )
        attempts.append(
            {"policy": "earliest_viable_15m_cap1", **attempt}
        )
        if trade is not None:
            trades.append(trade)

    grouped = scored.groupby(EPISODE_KEYS, sort=False)
    primary_armed = 0
    primary_signals = 0
    for keys, group in grouped:
        key = (str(keys[0]), str(keys[1]))
        if key not in viable_keys:
            continue
        primary_armed += 1
        candidates = group.loc[
            pd.to_numeric(
                group["predicted_episode_rank_15m"], errors="coerce"
            ).ge(rank_gate)
        ]
        if candidates.empty:
            continue
        row = candidates.sort_values("t", kind="stable").iloc[0]
        primary_signals += 1
        p = viability_by_key[key]
        attempt, trade = _attempt_row(
            row,
            policy="viability_rank_15m_cap1",
            viability_probability=p,
            rank_score=float(row["predicted_episode_rank_15m"]),
        )
        attempts.append(
            {"policy": "viability_rank_15m_cap1", **attempt}
        )
        if trade is not None:
            trades.append(trade)

    for policy in POLICIES:
        rows = [row for row in attempts if row["policy"] == policy]
        path_rows.append(
            {
                "policy": policy,
                "attempted": len(rows),
                "evaluated": sum(
                    row["evaluation_reason"] == "evaluated" for row in rows
                ),
                "unevaluable": sum(
                    row["evaluation_reason"] != "evaluated" for row in rows
                ),
                "armed_episodes": (
                    primary_armed
                    if policy == "viability_rank_15m_cap1"
                    else int(len(viable_keys))
                    if policy == "earliest_viable_15m_cap1"
                    else int(len(anchor_map))
                ),
                "rank_signals": (
                    primary_signals
                    if policy == "viability_rank_15m_cap1"
                    else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(trades),
        pd.DataFrame(attempts),
        pd.DataFrame(path_rows),
    )


def _metrics(
    trades: pd.DataFrame,
    *,
    month: str,
    policy: str,
    rank_gate: float,
) -> dict[str, object]:
    selected = trades.loc[trades["policy"].eq(policy)].copy()
    if selected.empty:
        return {
            "month": month,
            "policy": policy,
            "rank_gate": rank_gate,
            "trades": 0,
        }

    base = pd.to_numeric(
        selected["realized_base_net_return_pct"], errors="coerce"
    )
    gross = pd.to_numeric(
        selected["realized_gross_return_pct"], errors="coerce"
    )
    stress = pd.to_numeric(
        selected["realized_stress_net_return_pct"], errors="coerce"
    )
    daily = selected.assign(_base=base).groupby("trading_day")["_base"].mean()

    return {
        "month": month,
        "policy": policy,
        "rank_gate": rank_gate,
        "trades": int(len(selected)),
        "days": int(selected["trading_day"].nunique()),
        "gross_mean_pct": float(gross.mean()),
        "base_mean_pct": float(base.mean()),
        "base_median_pct": float(base.median()),
        "base_positive_rate": float(base.gt(0).mean()),
        "actual_severe_loss_rate": float(base.le(SEVERE_LOSS_PCT).mean()),
        "base_p05_pct": float(base.quantile(0.05)),
        "stress_mean_pct": float(stress.mean()),
        "day_balanced_base_mean_pct": float(daily.mean()),
        "worst_day_mean_pct": float(daily.min()),
        "median_entry_minute": float(
            pd.to_numeric(
                selected["minutes_since_10pct_cross"], errors="coerce"
            ).median()
        ),
        "viability_probability_mean": float(
            pd.to_numeric(
                selected["selected_viability_probability"], errors="coerce"
            ).mean()
        ),
        "rank_score_mean": float(
            pd.to_numeric(
                selected["selected_predicted_rank_score"], errors="coerce"
            ).mean()
        ),
    }


def viability_diagnostics(
    evaluation: pd.DataFrame,
    fitted: ViabilityModel,
    *,
    month: str,
) -> dict[str, object]:
    anchors = _first_eligible_rows(evaluation)
    target = pd.to_numeric(anchors[_target_column(HORIZON)], errors="coerce")
    valid = target.notna()
    labeled = anchors.loc[valid].copy()
    y = target.loc[valid].gt(0.0).astype(int)
    p = predict_viability(labeled, fitted)
    pred = p >= VIABILITY_THRESHOLD

    return {
        "month": month,
        "anchor_rows": int(len(anchors)),
        "labeled_anchor_rows": int(len(labeled)),
        "anchor_label_coverage": float(len(labeled) / len(anchors))
        if len(anchors)
        else np.nan,
        "actual_positive_rate": float(y.mean()) if len(y) else np.nan,
        "predicted_viable_rate": float(pred.mean()) if len(y) else np.nan,
        "viability_auc": float(roc_auc_score(y, p))
        if y.nunique() >= 2
        else np.nan,
        "viability_balanced_accuracy": float(
            balanced_accuracy_score(y, pred.astype(int))
        )
        if y.nunique() >= 2
        else np.nan,
        "viability_log_loss": float(log_loss(y, p, labels=[0, 1]))
        if len(y)
        else np.nan,
        "viability_brier": float(brier_score_loss(y, p))
        if len(y)
        else np.nan,
        "viability_probability_mean": float(np.mean(p))
        if len(p)
        else np.nan,
    }


def rank_diagnostics(
    evaluation: pd.DataFrame,
    fitted: RankModel,
    *,
    month: str,
) -> dict[str, object]:
    labeled = evaluation.loc[_eligible(evaluation)].copy()
    target = pd.to_numeric(
        labeled[_target_column(HORIZON)], errors="coerce"
    )
    labeled = labeled.loc[target.notna()].copy()
    labeled["_target_rank"] = episode_percentile_target(labeled, HORIZON)
    labeled = labeled.loc[labeled["_target_rank"].notna()].copy()
    if labeled.empty:
        return {"month": month, "rank_rows": 0}

    prediction = pd.Series(
        predict_rank(labeled, fitted),
        index=labeled.index,
        dtype=float,
    )
    target_rank = pd.to_numeric(labeled["_target_rank"], errors="coerce")
    episode_corrs: list[float] = []
    for _, group in labeled.assign(_prediction=prediction).groupby(
        EPISODE_KEYS,
        sort=False,
    ):
        if len(group) < MIN_EPISODE_LABELS:
            continue
        corr = group["_prediction"].corr(
            group["_target_rank"],
            method="spearman",
        )
        if pd.notna(corr):
            episode_corrs.append(float(corr))

    return {
        "month": month,
        "rank_rows": int(len(labeled)),
        "rank_episodes": int(
            labeled[EPISODE_KEYS].drop_duplicates().shape[0]
        ),
        "rank_global_spearman": float(
            prediction.corr(target_rank, method="spearman")
        ),
        "rank_episode_median_spearman": float(np.median(episode_corrs))
        if episode_corrs
        else np.nan,
        "rank_episode_positive_rate": float(
            np.mean(np.asarray(episode_corrs) > 0)
        )
        if episode_corrs
        else np.nan,
    }


def _common_episode_comparison(
    trades: pd.DataFrame,
    *,
    month: str,
    comparator: str,
) -> dict[str, object]:
    primary = trades.loc[
        trades["policy"].eq("viability_rank_15m_cap1")
    ].copy()
    other = trades.loc[trades["policy"].eq(comparator)].copy()
    if primary.empty or other.empty:
        return {
            "month": month,
            "comparator": comparator,
            "common_episodes": 0,
        }

    columns = EPISODE_KEYS + [
        "t",
        "realized_base_net_return_pct",
        "minutes_since_10pct_cross",
    ]
    merged = other.loc[:, columns].merge(
        primary.loc[:, columns],
        on=EPISODE_KEYS,
        suffixes=("_comparator", "_primary"),
    )
    if merged.empty:
        return {
            "month": month,
            "comparator": comparator,
            "common_episodes": 0,
        }

    delta = (
        merged["realized_base_net_return_pct_primary"]
        - merged["realized_base_net_return_pct_comparator"]
    )
    delay = (
        merged["minutes_since_10pct_cross_primary"]
        - merged["minutes_since_10pct_cross_comparator"]
    )
    return {
        "month": month,
        "comparator": comparator,
        "common_episodes": int(len(merged)),
        "primary_minus_comparator_base_mean_pct": float(delta.mean()),
        "primary_better_rate": float(delta.gt(0).mean()),
        "primary_later_rate": float(delay.gt(0).mean()),
        "median_entry_delay_minutes": float(delay.median()),
    }


def run_fold(
    dataset_paths: dict[str, Path],
    evaluation_month: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float | int | bool | str],
]:
    (
        rank_fit,
        rank_cal,
        viability_fit,
        viability_cal,
        evaluation,
        provenance,
    ) = load_fold(dataset_paths, evaluation_month)

    print(
        "fold rows",
        {
            "rank_fit": len(rank_fit),
            "rank_cal": len(rank_cal),
            "viability_fit": len(viability_fit),
            "viability_cal": len(viability_cal),
            "evaluation": len(evaluation),
        },
        flush=True,
    )

    viability_model = train_viability_model(
        viability_fit,
        viability_cal,
    )
    rank_model = train_rank_model(rank_fit)
    rank_gate = fit_rank_gate(rank_cal, rank_model)

    viability_diag = viability_diagnostics(
        evaluation,
        viability_model,
        month=evaluation_month,
    )
    rank_diag = rank_diagnostics(
        evaluation,
        rank_model,
        month=evaluation_month,
    )
    diagnostics = pd.DataFrame([{**viability_diag, **rank_diag}])

    trades, attempts, paths = select_policies(
        evaluation,
        viability_model,
        rank_model,
        rank_gate,
    )
    if not trades.empty:
        trades["month"] = evaluation_month
    if not attempts.empty:
        attempts["month"] = evaluation_month
    paths["month"] = evaluation_month

    details = pd.DataFrame(
        [
            _metrics(
                trades,
                month=evaluation_month,
                policy=policy,
                rank_gate=rank_gate,
            )
            for policy in POLICIES
        ]
    )
    comparisons = pd.DataFrame(
        [
            _common_episode_comparison(
                trades,
                month=evaluation_month,
                comparator="earliest_eligible_15m_cap1",
            ),
            _common_episode_comparison(
                trades,
                month=evaluation_month,
                comparator="earliest_viable_15m_cap1",
            ),
        ]
    )
    coverage = pd.DataFrame(
        [_coverage_row(evaluation, evaluation_month)]
    )
    bootstrap = day_cluster_bootstrap(
        trades,
        policy="viability_rank_15m_cap1",
        samples=BOOTSTRAP_SAMPLES,
    )

    return (
        details,
        trades,
        attempts,
        diagnostics,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    )


def render_report(
    details: pd.DataFrame,
    attempts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    comparisons: pd.DataFrame,
    coverage: pd.DataFrame,
    provenance: pd.DataFrame,
    bootstrap: dict[str, object],
) -> str:
    reason_summary = (
        attempts.groupby(["policy", "evaluation_reason"])
        .size()
        .rename("attempts")
        .reset_index()
        if not attempts.empty
        else pd.DataFrame()
    )
    return "\n".join(
        [
            "=== MoneyMaker Expanded-History Episode Viability + Timing Rank v2.3 ===",
            "episode_anchor=first eligible state",
            "viability_target=15m BASE-net return > 0",
            "viability_gate=chronological Platt calibrated probability >= 0.50",
            "timing_target=within-episode 15m BASE-net percentile rank",
            "rank_gate=calibration predicted-rank 75th percentile",
            "features=explicit multi-source causal feature frame",
            "evaluation=strict past-only",
            "NOTE=development only; April 2026+ sealed",
            "",
            "=== Date provenance ===",
            provenance.to_string(index=False),
            "",
            "=== Holdout diagnostics ===",
            diagnostics.to_string(index=False),
            "",
            "=== Month-by-month policies ===",
            details.to_string(index=False),
            "",
            "=== Attempt paths ===",
            reason_summary.to_string(index=False),
            "",
            "=== Common-episode comparisons ===",
            comparisons.to_string(index=False),
            "",
            "=== External-data coverage ===",
            coverage.to_string(index=False),
            "",
            "=== Primary pooled day bootstrap ===",
            str(bootstrap),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.expanded_episode_viability_rank"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
    )
    parser.add_argument("--evaluation-month", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--attempts-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--comparisons-csv", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--provenance-csv", type=Path, required=True)
    args = parser.parse_args()

    dataset_paths = dict(args.dataset)
    (
        details,
        trades,
        attempts,
        diagnostics,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    ) = run_fold(dataset_paths, args.evaluation_month)

    report = render_report(
        details,
        attempts,
        diagnostics,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    )
    print(report, flush=True)

    output_paths = (
        args.report,
        args.details_csv,
        args.trades_csv,
        args.attempts_csv,
        args.diagnostics_csv,
        args.comparisons_csv,
        args.coverage_csv,
        args.provenance_csv,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    attempts.to_csv(args.attempts_csv, index=False)
    diagnostics.to_csv(args.diagnostics_csv, index=False)
    comparisons.to_csv(args.comparisons_csv, index=False)
    coverage.to_csv(args.coverage_csv, index=False)
    provenance.to_csv(args.provenance_csv, index=False)

    bootstrap_path = args.report.with_name(
        args.report.stem + "-bootstrap.csv"
    )
    pd.DataFrame([bootstrap]).to_csv(bootstrap_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
