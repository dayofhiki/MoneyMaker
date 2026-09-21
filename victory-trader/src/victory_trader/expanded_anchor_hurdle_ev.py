from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .expanded_episode_viability_rank import (
    HORIZON,
    _evaluation_reason,
    _first_eligible_rows,
    _fit_calibration_days,
    read_model_panel,
)
from .state_action_risk import SEVERE_LOSS_PCT
from .state_action_value import _target_column
from .state_multi_source_value import _coverage_row, multi_source_action_feature_frame
from .state_rank_turn import day_cluster_bootstrap


PROBABILITY_THRESHOLD = 0.50
MIN_PROBABILITY_CALIBRATION = 100
MIN_MAGNITUDE_CALIBRATION = 75
MIN_MAGNITUDE_CALIBRATION_DAYS = 5
BOOTSTRAP_SAMPLES = 10_000
POLICIES = (
    "earliest_eligible_15m_cap1",
    "probability_half_15m_cap1",
    "hurdle_ev_15m_cap1",
)


@dataclass(frozen=True)
class ProbabilityModel:
    model: HistGradientBoostingClassifier
    feature_columns: tuple[str, ...]
    platt: LogisticRegression


@dataclass(frozen=True)
class MagnitudeModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    offset: float
    label: str


def load_fold(
    dataset_paths: dict[str, Path],
    evaluation_month: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if evaluation_month not in dataset_paths:
        raise ValueError(f"missing evaluation month: {evaluation_month}")

    fit_days, calibration_days = _fit_calibration_days(
        dataset_paths,
        evaluation_month,
    )
    fit_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
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
            fit_parts.append(_first_eligible_rows(fit_panel))

        cal_panel = panel.loc[day.isin(calibration_days)].copy()
        if not cal_panel.empty:
            calibration_parts.append(_first_eligible_rows(cal_panel))

        del panel, day, fit_panel, cal_panel
        gc.collect()

    fit = pd.concat(fit_parts, ignore_index=True)
    calibration = pd.concat(calibration_parts, ignore_index=True)
    evaluation = read_model_panel(dataset_paths[evaluation_month])
    evaluation_anchors = _first_eligible_rows(evaluation)

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
                "fit_anchors": len(fit),
                "calibration_anchors": len(calibration),
                "evaluation_anchors": len(evaluation_anchors),
            }
        ]
    )
    if (
        str(provenance.iloc[0]["calibration_max_day"])
        >= str(provenance.iloc[0]["evaluation_min_day"])
    ):
        raise ValueError("training/calibration overlaps evaluation")

    return fit, calibration, evaluation, provenance


def _usable_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = multi_source_action_feature_frame(frame)
    columns = [
        column for column in features.columns if features[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable multi-source features")
    return features, columns


def train_probability_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> ProbabilityModel:
    target_col = _target_column(HORIZON)

    fit_target = pd.to_numeric(fit[target_col], errors="coerce")
    fit_valid = fit_target.notna()
    fit_labeled = fit.loc[fit_valid].copy()
    y_fit = fit_target.loc[fit_valid].gt(0.0).astype(int)
    if fit_labeled.empty or y_fit.nunique() < 2:
        raise ValueError("probability fit set lacks both classes")

    x_fit, columns = _usable_features(fit_labeled)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=20261011,
    )
    model.fit(x_fit.loc[:, columns], y_fit)

    cal_target = pd.to_numeric(calibration[target_col], errors="coerce")
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    y_cal = cal_target.loc[cal_valid].gt(0.0).astype(int)
    if len(cal) < MIN_PROBABILITY_CALIBRATION:
        raise ValueError(
            f"fewer than {MIN_PROBABILITY_CALIBRATION} labeled calibration anchors"
        )
    if y_cal.nunique() < 2:
        raise ValueError("probability calibration set lacks both classes")

    raw = model.predict_proba(
        multi_source_action_feature_frame(cal).reindex(columns=columns)
    )[:, 1]
    raw = np.clip(raw.astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)

    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=20261011,
    )
    platt.fit(logits, y_cal)
    return ProbabilityModel(
        model=model,
        feature_columns=tuple(columns),
        platt=platt,
    )


def predict_probability(
    frame: pd.DataFrame,
    fitted: ProbabilityModel,
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


def _signed_magnitude(
    target: pd.Series,
    *,
    positive: bool,
) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(target, errors="coerce")
    if positive:
        mask = numeric.gt(0.0)
        magnitude = numeric.loc[mask].astype(float)
    else:
        mask = numeric.le(0.0)
        magnitude = -numeric.loc[mask].astype(float)
    return mask, magnitude


def train_magnitude_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    positive: bool,
) -> MagnitudeModel:
    target_col = _target_column(HORIZON)
    label = "win" if positive else "loss"

    fit_mask, fit_magnitude = _signed_magnitude(
        fit[target_col],
        positive=positive,
    )
    fit_rows = fit.loc[fit_mask].copy()
    if fit_rows.empty:
        raise ValueError(f"no {label} magnitude fit rows")

    lower = float(fit_magnitude.quantile(0.005))
    upper = float(fit_magnitude.quantile(0.995))
    y_fit = fit_magnitude.clip(lower=lower, upper=upper)

    x_fit, columns = _usable_features(fit_rows)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=20261012 if positive else 20261013,
    )
    model.fit(x_fit.loc[:, columns], y_fit)

    cal_mask, cal_magnitude = _signed_magnitude(
        calibration[target_col],
        positive=positive,
    )
    cal_rows = calibration.loc[cal_mask].copy()
    if len(cal_rows) < MIN_MAGNITUDE_CALIBRATION:
        raise ValueError(
            f"fewer than {MIN_MAGNITUDE_CALIBRATION} {label} calibration anchors"
        )
    if cal_rows["trading_day"].astype(str).nunique() < MIN_MAGNITUDE_CALIBRATION_DAYS:
        raise ValueError(
            f"fewer than {MIN_MAGNITUDE_CALIBRATION_DAYS} {label} calibration days"
        )

    raw_cal = model.predict(
        multi_source_action_feature_frame(cal_rows).reindex(columns=columns)
    ).astype(float)
    offset = float(np.mean(cal_magnitude.to_numpy(dtype=float) - raw_cal))

    return MagnitudeModel(
        model=model,
        feature_columns=tuple(columns),
        offset=offset,
        label=label,
    )


def predict_magnitude(
    frame: pd.DataFrame,
    fitted: MagnitudeModel,
) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    raw = fitted.model.predict(
        multi_source_action_feature_frame(frame).reindex(
            columns=fitted.feature_columns
        )
    ).astype(float)
    return np.maximum(0.0, raw + fitted.offset)


def hurdle_ev(
    probability: np.ndarray,
    win_magnitude: np.ndarray,
    loss_magnitude: np.ndarray,
) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    gain = np.asarray(win_magnitude, dtype=float)
    loss = np.asarray(loss_magnitude, dtype=float)
    return p * gain - (1.0 - p) * loss


def score_anchors(
    evaluation: pd.DataFrame,
    probability_model: ProbabilityModel,
    win_model: MagnitudeModel,
    loss_model: MagnitudeModel,
) -> pd.DataFrame:
    anchors = _first_eligible_rows(evaluation).copy()
    anchors["predicted_win_probability"] = predict_probability(
        anchors,
        probability_model,
    )
    anchors["predicted_win_magnitude_pct"] = predict_magnitude(
        anchors,
        win_model,
    )
    anchors["predicted_loss_magnitude_pct"] = predict_magnitude(
        anchors,
        loss_model,
    )
    anchors["predicted_hurdle_ev_pct"] = hurdle_ev(
        anchors["predicted_win_probability"].to_numpy(dtype=float),
        anchors["predicted_win_magnitude_pct"].to_numpy(dtype=float),
        anchors["predicted_loss_magnitude_pct"].to_numpy(dtype=float),
    )
    return anchors


def _attempt_row(
    row: pd.Series,
    *,
    policy: str,
) -> tuple[dict[str, object], dict[str, object] | None]:
    reason, flags = _evaluation_reason(row)
    attempt = {
        "policy": policy,
        "trading_day": row.get("trading_day"),
        "ticker": row.get("ticker"),
        "decision_t": row.get("t"),
        "action_horizon_min": HORIZON,
        "entry_price": row.get("entry_price", np.nan),
        "selected_win_probability": row.get(
            "predicted_win_probability", np.nan
        ),
        "selected_win_magnitude_pct": row.get(
            "predicted_win_magnitude_pct", np.nan
        ),
        "selected_loss_magnitude_pct": row.get(
            "predicted_loss_magnitude_pct", np.nan
        ),
        "selected_hurdle_ev_pct": row.get(
            "predicted_hurdle_ev_pct", np.nan
        ),
        "evaluation_reason": reason,
        **flags,
    }
    if reason != "evaluated":
        return attempt, None

    trade = row.to_dict()
    trade["policy"] = policy
    trade["action_horizon_min"] = HORIZON
    trade["selected_win_probability"] = attempt[
        "selected_win_probability"
    ]
    trade["selected_win_magnitude_pct"] = attempt[
        "selected_win_magnitude_pct"
    ]
    trade["selected_loss_magnitude_pct"] = attempt[
        "selected_loss_magnitude_pct"
    ]
    trade["selected_hurdle_ev_pct"] = attempt[
        "selected_hurdle_ev_pct"
    ]
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
    scored_anchors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attempts: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []

    for _, row in scored_anchors.iterrows():
        for policy, selected in (
            ("earliest_eligible_15m_cap1", True),
            (
                "probability_half_15m_cap1",
                float(row["predicted_win_probability"])
                >= PROBABILITY_THRESHOLD,
            ),
            (
                "hurdle_ev_15m_cap1",
                float(row["predicted_hurdle_ev_pct"]) > 0.0,
            ),
        ):
            if not selected:
                continue
            attempt, trade = _attempt_row(row, policy=policy)
            attempts.append(attempt)
            if trade is not None:
                trades.append(trade)

    path_rows = []
    for policy in POLICIES:
        rows = [row for row in attempts if row["policy"] == policy]
        path_rows.append(
            {
                "policy": policy,
                "anchors": int(len(scored_anchors)),
                "attempted": len(rows),
                "evaluated": sum(
                    row["evaluation_reason"] == "evaluated" for row in rows
                ),
                "unevaluable": sum(
                    row["evaluation_reason"] != "evaluated" for row in rows
                ),
            }
        )

    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        trade_frame = scored_anchors.iloc[0:0].copy()
        trade_frame["policy"] = pd.Series(dtype=str)
        trade_frame["action_horizon_min"] = pd.Series(dtype=int)
        trade_frame["selected_win_probability"] = pd.Series(dtype=float)
        trade_frame["selected_win_magnitude_pct"] = pd.Series(dtype=float)
        trade_frame["selected_loss_magnitude_pct"] = pd.Series(dtype=float)
        trade_frame["selected_hurdle_ev_pct"] = pd.Series(dtype=float)
        trade_frame["realized_gross_return_pct"] = pd.Series(dtype=float)
        trade_frame["realized_base_net_return_pct"] = pd.Series(dtype=float)
        trade_frame["realized_stress_net_return_pct"] = pd.Series(dtype=float)

    return trade_frame, pd.DataFrame(attempts), pd.DataFrame(path_rows)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    valid = pd.to_numeric(a, errors="coerce").notna() & pd.to_numeric(
        b, errors="coerce"
    ).notna()
    if int(valid.sum()) < 2:
        return np.nan
    return float(
        pd.to_numeric(a.loc[valid], errors="coerce").corr(
            pd.to_numeric(b.loc[valid], errors="coerce"),
            method="spearman",
        )
    )


def probability_diagnostics(
    scored: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    target = pd.to_numeric(scored[_target_column(HORIZON)], errors="coerce")
    valid = target.notna()
    y = target.loc[valid].gt(0.0).astype(int)
    p = pd.to_numeric(
        scored.loc[valid, "predicted_win_probability"],
        errors="coerce",
    ).to_numpy(dtype=float)

    return {
        "month": month,
        "anchor_rows": int(len(scored)),
        "labeled_anchor_rows": int(valid.sum()),
        "anchor_label_coverage": float(valid.mean()) if len(valid) else np.nan,
        "actual_positive_rate": float(y.mean()) if len(y) else np.nan,
        "probability_auc": float(roc_auc_score(y, p))
        if y.nunique() >= 2
        else np.nan,
        "probability_brier": float(brier_score_loss(y, p))
        if len(y)
        else np.nan,
        "probability_log_loss": float(log_loss(y, p, labels=[0, 1]))
        if len(y)
        else np.nan,
        "predicted_probability_mean": float(np.mean(p)) if len(p) else np.nan,
        "predicted_probability_half_rate": float(np.mean(p >= 0.50))
        if len(p)
        else np.nan,
    }


def magnitude_diagnostics(
    scored: pd.DataFrame,
    *,
    month: str,
    positive: bool,
) -> dict[str, object]:
    label = "win" if positive else "loss"
    target = pd.to_numeric(scored[_target_column(HORIZON)], errors="coerce")
    if positive:
        mask = target.gt(0.0)
        actual = target.loc[mask]
        prediction = scored.loc[mask, "predicted_win_magnitude_pct"]
    else:
        mask = target.le(0.0)
        actual = -target.loc[mask]
        prediction = scored.loc[mask, "predicted_loss_magnitude_pct"]

    return {
        "month": month,
        f"{label}_holdout_rows": int(mask.sum()),
        f"{label}_actual_magnitude_mean_pct": float(actual.mean())
        if len(actual)
        else np.nan,
        f"{label}_predicted_magnitude_mean_pct": float(
            pd.to_numeric(prediction, errors="coerce").mean()
        )
        if len(prediction)
        else np.nan,
        f"{label}_magnitude_spearman": _spearman(
            pd.Series(prediction),
            pd.Series(actual),
        ),
    }


def hurdle_diagnostics(
    scored: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    target = pd.to_numeric(scored[_target_column(HORIZON)], errors="coerce")
    ev = pd.to_numeric(scored["predicted_hurdle_ev_pct"], errors="coerce")
    valid = target.notna() & ev.notna()
    selected = valid & ev.gt(0.0)

    return {
        "month": month,
        "hurdle_ev_mean_pct": float(ev.loc[valid].mean())
        if valid.any()
        else np.nan,
        "hurdle_ev_selected_rate": float(selected.sum() / valid.sum())
        if valid.any()
        else np.nan,
        "hurdle_ev_spearman_vs_realized_base": _spearman(
            ev.loc[valid],
            target.loc[valid],
        ),
        "selected_predicted_hurdle_ev_mean_pct": float(
            ev.loc[selected].mean()
        )
        if selected.any()
        else np.nan,
        "selected_realized_base_mean_pct": float(
            target.loc[selected].mean()
        )
        if selected.any()
        else np.nan,
    }


def _metrics(
    trades: pd.DataFrame,
    *,
    month: str,
    policy: str,
) -> dict[str, object]:
    selected = trades.loc[trades["policy"].eq(policy)].copy()
    if selected.empty:
        return {"month": month, "policy": policy, "trades": 0}

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
        "trades": int(len(selected)),
        "days": int(selected["trading_day"].nunique()),
        "gross_mean_pct": float(gross.mean()),
        "base_mean_pct": float(base.mean()),
        "base_median_pct": float(base.median()),
        "base_positive_rate": float(base.gt(0.0).mean()),
        "base_p05_pct": float(base.quantile(0.05)),
        "actual_severe_loss_rate": float(
            base.le(SEVERE_LOSS_PCT).mean()
        ),
        "stress_mean_pct": float(stress.mean()),
        "day_balanced_base_mean_pct": float(daily.mean()),
        "worst_day_mean_pct": float(daily.min()),
        "selected_probability_mean": float(
            pd.to_numeric(
                selected["selected_win_probability"],
                errors="coerce",
            ).mean()
        ),
        "selected_hurdle_ev_mean_pct": float(
            pd.to_numeric(
                selected["selected_hurdle_ev_pct"],
                errors="coerce",
            ).mean()
        ),
    }


def _common_episode_comparison(
    trades: pd.DataFrame,
    *,
    month: str,
    comparator: str,
) -> dict[str, object]:
    primary = trades.loc[
        trades["policy"].eq("hurdle_ev_15m_cap1")
    ].copy()
    other = trades.loc[trades["policy"].eq(comparator)].copy()
    if primary.empty or other.empty:
        return {
            "month": month,
            "comparator": comparator,
            "common_episodes": 0,
        }

    keys = ["trading_day", "ticker"]
    columns = keys + ["realized_base_net_return_pct"]
    merged = other.loc[:, columns].merge(
        primary.loc[:, columns],
        on=keys,
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
    return {
        "month": month,
        "comparator": comparator,
        "common_episodes": int(len(merged)),
        "primary_minus_comparator_base_mean_pct": float(delta.mean()),
        "primary_better_rate": float(delta.gt(0.0).mean()),
    }


def calibration_diagnostics(
    calibration: pd.DataFrame,
    win_model: MagnitudeModel,
    loss_model: MagnitudeModel,
    *,
    month: str,
) -> dict[str, object]:
    target = pd.to_numeric(
        calibration[_target_column(HORIZON)],
        errors="coerce",
    )
    win_mask = target.gt(0.0)
    loss_mask = target.le(0.0)

    return {
        "month": month,
        "win_calibration_rows": int(win_mask.sum()),
        "win_calibration_days": int(
            calibration.loc[win_mask, "trading_day"].astype(str).nunique()
        ),
        "win_calibration_offset_pct": float(win_model.offset),
        "loss_calibration_rows": int(loss_mask.sum()),
        "loss_calibration_days": int(
            calibration.loc[loss_mask, "trading_day"].astype(str).nunique()
        ),
        "loss_calibration_offset_pct": float(loss_model.offset),
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
    dict[str, object],
]:
    fit, calibration, evaluation, provenance = load_fold(
        dataset_paths,
        evaluation_month,
    )
    print(
        "fold rows",
        {
            "fit_anchors": len(fit),
            "calibration_anchors": len(calibration),
            "evaluation_rows": len(evaluation),
        },
        flush=True,
    )

    probability_model = train_probability_model(fit, calibration)
    win_model = train_magnitude_model(
        fit,
        calibration,
        positive=True,
    )
    loss_model = train_magnitude_model(
        fit,
        calibration,
        positive=False,
    )

    scored = score_anchors(
        evaluation,
        probability_model,
        win_model,
        loss_model,
    )
    trades, attempts, paths = select_policies(scored)
    if not trades.empty:
        trades["month"] = evaluation_month
    if not attempts.empty:
        attempts["month"] = evaluation_month
    paths["month"] = evaluation_month

    diagnostics = pd.DataFrame(
        [
            {
                **probability_diagnostics(
                    scored,
                    month=evaluation_month,
                ),
                **magnitude_diagnostics(
                    scored,
                    month=evaluation_month,
                    positive=True,
                ),
                **magnitude_diagnostics(
                    scored,
                    month=evaluation_month,
                    positive=False,
                ),
                **hurdle_diagnostics(
                    scored,
                    month=evaluation_month,
                ),
                **calibration_diagnostics(
                    calibration,
                    win_model,
                    loss_model,
                    month=evaluation_month,
                ),
            }
        ]
    )

    details = pd.DataFrame(
        [
            _metrics(
                trades,
                month=evaluation_month,
                policy=policy,
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
                comparator="probability_half_15m_cap1",
            ),
        ]
    )
    coverage = pd.DataFrame(
        [_coverage_row(evaluation, evaluation_month)]
    )
    bootstrap = day_cluster_bootstrap(
        trades,
        policy="hurdle_ev_15m_cap1",
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
            "=== MoneyMaker Expanded-History Anchor Hurdle EV v2.4 ===",
            "anchor=first causal eligible state per ticker-day",
            "target=15m BASE-net return",
            "decomposition=P(win)*E[gain|win]-(1-P(win))*E[loss|loss]",
            "primary=hurdle EV > 0",
            "features=explicit multi-source causal feature frame for all heads",
            "evaluation=strict past-only",
            "NOTE=development only; April 2026+ sealed",
            "",
            "=== Date provenance ===",
            provenance.to_string(index=False),
            "",
            "=== Holdout and calibration diagnostics ===",
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
        prog="python -m victory_trader.expanded_anchor_hurdle_ev"
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
