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
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .expanded_anchor_hurdle_ev import (
    MIN_MAGNITUDE_CALIBRATION,
    MIN_MAGNITUDE_CALIBRATION_DAYS,
    MIN_PROBABILITY_CALIBRATION,
    MagnitudeModel,
    ProbabilityModel,
    hurdle_ev,
    predict_magnitude,
    predict_probability,
    train_magnitude_model,
    train_probability_model,
)
from .expanded_episode_viability_rank import (
    HORIZON,
    _evaluation_reason,
    _first_eligible_rows,
    _fit_calibration_days,
)
from .expanded_execution_feasible_hurdle_ev import execution_feasible_mask
from .state_action_risk import SEVERE_LOSS_PCT
from .state_action_value import _target_column
from .state_multi_source_value import (
    _coverage_row,
    multi_source_action_feature_frame,
)
from .state_rank_turn import day_cluster_bootstrap


BOOTSTRAP_SAMPLES = 10_000
POLICIES = (
    "feasible_earliest_15m_cap1",
    "feasible_base_hurdle_ev_15m_cap1",
    "supply_hurdle_ev_15m_cap1",
)
PRIMARY = "supply_hurdle_ev_15m_cap1"

SUPPLY_FEATURES = (
    "supply_log_weighted_shares_prior",
    "supply_log_share_class_shares_prior",
    "supply_log_implied_market_cap_prior",
    "supply_weighted_share_turnover_5m",
    "supply_share_class_turnover_5m",
    "supply_market_cap_turnover_5m",
    "supply_share_class_to_weighted_ratio",
)


def supply_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = multi_source_action_feature_frame(frame).copy()

    weighted = pd.to_numeric(
        frame["supply_weighted_shares_outstanding_prior"],
        errors="coerce",
    )
    share_class = pd.to_numeric(
        frame["supply_share_class_shares_outstanding_prior"],
        errors="coerce",
    )
    previous_close = pd.to_numeric(
        frame["previous_close"],
        errors="coerce",
    )
    volume_5m = pd.to_numeric(frame["volume_5m"], errors="coerce")
    dollar_volume_5m = pd.to_numeric(
        frame["dollar_volume_5m"],
        errors="coerce",
    )

    implied_market_cap = weighted * previous_close

    result["supply_log_weighted_shares_prior"] = np.log(
        weighted.where(weighted > 0)
    )
    result["supply_log_share_class_shares_prior"] = np.log(
        share_class.where(share_class > 0)
    )
    result["supply_log_implied_market_cap_prior"] = np.log(
        implied_market_cap.where(implied_market_cap > 0)
    )
    result["supply_weighted_share_turnover_5m"] = (
        volume_5m / weighted.where(weighted > 0)
    )
    result["supply_share_class_turnover_5m"] = (
        volume_5m / share_class.where(share_class > 0)
    )
    result["supply_market_cap_turnover_5m"] = (
        dollar_volume_5m / implied_market_cap.where(implied_market_cap > 0)
    )
    result["supply_share_class_to_weighted_ratio"] = (
        share_class / weighted.where(weighted > 0)
    )
    return result


def _usable_supply_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    features = supply_feature_frame(frame)
    columns = [
        column for column in features.columns if features[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable supply-state features")
    return features, columns


def train_supply_probability_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> ProbabilityModel:
    target_col = _target_column(HORIZON)
    target = pd.to_numeric(fit[target_col], errors="coerce")
    valid = target.notna()
    labeled = fit.loc[valid].copy()
    y_fit = target.loc[valid].gt(0.0).astype(int)
    if labeled.empty or y_fit.nunique() < 2:
        raise ValueError("supply probability fit lacks both classes")

    x_fit, columns = _usable_supply_features(labeled)
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

    cal_target = pd.to_numeric(
        calibration[target_col], errors="coerce"
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    y_cal = cal_target.loc[cal_valid].gt(0.0).astype(int)
    if len(cal) < MIN_PROBABILITY_CALIBRATION:
        raise ValueError("insufficient supply probability calibration rows")
    if y_cal.nunique() < 2:
        raise ValueError("supply probability calibration lacks both classes")

    raw = model.predict_proba(
        supply_feature_frame(cal).reindex(columns=columns)
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


def predict_supply_probability(
    frame: pd.DataFrame,
    fitted: ProbabilityModel,
) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    raw = fitted.model.predict_proba(
        supply_feature_frame(frame).reindex(
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


def train_supply_magnitude_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    positive: bool,
) -> MagnitudeModel:
    target_col = _target_column(HORIZON)
    label = "win" if positive else "loss"

    fit_mask, fit_magnitude = _signed_magnitude(
        fit[target_col], positive=positive
    )
    fit_rows = fit.loc[fit_mask].copy()
    if fit_rows.empty:
        raise ValueError(f"no supply {label} magnitude fit rows")

    low = float(fit_magnitude.quantile(0.005))
    high = float(fit_magnitude.quantile(0.995))
    y_fit = fit_magnitude.clip(lower=low, upper=high)

    x_fit, columns = _usable_supply_features(fit_rows)
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
        calibration[target_col], positive=positive
    )
    cal_rows = calibration.loc[cal_mask].copy()
    if len(cal_rows) < MIN_MAGNITUDE_CALIBRATION:
        raise ValueError(
            f"insufficient supply {label} magnitude calibration rows"
        )
    if (
        cal_rows["trading_day"].astype(str).nunique()
        < MIN_MAGNITUDE_CALIBRATION_DAYS
    ):
        raise ValueError(
            f"insufficient supply {label} magnitude calibration days"
        )

    raw = model.predict(
        supply_feature_frame(cal_rows).reindex(columns=columns)
    ).astype(float)
    offset = float(
        np.mean(cal_magnitude.to_numpy(dtype=float) - raw)
    )
    return MagnitudeModel(
        model=model,
        feature_columns=tuple(columns),
        offset=offset,
        label=label,
    )


def predict_supply_magnitude(
    frame: pd.DataFrame,
    fitted: MagnitudeModel,
) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    raw = fitted.model.predict(
        supply_feature_frame(frame).reindex(
            columns=fitted.feature_columns
        )
    ).astype(float)
    return np.maximum(0.0, raw + fitted.offset)


def load_fold(
    dataset_paths: dict[str, Path],
    evaluation_month: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_days, calibration_days = _fit_calibration_days(
        dataset_paths,
        evaluation_month,
    )
    prior_months = [
        label for label in sorted(dataset_paths) if label < evaluation_month
    ]
    if not prior_months:
        raise ValueError("no strictly-prior training months")

    fit_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    for label in prior_months:
        panel = pd.read_parquet(dataset_paths[label])
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
    evaluation_all = _first_eligible_rows(
        pd.read_parquet(dataset_paths[evaluation_month])
    )

    pre_fit = len(fit)
    pre_cal = len(calibration)
    pre_eval = len(evaluation_all)

    fit = fit.loc[execution_feasible_mask(fit)].copy()
    calibration = calibration.loc[
        execution_feasible_mask(calibration)
    ].copy()
    evaluation = evaluation_all.loc[
        execution_feasible_mask(evaluation_all)
    ].copy()

    provenance = pd.DataFrame(
        [
            {
                "evaluation_month": evaluation_month,
                "fit_min_day": min(fit_days),
                "fit_max_day": max(fit_days),
                "calibration_min_day": min(calibration_days),
                "calibration_max_day": max(calibration_days),
                "evaluation_min_day": str(
                    evaluation_all["trading_day"].astype(str).min()
                ),
                "evaluation_max_day": str(
                    evaluation_all["trading_day"].astype(str).max()
                ),
                "fit_days": len(fit_days),
                "calibration_days": len(calibration_days),
                "training_months": ",".join(prior_months),
                "fit_anchors_before_gate": pre_fit,
                "fit_anchors_after_gate": len(fit),
                "calibration_anchors_before_gate": pre_cal,
                "calibration_anchors_after_gate": len(calibration),
                "evaluation_anchors_before_gate": pre_eval,
                "evaluation_anchors_after_gate": len(evaluation),
            }
        ]
    )
    if max(calibration_days) >= provenance.iloc[0]["evaluation_min_day"]:
        raise ValueError("calibration overlaps evaluation")
    return fit, calibration, evaluation, provenance


def score_models(
    evaluation: pd.DataFrame,
    base_probability: ProbabilityModel,
    base_win: MagnitudeModel,
    base_loss: MagnitudeModel,
    supply_probability: ProbabilityModel,
    supply_win: MagnitudeModel,
    supply_loss: MagnitudeModel,
) -> pd.DataFrame:
    scored = evaluation.copy()

    scored["base_win_probability"] = predict_probability(
        scored, base_probability
    )
    scored["base_win_magnitude_pct"] = predict_magnitude(
        scored, base_win
    )
    scored["base_loss_magnitude_pct"] = predict_magnitude(
        scored, base_loss
    )
    scored["base_hurdle_ev_pct"] = hurdle_ev(
        scored["base_win_probability"].to_numpy(dtype=float),
        scored["base_win_magnitude_pct"].to_numpy(dtype=float),
        scored["base_loss_magnitude_pct"].to_numpy(dtype=float),
    )

    scored["supply_win_probability"] = predict_supply_probability(
        scored, supply_probability
    )
    scored["supply_win_magnitude_pct"] = predict_supply_magnitude(
        scored, supply_win
    )
    scored["supply_loss_magnitude_pct"] = predict_supply_magnitude(
        scored, supply_loss
    )
    scored["supply_hurdle_ev_pct"] = hurdle_ev(
        scored["supply_win_probability"].to_numpy(dtype=float),
        scored["supply_win_magnitude_pct"].to_numpy(dtype=float),
        scored["supply_loss_magnitude_pct"].to_numpy(dtype=float),
    )
    return scored


def _attempt(
    row: pd.Series,
    *,
    policy: str,
    score_column: str | None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    reason, flags = _evaluation_reason(row)
    attempt = {
        "policy": policy,
        "trading_day": row.get("trading_day"),
        "ticker": row.get("ticker"),
        "decision_t": row.get("t"),
        "action_horizon_min": HORIZON,
        "entry_price": row.get("entry_price", np.nan),
        "selected_decision_score": (
            row.get(score_column, np.nan)
            if score_column is not None
            else np.nan
        ),
        "evaluation_reason": reason,
        **flags,
    }
    if reason != "evaluated":
        return attempt, None

    trade = row.to_dict()
    trade["policy"] = policy
    trade["action_horizon_min"] = HORIZON
    trade["selected_decision_score"] = attempt[
        "selected_decision_score"
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
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []

    for _, row in scored.iterrows():
        decisions = (
            ("feasible_earliest_15m_cap1", True, None),
            (
                "feasible_base_hurdle_ev_15m_cap1",
                float(row["base_hurdle_ev_pct"]) > 0.0,
                "base_hurdle_ev_pct",
            ),
            (
                "supply_hurdle_ev_15m_cap1",
                float(row["supply_hurdle_ev_pct"]) > 0.0,
                "supply_hurdle_ev_pct",
            ),
        )
        for policy, selected, score_column in decisions:
            if not selected:
                continue
            attempt, trade = _attempt(
                row,
                policy=policy,
                score_column=score_column,
            )
            attempts.append(attempt)
            if trade is not None:
                trades.append(trade)

    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        trade_frame = scored.iloc[0:0].copy()
        trade_frame["policy"] = pd.Series(dtype=str)
        trade_frame["action_horizon_min"] = pd.Series(dtype=int)
        trade_frame["selected_decision_score"] = pd.Series(dtype=float)
        trade_frame["realized_gross_return_pct"] = pd.Series(dtype=float)
        trade_frame["realized_base_net_return_pct"] = pd.Series(dtype=float)
        trade_frame["realized_stress_net_return_pct"] = pd.Series(dtype=float)
    return trade_frame, pd.DataFrame(attempts)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 2:
        return np.nan
    return float(x.loc[valid].corr(y.loc[valid], method="spearman"))


def _head_diagnostics(
    scored: pd.DataFrame,
    *,
    prefix: str,
) -> dict[str, object]:
    target = pd.to_numeric(scored[_target_column(HORIZON)], errors="coerce")
    valid = target.notna()
    y = target.loc[valid].gt(0.0).astype(int)
    probability = pd.to_numeric(
        scored.loc[valid, f"{prefix}_win_probability"],
        errors="coerce",
    )
    win = target.gt(0.0)
    loss = target.le(0.0)
    ev = pd.to_numeric(
        scored[f"{prefix}_hurdle_ev_pct"],
        errors="coerce",
    )
    selected = valid & ev.gt(0.0)

    return {
        f"{prefix}_probability_auc": (
            float(roc_auc_score(y, probability))
            if y.nunique() >= 2
            else np.nan
        ),
        f"{prefix}_probability_brier": (
            float(brier_score_loss(y, probability)) if len(y) else np.nan
        ),
        f"{prefix}_probability_log_loss": (
            float(log_loss(y, probability, labels=[0, 1]))
            if len(y)
            else np.nan
        ),
        f"{prefix}_win_magnitude_spearman": _spearman(
            scored.loc[win, f"{prefix}_win_magnitude_pct"],
            target.loc[win],
        ),
        f"{prefix}_loss_magnitude_spearman": _spearman(
            scored.loc[loss, f"{prefix}_loss_magnitude_pct"],
            -target.loc[loss],
        ),
        f"{prefix}_hurdle_ev_spearman": _spearman(ev, target),
        f"{prefix}_selected_rate": float(selected.sum() / valid.sum())
        if valid.any()
        else np.nan,
        f"{prefix}_selected_predicted_ev_mean_pct": float(
            ev.loc[selected].mean()
        )
        if selected.any()
        else np.nan,
        f"{prefix}_selected_realized_base_mean_pct": float(
            target.loc[selected].mean()
        )
        if selected.any()
        else np.nan,
    }


def supply_coverage(
    scored: pd.DataFrame,
) -> dict[str, object]:
    weighted = pd.to_numeric(
        scored["supply_weighted_shares_outstanding_prior"],
        errors="coerce",
    )
    share_class = pd.to_numeric(
        scored["supply_share_class_shares_outstanding_prior"],
        errors="coerce",
    )
    previous_close = pd.to_numeric(
        scored["previous_close"], errors="coerce"
    )
    implied = weighted * previous_close
    query_date = pd.to_datetime(
        scored["supply_query_date"], errors="coerce"
    )
    trading_day = pd.to_datetime(
        scored["trading_day"], errors="coerce"
    )
    return {
        "supply_query_success_rate": float(
            pd.to_numeric(
                scored["supply_query_success"], errors="coerce"
            ).eq(1.0).mean()
        ),
        "supply_weighted_shares_coverage": float(weighted.notna().mean()),
        "supply_share_class_coverage": float(share_class.notna().mean()),
        "supply_implied_market_cap_coverage": float(implied.notna().mean()),
        "supply_strict_prior_query_dates": bool(
            query_date.notna().all()
            and query_date.lt(trading_day).all()
        ),
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
        "selected_score_mean_pct": float(
            pd.to_numeric(
                selected["selected_decision_score"], errors="coerce"
            ).mean()
        ),
    }


def _selection_comparison(
    trades: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    keys = ["trading_day", "ticker"]
    primary = trades.loc[
        trades["policy"].eq(PRIMARY), keys
    ].drop_duplicates()
    baseline = trades.loc[
        trades["policy"].eq("feasible_base_hurdle_ev_15m_cap1"), keys
    ].drop_duplicates()
    p = set(map(tuple, primary.to_numpy()))
    b = set(map(tuple, baseline.to_numpy()))
    return {
        "month": month,
        "primary_selected": len(p),
        "baseline_selected": len(b),
        "overlap": len(p & b),
        "primary_only": len(p - b),
        "baseline_only": len(b - p),
    }


def run_fold(dataset_paths: dict[str, Path], evaluation_month: str):
    fit, calibration, evaluation, provenance = load_fold(
        dataset_paths, evaluation_month
    )

    print(
        "v2.7 fold",
        {
            "fit": len(fit),
            "calibration": len(calibration),
            "evaluation": len(evaluation),
        },
        flush=True,
    )

    base_probability = train_probability_model(fit, calibration)
    base_win = train_magnitude_model(fit, calibration, positive=True)
    base_loss = train_magnitude_model(fit, calibration, positive=False)

    supply_probability = train_supply_probability_model(fit, calibration)
    supply_win = train_supply_magnitude_model(
        fit, calibration, positive=True
    )
    supply_loss = train_supply_magnitude_model(
        fit, calibration, positive=False
    )

    scored = score_models(
        evaluation,
        base_probability,
        base_win,
        base_loss,
        supply_probability,
        supply_win,
        supply_loss,
    )
    trades, attempts = select_policies(scored)
    if not trades.empty:
        trades["month"] = evaluation_month
    if not attempts.empty:
        attempts["month"] = evaluation_month

    target = pd.to_numeric(
        scored[_target_column(HORIZON)], errors="coerce"
    )
    diagnostics = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "anchor_rows": int(len(scored)),
                "labeled_anchor_rows": int(target.notna().sum()),
                **supply_coverage(scored),
                **_head_diagnostics(scored, prefix="base"),
                **_head_diagnostics(scored, prefix="supply"),
            }
        ]
    )
    details = pd.DataFrame(
        [
            _metrics(trades, month=evaluation_month, policy=policy)
            for policy in POLICIES
        ]
    )
    comparisons = pd.DataFrame(
        [_selection_comparison(trades, month=evaluation_month)]
    )
    coverage = pd.DataFrame(
        [_coverage_row(scored, evaluation_month)]
    )
    bootstrap = day_cluster_bootstrap(
        trades,
        policy=PRIMARY,
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
    details,
    attempts,
    diagnostics,
    comparisons,
    coverage,
    provenance,
    bootstrap,
):
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
            "=== MoneyMaker Point-in-Time Supply-State Hurdle EV v2.7 ===",
            "universe=exact v2.6 execution-feasible first anchors",
            "baseline=exact v2.4 multi-source hurdle heads inside same universe",
            "primary=baseline feature frame plus seven preregistered D-1 supply features",
            "decision=supply hurdle EV > 0",
            "evaluation=strict past-only; April 2026+ sealed",
            "",
            "=== Date provenance ===",
            provenance.to_string(index=False),
            "",
            "=== Model and supply diagnostics ===",
            diagnostics.to_string(index=False),
            "",
            "=== Policies ===",
            details.to_string(index=False),
            "",
            "=== Attempt paths ===",
            reason_summary.to_string(index=False),
            "",
            "=== Supply vs baseline selection ===",
            comparisons.to_string(index=False),
            "",
            "=== Existing external-data coverage ===",
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
        prog="python -m victory_trader.expanded_supply_hurdle_ev"
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

    (
        details,
        trades,
        attempts,
        diagnostics,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    ) = run_fold(dict(args.dataset), args.evaluation_month)

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

    for path in (
        args.report,
        args.details_csv,
        args.trades_csv,
        args.attempts_csv,
        args.diagnostics_csv,
        args.comparisons_csv,
        args.coverage_csv,
        args.provenance_csv,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    attempts.to_csv(args.attempts_csv, index=False)
    diagnostics.to_csv(args.diagnostics_csv, index=False)
    comparisons.to_csv(args.comparisons_csv, index=False)
    coverage.to_csv(args.coverage_csv, index=False)
    provenance.to_csv(args.provenance_csv, index=False)
    pd.DataFrame([bootstrap]).to_csv(
        args.report.with_name(args.report.stem + "-bootstrap.csv"),
        index=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
