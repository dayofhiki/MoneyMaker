from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .expanded_anchor_hurdle_ev import (
    HORIZON,
    MagnitudeModel,
    ProbabilityModel,
    hurdle_ev,
    predict_magnitude,
    predict_probability,
    train_magnitude_model,
    train_probability_model,
)
from .expanded_episode_viability_rank import (
    _evaluation_reason,
    _first_eligible_rows,
    _scoreable_days,
    read_model_panel,
)
from .state_action_risk import SEVERE_LOSS_PCT
from .state_action_value import _target_column
from .state_multi_source_value import _coverage_row
from .state_rank_turn import day_cluster_bootstrap


MIN_FINAL_CALIBRATION = 250
MIN_FINAL_CALIBRATION_DAYS = 5
BOOTSTRAP_SAMPLES = 10_000
POLICIES = (
    "earliest_eligible_15m_cap1",
    "raw_hurdle_ev_15m_cap1",
    "calibrated_hurdle_ev_15m_cap1",
)


@dataclass(frozen=True)
class FinalScaleCalibration:
    intercept: float
    slope: float
    rows: int
    days: int
    valid: bool


def _split_days(
    dataset_paths: dict[str, Path],
    evaluation_month: str,
) -> tuple[set[str], set[str], set[str]]:
    days = _scoreable_days(dataset_paths, evaluation_month)
    fit_cut = max(1, int(len(days) * 0.80))
    fit_cut = min(fit_cut, len(days) - 2)

    fit_days = days[:fit_cut]
    cal_days = days[fit_cut:]
    if len(cal_days) < 2:
        raise ValueError("fewer than two chronological calibration days")

    component_count = len(cal_days) // 2
    if component_count < 1:
        raise ValueError("component calibration block is empty")

    component_days = cal_days[:component_count]
    final_days = cal_days[component_count:]
    if not final_days:
        raise ValueError("final-scale calibration block is empty")

    return set(fit_days), set(component_days), set(final_days)


def load_fold(
    dataset_paths: dict[str, Path],
    evaluation_month: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if evaluation_month not in dataset_paths:
        raise ValueError(f"missing evaluation month: {evaluation_month}")

    fit_days, component_days, final_days = _split_days(
        dataset_paths,
        evaluation_month,
    )

    fit_parts: list[pd.DataFrame] = []
    component_parts: list[pd.DataFrame] = []
    final_parts: list[pd.DataFrame] = []

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

        component_panel = panel.loc[day.isin(component_days)].copy()
        if not component_panel.empty:
            component_parts.append(_first_eligible_rows(component_panel))

        final_panel = panel.loc[day.isin(final_days)].copy()
        if not final_panel.empty:
            final_parts.append(_first_eligible_rows(final_panel))

        del panel, day, fit_panel, component_panel, final_panel
        gc.collect()

    fit = pd.concat(fit_parts, ignore_index=True)
    component_cal = pd.concat(component_parts, ignore_index=True)
    final_cal = pd.concat(final_parts, ignore_index=True)
    evaluation = read_model_panel(dataset_paths[evaluation_month])

    evaluation_min = str(evaluation["trading_day"].astype(str).min())
    provenance = pd.DataFrame(
        [
            {
                "evaluation_month": evaluation_month,
                "fit_min_day": min(fit_days),
                "fit_max_day": max(fit_days),
                "component_cal_min_day": min(component_days),
                "component_cal_max_day": max(component_days),
                "final_cal_min_day": min(final_days),
                "final_cal_max_day": max(final_days),
                "evaluation_min_day": evaluation_min,
                "evaluation_max_day": str(
                    evaluation["trading_day"].astype(str).max()
                ),
                "fit_days": len(fit_days),
                "component_cal_days": len(component_days),
                "final_cal_days": len(final_days),
                "training_months": ",".join(prior_months),
                "fit_anchors": len(fit),
                "component_cal_anchors": len(component_cal),
                "final_cal_anchors": len(final_cal),
            }
        ]
    )
    if str(max(final_days)) >= evaluation_min:
        raise ValueError("final calibration overlaps evaluation")

    return fit, component_cal, final_cal, evaluation, provenance


def score_anchors(
    frame: pd.DataFrame,
    probability_model: ProbabilityModel,
    win_model: MagnitudeModel,
    loss_model: MagnitudeModel,
) -> pd.DataFrame:
    anchors = _first_eligible_rows(frame).copy()
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
    anchors["raw_hurdle_ev_pct"] = hurdle_ev(
        anchors["predicted_win_probability"].to_numpy(dtype=float),
        anchors["predicted_win_magnitude_pct"].to_numpy(dtype=float),
        anchors["predicted_loss_magnitude_pct"].to_numpy(dtype=float),
    )
    return anchors


def fit_final_scale(
    final_calibration: pd.DataFrame,
    probability_model: ProbabilityModel,
    win_model: MagnitudeModel,
    loss_model: MagnitudeModel,
) -> tuple[FinalScaleCalibration, pd.DataFrame]:
    scored = score_anchors(
        final_calibration,
        probability_model,
        win_model,
        loss_model,
    )
    target = pd.to_numeric(
        scored[_target_column(HORIZON)],
        errors="coerce",
    )
    raw = pd.to_numeric(scored["raw_hurdle_ev_pct"], errors="coerce")
    valid = target.notna() & raw.notna()
    labeled = scored.loc[valid].copy()
    y = target.loc[valid].to_numpy(dtype=float)
    x = raw.loc[valid].to_numpy(dtype=float).reshape(-1, 1)

    rows = int(len(labeled))
    days = int(labeled["trading_day"].astype(str).nunique())
    if rows < MIN_FINAL_CALIBRATION or days < MIN_FINAL_CALIBRATION_DAYS:
        calibration = FinalScaleCalibration(
            intercept=np.nan,
            slope=np.nan,
            rows=rows,
            days=days,
            valid=False,
        )
        scored["calibrated_hurdle_ev_pct"] = np.nan
        return calibration, scored

    model = LinearRegression()
    model.fit(x, y)
    intercept = float(model.intercept_)
    slope = float(model.coef_[0])
    valid_scale = bool(np.isfinite(slope) and slope > 0.0)

    calibration = FinalScaleCalibration(
        intercept=intercept,
        slope=slope,
        rows=rows,
        days=days,
        valid=valid_scale,
    )
    scored["calibrated_hurdle_ev_pct"] = (
        intercept + slope * scored["raw_hurdle_ev_pct"]
        if valid_scale
        else np.nan
    )
    return calibration, scored


def apply_final_scale(
    scored: pd.DataFrame,
    calibration: FinalScaleCalibration,
) -> pd.DataFrame:
    result = scored.copy()
    if calibration.valid:
        result["calibrated_hurdle_ev_pct"] = (
            calibration.intercept
            + calibration.slope
            * pd.to_numeric(result["raw_hurdle_ev_pct"], errors="coerce")
        )
    else:
        result["calibrated_hurdle_ev_pct"] = np.nan
    return result


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
        "selected_raw_hurdle_ev_pct": row.get(
            "raw_hurdle_ev_pct", np.nan
        ),
        "selected_calibrated_hurdle_ev_pct": row.get(
            "calibrated_hurdle_ev_pct", np.nan
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
    trade["selected_raw_hurdle_ev_pct"] = attempt[
        "selected_raw_hurdle_ev_pct"
    ]
    trade["selected_calibrated_hurdle_ev_pct"] = attempt[
        "selected_calibrated_hurdle_ev_pct"
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
    calibration: FinalScaleCalibration,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attempts: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []

    for _, row in scored.iterrows():
        raw = float(row["raw_hurdle_ev_pct"])
        calibrated = row.get("calibrated_hurdle_ev_pct", np.nan)

        decisions = (
            ("earliest_eligible_15m_cap1", True),
            ("raw_hurdle_ev_15m_cap1", raw > 0.0),
            (
                "calibrated_hurdle_ev_15m_cap1",
                bool(
                    calibration.valid
                    and pd.notna(calibrated)
                    and float(calibrated) > 0.0
                ),
            ),
        )
        for policy, selected in decisions:
            if not selected:
                continue
            attempt, trade = _attempt_row(row, policy=policy)
            attempts.append(attempt)
            if trade is not None:
                trades.append(trade)

    paths = []
    for policy in POLICIES:
        rows = [x for x in attempts if x["policy"] == policy]
        paths.append(
            {
                "policy": policy,
                "final_scale_valid": calibration.valid,
                "final_scale_slope": calibration.slope,
                "final_scale_intercept": calibration.intercept,
                "anchors": int(len(scored)),
                "attempted": len(rows),
                "evaluated": sum(
                    x["evaluation_reason"] == "evaluated" for x in rows
                ),
                "unevaluable": sum(
                    x["evaluation_reason"] != "evaluated" for x in rows
                ),
            }
        )

    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        trade_frame = scored.iloc[0:0].copy()
        trade_frame["policy"] = pd.Series(dtype=str)
        trade_frame["action_horizon_min"] = pd.Series(dtype=int)
        trade_frame["selected_win_probability"] = pd.Series(dtype=float)
        trade_frame["selected_raw_hurdle_ev_pct"] = pd.Series(dtype=float)
        trade_frame["selected_calibrated_hurdle_ev_pct"] = pd.Series(
            dtype=float
        )
        trade_frame["realized_gross_return_pct"] = pd.Series(dtype=float)
        trade_frame["realized_base_net_return_pct"] = pd.Series(dtype=float)
        trade_frame["realized_stress_net_return_pct"] = pd.Series(dtype=float)

    return trade_frame, pd.DataFrame(attempts), pd.DataFrame(paths)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 2:
        return np.nan
    return float(x.loc[valid].corr(y.loc[valid], method="spearman"))


def diagnostics(
    scored: pd.DataFrame,
    final_scored: pd.DataFrame,
    calibration: FinalScaleCalibration,
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

    win_mask = target.gt(0.0)
    loss_mask = target.le(0.0)

    final_target = pd.to_numeric(
        final_scored[_target_column(HORIZON)],
        errors="coerce",
    )
    final_raw = pd.to_numeric(
        final_scored["raw_hurdle_ev_pct"],
        errors="coerce",
    )
    final_calibrated = pd.to_numeric(
        final_scored["calibrated_hurdle_ev_pct"],
        errors="coerce",
    )

    raw = pd.to_numeric(scored["raw_hurdle_ev_pct"], errors="coerce")
    calibrated = pd.to_numeric(
        scored["calibrated_hurdle_ev_pct"],
        errors="coerce",
    )

    return {
        "month": month,
        "anchor_rows": int(len(scored)),
        "labeled_anchor_rows": int(valid.sum()),
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
        "win_magnitude_spearman": _spearman(
            scored.loc[win_mask, "predicted_win_magnitude_pct"],
            target.loc[win_mask],
        ),
        "loss_magnitude_spearman": _spearman(
            scored.loc[loss_mask, "predicted_loss_magnitude_pct"],
            -target.loc[loss_mask],
        ),
        "final_calibration_rows": calibration.rows,
        "final_calibration_days": calibration.days,
        "final_scale_intercept": calibration.intercept,
        "final_scale_slope": calibration.slope,
        "final_scale_valid": calibration.valid,
        "final_raw_ev_spearman": _spearman(final_raw, final_target),
        "final_calibrated_ev_spearman": _spearman(
            final_calibrated,
            final_target,
        ),
        "evaluation_raw_ev_spearman": _spearman(raw, target),
        "evaluation_calibrated_ev_spearman": _spearman(
            calibrated,
            target,
        ),
        "raw_selected_rate": float(raw.gt(0.0).mean()),
        "calibrated_selected_rate": float(calibrated.gt(0.0).mean())
        if calibration.valid
        else 0.0,
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
        "raw_ev_mean_pct": float(
            pd.to_numeric(
                selected["selected_raw_hurdle_ev_pct"],
                errors="coerce",
            ).mean()
        ),
        "calibrated_ev_mean_pct": float(
            pd.to_numeric(
                selected["selected_calibrated_hurdle_ev_pct"],
                errors="coerce",
            ).mean()
        ),
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
    fit, component_cal, final_cal, evaluation, provenance = load_fold(
        dataset_paths,
        evaluation_month,
    )

    print(
        "fold rows",
        {
            "fit": len(fit),
            "component_cal": len(component_cal),
            "final_cal": len(final_cal),
            "evaluation": len(evaluation),
        },
        flush=True,
    )

    probability_model = train_probability_model(fit, component_cal)
    win_model = train_magnitude_model(
        fit,
        component_cal,
        positive=True,
    )
    loss_model = train_magnitude_model(
        fit,
        component_cal,
        positive=False,
    )

    calibration, final_scored = fit_final_scale(
        final_cal,
        probability_model,
        win_model,
        loss_model,
    )

    scored = score_anchors(
        evaluation,
        probability_model,
        win_model,
        loss_model,
    )
    scored = apply_final_scale(scored, calibration)

    trades, attempts, paths = select_policies(scored, calibration)
    if not trades.empty:
        trades["month"] = evaluation_month
    if not attempts.empty:
        attempts["month"] = evaluation_month
    paths["month"] = evaluation_month

    diag = pd.DataFrame(
        [
            diagnostics(
                scored,
                final_scored,
                calibration,
                month=evaluation_month,
            )
        ]
    )
    details = pd.DataFrame(
        [
            _metrics(trades, month=evaluation_month, policy=policy)
            for policy in POLICIES
        ]
    )
    coverage = pd.DataFrame(
        [_coverage_row(evaluation, evaluation_month)]
    )
    bootstrap = day_cluster_bootstrap(
        trades,
        policy="calibrated_hurdle_ev_15m_cap1",
        samples=BOOTSTRAP_SAMPLES,
    )

    comparisons = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "primary": "calibrated_hurdle_ev_15m_cap1",
                "comparator": "earliest_eligible_15m_cap1",
            },
            {
                "month": evaluation_month,
                "primary": "calibrated_hurdle_ev_15m_cap1",
                "comparator": "raw_hurdle_ev_15m_cap1",
            },
        ]
    )
    return (
        details,
        trades,
        attempts,
        diag,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    )


def render_report(
    details: pd.DataFrame,
    attempts: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
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
            "=== MoneyMaker Final-Scale Calibrated Hurdle EV v2.5 ===",
            "anchor=first causal eligible state per ticker-day",
            "component_calibration=earlier half of prior 20% calibration days",
            "final_scale_calibration=later half, disjoint",
            "final_mapping=OLS BASE return ~ raw hurdle EV",
            "primary=calibrated hurdle EV > 0 when slope > 0",
            "evaluation=strict past-only",
            "NOTE=development only; April 2026+ sealed",
            "",
            "=== Date provenance ===",
            provenance.to_string(index=False),
            "",
            "=== Diagnostics ===",
            diagnostics_frame.to_string(index=False),
            "",
            "=== Month-by-month policies ===",
            details.to_string(index=False),
            "",
            "=== Attempt paths ===",
            reason_summary.to_string(index=False),
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
        prog="python -m victory_trader.expanded_final_scale_hurdle_ev"
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
        diag,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    ) = run_fold(dict(args.dataset), args.evaluation_month)

    report = render_report(
        details,
        attempts,
        diag,
        coverage,
        provenance,
        bootstrap,
    )
    print(report, flush=True)

    outputs = (
        args.report,
        args.details_csv,
        args.trades_csv,
        args.attempts_csv,
        args.diagnostics_csv,
        args.comparisons_csv,
        args.coverage_csv,
        args.provenance_csv,
    )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    attempts.to_csv(args.attempts_csv, index=False)
    diag.to_csv(args.diagnostics_csv, index=False)
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
