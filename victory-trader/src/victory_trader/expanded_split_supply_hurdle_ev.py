from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from .expanded_anchor_hurdle_ev import (
    MIN_MAGNITUDE_CALIBRATION,
    MIN_MAGNITUDE_CALIBRATION_DAYS,
    MIN_PROBABILITY_CALIBRATION,
    MagnitudeModel,
    ProbabilityModel,
    hurdle_ev,
)
from .expanded_episode_viability_rank import HORIZON
from .expanded_supply_hurdle_ev import (
    BOOTSTRAP_SAMPLES,
    _attempt,
    _head_diagnostics,
    _metrics,
    _signed_magnitude,
    load_fold,
    predict_supply_magnitude,
    predict_supply_probability,
    supply_coverage,
    supply_feature_frame,
    train_supply_magnitude_model,
    train_supply_probability_model,
)
from .state_action_value import _target_column
from .state_multi_source_value import _coverage_row
from .state_rank_turn import day_cluster_bootstrap


POLICIES = (
    "feasible_earliest_15m_cap1",
    "supply_hurdle_ev_15m_cap1",
    "split_supply_hurdle_ev_15m_cap1",
)
PRIMARY = "split_supply_hurdle_ev_15m_cap1"

SPLIT_FEATURES = (
    "split_any_730d",
    "reverse_split_count_730d",
    "reverse_split_count_365d",
    "forward_split_count_730d",
    "stock_dividend_count_730d",
    "split_log1p_days_since_latest_reverse",
    "split_log_latest_reverse_consolidation",
    "split_log_max_reverse_consolidation",
)


def split_supply_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = supply_feature_frame(frame).copy()

    result["split_any_730d"] = pd.to_numeric(
        frame["split_any_730d"], errors="coerce"
    )
    for column in (
        "reverse_split_count_730d",
        "reverse_split_count_365d",
        "forward_split_count_730d",
        "stock_dividend_count_730d",
    ):
        result[column] = pd.to_numeric(frame[column], errors="coerce")

    days = pd.to_numeric(
        frame["split_days_since_latest_reverse"], errors="coerce"
    )
    result["split_log1p_days_since_latest_reverse"] = np.log1p(
        days.where(days >= 0)
    )

    latest = pd.to_numeric(
        frame["split_latest_reverse_consolidation"], errors="coerce"
    )
    result["split_log_latest_reverse_consolidation"] = np.log(
        latest.where(latest > 0)
    )

    maximum = pd.to_numeric(
        frame["split_max_reverse_consolidation"], errors="coerce"
    )
    result["split_log_max_reverse_consolidation"] = np.log(
        maximum.where(maximum > 0)
    )
    return result


def _usable_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    features = split_supply_feature_frame(frame)
    columns = [
        column for column in features.columns if features[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable split-supply features")
    return features, columns


def train_split_probability_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> ProbabilityModel:
    target_col = _target_column(HORIZON)
    target = pd.to_numeric(fit[target_col], errors="coerce")
    valid = target.notna()
    labeled = fit.loc[valid].copy()
    y_fit = target.loc[valid].gt(0.0).astype(int)
    if labeled.empty or y_fit.nunique() < 2:
        raise ValueError("split-supply probability fit lacks both classes")

    x_fit, columns = _usable_features(labeled)
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
        raise ValueError("insufficient split-supply probability calibration rows")
    if y_cal.nunique() < 2:
        raise ValueError("split-supply probability calibration lacks both classes")

    raw = model.predict_proba(
        split_supply_feature_frame(cal).reindex(columns=columns)
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


def predict_split_probability(
    frame: pd.DataFrame,
    fitted: ProbabilityModel,
) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    raw = fitted.model.predict_proba(
        split_supply_feature_frame(frame).reindex(
            columns=fitted.feature_columns
        )
    )[:, 1]
    raw = np.clip(raw.astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    return fitted.platt.predict_proba(logits)[:, 1].astype(float)


def train_split_magnitude_model(
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
        raise ValueError(f"no split-supply {label} magnitude fit rows")

    low = float(fit_magnitude.quantile(0.005))
    high = float(fit_magnitude.quantile(0.995))
    y_fit = fit_magnitude.clip(lower=low, upper=high)

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
        calibration[target_col], positive=positive
    )
    cal_rows = calibration.loc[cal_mask].copy()
    if len(cal_rows) < MIN_MAGNITUDE_CALIBRATION:
        raise ValueError(
            f"insufficient split-supply {label} magnitude calibration rows"
        )
    if (
        cal_rows["trading_day"].astype(str).nunique()
        < MIN_MAGNITUDE_CALIBRATION_DAYS
    ):
        raise ValueError(
            f"insufficient split-supply {label} magnitude calibration days"
        )

    raw = model.predict(
        split_supply_feature_frame(cal_rows).reindex(columns=columns)
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


def predict_split_magnitude(
    frame: pd.DataFrame,
    fitted: MagnitudeModel,
) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    raw = fitted.model.predict(
        split_supply_feature_frame(frame).reindex(
            columns=fitted.feature_columns
        )
    ).astype(float)
    return np.maximum(0.0, raw + fitted.offset)


def score_models(
    evaluation: pd.DataFrame,
    supply_probability: ProbabilityModel,
    supply_win: MagnitudeModel,
    supply_loss: MagnitudeModel,
    split_probability: ProbabilityModel,
    split_win: MagnitudeModel,
    split_loss: MagnitudeModel,
) -> pd.DataFrame:
    scored = evaluation.copy()

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

    scored["split_win_probability"] = predict_split_probability(
        scored, split_probability
    )
    scored["split_win_magnitude_pct"] = predict_split_magnitude(
        scored, split_win
    )
    scored["split_loss_magnitude_pct"] = predict_split_magnitude(
        scored, split_loss
    )
    scored["split_hurdle_ev_pct"] = hurdle_ev(
        scored["split_win_probability"].to_numpy(dtype=float),
        scored["split_win_magnitude_pct"].to_numpy(dtype=float),
        scored["split_loss_magnitude_pct"].to_numpy(dtype=float),
    )
    return scored


def select_policies(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []

    for _, row in scored.iterrows():
        decisions = (
            ("feasible_earliest_15m_cap1", True, None),
            (
                "supply_hurdle_ev_15m_cap1",
                float(row["supply_hurdle_ev_pct"]) > 0.0,
                "supply_hurdle_ev_pct",
            ),
            (
                "split_supply_hurdle_ev_15m_cap1",
                float(row["split_hurdle_ev_pct"]) > 0.0,
                "split_hurdle_ev_pct",
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


def split_coverage(scored: pd.DataFrame) -> dict[str, object]:
    reverse = pd.to_numeric(
        scored["reverse_split_count_730d"], errors="coerce"
    ).gt(0)
    return {
        "split_query_success": float(
            pd.to_numeric(
                scored["split_history_query_success"], errors="coerce"
            ).eq(1.0).mean()
        ),
        "split_strict_prior": bool(
            pd.to_numeric(
                scored["split_history_strict_prior"], errors="coerce"
            ).eq(1.0).all()
        ),
        "reverse_split_anchor_rate_730d": float(reverse.mean()),
        "reverse_split_anchors_730d": int(reverse.sum()),
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
    comparator = trades.loc[
        trades["policy"].eq("supply_hurdle_ev_15m_cap1"), keys
    ].drop_duplicates()
    p = set(map(tuple, primary.to_numpy()))
    c = set(map(tuple, comparator.to_numpy()))
    return {
        "month": month,
        "primary_selected": len(p),
        "supply_selected": len(c),
        "overlap": len(p & c),
        "primary_only": len(p - c),
        "supply_only": len(c - p),
    }


def run_fold(dataset_paths: dict[str, Path], evaluation_month: str):
    fit, calibration, evaluation, provenance = load_fold(
        dataset_paths, evaluation_month
    )

    print(
        "v2.8 fold",
        {
            "fit": len(fit),
            "calibration": len(calibration),
            "evaluation": len(evaluation),
        },
        flush=True,
    )

    supply_probability = train_supply_probability_model(fit, calibration)
    supply_win = train_supply_magnitude_model(
        fit, calibration, positive=True
    )
    supply_loss = train_supply_magnitude_model(
        fit, calibration, positive=False
    )

    split_probability = train_split_probability_model(fit, calibration)
    split_win = train_split_magnitude_model(
        fit, calibration, positive=True
    )
    split_loss = train_split_magnitude_model(
        fit, calibration, positive=False
    )

    scored = score_models(
        evaluation,
        supply_probability,
        supply_win,
        supply_loss,
        split_probability,
        split_win,
        split_loss,
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
                **split_coverage(scored),
                **_head_diagnostics(scored, prefix="supply"),
                **_head_diagnostics(scored, prefix="split"),
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
        [{**_coverage_row(scored, evaluation_month), **split_coverage(scored)}]
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
            "=== MoneyMaker Split-State Hurdle EV v2.8 ===",
            "universe=exact v2.7 execution-feasible first anchors",
            "comparator=exact v2.7 supply-state hurdle heads",
            "primary=supply feature frame plus eight preregistered split-history features",
            "decision=split-supply hurdle EV > 0",
            "evaluation=strict past-only; April 2026+ sealed",
            "",
            "=== Date provenance ===",
            provenance.to_string(index=False),
            "",
            "=== Model / supply / split diagnostics ===",
            diagnostics.to_string(index=False),
            "",
            "=== Policies ===",
            details.to_string(index=False),
            "",
            "=== Attempt paths ===",
            reason_summary.to_string(index=False),
            "",
            "=== Split-supply vs supply selection ===",
            comparisons.to_string(index=False),
            "",
            "=== Existing external-data / split coverage ===",
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
        prog="python -m victory_trader.expanded_split_supply_hurdle_ev"
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
