from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .expanded_anchor_hurdle_ev import (
    MIN_MAGNITUDE_CALIBRATION,
    MIN_MAGNITUDE_CALIBRATION_DAYS,
    MIN_PROBABILITY_CALIBRATION,
    MagnitudeModel,
    ProbabilityModel,
    hurdle_ev,
)
from .expanded_filing_semantics_hurdle_ev import (
    filing_semantic_split_supply_feature_frame,
    semantic_coverage,
)
from .expanded_supply_hurdle_ev import (
    BOOTSTRAP_SAMPLES,
    _metrics,
    _signed_magnitude,
    load_fold,
)
from .state_action_value import _target_column
from .state_multi_source_value import _coverage_row
from .state_rank_turn import day_cluster_bootstrap


HORIZONS = (1, 2, 5, 10, 15, 30)
BROAD = "feasible_earliest_15m_cap1"
FIXED = "filing_semantic_split_supply_hurdle_ev_15m_cap1"
PRIMARY = "adaptive_filing_semantic_hurdle_ev_cap1"
POLICIES = (BROAD, FIXED, PRIMARY)


@dataclass(frozen=True)
class HorizonModels:
    probability: ProbabilityModel
    win: MagnitudeModel
    loss: MagnitudeModel


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return filing_semantic_split_supply_feature_frame(frame)


def _usable_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = _feature_frame(frame)
    columns = [column for column in features if features[column].notna().any()]
    if not columns:
        raise ValueError("no usable adaptive semantic features")
    return features, columns


def train_probability_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    horizon: int,
) -> ProbabilityModel:
    target_col = _target_column(horizon)
    target = pd.to_numeric(fit[target_col], errors="coerce")
    valid = target.notna()
    labeled = fit.loc[valid].copy()
    y_fit = target.loc[valid].gt(0.0).astype(int)
    if labeled.empty or y_fit.nunique() < 2:
        raise ValueError(f"{horizon}m probability fit lacks both classes")

    x_fit, columns = _usable_features(labeled)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=20261021,
    )
    model.fit(x_fit.loc[:, columns], y_fit)

    cal_target = pd.to_numeric(calibration[target_col], errors="coerce")
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    y_cal = cal_target.loc[cal_valid].gt(0.0).astype(int)
    if len(cal) < MIN_PROBABILITY_CALIBRATION or y_cal.nunique() < 2:
        raise ValueError(f"insufficient {horizon}m probability calibration")
    raw = model.predict_proba(
        _feature_frame(cal).reindex(columns=columns)
    )[:, 1]
    raw = np.clip(raw.astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=20261021,
    )
    platt.fit(logits, y_cal)
    return ProbabilityModel(
        model=model,
        feature_columns=tuple(columns),
        platt=platt,
    )


def predict_probability(frame: pd.DataFrame, fitted: ProbabilityModel) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    raw = fitted.model.predict_proba(
        _feature_frame(frame).reindex(columns=fitted.feature_columns)
    )[:, 1]
    raw = np.clip(raw.astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    return fitted.platt.predict_proba(logits)[:, 1].astype(float)


def train_magnitude_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    horizon: int,
    *,
    positive: bool,
) -> MagnitudeModel:
    target_col = _target_column(horizon)
    label = "win" if positive else "loss"
    fit_mask, fit_magnitude = _signed_magnitude(
        fit[target_col], positive=positive
    )
    fit_rows = fit.loc[fit_mask].copy()
    if fit_rows.empty:
        raise ValueError(f"no {horizon}m {label} magnitude fit rows")

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
        random_state=20261022 if positive else 20261023,
    )
    model.fit(x_fit.loc[:, columns], y_fit)

    cal_mask, cal_magnitude = _signed_magnitude(
        calibration[target_col], positive=positive
    )
    cal_rows = calibration.loc[cal_mask].copy()
    if len(cal_rows) < MIN_MAGNITUDE_CALIBRATION:
        raise ValueError(
            f"insufficient {horizon}m {label} magnitude calibration rows"
        )
    if (
        cal_rows["trading_day"].astype(str).nunique()
        < MIN_MAGNITUDE_CALIBRATION_DAYS
    ):
        raise ValueError(
            f"insufficient {horizon}m {label} magnitude calibration days"
        )
    raw = model.predict(
        _feature_frame(cal_rows).reindex(columns=columns)
    ).astype(float)
    offset = float(np.mean(cal_magnitude.to_numpy(dtype=float) - raw))
    return MagnitudeModel(
        model=model,
        feature_columns=tuple(columns),
        offset=offset,
        label=label,
    )


def predict_magnitude(frame: pd.DataFrame, fitted: MagnitudeModel) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    raw = fitted.model.predict(
        _feature_frame(frame).reindex(columns=fitted.feature_columns)
    ).astype(float)
    return np.maximum(0.0, raw + fitted.offset)


def train_horizon_models(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> dict[int, HorizonModels]:
    result: dict[int, HorizonModels] = {}
    for horizon in HORIZONS:
        result[horizon] = HorizonModels(
            probability=train_probability_model(fit, calibration, horizon),
            win=train_magnitude_model(
                fit, calibration, horizon, positive=True
            ),
            loss=train_magnitude_model(
                fit, calibration, horizon, positive=False
            ),
        )
    return result


def score_horizons(
    evaluation: pd.DataFrame,
    models: dict[int, HorizonModels],
) -> pd.DataFrame:
    scored = evaluation.copy()
    ev_columns: list[str] = []
    for horizon in HORIZONS:
        fitted = models[horizon]
        p = predict_probability(scored, fitted.probability)
        win = predict_magnitude(scored, fitted.win)
        loss = predict_magnitude(scored, fitted.loss)
        prefix = f"adaptive_{horizon}m"
        scored[f"{prefix}_win_probability"] = p
        scored[f"{prefix}_win_magnitude_pct"] = win
        scored[f"{prefix}_loss_magnitude_pct"] = loss
        scored[f"{prefix}_hurdle_ev_pct"] = hurdle_ev(p, win, loss)
        ev_columns.append(f"{prefix}_hurdle_ev_pct")

    matrix = scored.loc[:, ev_columns].to_numpy(dtype=float)
    best_index = np.argmax(matrix, axis=1)
    horizon_array = np.asarray(HORIZONS, dtype=int)
    scored["adaptive_best_horizon_min"] = horizon_array[best_index]
    scored["adaptive_best_hurdle_ev_pct"] = matrix[
        np.arange(len(scored)), best_index
    ]
    return scored


def _evaluation_reason(
    row: pd.Series,
    horizon: int,
) -> tuple[str, dict[str, bool]]:
    entry = pd.to_numeric(
        pd.Series([row.get("entry_price", np.nan)]), errors="coerce"
    ).iloc[0]
    gross = pd.to_numeric(
        pd.Series([row.get(f"buy_return_{horizon}m_pct", np.nan)]),
        errors="coerce",
    ).iloc[0]
    base = pd.to_numeric(
        pd.Series([row.get(_target_column(horizon), np.nan)]),
        errors="coerce",
    ).iloc[0]
    stress = pd.to_numeric(
        pd.Series(
            [row.get(f"buy_return_{horizon}m_stress_net_return_pct", np.nan)]
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


def _attempt(
    row: pd.Series,
    *,
    policy: str,
    horizon: int,
    score: float,
) -> tuple[dict[str, object], dict[str, object] | None]:
    reason, flags = _evaluation_reason(row, horizon)
    attempt = {
        "policy": policy,
        "trading_day": row.get("trading_day"),
        "ticker": row.get("ticker"),
        "decision_t": row.get("t"),
        "action_horizon_min": horizon,
        "entry_price": row.get("entry_price", np.nan),
        "selected_decision_score": score,
        "evaluation_reason": reason,
        **flags,
    }
    if reason != "evaluated":
        return attempt, None

    trade = row.to_dict()
    trade["policy"] = policy
    trade["action_horizon_min"] = horizon
    trade["selected_decision_score"] = score
    trade["realized_gross_return_pct"] = float(
        row[f"buy_return_{horizon}m_pct"]
    )
    trade["realized_base_net_return_pct"] = float(
        row[_target_column(horizon)]
    )
    trade["realized_stress_net_return_pct"] = float(
        row[f"buy_return_{horizon}m_stress_net_return_pct"]
    )
    return attempt, trade


def select_policies(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    for _, row in scored.iterrows():
        fixed_ev = float(row["adaptive_15m_hurdle_ev_pct"])
        best_ev = float(row["adaptive_best_hurdle_ev_pct"])
        best_horizon = int(row["adaptive_best_horizon_min"])
        decisions = (
            (BROAD, True, 15, np.nan),
            (FIXED, fixed_ev > 0.0, 15, fixed_ev),
            (PRIMARY, best_ev > 0.0, best_horizon, best_ev),
        )
        for policy, selected, horizon, score in decisions:
            if not selected:
                continue
            attempt, trade = _attempt(
                row,
                policy=policy,
                horizon=horizon,
                score=score,
            )
            attempts.append(attempt)
            if trade is not None:
                trades.append(trade)
    return pd.DataFrame(trades), pd.DataFrame(attempts)


def _safe_auc(y: pd.Series, p: pd.Series) -> float:
    valid = y.notna() & p.notna()
    if valid.sum() == 0 or y.loc[valid].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y.loc[valid].astype(int), p.loc[valid]))


def horizon_diagnostics(scored: pd.DataFrame, month: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        target = pd.to_numeric(scored[_target_column(horizon)], errors="coerce")
        p = pd.to_numeric(
            scored[f"adaptive_{horizon}m_win_probability"], errors="coerce"
        )
        ev = pd.to_numeric(
            scored[f"adaptive_{horizon}m_hurdle_ev_pct"], errors="coerce"
        )
        valid = target.notna() & p.notna() & ev.notna()
        y = target.gt(0.0).astype(int)
        if valid.any():
            brier = float(brier_score_loss(y.loc[valid], p.loc[valid]))
            clipped = p.loc[valid].clip(1e-9, 1.0 - 1e-9)
            ll = float(log_loss(y.loc[valid], clipped, labels=[0, 1]))
            spearman = float(ev.loc[valid].corr(target.loc[valid], method="spearman"))
        else:
            brier = ll = spearman = float("nan")
        rows.append(
            {
                "month": month,
                "horizon_min": horizon,
                "labeled_rows": int(valid.sum()),
                "positive_rate": float(y.loc[valid].mean()) if valid.any() else np.nan,
                "probability_auc": _safe_auc(y.loc[valid], p.loc[valid]),
                "probability_brier": brier,
                "probability_log_loss": ll,
                "hurdle_ev_spearman": spearman,
                "predicted_ev_positive_rate": float(ev.gt(0.0).mean()),
                "predicted_ev_mean_pct": float(ev.mean()),
                "realized_base_mean_pct": float(target.loc[valid].mean())
                if valid.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def action_summary(
    trades: pd.DataFrame,
    attempts: pd.DataFrame,
    month: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy in POLICIES:
        pa = attempts.loc[attempts["policy"].eq(policy)].copy()
        pt = trades.loc[trades["policy"].eq(policy)].copy()
        row: dict[str, object] = {
            "month": month,
            "policy": policy,
            "attempts": int(len(pa)),
            "evaluated": int(len(pt)),
            "evaluation_rate": float(len(pt) / len(pa)) if len(pa) else np.nan,
        }
        for horizon in HORIZONS:
            row[f"action_{horizon}m_rate"] = (
                float(
                    pd.to_numeric(
                        pa["action_horizon_min"], errors="coerce"
                    ).eq(horizon).mean()
                )
                if len(pa)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def oracle_summary(trades: pd.DataFrame, month: str) -> pd.DataFrame:
    selected = trades.loc[trades["policy"].eq(PRIMARY)].copy()
    if selected.empty:
        return pd.DataFrame(
            [{"month": month, "policy": PRIMARY, "evaluated": 0}]
        )

    base_matrix = pd.DataFrame(
        {
            horizon: pd.to_numeric(
                selected[_target_column(horizon)], errors="coerce"
            )
            for horizon in HORIZONS
        },
        index=selected.index,
    )
    oracle = base_matrix.max(axis=1, skipna=True)
    chosen = pd.to_numeric(
        selected["realized_base_net_return_pct"], errors="coerce"
    )
    best_matches: list[bool] = []
    for idx, row in base_matrix.iterrows():
        finite = row.dropna()
        if finite.empty:
            best_matches.append(False)
            continue
        chosen_horizon = int(selected.loc[idx, "action_horizon_min"])
        best_matches.append(
            bool(
                pd.notna(row.get(chosen_horizon))
                and float(row[chosen_horizon]) >= float(finite.max()) - 1e-12
            )
        )

    return pd.DataFrame(
        [
            {
                "month": month,
                "policy": PRIMARY,
                "evaluated": int(len(selected)),
                "chosen_base_mean_pct": float(chosen.mean()),
                "oracle_base_mean_pct": float(oracle.mean()),
                "mean_oracle_regret_pct": float((oracle - chosen).mean()),
                "chosen_is_best_horizon_rate": float(np.mean(best_matches)),
                "any_base_positive_rate": float((base_matrix > 0.0).any(axis=1).mean()),
            }
        ]
    )


def selection_comparison(trades: pd.DataFrame, month: str) -> pd.DataFrame:
    keys = ["trading_day", "ticker"]
    primary = set(
        map(
            tuple,
            trades.loc[trades["policy"].eq(PRIMARY), keys].drop_duplicates().to_numpy(),
        )
    )
    fixed = set(
        map(
            tuple,
            trades.loc[trades["policy"].eq(FIXED), keys].drop_duplicates().to_numpy(),
        )
    )
    return pd.DataFrame(
        [
            {
                "month": month,
                "primary_selected": len(primary),
                "fixed15_selected": len(fixed),
                "overlap": len(primary & fixed),
                "primary_only": len(primary - fixed),
                "fixed15_only": len(fixed - primary),
            }
        ]
    )


def run_fold(dataset_paths: dict[str, Path], evaluation_month: str):
    fit, calibration, evaluation, provenance = load_fold(
        dataset_paths, evaluation_month
    )
    print(
        "v3.2 fold",
        {
            "month": evaluation_month,
            "fit": len(fit),
            "calibration": len(calibration),
            "evaluation": len(evaluation),
        },
        flush=True,
    )
    models = train_horizon_models(fit, calibration)
    scored = score_horizons(evaluation, models)
    trades, attempts = select_policies(scored)
    if not trades.empty:
        trades["month"] = evaluation_month
    if not attempts.empty:
        attempts["month"] = evaluation_month

    details = pd.DataFrame(
        [
            _metrics(trades, month=evaluation_month, policy=policy)
            for policy in POLICIES
        ]
    )
    diagnostics = horizon_diagnostics(scored, evaluation_month)
    actions = action_summary(trades, attempts, evaluation_month)
    oracle = oracle_summary(trades, evaluation_month)
    comparisons = selection_comparison(trades, evaluation_month)
    coverage = pd.DataFrame(
        [
            {
                **_coverage_row(scored, evaluation_month),
                **semantic_coverage(scored),
            }
        ]
    )
    bootstrap = pd.DataFrame(
        [
            day_cluster_bootstrap(
                trades,
                policy=PRIMARY,
                samples=BOOTSTRAP_SAMPLES,
            )
        ]
    )
    return (
        details,
        trades,
        attempts,
        diagnostics,
        actions,
        oracle,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    )


def render_report(
    details: pd.DataFrame,
    attempts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    actions: pd.DataFrame,
    oracle: pd.DataFrame,
    comparisons: pd.DataFrame,
    coverage: pd.DataFrame,
    provenance: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    reasons = (
        attempts.groupby(["policy", "evaluation_reason"])
        .size()
        .rename("attempts")
        .reset_index()
        if not attempts.empty
        else pd.DataFrame()
    )
    return "\n".join(
        [
            "=== MoneyMaker Adaptive Multi-Horizon Hurdle EV v3.2 ===",
            "universe=execution-feasible first anchors from frozen v3.0 semantic feature set",
            "actions=BUY_1M/2M/5M/10M/15M/30M or SKIP",
            "decision=max calibrated horizon-specific hurdle EV; BUY iff max EV > 0",
            "April 2026+ sealed",
            "",
            "=== Details ===",
            details.to_string(index=False),
            "",
            "=== Action mix ===",
            actions.to_string(index=False),
            "",
            "=== Horizon diagnostics ===",
            diagnostics.to_string(index=False),
            "",
            "=== Oracle regret diagnostic ===",
            oracle.to_string(index=False),
            "",
            "=== Selection comparison ===",
            comparisons.to_string(index=False),
            "",
            "=== Attempt reasons ===",
            reasons.to_string(index=False),
            "",
            "=== Coverage ===",
            coverage.to_string(index=False),
            "",
            "=== Provenance ===",
            provenance.to_string(index=False),
            "",
            "=== Primary pooled day bootstrap ===",
            bootstrap.to_string(index=False),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.expanded_adaptive_hurdle_ev"
    )
    parser.add_argument("--evaluation-month", required=True)
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--attempts-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--actions-csv", type=Path, required=True)
    parser.add_argument("--oracle-csv", type=Path, required=True)
    parser.add_argument("--comparisons-csv", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--provenance-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-csv", type=Path, required=True)
    args = parser.parse_args()

    outputs = run_fold(dict(args.dataset), args.evaluation_month)
    (
        details,
        trades,
        attempts,
        diagnostics,
        actions,
        oracle,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    ) = outputs
    report = render_report(
        details,
        attempts,
        diagnostics,
        actions,
        oracle,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    )
    print(report, flush=True)

    frames = (
        (args.details_csv, details),
        (args.trades_csv, trades),
        (args.attempts_csv, attempts),
        (args.diagnostics_csv, diagnostics),
        (args.actions_csv, actions),
        (args.oracle_csv, oracle),
        (args.comparisons_csv, comparisons),
        (args.coverage_csv, coverage),
        (args.provenance_csv, provenance),
        (args.bootstrap_csv, bootstrap),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    for path, frame in frames:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
