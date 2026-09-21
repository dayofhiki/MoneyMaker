from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .expanded_adaptive_hurdle_ev import (
    BROAD,
    FIXED,
    HORIZONS,
    _attempt,
    _feature_frame,
    score_horizons,
    train_horizon_models,
)
from .expanded_anchor_hurdle_ev import MIN_PROBABILITY_CALIBRATION
from .expanded_filing_semantics_hurdle_ev import semantic_coverage
from .expanded_supply_hurdle_ev import BOOTSTRAP_SAMPLES, _metrics, load_fold
from .state_action_value import _target_column
from .state_multi_source_value import _coverage_row
from .state_rank_turn import day_cluster_bootstrap


V32 = "adaptive_filing_semantic_hurdle_ev_cap1"
SHARED = "policy_calibrated_shared_q_cap1"
PRIMARY = "opportunity_gated_shared_q_cap1"
POLICIES = (BROAD, FIXED, V32, SHARED, PRIMARY)
MIN_CALIBRATION_DAYS = 5
REGULAR_SESSION_LAST_MINUTE = 389


@dataclass(frozen=True)
class OpportunityModel:
    model: HistGradientBoostingClassifier
    feature_columns: tuple[str, ...]
    platt: LogisticRegression


@dataclass(frozen=True)
class SharedQModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]


def decision_feasible(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Known-at-decision clock feasibility; never inspect future label availability."""
    minute = pd.to_numeric(frame["minutes_from_regular_open"], errors="coerce")
    return minute.ge(0.0) & minute.le(REGULAR_SESSION_LAST_MINUTE - horizon)


def _state_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = _feature_frame(frame)
    columns = [column for column in features if features[column].notna().any()]
    if not columns:
        raise ValueError("no usable shared-Q state features")
    return features, columns


def opportunity_target(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    values: list[np.ndarray] = []
    available: list[np.ndarray] = []
    for horizon in HORIZONS:
        target = pd.to_numeric(frame[_target_column(horizon)], errors="coerce")
        usable = decision_feasible(frame, horizon) & target.notna()
        values.append(target.to_numpy(dtype=float))
        available.append(usable.to_numpy(dtype=bool))
    matrix = np.column_stack(values)
    mask = np.column_stack(available)
    any_available = mask.any(axis=1)
    masked = np.where(mask, matrix, -np.inf)
    best = masked.max(axis=1)
    target = pd.Series(best > 0.0, index=frame.index, dtype=bool)
    valid = pd.Series(any_available, index=frame.index, dtype=bool)
    return target, valid


def train_opportunity_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> OpportunityModel:
    y_fit, fit_valid = opportunity_target(fit)
    labeled = fit.loc[fit_valid].copy()
    y_fit = y_fit.loc[fit_valid].astype(int)
    if labeled.empty or y_fit.nunique() < 2:
        raise ValueError("opportunity fit lacks both classes")
    x_fit, columns = _state_features(labeled)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=20261031,
    )
    model.fit(x_fit.loc[:, columns], y_fit)

    y_cal, cal_valid = opportunity_target(calibration)
    cal = calibration.loc[cal_valid].copy()
    y_cal = y_cal.loc[cal_valid].astype(int)
    if len(cal) < MIN_PROBABILITY_CALIBRATION or y_cal.nunique() < 2:
        raise ValueError("insufficient opportunity calibration")
    raw = model.predict_proba(
        _feature_frame(cal).reindex(columns=columns)
    )[:, 1]
    raw = np.clip(raw.astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=20261031,
    )
    platt.fit(logits, y_cal)
    return OpportunityModel(model, tuple(columns), platt)


def predict_opportunity(frame: pd.DataFrame, fitted: OpportunityModel) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    raw = fitted.model.predict_proba(
        _feature_frame(frame).reindex(columns=fitted.feature_columns)
    )[:, 1]
    raw = np.clip(raw.astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    return fitted.platt.predict_proba(logits)[:, 1].astype(float)


def _action_features(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    features = _feature_frame(frame).copy()
    features["action_horizon_min"] = float(horizon)
    features["action_log_horizon"] = float(np.log1p(horizon))
    for candidate in HORIZONS:
        features[f"action_is_{candidate}m"] = float(candidate == horizon)
    return features


def train_shared_q(fit: pd.DataFrame) -> SharedQModel:
    frames: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    for horizon in HORIZONS:
        target = pd.to_numeric(fit[_target_column(horizon)], errors="coerce")
        valid = decision_feasible(fit, horizon) & target.notna()
        if not valid.any():
            continue
        frames.append(_action_features(fit.loc[valid], horizon))
        targets.append(target.loc[valid])
    if not frames:
        raise ValueError("shared Q has no labeled feasible actions")
    x_fit = pd.concat(frames, ignore_index=True)
    y_fit = pd.concat(targets, ignore_index=True).astype(float)
    columns = [column for column in x_fit if x_fit[column].notna().any()]
    low = float(y_fit.quantile(0.005))
    high = float(y_fit.quantile(0.995))
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=20261032,
    )
    model.fit(x_fit.loc[:, columns], y_fit.clip(lower=low, upper=high))
    return SharedQModel(model, tuple(columns))


def score_shared_actions(
    frame: pd.DataFrame,
    opportunity: OpportunityModel,
    shared_q: SharedQModel,
) -> pd.DataFrame:
    scored = frame.copy()
    scored["opportunity_probability"] = predict_opportunity(scored, opportunity)
    q_columns: list[str] = []
    feasible_columns: list[str] = []
    for horizon in HORIZONS:
        q_column = f"shared_q_{horizon}m_raw_pct"
        feasible_column = f"action_{horizon}m_clock_feasible"
        scored[q_column] = shared_q.model.predict(
            _action_features(scored, horizon).reindex(
                columns=shared_q.feature_columns
            )
        ).astype(float)
        scored[feasible_column] = decision_feasible(scored, horizon)
        q_columns.append(q_column)
        feasible_columns.append(feasible_column)

    q = scored.loc[:, q_columns].to_numpy(dtype=float)
    feasible = scored.loc[:, feasible_columns].to_numpy(dtype=bool)
    masked = np.where(feasible, q, -np.inf)
    any_feasible = feasible.any(axis=1)
    best_index = np.argmax(masked, axis=1)
    horizons = np.asarray(HORIZONS, dtype=int)
    scored["shared_best_horizon_min"] = np.where(
        any_feasible, horizons[best_index], 0
    )
    scored["shared_best_raw_q_pct"] = np.where(
        any_feasible,
        masked[np.arange(len(scored)), best_index],
        np.nan,
    )
    return scored


def _chosen_target(scored: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=scored.index, dtype=float)
    chosen = pd.to_numeric(scored["shared_best_horizon_min"], errors="coerce")
    for horizon in HORIZONS:
        mask = chosen.eq(horizon)
        result.loc[mask] = pd.to_numeric(
            scored.loc[mask, _target_column(horizon)], errors="coerce"
        )
    return result


def policy_level_correction(scored_calibration: pd.DataFrame) -> dict[str, object]:
    actual = _chosen_target(scored_calibration)
    predicted = pd.to_numeric(
        scored_calibration["shared_best_raw_q_pct"], errors="coerce"
    )
    gate = pd.to_numeric(
        scored_calibration["opportunity_probability"], errors="coerce"
    )
    selected = gate.gt(0.5) & predicted.gt(0.0)
    valid = selected & actual.notna() & predicted.notna()
    rows = scored_calibration.loc[valid, ["trading_day"]].copy()
    rows["optimism_pct"] = predicted.loc[valid] - actual.loc[valid]
    daily = rows.groupby(rows["trading_day"].astype(str))["optimism_pct"].mean()
    if len(daily) < MIN_CALIBRATION_DAYS:
        correction = float("inf")
        mean = se = float("nan")
    else:
        mean = float(daily.mean())
        se = float(daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 else 0.0
        correction = max(0.0, mean + 1.645 * se)
    return {
        "selected_rows": int(selected.sum()),
        "evaluated_selected_rows": int(valid.sum()),
        "selected_days": int(len(daily)),
        "mean_daily_max_selection_optimism_pct": mean,
        "daily_optimism_se_pct": se,
        "policy_level_correction_pct": correction,
    }


def apply_policy_correction(
    scored: pd.DataFrame,
    calibration_summary: dict[str, object],
) -> pd.DataFrame:
    result = scored.copy()
    correction = float(calibration_summary["policy_level_correction_pct"])
    raw = pd.to_numeric(result["shared_best_raw_q_pct"], errors="coerce")
    result["shared_best_corrected_q_pct"] = raw - correction
    return result


def select_policies(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    for _, row in scored.iterrows():
        fixed_ev = float(row["adaptive_15m_hurdle_ev_pct"])
        v32_ev = float(row["adaptive_best_hurdle_ev_pct"])
        v32_horizon = int(row["adaptive_best_horizon_min"])
        shared_q = float(row["shared_best_corrected_q_pct"])
        shared_horizon = int(row["shared_best_horizon_min"])
        gate = float(row["opportunity_probability"])
        decisions = (
            (BROAD, True, 15, np.nan),
            (FIXED, fixed_ev > 0.0, 15, fixed_ev),
            (V32, v32_ev > 0.0, v32_horizon, v32_ev),
            (SHARED, shared_horizon > 0 and shared_q > 0.0, shared_horizon, shared_q),
            (
                PRIMARY,
                shared_horizon > 0 and gate > 0.5 and shared_q > 0.0,
                shared_horizon,
                shared_q,
            ),
        )
        for policy, selected, horizon, score in decisions:
            if not selected:
                continue
            attempt, trade = _attempt(
                row, policy=policy, horizon=horizon, score=score
            )
            attempt["opportunity_probability"] = gate
            attempts.append(attempt)
            if trade is not None:
                trade["opportunity_probability"] = gate
                trades.append(trade)
    return pd.DataFrame(trades), pd.DataFrame(attempts)


def _safe_auc(y: pd.Series, p: pd.Series) -> float:
    valid = y.notna() & p.notna()
    if valid.sum() == 0 or y.loc[valid].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y.loc[valid].astype(int), p.loc[valid]))


def opportunity_diagnostics(scored: pd.DataFrame, month: str) -> pd.DataFrame:
    target, valid = opportunity_target(scored)
    p = pd.to_numeric(scored["opportunity_probability"], errors="coerce")
    valid = valid & p.notna()
    y = target.loc[valid].astype(int)
    clipped = p.loc[valid].clip(1e-9, 1.0 - 1e-9)
    return pd.DataFrame(
        [
            {
                "month": month,
                "labeled_rows": int(valid.sum()),
                "opportunity_rate": float(y.mean()),
                "probability_auc": _safe_auc(y, p.loc[valid]),
                "probability_brier": float(brier_score_loss(y, p.loc[valid])),
                "probability_log_loss": float(log_loss(y, clipped, labels=[0, 1])),
                "gate_positive_rate": float(p.gt(0.5).mean()),
            }
        ]
    )


def raw_shared_diagnostics(scored: pd.DataFrame, month: str) -> pd.DataFrame:
    outcome = pd.DataFrame(index=scored.index)
    for horizon in HORIZONS:
        values = pd.to_numeric(scored[_target_column(horizon)], errors="coerce")
        outcome[horizon] = values.where(decision_feasible(scored, horizon))
    chosen = _chosen_target(scored)
    oracle = outcome.max(axis=1, skipna=True)
    raw_q = pd.to_numeric(scored["shared_best_raw_q_pct"], errors="coerce")
    gate = pd.to_numeric(scored["opportunity_probability"], errors="coerce")
    chosen_horizon = pd.to_numeric(
        scored["shared_best_horizon_min"], errors="coerce"
    )
    subsets = {
        "all_clock_feasible": chosen_horizon.gt(0),
        "gate_positive": chosen_horizon.gt(0) & gate.gt(0.5),
        "raw_q_positive": chosen_horizon.gt(0) & raw_q.gt(0.0),
        "gate_and_raw_q_positive": (
            chosen_horizon.gt(0) & gate.gt(0.5) & raw_q.gt(0.0)
        ),
    }
    rows: list[dict[str, object]] = []
    for label, selected in subsets.items():
        valid = selected & chosen.notna() & oracle.notna() & raw_q.notna()
        exact: list[bool] = []
        for idx in scored.index[valid]:
            horizon = int(chosen_horizon.loc[idx])
            row = outcome.loc[idx].dropna()
            exact.append(
                bool(
                    not row.empty
                    and pd.notna(outcome.loc[idx, horizon])
                    and float(outcome.loc[idx, horizon])
                    >= float(row.max()) - 1e-12
                )
            )
        entry: dict[str, object] = {
            "month": month,
            "subset": label,
            "states": int(selected.sum()),
            "chosen_evaluated": int(valid.sum()),
            "chosen_evaluation_rate": (
                float(valid.sum() / selected.sum()) if selected.any() else np.nan
            ),
            "chosen_base_mean_pct": float(chosen.loc[valid].mean())
            if valid.any()
            else np.nan,
            "oracle_base_mean_pct": float(oracle.loc[valid].mean())
            if valid.any()
            else np.nan,
            "mean_oracle_regret_pct": float(
                (oracle.loc[valid] - chosen.loc[valid]).mean()
            )
            if valid.any()
            else np.nan,
            "chosen_is_best_horizon_rate": float(np.mean(exact))
            if exact
            else np.nan,
            "any_base_positive_rate": float(
                outcome.loc[valid].gt(0.0).any(axis=1).mean()
            )
            if valid.any()
            else np.nan,
            "raw_q_chosen_spearman": float(
                raw_q.loc[valid].corr(chosen.loc[valid], method="spearman")
            )
            if valid.sum() > 1
            else np.nan,
            "mean_selected_q_optimism_pct": float(
                (raw_q.loc[valid] - chosen.loc[valid]).mean()
            )
            if valid.any()
            else np.nan,
        }
        for horizon in HORIZONS:
            entry[f"action_{horizon}m_rate"] = (
                float(chosen_horizon.loc[selected].eq(horizon).mean())
                if selected.any()
                else np.nan
            )
        rows.append(entry)
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
                float(pd.to_numeric(pa["action_horizon_min"], errors="coerce").eq(horizon).mean())
                if len(pa)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def action_rank_diagnostics(trades: pd.DataFrame, month: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy in (V32, SHARED, PRIMARY):
        selected = trades.loc[trades["policy"].eq(policy)].copy()
        if selected.empty:
            rows.append({"month": month, "policy": policy, "evaluated": 0})
            continue
        matrix = pd.DataFrame(
            {
                horizon: pd.to_numeric(
                    selected[_target_column(horizon)], errors="coerce"
                )
                for horizon in HORIZONS
            },
            index=selected.index,
        )
        oracle = matrix.max(axis=1, skipna=True)
        chosen = pd.to_numeric(selected["realized_base_net_return_pct"], errors="coerce")
        exact: list[bool] = []
        for idx, outcomes in matrix.iterrows():
            finite = outcomes.dropna()
            horizon = int(selected.loc[idx, "action_horizon_min"])
            exact.append(
                bool(
                    not finite.empty
                    and pd.notna(outcomes.get(horizon))
                    and float(outcomes[horizon]) >= float(finite.max()) - 1e-12
                )
            )
        rows.append(
            {
                "month": month,
                "policy": policy,
                "evaluated": int(len(selected)),
                "chosen_base_mean_pct": float(chosen.mean()),
                "oracle_base_mean_pct": float(oracle.mean()),
                "mean_oracle_regret_pct": float((oracle - chosen).mean()),
                "chosen_is_best_horizon_rate": float(np.mean(exact)),
                "any_base_positive_rate": float((matrix > 0.0).any(axis=1).mean()),
            }
        )
    return pd.DataFrame(rows)


def selection_comparison(trades: pd.DataFrame, month: str) -> pd.DataFrame:
    keys = ["trading_day", "ticker"]
    selected: dict[str, set[tuple[object, ...]]] = {}
    for policy in (FIXED, V32, SHARED, PRIMARY):
        selected[policy] = set(
            map(
                tuple,
                trades.loc[trades["policy"].eq(policy), keys]
                .drop_duplicates()
                .to_numpy(),
            )
        )
    primary = selected[PRIMARY]
    return pd.DataFrame(
        [
            {
                "month": month,
                "primary_selected": len(primary),
                "fixed15_selected": len(selected[FIXED]),
                "v32_selected": len(selected[V32]),
                "shared_ungated_selected": len(selected[SHARED]),
                "primary_v32_overlap": len(primary & selected[V32]),
                "primary_only_vs_v32": len(primary - selected[V32]),
                "v32_only_vs_primary": len(selected[V32] - primary),
            }
        ]
    )


def run_fold(dataset_paths: dict[str, Path], evaluation_month: str):
    fit, calibration, evaluation, provenance = load_fold(
        dataset_paths, evaluation_month
    )
    print(
        "v3.3 fold",
        {
            "month": evaluation_month,
            "fit": len(fit),
            "calibration": len(calibration),
            "evaluation": len(evaluation),
        },
        flush=True,
    )
    opportunity = train_opportunity_model(fit, calibration)
    shared_q = train_shared_q(fit)
    calibration_scored = score_shared_actions(calibration, opportunity, shared_q)
    correction = policy_level_correction(calibration_scored)

    scored = score_shared_actions(evaluation, opportunity, shared_q)
    scored = apply_policy_correction(scored, correction)
    scored = score_horizons(evaluation=scored, models=train_horizon_models(fit, calibration))
    trades, attempts = select_policies(scored)
    if not trades.empty:
        trades["month"] = evaluation_month
    if not attempts.empty:
        attempts["month"] = evaluation_month

    details = pd.DataFrame(
        [_metrics(trades, month=evaluation_month, policy=policy) for policy in POLICIES]
    )
    calibration_frame = pd.DataFrame([{"month": evaluation_month, **correction}])
    opportunity_frame = opportunity_diagnostics(scored, evaluation_month)
    raw_diagnostics = raw_shared_diagnostics(scored, evaluation_month)
    actions = action_summary(trades, attempts, evaluation_month)
    ranks = action_rank_diagnostics(trades, evaluation_month)
    comparisons = selection_comparison(trades, evaluation_month)
    coverage = pd.DataFrame(
        [{**_coverage_row(scored, evaluation_month), **semantic_coverage(scored)}]
    )
    bootstrap = pd.DataFrame(
        [
            day_cluster_bootstrap(
                trades, policy=PRIMARY, samples=BOOTSTRAP_SAMPLES
            )
        ]
    )
    return (
        details,
        trades,
        attempts,
        calibration_frame,
        opportunity_frame,
        raw_diagnostics,
        actions,
        ranks,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    )


def render_report(outputs: tuple[pd.DataFrame, ...]) -> str:
    (
        details,
        _trades,
        attempts,
        calibration,
        opportunity,
        raw_diagnostics,
        actions,
        ranks,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    ) = outputs
    reasons = (
        attempts.groupby(["policy", "evaluation_reason"])
        .size()
        .rename("attempts")
        .reset_index()
        if not attempts.empty
        else pd.DataFrame()
    )
    sections = (
        ("Details", details),
        ("Policy-level calibration", calibration),
        ("Opportunity gate", opportunity),
        ("Raw shared-Q diagnostics (non-policy)", raw_diagnostics),
        ("Action mix", actions),
        ("Action ranking and oracle regret", ranks),
        ("Selection comparison", comparisons),
        ("Attempt reasons", reasons),
        ("Coverage", coverage),
        ("Provenance", provenance),
        ("Primary pooled day bootstrap", bootstrap),
    )
    lines = [
        "=== MoneyMaker Opportunity-Gated Shared Action Value v3.3 ===",
        "entry gate=P(any clock-feasible horizon BASE-positive) > 0.5",
        "action=shared Q over 1/2/5/10/15/30m with policy-level optimism correction",
        "clock feasibility is known at decision; future label availability is never a mask",
        "April 2026+ sealed",
    ]
    for title, frame in sections:
        lines.extend(("", f"=== {title} ===", frame.to_string(index=False)))
    return "\n".join(lines)


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.expanded_opportunity_shared_q"
    )
    parser.add_argument("--evaluation-month", required=True)
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--report", type=Path, required=True)
    for name in (
        "details", "trades", "attempts", "calibration", "opportunity",
        "raw-diagnostics", "actions", "ranks", "comparisons", "coverage",
        "provenance", "bootstrap",
    ):
        parser.add_argument(f"--{name}-csv", type=Path, required=True)
    args = parser.parse_args()

    outputs = run_fold(dict(args.dataset), args.evaluation_month)
    report = render_report(outputs)
    print(report, flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    paths = (
        args.details_csv, args.trades_csv, args.attempts_csv,
        args.calibration_csv, args.opportunity_csv, args.raw_diagnostics_csv,
        args.actions_csv, args.ranks_csv, args.comparisons_csv,
        args.coverage_csv, args.provenance_csv, args.bootstrap_csv,
    )
    for path, frame in zip(paths, outputs, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
