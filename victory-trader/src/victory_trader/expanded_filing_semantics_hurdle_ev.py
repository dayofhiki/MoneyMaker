from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from victory_trader.expanded_anchor_hurdle_ev import (
    MIN_MAGNITUDE_CALIBRATION,
    MIN_MAGNITUDE_CALIBRATION_DAYS,
    MIN_PROBABILITY_CALIBRATION,
    MagnitudeModel,
    ProbabilityModel,
    hurdle_ev,
)
from victory_trader.expanded_episode_viability_rank import HORIZON
from victory_trader.expanded_split_supply_hurdle_ev import (
    predict_split_magnitude,
    predict_split_probability,
    split_coverage,
    split_supply_feature_frame,
    train_split_magnitude_model,
    train_split_probability_model,
)
from victory_trader.expanded_supply_hurdle_ev import (
    BOOTSTRAP_SAMPLES,
    _attempt,
    _head_diagnostics,
    _metrics,
    _signed_magnitude,
    load_fold,
    supply_coverage,
)
from victory_trader.state_action_value import _target_column
from victory_trader.state_multi_source_value import _coverage_row
from victory_trader.state_rank_turn import day_cluster_bootstrap


POLICIES = (
    "feasible_earliest_15m_cap1",
    "split_supply_hurdle_ev_15m_cap1",
    "filing_semantic_split_supply_hurdle_ev_15m_cap1",
)
PRIMARY = "filing_semantic_split_supply_hurdle_ev_15m_cap1"
SEMANTIC_FEATURES = (
    "filing_semantic_log1p_equity_supply_30d",
    "filing_semantic_log1p_equity_supply_7d",
    "filing_semantic_log1p_debt_30d",
    "filing_semantic_log1p_debt_7d",
    "filing_semantic_log1p_operating_30d",
    "filing_semantic_log1p_operating_7d",
    "filing_semantic_log1p_adverse_30d",
    "filing_semantic_log1p_adverse_7d",
    "filing_semantic_log1p_total_30d",
    "filing_semantic_log1p_latest_age_days",
    "filing_semantic_supply_minus_operating_balance",
)


def filing_semantic_split_supply_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = split_supply_feature_frame(frame).copy()
    raw: dict[str, pd.Series] = {}
    for bucket, source in (
        ("equity_supply_30d", "filing_equity_supply_accessions_30d"),
        ("equity_supply_7d", "filing_equity_supply_accessions_7d"),
        ("debt_30d", "filing_debt_accessions_30d"),
        ("debt_7d", "filing_debt_accessions_7d"),
        ("operating_30d", "filing_operating_catalyst_accessions_30d"),
        ("operating_7d", "filing_operating_catalyst_accessions_7d"),
        ("adverse_30d", "filing_adverse_accessions_30d"),
        ("adverse_7d", "filing_adverse_accessions_7d"),
        ("total_30d", "filing_semantic_accessions_30d"),
    ):
        values = pd.to_numeric(frame[source], errors="coerce").clip(lower=0)
        raw[bucket] = values
        result[f"filing_semantic_log1p_{bucket}"] = np.log1p(values)
    age = pd.to_numeric(
        frame["filing_latest_semantic_age_days"], errors="coerce"
    )
    result["filing_semantic_log1p_latest_age_days"] = np.log1p(
        age.where(age >= 0)
    )
    result["filing_semantic_supply_minus_operating_balance"] = (
        raw["equity_supply_30d"] - raw["operating_30d"]
    ) / (raw["equity_supply_30d"] + raw["operating_30d"] + 1.0)
    return result


def _usable_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = filing_semantic_split_supply_feature_frame(frame)
    columns = [column for column in features if features[column].notna().any()]
    if not columns:
        raise ValueError("no usable filing-semantic-split-supply features")
    return features, columns


def train_semantic_probability_model(
    fit: pd.DataFrame, calibration: pd.DataFrame
) -> ProbabilityModel:
    target_col = _target_column(HORIZON)
    target = pd.to_numeric(fit[target_col], errors="coerce")
    valid = target.notna()
    labeled = fit.loc[valid].copy()
    y_fit = target.loc[valid].gt(0.0).astype(int)
    if labeled.empty or y_fit.nunique() < 2:
        raise ValueError("semantic probability fit lacks both classes")
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
        raise ValueError("insufficient semantic probability calibration")
    raw = model.predict_proba(
        filing_semantic_split_supply_feature_frame(cal).reindex(columns=columns)
    )[:, 1]
    raw = np.clip(raw.astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=1000, random_state=20261021
    )
    platt.fit(logits, y_cal)
    return ProbabilityModel(model=model, feature_columns=tuple(columns), platt=platt)


def predict_semantic_probability(
    frame: pd.DataFrame, fitted: ProbabilityModel
) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    raw = fitted.model.predict_proba(
        filing_semantic_split_supply_feature_frame(frame).reindex(
            columns=fitted.feature_columns
        )
    )[:, 1]
    raw = np.clip(raw.astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    return fitted.platt.predict_proba(logits)[:, 1].astype(float)


def train_semantic_magnitude_model(
    fit: pd.DataFrame, calibration: pd.DataFrame, *, positive: bool
) -> MagnitudeModel:
    target_col = _target_column(HORIZON)
    label = "win" if positive else "loss"
    fit_mask, fit_magnitude = _signed_magnitude(fit[target_col], positive=positive)
    fit_rows = fit.loc[fit_mask].copy()
    if fit_rows.empty:
        raise ValueError(f"no semantic {label} magnitude fit rows")
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
        raise ValueError(f"insufficient semantic {label} calibration rows")
    if (
        cal_rows["trading_day"].astype(str).nunique()
        < MIN_MAGNITUDE_CALIBRATION_DAYS
    ):
        raise ValueError(f"insufficient semantic {label} calibration days")
    raw = model.predict(
        filing_semantic_split_supply_feature_frame(cal_rows).reindex(columns=columns)
    ).astype(float)
    offset = float(np.mean(cal_magnitude.to_numpy(dtype=float) - raw))
    return MagnitudeModel(
        model=model, feature_columns=tuple(columns), offset=offset, label=label
    )


def predict_semantic_magnitude(
    frame: pd.DataFrame, fitted: MagnitudeModel
) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    raw = fitted.model.predict(
        filing_semantic_split_supply_feature_frame(frame).reindex(
            columns=fitted.feature_columns
        )
    ).astype(float)
    return np.maximum(0.0, raw + fitted.offset)


def score_models(
    evaluation: pd.DataFrame,
    split_probability: ProbabilityModel,
    split_win: MagnitudeModel,
    split_loss: MagnitudeModel,
    semantic_probability: ProbabilityModel,
    semantic_win: MagnitudeModel,
    semantic_loss: MagnitudeModel,
) -> pd.DataFrame:
    scored = evaluation.copy()
    scored["split_win_probability"] = predict_split_probability(
        scored, split_probability
    )
    scored["split_win_magnitude_pct"] = predict_split_magnitude(scored, split_win)
    scored["split_loss_magnitude_pct"] = predict_split_magnitude(scored, split_loss)
    scored["split_hurdle_ev_pct"] = hurdle_ev(
        scored["split_win_probability"],
        scored["split_win_magnitude_pct"],
        scored["split_loss_magnitude_pct"],
    )
    scored["semantic_win_probability"] = predict_semantic_probability(
        scored, semantic_probability
    )
    scored["semantic_win_magnitude_pct"] = predict_semantic_magnitude(scored, semantic_win)
    scored["semantic_loss_magnitude_pct"] = predict_semantic_magnitude(scored, semantic_loss)
    scored["semantic_hurdle_ev_pct"] = hurdle_ev(
        scored["semantic_win_probability"],
        scored["semantic_win_magnitude_pct"],
        scored["semantic_loss_magnitude_pct"],
    )
    return scored


def select_policies(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    for _, row in scored.iterrows():
        decisions = (
            ("feasible_earliest_15m_cap1", True, None),
            (
                "split_supply_hurdle_ev_15m_cap1",
                float(row["split_hurdle_ev_pct"]) > 0.0,
                "split_hurdle_ev_pct",
            ),
            (
                PRIMARY,
                float(row["semantic_hurdle_ev_pct"]) > 0.0,
                "semantic_hurdle_ev_pct",
            ),
        )
        for policy, selected, score_column in decisions:
            if not selected:
                continue
            attempt, trade = _attempt(row, policy=policy, score_column=score_column)
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


def semantic_coverage(scored: pd.DataFrame) -> dict[str, object]:
    return {
        "semantic_query_success": float(
            scored["filing_semantics_query_success"].eq(1.0).mean()
        ),
        "semantic_strict_prior": bool(
            scored["filing_semantics_strict_prior"].eq(1.0).all()
        ),
        "semantic_30d_rate": float(
            scored["filing_semantic_accessions_30d"].gt(0).mean()
        ),
        "semantic_equity_supply_30d_rate": float(
            scored["filing_equity_supply_accessions_30d"].gt(0).mean()
        ),
        "semantic_debt_30d_rate": float(
            scored["filing_debt_accessions_30d"].gt(0).mean()
        ),
        "semantic_operating_30d_rate": float(
            scored["filing_operating_catalyst_accessions_30d"].gt(0).mean()
        ),
        "semantic_adverse_30d_rate": float(
            scored["filing_adverse_accessions_30d"].gt(0).mean()
        ),
    }


def _selection_comparison(trades: pd.DataFrame, month: str) -> dict[str, object]:
    keys = ["trading_day", "ticker"]
    primary = set(
        map(tuple, trades.loc[trades["policy"].eq(PRIMARY), keys].to_numpy())
    )
    comparator = set(
        map(
            tuple,
            trades.loc[
                trades["policy"].eq("split_supply_hurdle_ev_15m_cap1"), keys
            ].to_numpy(),
        )
    )
    return {
        "month": month,
        "primary_selected": len(primary),
        "split_selected": len(comparator),
        "overlap": len(primary & comparator),
        "primary_only": len(primary - comparator),
        "split_only": len(comparator - primary),
    }


def run_fold(dataset_paths: dict[str, Path], evaluation_month: str):
    fit, calibration, evaluation, provenance = load_fold(
        dataset_paths, evaluation_month
    )
    split_probability = train_split_probability_model(fit, calibration)
    split_win = train_split_magnitude_model(fit, calibration, positive=True)
    split_loss = train_split_magnitude_model(fit, calibration, positive=False)
    semantic_probability = train_semantic_probability_model(fit, calibration)
    semantic_win = train_semantic_magnitude_model(fit, calibration, positive=True)
    semantic_loss = train_semantic_magnitude_model(fit, calibration, positive=False)
    scored = score_models(
        evaluation,
        split_probability,
        split_win,
        split_loss,
        semantic_probability,
        semantic_win,
        semantic_loss,
    )
    trades, attempts = select_policies(scored)
    if not trades.empty:
        trades["month"] = evaluation_month
    if not attempts.empty:
        attempts["month"] = evaluation_month
    target = pd.to_numeric(scored[_target_column(HORIZON)], errors="coerce")
    diagnostics = pd.DataFrame(
        [{
            "month": evaluation_month,
            "anchor_rows": len(scored),
            "labeled_anchor_rows": int(target.notna().sum()),
            **supply_coverage(scored),
            **split_coverage(scored),
            **semantic_coverage(scored),
            **_head_diagnostics(scored, prefix="split"),
            **_head_diagnostics(scored, prefix="semantic"),
        }]
    )
    details = pd.DataFrame(
        [_metrics(trades, month=evaluation_month, policy=policy) for policy in POLICIES]
    )
    comparisons = pd.DataFrame([_selection_comparison(trades, evaluation_month)])
    coverage = pd.DataFrame(
        [{**_coverage_row(scored, evaluation_month), **split_coverage(scored), **semantic_coverage(scored)}]
    )
    bootstrap = day_cluster_bootstrap(
        trades, policy=PRIMARY, samples=BOOTSTRAP_SAMPLES
    )
    return details, trades, attempts, diagnostics, comparisons, coverage, provenance, bootstrap


def render_report(details, attempts, diagnostics, comparisons, coverage, provenance, bootstrap):
    reasons = (
        attempts.groupby(["policy", "evaluation_reason"]).size().rename("attempts").reset_index()
        if not attempts.empty else pd.DataFrame()
    )
    return "\n".join([
        "=== MoneyMaker Filing-Semantic Hurdle EV v3.0 ===",
        "universe=exact v2.8 execution-feasible first anchors",
        "comparator=exact v2.8 split+supply hurdle heads",
        "primary=comparator features plus eleven preregistered strictly-prior filing-semantic features",
        "decision=semantic hurdle EV > 0; April 2026+ sealed",
        "", "=== Date provenance ===", provenance.to_string(index=False),
        "", "=== Diagnostics ===", diagnostics.to_string(index=False),
        "", "=== Policies ===", details.to_string(index=False),
        "", "=== Attempt paths ===", reasons.to_string(index=False),
        "", "=== Selection comparison ===", comparisons.to_string(index=False),
        "", "=== Coverage ===", coverage.to_string(index=False),
        "", "=== Primary pooled day bootstrap ===", str(bootstrap),
    ])


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--evaluation-month", required=True)
    for name in ("report", "details_csv", "trades_csv", "attempts_csv", "diagnostics_csv", "comparisons_csv", "coverage_csv", "provenance_csv"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    outputs = run_fold(dict(args.dataset), args.evaluation_month)
    details, trades, attempts, diagnostics, comparisons, coverage, provenance, bootstrap = outputs
    report = render_report(details, attempts, diagnostics, comparisons, coverage, provenance, bootstrap)
    print(report, flush=True)
    frames = {
        args.details_csv: details,
        args.trades_csv: trades,
        args.attempts_csv: attempts,
        args.diagnostics_csv: diagnostics,
        args.comparisons_csv: comparisons,
        args.coverage_csv: coverage,
        args.provenance_csv: provenance,
    }
    for path, frame in frames.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    pd.DataFrame([bootstrap]).to_csv(
        args.report.with_name(args.report.stem + "-bootstrap.csv"), index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
