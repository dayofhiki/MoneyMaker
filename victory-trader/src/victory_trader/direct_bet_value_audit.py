"""Request289: audit direct causal bet-value predictions from Request288.

No model is refit and no market data is fetched. Request288 already produced
outer-fold OOF direct return predictions on the original first 2% pullback.
This diagnostic asks whether those predictions actually order realized returns
inside the held-out stage-1 shortlist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUEST_ID = 289
MIN_RESOLVED = 30
MIN_SPEARMAN = 0.20


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _quartile_report(frame: pd.DataFrame) -> dict:
    if len(frame) < 20:
        return {}
    rank = _numeric(frame.direct_predicted_return_pct).rank(
        method="first",
        pct=True,
    )
    bucket = np.minimum(
        4,
        np.maximum(1, np.ceil(rank * 4).astype(int)),
    )
    temp = frame.assign(_quartile=bucket)
    report = {}
    for q, part in temp.groupby("_quartile", sort=True):
        actual = _numeric(part.trade_return_pct)
        report[f"q{int(q)}"] = {
            "rows": int(len(part)),
            "predicted_mean_pct": float(
                _numeric(part.direct_predicted_return_pct).mean()
            ),
            "realized_mean_pct": float(actual.mean()),
            "positive_rate": float(actual.gt(0).mean()),
            "severe_rate_le_minus2": float(actual.le(-2).mean()),
            "upside_rate_ge_plus2": float(actual.ge(2).mean()),
        }
    return report


def _fraction_report(
    frame: pd.DataFrame,
    fraction: float,
) -> dict:
    if frame.empty:
        return {}
    score = _numeric(frame.direct_predicted_return_pct)
    threshold = float(score.quantile(1.0 - float(fraction)))
    selected = frame.loc[score.ge(threshold)].copy()
    actual = _numeric(selected.trade_return_pct)
    return {
        "diagnostic_only": True,
        "fraction": float(fraction),
        "oof_score_threshold_pct": threshold,
        "rows": int(len(selected)),
        "realized_mean_pct": float(actual.mean()),
        "positive_rate": float(actual.gt(0).mean()),
        "severe_rate_le_minus2": float(actual.le(-2).mean()),
        "upside_rate_ge_plus2": float(actual.ge(2).mean()),
    }


def audit(rows: pd.DataFrame) -> dict:
    shortlist = rows.loc[
        rows.selected_prehot.fillna(False).astype(bool)
    ].copy()
    resolved = shortlist.loc[
        shortlist.resolved.astype(bool)
        & _numeric(shortlist.trade_return_pct).notna()
        & _numeric(shortlist.direct_predicted_return_pct).notna()
    ].copy()
    actual = _numeric(resolved.trade_return_pct)
    score = _numeric(resolved.direct_predicted_return_pct)
    spearman = actual.corr(score, method="spearman")
    spearman = None if pd.isna(spearman) else float(spearman)

    positive_score = resolved.loc[
        score.gt(0)
    ].copy()
    nonpositive_score = resolved.loc[
        ~score.gt(0)
    ].copy()
    pos_actual = _numeric(positive_score.trade_return_pct)
    neg_actual = _numeric(nonpositive_score.trade_return_pct)

    decision = shortlist.copy()
    decision["direct_take"] = _numeric(
        decision.direct_predicted_return_pct
    ).gt(0)
    decision["decision_resolved"] = (
        ~decision.direct_take
        | decision.resolved.astype(bool)
    )
    decision["decision_return_pct"] = 0.0
    active = decision.direct_take & decision.resolved.astype(bool)
    decision.loc[active, "decision_return_pct"] = _numeric(
        decision.loc[active, "trade_return_pct"]
    )
    decision.loc[
        ~decision.decision_resolved,
        "decision_return_pct",
    ] = np.nan
    valid = decision.loc[decision.decision_resolved].copy()
    daily = valid.groupby(
        valid.trading_day.astype(str),
        sort=True,
    ).decision_return_pct.mean()

    quartiles = _quartile_report(resolved)
    q1 = quartiles.get("q1", {})
    q4 = quartiles.get("q4", {})
    q_spread = (
        float(q4["realized_mean_pct"] - q1["realized_mean_pct"])
        if q1 and q4
        else None
    )

    return {
        "shortlisted_pullback_rows": int(len(shortlist)),
        "resolved_shortlist_rows": int(len(resolved)),
        "baseline_realized_mean_pct": (
            float(actual.mean()) if len(actual) else None
        ),
        "prediction_distribution_pct": {
            "min": float(score.min()) if len(score) else None,
            "q25": float(score.quantile(0.25)) if len(score) else None,
            "median": float(score.median()) if len(score) else None,
            "q75": float(score.quantile(0.75)) if len(score) else None,
            "max": float(score.max()) if len(score) else None,
            "mean": float(score.mean()) if len(score) else None,
        },
        "direct_value_spearman": spearman,
        "predicted_positive_resolved_bets": int(len(positive_score)),
        "predicted_positive_realized_mean_pct": (
            float(pos_actual.mean()) if len(pos_actual) else None
        ),
        "predicted_positive_positive_rate": (
            float(pos_actual.gt(0).mean()) if len(pos_actual) else None
        ),
        "predicted_positive_severe_rate_le_minus2": (
            float(pos_actual.le(-2).mean()) if len(pos_actual) else None
        ),
        "predicted_nonpositive_counterfactual_mean_pct": (
            float(neg_actual.mean()) if len(neg_actual) else None
        ),
        "positive_minus_nonpositive_realized_mean_pct": (
            float(pos_actual.mean() - neg_actual.mean())
            if len(pos_actual) and len(neg_actual)
            else None
        ),
        "cash_adjusted_direct_positive_mean_pct": (
            float(_numeric(valid.decision_return_pct).mean())
            if len(valid)
            else None
        ),
        "decision_resolution_or_cash_rate": (
            float(decision.decision_resolved.mean())
            if len(decision)
            else 0.0
        ),
        "positive_decision_mean_days": int((daily > 0).sum()),
        "quartiles_low_to_high": quartiles,
        "q4_minus_q1_realized_mean_pct": q_spread,
        "top_half_diagnostic": _fraction_report(resolved, 0.50),
        "top_quartile_diagnostic": _fraction_report(resolved, 0.25),
        "by_day": {
            str(day): {
                "resolved": int(
                    part.resolved.astype(bool).sum()
                ),
                "predicted_positive": int(
                    _numeric(part.direct_predicted_return_pct).gt(0).sum()
                ),
            }
            for day, part in shortlist.groupby(
                shortlist.trading_day.astype(str),
                sort=True,
            )
        },
    }


def evaluate(
    rows_path: Path,
    output_path: Path,
) -> int:
    rows = pd.read_parquet(rows_path)
    report = audit(rows)
    quartiles = report["quartiles_low_to_high"]
    q4 = quartiles.get("q4", {})
    checks = {
        "support": report["resolved_shortlist_rows"] >= MIN_RESOLVED,
        "ranking_signal": (
            report["direct_value_spearman"] is not None
            and report["direct_value_spearman"] >= MIN_SPEARMAN
        ),
        "q4_beats_q1": (
            report["q4_minus_q1_realized_mean_pct"] is not None
            and report["q4_minus_q1_realized_mean_pct"] > 0
        ),
        "q4_realized_positive": (
            bool(q4)
            and q4["realized_mean_pct"] > 0
        ),
    }
    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "source_request": 288,
        "model_refit": False,
        "market_data_refetch": False,
        "report": report,
        "checks": checks,
        "direct_bet_ordering_gate_pass": bool(all(checks.values())),
        "interpretation_contract": (
            "quartile/top-fraction reports are diagnostic OOF ordering checks, "
            "not deployable thresholds; only predicted return > 0 is treated "
            "as an absolute action rule"
        ),
        "next_boundary": (
            "if ordering is coherent, build nested training-only calibration "
            "and stake sizing for direct expected return"
            if bool(all(checks.values()))
            else (
                "if ordering is only partial, locate the information horizon "
                "where direct EV becomes usable before any stake sizing"
            )
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(args.rows, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
