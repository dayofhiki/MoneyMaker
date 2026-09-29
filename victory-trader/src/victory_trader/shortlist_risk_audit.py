"""Request283: shortlist-conditioned pullback risk audit.

Request282 showed a globally useful severe-loss signal (OOF AUC about 0.67)
but only tiny economic improvement after applying it inside the stage-1
pre-HOT shortlist. This request changes no trading action. It reconstructs the
same cross-fitted stage-1 shortlist and stage-2 severe-risk scores, then
measures discrimination only where the veto is actually used.

No threshold is promoted from this diagnostic and no new dates are opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .causal_minute_controller_rebuild import causal_execution_scan
from .downstream_aligned_candidate import (
    attach_downstream_target,
    candidate_columns,
)
from .full_hot_fixed_policy_value import attach_fixed_value
from .joint_pullback_admission import (
    FIXED_MAX_HOLD_MINUTES,
    FIXED_STOP_LOSS_PCT,
    FIXED_TRAIL_PCT,
)
from .lagged_minute_context import CROSSFIT_DAYS
from .pullback_trailing_exit import (
    apply_exit_rule,
    build_pullback_episodes,
)
from .two_stage_risk_veto import (
    attach_ticker_history,
    fold_stage1,
    fold_stage2,
    prepare_pullback_states,
)

REQUEST_ID = 283
RISK_REJECT_FRACTION = 0.50
MIN_CONDITIONED_RESOLVED = 30
MIN_CONDITIONED_AUC = 0.60
MIN_AP_LIFT = 0.05
MIN_REJECTED_SEVERE_GAP = 0.10
MIN_ACCEPTED_MEAN_ADVANTAGE_PCT = 0.20


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def risk_diagnostics(
    frame: pd.DataFrame,
    *,
    threshold_column: str = "severe_threshold",
) -> dict:
    work = frame.copy()
    target = _numeric(work.trade_return_pct)
    valid = (
        work.resolved.astype(bool)
        & target.notna()
        & _numeric(work.severe_probability).notna()
    )
    work = work.loc[valid].copy()
    target = _numeric(work.trade_return_pct)
    severe = target.le(-2.0).astype(int)
    score = _numeric(work.severe_probability)
    if len(work) and severe.nunique() == 2:
        auc = float(roc_auc_score(severe, score))
        ap = float(average_precision_score(severe, score))
    else:
        auc = None
        ap = None
    prevalence = float(severe.mean()) if len(severe) else None
    ap_lift = (
        float(ap - prevalence)
        if ap is not None and prevalence is not None
        else None
    )

    threshold = _numeric(work[threshold_column])
    rejected = score > threshold
    accepted = ~rejected
    rejected_rows = work.loc[rejected].copy()
    accepted_rows = work.loc[accepted].copy()
    rejected_return = _numeric(rejected_rows.trade_return_pct)
    accepted_return = _numeric(accepted_rows.trade_return_pct)

    def severe_rate(rows: pd.DataFrame) -> float | None:
        values = _numeric(rows.trade_return_pct)
        return float(values.le(-2.0).mean()) if len(values) else None

    def positive_rate(rows: pd.DataFrame) -> float | None:
        values = _numeric(rows.trade_return_pct)
        return float(values.gt(0).mean()) if len(values) else None

    rejected_severe = severe_rate(rejected_rows)
    accepted_severe = severe_rate(accepted_rows)
    severe_gap = (
        float(rejected_severe - accepted_severe)
        if rejected_severe is not None
        and accepted_severe is not None
        else None
    )
    mean_advantage = (
        float(accepted_return.mean() - rejected_return.mean())
        if len(accepted_return) and len(rejected_return)
        else None
    )

    quantiles = []
    if len(work) >= 8:
        ranked = work.copy()
        ranked["risk_bucket"] = pd.qcut(
            _numeric(ranked.severe_probability).rank(method="first"),
            q=min(4, len(ranked)),
            labels=False,
            duplicates="drop",
        )
        for bucket, part in ranked.groupby("risk_bucket", sort=True):
            values = _numeric(part.trade_return_pct)
            quantiles.append({
                "bucket": int(bucket),
                "rows": int(len(part)),
                "mean_severe_probability": float(
                    _numeric(part.severe_probability).mean()
                ),
                "trade_mean_pct": float(values.mean()),
                "positive_rate": float(values.gt(0).mean()),
                "severe_rate": float(values.le(-2.0).mean()),
            })

    return {
        "resolved_rows": int(len(work)),
        "severe_prevalence": prevalence,
        "auc": auc,
        "average_precision": ap,
        "average_precision_lift_vs_prevalence": ap_lift,
        "rejected_rows": int(rejected.sum()),
        "accepted_rows": int(accepted.sum()),
        "rejected_trade_mean_pct": (
            float(rejected_return.mean())
            if len(rejected_return)
            else None
        ),
        "accepted_trade_mean_pct": (
            float(accepted_return.mean())
            if len(accepted_return)
            else None
        ),
        "accepted_mean_advantage_pct": mean_advantage,
        "rejected_positive_rate": positive_rate(rejected_rows),
        "accepted_positive_rate": positive_rate(accepted_rows),
        "rejected_severe_rate": rejected_severe,
        "accepted_severe_rate": accepted_severe,
        "rejected_minus_accepted_severe_rate": severe_gap,
        "risk_quartiles": quantiles,
    }


def build_oof(
    candidates: pd.DataFrame,
    states: pd.DataFrame,
    prehot_columns: tuple[str, ...],
    post_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prehot_parts = []
    risk_parts = []
    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train_candidates = candidates.loc[
            ~candidates.trading_day.astype(str).eq(held_day)
        ].copy()
        held_candidates = candidates.loc[
            candidates.trading_day.astype(str).eq(held_day)
        ].copy()
        prehot, _ = fold_stage1(
            train_candidates,
            held_candidates,
            prehot_columns,
            20261520 + fold_index * 10,
        )
        prehot_parts.append(prehot)

        train_states = states.loc[
            ~states.trading_day.astype(str).eq(held_day)
        ].copy()
        held_states = states.loc[
            states.trading_day.astype(str).eq(held_day)
        ].copy()
        risk, _ = fold_stage2(
            train_states,
            held_states,
            post_columns,
            20261530 + fold_index * 10,
        )
        risk_parts.append(risk)

    return (
        pd.concat(prehot_parts, ignore_index=True),
        pd.concat(risk_parts, ignore_index=True),
    )


def evaluate(
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    raw_scan = pd.read_parquet(scan_path)
    raw_scan = raw_scan.loc[
        raw_scan.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    causal_scan = causal_execution_scan(raw_scan)

    first_hot = pd.read_parquet(first_hot_path).drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    first_hot = attach_fixed_value(first_hot, causal_scan)

    episodes = build_pullback_episodes(first_hot, causal_scan)
    policy = apply_exit_rule(
        episodes,
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name="request283_frozen_downstream",
    )
    labeled = attach_downstream_target(first_hot, policy)
    candidates, ticker_extra = attach_ticker_history(
        labeled,
        raw_scan,
    )
    prehot_columns = tuple(
        dict.fromkeys([
            *candidate_columns(candidates),
            *ticker_extra,
        ])
    )
    states, post_columns = prepare_pullback_states(
        candidates,
        ticker_extra,
        raw_scan,
        causal_scan,
        policy,
    )

    prehot_oof, risk_oof = build_oof(
        candidates,
        states,
        prehot_columns,
        post_columns,
    )
    joined = states.loc[:, [
        "trading_day",
        "ticker",
        "hot_t",
        "resolved",
        "trade_return_pct",
    ]].merge(
        prehot_oof.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "selected_prehot",
            "prehot_predicted_value_pct",
            "prehot_threshold",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="inner",
        validate="one_to_one",
    ).merge(
        risk_oof.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "severe_probability",
            "severe_threshold",
            "risk_accepted",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="inner",
        validate="one_to_one",
    )

    all_diag = risk_diagnostics(joined)
    conditioned = joined.loc[
        joined.selected_prehot.astype(bool)
    ].copy()
    conditioned_diag = risk_diagnostics(conditioned)

    checks = {
        "conditioned_support": (
            conditioned_diag["resolved_rows"]
            >= MIN_CONDITIONED_RESOLVED
        ),
        "conditioned_auc": (
            conditioned_diag["auc"] is not None
            and conditioned_diag["auc"] >= MIN_CONDITIONED_AUC
        ),
        "conditioned_ap_lift": (
            conditioned_diag[
                "average_precision_lift_vs_prevalence"
            ] is not None
            and conditioned_diag[
                "average_precision_lift_vs_prevalence"
            ] >= MIN_AP_LIFT
        ),
        "rejected_severe_gap": (
            conditioned_diag[
                "rejected_minus_accepted_severe_rate"
            ] is not None
            and conditioned_diag[
                "rejected_minus_accepted_severe_rate"
            ] >= MIN_REJECTED_SEVERE_GAP
        ),
        "accepted_mean_advantage": (
            conditioned_diag[
                "accepted_mean_advantage_pct"
            ] is not None
            and conditioned_diag[
                "accepted_mean_advantage_pct"
            ] >= MIN_ACCEPTED_MEAN_ADVANTAGE_PCT
        ),
    }
    conditioned_signal_pass = bool(all(checks.values()))

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "stage1_selection_fraction": 0.20,
        "stage2_risk_reject_fraction": RISK_REJECT_FRACTION,
        "feature_counts": {
            "prehot": len(prehot_columns),
            "posthot": len(post_columns),
            "ticker_history": len(ticker_extra),
        },
        "global_pullback_risk": all_diag,
        "stage1_shortlist_conditioned_risk": conditioned_diag,
        "checks": checks,
        "shortlist_conditioned_signal_pass": conditioned_signal_pass,
        "interpretation": (
            "This request changes no action. It asks whether the severe-loss "
            "signal remains useful after conditioning on the exact pre-HOT "
            "population where Request282 actually applied the veto."
        ),
        "next_boundary": (
            "replace binary ENTER/CASH with causal WAIT/recheck because "
            "risk remains informative inside the shortlist but the first "
            "pullback can be temporarily risky"
            if conditioned_signal_pass
            else (
                "stage2 global risk discrimination does not transfer "
                "cleanly to the shortlist; rebuild shortlist-conditioned "
                "risk representation before adding action complexity"
            )
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    joined.to_parquet(rows_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
