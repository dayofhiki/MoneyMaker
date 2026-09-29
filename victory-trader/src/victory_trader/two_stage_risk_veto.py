"""Request282: two-stage pre-HOT shortlist plus pullback risk veto.

Stage 1 uses Request280's best representation: downstream-value scoring with
exact pre-HOT ticker history. Stage 2 waits for the already-frozen first causal
2% pullback, then uses Request277-style strictly lagged completed-minute
context to estimate severe-loss risk.

The only new action is a preregistered veto: reject the highest 50% severe-risk
pullback states according to a threshold derived solely from the outer fold's
training states. Entry and exit mechanics remain frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .causal_minute_controller_rebuild import causal_execution_scan
from .downstream_aligned_candidate import (
    MIN_RESOLUTION_OR_CASH,
    SELECTION_FRACTION,
    attach_downstream_target,
    candidate_columns,
    fit_pair,
    selection_metrics,
    threshold_from_training,
)
from .full_hot_fixed_policy_value import attach_fixed_value, xframe
from .joint_pullback_admission import (
    FIXED_MAX_HOLD_MINUTES,
    FIXED_STOP_LOSS_PCT,
    FIXED_TRAIL_PCT,
    build_pullback_state_rows,
    feature_columns,
    feature_x,
)
from .lagged_minute_context import (
    CROSSFIT_DAYS,
    fit_triplet,
)
from .prehot_context_ablation import (
    build_ticker_context,
    usable_context_columns,
)
from .pullback_trailing_exit import (
    apply_exit_rule,
    build_pullback_episodes,
)
from .rich_post_hot_state import (
    CURRENT_MINUTE_FEATURES,
    attach_minute_features,
)

REQUEST_ID = 282
RISK_REJECT_FRACTION = 0.50
MIN_SHORTLISTED = 100
MIN_ADMITTED_RESOLVED = 15
MIN_POSITIVE_DAYS = 3
MAX_SEVERE_TRADE_RATE = 0.25
MIN_SEVERE_AUC = 0.65


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def attach_ticker_history(
    candidates: pd.DataFrame,
    raw_scan: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    context, names = build_ticker_context(raw_scan)
    enriched = candidates.merge(
        context,
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
    )
    extra = usable_context_columns(enriched, names)
    return enriched, extra


def prepare_pullback_states(
    candidates: pd.DataFrame,
    ticker_extra: tuple[str, ...],
    raw_scan: pd.DataFrame,
    causal_scan: pd.DataFrame,
    policy: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    scored = candidates.copy()
    # build_pullback_state_rows requires this causal static field. The
    # placeholder is deliberately removed from the learned feature set below.
    scored["candidate_probability"] = 0.5
    states = build_pullback_state_rows(scored, causal_scan)
    states = states.merge(
        policy.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "resolved",
            "trade_return_pct",
            "exit_t",
            "exit_reason",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="one_to_one",
    )
    if states.resolved.isna().any():
        raise ValueError("Request282 state/outcome merge lost rows")

    candidate_context = candidates.loc[:, [
        "trading_day",
        "ticker",
        "t",
        *ticker_extra,
    ]].rename(columns={"t": "hot_t"})
    states = states.merge(
        candidate_context,
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="one_to_one",
    )
    states["state_t"] = _numeric(
        states.entry_t
    ).astype("int64")
    enriched = attach_minute_features(states, raw_scan)

    base_columns = tuple(
        name
        for name in feature_columns(enriched)
        if name != "candidate_probability"
    )
    minute_columns = tuple(
        name
        for name in CURRENT_MINUTE_FEATURES
        if name in enriched
        and _numeric(enriched[name]).notna().any()
    )
    columns = tuple(
        dict.fromkeys([
            *base_columns,
            *ticker_extra,
            *minute_columns,
        ])
    )
    return enriched, columns


def fold_stage1(
    train: pd.DataFrame,
    held: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
) -> tuple[pd.DataFrame, float]:
    fit = train.loc[
        _numeric(train.downstream_value_pct).notna()
    ].copy()
    pair = fit_pair(
        fit,
        "downstream_value_pct",
        columns,
        seed,
    )
    train_scores = pair.regressor.predict(
        xframe(train, columns)
    )
    threshold = threshold_from_training(
        train_scores,
        fraction=SELECTION_FRACTION,
    )
    result = held.loc[:, [
        "trading_day",
        "ticker",
        "hot_t",
    ]].copy()
    result["prehot_predicted_value_pct"] = (
        pair.regressor.predict(xframe(held, columns))
    )
    result["prehot_threshold"] = threshold
    result["selected_prehot"] = (
        result.prehot_predicted_value_pct >= threshold
    )
    return result, threshold


def fold_stage2(
    train_states: pd.DataFrame,
    held_states: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
) -> tuple[pd.DataFrame, float]:
    _, _, severe = fit_triplet(
        train_states,
        columns,
        seed,
    )
    train_probability = severe.predict_proba(
        feature_x(train_states, columns)
    )[:, 1]
    threshold = float(
        np.quantile(
            train_probability,
            1.0 - RISK_REJECT_FRACTION,
        )
    )
    out = held_states.loc[:, [
        "trading_day",
        "ticker",
        "hot_t",
        "resolved",
        "trade_return_pct",
    ]].copy()
    probability = severe.predict_proba(
        feature_x(held_states, columns)
    )[:, 1]
    out["severe_probability"] = probability
    out["severe_threshold"] = threshold
    out["risk_accepted"] = probability <= threshold
    return out, threshold


def two_stage_metrics(
    policy: pd.DataFrame,
    prehot: pd.DataFrame,
    risk: pd.DataFrame,
) -> dict:
    decisions = policy.merge(
        prehot.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "selected_prehot",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="one_to_one",
    )
    decisions = decisions.merge(
        risk.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "risk_accepted",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="one_to_one",
    )
    decisions["selected_prehot"] = (
        decisions.selected_prehot.fillna(False).astype(bool)
    )
    decisions["risk_accepted"] = (
        decisions.risk_accepted.fillna(False).astype(bool)
    )
    entered = decisions.entered.astype(bool)
    resolved = decisions.resolved.astype(bool)
    decisions["admitted"] = (
        decisions.selected_prehot
        & entered
        & decisions.risk_accepted
    )
    decisions["decision_resolved"] = (
        ~decisions.admitted | resolved
    )
    decisions["decision_return_pct"] = 0.0
    admitted_resolved = decisions.admitted & resolved
    decisions.loc[
        admitted_resolved,
        "decision_return_pct",
    ] = _numeric(
        decisions.loc[
            admitted_resolved,
            "trade_return_pct",
        ]
    )
    decisions.loc[
        ~decisions.decision_resolved,
        "decision_return_pct",
    ] = np.nan

    valid = decisions.loc[
        decisions.decision_resolved
    ].copy()
    trades = decisions.loc[
        admitted_resolved
    ].copy()
    returns = _numeric(trades.trade_return_pct)
    daily = valid.groupby(
        valid.trading_day.astype(str),
        sort=True,
    ).decision_return_pct.mean()
    shortlist_trigger = (
        decisions.selected_prehot & entered
    )
    risk_rejected = (
        shortlist_trigger & ~decisions.risk_accepted
    )
    return {
        "candidate_episodes": int(len(decisions)),
        "shortlisted_candidates": int(
            decisions.selected_prehot.sum()
        ),
        "shortlisted_pullback_triggers": int(
            shortlist_trigger.sum()
        ),
        "risk_rejected_triggers": int(risk_rejected.sum()),
        "admitted_triggers": int(decisions.admitted.sum()),
        "unresolved_admitted": int(
            (decisions.admitted & ~resolved).sum()
        ),
        "resolution_or_cash_rate": float(
            decisions.decision_resolved.mean()
        ),
        "candidate_mean_pct": float(
            _numeric(valid.decision_return_pct).mean()
        ),
        "day_balanced_candidate_mean_pct": float(
            daily.mean()
        ),
        "positive_candidate_mean_days": int(
            (daily > 0).sum()
        ),
        "trade_count": int(len(trades)),
        "trade_mean_pct": (
            float(returns.mean()) if len(returns) else None
        ),
        "trade_positive_rate": (
            float(returns.gt(0).mean())
            if len(returns)
            else None
        ),
        "trade_severe_loss_rate_le_minus2": (
            float(returns.le(-2.0).mean())
            if len(returns)
            else None
        ),
        "by_day": {
            str(day): {
                "candidates": int(len(part)),
                "shortlisted": int(
                    part.selected_prehot.sum()
                ),
                "admitted": int(part.admitted.sum()),
                "candidate_mean_pct": float(
                    _numeric(
                        part.loc[
                            part.decision_resolved,
                            "decision_return_pct",
                        ]
                    ).mean()
                ),
            }
            for day, part in decisions.groupby(
                decisions.trading_day.astype(str),
                sort=True,
            )
        },
    }


def severe_oof_auc(states: pd.DataFrame) -> float | None:
    target = _numeric(states.trade_return_pct)
    valid = (
        states.resolved.astype(bool)
        & target.notna()
        & _numeric(states.severe_probability).notna()
    )
    frame = states.loc[valid].copy()
    y = _numeric(frame.trade_return_pct).le(-2.0).astype(int)
    if len(frame) < 20 or y.nunique() != 2:
        return None
    return float(
        roc_auc_score(
            y,
            _numeric(frame.severe_probability),
        )
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
        policy_name="request282_frozen_downstream",
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

    prehot_parts = []
    risk_parts = []
    fold_thresholds = {}
    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train_candidates = candidates.loc[
            ~candidates.trading_day.astype(str).eq(
                held_day
            )
        ].copy()
        held_candidates = candidates.loc[
            candidates.trading_day.astype(str).eq(
                held_day
            )
        ].copy()
        prehot, prehot_threshold = fold_stage1(
            train_candidates,
            held_candidates,
            prehot_columns,
            20261450 + fold_index * 10,
        )
        prehot_parts.append(prehot)

        train_states = states.loc[
            ~states.trading_day.astype(str).eq(held_day)
        ].copy()
        held_states = states.loc[
            states.trading_day.astype(str).eq(held_day)
        ].copy()
        risk, severe_threshold = fold_stage2(
            train_states,
            held_states,
            post_columns,
            20261460 + fold_index * 10,
        )
        risk_parts.append(risk)
        fold_thresholds[str(held_day)] = {
            "prehot_value_threshold": float(
                prehot_threshold
            ),
            "severe_probability_threshold": float(
                severe_threshold
            ),
            "train_pullback_states": int(
                len(train_states)
            ),
            "held_pullback_states": int(
                len(held_states)
            ),
        }

    prehot_oof = pd.concat(
        prehot_parts,
        ignore_index=True,
    )
    risk_oof = pd.concat(
        risk_parts,
        ignore_index=True,
    )
    prehot_baseline = selection_metrics(
        policy,
        prehot_oof,
        "selected_prehot",
    )
    combined = two_stage_metrics(
        policy,
        prehot_oof,
        risk_oof,
    )
    severe_auc = severe_oof_auc(risk_oof)

    baseline_mean = prehot_baseline["candidate_mean_pct"]
    combined_mean = combined["candidate_mean_pct"]
    gain = (
        float(combined_mean - baseline_mean)
        if baseline_mean is not None
        and combined_mean is not None
        else None
    )
    checks = {
        "minimum_shortlisted": (
            combined["shortlisted_candidates"]
            >= MIN_SHORTLISTED
        ),
        "minimum_admitted_resolved": (
            combined["trade_count"]
            >= MIN_ADMITTED_RESOLVED
        ),
        "resolution_or_cash": (
            combined["resolution_or_cash_rate"]
            >= MIN_RESOLUTION_OR_CASH
        ),
        "candidate_mean_positive": (
            combined_mean is not None
            and combined_mean > 0
        ),
        "day_balanced_mean_positive": (
            combined[
                "day_balanced_candidate_mean_pct"
            ] > 0
        ),
        "positive_days": (
            combined["positive_candidate_mean_days"]
            >= MIN_POSITIVE_DAYS
        ),
        "gain_vs_prehot_only": (
            gain is not None and gain > 0
        ),
        "severe_trade_rate": (
            combined[
                "trade_severe_loss_rate_le_minus2"
            ] is not None
            and combined[
                "trade_severe_loss_rate_le_minus2"
            ] <= MAX_SEVERE_TRADE_RATE
        ),
        "severe_oof_auc": (
            severe_auc is not None
            and severe_auc >= MIN_SEVERE_AUC
        ),
    }
    gate = bool(all(checks.values()))

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "stage1_selection_fraction": SELECTION_FRACTION,
        "stage2_risk_reject_fraction": RISK_REJECT_FRACTION,
        "feature_counts": {
            "prehot": len(prehot_columns),
            "posthot": len(post_columns),
            "ticker_history": len(ticker_extra),
        },
        "fixed_downstream_policy": {
            "entry_trigger": "first causal 2% pullback",
            "hard_stop_loss_pct": FIXED_STOP_LOSS_PCT,
            "trailing_drawdown_pct": FIXED_TRAIL_PCT,
            "max_hold_minutes": FIXED_MAX_HOLD_MINUTES,
            "risk_veto_action": "cash",
            "missing_exact_deadline": "unresolved",
        },
        "fold_thresholds": fold_thresholds,
        "pullback_severe_oof_auc": severe_auc,
        "prehot_only_top20": prehot_baseline,
        "two_stage_risk_veto": combined,
        "candidate_mean_gain_vs_prehot_only_pct": gain,
        "checks": checks,
        "two_stage_economic_gate_pass": gate,
        "next_boundary": (
            "freeze the two-stage architecture and validate it without "
            "retuning on May11-20"
            if gate
            else (
                "inspect which risk-veto errors remain and whether a "
                "continuous WAIT/recheck action is required at the pullback "
                "instead of immediate ENTER-versus-CASH"
            )
        ),
        "timing_contract": (
            "stage1 sees only HOT-time and earlier ticker context; stage2 "
            "uses pullback-path prefix plus the strictly completed prior "
            "minute; all thresholds are derived from outer-fold training "
            "days only"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = prehot_oof.merge(
        risk_oof.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "severe_probability",
            "severe_threshold",
            "risk_accepted",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="one_to_one",
    )
    rows.to_parquet(rows_output, index=False)
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
