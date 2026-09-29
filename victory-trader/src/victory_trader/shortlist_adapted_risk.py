"""Request284: stage-2 severe-risk adaptation to the stage-1 shortlist.

Request283 showed that the global severe-risk model loses most of its ordering
inside the exact pre-HOT shortlist. This request changes no trading action.
It compares the original global severe-risk model with a shortlist-aware model
that:

1. adds the causal stage-1 predicted value and value margin to stage-2 context;
2. emphasizes training pullbacks that would have entered the stage-1 shortlist;
3. derives every shortlist-emphasis flag without using the outer held-out day
   or the row's own downstream label.

The final evaluation remains May5-8 leave-one-day-out and is conditioned on the
held-out stage-1 shortlist. No new dates are opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .causal_minute_controller_rebuild import causal_execution_scan
from .downstream_aligned_candidate import (
    SELECTION_FRACTION,
    attach_downstream_target,
    candidate_columns,
    fit_pair,
    threshold_from_training,
)
from .full_hot_fixed_policy_value import attach_fixed_value, xframe
from .joint_pullback_admission import (
    FIXED_MAX_HOLD_MINUTES,
    FIXED_STOP_LOSS_PCT,
    FIXED_TRAIL_PCT,
    feature_x,
)
from .lagged_minute_context import (
    CROSSFIT_DAYS,
    day_weights,
    fit_triplet,
)
from .pullback_trailing_exit import (
    apply_exit_rule,
    build_pullback_episodes,
)
from .two_stage_risk_veto import (
    attach_ticker_history,
    prepare_pullback_states,
)

REQUEST_ID = 284
SHORTLIST_WEIGHT = 4.0
RISK_REJECT_FRACTION = 0.50
MIN_CONDITIONED_ROWS = 30
MIN_CONDITIONED_AUC = 0.62
MIN_CONDITIONED_AP_LIFT = 0.08
MIN_AUC_GAIN_VS_GLOBAL = 0.05
MIN_REJECTED_SEVERE_GAP = 0.15

STAGE1_CONTEXT = (
    "prehot_predicted_value_pct",
    "prehot_value_margin_pct",
)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def score_stage1(
    fit_candidates: pd.DataFrame,
    score_candidates: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
) -> tuple[pd.DataFrame, float]:
    fit = fit_candidates.loc[
        _numeric(fit_candidates.downstream_value_pct).notna()
    ].copy()
    pair = fit_pair(
        fit,
        "downstream_value_pct",
        columns,
        seed,
    )
    fit_prediction = pair.regressor.predict(
        xframe(fit_candidates, columns)
    )
    threshold = threshold_from_training(
        fit_prediction,
        fraction=SELECTION_FRACTION,
    )
    prediction = pair.regressor.predict(
        xframe(score_candidates, columns)
    )
    out = score_candidates.loc[:, [
        "trading_day",
        "ticker",
        "hot_t",
    ]].copy()
    out["prehot_predicted_value_pct"] = prediction
    out["prehot_threshold"] = threshold
    out["prehot_value_margin_pct"] = prediction - threshold
    out["selected_prehot"] = prediction >= threshold
    return out, float(threshold)


def nested_training_stage1(
    outer_train: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
) -> pd.DataFrame:
    """OOF stage-1 context using only outer-training days.

    Each outer-training day is scored by a model fit on the other two days.
    Therefore neither the outer held-out day nor the scored row's own target is
    used to decide whether that row receives shortlist emphasis in stage-2 fit.
    """
    days = tuple(
        sorted(outer_train.trading_day.astype(str).unique())
    )
    if len(days) != 3:
        raise ValueError(
            f"Request284 expected three outer-training days, got {days}"
        )
    parts = []
    for nested_index, held_day in enumerate(days):
        nested_fit = outer_train.loc[
            ~outer_train.trading_day.astype(str).eq(held_day)
        ].copy()
        nested_held = outer_train.loc[
            outer_train.trading_day.astype(str).eq(held_day)
        ].copy()
        scored, _ = score_stage1(
            nested_fit,
            nested_held,
            columns,
            seed + nested_index * 10,
        )
        parts.append(scored)
    return pd.concat(parts, ignore_index=True)


def attach_stage1_context(
    states: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    return states.merge(
        scores.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "prehot_predicted_value_pct",
            "prehot_value_margin_pct",
            "selected_prehot",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="one_to_one",
    )


def fit_adapted_severe(
    train: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
) -> HistGradientBoostingClassifier:
    target = _numeric(train.trade_return_pct)
    fit = train.loc[
        train.resolved.astype(bool) & target.notna()
    ].copy()
    y = _numeric(fit.trade_return_pct).le(-2.0).astype(int)
    if len(fit) < 120 or y.nunique() != 2:
        raise ValueError(
            "Request284 adapted severe fit lacks support/classes"
        )
    weights = day_weights(fit)
    emphasis = np.where(
        fit.selected_prehot.fillna(False).astype(bool).to_numpy(),
        SHORTLIST_WEIGHT,
        1.0,
    )
    weights = weights * emphasis
    weights = weights / float(np.mean(weights))

    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(
        feature_x(fit, columns),
        y.to_numpy(int),
        sample_weight=weights,
    )
    return model


def fit_global_severe(
    train: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
) -> HistGradientBoostingClassifier:
    _, _, severe = fit_triplet(train, columns, seed)
    return severe


def threshold_from_shortlist_train(
    probabilities: np.ndarray,
    selected: pd.Series,
) -> float:
    values = np.asarray(probabilities, dtype=float)
    mask = selected.fillna(False).astype(bool).to_numpy()
    source = values[mask & np.isfinite(values)]
    if len(source) < 15:
        source = values[np.isfinite(values)]
    if len(source) < 20:
        raise ValueError("Request284 risk-threshold support too small")
    return float(
        np.quantile(
            source,
            1.0 - RISK_REJECT_FRACTION,
        )
    )


def diagnostics(
    frame: pd.DataFrame,
    probability_column: str,
    threshold_column: str,
) -> dict:
    work = frame.loc[
        frame.resolved.astype(bool)
        & _numeric(frame.trade_return_pct).notna()
        & _numeric(frame[probability_column]).notna()
    ].copy()
    target = _numeric(work.trade_return_pct)
    severe = target.le(-2.0).astype(int)
    score = _numeric(work[probability_column])
    auc = (
        float(roc_auc_score(severe, score))
        if len(work) >= 20 and severe.nunique() == 2
        else None
    )
    ap = (
        float(average_precision_score(severe, score))
        if len(work) and severe.nunique() == 2
        else None
    )
    prevalence = float(severe.mean()) if len(severe) else None
    ap_lift = (
        float(ap - prevalence)
        if ap is not None and prevalence is not None
        else None
    )

    rejected = score > _numeric(work[threshold_column])
    accepted = ~rejected
    rejected_rows = work.loc[rejected]
    accepted_rows = work.loc[accepted]

    def mean_return(rows: pd.DataFrame) -> float | None:
        values = _numeric(rows.trade_return_pct)
        return float(values.mean()) if len(values) else None

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
    quartiles = []
    if len(work) >= 8:
        ranked = work.copy()
        ranked["_bucket"] = pd.qcut(
            _numeric(ranked[probability_column]).rank(method="first"),
            q=4,
            labels=False,
            duplicates="drop",
        )
        for bucket, part in ranked.groupby("_bucket", sort=True):
            values = _numeric(part.trade_return_pct)
            quartiles.append({
                "bucket": int(bucket),
                "rows": int(len(part)),
                "mean_risk": float(
                    _numeric(part[probability_column]).mean()
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
        "rejected_trade_mean_pct": mean_return(rejected_rows),
        "accepted_trade_mean_pct": mean_return(accepted_rows),
        "rejected_positive_rate": positive_rate(rejected_rows),
        "accepted_positive_rate": positive_rate(accepted_rows),
        "rejected_severe_rate": rejected_severe,
        "accepted_severe_rate": accepted_severe,
        "rejected_minus_accepted_severe_rate": severe_gap,
        "risk_quartiles": quartiles,
    }


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
        policy_name="request284_frozen_downstream",
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
    adapted_columns = tuple(
        dict.fromkeys([
            *post_columns,
            *STAGE1_CONTEXT,
        ])
    )

    parts = []
    thresholds = {}
    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        outer_train_candidates = candidates.loc[
            ~candidates.trading_day.astype(str).eq(held_day)
        ].copy()
        held_candidates = candidates.loc[
            candidates.trading_day.astype(str).eq(held_day)
        ].copy()

        nested_train_stage1 = nested_training_stage1(
            outer_train_candidates,
            prehot_columns,
            20261600 + fold_index * 100,
        )
        held_stage1, stage1_threshold = score_stage1(
            outer_train_candidates,
            held_candidates,
            prehot_columns,
            20261650 + fold_index * 100,
        )

        train_states = states.loc[
            ~states.trading_day.astype(str).eq(held_day)
        ].copy()
        held_states = states.loc[
            states.trading_day.astype(str).eq(held_day)
        ].copy()
        train_states = attach_stage1_context(
            train_states,
            nested_train_stage1,
        )
        held_states = attach_stage1_context(
            held_states,
            held_stage1,
        )

        global_model = fit_global_severe(
            train_states,
            post_columns,
            20261700 + fold_index * 10,
        )
        adapted_model = fit_adapted_severe(
            train_states,
            adapted_columns,
            20261750 + fold_index * 10,
        )

        global_train_prob = global_model.predict_proba(
            feature_x(train_states, post_columns)
        )[:, 1]
        adapted_train_prob = adapted_model.predict_proba(
            feature_x(train_states, adapted_columns)
        )[:, 1]
        global_threshold = threshold_from_shortlist_train(
            global_train_prob,
            train_states.selected_prehot,
        )
        adapted_threshold = threshold_from_shortlist_train(
            adapted_train_prob,
            train_states.selected_prehot,
        )

        held = held_states.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "resolved",
            "trade_return_pct",
            "prehot_predicted_value_pct",
            "prehot_value_margin_pct",
            "selected_prehot",
        ]].copy()
        held["global_severe_probability"] = (
            global_model.predict_proba(
                feature_x(held_states, post_columns)
            )[:, 1]
        )
        held["adapted_severe_probability"] = (
            adapted_model.predict_proba(
                feature_x(held_states, adapted_columns)
            )[:, 1]
        )
        held["global_severe_threshold"] = global_threshold
        held["adapted_severe_threshold"] = adapted_threshold
        held["fold_holdout_day"] = held_day
        parts.append(held)
        thresholds[str(held_day)] = {
            "stage1_threshold": float(stage1_threshold),
            "global_risk_threshold": float(global_threshold),
            "adapted_risk_threshold": float(adapted_threshold),
            "outer_train_pullback_states": int(len(train_states)),
            "nested_train_shortlisted_pullbacks": int(
                train_states.selected_prehot.fillna(False).astype(bool).sum()
            ),
            "held_shortlisted_pullbacks": int(
                held.selected_prehot.fillna(False).astype(bool).sum()
            ),
        }

    oof = pd.concat(parts, ignore_index=True)
    conditioned = oof.loc[
        oof.selected_prehot.fillna(False).astype(bool)
    ].copy()

    global_diag = diagnostics(
        conditioned,
        "global_severe_probability",
        "global_severe_threshold",
    )
    adapted_diag = diagnostics(
        conditioned,
        "adapted_severe_probability",
        "adapted_severe_threshold",
    )
    auc_gain = (
        float(adapted_diag["auc"] - global_diag["auc"])
        if adapted_diag["auc"] is not None
        and global_diag["auc"] is not None
        else None
    )

    checks = {
        "conditioned_support": (
            adapted_diag["resolved_rows"] >= MIN_CONDITIONED_ROWS
        ),
        "adapted_auc": (
            adapted_diag["auc"] is not None
            and adapted_diag["auc"] >= MIN_CONDITIONED_AUC
        ),
        "adapted_ap_lift": (
            adapted_diag[
                "average_precision_lift_vs_prevalence"
            ] is not None
            and adapted_diag[
                "average_precision_lift_vs_prevalence"
            ] >= MIN_CONDITIONED_AP_LIFT
        ),
        "auc_gain_vs_global": (
            auc_gain is not None
            and auc_gain >= MIN_AUC_GAIN_VS_GLOBAL
        ),
        "rejected_severe_gap": (
            adapted_diag[
                "rejected_minus_accepted_severe_rate"
            ] is not None
            and adapted_diag[
                "rejected_minus_accepted_severe_rate"
            ] >= MIN_REJECTED_SEVERE_GAP
        ),
    }
    gate = bool(all(checks.values()))

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "shortlist_weight": SHORTLIST_WEIGHT,
        "stage1_selection_fraction": SELECTION_FRACTION,
        "feature_counts": {
            "prehot": len(prehot_columns),
            "global_posthot": len(post_columns),
            "adapted_posthot": len(adapted_columns),
        },
        "fold_thresholds": thresholds,
        "global_model_conditioned": global_diag,
        "shortlist_adapted_model_conditioned": adapted_diag,
        "conditioned_auc_gain_vs_global": auc_gain,
        "checks": checks,
        "shortlist_adapted_signal_pass": gate,
        "timing_contract": (
            "outer-held day is excluded from every stage-2 fit; shortlist "
            "emphasis for outer-training states is generated by nested "
            "stage-1 models that exclude each scored training day; the "
            "held-day stage-1 score uses only outer-training days"
        ),
        "next_boundary": (
            "use the adapted conditioned risk state in a causal "
            "ENTER/WAIT/REJECT recheck controller"
            if gate
            else (
                "shortlist-conditioned risk remains too weak; test whether "
                "new information after the first pullback changes outcome "
                "separability before adding a recurrent action policy"
            )
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    oof.to_parquet(rows_output, index=False)
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
