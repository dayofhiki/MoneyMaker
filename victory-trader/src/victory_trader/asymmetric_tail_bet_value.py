"""Request288: asymmetric tail-state bet value.

Request287 showed that "return > 0" is a poor win target for a momentum bet.
This request models three economically meaningful outcome regimes instead:

- severe downside: return <= -2%
- middle/noise: -2% < return < +2%
- meaningful upside: return >= +2%

The classifier estimates the three causal probabilities.  Each regime receives
its outer-training realized payoff, and their probability-weighted sum is the
predicted bet EV.  Entry/exits remain exactly frozen.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score

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
    feature_x,
)
from .lagged_minute_context import CROSSFIT_DAYS, day_weights
from .pullback_trailing_exit import (
    apply_exit_rule,
    build_pullback_episodes,
)
from .rich_post_hot_state import attach_second_features
from .second_resolution_pullback_state import usable_second_columns
from .shortlist_adapted_risk import (
    STAGE1_CONTEXT,
    attach_stage1_context,
    nested_training_stage1,
    score_stage1,
)
from .two_stage_risk_veto import (
    attach_ticker_history,
    prepare_pullback_states,
)

REQUEST_ID = 288
DOWNSIDE_THRESHOLD_PCT = -2.0
UPSIDE_THRESHOLD_PCT = 2.0
MIN_CLASS_SUPPORT = 15
MIN_CONDITIONED_RESOLVED = 30
MIN_ACCEPTED_RESOLVED = 10
MIN_EV_SPEARMAN_GAIN = 0.05
MIN_POSITIVE_DAYS = 3


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def regime_labels(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.ones(len(values), dtype=int)
    out[values <= DOWNSIDE_THRESHOLD_PCT] = 0
    out[values >= UPSIDE_THRESHOLD_PCT] = 2
    return out


@dataclass(frozen=True)
class TailModels:
    direct: HistGradientBoostingRegressor
    regime: HistGradientBoostingClassifier
    payoffs: tuple[float, float, float]


def fit_tail_models(
    train: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
) -> TailModels:
    target = _numeric(train.trade_return_pct)
    fit = train.loc[
        train.resolved.astype(bool) & target.notna()
    ].copy()
    y = _numeric(fit.trade_return_pct).to_numpy(float)
    if len(fit) < 120:
        raise ValueError(
            f"Request288 only {len(fit)} resolved training pullbacks"
        )

    labels = regime_labels(y)
    counts = np.bincount(labels, minlength=3)
    if int(counts.min()) < MIN_CLASS_SUPPORT:
        raise ValueError(
            f"Request288 regime support too small: {counts.tolist()}"
        )

    weights = day_weights(fit)
    x = feature_x(fit, columns)
    low, high = np.quantile(y, [0.01, 0.99])
    clipped = np.clip(y, low, high)

    direct = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=11,
        min_samples_leaf=15,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=seed,
    )
    direct.fit(x, clipped, sample_weight=weights)

    regime = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=200,
        max_leaf_nodes=11,
        min_samples_leaf=15,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=seed + 1,
    )
    regime.fit(x, labels, sample_weight=weights)

    payoffs = []
    for label in (0, 1, 2):
        mask = labels == label
        payoffs.append(
            float(np.average(clipped[mask], weights=weights[mask]))
        )
    return TailModels(
        direct=direct,
        regime=regime,
        payoffs=tuple(payoffs),
    )


def score_tail_bets(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    models: TailModels,
) -> pd.DataFrame:
    out = frame.copy()
    x = feature_x(out, columns)
    raw = models.regime.predict_proba(x)
    probabilities = np.zeros((len(out), 3), dtype=float)
    for index, label in enumerate(models.regime.classes_):
        probabilities[:, int(label)] = raw[:, index]

    payoff = np.asarray(models.payoffs, dtype=float)
    ev = probabilities @ payoff
    out["direct_predicted_return_pct"] = models.direct.predict(x)
    out["tail_downside_probability"] = probabilities[:, 0]
    out["tail_middle_probability"] = probabilities[:, 1]
    out["tail_upside_probability"] = probabilities[:, 2]
    out["tail_downside_payoff_pct"] = payoff[0]
    out["tail_middle_payoff_pct"] = payoff[1]
    out["tail_upside_payoff_pct"] = payoff[2]
    out["tail_expected_value_pct"] = ev
    out["tail_take"] = ev > 0.0
    return out


def _spearman(actual: pd.Series, score: pd.Series) -> float | None:
    a = _numeric(actual)
    s = _numeric(score)
    valid = a.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    value = a.loc[valid].corr(s.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _auc(target: pd.Series, score: pd.Series) -> float | None:
    y = _numeric(target)
    s = _numeric(score)
    valid = y.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    y = y.loc[valid].astype(int)
    if y.nunique() != 2:
        return None
    return float(roc_auc_score(y, s.loc[valid]))


def _quartiles(work: pd.DataFrame) -> dict:
    if len(work) < 20:
        return {}
    ranked = _numeric(work.tail_expected_value_pct).rank(
        method="first",
        pct=True,
    )
    bucket = np.minimum(
        4,
        np.maximum(1, np.ceil(ranked * 4).astype(int)),
    )
    temp = work.assign(_quartile=bucket)
    out = {}
    for q, part in temp.groupby("_quartile", sort=True):
        actual = _numeric(part.trade_return_pct)
        out[f"q{int(q)}"] = {
            "rows": int(len(part)),
            "predicted_ev_mean_pct": float(
                _numeric(part.tail_expected_value_pct).mean()
            ),
            "realized_trade_mean_pct": float(actual.mean()),
            "upside_rate_ge_plus2": float(
                actual.ge(UPSIDE_THRESHOLD_PCT).mean()
            ),
            "severe_rate_le_minus2": float(
                actual.le(DOWNSIDE_THRESHOLD_PCT).mean()
            ),
        }
    return out


def tail_metrics(frame: pd.DataFrame) -> dict:
    shortlist = frame.loc[
        frame.selected_prehot.fillna(False).astype(bool)
    ].copy()
    resolved = shortlist.loc[
        shortlist.resolved.astype(bool)
        & _numeric(shortlist.trade_return_pct).notna()
    ].copy()
    actual = _numeric(resolved.trade_return_pct)
    labels = regime_labels(actual.to_numpy(float))
    downside = pd.Series(labels == 0, index=resolved.index).astype(int)
    upside = pd.Series(labels == 2, index=resolved.index).astype(int)

    take = resolved.tail_take.fillna(False).astype(bool)
    accepted = resolved.loc[take].copy()
    rejected = resolved.loc[~take].copy()
    accepted_return = _numeric(accepted.trade_return_pct)
    rejected_return = _numeric(rejected.trade_return_pct)

    decision = shortlist.copy()
    decision["decision_resolved"] = (
        ~decision.tail_take.fillna(False).astype(bool)
        | decision.resolved.astype(bool)
    )
    decision["decision_return_pct"] = 0.0
    active = (
        decision.tail_take.fillna(False).astype(bool)
        & decision.resolved.astype(bool)
    )
    decision.loc[active, "decision_return_pct"] = _numeric(
        decision.loc[active, "trade_return_pct"]
    )
    decision.loc[
        ~decision.decision_resolved,
        "decision_return_pct",
    ] = np.nan
    valid_decisions = decision.loc[
        decision.decision_resolved
    ].copy()
    daily = valid_decisions.groupby(
        valid_decisions.trading_day.astype(str),
        sort=True,
    ).decision_return_pct.mean()

    direct_sp = _spearman(
        actual,
        resolved.direct_predicted_return_pct,
    )
    ev_sp = _spearman(
        actual,
        resolved.tail_expected_value_pct,
    )

    return {
        "shortlisted_pullback_rows": int(len(shortlist)),
        "resolved_shortlist_rows": int(len(resolved)),
        "resolved_regime_counts": {
            "severe_downside": int((labels == 0).sum()),
            "middle": int((labels == 1).sum()),
            "meaningful_upside": int((labels == 2).sum()),
        },
        "baseline_all_trade_mean_pct": (
            float(actual.mean()) if len(actual) else None
        ),
        "direct_value_spearman": direct_sp,
        "tail_ev_spearman": ev_sp,
        "tail_ev_spearman_gain_vs_direct": (
            float(ev_sp - direct_sp)
            if ev_sp is not None and direct_sp is not None
            else None
        ),
        "downside_auc": _auc(
            downside,
            resolved.tail_downside_probability,
        ),
        "upside_auc": _auc(
            upside,
            resolved.tail_upside_probability,
        ),
        "ev_positive_bets_resolved": int(len(accepted)),
        "ev_nonpositive_rejected_resolved": int(len(rejected)),
        "ev_positive_trade_mean_pct": (
            float(accepted_return.mean())
            if len(accepted_return)
            else None
        ),
        "ev_positive_trade_upside_rate_ge_plus2": (
            float(accepted_return.ge(UPSIDE_THRESHOLD_PCT).mean())
            if len(accepted_return)
            else None
        ),
        "ev_positive_trade_severe_rate_le_minus2": (
            float(accepted_return.le(DOWNSIDE_THRESHOLD_PCT).mean())
            if len(accepted_return)
            else None
        ),
        "rejected_counterfactual_trade_mean_pct": (
            float(rejected_return.mean())
            if len(rejected_return)
            else None
        ),
        "accepted_minus_rejected_mean_pct": (
            float(accepted_return.mean() - rejected_return.mean())
            if len(accepted_return) and len(rejected_return)
            else None
        ),
        "cash_adjusted_decision_mean_pct": (
            float(_numeric(valid_decisions.decision_return_pct).mean())
            if len(valid_decisions)
            else None
        ),
        "decision_resolution_or_cash_rate": (
            float(decision.decision_resolved.mean())
            if len(decision)
            else 0.0
        ),
        "positive_decision_mean_days": int((daily > 0).sum()),
        "by_day": {
            str(day): {
                "decisions": int(len(part)),
                "bets": int(
                    part.tail_take.fillna(False).astype(bool).sum()
                ),
                "decision_mean_pct": float(
                    _numeric(
                        part.loc[
                            part.decision_resolved,
                            "decision_return_pct",
                        ]
                    ).mean()
                ),
            }
            for day, part in decision.groupby(
                decision.trading_day.astype(str),
                sort=True,
            )
        },
        "ev_quartiles_low_to_high": _quartiles(resolved),
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

    policy = apply_exit_rule(
        build_pullback_episodes(first_hot, causal_scan),
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name="request288_frozen_h0",
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
    states, baseline_columns = prepare_pullback_states(
        candidates,
        ticker_extra,
        raw_scan,
        causal_scan,
        policy,
    )
    states, second_audit = attach_second_features(states)
    second_columns = usable_second_columns(states)
    representation_columns = tuple(
        dict.fromkeys([
            *baseline_columns,
            *second_columns,
            *STAGE1_CONTEXT,
        ])
    )

    parts = []
    fold_audit = {}
    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        outer_train_candidates = candidates.loc[
            ~candidates.trading_day.astype(str).eq(held_day)
        ].copy()
        held_candidates = candidates.loc[
            candidates.trading_day.astype(str).eq(held_day)
        ].copy()

        nested_scores = nested_training_stage1(
            outer_train_candidates,
            prehot_columns,
            20262500 + fold_index * 100,
        )
        held_scores, held_threshold = score_stage1(
            outer_train_candidates,
            held_candidates,
            prehot_columns,
            20262550 + fold_index * 100,
        )

        train_states = states.loc[
            ~states.trading_day.astype(str).eq(held_day)
        ].copy()
        held_states = states.loc[
            states.trading_day.astype(str).eq(held_day)
        ].copy()
        train_states = attach_stage1_context(
            train_states,
            nested_scores,
        )
        held_states = attach_stage1_context(
            held_states,
            held_scores,
        )

        models = fit_tail_models(
            train_states,
            representation_columns,
            20262600 + fold_index * 100,
        )
        scored = score_tail_bets(
            held_states,
            representation_columns,
            models,
        )
        scored["fold_holdout_day"] = held_day
        parts.append(scored)
        fold_audit[str(held_day)] = {
            "prehot_threshold": float(held_threshold),
            "train_pullback_states": int(len(train_states)),
            "held_pullback_states": int(len(held_states)),
            "held_shortlisted_pullbacks": int(
                held_states.selected_prehot
                .fillna(False)
                .astype(bool)
                .sum()
            ),
            "regime_training_payoffs_pct": {
                "severe_downside": float(models.payoffs[0]),
                "middle": float(models.payoffs[1]),
                "meaningful_upside": float(models.payoffs[2]),
            },
        }

    oof = pd.concat(parts, ignore_index=True)
    metrics = tail_metrics(oof)
    gain = metrics["tail_ev_spearman_gain_vs_direct"]
    accepted_mean = metrics["ev_positive_trade_mean_pct"]
    baseline_mean = metrics["baseline_all_trade_mean_pct"]
    cash_mean = metrics["cash_adjusted_decision_mean_pct"]

    checks = {
        "conditioned_support": (
            metrics["resolved_shortlist_rows"]
            >= MIN_CONDITIONED_RESOLVED
        ),
        "accepted_support": (
            metrics["ev_positive_bets_resolved"]
            >= MIN_ACCEPTED_RESOLVED
        ),
        "ev_ranking_gain_vs_direct": (
            gain is not None
            and gain >= MIN_EV_SPEARMAN_GAIN
        ),
        "accepted_beats_all_shortlist": (
            accepted_mean is not None
            and baseline_mean is not None
            and accepted_mean > baseline_mean
        ),
        "accepted_trade_mean_positive": (
            accepted_mean is not None
            and accepted_mean > 0
        ),
        "cash_adjusted_mean_positive": (
            cash_mean is not None
            and cash_mean > 0
        ),
        "positive_days": (
            metrics["positive_decision_mean_days"]
            >= MIN_POSITIVE_DAYS
        ),
    }
    gate = bool(all(checks.values()))

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "decision_philosophy": (
            "price asymmetric momentum outcomes instead of demanding binary "
            "prediction correctness"
        ),
        "outcome_regimes": {
            "severe_downside": "<= -2%",
            "middle": "(-2%, +2%)",
            "meaningful_upside": ">= +2%",
        },
        "fixed_execution": {
            "entry": "original first causal 2% pullback",
            "hard_stop_loss_pct": FIXED_STOP_LOSS_PCT,
            "trailing_drawdown_pct": FIXED_TRAIL_PCT,
            "max_hold_minutes": FIXED_MAX_HOLD_MINUTES,
            "missing_exact_deadline": "unresolved",
        },
        "bet_value_formula": (
            "sum over {downside,middle,upside}: "
            "P(regime|causal state) * outer-training mean payoff(regime)"
        ),
        "bet_action": "ENTER iff predicted tail EV > 0; otherwise CASH",
        "feature_counts": {
            "baseline": len(baseline_columns),
            "completed_second_extra": len(second_columns),
            "stage1_context": len(STAGE1_CONTEXT),
            "total": len(representation_columns),
        },
        "second_data_audit": second_audit,
        "fold_audit": fold_audit,
        "metrics": metrics,
        "checks": checks,
        "asymmetric_tail_bet_gate_pass": gate,
        "timing_contract": (
            "stage1 context is outer/nested out-of-fold; pullback features "
            "use only causal path and completed seconds through decision_t-1s; "
            "regime payoffs are derived from outer-fold training outcomes only"
        ),
        "next_boundary": (
            "if tail EV orders outcomes, translate EV and downside probability "
            "into causal stake sizing before adding WAIT"
            if gate
            else (
                "if upside remains unlearnable while downside remains useful, "
                "stop trying to forecast the exact winner at h0 and test whether "
                "short fixed sub-minute rechecks create an upside signal cheaply"
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
