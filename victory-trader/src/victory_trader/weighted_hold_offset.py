"""R303: isolate weighted training-residual calibration for frozen pi1."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .clock_context_hold_value import paired_intervals, risk_upside
from .frozen_entry_second_hold_exit import KEYS
from .hierarchical_crack_entry_controller import _episode_day_weights, _predict
from .hold_policy_stage_audit import SELF_TARGET, mismatch_audit
from .lagged_minute_context import CROSSFIT_DAYS
from .pending_exit_integrity import regular_bars, session_limits
from .preentry_context_hold_value import ALL_FEATURES, PREDICTION, availability, common_diagnostics
from .risk_reachable_hold_value import (
    TARGET_PI0, arm_report, continuation_targets, model_fit,
    replay_scored, signal, validate_fit_days, verify_reproduction,
)

REQUEST_ID = 303


def weighted_offset_model(train: pd.DataFrame, fitted):
    """Change only offset; original model/columns/clipping remain untouched."""
    if fitted.log_target:
        raise ValueError("R303 requires non-log pi1")
    target = pd.to_numeric(train[TARGET_PI0], errors="coerce")
    fit = train.loc[train.risk_reachable_hold & target.notna()].copy()
    y = pd.to_numeric(fit[TARGET_PI0]).to_numpy(float)
    weights = _episode_day_weights(fit)
    # _predict includes the historical unweighted offset. Subtract it to
    # recover the identical raw tree prediction without touching fitted trees.
    raw = _predict(fit, fitted)-fitted.offset
    if not np.isfinite(y).all() or not np.isfinite(raw).all():
        raise ValueError("nonfinite training residual")
    offset = float(np.average(y-raw, weights=weights))
    corrected = replace(fitted, offset=offset)
    audit = {
        "training_rows": len(fit),
        "old_offset_pp": fitted.offset,
        "weighted_offset_pp": offset,
        "score_shift_pp": offset-fitted.offset,
        "weighted_residual_before_pp": float(np.average(y-raw-fitted.offset, weights=weights)),
        "weighted_residual_after_pp": float(np.average(y-_predict(fit, corrected), weights=weights)),
    }
    if abs(audit["weighted_residual_after_pp"]) > 1e-9:
        raise ValueError("weighted residual not centered")
    return corrected, audit


def crossfit_weighted_offset(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parts, provenance = [], {}
    for index, held_day in enumerate(CROSSFIT_DAYS):
        train = source.loc[source.trading_day.ne(held_day)]
        held = source.loc[source.trading_day.eq(held_day)].copy()
        fit_days = sorted(train.trading_day.unique())
        validate_fit_days(fit_days, excluded=[held_day])
        seed = 20263975+index*100
        baseline = model_fit(train, TARGET_PI0, seed, ALL_FEATURES)
        old_prediction = _predict(held, baseline)
        saved_error = float(np.max(np.abs(old_prediction-held[PREDICTION].to_numpy(float))))
        if saved_error > 1e-9:
            raise ValueError("saved R302 pi1 prediction mismatch")
        corrected, audit = weighted_offset_model(train, baseline)
        held[PREDICTION] = _predict(held, corrected)
        if not np.allclose(held[PREDICTION]-old_prediction, audit["score_shift_pp"], atol=1e-9, rtol=0):
            raise ValueError("offset changed more than a constant score shift")
        parts.append(held)
        provenance[held_day] = {
            "fit_days": fit_days, "seed": seed, "target": TARGET_PI0,
            "features": list(ALL_FEATURES), "max_saved_prediction_error": saved_error,
            "held_score_sign_changes": int((held[PREDICTION].gt(0).to_numpy() != (old_prediction > 0)).sum()),
            **audit,
        }
        print(f"R303 weighted offset fold complete {held_day}", flush=True)
    return pd.concat(parts, ignore_index=True), provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("states", "decisions", "reference-decisions", "raw-seconds", "output", "states-output", "decisions-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_parquet(args.states)
    source = source.loc[source.arm.eq("B")].copy().reset_index(drop=True)
    if set(source.trading_day.astype(str)) != set(CROSSFIT_DAYS) or source.duplicated([*KEYS, "decision_t"]).any() or len(source[KEYS].drop_duplicates()) != 86:
        raise ValueError("R303 requires exact unique 86 May5-8 positions")
    raw = pd.read_parquet(args.raw_seconds)
    if not set(raw.trading_day.astype(str)).issubset(CROSSFIT_DAYS):
        raise ValueError("new raw dates forbidden")
    contexts = {}
    for key, _ in source.groupby(KEYS, sort=False):
        opening, closing = session_limits(str(key[0]))
        bars = raw.loc[raw.trading_day.astype(str).eq(str(key[0])) & raw.ticker.astype(str).eq(str(key[1]))]
        contexts[key] = (regular_bars(bars, opening, closing), opening, closing)
    original = pd.read_parquet(args.decisions)
    original = original.loc[original.arm.eq("B_fixed_reference_pi1")]
    A = replay_scored(source, contexts, "A_R302_unweighted_offset", next_event=True)
    reproduction = verify_reproduction(A, original)
    print("R303 exact R302 pi1 replay passed", flush=True)
    Bstates, folds = crossfit_weighted_offset(source)
    B = replay_scored(Bstates, contexts, "B_weighted_offset_pi1", next_event=True)
    Astate = continuation_targets(source, contexts, prediction_column=PREDICTION, output_column=SELF_TARGET)
    Bstate = continuation_targets(Bstates, contexts, prediction_column=PREDICTION, output_column=SELF_TARGET)
    # Outer future outcomes are diagnostic only; both models were fitted above.
    reference = pd.read_parquet(args.reference_decisions)
    old = reference.loc[reference.stage.eq("rebuilt_label_refit") & reference.policy.eq("old_10m_2pct_trailing")].copy()
    arms, frames = {"A": A, "B": B}, {"A": Astate, "B": Bstate}
    summaries, common, support, mismatch = {}, {}, {}, {}
    for name, decisions in arms.items():
        summary = arm_report(decisions, old)
        summary["mean_gross_pct"] = float(decisions.gross_return_pct.mean())
        summary["mean_cost_drag_pp"] = float((decisions.gross_return_pct-decisions.base_net_return_pct).mean())
        summary["median_hold_seconds"] = float(decisions.hold_seconds.median())
        summary["risk_reachable_upside"] = risk_upside(source, contexts, decisions)
        summaries[name] = summary
        common[name] = common_diagnostics(frames[name], source, decisions)
        support[name] = availability(frames[name], decisions)
        mismatch[name] = mismatch_audit(frames[name], decisions, TARGET_PI0)
    comparisons = {"B_minus_A": paired_intervals(B, A), "B_minus_old": paired_intervals(B, old)}
    rank = signal(Bstate, TARGET_PI0, PREDICTION)["day_episode_weighted_rank_correlation"]
    primary = summaries["B"]
    checks = {
        "exact_R302_replay": reproduction["maximum_return_error"] <= 1e-9,
        "exact_saved_pi1_predictions": all(fold["max_saved_prediction_error"] <= 1e-9 for fold in folds.values()),
        "weighted_training_residual_zero": all(abs(fold["weighted_residual_after_pp"]) <= 1e-9 for fold in folds.values()),
        "all_entries_reconciled": all(len(frame) == 86 for frame in arms.values()),
        "resolution_at_least_95pct": primary["resolution_rate"] >= .95,
        "positive_base_mean": primary["mean_base_net_pct_resolved"] > 0,
        "positive_at_least_three_days": sum(v > 0 for v in primary["daily_base_net_pct"].values()) >= 3,
        "beats_old_matched": comparisons["B_minus_old"]["mean_gain_pp"] > 0,
        "beats_R302_unweighted_offset": comparisons["B_minus_A"]["mean_gain_pp"] > 0,
        "positive_without_largest_trade": primary["mean_without_largest_trade_pct"] > 0,
        "positive_weighted_learned_target_ranking": rank is not None and rank > 0,
    }
    result = {
        "request_id": REQUEST_ID, "primary": "B_weighted_offset_pi1", "development_only": True,
        "promotion_eligible": False, "June_HOLD_opened": False, "network_requests": 0,
        "candidate_episodes": 97, "entries": 86, "R302_reproduction": reproduction,
        "arms": summaries, "frozen_pi0_reference": arm_report(old.assign(policy="next_event_primary"), old),
        "common_label_diagnostics": common, "reference_vs_self_policy_audit": mismatch,
        "feature_availability_and_early_exits": support, "attribution": comparisons,
        "offset_outer_folds": folds, "checks": checks, "research_gate_pass": all(checks.values()),
        "target_contract": "Both arms learn fixed WAIT-follow-pi0. Only the additive training residual offset changes, from unweighted to day/episode weighted. Original unclipped target residuals are retained. Self-follow labels are held-out diagnostics, never fitted or selected.",
        "limitations": "Four reused days; upstream entry OOF not nested; next-print reference fills are not quotes; event mean is not account return; bounded off-policy improvement is not convergence. No result-driven threshold or arm selection.",
    }
    for path in (args.output, args.states_output, args.decisions_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([frame.assign(arm=name) for name, frame in frames.items()], ignore_index=True).to_parquet(args.states_output, index=False)
    pd.concat(list(arms.values()), ignore_index=True).to_parquet(args.decisions_output, index=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({"checks": checks, "research_gate_pass": all(checks.values())}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
