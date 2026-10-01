"""R302: compare fixed-reference pi1 to nested pi2 and audit self continuation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clock_context_hold_value import paired_intervals, risk_upside
from .frozen_entry_second_hold_exit import KEYS
from .hierarchical_crack_entry_controller import _predict
from .lagged_minute_context import CROSSFIT_DAYS
from .pending_exit_integrity import regular_bars, session_limits
from .preentry_context_hold_value import ALL_FEATURES, PREDICTION, availability, calibration, common_diagnostics
from .risk_reachable_hold_value import (
    TARGET_PI0, TARGET_PI1, arm_report, continuation_targets, model_fit,
    replay_scored, signal, validate_fit_days, verify_reproduction,
)

REQUEST_ID = 302
SELF_TARGET = "wait_follow_deployed_policy_advantage_pct"


def crossfit_one_stage(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parts, provenance = [], {}
    for index, held_day in enumerate(CROSSFIT_DAYS):
        train = source.loc[source.trading_day.ne(held_day)]
        held = source.loc[source.trading_day.eq(held_day)].copy()
        fit_days = sorted(train.trading_day.unique())
        validate_fit_days(fit_days, excluded=[held_day])
        seed = 20263975+index*100
        model = model_fit(train, TARGET_PI0, seed, ALL_FEATURES)
        predictions = _predict(held, model)
        if not np.allclose(predictions, held.pi1_prediction_pct, atol=1e-9, rtol=0):
            raise ValueError("saved R301 outer pi1 prediction mismatch")
        held[PREDICTION] = predictions
        parts.append(held)
        provenance[held_day] = {"fit_days": fit_days, "seed": seed, "target": TARGET_PI0, "features": list(ALL_FEATURES), "max_saved_prediction_error": float(np.max(np.abs(predictions-held.pi1_prediction_pct.to_numpy(float))))}
        print(f"R302 fixed-reference pi1 fold complete {held_day}", flush=True)
    return pd.concat(parts, ignore_index=True), provenance


def mismatch_audit(states: pd.DataFrame, decisions: pd.DataFrame, learned_reference: str) -> dict:
    first = states.sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    visited = states.merge(decisions[KEYS+["submission_t"]], on=KEYS, validate="many_to_one")
    visited = visited.loc[visited.decision_t.le(visited.submission_t)]
    output = {}
    for name, frame in (("all_reachable", states), ("first_decision", first), ("policy_visited", visited)):
        eligible = frame.loc[frame.risk_reachable_hold & np.isfinite(frame[learned_reference]) & np.isfinite(frame[SELF_TARGET])]
        difference = eligible[SELF_TARGET]-eligible[learned_reference]
        output[name] = {
            "states": len(eligible),
            "sign_disagreements": int((eligible[SELF_TARGET].gt(0) != eligible[learned_reference].gt(0)).sum()),
            "mean_absolute_reference_self_difference_pp": float(difference.abs().mean()) if len(eligible) else None,
            "reference_ranking": signal(frame, learned_reference, PREDICTION),
            "self_ranking": signal(frame, SELF_TARGET, PREDICTION),
            "reference_calibration": calibration(frame, learned_reference),
            "self_calibration": calibration(frame, SELF_TARGET),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("states", "decisions", "reference-decisions", "raw-seconds", "output", "states-output", "decisions-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_parquet(args.states)
    source = source.loc[source.arm.eq("B")].copy().reset_index(drop=True)
    if set(source.trading_day.astype(str)) != set(CROSSFIT_DAYS) or source.duplicated([*KEYS, "decision_t"]).any() or len(source[KEYS].drop_duplicates()) != 86:
        raise ValueError("R302 requires exact unique 86 May5-8 positions")
    raw = pd.read_parquet(args.raw_seconds)
    if not set(raw.trading_day.astype(str)).issubset(CROSSFIT_DAYS):
        raise ValueError("new raw dates forbidden")
    contexts = {}
    for key, _ in source.groupby(KEYS, sort=False):
        opening, closing = session_limits(str(key[0]))
        bars = raw.loc[raw.trading_day.astype(str).eq(str(key[0])) & raw.ticker.astype(str).eq(str(key[1]))]
        contexts[key] = (regular_bars(bars, opening, closing), opening, closing)
    original = pd.read_parquet(args.decisions)
    original = original.loc[original.arm.eq("B_preentry_context")]
    A = replay_scored(source, contexts, "A_R301_pi2", next_event=True)
    reproduction = verify_reproduction(A, original)
    print("R302 exact R301 pi2 replay passed", flush=True)
    Bstates, folds = crossfit_one_stage(source)
    B = replay_scored(Bstates, contexts, "B_fixed_reference_pi1", next_event=True)
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
        mismatch[name] = mismatch_audit(frames[name], decisions, TARGET_PI1 if name == "A" else TARGET_PI0)
    comparisons = {"B_minus_A": paired_intervals(B, A), "B_minus_old": paired_intervals(B, old)}
    rank = signal(Bstate, TARGET_PI0, PREDICTION)["day_episode_weighted_rank_correlation"]
    primary = summaries["B"]
    checks = {
        "exact_R301_replay": reproduction["maximum_return_error"] <= 1e-9,
        "exact_saved_pi1_predictions": all(fold["max_saved_prediction_error"] <= 1e-9 for fold in folds.values()),
        "all_entries_reconciled": all(len(frame) == 86 for frame in arms.values()),
        "resolution_at_least_95pct": primary["resolution_rate"] >= .95,
        "positive_base_mean": primary["mean_base_net_pct_resolved"] > 0,
        "positive_at_least_three_days": sum(v > 0 for v in primary["daily_base_net_pct"].values()) >= 3,
        "beats_old_matched": comparisons["B_minus_old"]["mean_gain_pp"] > 0,
        "beats_R301_pi2": comparisons["B_minus_A"]["mean_gain_pp"] > 0,
        "positive_without_largest_trade": primary["mean_without_largest_trade_pct"] > 0,
        "positive_weighted_learned_target_ranking": rank is not None and rank > 0,
    }
    result = {
        "request_id": REQUEST_ID, "primary": "B_fixed_reference_pi1", "development_only": True,
        "promotion_eligible": False, "June_HOLD_opened": False, "network_requests": 0,
        "candidate_episodes": 97, "entries": 86, "R301_reproduction": reproduction,
        "arms": summaries, "frozen_pi0_reference": arm_report(old.assign(policy="next_event_primary"), old),
        "common_label_diagnostics": common, "reference_vs_self_policy_audit": mismatch,
        "feature_availability_and_early_exits": support, "attribution": comparisons,
        "pi1_outer_folds": folds, "checks": checks, "research_gate_pass": all(checks.values()),
        "target_contract": "B learns WAIT-follow-pi0; A/pi2 learned inner WAIT-follow-pi1 targets. A diagnostic outer pi1 target is not the same fitted inner training target. Self-follow labels are held-out diagnostics, never fitted or selected.",
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
