"""R301: retain causal pre-entry chart history for initial HOLD decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clock_context_hold_value import (
    CLOCK_FEATURES, CONTEXT_FEATURES, causal_features, paired_intervals,
    risk_upside, signal_diagnostics,
)
from .frozen_entry_second_hold_exit import FEATURES, KEYS
from .hierarchical_crack_entry_controller import _episode_day_weights
from .lagged_minute_context import CROSSFIT_DAYS
from .pending_exit_integrity import regular_bars, session_limits
from .risk_reachable_hold_value import (
    TARGET_PI0, TARGET_PI1, arm_report, continuation_targets,
    crossfit_improvement, replay_scored, signal, verify_reproduction,
)

REQUEST_ID = 301
ALL_FEATURES = FEATURES+CLOCK_FEATURES+CONTEXT_FEATURES
PREDICTION = "predicted_next_event_advantage_pct"


def calibration(frame: pd.DataFrame, target: str) -> dict:
    valid = frame.loc[frame.risk_reachable_hold & np.isfinite(frame[target]) & np.isfinite(frame[PREDICTION])]
    if valid.empty:
        return {"states": 0}
    weights = _episode_day_weights(valid)
    truth, hold = valid[target].gt(0).to_numpy(), valid[PREDICTION].gt(0).to_numpy()
    output = {
        "states": len(valid), "weighted_predicted_mean_pct": float(np.average(valid[PREDICTION], weights=weights)),
        "weighted_target_mean_pct": float(np.average(valid[target], weights=weights)),
        "weighted_sign_accuracy": float(np.average(truth == hold, weights=weights)),
        "confusion_counts": {"hold_positive_target": int((hold & truth).sum()), "hold_nonpositive_target": int((hold & ~truth).sum()), "exit_positive_target": int((~hold & truth).sum()), "exit_nonpositive_target": int((~hold & ~truth).sum())},
    }
    for name, mask in (("predicted_hold", hold), ("predicted_exit", ~hold)):
        output[name] = {
            "states": int(mask.sum()),
            "weighted_target_mean_pct": float(np.average(valid[target].to_numpy()[mask], weights=weights[mask])) if mask.any() else None,
        }
    return output


def common_diagnostics(states: pd.DataFrame, baseline: pd.DataFrame, decisions: pd.DataFrame) -> dict:
    labels = baseline[KEYS+["decision_t", TARGET_PI0, TARGET_PI1]].rename(columns={TARGET_PI0: "common_pi0", TARGET_PI1: "common_A_pi1"})
    joined = states.merge(labels, on=KEYS+["decision_t"], validate="one_to_one")
    first = joined.sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    visited = joined.merge(decisions[KEYS+["submission_t"]], on=KEYS, validate="many_to_one")
    visited = visited.loc[visited.decision_t.le(visited.submission_t)]
    return {
        target: {name: {"ranking": signal(frame, target, PREDICTION), "calibration": calibration(frame, target)} for name, frame in (("all_reachable", joined), ("first_decision", first), ("policy_visited", visited))}
        for target in ("common_pi0", "common_A_pi1")
    }


def availability(states: pd.DataFrame, decisions: pd.DataFrame) -> dict:
    names = list(CLOCK_FEATURES+CONTEXT_FEATURES)
    first = states.sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    joined = states.merge(decisions[KEYS+["submission_t", "entry_t", "exit_reason"]].rename(columns={"entry_t": "decision_entry_t"}), on=KEYS, validate="many_to_one")
    exits = joined.loc[joined.decision_t.eq(joined.submission_t) & joined.exit_reason.eq("model_exit")]
    reachable = states.loc[states.risk_reachable_hold]
    scopes = {"all_reachable": reachable, "first_reachable": first.loc[first.risk_reachable_hold], "model_exit_states": exits}
    output = {name: {"states": len(group), "available_counts": {column: int(group[column].notna().sum()) for column in names}} for name, group in scopes.items()}
    output["by_day_reachable"] = {str(day): {column: int(group[column].notna().sum()) for column in names} for day, group in reachable.groupby("trading_day")}
    matched = decisions.merge(first[KEYS+["decision_t"]], on=KEYS, validate="one_to_one")
    output["first_decision_model_exits"] = int((matched.exit_reason.eq("model_exit") & matched.submission_t.eq(matched.decision_t)).sum())
    output["submitted_within_20s_of_entry"] = int((matched.submission_t-matched.entry_t).lt(20000).sum())
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("states", "decisions", "reference-decisions", "raw-seconds", "output", "states-output", "decisions-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_parquet(args.states)
    source = source.loc[source.arm.eq("C")].copy().reset_index(drop=True)
    if set(source.trading_day.astype(str)) != set(CROSSFIT_DAYS) or source.duplicated([*KEYS, "decision_t"]).any() or len(source[KEYS].drop_duplicates()) != 86:
        raise ValueError("R301 requires exact unique 86 May5-8 positions")
    raw = pd.read_parquet(args.raw_seconds)
    if not set(raw.trading_day.astype(str)).issubset(CROSSFIT_DAYS):
        raise ValueError("new raw dates forbidden")
    contexts = {}
    for key, _ in source.groupby(KEYS, sort=False):
        opening, closing = session_limits(str(key[0]))
        bars = raw.loc[raw.trading_day.astype(str).eq(str(key[0])) & raw.ticker.astype(str).eq(str(key[1]))]
        contexts[key] = (regular_bars(bars, opening, closing), opening, closing)
    saved_decisions = pd.read_parquet(args.decisions)
    original = saved_decisions.loc[saved_decisions.arm.eq("C")]
    r299 = saved_decisions.loc[saved_decisions.arm.eq("A_R299")]
    A = replay_scored(source, contexts, "A_R300_C", next_event=True)
    reproduction = verify_reproduction(A, original)
    print("R301 exact R300 C replay passed", flush=True)
    enriched = causal_features(source, contexts, include_preentry=True)
    targets = continuation_targets(enriched, contexts, prediction_column=None, output_column=TARGET_PI0)
    Bstates, folds = crossfit_improvement(targets, contexts, features=ALL_FEATURES)
    B = replay_scored(Bstates, contexts, "B_preentry_context", next_event=True)
    reference = pd.read_parquet(args.reference_decisions)
    old = reference.loc[reference.stage.eq("rebuilt_label_refit") & reference.policy.eq("old_10m_2pct_trailing")]
    arms, frames = {"A": A, "B": B}, {"A": source, "B": Bstates}
    summaries, signals, common, support = {}, {}, {}, {}
    for name, decisions in arms.items():
        summary = arm_report(decisions, old)
        summary["mean_gross_pct"] = float(decisions.gross_return_pct.mean())
        summary["mean_cost_drag_pp"] = float((decisions.gross_return_pct-decisions.base_net_return_pct).mean())
        summary["median_hold_seconds"] = float(decisions.hold_seconds.median())
        summary["risk_reachable_upside"] = risk_upside(source, contexts, decisions)
        summaries[name] = summary
        signals[name] = signal_diagnostics(frames[name], decisions)
        common[name] = common_diagnostics(frames[name], source, decisions)
        support[name] = availability(frames[name], decisions)
    comparisons = {"B_minus_A": paired_intervals(B, A), "B_minus_R299_D": paired_intervals(B, r299)}
    primary = summaries["B"]
    rank = signals["B"]["all_reachable"]["day_episode_weighted_rank_correlation"]
    checks = {
        "exact_R300_replay": reproduction["maximum_return_error"] <= 1e-9,
        "all_entries_reconciled": all(len(frame) == 86 for frame in arms.values()),
        "resolution_at_least_95pct": primary["resolution_rate"] >= .95,
        "positive_base_mean": primary["mean_base_net_pct_resolved"] > 0,
        "positive_at_least_three_days": sum(v > 0 for v in primary["daily_base_net_pct"].values()) >= 3,
        "beats_old_matched": primary["matched_vs_old"]["mean_gain_pp"] > 0,
        "beats_R300_C": comparisons["B_minus_A"]["mean_gain_pp"] > 0,
        "beats_R299_D": comparisons["B_minus_R299_D"]["mean_gain_pp"] > 0,
        "positive_without_largest_trade": primary["mean_without_largest_trade_pct"] > 0,
        "positive_weighted_hold_ranking": rank is not None and rank > 0,
    }
    result = {
        "request_id": REQUEST_ID, "primary": "B_preentry_context", "development_only": True,
        "promotion_eligible": False, "June_HOLD_opened": False, "network_requests": 0,
        "candidate_episodes": 97, "entries": 86, "reachable_hold_states": int(source.risk_reachable_hold.sum()),
        "feature_columns": list(ALL_FEATURES), "R300_reproduction": reproduction,
        "arms": summaries, "signals": signals, "common_label_diagnostics": common,
        "feature_availability_and_early_exits": support, "attribution": comparisons,
        "nested_folds": folds, "checks": checks, "research_gate_pass": all(checks.values()),
        "limitations": "Four reused days; upstream entry OOF not nested; next-print fills are not quotes; event means are not account return. pi1 labels differ across arms; common-label diagnostics are counterfactual, not optimal or profit guarantees. No parameter/arm selection after result.",
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
