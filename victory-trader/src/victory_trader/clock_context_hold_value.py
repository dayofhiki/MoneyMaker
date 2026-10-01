"""R300: causal elapsed-time and local close-range HOLD feature ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .frozen_entry_second_hold_exit import FEATURES, KEYS
from .lagged_minute_context import CROSSFIT_DAYS
from .pending_exit_integrity import regular_bars, session_limits
from .risk_reachable_hold_value import (
    TARGET_PI0, TARGET_PI1, arm_report, continuation_targets,
    crossfit_improvement, replay_scored, signal, verify_reproduction,
)

REQUEST_ID = 300
CLOCK_FEATURES = (
    "clock_return_5s_pct", "clock_return_20s_pct", "clock_return_60s_pct",
    "clock_observed_count_20s", "clock_observed_count_60s", "clock_volume_ratio_20s",
)
CONTEXT_FEATURES = ("clock_close_location_60s", "clock_drawdown_60s_pct")


def causal_features(states: pd.DataFrame, contexts: dict) -> pd.DataFrame:
    """Use completed regular bars since entry; never pad or backfill gaps."""
    frame = states.copy()
    for name in CLOCK_FEATURES + CONTEXT_FEATURES:
        frame[name] = np.nan
    for key, group in frame.groupby(KEYS, sort=False):
        bars, _, _ = contexts[key]
        entry_t = int(group.entry_t.iloc[0])
        entry_price = float(group.entry_price.iloc[0])
        bars = bars.loc[bars.t.ge(entry_t)].sort_values("t")
        ends = bars.t.to_numpy(np.int64) + 1000
        close = bars.c.to_numpy(float)
        volume = bars.v.to_numpy(float)
        if len(np.unique(ends)) != len(ends) or not np.isfinite(close).all() or (close <= 0).any() or not np.isfinite(volume).all() or (volume < 0).any():
            raise ValueError("invalid causal feature bars")
        cumulative_volume = np.r_[0., np.cumsum(volume)]
        values = []
        for row in group.itertuples():
            t = int(row.decision_t)
            right = int(np.searchsorted(ends, t, side="right"))
            if right == 0 or t < entry_t:
                raise ValueError("state has no completed causal close")
            current = close[right-1]
            out = {}
            for window in (5, 20, 60):
                left = int(np.searchsorted(ends, t-window*1000, side="right"))
                if t-entry_t >= window*1000:
                    reference = close[left-1] if left else entry_price
                    out[f"clock_return_{window}s_pct"] = 100*(current/reference-1)
                else:
                    out[f"clock_return_{window}s_pct"] = np.nan
                if window in (20, 60):
                    out[f"clock_observed_count_{window}s"] = right-left
                if window == 20:
                    previous = int(np.searchsorted(ends, t-40000, side="right"))
                    denominator = cumulative_volume[left]-cumulative_volume[previous]
                    out["clock_volume_ratio_20s"] = (cumulative_volume[right]-cumulative_volume[left])/denominator if t-entry_t >= 40000 and denominator > 0 else np.nan
                if window == 60:
                    recent = close[left:right]
                    if t-entry_t >= 60000 and len(recent):
                        low, high = float(recent.min()), float(recent.max())
                        out["clock_close_location_60s"] = (current-low)/(high-low) if high > low else .5
                        out["clock_drawdown_60s_pct"] = 100*(current/high-1)
                    else:
                        out["clock_close_location_60s"] = np.nan
                        out["clock_drawdown_60s_pct"] = np.nan
            values.append(out)
        added = pd.DataFrame(values, index=group.index)
        frame.loc[group.index, list(added.columns)] = added
    return frame


def paired_intervals(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    pair = a.loc[a.resolved].merge(b.loc[b.resolved], on=KEYS, suffixes=("_a", "_b"), validate="one_to_one")
    pair["gain"] = pair.base_net_return_pct_a-pair.base_net_return_pct_b
    output = {"matched_entries": len(pair), "mean_gain_pp": float(pair.gain.mean()), "bootstrap": {}}
    for index, columns in enumerate((["trading_day"], ["trading_day", "ticker"])):
        grouped = pair.groupby(columns).gain.agg(["sum", "count"])
        rng = np.random.default_rng(20263000+index)
        choices = rng.integers(0, len(grouped), size=(10000, len(grouped)))
        sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
        samples = sums[choices].sum(axis=1)/counts[choices].sum(axis=1)
        output["bootstrap"]["day" if index == 0 else "ticker_day"] = {
            "clusters": len(grouped), "gain_ci95_pp": np.quantile(samples, [.025, .975]).tolist(),
        }
    return output


def signal_diagnostics(states: pd.DataFrame, decisions: pd.DataFrame) -> dict:
    target, prediction = TARGET_PI1, "predicted_next_event_advantage_pct"
    eligible = states.loc[states.risk_reachable_hold].copy()
    first = states.sort_values("decision_t").groupby(KEYS, sort=False).head(1)
    # States after this policy's submission are counterfactual, not deployed.
    joined = states.merge(decisions[KEYS+["submission_t"]], on=KEYS, validate="many_to_one")
    visited = joined.loc[joined.decision_t.le(joined.submission_t)]
    signs = {}
    for day, group in eligible.groupby("trading_day"):
        signs[str(day)] = {
            "states": len(group), "target_positive_rate": float(group[target].gt(0).mean()),
            "target_zero_rate": float(group[target].eq(0).mean()),
            "prediction_positive_rate": float(group[prediction].gt(0).mean()),
        }
    return {
        "all_reachable": signal(states, target, prediction),
        "per_day": {str(day): signal(group, target, prediction) for day, group in states.groupby("trading_day")},
        "first_decision_reachable": signal(first, target, prediction),
        "policy_visited_reachable": signal(visited, target, prediction),
        "signs_by_day": signs,
    }


def risk_upside(states: pd.DataFrame, contexts: dict, decisions: pd.DataFrame) -> dict:
    """Future-knowledge diagnostic; stop-submission fill is included, later fills are not."""
    rows = []
    for key, group in states.groupby(KEYS, sort=False):
        bars, _, closing = contexts[key]
        times = bars.t.to_numpy(np.int64)
        ends = []
        for row in group.itertuples():
            if row.risk_reachable_hold:
                ends.append(int(row.decision_t))
        stop = float(group.forced_stop_t.iloc[0])
        cap = min(int(group.hot_t.iloc[0])+3600000, closing)
        ends.append(min(int(stop), cap) if np.isfinite(stop) else cap)
        positions = np.searchsorted(times, ends)
        positions = positions[positions < len(bars)]
        price = float(group.entry_price.iloc[0])
        maximum = 100*(bars.o.to_numpy(float)[positions].max()/price-1) if len(positions) else np.nan
        rows.append({**dict(zip(KEYS, key)), "reachable_mfe_open_pct": maximum})
    pair = decisions.merge(pd.DataFrame(rows), on=KEYS, validate="one_to_one")
    output = {}
    for level in (5, 10, 20):
        selected = pair.loc[pair.reachable_mfe_open_pct.ge(level)]
        output[f"plus{level}"] = {
            "opportunities": len(selected),
            "net_reached_count": int(selected.base_net_return_pct.ge(level).sum()),
            "net_reached_rate": float(selected.base_net_return_pct.ge(level).mean()) if len(selected) else None,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("states", "decisions", "reference-decisions", "raw-seconds", "output", "states-output", "decisions-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    saved = pd.read_parquet(args.states)
    source = saved.loc[saved.arm.eq("D")].copy().reset_index(drop=True)
    if set(source.trading_day.astype(str)) != set(CROSSFIT_DAYS) or source.duplicated([*KEYS, "decision_t"]).any() or len(source[KEYS].drop_duplicates()) != 86:
        raise ValueError("R300 requires unique 86 May5-8 positions")
    raw = pd.read_parquet(args.raw_seconds)
    if not set(raw.trading_day.astype(str)).issubset(CROSSFIT_DAYS):
        raise ValueError("new raw dates forbidden")
    contexts = {}
    for key, _ in source.groupby(KEYS, sort=False):
        opening, closing = session_limits(str(key[0]))
        bars = raw.loc[raw.trading_day.astype(str).eq(str(key[0])) & raw.ticker.astype(str).eq(str(key[1]))]
        contexts[key] = (regular_bars(bars, opening, closing), opening, closing)
    original = pd.read_parquet(args.decisions)
    original = original.loc[original.arm.eq("D_next_event")]
    A = replay_scored(source, contexts, "A_R299", next_event=True)
    reproduction = verify_reproduction(A, original)
    print("R300 exact R299 D replay passed", flush=True)
    enriched = causal_features(source, contexts)
    targets = continuation_targets(enriched, contexts, prediction_column=None, output_column=TARGET_PI0)
    frames, arms, folds = {"A": source}, {"A": A}, {}
    for name, features in (("B", FEATURES+CLOCK_FEATURES), ("C", FEATURES+CLOCK_FEATURES+CONTEXT_FEATURES)):
        print(f"R300 arm {name} feature count {len(features)}", flush=True)
        scored, provenance = crossfit_improvement(targets, contexts, features=features)
        frames[name] = scored
        arms[name] = replay_scored(scored, contexts, name, next_event=True)
        folds[name] = provenance
    reference = pd.read_parquet(args.reference_decisions)
    old = reference.loc[reference.stage.eq("rebuilt_label_refit") & reference.policy.eq("old_10m_2pct_trailing")]
    summaries, signals = {}, {}
    for name, decisions in arms.items():
        summary = arm_report(decisions, old)
        summary["mean_gross_pct"] = float(decisions.gross_return_pct.mean())
        summary["mean_cost_drag_pp"] = float((decisions.gross_return_pct-decisions.base_net_return_pct).mean())
        summary["median_hold_seconds"] = float(decisions.hold_seconds.median())
        summary["risk_reachable_upside"] = risk_upside(source, contexts, decisions)
        summaries[name] = summary
        signals[name] = signal_diagnostics(frames[name], decisions)
    comparisons = {"B_minus_A": paired_intervals(arms["B"], A), "C_minus_B": paired_intervals(arms["C"], arms["B"]), "C_minus_A": paired_intervals(arms["C"], A)}
    primary = summaries["C"]
    rank = signals["C"]["all_reachable"]["day_episode_weighted_rank_correlation"]
    checks = {
        "exact_R299_replay": reproduction["maximum_return_error"] <= 1e-9,
        "all_entries_reconciled": all(len(decisions) == 86 for decisions in arms.values()),
        "resolution_at_least_95pct": primary["resolution_rate"] >= .95,
        "positive_base_mean": primary["mean_base_net_pct_resolved"] > 0,
        "positive_at_least_three_days": sum(v > 0 for v in primary["daily_base_net_pct"].values()) >= 3,
        "beats_old_matched": primary["matched_vs_old"]["mean_gain_pp"] > 0,
        "beats_R299_D": comparisons["C_minus_A"]["mean_gain_pp"] > 0,
        "positive_without_largest_trade": primary["mean_without_largest_trade_pct"] > 0,
        "positive_weighted_hold_ranking": rank is not None and rank > 0,
    }
    availability = {}
    for day, group in enriched.groupby("trading_day"):
        eligible = group.loc[group.risk_reachable_hold]
        availability[str(day)] = {name: int(eligible[name].notna().sum()) for name in CLOCK_FEATURES+CONTEXT_FEATURES}
    result = {
        "request_id": REQUEST_ID, "primary": "C_clock_and_context", "development_only": True,
        "promotion_eligible": False, "June_HOLD_opened": False, "network_requests": 0,
        "candidate_episodes": 97, "entries": 86, "reachable_hold_states": int(source.risk_reachable_hold.sum()),
        "feature_columns": {"A": list(FEATURES), "B": list(FEATURES+CLOCK_FEATURES), "C": list(FEATURES+CLOCK_FEATURES+CONTEXT_FEATURES)},
        "reachable_feature_availability_by_day": availability, "R299_reproduction": reproduction,
        "arms": summaries, "signals": signals, "attribution": comparisons, "nested_folds": folds,
        "checks": checks, "research_gate_pass": all(checks.values()),
        "limitations": "Four reused days; upstream entry OOF not nested within HOLD; next-print reference fills are not quotes; event means are not portfolio returns. pi1 downstream labels differ by feature arm. No arm selection after primary failure.",
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
