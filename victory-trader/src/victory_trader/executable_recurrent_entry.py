"""Request 214: executable local value + recurrent transition timing.

Development only. No future maximum as the entry-value training target, no
forced terminal purchase, no realized outcome in the decision function.
A fixed 3-minute exit isolates the entry change; this is not a full trader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS, net_round_trip_return_pct
from .opportunity_admission_transition_timing import (
    train_admission_model, predict_admission_ev,
)
from .pullback_turn_transition_entry import (
    attach_transition_features, train_model, predict,
)
from .recurrent_wait_entry_action_value import build_watch_states, MAX_WAIT_MINUTES
from .relative_recurrent_entry_timing import (
    attach_relative_advantage, _metrics, _policy_trades, _safe_spearman,
)
from .supply_admission_transition_timing import _difference_bootstrap

REQUEST_ID = 214
DECISION_COLUMNS = EPISODE_KEYS + [
    "minutes_since_hot", "predicted_entry_ev_pct", "predicted_relative_advantage_pct",
]


def decide(scored: pd.DataFrame, *, reobserve: bool = True) -> pd.DataFrame:
    """One terminal decision per observed episode, independent of future labels.

    Rejected candidates remain on WATCH. Missing checkpoints stop the replay,
    rather than skipping ahead with hindsight. Minute 5 still requires EV > 0.
    """
    work = scored.loc[:, DECISION_COLUMNS].copy()
    if work.duplicated(EPISODE_KEYS + ["minutes_since_hot"]).any():
        raise ValueError("duplicate episode checkpoint")
    records = []
    for key, group in work.groupby(EPISODE_KEYS, sort=True):
        by_minute = group.set_index("minutes_since_hot")
        record = dict(zip(EPISODE_KEYS, key))
        record.update(action="ABSTAIN", entry_minute_after_hot=np.nan, wait_actions=0)
        for minute in range(1, MAX_WAIT_MINUTES + 1):
            if minute not in by_minute.index:
                record["action"] = "COVERAGE_MISS"
                break
            row = by_minute.loc[minute]
            ev = float(row["predicted_entry_ev_pct"])
            advantage = float(row["predicted_relative_advantage_pct"])
            if not np.isfinite(ev) or (minute < MAX_WAIT_MINUTES and not np.isfinite(advantage)):
                record["action"] = "PREDICTION_MISS"
                break
            if ev > 0 and (minute == MAX_WAIT_MINUTES or advantage >= 0):
                record.update(action="ENTER", entry_minute_after_hot=minute)
                break
            if not reobserve:
                break
            if minute < MAX_WAIT_MINUTES:
                record["wait_actions"] += 1
        records.append(record)
    return pd.DataFrame(records, columns=EPISODE_KEYS + [
        "action", "entry_minute_after_hot", "wait_actions",
    ])


def settle(decisions: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """Join outcomes only after decisions; unresolved entries remain explicit."""
    labels = states.loc[:, EPISODE_KEYS + ["minutes_since_hot", "enter_3m_base_pct"]]
    labels = labels.rename(columns={"minutes_since_hot": "entry_minute_after_hot"})
    result = decisions.merge(labels, on=EPISODE_KEYS + ["entry_minute_after_hot"],
                             how="left", validate="one_to_one")
    result["realized_base_return_pct"] = result.pop("enter_3m_base_pct")
    result.loc[result.action.eq("ABSTAIN"), "realized_base_return_pct"] = 0.0
    result["resolved"] = result.action.eq("ABSTAIN") | (
        result.action.eq("ENTER") & np.isfinite(result.realized_base_return_pct)
    )
    return result


def add_execution_labels(states: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Evaluation-only gross/stress labels from the same entry/exit references."""
    result = states.copy()
    lookup = {}
    for row in positions.itertuples(index=False):
        if float(row.minutes_held).is_integer():
            key = (str(row.trading_day), str(row.ticker), int(row.hot_t), int(row.minutes_held))
            if key in lookup:
                raise ValueError("duplicate position checkpoint")
            lookup[key] = float(row.exit_reference_open)
    stress = next(s for s in DEFAULT_EXECUTION_SCENARIOS if s.name == "stress")
    gross_values, stress_values = [], []
    for row in result.itertuples(index=False):
        key = (str(row.trading_day), str(row.ticker), int(row.hot_t))
        minute = int(row.minutes_since_hot)
        entry, exit_ = lookup.get((*key, minute), np.nan), lookup.get((*key, minute + 3), np.nan)
        gross = (exit_ / entry - 1) * 100 if np.isfinite(entry) and np.isfinite(exit_) and min(entry, exit_) > 0 else np.nan
        gross_values.append(gross)
        stress_values.append(net_round_trip_return_pct(entry, gross, stress) if np.isfinite(gross) else np.nan)
    result["gross_return_pct"] = gross_values
    result["stress_return_pct"] = stress_values
    return result


def policy_report(decisions: pd.DataFrame, states: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    settled = settle(decisions, states)
    entered = settled.loc[settled.action.eq("ENTER") & settled.resolved].copy()
    outcome = states.rename(columns={"minutes_since_hot": "entry_minute_after_hot"})
    entered = entered.merge(outcome[EPISODE_KEYS + ["entry_minute_after_hot", "gross_return_pct", "stress_return_pct"]],
                            on=EPISODE_KEYS + ["entry_minute_after_hot"], validate="one_to_one")
    daily = settled.groupby("trading_day").realized_base_return_pct.mean()
    complete = bool(settled.resolved.all())
    report = {
        "episodes": len(settled), "decisions": settled.action.value_counts().to_dict(),
        "unresolved_episodes": int((~settled.resolved).sum()),
        "unresolved_entry_outcomes": int((settled.action.eq("ENTER") & ~settled.resolved).sum()),
        "wait_actions": int(settled.wait_actions.sum()),
        "entered_after_wait": int(entered.entry_minute_after_hot.gt(1).sum()),
        "resolved_trade_metrics": _metrics(entered),
        "mean_gross_pct": float(entered.gross_return_pct.mean()) if len(entered) else None,
        "mean_stress_pct": float(entered.stress_return_pct.mean()) if len(entered) else None,
        # Never silently price missing observations/entry outcomes as abstention.
        "all_opportunity_daily_base_pct": daily.to_dict() if complete else None,
        "fully_resolved": complete,
    }
    return report, settled


def development_gate(report: dict, difference: dict) -> bool:
    metrics = report["resolved_trade_metrics"]
    daily = metrics.get("by_day", {})
    return bool(report["fully_resolved"] and metrics.get("trades", 0) >= 30
                and metrics.get("day_balanced_mean_pct", -np.inf) > 0
                and sum(v > 0 for v in daily.values()) >= 4
                and (difference.get("ci_low_pct") or -np.inf) > 0)


def prepare(positions: pd.DataFrame) -> pd.DataFrame:
    return attach_transition_features(attach_relative_advantage(build_watch_states(positions)))


def evaluate(fit_path: Path, calibration_path: Path, fresh_dir: Path, output: Path) -> dict:
    paths = [fit_path, calibration_path, *sorted(fresh_dir.glob("*-positions.parquet"))]
    if len(paths) != 7:
        raise ValueError("requires exactly five request178 development shards")
    positions = [pd.read_parquet(p) for p in paths]
    fit, calibration, fresh = prepare(positions[0]), prepare(positions[1]), prepare(pd.concat(positions[2:], ignore_index=True))
    if sorted(fresh.trading_day.unique()) != list(FRESH_DAYS):
        raise ValueError("unexpected development dates")
    if not max(fit.trading_day) < min(calibration.trading_day) <= max(calibration.trading_day) < min(fresh.trading_day):
        raise ValueError("fit/calibration/evaluation dates must be strictly ordered")
    timing = train_model(fit, calibration)
    value = train_admission_model(fit, calibration, target_column="enter_3m_base_pct")
    fresh["predicted_relative_advantage_pct"] = predict(fresh, timing)
    fresh["predicted_entry_ev_pct"] = predict_admission_ev(fresh, value)[0]
    fresh = add_execution_labels(fresh, pd.concat(positions[2:], ignore_index=True))
    recurrent, recurrent_rows = policy_report(decide(fresh), fresh)
    once, once_rows = policy_report(decide(fresh, reobserve=False), fresh)
    baseline_trades, baseline_counts = _policy_trades(fresh, recurrent=True)
    # Reconstruct #208 decisions before examining whether future labels exist.
    baseline_states = fresh.copy()
    baseline_states["predicted_entry_ev_pct"] = 1.0
    baseline, baseline_rows = policy_report(decide(baseline_states), fresh)
    total_episodes = len(pd.concat(positions[2:])[EPISODE_KEYS].drop_duplicates())
    observed_episodes = len(recurrent_rows)
    difference = {}
    if recurrent["fully_resolved"] and baseline["fully_resolved"]:
        difference = _difference_bootstrap(pd.Series(recurrent["all_opportunity_daily_base_pct"]),
                                          pd.Series(baseline["all_opportunity_daily_base_pct"]))
    gate = development_gate(recurrent, difference) and total_episodes == observed_episodes
    valid = np.isfinite(fresh.enter_3m_base_pct)
    result = {
        "request_id": REQUEST_ID, "opens_new_dates": False, "promotion_eligible": False,
        "promotion_gate_pass": False, "development_gate_pass": bool(gate),
        "target": "realized 3m BASE return from ENTER at this checkpoint; no future maximum",
        "decision": "EV>0 AND (relative advantage>=0 OR minute=5); otherwise WAIT/ABSTAIN",
        "exit_policy": "fixed 3m, entry diagnostic only", "account_return": None,
        "total_source_episodes": total_episodes, "observed_episodes": observed_episodes,
        "source_episodes_without_any_watch_state": total_episodes - observed_episodes,
        "recurrent": recurrent, "one_shot_same_model": once,
        "request208_explicit_missing": baseline,
        "request208_legacy_resolved_only": _metrics(baseline_trades),
        "request208_legacy_counts": baseline_counts,
        "recurrent_minus_208_all_opportunity": difference,
        "entry_value_spearman": _safe_spearman(fresh.enter_3m_base_pct, fresh.predicted_entry_ev_pct),
        "mean_predicted_ev_pct": float(fresh.loc[valid, "predicted_entry_ev_pct"].mean()),
        "mean_realized_enter_pct": float(fresh.loc[valid, "enter_3m_base_pct"].mean()),
        "features": list(value.columns),
        "runtime": {"python": platform.python_version(), **{
            name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl"]
        }},
        "predicted_entry_ev_range_pct": [float(fresh.predicted_entry_ev_pct.min()), float(fresh.predicted_entry_ev_pct.max())],
        "positive_ev_states": int(fresh.predicted_entry_ev_pct.gt(0).sum()),
        "watch_states": len(fresh),
        "frozen_gate": {"min_resolved_trades": 30, "min_positive_days": 4,
                        "positive_day_balanced_base": True, "positive_paired_daily_bootstrap_low": True,
                        "all_source_episodes_and_outcomes_resolved": True},
        "inputs": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "dates": {"fit": sorted(fit.trading_day.unique()), "calibration": sorted(calibration.trading_day.unique()),
                  "development": sorted(fresh.trading_day.unique())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    pd.concat([recurrent_rows.assign(policy="recurrent214"), once_rows.assign(policy="one_shot214"),
               baseline_rows.assign(policy="request208")]).to_parquet(output.with_suffix(".parquet"), index=False)
    fresh.to_parquet(output.with_name(output.stem + "-states.parquet"), index=False)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument("--calibration-positions", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.fit_positions, args.calibration_positions, args.fresh_dir, args.output)


if __name__ == "__main__":
    main()
