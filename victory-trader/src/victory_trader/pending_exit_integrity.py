"""R298: isolate clock/pending-exit corrections from continuation-label refits.

All entry and model choices are frozen. Only reused May development is accepted.
Next-print references are a scenario, not a claim about real order-book fills.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS
from .frozen_entry_second_hold_exit import (
    FEATURES, HARD_STOP_PCT, HORIZONS_S, KEYS, POLICIES,
    check_development, crossfit, net, replay as legacy_replay, report,
)
from .lagged_minute_context import CROSSFIT_DAYS
from .market_calendar import regular_session_bounds
from .massive_client import MassiveClient
from .second_path_attention_probe import SECOND_CACHE_DIR, _second_frame


REQUEST_ID = 298


def session_limits(day: str) -> tuple[int, int]:
    bounds = regular_session_bounds(date.fromisoformat(day))
    if bounds is None:
        raise ValueError(f"no regular session for {day}")
    return tuple(int(x.timestamp() * 1000) for x in bounds)


def regular_bars(bars: pd.DataFrame, opening: int, closing: int) -> pd.DataFrame:
    result = bars.loc[bars.t.ge(opening) & bars.t.lt(closing)].sort_values("t").reset_index(drop=True)
    if result.t.duplicated().any():
        raise ValueError("duplicate raw second timestamps")
    if len(result) and (not np.isfinite(result[["o", "h", "l", "c"]].to_numpy(float)).all() or (result[["o", "h", "l", "c"]] <= 0).any().any()):
        raise ValueError("invalid raw prices")
    return result


def fill_reference(bars: pd.DataFrame, submission_t: int) -> tuple[int, float] | None:
    """First regular-session open after submission, including long pending gaps."""
    times = bars.t.to_numpy(np.int64)
    pos = int(np.searchsorted(times, int(submission_t)))
    if pos >= len(times):
        return None
    return int(times[pos]), float(bars.o.iloc[pos])


def submission(group: pd.DataFrame, policy: str, cap: int) -> tuple[int, str]:
    ordered = group.sort_values("decision_t")
    entry_t = int(ordered.entry_t.iloc[0])
    timer, reason = cap, "terminal_cap"
    if policy == "immediate":
        return min(entry_t + 1000, cap), "immediate"
    if policy.startswith("fixed_"):
        horizon = int(policy.split("_")[1][:-1])
        if entry_t + horizon * 1000 <= cap:
            timer, reason = entry_t + horizon * 1000, "fixed_horizon"
    elif policy == "old_10m_2pct_trailing" and entry_t + 600000 <= cap:
        timer, reason = entry_t + 600000, "max_hold"
    # A market-data signal can preempt a timer, but never a submitted exit.
    # If there are no prints before a timer, the timer still fires exactly on time.
    for _, row in ordered.loc[ordered.decision_t.le(timer)].iterrows():
        signal = None
        if row.observed_net_return_pct <= HARD_STOP_PCT:
            signal = "hard_stop"
        elif policy == "old_10m_2pct_trailing" and row.drawdown_pct <= -2:
            signal = "trailing_stop"
        elif policy.startswith("dynamic_"):
            horizon = int(policy.split("_")[1][:-1])
            prediction = float(row[f"predicted_hold_{horizon}s_pct"])
            if not np.isfinite(prediction) or prediction <= 0:
                signal = "model_exit"
        if signal:
            return int(row.decision_t), signal
    return int(timer), reason


def pending_replay(group: pd.DataFrame, bars: pd.DataFrame, policy: str, opening: int, closing: int) -> dict:
    ordered = group.sort_values("decision_t")
    first = ordered.iloc[0]
    entry_t, price = int(first.entry_t), float(first.entry_price)
    cap = min(int(first.hot_t) + 3600000, closing)
    if not opening <= entry_t < closing or cap <= entry_t:
        raise ValueError("invalid entry/session/cap")
    raw = regular_bars(bars, opening, closing)
    submit_t, reason = submission(ordered, policy, cap)
    fill = fill_reference(raw, submit_t)
    filled = fill is not None
    fill_t, fill_price = fill if fill else (None, None)
    window = raw.loc[raw.t.ge(entry_t) & raw.t.lt(cap)]
    opportunity = max(0., float((window.o.max() / price - 1) * 100)) if len(window) else np.nan
    last_bar = raw.loc[(raw.t + 1000).le(submit_t) & raw.t.ge(entry_t)]
    last_close = float(last_bar.c.iloc[-1]) if len(last_bar) else price
    last_observed_t = int(last_bar.t.iloc[-1]) + 1000 if len(last_bar) else entry_t
    held = raw.loc[raw.t.ge(entry_t) & raw.t.lt(fill_t if filled else closing)]
    later = window.loc[window.t.gt(fill_t)] if filled else pd.DataFrame()
    pre = ordered.loc[ordered.decision_t.le(fill_t if filled else closing)]
    lag_s = (fill_t - submit_t) / 1000 if filled else np.nan
    result = {
        "trading_day": str(first.trading_day), "ticker": str(first.ticker), "hot_t": int(first.hot_t), "policy": policy, "resolved": filled,
        "order_state": "FILLED" if filled else "RIGHT_CENSORED",
        "exit_reason": reason if filled else "right_censored_" + reason,
        "signal_t": submit_t, "submission_t": submit_t,
        "entry_t": entry_t, "entry_price": price, "exit_t": fill_t,
        "session_open_t": opening, "session_close_t": closing, "observation_cap_t": cap,
        "execution_delay_s": lag_s, "pending_duration_s": ((fill_t if filled else closing) - submit_t) / 1000, "within_3s": bool(filled and lag_s <= 3),
        "delay_bucket": ("0_3s" if lag_s <= 3 else "4_30s" if lag_s <= 30 else "31_300s" if lag_s <= 300 else "over_300s") if filled else "censored",
        "last_observed_price_age_s": (submit_t - last_observed_t) / 1000,
        "fill_jump_vs_last_observed_pct": (fill_price / last_close - 1) * 100 if filled else np.nan,
        "hold_seconds": (fill_t - entry_t) / 1000 if filled else np.nan,
        "exposure_past_cap_s": max(0., (fill_t - cap) / 1000) if filled else np.nan,
        "gross_return_pct": (fill_price / price - 1) * 100 if filled else np.nan,
        "mae_pct": min(0., (float(held.l.min()) / price - 1) * 100, (fill_price / price - 1) * 100) if filled and len(held) else np.nan,
        "mae_observed_close_pct": min(0., float(pre.observed_return_pct.min())) if len(pre) else np.nan,
        "available_mfe_open_pct": opportunity,
        "later_left_on_table_pp": max(0., float((later.o.max() - fill_price) / price * 100)) if len(later) else np.nan,
        "opportunity_active_seconds": len(window),
        "opportunity_last_print_age_s": (cap - int(window.t.iloc[-1]) - 1000) / 1000 if len(window) else np.nan,
        "opportunity_max_interprint_gap_s": float(window.t.diff().max() / 1000) if len(window) > 1 else np.nan,
        "cost_only_net_pct": net(price, price),
        "stop_cost_only_breach": bool(reason == "hard_stop" and net(price, price) <= HARD_STOP_PCT),
    }
    for scenario in DEFAULT_EXECUTION_SCENARIOS:
        result[f"{scenario.name}_net_return_pct"] = net(price, fill_price, scenario) if filled else np.nan
    return result


def rebuild_targets(group: pd.DataFrame, bars: pd.DataFrame, closing: int) -> pd.DataFrame:
    frame = group.copy()
    bars = bars.loc[bars.t.lt(closing)].sort_values("t").reset_index(drop=True)
    price = float(frame.entry_price.iloc[0])
    cap = min(int(frame.hot_t.iloc[0]) + 3600000, closing)
    times = bars.t.to_numpy(np.int64)
    prices = bars.o.to_numpy(float)
    def liquidate(submissions):
        positions = np.searchsorted(times, submissions)
        valid = positions < len(times)
        result = np.full(len(frame), np.nan)
        delays = np.full(len(frame), np.nan)
        for i in np.flatnonzero(valid):
            result[i] = net(price, float(prices[positions[i]]))
            delays[i] = (int(times[positions[i]]) - int(submissions[i])) / 1000
        return result, delays
    now, now_delay = liquidate(frame.decision_t.to_numpy(np.int64))
    frame["exit_now_pct"] = now
    frame["exit_now_label_delay_s"] = now_delay
    for h in HORIZONS_S:
        submit = np.minimum(frame.decision_t.to_numpy(np.int64) + h * 1000, cap)
        value, delay = liquidate(submit)
        frame[f"hold_{h}s_advantage_pct"] = value - now
        frame[f"hold_{h}s_label_delay_s"] = delay
    return frame


def reproduce_legacy(states: pd.DataFrame, original: pd.DataFrame) -> dict:
    rows = pd.DataFrame([legacy_replay(g, p) for _, g in states.groupby(KEYS, sort=False) for p in POLICIES])
    match = rows.merge(original, on=[*KEYS, "policy"], suffixes=("_new", "_original"), validate="one_to_one")
    if len(match) != len(original):
        raise ValueError("legacy reproduction lost rows")
    equal = match.resolved_new.eq(match.resolved_original) & match.exit_reason_new.eq(match.exit_reason_original)
    valid = match.resolved_original
    delta = (match.loc[valid, "base_net_return_pct_new"] - match.loc[valid, "base_net_return_pct_original"]).abs()
    if not equal.all() or not np.allclose(delta, 0, atol=1e-9, rtol=0):
        raise ValueError("R297 reproduction mismatch")
    return {"rows": len(match), "return_max_absolute_error": float(delta.max()), "reasons_identical": True}


def extended_report(rows: pd.DataFrame, total: int) -> dict:
    result = report(rows, total)
    for policy in POLICIES:
        frame = rows.loc[rows.policy.eq(policy)]
        valid = frame.loc[frame.resolved]
        y = valid.base_net_return_pct.sort_values(ascending=False)
        best = valid.loc[valid.base_net_return_pct.idxmax()] if len(valid) else None
        result[policy].update({
            "order_states": frame.order_state.value_counts().to_dict(),
            "delay_buckets": frame.delay_bucket.value_counts().to_dict(),
            "delay_seconds_quantiles": valid.execution_delay_s.quantile([.5, .9, .95, 1.]).to_dict() if len(valid) else {},
            "mean_without_largest_trade_pct": float(y.iloc[1:].mean()) if len(y) > 1 else None,
            "largest_trade": {"ticker": str(best.ticker), "day": str(best.trading_day), "base_net_pct": float(best.base_net_return_pct), "hold_seconds": float(best.hold_seconds), "delay_s": float(best.execution_delay_s)} if best is not None else None,
            "cost_only_stop_breaches": int(frame.stop_cost_only_breach.sum()),
            "fills_past_cap": int(valid.exposure_past_cap_s.gt(0).sum()),
            "within_3s_diagnostic": {"count": int(valid.within_3s.sum()), "mean_base_net_pct": float(valid.loc[valid.within_3s, "base_net_return_pct"].mean()) if valid.within_3s.any() else None, "post_action_subset_not_population": True},
            "left_on_table_observable_count": int(valid.later_left_on_table_pp.notna().sum()),
            "observed_opportunity_paths": int(frame.available_mfe_open_pct.notna().sum()),
            "endpoint_age_quantiles_s": frame.opportunity_last_print_age_s.dropna().quantile([.5,.9,.95,1.]).to_dict(),
        })
    return result


def signal_report(states: pd.DataFrame) -> dict:
    result = {}
    for h in HORIZONS_S:
        target, pred = f"hold_{h}s_advantage_pct", f"predicted_hold_{h}s_pct"
        valid = states[[target, pred, f"hold_{h}s_label_delay_s", "exit_now_label_delay_s"]].dropna()
        short = valid.loc[valid[f"hold_{h}s_label_delay_s"].le(3) & valid.exit_now_label_delay_s.le(3)]
        def correlation(frame):
            v = frame[target].corr(frame[pred], method="spearman") if len(frame) > 1 else np.nan
            return float(v) if np.isfinite(v) else None
        result[str(h)] = {"label_coverage": float(states[target].notna().mean()), "oof_spearman": correlation(valid), "within_3s_label_rows": len(short), "within_3s_spearman": correlation(short), "future_label_delay_quantiles_s": valid[f"hold_{h}s_label_delay_s"].quantile([.5,.9,.95,1.]).to_dict()}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("legacy-states", "legacy-decisions", "output", "states-output", "decisions-output", "raw-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    states = pd.read_parquet(args.legacy_states)
    check_development(states.assign(held_out_day=states.trading_day))
    if set(states.trading_day.astype(str)) != set(CROSSFIT_DAYS):
        raise ValueError("all four development days required")
    original = pd.read_parquet(args.legacy_decisions)
    reproduction = reproduce_legacy(states, original)
    if states[KEYS].drop_duplicates().shape[0] != 86:
        raise ValueError("R298 requires the exact R297 86 positions")
    settings = load_settings()
    # Reuse R297's exact raw acquisition cache, avoiding redundant collection.
    client = MassiveClient(settings.massive_api_key, cache_dir=SECOND_CACHE_DIR / "request297", request_interval_seconds=0.02)
    cache, contexts, rebuilt_parts, corrected_rows = {}, {}, [], []
    for index, (key, group) in enumerate(states.groupby(KEYS, sort=False)):
        day, ticker, _ = key
        opening, closing = session_limits(str(day))
        raw_key = (str(day), str(ticker))
        if raw_key not in cache:
            cache[raw_key] = regular_bars(_second_frame(client.second_bars_range(str(ticker), date.fromisoformat(str(day)), date.fromisoformat(str(day)), adjusted=False)), opening, closing)
        bars = cache[raw_key]
        contexts[key] = (bars, opening, closing)
        corrected_rows.extend(pending_replay(group, bars, p, opening, closing) for p in POLICIES)
        rebuilt_parts.append(rebuild_targets(group, bars, closing))
        print(f"R298 path {index+1}/86 {day} {ticker}", flush=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([bars.assign(trading_day=key[0], ticker=key[1]) for key, bars in cache.items()], ignore_index=True).to_parquet(args.raw_output, index=False)
    rebuilt = pd.concat(rebuilt_parts, ignore_index=True)
    # Freeze features and scores for execution-only attribution before refitting.
    pd.testing.assert_frame_equal(states.sort_values([*KEYS,"decision_t"])[list(FEATURES)].reset_index(drop=True), rebuilt.sort_values([*KEYS,"decision_t"])[list(FEATURES)].reset_index(drop=True))
    args.states_output.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.to_parquet(args.states_output, index=False)
    scored, folds = crossfit(rebuilt)
    refit_rows = []
    for key, group in scored.groupby(KEYS, sort=False):
        bars, opening, closing = contexts[key]
        refit_rows.extend(pending_replay(group, bars, p, opening, closing) for p in POLICIES)
    corrected = pd.DataFrame(corrected_rows).assign(stage="execution_only")
    refit = pd.DataFrame(refit_rows).assign(stage="rebuilt_label_refit")
    decisions = pd.concat([corrected, refit], ignore_index=True)
    checks = {
        "reproduction": True,
        "all_positions_reconciled": all(len(g) == 86 for _, g in decisions.groupby(["stage","policy"])),
        "fills_after_submission": bool(decisions.loc[decisions.resolved, "exit_t"].ge(decisions.loc[decisions.resolved, "submission_t"]).all()),
        "fills_before_close": bool(decisions.loc[decisions.resolved, "exit_t"].lt(decisions.loc[decisions.resolved, "session_close_t"]).all()),
    }
    fixed = decisions.loc[decisions.exit_reason.eq("fixed_horizon")]
    checks["exact_fixed_timers"] = all(int(r.submission_t) == int(r.entry_t) + int(r.policy.split('_')[1][:-1])*1000 for r in fixed.itertuples())
    primary = refit.loc[refit.policy.eq("dynamic_300s")]
    checks["primary_resolution_at_least_95pct"] = float(primary.resolved.mean()) >= .95
    stats = {"execution_only": extended_report(corrected, 97), "rebuilt_label_refit": extended_report(refit, 97)}
    primary_stats = stats["rebuilt_label_refit"]["dynamic_300s"]
    matched = stats["rebuilt_label_refit"].get("matched_primary_vs_old", {})
    economics = {
        "positive_base_mean": (primary_stats["mean_base_net_pct_resolved"] or 0) > 0,
        "positive_at_least_3_of_4_days": sum(v > 0 for v in primary_stats["daily_base_net_pct"].values()) >= 3,
        "beats_old_matched": matched.get("mean_gain_pp", -np.inf) > 0,
    }
    result = {
        "request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False, "opens_new_dates": False,
        "June_HOLD_opened": False, "entries":86, "candidate_episodes":97,
        "entry_features_thresholds_and_stop_retuned": False, "same_features_and_model_seeds_as_R297": True,
        "reproduction": reproduction, "api_stats":client.stats.to_dict(), "folds":folds,
        "raw_seconds_artifact": str(args.raw_output),
        "signals_rebuilt":signal_report(scored), "signals_original_scores_on_rebuilt_labels":signal_report(rebuilt),
        "metrics":stats, "integrity_checks":checks, "integrity_gate_pass":all(checks.values()),
        "economics_checks": economics, "research_gate_pass": all(checks.values()) and all(economics.values()),
        "fill_assumption":"First regular-session second open at/after submission is a reference scenario; no quote/order-book fill guarantee. Long gaps remain paid exposure, not discarded trades.",
        "opportunity_assumption":"Observed-window MFE bounds, not proof of complete tradable opportunity. Endpoint execution is separate; missing later prints are not zero missed upside.",
        "limitations":"Reused four-day development; upstream OOF not nested within HOLD folds. Event P&L is not portfolio/account return.",
        "next_boundary":"Repair any remaining integrity issue first; if HOLD signal remains weak, design policy-consistent continuation on separate development. Keep June sealed until HOLD policy frozen.",
    }
    args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(args.decisions_output,index=False)
    scored.to_parquet(args.states_output,index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({"integrity_gate_pass":result["integrity_gate_pass"],"integrity_checks":checks},indent=2),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
