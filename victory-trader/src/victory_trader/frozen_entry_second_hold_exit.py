"""R297: development-only causal second HOLD/EXIT with R296 entry gates frozen.

Uses existing R294 OOF entry scores (not refitted scores on the same days).
The whole upstream selector is not nested inside HOLD folds: this is explicitly
an exploratory, conditional position experiment, not an untouched validation.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS, net_round_trip_return_pct
from .fresh_crack_entry_validation import (
    FROZEN_UTILITY_THRESHOLD, FROZEN_TURN_THRESHOLD,
)
from .hierarchical_crack_entry_controller import _fit, _predict
from .lagged_minute_context import CROSSFIT_DAYS
from .market_calendar import regular_session_bounds
from .massive_client import MassiveClient
from .second_path_attention_probe import SECOND_CACHE_DIR, _second_frame

REQUEST_ID = 297
KEYS = ["trading_day", "ticker", "hot_t"]
HORIZONS_S = (60, 300)
BASE = DEFAULT_EXECUTION_SCENARIOS[1]
MAX_LAG_MS = 3000
HARD_STOP_PCT = -2.5
FEATURES = (
    "position_age_s", "observed_return_pct", "observed_net_return_pct",
    "running_peak_pct", "running_trough_pct", "drawdown_pct", "bounce_pct",
    "seconds_since_peak", "seconds_since_low", "last_return_pct",
    "return_sum_5", "return_sum_20", "positive_fraction_20",
    "volume_ratio_5_20", "log_volume", "log_transactions", "active_gap_s",
    "entry_setup_score_pct", "entry_value_score_pct", "entry_turn_score_pct",
)
POLICIES = (
    "immediate", "fixed_60s", "fixed_300s", "fixed_600s",
    "old_10m_2pct_trailing", "terminal_60m", "dynamic_60s", "dynamic_300s",
)


def check_development(frame: pd.DataFrame) -> None:
    if frame.empty or not set(frame.trading_day.astype(str)).issubset(CROSSFIT_DAYS):
        raise ValueError("R297 accepts May5-8 development only; fresh June is sealed")
    if "held_out_day" not in frame or not frame.held_out_day.astype(str).eq(frame.trading_day.astype(str)).all():
        raise ValueError("entry scores must have matching held-out-day provenance")
    if frame.duplicated([*KEYS, "decision_t"]).any():
        raise ValueError("duplicate decision states")


def frozen_entries(scored: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    check_development(scored)
    rows = []
    groups = scored.groupby(KEYS, sort=False)
    for key, episode in groups:
        ordered = episode.sort_values("decision_t")
        eligible = ordered.loc[
            ordered.predicted_enter_utility_pct.ge(FROZEN_UTILITY_THRESHOLD)
            & ordered.predicted_turn_15s_pct.ge(FROZEN_TURN_THRESHOLD)
        ]
        if eligible.empty:
            continue
        chosen = eligible.iloc[0]
        rows.append({
            **dict(zip(KEYS, key)),
            "entry_t": int(chosen.execution_t),
            "entry_price": float(chosen.execution_price),
            "entry_setup_score_pct": float(chosen.rich_setup_score_pct),
            "entry_value_score_pct": float(chosen.predicted_enter_utility_pct),
            "entry_turn_score_pct": float(chosen.predicted_turn_15s_pct),
        })
    return pd.DataFrame(rows), len(groups)


def net(entry: float, exit_price: float, scenario=BASE) -> float:
    return net_round_trip_return_pct(entry, (exit_price / entry - 1) * 100, scenario)


def position_states(seconds: pd.DataFrame, entry: dict, deadline: int) -> pd.DataFrame:
    """Features use completed bars only; following opens are execution labels."""
    bars = seconds.sort_values("t").reset_index(drop=True)
    if bars.t.duplicated().any():
        raise ValueError("duplicate second bars")
    if not (float(entry["entry_price"]) > 0 and deadline > int(entry["entry_t"])):
        return pd.DataFrame()
    price = float(entry["entry_price"])
    b = bars.loc[
        bars.t.ge(int(entry["entry_t"])) & (bars.t + 1000).le(deadline)
    ].copy().reset_index(drop=True)
    if b.empty:
        return pd.DataFrame()
    if not np.isfinite(b[["o", "h", "l", "c"]].to_numpy(float)).all() or (b[["o", "h", "l", "c"]] <= 0).any().any():
        raise ValueError("invalid second prices")
    t = b.t.to_numpy(np.int64)
    c = b.c.astype(float)
    high = c.cummax().clip(lower=price)
    low = c.cummin().clip(upper=price)
    peak_t = pd.Series(np.where(c.eq(high), t + 1000, np.nan)).ffill().fillna(int(entry["entry_t"]))
    low_t = pd.Series(np.where(c.eq(low), t + 1000, np.nan)).ffill().fillna(int(entry["entry_t"]))
    returns = c.pct_change().fillna(c.iloc[0] / price - 1) * 100
    frame = pd.DataFrame({
        "decision_t": t + 1000,
        "position_age_s": (t + 1000 - int(entry["entry_t"])) / 1000,
        "observed_return_pct": (c / price - 1) * 100,
        "observed_net_return_pct": [net(price, value) for value in c],
        "running_peak_pct": (high / price - 1) * 100,
        "running_trough_pct": (low / price - 1) * 100,
        "drawdown_pct": (c / high - 1) * 100,
        "bounce_pct": (c / low - 1) * 100,
        "seconds_since_peak": (t + 1000 - peak_t) / 1000,
        "seconds_since_low": (t + 1000 - low_t) / 1000,
        "last_return_pct": returns,
        "return_sum_5": returns.rolling(5, min_periods=1).sum(),
        "return_sum_20": returns.rolling(20, min_periods=1).sum(),
        "positive_fraction_20": returns.gt(0).rolling(20, min_periods=1).mean(),
        "volume_ratio_5_20": b.v.rolling(5, min_periods=1).mean() / b.v.rolling(20, min_periods=1).mean().replace(0, np.nan),
        "log_volume": np.log1p(b.v.clip(lower=0)),
        "log_transactions": np.log1p(b.n.clip(lower=0)),
        "active_gap_s": b.t.diff().fillna(b.t.iloc[0] - int(entry["entry_t"])) / 1000,
    })
    for name in (*KEYS, "entry_t", "entry_price", "entry_setup_score_pct", "entry_value_score_pct", "entry_turn_score_pct"):
        frame[name] = entry[name]
    times = bars.t.to_numpy(np.int64)
    positions = np.searchsorted(times, frame.decision_t.to_numpy(np.int64))
    valid = positions < len(bars)
    fill_t = np.full(len(frame), np.nan)
    fill_price = np.full(len(frame), np.nan)
    fill_t[valid] = times[positions[valid]]
    fill_price[valid] = bars.o.to_numpy(float)[positions[valid]]
    lag = fill_t - frame.decision_t.to_numpy(float)
    valid &= (lag <= MAX_LAG_MS) & (fill_t <= deadline) & (fill_price > 0)
    frame["observed_low_return_pct"] = (b.l.astype(float) / price - 1) * 100
    frame["execution_t"] = fill_t
    frame["exit_price"] = fill_price
    frame["executable"] = valid
    frame["exit_now_pct"] = [net(price, p) if ok else np.nan for p, ok in zip(fill_price, valid)]
    complete = bool(valid[-1] and deadline - int(frame.decision_t.iloc[-1]) <= MAX_LAG_MS)
    frame["terminal_complete"] = complete
    # Targets compare future executable liquidation against executable EXIT now.
    for horizon in HORIZONS_S:
        target_t = np.minimum(frame.decision_t.to_numpy(np.int64) + horizon * 1000, deadline)
        if complete:
            target_t = np.minimum(target_t, int(frame.execution_t.iloc[-1]))
        pos = np.searchsorted(times, target_t)
        ok = pos < len(times)
        future_t = np.full(len(frame), np.nan)
        future_p = np.full(len(frame), np.nan)
        future_t[ok] = times[pos[ok]]
        future_p[ok] = bars.o.to_numpy(float)[pos[ok]]
        ok &= (future_t - target_t <= MAX_LAG_MS) & (future_t <= deadline) & valid
        value = np.array([net(price, p) if flag else np.nan for p, flag in zip(future_p, ok)])
        frame[f"hold_{horizon}s_advantage_pct"] = value - frame.exit_now_pct.to_numpy(float)
    return frame


def build_states(entries: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    settings = load_settings()
    client = MassiveClient(settings.massive_api_key, cache_dir=SECOND_CACHE_DIR / "request297", request_interval_seconds=0.02)
    pieces, missing, cache = [], [], {}
    for index, raw in enumerate(entries.to_dict("records")):
        key = (str(raw["trading_day"]), str(raw["ticker"]))
        bounds = regular_session_bounds(date.fromisoformat(key[0]))
        if bounds is None:
            raise ValueError("development date is closed")
        opening, closing = (int(value.timestamp() * 1000) for value in bounds)
        if not opening <= int(raw["entry_t"]) < closing:
            raise ValueError("entry outside regular session")
        deadline = min(int(raw["hot_t"]) + 3600000, closing - 1000)
        if key not in cache:
            cache[key] = _second_frame(client.second_bars_range(key[1], date.fromisoformat(key[0]), date.fromisoformat(key[0]), adjusted=False))
        frame = position_states(cache[key], raw, deadline)
        if frame.empty:
            missing.append({**{k: raw[k] for k in KEYS}, "reason": "no_position_states"})
        else:
            pieces.append(frame)
        print(f"position path {index + 1}/{len(entries)} {key}: {len(frame)} states", flush=True)
    if not pieces:
        raise ValueError("no position states")
    return pd.concat(pieces, ignore_index=True), {"missing": missing, "api_stats": client.stats.to_dict()}


def replay(group: pd.DataFrame, policy: str) -> dict:
    ordered = group.sort_values("decision_t").reset_index(drop=True)
    first = ordered.iloc[0]
    chosen = len(ordered) - 1
    reason = "terminal_cap"
    for i, row in ordered.iterrows():
        if policy == "immediate":
            chosen, reason = i, "immediate"
            break
        if row.observed_net_return_pct <= HARD_STOP_PCT:
            chosen, reason = i, "hard_stop"
            break
        if policy.startswith("fixed_"):
            horizon = int(policy.split("_")[1][:-1])
            if row.position_age_s >= horizon:
                chosen, reason = i, "fixed_horizon"
                break
        elif policy == "old_10m_2pct_trailing":
            if row.drawdown_pct <= -2.0 or row.position_age_s >= 600:
                chosen, reason = i, "trailing_stop" if row.drawdown_pct <= -2 else "max_hold"
                break
        elif policy.startswith("dynamic_"):
            horizon = int(policy.split("_")[1][:-1])
            prediction = row[f"predicted_hold_{horizon}s_pct"]
            if not np.isfinite(prediction) or prediction <= 0:
                chosen, reason = i, "model_exit"
                break
    row = ordered.iloc[chosen]
    resolved = bool(row.executable and (reason != "terminal_cap" or row.terminal_complete))
    price = float(row.entry_price)
    all_valid = ordered.loc[ordered.executable]
    available = float((all_valid.exit_price.max() / price - 1) * 100) if row.terminal_complete else np.nan
    later = all_valid.loc[all_valid.execution_t.gt(row.execution_t)]
    pre_exit = ordered.iloc[:chosen + 1]
    result = {
        **{k: first[k] for k in KEYS}, "policy": policy,
        "resolved": resolved, "exit_reason": reason if resolved else "unresolved_" + reason,
        "entry_price": price, "exit_t": int(row.execution_t) if resolved else None,
        "hold_seconds": float((row.execution_t - row.entry_t) / 1000) if resolved else None,
        "gross_return_pct": float((row.exit_price / price - 1) * 100) if resolved else np.nan,
        "mae_observed_close_pct": float(min(0, pre_exit.observed_return_pct.min())),
        "mae_pct": float(min(0, pre_exit.observed_low_return_pct.min(), (row.exit_price / price - 1) * 100)) if resolved else np.nan,
        "available_mfe_open_pct": max(0, available) if np.isfinite(available) else np.nan,
        "later_left_on_table_pp": max(0, float((later.exit_price.max() - row.exit_price) / price * 100)) if resolved and row.terminal_complete and len(later) else (0.0 if resolved and row.terminal_complete else np.nan),
    }
    for scenario in DEFAULT_EXECUTION_SCENARIOS:
        result[f"{scenario.name}_net_return_pct"] = net(price, float(row.exit_price), scenario) if resolved else np.nan
    return result


def crossfit(states: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scored, folds = [], {}
    for index, day in enumerate(CROSSFIT_DAYS):
        train = states.loc[states.trading_day.ne(day)]
        held = states.loc[states.trading_day.eq(day)].copy()
        if held.empty:
            raise ValueError(f"missing HOLD evaluation day: {day}")
        folds[day] = {"train_days": sorted(train.trading_day.unique().tolist()), "held_states": len(held)}
        for horizon in HORIZONS_S:
            model = _fit(train, FEATURES, f"hold_{horizon}s_advantage_pct", 20263700 + index * 10 + horizon, log_target=False, episode_weighting=True)
            held[f"predicted_hold_{horizon}s_pct"] = _predict(held, model)
        scored.append(held)
    return pd.concat(scored, ignore_index=True), folds


def report(rows: pd.DataFrame, total_candidates: int) -> dict:
    output = {}
    for policy, group in rows.groupby("policy", sort=False):
        valid = group.loc[group.resolved]
        y = valid.base_net_return_pct
        daily = valid.groupby("trading_day").base_net_return_pct.mean()
        item = {
            "entries": len(group), "resolved": len(valid), "resolution_rate": len(valid) / len(group),
            "mean_base_net_pct_resolved": float(y.mean()) if len(y) else None,
            "cash_adjusted_mean_base_net_pct": float(y.sum() / total_candidates) if len(valid) == len(group) else None,
            "median_base_net_pct": float(y.median()) if len(y) else None,
            "positive_rate": float(y.gt(0).mean()) if len(y) else None,
            "loss_below_minus5_rate": float(y.lt(-5).mean()) if len(y) else None,
            "cvar5_pct": float(y.loc[y.le(y.quantile(.05))].mean()) if len(y) else None,
            "mean_hold_s": float(valid.hold_seconds.mean()) if len(valid) else None,
            "mean_mae_observed_close_pct": float(valid.mae_observed_close_pct.mean()) if len(valid) else None,
            "mean_mae_pct": float(valid.mae_pct.mean()) if len(valid) else None,
            "mean_left_on_table_pp": float(valid.later_left_on_table_pp.mean()) if valid.later_left_on_table_pp.notna().any() else None,
            "daily_base_net_pct": {str(k): float(v) for k, v in daily.items()},
            "exit_reasons": group.exit_reason.value_counts().to_dict(),
            "cost_scenarios": {s.name: float(valid[f"{s.name}_net_return_pct"].mean()) if len(valid) else None for s in DEFAULT_EXECUTION_SCENARIOS},
            "tail_capture": {},
        }
        for threshold in (5, 10, 20):
            tail = valid.loc[valid.available_mfe_open_pct.ge(threshold)]
            item["tail_capture"][f"plus{threshold}"] = {
                "opportunities": len(tail),
                "mean_gross_capture_fraction": float((tail.gross_return_pct / tail.available_mfe_open_pct).mean()) if len(tail) else None,
                "realized_net_reached_rate": float(tail.base_net_return_pct.ge(threshold).mean()) if len(tail) else None,
            }
        output[policy] = item
    primary = rows.loc[rows.policy.eq("dynamic_300s") & rows.resolved]
    old = rows.loc[rows.policy.eq("old_10m_2pct_trailing") & rows.resolved]
    pair = primary.merge(old, on=KEYS, suffixes=("_dynamic", "_old"))
    if len(pair):
        pair["gain"] = pair.base_net_return_pct_dynamic - pair.base_net_return_pct_old
        daily = pair.groupby("trading_day").gain.mean()
        rng = np.random.default_rng(20263797)
        intervals = {}
        for label, cluster in (("day", "trading_day"), ("ticker_day", "ticker_day")):
            work = pair.assign(ticker_day=pair.trading_day + "|" + pair.ticker)
            values = [g.gain.to_numpy(float) for _, g in work.groupby(cluster)]
            draws = [float(np.concatenate([values[i] for i in rng.integers(0, len(values), len(values))]).mean()) for _ in range(1000)]
            intervals[label] = {"clusters": len(values), "gain_ci95_pp": np.quantile(draws, [.025, .975]).tolist()}
        output["matched_primary_vs_old"] = {"episodes": len(pair), "mean_gain_pp": float(pair.gain.mean()), "daily_gain_pp": daily.to_dict(), "bootstrap": intervals}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states-output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path, required=True)
    args = parser.parse_args()
    entries, total = frozen_entries(pd.read_parquet(args.entry_scored))
    if entries.empty:
        raise ValueError("no frozen entries")
    states, acquisition = build_states(entries)
    args.states_output.parent.mkdir(parents=True, exist_ok=True)
    # Persist acquisition before fitting so a later numerical error loses no data.
    states.to_parquet(args.states_output, index=False)
    scored, folds = crossfit(states)
    rows = [replay(group, policy) for _, group in scored.groupby(KEYS, sort=False) for policy in POLICIES]
    present = set(tuple(x) for x in scored[KEYS].drop_duplicates().itertuples(index=False, name=None))
    for raw in entries.to_dict("records"):
        if tuple(raw[k] for k in KEYS) not in present:
            rows.extend({**{k: raw[k] for k in KEYS}, "policy": p, "resolved": False, "exit_reason": "no_position_states"} for p in POLICIES)
    decisions = pd.DataFrame(rows)
    metrics = report(decisions, total)
    dynamic = metrics["dynamic_300s"]
    comparison = metrics.get("matched_primary_vs_old", {})
    checks = {
        "resolution_at_least_95pct": dynamic["resolution_rate"] >= .95,
        "positive_base_return": (dynamic["mean_base_net_pct_resolved"] or 0) > 0,
        "beats_old_matched": comparison.get("mean_gain_pp", -np.inf) > 0,
        "positive_at_least_3_of_4_days": sum(v > 0 for v in dynamic["daily_base_net_pct"].values()) >= 3,
    }
    result = {
        "request_id": REQUEST_ID, "development_only": True, "promotion_eligible": False,
        "opens_new_dates": False, "fresh_June_HOLD_validation_opened": False,
        "entry_contract": {"scores": "existing R294 day-OOF", "utility_threshold": FROZEN_UTILITY_THRESHOLD, "turn_threshold": FROZEN_TURN_THRESHOLD, "override": "disabled", "retuned": False},
        "limitation": "Conditional exploratory position crossfit; upstream OOF selector/entry training is not nested within HOLD folds. Four reused development days cannot establish independent profitability.",
        "position_features": list(FEATURES), "primary": "dynamic_300s", "sensitivity": "dynamic_60s",
        "hold_rule": "positive predicted fixed-horizon liquidation advantage; re-evaluate every completed active second; no threshold search",
        "risk": {"common_hard_stop_net_pct": HARD_STOP_PCT, "terminal": "HOT+60m or regular close minus 1s", "max_execution_lag_ms": MAX_LAG_MS},
        "old_comparator": "Frozen -2.5 net stop / 2pct close-trailing / 10m cap replayed at second resolution on identical entries; not an exact reproduction of old minute replay",
        "candidate_episodes": total, "entries": len(entries), "position_states": len(states), "acquisition": acquisition,
        "outer_folds": folds, "metrics": metrics, "checks": checks, "research_gate_pass": all(checks.values()),
        "next_boundary": "Analyze development results; freeze HOLD before scoring June15-22 once. No fresh-date policy tuning. Event returns are not portfolio returns.",
    }
    scored.to_parquet(args.states_output, index=False)
    args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(args.decisions_output, index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"research_gate_pass": result["research_gate_pass"], "checks": checks}, indent=2), flush=True)
    # A negative research finding is a completed run, not a workflow failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
