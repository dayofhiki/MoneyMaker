"""Request243: chronological policy-rollout action values, development only.

No realized-future maximization is used to make a training target. A frozen
policy trained on earlier days acts on later-day causal states; its realized
continuations label ENTER, WAIT and HOLD-minus-EXIT. This is ONE conservative
policy-improvement experiment, not a converged optimal trader or portfolio.

The source position artifact timestamps are completed-minute availability times.
exit_reference_open is the following bar's open at that timestamp, an execution
LABEL, never a feature. Missing references are unresolved, never cash returns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version

import joblib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .execution_costs import (
    DEFAULT_EXECUTION_SCENARIOS,
    modeled_sell_fill,
    net_round_trip_return_pct,
)

REQUEST_ID = 243
KEYS = ["trading_day", "ticker", "hot_t"]
FIT_DAYS = [
    "2026-04-30",
    "2026-05-01",
    "2026-05-04",
    "2026-05-05",
    "2026-05-06",
    "2026-05-07",
    "2026-05-08",
]
CAL_DAYS = [
    "2026-05-11",
    "2026-05-12",
    "2026-05-13",
    "2026-05-14",
    "2026-05-15",
    "2026-05-18",
    "2026-05-19",
    "2026-05-20",
]
DEV_DAYS = ["2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26", "2026-06-29"]
WATCH_MINUTES = 5
WATCH_EVENTS = 5
# Request171 materializes minutes 1..29. Minute 29 is a precommitted exit
# deadline, not the hindsight last available row. Absent deadline => unresolved.
EXIT_DEADLINE = 29
BASE = next(s for s in DEFAULT_EXECUTION_SCENARIOS if s.name == "base")
STRESS = next(s for s in DEFAULT_EXECUTION_SCENARIOS if s.name == "stress")
CAUSAL_SOURCE_FEATURES = (
    "attention_score",
    "attention_rank",
    "return_from_previous_close_pct",
    "minute_body_return_pct",
    "minute_range_pct",
    "log_minute_volume",
    "log_minute_transactions",
    "minute_return_1m_pct",
    "return_accel_1m_pct",
    "volume_ratio_prev1",
    "transactions_ratio_prev1",
    "active_seconds_60",
    "active_seconds_10",
    "last_activity_age_s",
    "sec_last10_return_pct",
    "sec_prev10_return_pct",
    "sec_accel_10_pct",
    "sec_first30_return_pct",
    "sec_last30_return_pct",
    "sec_accel_30_pct",
    "sec_realized_vol_pct",
    "sec_positive_fraction",
    "sec_max_drawdown_pct",
    "sec_max_runup_pct",
    "sec_volume_last10_share",
    "sec_transactions_last10_share",
    "sec_volume_burst_10",
    "sec_transactions_burst_10",
    "sec_last5_return_pct",
    "sec_prev5_return_pct",
    "sec_accel_5_pct",
    "sec_last15_return_pct",
    "sec_prev15_return_pct",
    "sec_accel_15_pct",
    "sec_volume_burst_5",
    "sec_transactions_burst_5",
    "sec_volume_last5_share",
    "sec_transactions_last5_share",
    "sec_signed_volume_imbalance_60",
    "sec_signed_transactions_imbalance_60",
    "sec_return_efficiency_60",
    "sec_sign_flip_rate_60",
    "sec_seconds_since_high",
    "sec_seconds_since_low",
    "sec_close_location_60",
    "sec_close_vs_vwap_pct",
    "sec_volatility_ratio_10_to_prev50",
    "sec_mean_trade_size_ratio_10_to_prev50",
)
PATH_FEATURES = (
    "elapsed_minutes",
    "event_index",
    "gap_minutes",
    "path_return_pct",
    "path_drawdown_pct",
    "path_rebound_pct",
    "event_return_pct",
    "lag1_event_return_pct",
    "lag2_event_return_pct",
    "log_current_price",
    "base_zero_move_cost_proxy_pct",
    "minutes_since_open",
    "minutes_to_close",
)
TARGETS = ("enter_value", "wait_value", "hold_advantage")


@dataclass
class Policy:
    heads: dict
    columns: tuple[str, ...]
    training_days: tuple[str, ...]
    support: dict


def finite(value):
    return bool(pd.notna(value) and np.isfinite(float(value)))


def net(entry, exit_price, scenario=BASE):
    if not (finite(entry) and finite(exit_price) and entry > 0 and exit_price > 0):
        return np.nan
    return float(
        net_round_trip_return_pct(entry, (exit_price / entry - 1) * 100, scenario)
    )


def build_states(positions):
    required = {*KEYS, "state_t", "log_current_close", "exit_reference_open"}
    if required - set(positions):
        raise ValueError(f"missing source columns: {required - set(positions)}")
    if positions.duplicated(KEYS + ["state_t"]).any():
        raise ValueError("duplicate episode state timestamp")
    records = []
    for _, group in positions.groupby(KEYS, sort=False):
        previous_price = first_price = peak = trough = np.nan
        previous_t = np.nan
        lag1 = lag2 = np.nan
        ordinal = 0
        for row in group.sort_values("state_t").to_dict("records"):
            elapsed = (float(row["state_t"]) - float(row["hot_t"])) / 60000
            if not 0 < elapsed <= EXIT_DEADLINE:
                continue
            ordinal += 1
            log_close = row["log_current_close"]
            price = float(np.exp(log_close)) if finite(log_close) else np.nan
            # Never fall back to a future fill when the observable close is absent.
            r = {k: row[k] for k in KEYS + ["state_t"]}
            for c in CAUSAL_SOURCE_FEATURES:
                r[c] = row.get(c, np.nan)
            change = (
                (price / previous_price - 1) * 100
                if finite(previous_price) and finite(price)
                else np.nan
            )
            if finite(price):
                first_price = price if not finite(first_price) else first_price
                peak = max(peak, price) if finite(peak) else price
                trough = min(trough, price) if finite(trough) else price
            r.update(
                elapsed_minutes=elapsed,
                event_index=ordinal,
                gap_minutes=(row["state_t"] - previous_t) / 60000
                if finite(previous_t)
                else np.nan,
                path_return_pct=(price / first_price - 1) * 100
                if finite(first_price)
                else np.nan,
                path_drawdown_pct=(price / peak - 1) * 100 if finite(peak) else np.nan,
                path_rebound_pct=(price / trough - 1) * 100
                if finite(trough)
                else np.nan,
                event_return_pct=change,
                lag1_event_return_pct=lag1,
                lag2_event_return_pct=lag2,
                log_current_price=log_close,
                base_zero_move_cost_proxy_pct=-net(price, price),
                minutes_since_open=row.get("minutes_since_open", np.nan),
                minutes_to_close=row.get("minutes_to_close", np.nan),
                execution_open=row["exit_reference_open"],
            )
            r["can_enter"] = bool(
                elapsed <= WATCH_MINUTES and ordinal <= WATCH_EVENTS and finite(price)
            )
            r["terminal"] = bool(
                elapsed >= EXIT_DEADLINE
                or (finite(r["minutes_to_close"]) and r["minutes_to_close"] <= 1)
            )
            records.append(r)
            previous_price, previous_t = price, row["state_t"]
            lag2, lag1 = lag1, change
    return pd.DataFrame(records).reset_index(drop=True)


def features(frame, columns):
    return (
        frame.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def actions(frame, policy, mode="full"):
    """Decision interface deliberately accepts causal inputs, never fill labels."""
    if policy is None:
        q = np.tile([1.0, 0.0, -1.0], (len(frame), 1))
    else:
        x = features(frame, policy.columns)
        q = np.column_stack([policy.heads[t].predict(x) for t in TARGETS])
    q_enter, q_wait, q_hold = q.T
    allowed = frame["can_enter"].to_numpy(bool) & ~frame["terminal"].to_numpy(bool)
    enter = allowed & (q_enter > 0) & (q_enter >= q_wait)
    wait = allowed & ~enter & (q_wait > 0)
    if mode == "no_wait":
        # A true one-shot ablation: later states are never reconsidered.
        enter = allowed & frame["event_index"].eq(1).to_numpy() & (q_enter > 0)
        wait[:] = False
    flat = np.where(enter, "ENTER", np.where(wait, "WAIT", "ABSTAIN"))
    held = np.where((q_hold > 0) & ~frame["terminal"].to_numpy(bool), "HOLD", "EXIT")
    if mode == "no_hold":
        held[:] = "EXIT"
    return flat, held, q


def rollout(frame, policy, mode="full"):
    """Backward bookkeeping of a CAUSALLY chosen policy, not max of outcomes.

    Each q target is the realized continuation under a fixed teacher. A WAIT
    that never enters has known cash=0. An ENTER without an observed fill/exit
    has unknown P&L, even if later data would make it convenient to drop it.
    """
    out = frame.reset_index(drop=True).copy()
    decision_columns = list(
        dict.fromkeys(
            [*CAUSAL_SOURCE_FEATURES, *PATH_FEATURES, "can_enter", "terminal"]
        )
    )
    flat, held, q = actions(out[decision_columns], policy, mode)
    for j, t in enumerate(TARGETS):
        out["predicted_" + t] = q[:, j]
        out[t] = np.nan
    out["flat_action"], out["held_action"] = flat, held
    out["policy_value"] = np.nan
    out["policy_entry_index"] = -1
    out["policy_exit_index"] = -1
    out["forced_enter_exit_index"] = -1
    for _, group in out.groupby(KEYS, sort=False):
        ids = list(group.index)
        n = len(ids)
        held_exit = [-1] * n
        flat_value = [0.0] * n
        flat_entry, flat_exit = [-1] * n, [-1] * n
        for k in range(n - 1, -1, -1):
            i = ids[k]
            next_exit = held_exit[k + 1] if k + 1 < n else -1
            # The deadline is observable; never liquidate merely because this
            # happens to be the last row in a historical file.
            held_exit[k] = i if held[i] == "EXIT" else next_exit
            enter_value = np.nan
            hold_advantage = np.nan
            if next_exit >= 0:
                enter_value = net(
                    out.at[i, "execution_open"], out.at[next_exit, "execution_open"]
                )
                now = out.at[i, "execution_open"]
                later = out.at[next_exit, "execution_open"]
                if finite(now) and finite(later) and now > 0 and later > 0:
                    now_sell = modeled_sell_fill(now, BASE)
                    if now_sell > 0:
                        hold_advantage = (
                            modeled_sell_fill(later, BASE) / now_sell - 1
                        ) * 100
            wait_value = flat_value[k + 1] if k + 1 < n else 0.0
            if bool(out.at[i, "can_enter"]) and not bool(out.at[i, "terminal"]):
                out.at[i, "enter_value"] = enter_value
                out.at[i, "wait_value"] = wait_value
                out.at[i, "forced_enter_exit_index"] = next_exit
            if not bool(out.at[i, "terminal"]):
                out.at[i, "hold_advantage"] = hold_advantage
            if flat[i] == "ENTER":
                flat_value[k], flat_entry[k], flat_exit[k] = enter_value, i, next_exit
            elif flat[i] == "WAIT" and k + 1 < n:
                flat_value[k], flat_entry[k], flat_exit[k] = (
                    flat_value[k + 1],
                    flat_entry[k + 1],
                    flat_exit[k + 1],
                )
            out.at[i, "policy_value"] = flat_value[k]
            out.at[i, "policy_entry_index"] = flat_entry[k]
            out.at[i, "policy_exit_index"] = flat_exit[k]
    return out


def fit_policy(labeled, *, min_rows=300):
    columns = tuple(
        c
        for c in dict.fromkeys([*CAUSAL_SOURCE_FEATURES, *PATH_FEATURES])
        if c in labeled and pd.to_numeric(labeled[c], errors="coerce").notna().any()
    )
    heads, support = {}, {}
    for j, target in enumerate(TARGETS):
        valid = np.isfinite(pd.to_numeric(labeled[target], errors="coerce"))
        train = labeled.loc[valid]
        episodes = len(train[KEYS].drop_duplicates())
        if len(train) < min_rows or episodes < max(2, min_rows // 6):
            raise ValueError(f"{target}: only {len(train)} rows / {episodes} episodes")
        y = train[target].to_numpy(float)
        low, high = np.quantile(y, [0.005, 0.995])
        # Equal day, then equal episode: dense paths do not dominate the loss.
        sizes = train.groupby(KEYS)["state_t"].transform("size").to_numpy(float)
        ep_per_day = train[KEYS].drop_duplicates().groupby("trading_day").size()
        w = 1 / sizes / train.trading_day.map(ep_per_day).to_numpy(float)
        w /= w.mean()
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=7,
            min_samples_leaf=60,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=20261343 + j,
        )
        model.fit(features(train, columns), np.clip(y, low, high), sample_weight=w)
        heads[target] = model
        support[target] = {
            "rows": len(train),
            "episodes": episodes,
            "unresolved_target_rows": int(
                (
                    ~valid
                    & (
                        labeled.can_enter
                        if target != "hold_advantage"
                        else ~labeled.terminal
                    )
                ).sum()
            ),
            "winsor_low": float(low),
            "winsor_high": float(high),
        }
    return Policy(heads, columns, tuple(sorted(labeled.trading_day.unique())), support)


def chronological_fit(states, *, min_rows=300):
    days = sorted(states.trading_day.unique())
    if days != FIT_DAYS:
        raise ValueError(f"fit dates changed: {days}")
    early = states.loc[states.trading_day.isin(FIT_DAYS[:3])].reset_index(drop=True)
    later = states.loc[states.trading_day.isin(FIT_DAYS[3:])].reset_index(drop=True)
    teacher = fit_policy(rollout(early, None), min_rows=min_rows)
    # Teacher has NEVER trained on the later target-generation dates.
    if max(teacher.training_days) >= min(later.trading_day):
        raise ValueError("teacher must precede every student training day")
    targets = rollout(later, teacher)
    student = fit_policy(targets, min_rows=min_rows)
    return teacher, student, targets


def episode_results(scored):
    rows = []
    for key, part in scored.groupby(KEYS, sort=False):
        first = part.iloc[0]
        entry, exit_index = int(first.policy_entry_index), int(first.policy_exit_index)
        value = float(first.policy_value)
        executed = entry >= 0
        unresolved = executed and not finite(value)
        stress_value = np.nan if unresolved else 0.0
        entry_t = exit_t = None
        if executed:
            entry_t = int(scored.at[entry, "state_t"])
            if exit_index >= 0:
                exit_t = int(scored.at[exit_index, "state_t"])
                stress_value = net(
                    scored.at[entry, "execution_open"],
                    scored.at[exit_index, "execution_open"],
                    STRESS,
                )
        initial_waits = 0
        for action in part.flat_action:
            if action != "WAIT":
                break
            initial_waits += 1
        actual_holds = 0
        if entry >= 0:
            actual_holds = int(
                part.loc[
                    (part.index > entry)
                    & ((part.index < exit_index) if exit_index >= 0 else True),
                    "held_action",
                ]
                .eq("HOLD")
                .sum()
            )
        rows.append(
            dict(zip(KEYS, key))
            | {
                "entered": executed,
                "unresolved": unresolved,
                "net_pct": value,
                "stress_pct": stress_value,
                "entry_t": entry_t,
                "exit_t": exit_t,
                "wait_actions": initial_waits,
                "hold_actions": actual_holds,
            }
        )
    return pd.DataFrame(rows)


def metrics(episodes, days):
    trades = episodes.loc[episodes.entered & ~episodes.unresolved]
    # An unresolved trade poisons its day's mean: it is NOT silently dropped.
    daily = {}
    for day in days:
        part = episodes.loc[episodes.trading_day.eq(day)]
        daily[day] = (
            None
            if len(part) == 0 or part.unresolved.any()
            else float(part.net_pct.mean())
        )
    complete = all(v is not None for v in daily.values())
    values = np.array(list(daily.values()), float) if complete else np.array([])
    low = None
    if len(values) >= 2:
        rng = np.random.default_rng(20261343)
        low = float(
            np.quantile(
                rng.choice(values, (10000, len(values)), replace=True).mean(axis=1),
                0.025,
            )
        )
    return {
        "episodes": len(episodes),
        "entries": int(episodes.entered.sum()),
        "resolved_trades": len(trades),
        "unresolved_trades": int(episodes.unresolved.sum()),
        "execution_rate": float(episodes.entered.mean()) if len(episodes) else 0.0,
        "resolved_trade_mean_pct": float(trades.net_pct.mean())
        if len(trades)
        else None,
        "resolved_stress_trade_mean_pct": float(trades.stress_pct.mean())
        if len(trades)
        else None,
        "day_balanced_episode_mean_pct": float(values.mean()) if len(values) else None,
        "note": "equal episode-notional research proxy; not portfolio return or compounded balance",
        "by_day_episode_mean_pct": daily,
        "positive_days": sum(v is not None and v > 0 for v in daily.values()),
        "day_cluster_bootstrap_low_pct": low,
        "severe_loss_rate_le_minus2": float(trades.net_pct.le(-2).mean())
        if len(trades)
        else None,
        "wait_actions": int(episodes.wait_actions.sum()),
        "hold_actions": int(episodes.hold_actions.sum()),
    }


def evaluate_block(states, teacher, student, days):
    reports, all_rows = {}, []
    for name, policy, mode in [
        ("full", student, "full"),
        ("earlier_teacher", teacher, "full"),
        ("no_wait", student, "no_wait"),
        ("no_hold", student, "no_hold"),
        ("first_event_one_step", None, "full"),
    ]:
        rows = episode_results(rollout(states, policy, mode))
        reports[name] = metrics(rows, days)
        rows["policy"] = name
        all_rows.append(rows)
    reports["cash"] = {"day_balanced_episode_mean_pct": 0.0}
    full = reports["full"]
    checks = {
        "all_trades_resolved": full["unresolved_trades"] == 0,
        "at_least_20_trades": full["resolved_trades"] >= 20,
        "positive_episode_mean": (full["day_balanced_episode_mean_pct"] or 0) > 0,
        "positive_trade_mean": (full["resolved_trade_mean_pct"] or 0) > 0,
        "positive_cluster_lower_bound": (full["day_cluster_bootstrap_low_pct"] or 0)
        > 0,
        "positive_day_majority": full["positive_days"] >= (len(days) // 2 + 1),
        "uses_wait_and_hold": full["wait_actions"] >= 5 and full["hold_actions"] >= 5,
    }
    for name in ("earlier_teacher", "no_wait", "no_hold", "first_event_one_step"):
        reference = reports[name]["day_balanced_episode_mean_pct"]
        candidate = full["day_balanced_episode_mean_pct"]
        checks["beats_" + name] = (
            candidate is not None and reference is not None and candidate > reference
        )
    return {
        "policies": reports,
        "checks": checks,
        "gate_pass": all(checks.values()),
    }, pd.concat(all_rows, ignore_index=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(fit_path, cal_path, dev_dir, output, rows_output):
    fit = build_states(pd.read_parquet(fit_path))
    cal = build_states(pd.read_parquet(cal_path))
    if sorted(cal.trading_day.unique()) != CAL_DAYS:
        raise ValueError("calibration dates changed")
    teacher, student, targets = chronological_fit(fit)
    cal_report, cal_rows = evaluate_block(cal, teacher, student, CAL_DAYS)
    cal_rows["split"] = "calibration"
    all_rows = [cal_rows]
    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "promotion_gate_pass": False,
        "population": "Request171/178 BUY-gated archived episodes; not full-HOT or full-market coverage",
        "runtime": {
            k: version(k) for k in ["numpy", "pandas", "scikit-learn", "pyarrow"]
        },
        "python": platform.python_version(),
        "teacher_days": teacher.training_days,
        "student_days": student.training_days,
        "calibration_days": CAL_DAYS,
        "development_days": DEV_DAYS,
        "teacher_support": teacher.support,
        "student_support": student.support,
        "features": student.columns,
        "calibration": cal_report,
        "development": {"status": "not_opened_calibration_gate_failed"},
        "execution": {
            "source_state_time": "completed bar availability; execution_open is next-bar open label",
            "watch_minutes": WATCH_MINUTES,
            "watch_events": WATCH_EVENTS,
            "precommitted_exit_minute": EXIT_DEADLINE,
            "missing_execution": "unresolved, never zero or hindsight last-row exit",
            "reentry": False,
            "portfolio_capital_constraints": False,
        },
        "input_sha256": {
            str(fit_path): sha256(fit_path),
            str(cal_path): sha256(cal_path),
        },
    }
    if cal_report["gate_pass"]:
        paths = [dev_dir / f"{day}-positions.parquet" for day in DEV_DAYS]
        dev = build_states(
            pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        )
        if sorted(dev.trading_day.unique()) != DEV_DAYS:
            raise ValueError("development dates changed")
        result["development"], dev_rows = evaluate_block(
            dev, teacher, student, DEV_DAYS
        )
        dev_rows["split"] = "development"
        all_rows.append(dev_rows)
        result["input_sha256"].update({str(p): sha256(p) for p in paths})
    result["development_gate_pass"] = bool(
        cal_report["gate_pass"] and result["development"].get("gate_pass", False)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.concat(all_rows, ignore_index=True).to_parquet(rows_output, index=False)
    targets.to_parquet(
        rows_output.with_name("request243-student-targets.parquet"), index=False
    )
    joblib.dump(
        {"teacher": teacher, "student": student},
        rows_output.with_name("request243-policies.joblib"),
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fit-positions", type=Path, required=True)
    p.add_argument("--calibration-positions", type=Path, required=True)
    p.add_argument("--fresh-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows-output", type=Path, required=True)
    a = p.parse_args()
    return evaluate(
        a.fit_positions, a.calibration_positions, a.fresh_dir, a.output, a.rows_output
    )


if __name__ == "__main__":
    raise SystemExit(main())
