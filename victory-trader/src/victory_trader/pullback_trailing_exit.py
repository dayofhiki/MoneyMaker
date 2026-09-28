"""Request273: causal pullback entry with precommitted trailing/risk exits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .causal_minute_controller_rebuild import causal_execution_scan
from .causal_pullback_entry_audit import (
    ENTRY_WINDOW_MINUTES,
    RISK_CAP_MINUTES,
    pick_entry,
)
from .full_hot_fixed_policy_value import attach_fixed_value, event_lookup
from .learned_pullback_entry import CONTROLLER_TRAIN_DAYS, TEST_DAYS
from .recurrent_wait_entry_action_value import _base_return
from .rich_post_hot_state import score_and_select_candidates

REQUEST_ID = 273
RULE_FIT_DAYS = ("2026-05-05", "2026-05-06", "2026-05-07")
RULE_CHECK_DAY = "2026-05-08"
STOP_LOSSES = (-1.5, -2.5, -4.0)
TRAILS = (1.0, 2.0, 3.0)
MAX_HOLD_MINUTES = (5, 10, 20, 30)

MIN_FIT_ENTRIES = 25
MIN_RESOLUTION = 0.95
MIN_FIT_GAIN = 0.10
MAX_FIT_SEVERE = 0.25
MIN_FIT_POSITIVE_DAYS = 2
MAX_CHECK_SEVERE = 0.25

MIN_TEST_ENTRIES = 20
MIN_TEST_POSITIVE_DAYS = 5
MIN_TEST_TRADE_POSITIVE_RATE = 0.40
MAX_TEST_SEVERE = 0.15
MIN_TEST_GAIN = 0.10


def build_pullback_episodes(
    selected: pd.DataFrame,
    causal_scan: pd.DataFrame,
) -> pd.DataFrame:
    lookup = event_lookup(causal_scan)
    rows = []
    for raw in selected.itertuples(index=False):
        day = str(raw.trading_day)
        ticker = str(raw.ticker).upper()
        hot_t = int(raw.t)
        entry_limit = hot_t + ENTRY_WINDOW_MINUTES * 60_000
        risk_cap = hot_t + RISK_CAP_MINUTES * 60_000
        episode = lookup.get((day, ticker), [])
        entry_events = [
            (int(t), float(price))
            for t, price in episode
            if hot_t < int(t) <= entry_limit and float(price) > 0
        ]
        entry = pick_entry(entry_events, "pullback_2")
        if entry is None:
            rows.append(
                {
                    "trading_day": day,
                    "ticker": ticker,
                    "hot_t": hot_t,
                    "entered": False,
                    "entry_t": None,
                    "entry_price": None,
                    "path": [],
                }
            )
            continue
        entry_t, entry_price = entry
        path = [
            (int(t), float(price))
            for t, price in episode
            if int(entry_t) < int(t) <= risk_cap and float(price) > 0
        ]
        rows.append(
            {
                "trading_day": day,
                "ticker": ticker,
                "hot_t": hot_t,
                "entered": True,
                "entry_t": int(entry_t),
                "entry_price": float(entry_price),
                "path": path,
            }
        )
    return pd.DataFrame(rows)


def apply_exit_rule(
    episodes: pd.DataFrame,
    *,
    stop_loss_pct: float,
    trail_pct: float,
    max_hold_minutes: int,
    policy_name: str,
) -> pd.DataFrame:
    rows = []
    for raw in episodes.to_dict("records"):
        base = {
            "trading_day": str(raw["trading_day"]),
            "ticker": str(raw["ticker"]).upper(),
            "hot_t": int(raw["hot_t"]),
            "entered": bool(raw["entered"]),
            "policy": policy_name,
        }
        if not bool(raw["entered"]):
            rows.append(
                {
                    **base,
                    "resolved": True,
                    "economic_return_pct": 0.0,
                    "trade_return_pct": np.nan,
                    "exit_t": None,
                    "exit_reason": "cash_no_pullback",
                }
            )
            continue
        path = list(raw["path"])
        if not path:
            rows.append(
                {
                    **base,
                    "resolved": False,
                    "economic_return_pct": np.nan,
                    "trade_return_pct": np.nan,
                    "exit_t": None,
                    "exit_reason": "no_post_entry_state",
                }
            )
            continue
        entry_t = int(raw["entry_t"])
        entry_price = float(raw["entry_price"])
        running_high = entry_price
        deadline = min(
            int(raw["hot_t"]) + RISK_CAP_MINUTES * 60_000,
            entry_t + int(max_hold_minutes) * 60_000,
        )
        chosen_t, chosen_price = None, None
        reason = "missing_deadline_state"
        for state_t, price in path:
            # A missing deadline cannot be settled at an earlier/later price.
            if int(state_t) > deadline:
                break
            running_high = max(running_high, float(price))
            ret = _base_return(entry_price, float(price))
            drawdown = (
                (float(price) / running_high - 1.0) * 100.0
                if running_high > 0
                else 0.0
            )
            elapsed = (int(state_t) - entry_t) / 60_000
            if ret <= float(stop_loss_pct):
                chosen_t, chosen_price = int(state_t), float(price)
                reason = "hard_stop"
                break
            if drawdown <= -float(trail_pct):
                chosen_t, chosen_price = int(state_t), float(price)
                reason = "trailing_stop"
                break
            if int(state_t) == deadline:
                chosen_t, chosen_price = int(state_t), float(price)
                reason = (
                    "max_hold" if elapsed >= int(max_hold_minutes)
                    else "terminal_cap"
                )
                break
        if chosen_t is None:
            rows.append({
                **base,
                "resolved": False,
                "economic_return_pct": np.nan,
                "trade_return_pct": np.nan,
                "exit_t": None,
                "exit_reason": reason,
            })
            continue
        trade_return = _base_return(entry_price, chosen_price)
        rows.append(
            {
                **base,
                "resolved": True,
                "economic_return_pct": float(trade_return),
                "trade_return_pct": float(trade_return),
                "exit_t": int(chosen_t),
                "exit_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def first_exit_baseline(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw in episodes.to_dict("records"):
        base = {
            "trading_day": str(raw["trading_day"]),
            "ticker": str(raw["ticker"]).upper(),
            "hot_t": int(raw["hot_t"]),
            "entered": bool(raw["entered"]),
            "policy": "pullback_first_exit",
        }
        if not bool(raw["entered"]):
            rows.append(
                {
                    **base,
                    "resolved": True,
                    "economic_return_pct": 0.0,
                    "trade_return_pct": np.nan,
                    "exit_t": None,
                    "exit_reason": "cash_no_pullback",
                }
            )
            continue
        path = list(raw["path"])
        if not path:
            rows.append(
                {
                    **base,
                    "resolved": False,
                    "economic_return_pct": np.nan,
                    "trade_return_pct": np.nan,
                    "exit_t": None,
                    "exit_reason": "no_post_entry_state",
                }
            )
            continue
        state_t, price = path[0]
        trade_return = _base_return(float(raw["entry_price"]), float(price))
        rows.append(
            {
                **base,
                "resolved": True,
                "economic_return_pct": float(trade_return),
                "trade_return_pct": float(trade_return),
                "exit_t": int(state_t),
                "exit_reason": "first_state",
            }
        )
    return pd.DataFrame(rows)


def policy_metrics(rows: pd.DataFrame) -> dict:
    resolved = rows.loc[rows.resolved.astype(bool)].copy()
    economic = pd.to_numeric(
        resolved.economic_return_pct,
        errors="coerce",
    )
    entered = rows.entered.astype(bool)
    trade = pd.to_numeric(
        resolved.loc[resolved.entered.astype(bool), "trade_return_pct"],
        errors="coerce",
    )
    daily = resolved.groupby(
        resolved.trading_day.astype(str),
        sort=True,
    ).economic_return_pct.mean()
    return {
        "candidate_episodes": int(len(rows)),
        "entries": int(entered.sum()),
        "cash_episodes": int((~entered).sum()),
        "unresolved_entries": int(
            (entered & ~rows.resolved.astype(bool)).sum()
        ),
        "resolution_or_cash_rate": (
            float(rows.resolved.astype(bool).mean()) if len(rows) else 0.0
        ),
        "candidate_mean_pct": (
            float(economic.mean()) if len(economic) else None
        ),
        "day_balanced_candidate_mean_pct": (
            float(daily.mean()) if len(daily) else None
        ),
        "positive_candidate_mean_days": int((daily > 0).sum()),
        "trade_count": int(len(trade)),
        "trade_mean_pct": float(trade.mean()) if len(trade) else None,
        "trade_positive_rate": (
            float(trade.gt(0).mean()) if len(trade) else None
        ),
        "trade_severe_loss_rate_le_minus2": (
            float(trade.le(-2).mean()) if len(trade) else None
        ),
        "by_day": {
            str(day): {
                "candidates": int(len(part)),
                "candidate_mean_pct": float(
                    pd.to_numeric(
                        part.economic_return_pct,
                        errors="coerce",
                    ).mean()
                ),
            }
            for day, part in resolved.groupby(
                resolved.trading_day.astype(str),
                sort=True,
            )
        },
    }


def evaluate_rule_grid(
    episodes: pd.DataFrame,
) -> tuple[dict | None, list[dict], dict]:
    baseline_rows = first_exit_baseline(episodes)
    baseline = policy_metrics(baseline_rows)
    results = []
    passing = []
    for stop_loss in STOP_LOSSES:
        for trail in TRAILS:
            for max_hold in MAX_HOLD_MINUTES:
                rows = apply_exit_rule(
                    episodes,
                    stop_loss_pct=stop_loss,
                    trail_pct=trail,
                    max_hold_minutes=max_hold,
                    policy_name="calibration_grid",
                )
                metrics = policy_metrics(rows)
                gain = (
                    metrics["candidate_mean_pct"]
                    - baseline["candidate_mean_pct"]
                    if metrics["candidate_mean_pct"] is not None
                    and baseline["candidate_mean_pct"] is not None
                    else None
                )
                checks = {
                    "entries": metrics["entries"] >= MIN_FIT_ENTRIES,
                    "resolution": (
                        metrics["resolution_or_cash_rate"] >= MIN_RESOLUTION
                    ),
                    "gain_vs_first": (
                        gain is not None and gain >= MIN_FIT_GAIN
                    ),
                    "trade_severe": (
                        metrics["trade_severe_loss_rate_le_minus2"] is not None
                        and metrics["trade_severe_loss_rate_le_minus2"]
                        <= MAX_FIT_SEVERE
                    ),
                    "positive_days": (
                        metrics["positive_candidate_mean_days"]
                        >= MIN_FIT_POSITIVE_DAYS
                    ),
                }
                row = {
                    "stop_loss_pct": float(stop_loss),
                    "trail_pct": float(trail),
                    "max_hold_minutes": int(max_hold),
                    "metrics": metrics,
                    "mean_gain_vs_first_exit_pct": gain,
                    "checks": checks,
                    "passes": bool(all(checks.values())),
                }
                results.append(row)
                if row["passes"]:
                    passing.append(row)
    if not passing:
        return None, results, baseline
    passing.sort(
        key=lambda row: (
            -(row["mean_gain_vs_first_exit_pct"] or -999.0),
            row["metrics"]["trade_severe_loss_rate_le_minus2"],
            row["max_hold_minutes"],
            row["trail_pct"],
        )
    )
    return passing[0], results, baseline


def subset_days(frame: pd.DataFrame, days: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[
        frame.trading_day.astype(str).isin(days)
    ].copy()


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    causal_scan = causal_execution_scan(scan)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, causal_scan)
    scored, train_threshold, test_threshold = score_and_select_candidates(
        first_hot
    )

    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(CONTROLLER_TRAIN_DAYS)
        & scored.candidate_probability.ge(train_threshold)
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_probability.ge(test_threshold)
    ].copy()

    train_episodes = build_pullback_episodes(
        train_selected,
        causal_scan,
    )
    test_episodes = build_pullback_episodes(
        test_selected,
        causal_scan,
    )
    fit_episodes = subset_days(train_episodes, RULE_FIT_DAYS)
    check_episodes = subset_days(train_episodes, (RULE_CHECK_DAY,))

    chosen, grid, fit_baseline = evaluate_rule_grid(fit_episodes)
    if chosen is None:
        result = {
            "request_id": REQUEST_ID,
            "development_only": True,
            "opens_new_dates": False,
            "promotion_eligible": False,
            "causal_reference_contract": True,
            "train_candidate_threshold_top20": train_threshold,
            "test_candidate_threshold_top5": test_threshold,
            "fit_rule_selected": False,
            "fit_baseline": fit_baseline,
            "fit_grid": grid,
            "economic_gate_pass": False,
            "next_boundary": (
                "joint candidate/pullback admission redesign"
            ),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0

    params = {
        "stop_loss_pct": float(chosen["stop_loss_pct"]),
        "trail_pct": float(chosen["trail_pct"]),
        "max_hold_minutes": int(chosen["max_hold_minutes"]),
    }

    check_rows = apply_exit_rule(
        check_episodes,
        **params,
        policy_name="chosen_rule_check",
    )
    check_baseline_rows = first_exit_baseline(check_episodes)
    check_metrics = policy_metrics(check_rows)
    check_baseline = policy_metrics(check_baseline_rows)
    check_gain = (
        check_metrics["candidate_mean_pct"]
        - check_baseline["candidate_mean_pct"]
    )
    check_gate = {
        "resolution": (
            check_metrics["resolution_or_cash_rate"] >= MIN_RESOLUTION
        ),
        "mean_gain_nonnegative": check_gain >= 0,
        "trade_severe": (
            check_metrics["trade_severe_loss_rate_le_minus2"] is not None
            and check_metrics["trade_severe_loss_rate_le_minus2"]
            <= MAX_CHECK_SEVERE
        ),
    }

    test_rows = apply_exit_rule(
        test_episodes,
        **params,
        policy_name="chosen_rule_test",
    )
    test_baseline_rows = first_exit_baseline(test_episodes)
    test_metrics = policy_metrics(test_rows)
    test_baseline = policy_metrics(test_baseline_rows)
    test_gain = (
        test_metrics["candidate_mean_pct"]
        - test_baseline["candidate_mean_pct"]
    )
    test_checks = {
        "check_day_gate": bool(all(check_gate.values())),
        "entries": test_metrics["entries"] >= MIN_TEST_ENTRIES,
        "resolution": (
            test_metrics["resolution_or_cash_rate"] >= MIN_RESOLUTION
        ),
        "candidate_mean_positive": (
            test_metrics["candidate_mean_pct"] is not None
            and test_metrics["candidate_mean_pct"] > 0
        ),
        "day_balanced_mean_positive": (
            test_metrics["day_balanced_candidate_mean_pct"] is not None
            and test_metrics["day_balanced_candidate_mean_pct"] > 0
        ),
        "positive_days": (
            test_metrics["positive_candidate_mean_days"]
            >= MIN_TEST_POSITIVE_DAYS
        ),
        "trade_positive_rate": (
            test_metrics["trade_positive_rate"] is not None
            and test_metrics["trade_positive_rate"]
            >= MIN_TEST_TRADE_POSITIVE_RATE
        ),
        "trade_severe": (
            test_metrics["trade_severe_loss_rate_le_minus2"] is not None
            and test_metrics["trade_severe_loss_rate_le_minus2"]
            <= MAX_TEST_SEVERE
        ),
        "gain_vs_first_exit": test_gain >= MIN_TEST_GAIN,
    }

    all_rows = [
        check_rows,
        check_baseline_rows,
        test_rows,
        test_baseline_rows,
    ]
    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "causal_reference_contract": True,
        "train_candidate_threshold_top20": train_threshold,
        "test_candidate_threshold_top5": test_threshold,
        "entry_rule": "pullback_2",
        "support": {
            "train_candidate_episodes": int(len(train_selected)),
            "test_candidate_episodes": int(len(test_selected)),
            "fit_candidate_episodes": int(len(fit_episodes)),
            "check_candidate_episodes": int(len(check_episodes)),
            "test_pullback_entries": int(test_episodes.entered.sum()),
        },
        "chosen_rule": params,
        "fit_selected_result": chosen,
        "fit_baseline": fit_baseline,
        "fit_grid": grid,
        "check_day": {
            "policy": check_metrics,
            "baseline": check_baseline,
            "mean_gain_vs_first_exit_pct": check_gain,
            "checks": check_gate,
            "pass": bool(all(check_gate.values())),
        },
        "development_test": {
            "policy": test_metrics,
            "baseline": test_baseline,
            "mean_gain_vs_first_exit_pct": test_gain,
            "checks": test_checks,
        },
        "economic_gate_pass": bool(all(test_checks.values())),
        "interpretation": (
            "Entry is the frozen causal 2% pullback rule. Exit-rule parameters "
            "are chosen on May5-7 only, sanity-checked on May8, then applied "
            "unchanged to top-5% May11-20 candidates. Cash is zero return."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.concat(all_rows, ignore_index=True).to_parquet(
        rows_output,
        index=False,
    )
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
