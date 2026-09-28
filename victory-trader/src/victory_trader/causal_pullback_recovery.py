"""Request276: causal pullback-recovery entry under a frozen exit policy.

Request275 showed useful ranking information at the raw 2% pullback state but
could not identify a positive subset. This experiment changes only the entry
trigger: after a 2% drawdown from the running post-HOT high, require an
observable recovery from the running pullback low before entering.

The candidate population and downstream exit policy are frozen. May5-7 select
one recovery threshold, May8 is a separate check, and May11-20 is touched only
if both earlier stages pass. June15-19 remains unopened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .attention_replay import MINUTE_MS
from .causal_minute_controller_rebuild import causal_execution_scan
from .causal_pullback_entry_audit import ENTRY_WINDOW_MINUTES, RISK_CAP_MINUTES
from .full_hot_fixed_policy_value import attach_fixed_value, event_lookup
from .pullback_trailing_exit import (
    apply_exit_rule,
    build_pullback_episodes,
    policy_metrics,
)
from .rich_post_hot_state import score_and_select_candidates

REQUEST_ID = 276
FIT_DAYS = ("2026-05-05", "2026-05-06", "2026-05-07")
CAL_DAY = "2026-05-08"
TEST_DAYS = (
    "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14",
    "2026-05-15", "2026-05-18", "2026-05-19", "2026-05-20",
)
RECOVERY_THRESHOLDS_PCT = (0.5, 1.0, 1.5)

FIXED_STOP_LOSS_PCT = -2.5
FIXED_TRAIL_PCT = 2.0
FIXED_MAX_HOLD_MINUTES = 10

MIN_FIT_ENTRIES = 15
MIN_FIT_RESOLUTION = 0.95
MIN_FIT_GAIN_VS_RAW = 0.10
MIN_FIT_POSITIVE_DAYS = 2
MAX_FIT_SEVERE = 0.20

MIN_CAL_ENTRIES = 5
MIN_CAL_RESOLUTION = 0.95
MAX_CAL_SEVERE = 0.25

MIN_TEST_ENTRIES = 20
MIN_TEST_RESOLUTION = 0.95
MIN_TEST_POSITIVE_DAYS = 5
MIN_TEST_TRADE_POSITIVE_RATE = 0.40
MAX_TEST_SEVERE = 0.15
MIN_TEST_GAIN_VS_RAW = 0.10


def pick_recovery_entry(
    events: list[tuple[int, float]],
    recovery_pct: float,
) -> tuple[int, float] | None:
    """Enter only after a 2% drawdown has visibly recovered."""
    if len(events) < 2:
        return None

    running_high = float(events[0][1])
    activated = False
    pullback_low = float(events[0][1])

    for t, raw_price in events[1:]:
        price = float(raw_price)
        running_high = max(running_high, price)
        drawdown_pct = (
            (price / running_high - 1.0) * 100.0
            if running_high > 0
            else 0.0
        )

        if not activated:
            if drawdown_pct <= -2.0:
                activated = True
                pullback_low = price
            continue

        pullback_low = min(pullback_low, price)
        recovery_from_low_pct = (
            (price / pullback_low - 1.0) * 100.0
            if pullback_low > 0
            else 0.0
        )
        if recovery_from_low_pct >= float(recovery_pct):
            return int(t), price

    return None


def build_recovery_episodes(
    selected: pd.DataFrame,
    scan: pd.DataFrame,
    *,
    recovery_pct: float,
) -> pd.DataFrame:
    lookup = event_lookup(scan)
    rows: list[dict[str, object]] = []

    for raw in selected.itertuples(index=False):
        day = str(raw.trading_day)
        ticker = str(raw.ticker).upper()
        hot_t = int(raw.t)
        entry_limit = hot_t + ENTRY_WINDOW_MINUTES * MINUTE_MS
        risk_cap = hot_t + RISK_CAP_MINUTES * MINUTE_MS
        episode = lookup.get((day, ticker), [])
        entry_events = [
            (int(t), float(price))
            for t, price in episode
            if hot_t < int(t) <= entry_limit and float(price) > 0
        ]
        entry = pick_recovery_entry(entry_events, recovery_pct)
        if entry is None:
            rows.append({
                "trading_day": day,
                "ticker": ticker,
                "hot_t": hot_t,
                "entered": False,
                "entry_t": None,
                "entry_price": None,
                "path": [],
            })
            continue

        entry_t, entry_price = entry
        path = [
            (int(t), float(price))
            for t, price in episode
            if int(entry_t) < int(t) <= risk_cap and float(price) > 0
        ]
        rows.append({
            "trading_day": day,
            "ticker": ticker,
            "hot_t": hot_t,
            "entered": True,
            "entry_t": int(entry_t),
            "entry_price": float(entry_price),
            "path": path,
        })

    return pd.DataFrame(rows)


def apply_fixed_exit(
    episodes: pd.DataFrame,
    policy_name: str,
) -> pd.DataFrame:
    return apply_exit_rule(
        episodes,
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name=policy_name,
    )


def subset_days(
    frame: pd.DataFrame,
    days: tuple[str, ...],
) -> pd.DataFrame:
    return frame.loc[
        frame.trading_day.astype(str).isin(days)
    ].copy()


def gain_vs(candidate: dict, baseline: dict) -> float | None:
    left = candidate.get("candidate_mean_pct")
    right = baseline.get("candidate_mean_pct")
    if left is None or right is None:
        return None
    return float(left - right)


def fit_checks(metrics: dict, gain: float | None) -> dict:
    return {
        "min_entries": metrics["entries"] >= MIN_FIT_ENTRIES,
        "resolution": (
            metrics["resolution_or_cash_rate"] >= MIN_FIT_RESOLUTION
        ),
        "candidate_mean_positive": (
            metrics["candidate_mean_pct"] is not None
            and metrics["candidate_mean_pct"] > 0
        ),
        "day_balanced_mean_positive": (
            metrics["day_balanced_candidate_mean_pct"] is not None
            and metrics["day_balanced_candidate_mean_pct"] > 0
        ),
        "trade_mean_positive": (
            metrics["trade_mean_pct"] is not None
            and metrics["trade_mean_pct"] > 0
        ),
        "positive_days": (
            metrics["positive_candidate_mean_days"]
            >= MIN_FIT_POSITIVE_DAYS
        ),
        "severe_loss": (
            metrics["trade_severe_loss_rate_le_minus2"] is not None
            and metrics["trade_severe_loss_rate_le_minus2"]
            <= MAX_FIT_SEVERE
        ),
        "gain_vs_raw_pullback": (
            gain is not None and gain >= MIN_FIT_GAIN_VS_RAW
        ),
    }


def cal_checks(metrics: dict, gain: float | None) -> dict:
    return {
        "min_entries": metrics["entries"] >= MIN_CAL_ENTRIES,
        "resolution": (
            metrics["resolution_or_cash_rate"] >= MIN_CAL_RESOLUTION
        ),
        "candidate_mean_nonnegative": (
            metrics["candidate_mean_pct"] is not None
            and metrics["candidate_mean_pct"] >= 0
        ),
        "trade_mean_nonnegative": (
            metrics["trade_mean_pct"] is not None
            and metrics["trade_mean_pct"] >= 0
        ),
        "severe_loss": (
            metrics["trade_severe_loss_rate_le_minus2"] is not None
            and metrics["trade_severe_loss_rate_le_minus2"]
            <= MAX_CAL_SEVERE
        ),
        "gain_vs_raw_pullback_positive": (
            gain is not None and gain > 0
        ),
    }


def development_checks(metrics: dict, gain: float | None) -> dict:
    return {
        "min_entries": metrics["entries"] >= MIN_TEST_ENTRIES,
        "resolution": (
            metrics["resolution_or_cash_rate"] >= MIN_TEST_RESOLUTION
        ),
        "candidate_mean_positive": (
            metrics["candidate_mean_pct"] is not None
            and metrics["candidate_mean_pct"] > 0
        ),
        "day_balanced_mean_positive": (
            metrics["day_balanced_candidate_mean_pct"] is not None
            and metrics["day_balanced_candidate_mean_pct"] > 0
        ),
        "positive_days": (
            metrics["positive_candidate_mean_days"]
            >= MIN_TEST_POSITIVE_DAYS
        ),
        "trade_positive_rate": (
            metrics["trade_positive_rate"] is not None
            and metrics["trade_positive_rate"]
            >= MIN_TEST_TRADE_POSITIVE_RATE
        ),
        "severe_loss": (
            metrics["trade_severe_loss_rate_le_minus2"] is not None
            and metrics["trade_severe_loss_rate_le_minus2"]
            <= MAX_TEST_SEVERE
        ),
        "gain_vs_raw_pullback": (
            gain is not None and gain >= MIN_TEST_GAIN_VS_RAW
        ),
    }


def evaluate(
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    scan = causal_execution_scan(pd.read_parquet(scan_path))
    first_hot = pd.read_parquet(first_hot_path).drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, scan)
    scored, top20_threshold, _ = score_and_select_candidates(first_hot)

    used_days = (*FIT_DAYS, CAL_DAY, *TEST_DAYS)
    selected = scored.loc[
        scored.trading_day.astype(str).isin(used_days)
        & scored.candidate_probability.ge(top20_threshold)
    ].copy()

    raw_rows = apply_fixed_exit(
        build_pullback_episodes(selected, scan),
        "raw_pullback_2",
    )
    policy_rows: dict[float, pd.DataFrame] = {}
    for recovery in RECOVERY_THRESHOLDS_PCT:
        policy_rows[float(recovery)] = apply_fixed_exit(
            build_recovery_episodes(
                selected,
                scan,
                recovery_pct=float(recovery),
            ),
            f"pullback2_recover{recovery:g}",
        )

    raw_fit = policy_metrics(subset_days(raw_rows, FIT_DAYS))
    fit_grid = []
    passing = []
    for recovery in RECOVERY_THRESHOLDS_PCT:
        metrics = policy_metrics(
            subset_days(
                policy_rows[float(recovery)],
                FIT_DAYS,
            )
        )
        gain = gain_vs(metrics, raw_fit)
        checks = fit_checks(metrics, gain)
        item = {
            "recovery_pct": float(recovery),
            "metrics": metrics,
            "candidate_mean_gain_vs_raw_pct": gain,
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
        fit_grid.append(item)
        if item["passes"]:
            passing.append(item)

    chosen = None
    if passing:
        passing.sort(
            key=lambda item: (
                -(item["metrics"]["candidate_mean_pct"] or -999.0),
                item["metrics"]["trade_severe_loss_rate_le_minus2"],
                item["recovery_pct"],
            )
        )
        chosen = passing[0]

    raw_cal = None
    calibration_metrics = None
    calibration_gain = None
    calibration_checks = None
    calibration_gate_pass = False

    if chosen is not None:
        chosen_rows = policy_rows[float(chosen["recovery_pct"])]
        raw_cal = policy_metrics(subset_days(raw_rows, (CAL_DAY,)))
        calibration_metrics = policy_metrics(
            subset_days(chosen_rows, (CAL_DAY,))
        )
        calibration_gain = gain_vs(calibration_metrics, raw_cal)
        calibration_checks = cal_checks(
            calibration_metrics,
            calibration_gain,
        )
        calibration_gate_pass = bool(
            all(calibration_checks.values())
        )

    raw_test = None
    dev_metrics = None
    dev_gain = None
    dev_checks = None
    dev_gate_pass = False

    if chosen is not None and calibration_gate_pass:
        chosen_rows = policy_rows[float(chosen["recovery_pct"])]
        raw_test = policy_metrics(subset_days(raw_rows, TEST_DAYS))
        dev_metrics = policy_metrics(
            subset_days(chosen_rows, TEST_DAYS)
        )
        dev_gain = gain_vs(dev_metrics, raw_test)
        dev_checks = development_checks(dev_metrics, dev_gain)
        dev_gate_pass = bool(all(dev_checks.values()))

    audit_rows = [
        raw_rows.assign(
            recovery_pct=np.nan,
            trigger_family="raw_pullback_2",
        )
    ]
    for recovery in RECOVERY_THRESHOLDS_PCT:
        audit_rows.append(
            policy_rows[float(recovery)].assign(
                recovery_pct=float(recovery),
                trigger_family="pullback_recovery",
            )
        )
    audit = pd.concat(audit_rows, ignore_index=True)

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "candidate_population": {
            "rule": (
                "frozen top-20% candidate threshold from "
                "score_and_select_candidates"
            ),
            "threshold": float(top20_threshold),
        },
        "entry_contract": (
            "observe >=2% drawdown from running post-HOT high, "
            "then require causal recovery from the running pullback "
            "low before entry"
        ),
        "recovery_thresholds_pct": list(RECOVERY_THRESHOLDS_PCT),
        "fixed_downstream_policy": {
            "hard_stop_loss_pct": FIXED_STOP_LOSS_PCT,
            "trailing_drawdown_pct": FIXED_TRAIL_PCT,
            "max_hold_minutes": FIXED_MAX_HOLD_MINUTES,
            "missing_deadline": "unresolved",
        },
        "fit_days": list(FIT_DAYS),
        "calibration_day": CAL_DAY,
        "development_test_days": list(TEST_DAYS),
        "fit_raw_pullback_baseline": raw_fit,
        "fit_grid": fit_grid,
        "fit_gate_pass": chosen is not None,
        "chosen_recovery": chosen,
        "calibration_raw_pullback_baseline": raw_cal,
        "calibration_metrics": calibration_metrics,
        "calibration_gain_vs_raw_pct": calibration_gain,
        "calibration_checks": calibration_checks,
        "calibration_gate_pass": calibration_gate_pass,
        "development_raw_pullback_baseline": raw_test,
        "development_metrics": dev_metrics,
        "development_gain_vs_raw_pct": dev_gain,
        "development_checks": dev_checks,
        "development_gate_pass": dev_gate_pass,
        "interpretation": (
            "Only open-price states observed before the entry "
            "decision define the recovery trigger. The exit policy "
            "is frozen from Request275. No future-best label, "
            "one-second feature, or new date is used. May11-20 "
            "remains untouched unless fit and May8 both pass."
        ),
        "next_boundary": (
            "learn admission around the confirmed recovery state"
            if dev_gate_pass
            else (
                "redesign candidate state or add strictly lagged "
                "completed-minute context; do not open June15-19"
            )
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    audit.to_parquet(rows_output, index=False)
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
