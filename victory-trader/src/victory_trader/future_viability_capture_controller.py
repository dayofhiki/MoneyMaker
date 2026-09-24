"""Request 192: future-viability + capture recurrent controller.

Composes Request 191's absolute future cost-cover observability with Request
189's relative capture timing into an honest one-minute-at-a-time HOLD/EXIT
policy. All decisions use causal POSITION state only; exact next executable
opens remain outcome references.

This request also reports a transparent KRW 1,000,000 reference-account ledger
for intuitive interpretation. The June 23-29 block is development-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .attention_replay import MINUTE_MS
from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .future_cost_cover_state_observability import (
    score_states,
    train_models,
)
from .market_regime_position_value import (
    attach_market_regime,
    build_market_regime,
)
from .oracle_capture_percentile_stopping import (
    BOOTSTRAP_SEED,
    MIN_COMPLETION_COVERAGE,
    MIN_START_COVERAGE,
    _complete_exit,
    _episode_rows,
    _first_later_exit,
    build_trajectories,
    matched_difference,
    predict_capture,
    summarize,
    train_capture_model,
    add_capture_target,
)
from .selected_hot_position_value_observability import _base_return

REQUEST_ID = 192
CAPTURE_THRESHOLD = 0.70
MAX_HOLD_MINUTES = 30
SEVERE_LOSS_PCT = -5.0

REFERENCE_START_KRW = 1_000_000.0
REFERENCE_MAX_POSITIONS = 5
REFERENCE_POSITION_FRACTION = 0.20

MIN_POSITIVE_DAYS = 4


def _marked_base_return(row: pd.Series) -> float:
    entry_open = pd.to_numeric(
        pd.Series([row.get("entry_open")]),
        errors="coerce",
    ).iloc[0]
    log_current = pd.to_numeric(
        pd.Series([row.get("log_current_close")]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(entry_open) or pd.isna(log_current):
        return np.nan
    current_close = float(np.exp(float(log_current)))
    return float(_base_return(float(entry_open), current_close))


def add_causal_mark(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["marked_base_return_pct"] = result.apply(
        _marked_base_return,
        axis=1,
    )
    return result


def _decision(row: pd.Series, probability_gate: float) -> tuple[bool, str]:
    mark = pd.to_numeric(
        pd.Series([row.get("marked_base_return_pct")]),
        errors="coerce",
    ).iloc[0]
    probability = pd.to_numeric(
        pd.Series([row.get("future_cost_cover_probability")]),
        errors="coerce",
    ).iloc[0]
    predicted_best = pd.to_numeric(
        pd.Series([row.get("predicted_best_future_base_return_pct")]),
        errors="coerce",
    ).iloc[0]
    capture = pd.to_numeric(
        pd.Series([row.get("predicted_capture_percentile")]),
        errors="coerce",
    ).iloc[0]

    values = [mark, probability, predicted_best, capture]
    if any(pd.isna(value) or not np.isfinite(float(value)) for value in values):
        return True, "unscoreable_exit"

    mark = float(mark)
    probability = float(probability)
    predicted_best = float(predicted_best)
    capture = float(capture)

    high_viability = probability >= float(probability_gate)

    if mark > 0:
        if capture >= CAPTURE_THRESHOLD:
            return True, "positive_capture_exit"
        if not high_viability:
            return True, "positive_viability_faded_exit"
        if predicted_best <= mark:
            return True, "positive_no_improvement_exit"
        return False, "positive_viable_hold"

    if high_viability and predicted_best > 0:
        return False, "negative_recovery_hold"
    return True, "negative_recovery_failed_exit"


def build_composed_trajectories(
    scored: pd.DataFrame,
    *,
    probability_gate: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    total = int(
        scored.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    records: list[dict[str, object]] = []

    for _, group in scored.groupby(EPISODE_KEYS, sort=False):
        by_minute = _episode_rows(group)
        first = by_minute.get(1)
        if first is None:
            continue

        minute = 1
        realized = np.nan
        exit_minute = np.nan
        status = "unresolved"
        reason = "unknown"
        holds = 0

        while minute < MAX_HOLD_MINUTES:
            row = by_minute.get(minute)
            if row is None:
                later = _first_later_exit(by_minute, minute)
                if later is not None:
                    later_minute, value = later
                    status = "completed"
                    reason = "missing_state_delayed_open_exit"
                    realized = value
                    exit_minute = later_minute
                else:
                    status = "unresolved"
                    reason = "missing_reached_state"
                break

            current_exit = pd.to_numeric(
                pd.Series([row.get("exit_now_base_return_pct")]),
                errors="coerce",
            ).iloc[0]

            should_exit, decision_reason = _decision(
                row,
                probability_gate,
            )
            if should_exit:
                (
                    status,
                    reason,
                    realized,
                    exit_minute,
                ) = _complete_exit(
                    by_minute,
                    minute,
                    current_exit,
                    decision_reason,
                )
                break

            holds += 1
            if minute == MAX_HOLD_MINUTES - 1:
                next_return = pd.to_numeric(
                    pd.Series(
                        [row.get("next_minute_base_return_pct")]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(next_return):
                    status = "completed"
                    reason = "forced_30m_cap"
                    realized = float(next_return)
                    exit_minute = float(MAX_HOLD_MINUTES)
                else:
                    status = "unresolved"
                    reason = "missing_forced_cap_exit"
                break

            next_row = by_minute.get(minute + 1)
            if next_row is None:
                next_return = pd.to_numeric(
                    pd.Series(
                        [row.get("next_minute_base_return_pct")]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(next_return):
                    status = "completed"
                    reason = "missing_state_next_minute_exit"
                    realized = float(next_return)
                    exit_minute = float(minute + 1)
                else:
                    later = _first_later_exit(by_minute, minute)
                    if later is not None:
                        later_minute, value = later
                        status = "completed"
                        reason = "missing_state_delayed_open_exit"
                        realized = value
                        exit_minute = later_minute
                    else:
                        status = "unresolved"
                        reason = "missing_state_and_exit"
                break

            minute += 1

        records.append(
            {
                "trading_day": str(first["trading_day"]),
                "ticker": str(first["ticker"]).upper(),
                "hot_t": int(first["hot_t"]),
                "policy": "future_viability_capture",
                "status": status,
                "exit_reason": reason,
                "base_net_return_pct": (
                    float(realized)
                    if pd.notna(realized)
                    else np.nan
                ),
                "minutes_held": exit_minute,
                "hold_decisions": int(holds),
            }
        )

    trajectories = pd.DataFrame(records)
    started = int(len(trajectories))
    return trajectories, {
        "total_position_episodes": total,
        "started_episodes": started,
        "start_state_coverage": (
            float(started / total)
            if total
            else None
        ),
    }


def _positive_day_count(
    trajectories: pd.DataFrame,
) -> tuple[int, dict[str, float | None]]:
    completed = trajectories.loc[
        trajectories["status"].eq("completed")
    ].copy()
    completed["base_net_return_pct"] = pd.to_numeric(
        completed["base_net_return_pct"],
        errors="coerce",
    )
    daily = (
        completed.dropna(
            subset=["base_net_return_pct"]
        )
        .groupby(
            completed["trading_day"].astype(str),
            sort=True,
        )["base_net_return_pct"]
        .mean()
    )
    by_day: dict[str, float | None] = {}
    positive = 0
    for day in FRESH_DAYS:
        value = (
            float(daily.loc[day])
            if day in daily.index
            else None
        )
        by_day[day] = value
        if value is not None and value > 0:
            positive += 1
    return positive, by_day


def simulate_reference_account(
    trajectories: pd.DataFrame,
) -> dict[str, object]:
    completed = trajectories.loc[
        trajectories["status"].eq("completed")
    ].copy()
    completed["base_net_return_pct"] = pd.to_numeric(
        completed["base_net_return_pct"],
        errors="coerce",
    )
    completed["minutes_held"] = pd.to_numeric(
        completed["minutes_held"],
        errors="coerce",
    )
    completed["hot_t"] = pd.to_numeric(
        completed["hot_t"],
        errors="coerce",
    )
    completed = completed.dropna(
        subset=[
            "base_net_return_pct",
            "minutes_held",
            "hot_t",
        ]
    ).copy()

    started = int(len(trajectories))
    coverage = (
        float(len(completed) / started)
        if started
        else None
    )

    events: list[tuple[int, int, str, dict[str, object]]] = []
    for row in completed.to_dict("records"):
        entry_t = int(row["hot_t"])
        exit_t = entry_t + int(
            round(float(row["minutes_held"]) * MINUTE_MS)
        )
        key = (
            str(row["trading_day"]),
            str(row["ticker"]),
            int(row["hot_t"]),
        )
        payload = {
            "key": key,
            "return_pct": float(row["base_net_return_pct"]),
        }
        events.append((entry_t, 1, "entry", payload))
        events.append((exit_t, 0, "exit", payload))

    events.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[3]["key"],
        )
    )

    cash = float(REFERENCE_START_KRW)
    open_positions: dict[tuple[str, str, int], float] = {}
    admitted: set[tuple[str, str, int]] = set()
    skipped_capacity = 0
    skipped_cash = 0
    max_concurrent_seen = 0
    min_cash = cash

    for _, _, kind, payload in events:
        key = payload["key"]
        if kind == "exit":
            if key not in admitted:
                continue
            principal = open_positions.pop(key)
            cash += principal * (
                1.0 + float(payload["return_pct"]) / 100.0
            )
            continue

        if len(open_positions) >= REFERENCE_MAX_POSITIONS:
            skipped_capacity += 1
            continue

        nominal_equity = cash + float(
            sum(open_positions.values())
        )
        target = (
            nominal_equity
            * REFERENCE_POSITION_FRACTION
        )
        allocation = min(cash, target)
        if allocation <= 0:
            skipped_cash += 1
            continue

        cash -= allocation
        open_positions[key] = allocation
        admitted.add(key)
        max_concurrent_seen = max(
            max_concurrent_seen,
            len(open_positions),
        )
        min_cash = min(min_cash, cash)

    if open_positions:
        raise ValueError(
            "request 192 reference ledger ended with open completed positions"
        )

    return {
        "starting_balance_krw": REFERENCE_START_KRW,
        "ending_balance_krw": float(cash),
        "net_profit_krw": float(
            cash - REFERENCE_START_KRW
        ),
        "total_return_pct": float(
            (cash / REFERENCE_START_KRW - 1.0) * 100.0
        ),
        "max_concurrent_positions": (
            REFERENCE_MAX_POSITIONS
        ),
        "position_fraction": (
            REFERENCE_POSITION_FRACTION
        ),
        "completed_episode_coverage": coverage,
        "completed_episodes_available": int(len(completed)),
        "admitted_trades": int(len(admitted)),
        "skipped_capacity": int(skipped_capacity),
        "skipped_cash": int(skipped_cash),
        "max_concurrent_seen": int(max_concurrent_seen),
        "minimum_cash_krw": float(min_cash),
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    trajectories_output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    history_scan = pd.read_parquet(history_scan_path)

    position_paths = sorted(
        fresh_dir.glob("*-positions.parquet")
    )
    scan_paths = sorted(
        fresh_dir.glob("*-scan.parquet")
    )
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError(
            "request 192 requires complete request 178 shards"
        )

    fresh = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    found = sorted(
        fresh["trading_day"].astype(str).unique()
    )
    if found != list(FRESH_DAYS):
        raise ValueError(
            f"request 192 expected {FRESH_DAYS}, found {found}"
        )

    historical_days = set(
        fit["trading_day"].astype(str)
    ) | set(
        calibration["trading_day"].astype(str)
    )
    hist_scan = history_scan.loc[
        history_scan["trading_day"].astype(str).isin(
            historical_days
        )
    ].copy()
    hist_regime = build_market_regime(hist_scan)
    fresh_regime = build_market_regime(fresh_scan)

    fit = attach_market_regime(fit, hist_regime)
    calibration = attach_market_regime(
        calibration,
        hist_regime,
    )
    fresh = attach_market_regime(
        fresh,
        fresh_regime,
    )

    future_models = train_models(fit, calibration)
    scored = score_states(fresh, future_models)

    capture_fit = add_capture_target(fit)
    capture_model = train_capture_model(capture_fit)
    scored["predicted_capture_percentile"] = (
        predict_capture(scored, capture_model)
    )
    scored = add_causal_mark(scored)

    candidate, candidate_coverage = (
        build_composed_trajectories(
            scored,
            probability_gate=future_models.probability_gate,
        )
    )
    minute1, minute1_coverage = build_trajectories(
        scored,
        policy="minute1_exit",
    )
    hold30, hold30_coverage = build_trajectories(
        scored,
        policy="hold30",
    )
    capture_only, capture_coverage = build_trajectories(
        scored,
        policy="capture_percentile",
        threshold=CAPTURE_THRESHOLD,
    )

    candidate_summary = summarize(
        candidate,
        seed=BOOTSTRAP_SEED + 200,
    )
    minute1_summary = summarize(
        minute1,
        seed=BOOTSTRAP_SEED + 201,
    )
    hold30_summary = summarize(
        hold30,
        seed=BOOTSTRAP_SEED + 202,
    )
    capture_summary = summarize(
        capture_only,
        seed=BOOTSTRAP_SEED + 203,
    )

    diff_minute1 = matched_difference(
        candidate,
        minute1,
        seed=BOOTSTRAP_SEED + 204,
    )
    diff_capture = matched_difference(
        candidate,
        capture_only,
        seed=BOOTSTRAP_SEED + 205,
    )

    positive_days, candidate_by_day = (
        _positive_day_count(candidate)
    )

    accounts = {
        "candidate": simulate_reference_account(candidate),
        "minute1": simulate_reference_account(minute1),
        "hold30": simulate_reference_account(hold30),
        "capture_only": simulate_reference_account(
            capture_only
        ),
    }

    c_day = candidate_summary[
        "day_balanced_base_mean_pct"
    ]
    c_low = candidate_summary[
        "day_bootstrap"
    ]["ci_low_pct"]
    c_severe = candidate_summary["severe_loss_rate"]
    b_severe = minute1_summary["severe_loss_rate"]
    d_minute = diff_minute1[
        "day_balanced_difference_pct"
    ]
    d_minute_low = diff_minute1[
        "bootstrap"
    ]["ci_low_pct"]
    d_capture = diff_capture[
        "day_balanced_difference_pct"
    ]
    account_end = accounts[
        "candidate"
    ]["ending_balance_krw"]

    gate = bool(
        candidate_coverage["start_state_coverage"]
        is not None
        and float(
            candidate_coverage["start_state_coverage"]
        )
        >= MIN_START_COVERAGE
        and candidate_summary["completion_coverage"]
        is not None
        and float(
            candidate_summary["completion_coverage"]
        )
        >= MIN_COMPLETION_COVERAGE
        and c_day is not None
        and float(c_day) > 0
        and c_low is not None
        and float(c_low) > 0
        and positive_days >= MIN_POSITIVE_DAYS
        and c_severe is not None
        and b_severe is not None
        and float(c_severe) <= float(b_severe)
        and d_minute is not None
        and float(d_minute) > 0
        and d_minute_low is not None
        and float(d_minute_low) > 0
        and d_capture is not None
        and float(d_capture) > 0
        and account_end is not None
        and float(account_end) > REFERENCE_START_KRW
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "policy_executed": True,
        "capture_threshold": CAPTURE_THRESHOLD,
        "future_probability_gate": (
            future_models.probability_gate
        ),
        "candidate_coverage": candidate_coverage,
        "minute1_coverage": minute1_coverage,
        "hold30_coverage": hold30_coverage,
        "capture_only_coverage": capture_coverage,
        "candidate": candidate_summary,
        "minute1_comparator": minute1_summary,
        "hold30_comparator": hold30_summary,
        "capture_only_comparator": capture_summary,
        "matched_candidate_minus_minute1": (
            diff_minute1
        ),
        "matched_candidate_minus_capture_only": (
            diff_capture
        ),
        "candidate_positive_return_days": int(
            positive_days
        ),
        "candidate_by_day_mean_pct": candidate_by_day,
        "reference_accounts": accounts,
        "frozen_gate": {
            "min_start_coverage": MIN_START_COVERAGE,
            "min_completion_coverage": (
                MIN_COMPLETION_COVERAGE
            ),
            "candidate_day_balanced_base_positive": True,
            "candidate_bootstrap_low_positive": True,
            "min_positive_return_days": MIN_POSITIVE_DAYS,
            "severe_loss_no_worse_than_minute1": True,
            "matched_minute1_day_balanced_positive": True,
            "matched_minute1_bootstrap_low_positive": True,
            "matched_capture_day_balanced_positive": True,
            "reference_account_must_finish_above_start": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    trajectories_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pd.concat(
        [
            candidate,
            minute1,
            hold30,
            capture_only,
        ],
        ignore_index=True,
    ).to_parquet(
        trajectories_output_path,
        index=False,
        compression="zstd",
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--history-scan",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fresh-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--trajectories-output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.trajectories_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
