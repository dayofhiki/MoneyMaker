"""Request 194: asymmetric risk/reward overlay bridge.

Calibration selects one predeclared hard-stop / half-take / trailing policy.
Fresh Request-178 dates are evaluation-only. Triggers are causal completed-state
marks; executions use the next causally executable open reference already
stored in POSITION rows, so threshold fills are never idealized.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .attention_replay import MINUTE_MS
from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .future_viability_capture_controller import (
    REFERENCE_MAX_POSITIONS,
    REFERENCE_POSITION_FRACTION,
    REFERENCE_START_KRW,
    _marked_base_return,
)
from .oracle_capture_percentile_stopping import (
    BOOTSTRAP_SEED,
    _episode_rows,
    build_trajectories,
    matched_difference,
    summarize,
)
from .selected_hot_position_value_observability import (
    _base_return,
    _open_map,
)

REQUEST_ID = 194
MAX_HOLD_MINUTES = 30
PARTIAL_FRACTION = 0.50
STOP_THRESHOLDS = (-3.0, -5.0, -7.0)
TAKE_THRESHOLDS = (5.0, 10.0, 15.0)
TRAIL_GAPS = (3.0, 5.0, 7.0)
MIN_COMPLETION_COVERAGE = 0.90
MIN_POSITIVE_DAYS = 4
CANONICAL = (-5.0, 10.0, 5.0)


@dataclass(frozen=True)
class PolicySpec:
    stop_pct: float | None
    take_pct: float | None
    trail_gap_pct: float | None
    partial_fraction: float = PARTIAL_FRACTION

    @property
    def name(self) -> str:
        stop = "none" if self.stop_pct is None else f"{self.stop_pct:g}"
        take = "none" if self.take_pct is None else f"{self.take_pct:g}"
        trail = (
            "none"
            if self.trail_gap_pct is None
            else f"{self.trail_gap_pct:g}"
        )
        return f"stop_{stop}__take_{take}__trail_{trail}"


def _num(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(
        pd.Series([row.get(column)]),
        errors="coerce",
    ).iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _finalize(
    *,
    partial_return: float,
    partial_fraction: float,
    final_return: float,
) -> float:
    if pd.notna(partial_return):
        return float(
            partial_fraction * partial_return
            + (1.0 - partial_fraction) * final_return
        )
    return float(final_return)


def _trajectory_record(
    first: pd.Series,
    spec: PolicySpec,
    *,
    status: str,
    reason: str,
    final_return: float,
    final_minute: float,
    partial_return: float,
    partial_minute: float,
) -> dict[str, object]:
    weighted = (
        _finalize(
            partial_return=partial_return,
            partial_fraction=spec.partial_fraction,
            final_return=final_return,
        )
        if status == "completed" and pd.notna(final_return)
        else np.nan
    )
    return {
        "trading_day": str(first["trading_day"]),
        "ticker": str(first["ticker"]).upper(),
        "hot_t": int(first["hot_t"]),
        "policy": spec.name,
        "status": status,
        "exit_reason": reason,
        "base_net_return_pct": weighted,
        "minutes_held": (
            float(final_minute)
            if pd.notna(final_minute)
            else np.nan
        ),
        "partial_taken": bool(pd.notna(partial_return)),
        "partial_return_pct": (
            float(partial_return)
            if pd.notna(partial_return)
            else np.nan
        ),
        "partial_minute": (
            float(partial_minute)
            if pd.notna(partial_minute)
            else np.nan
        ),
        "partial_fraction": (
            float(spec.partial_fraction)
            if pd.notna(partial_return)
            else 0.0
        ),
        "final_return_pct": (
            float(final_return)
            if pd.notna(final_return)
            else np.nan
        ),
        "final_fraction": (
            float(1.0 - spec.partial_fraction)
            if pd.notna(partial_return)
            else 1.0
        ),
    }


def build_policy_trajectories(
    positions: pd.DataFrame,
    spec: PolicySpec,
    scan: pd.DataFrame | None = None,
    *,
    opens: dict[tuple[str, str, int], float] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if opens is None:
        opens = (
            _open_map(scan)
            if scan is not None and not scan.empty
            else {}
        )
    total = int(
        positions.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    records: list[dict[str, object]] = []

    for _, group in positions.groupby(EPISODE_KEYS, sort=False):
        by_minute = _episode_rows(group)
        first = by_minute.get(1)
        if first is None:
            continue

        partial_return = np.nan
        partial_minute = np.nan
        post_partial_peak = -np.inf
        status = "unresolved"
        reason = "unknown"
        final_return = np.nan
        final_minute = np.nan

        minute = 1
        while minute < MAX_HOLD_MINUTES:
            row = by_minute.get(minute)
            if row is None:
                # A silent/missing minute is not an EXIT signal. In live
                # operation the controller would simply have no new bar-level
                # evidence at this exact minute, so preserve the position and
                # resume decisions at the next observed causal state.
                minute += 1
                continue

            mark = _marked_base_return(row)
            exit_now = _num(row, "exit_now_base_return_pct")
            if not np.isfinite(mark):
                reason = "unscoreable_mark"
                break

            if (
                spec.stop_pct is not None
                and mark <= float(spec.stop_pct)
            ):
                if pd.isna(exit_now):
                    reason = "hard_stop_missing_open"
                    break
                status = "completed"
                reason = "hard_stop"
                final_return = float(exit_now)
                final_minute = float(minute)
                break

            if pd.notna(partial_return):
                post_partial_peak = max(
                    float(post_partial_peak),
                    float(mark),
                )
                if (
                    spec.trail_gap_pct is not None
                    and mark
                    <= post_partial_peak
                    - float(spec.trail_gap_pct)
                ):
                    if pd.isna(exit_now):
                        reason = "trailing_exit_missing_open"
                        break
                    status = "completed"
                    reason = "trailing_exit"
                    final_return = float(exit_now)
                    final_minute = float(minute)
                    break
            elif (
                spec.take_pct is not None
                and mark >= float(spec.take_pct)
            ):
                if pd.isna(exit_now):
                    reason = "partial_take_missing_open"
                    break
                partial_return = float(exit_now)
                partial_minute = float(minute)
                post_partial_peak = float(mark)

            if minute == MAX_HOLD_MINUTES - 1:
                minute30 = _num(
                    row,
                    "next_minute_base_return_pct",
                )
                if pd.isna(minute30):
                    reason = "missing_forced_cap_exit"
                    break
                status = "completed"
                reason = "forced_30m_cap"
                final_return = float(minute30)
                final_minute = float(MAX_HOLD_MINUTES)
                break

            minute += 1

        if status == "unresolved" and reason == "unknown":
            day = str(first["trading_day"])
            ticker = str(first["ticker"]).upper()
            hot_t = int(first["hot_t"])
            entry_open = _num(first, "entry_open")
            cap_open = opens.get(
                (
                    day,
                    ticker,
                    hot_t
                    + MAX_HOLD_MINUTES * MINUTE_MS,
                ),
                np.nan,
            )
            if (
                np.isfinite(entry_open)
                and entry_open > 0
                and pd.notna(cap_open)
                and np.isfinite(float(cap_open))
                and float(cap_open) > 0
            ):
                status = "completed"
                reason = "forced_30m_cap_exact_scan"
                final_return = float(
                    _base_return(
                        float(entry_open),
                        float(cap_open),
                    )
                )
                final_minute = float(MAX_HOLD_MINUTES)
            else:
                reason = "missing_forced_cap_execution"

        records.append(
            _trajectory_record(
                first,
                spec,
                status=status,
                reason=reason,
                final_return=final_return,
                final_minute=final_minute,
                partial_return=partial_return,
                partial_minute=partial_minute,
            )
        )

    trajectories = pd.DataFrame(records)
    started = int(len(trajectories))
    return trajectories, {
        "total_position_episodes": total,
        "started_episodes": started,
        "start_state_coverage": (
            float(started / total) if total else None
        ),
    }


def summarize_enhanced(
    trajectories: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, object]:
    result = summarize(trajectories, seed=seed)
    completed = trajectories.loc[
        trajectories["status"].eq("completed")
    ].copy()
    values = pd.to_numeric(
        completed["base_net_return_pct"],
        errors="coerce",
    ).dropna()
    wins = values.loc[values.gt(0)]
    losses = values.loc[values.lt(0)]
    mean_win = float(wins.mean()) if len(wins) else None
    mean_loss = float(losses.mean()) if len(losses) else None
    payoff = (
        float(mean_win / abs(mean_loss))
        if mean_win is not None
        and mean_loss is not None
        and mean_loss < 0
        else None
    )
    result.update(
        {
            "winning_trades": int(len(wins)),
            "losing_trades": int(len(losses)),
            "zero_trades": int(values.eq(0).sum()),
            "mean_winning_return_pct": mean_win,
            "mean_losing_return_pct": mean_loss,
            "payoff_ratio": payoff,
            "partial_take_rate": (
                float(
                    completed["partial_taken"]
                    .fillna(False)
                    .astype(bool)
                    .mean()
                )
                if len(completed)
                and "partial_taken" in completed
                else 0.0
            ),
            "hard_stop_rate": (
                float(
                    completed["exit_reason"]
                    .eq("hard_stop")
                    .mean()
                )
                if len(completed)
                else None
            ),
            "trailing_exit_rate": (
                float(
                    completed["exit_reason"]
                    .eq("trailing_exit")
                    .mean()
                )
                if len(completed)
                else None
            ),
        }
    )
    return result


def choose_policy(
    calibration: pd.DataFrame,
    calibration_scan: pd.DataFrame | None = None,
) -> tuple[PolicySpec, dict[str, object]]:
    table: dict[str, object] = {}
    calibration_opens = (
        _open_map(calibration_scan)
        if calibration_scan is not None
        and not calibration_scan.empty
        else {}
    )
    eligible: list[
        tuple[float, float, float, float, float, float, PolicySpec]
    ] = []

    for index, (stop, take, trail) in enumerate(
        product(
            STOP_THRESHOLDS,
            TAKE_THRESHOLDS,
            TRAIL_GAPS,
        )
    ):
        spec = PolicySpec(stop, take, trail)
        trajectory, coverage = build_policy_trajectories(
            calibration,
            spec,
            opens=calibration_opens,
        )
        summary = summarize_enhanced(
            trajectory,
            seed=BOOTSTRAP_SEED + 300 + index,
        )
        table[spec.name] = {
            "spec": {
                "stop_pct": stop,
                "take_pct": take,
                "trail_gap_pct": trail,
                "partial_fraction": PARTIAL_FRACTION,
            },
            "coverage": coverage,
            "summary": summary,
        }
        completion = summary["completion_coverage"]
        day_mean = summary["day_balanced_base_mean_pct"]
        severe = summary["severe_loss_rate"]
        mean_hold = summary["mean_hold_minutes"]
        if (
            completion is not None
            and float(completion) >= MIN_COMPLETION_COVERAGE
            and day_mean is not None
            and severe is not None
            and mean_hold is not None
        ):
            eligible.append(
                (
                    -float(day_mean),
                    float(severe),
                    float(mean_hold),
                    float(stop),
                    float(take),
                    float(trail),
                    spec,
                )
            )

    if not eligible:
        raise ValueError(
            "request 194 no calibration policy reaches completion floor"
        )
    eligible.sort(key=lambda item: item[:-1])
    return eligible[0][-1], table


def _positive_days(
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
    for column in [
        "base_net_return_pct",
        "minutes_held",
        "hot_t",
        "partial_return_pct",
        "partial_minute",
        "partial_fraction",
        "final_return_pct",
        "final_fraction",
    ]:
        if column in completed:
            completed[column] = pd.to_numeric(
                completed[column],
                errors="coerce",
            )
    completed = completed.dropna(
        subset=[
            "base_net_return_pct",
            "minutes_held",
            "hot_t",
            "final_return_pct",
        ]
    ).copy()

    events: list[
        tuple[int, int, str, tuple[str, str, int], dict[str, float]]
    ] = []
    returns_by_key: dict[tuple[str, str, int], float] = {}

    for row in completed.to_dict("records"):
        key = (
            str(row["trading_day"]),
            str(row["ticker"]).upper(),
            int(row["hot_t"]),
        )
        returns_by_key[key] = float(row["base_net_return_pct"])
        entry_t = int(row["hot_t"])
        events.append((entry_t, 2, "entry", key, {}))

        if bool(row.get("partial_taken", False)):
            partial_minute = float(row["partial_minute"])
            events.append(
                (
                    entry_t
                    + int(round(partial_minute * MINUTE_MS)),
                    1,
                    "partial",
                    key,
                    {
                        "fraction": float(row["partial_fraction"]),
                        "return_pct": float(row["partial_return_pct"]),
                    },
                )
            )

        events.append(
            (
                entry_t
                + int(
                    round(
                        float(row["minutes_held"])
                        * MINUTE_MS
                    )
                ),
                0,
                "final",
                key,
                {
                    "return_pct": float(row["final_return_pct"]),
                },
            )
        )

    events.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[3],
        )
    )

    cash = float(REFERENCE_START_KRW)
    open_positions: dict[
        tuple[str, str, int],
        dict[str, float],
    ] = {}
    admitted: set[tuple[str, str, int]] = set()
    skipped_capacity = 0
    skipped_cash = 0
    max_concurrent_seen = 0

    peak_balance = float(REFERENCE_START_KRW)
    max_drawdown = 0.0

    def mark_balance() -> None:
        nonlocal peak_balance, max_drawdown
        nominal = cash + float(
            sum(
                item["remaining_principal"]
                for item in open_positions.values()
            )
        )
        peak_balance = max(peak_balance, nominal)
        if peak_balance > 0:
            drawdown = (
                nominal / peak_balance - 1.0
            ) * 100.0
            max_drawdown = min(
                max_drawdown,
                float(drawdown),
            )

    for _, _, kind, key, payload in events:
        if kind == "entry":
            if len(open_positions) >= REFERENCE_MAX_POSITIONS:
                skipped_capacity += 1
                continue
            nominal_equity = cash + float(
                sum(
                    item["remaining_principal"]
                    for item in open_positions.values()
                )
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
            open_positions[key] = {
                "initial_principal": allocation,
                "remaining_principal": allocation,
            }
            admitted.add(key)
            max_concurrent_seen = max(
                max_concurrent_seen,
                len(open_positions),
            )
            mark_balance()
            continue

        if key not in admitted or key not in open_positions:
            continue

        if kind == "partial":
            position = open_positions[key]
            piece = (
                position["initial_principal"]
                * float(payload["fraction"])
            )
            piece = min(
                piece,
                position["remaining_principal"],
            )
            position["remaining_principal"] -= piece
            cash += piece * (
                1.0
                + float(payload["return_pct"])
                / 100.0
            )
            mark_balance()
            continue

        position = open_positions.pop(key)
        principal = position["remaining_principal"]
        cash += principal * (
            1.0
            + float(payload["return_pct"])
            / 100.0
        )
        mark_balance()

    if open_positions:
        raise ValueError(
            "request 194 ledger ended with open completed positions"
        )

    admitted_returns = [
        returns_by_key[key]
        for key in admitted
        if key in returns_by_key
    ]
    wins = sum(value > 0 for value in admitted_returns)
    losses = sum(value < 0 for value in admitted_returns)
    started = int(len(trajectories))

    return {
        "starting_balance_krw": REFERENCE_START_KRW,
        "ending_balance_krw": float(cash),
        "net_profit_krw": float(
            cash - REFERENCE_START_KRW
        ),
        "total_return_pct": float(
            (cash / REFERENCE_START_KRW - 1.0)
            * 100.0
        ),
        "realized_ledger_max_drawdown_pct": float(
            max_drawdown
        ),
        "completed_episode_coverage": (
            float(len(completed) / started)
            if started
            else None
        ),
        "completed_episodes_available": int(
            len(completed)
        ),
        "admitted_trades": int(len(admitted)),
        "winning_trades": int(wins),
        "losing_trades": int(losses),
        "skipped_capacity": int(skipped_capacity),
        "skipped_cash": int(skipped_cash),
        "max_concurrent_seen": int(
            max_concurrent_seen
        ),
    }


def evaluate(
    calibration_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    trajectories_output_path: Path,
) -> int:
    calibration = pd.read_parquet(calibration_path)
    history_scan = pd.read_parquet(history_scan_path)
    calibration_days = set(
        calibration["trading_day"].astype(str)
    )
    calibration_scan = history_scan.loc[
        history_scan["trading_day"]
        .astype(str)
        .isin(calibration_days)
    ].copy()
    selected_spec, calibration_table = choose_policy(
        calibration,
        calibration_scan,
    )

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
            "request 194 requires complete request 178 positions and scans"
        )
    fresh = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    fresh_opens = _open_map(fresh_scan)
    found = sorted(
        fresh["trading_day"].astype(str).unique()
    )
    if found != list(FRESH_DAYS):
        raise ValueError(
            f"request 194 expected {FRESH_DAYS}, found {found}"
        )

    candidate, candidate_coverage = (
        build_policy_trajectories(
            fresh,
            selected_spec,
            opens=fresh_opens,
        )
    )
    minute1, minute1_coverage = build_trajectories(
        fresh,
        policy="minute1_exit",
    )
    hold30, hold30_coverage = build_policy_trajectories(
        fresh,
        PolicySpec(None, None, None),
        opens=fresh_opens,
    )

    stop_only_spec = PolicySpec(
        selected_spec.stop_pct,
        None,
        None,
    )
    stop_only, stop_only_coverage = (
        build_policy_trajectories(
            fresh,
            stop_only_spec,
            opens=fresh_opens,
        )
    )
    profit_only_spec = PolicySpec(
        None,
        selected_spec.take_pct,
        selected_spec.trail_gap_pct,
    )
    profit_only, profit_only_coverage = (
        build_policy_trajectories(
            fresh,
            profit_only_spec,
            opens=fresh_opens,
        )
    )
    canonical_spec = PolicySpec(*CANONICAL)
    canonical, canonical_coverage = (
        build_policy_trajectories(
            fresh,
            canonical_spec,
            opens=fresh_opens,
        )
    )

    candidate_summary = summarize_enhanced(
        candidate,
        seed=BOOTSTRAP_SEED + 400,
    )
    minute1_summary = summarize(
        minute1,
        seed=BOOTSTRAP_SEED + 401,
    )
    hold30_summary = summarize_enhanced(
        hold30,
        seed=BOOTSTRAP_SEED + 402,
    )
    stop_only_summary = summarize_enhanced(
        stop_only,
        seed=BOOTSTRAP_SEED + 403,
    )
    profit_only_summary = summarize_enhanced(
        profit_only,
        seed=BOOTSTRAP_SEED + 404,
    )
    canonical_summary = summarize_enhanced(
        canonical,
        seed=BOOTSTRAP_SEED + 405,
    )

    diff_minute1 = matched_difference(
        candidate,
        minute1,
        seed=BOOTSTRAP_SEED + 406,
    )
    diff_hold30 = matched_difference(
        candidate,
        hold30,
        seed=BOOTSTRAP_SEED + 407,
    )

    positive_days, by_day = _positive_days(
        candidate
    )
    account = simulate_reference_account(
        candidate
    )

    c_day = candidate_summary[
        "day_balanced_base_mean_pct"
    ]
    c_low = candidate_summary[
        "day_bootstrap"
    ]["ci_low_pct"]
    c_severe = candidate_summary[
        "severe_loss_rate"
    ]
    h_severe = hold30_summary[
        "severe_loss_rate"
    ]
    d_m = diff_minute1[
        "day_balanced_difference_pct"
    ]
    d_m_low = diff_minute1[
        "bootstrap"
    ]["ci_low_pct"]
    d_h = diff_hold30[
        "day_balanced_difference_pct"
    ]
    d_h_low = diff_hold30[
        "bootstrap"
    ]["ci_low_pct"]

    gate = bool(
        candidate_summary["completion_coverage"]
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
        and d_m is not None
        and float(d_m) > 0
        and d_m_low is not None
        and float(d_m_low) > 0
        and d_h is not None
        and float(d_h) > 0
        and d_h_low is not None
        and float(d_h_low) > 0
        and c_severe is not None
        and h_severe is not None
        and float(c_severe) < float(h_severe)
        and float(account["ending_balance_krw"])
        > REFERENCE_START_KRW
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "policy_executed": True,
        "selected_policy": {
            "name": selected_spec.name,
            "stop_pct": selected_spec.stop_pct,
            "take_pct": selected_spec.take_pct,
            "trail_gap_pct": selected_spec.trail_gap_pct,
            "partial_fraction": selected_spec.partial_fraction,
        },
        "canonical_policy": {
            "stop_pct": CANONICAL[0],
            "take_pct": CANONICAL[1],
            "trail_gap_pct": CANONICAL[2],
            "partial_fraction": PARTIAL_FRACTION,
        },
        "calibration_policy_table": calibration_table,
        "candidate_coverage": candidate_coverage,
        "candidate": candidate_summary,
        "minute1_coverage": minute1_coverage,
        "minute1_comparator": minute1_summary,
        "hold30_coverage": hold30_coverage,
        "hold30_comparator": hold30_summary,
        "stop_only_coverage": stop_only_coverage,
        "stop_only_comparator": stop_only_summary,
        "profit_only_coverage": profit_only_coverage,
        "profit_only_comparator": profit_only_summary,
        "canonical_coverage": canonical_coverage,
        "canonical_comparator": canonical_summary,
        "matched_candidate_minus_minute1": diff_minute1,
        "matched_candidate_minus_hold30": diff_hold30,
        "candidate_positive_return_days": int(
            positive_days
        ),
        "candidate_by_day_mean_pct": by_day,
        "reference_account": account,
        "frozen_gate": {
            "min_completion_coverage": MIN_COMPLETION_COVERAGE,
            "candidate_day_balanced_positive": True,
            "candidate_bootstrap_low_positive": True,
            "min_positive_return_days": MIN_POSITIVE_DAYS,
            "matched_minute1_positive_with_ci": True,
            "matched_hold30_positive_with_ci": True,
            "severe_loss_below_hold30": True,
            "reference_account_above_start": True,
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

    def tagged(
        frame: pd.DataFrame,
        label: str,
    ) -> pd.DataFrame:
        out = frame.copy()
        out["comparison_label"] = label
        return out

    pd.concat(
        [
            tagged(candidate, "candidate"),
            tagged(minute1, "minute1"),
            tagged(hold30, "hold30"),
            tagged(stop_only, "stop_only"),
            tagged(profit_only, "profit_only"),
            tagged(canonical, "canonical"),
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
        args.calibration,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.trajectories_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
