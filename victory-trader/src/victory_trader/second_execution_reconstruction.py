"""Request244: reconstruct Request243 policy execution from one-second aggregates.

This diagnostic does not alter policy actions. Historical one-second aggregates are
used only after actions are frozen, as synthetic execution references. Missing opens
remain unresolved. The experiment is not a broker-fill or profitability claim.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from .chronological_action_value import (
    BASE,
    STRESS,
    CAL_DAYS,
    EXIT_DEADLINE,
    build_states,
    chronological_fit,
    episode_results,
    rollout,
)
from .config import load_settings
from .execution_costs import modeled_buy_fill, modeled_sell_fill
from .massive_client import MassiveClient

REQUEST_ID = 244
EXPIRY_MS = 5_000
LATENCIES_MS = (0, 1_000, 2_000, 5_000)
PRIMARY_LATENCY_MS = 1_000
MIN_ENTRY_FILL_COVERAGE = 0.90
MIN_EXIT_FILL_COVERAGE = 0.90
MIN_RESOLUTION_RATE = 0.80
MIN_RESOLUTION_UPLIFT = 0.20
MINUTE_MS = 60_000


def observed_opens(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for raw in payload.get("results") or []:
        t = raw.get("t")
        o = raw.get("o")
        if t is None or o is None:
            continue
        try:
            t = int(t)
            o = float(o)
        except (TypeError, ValueError):
            continue
        if t >= 0 and isfinite(o) and o > 0:
            rows.append((t, o))
    rows.sort()
    if any(rows[i][0] == rows[i - 1][0] for i in range(1, len(rows))):
        raise ValueError("duplicate one-second timestamp")
    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    return (
        np.asarray([x[0] for x in rows], dtype=np.int64),
        np.asarray([x[1] for x in rows], dtype=float),
    )


def first_observed_open(
    times: np.ndarray,
    opens: np.ndarray,
    *,
    decision_t: int,
    latency_ms: int,
    expiry_ms: int = EXPIRY_MS,
) -> tuple[int | None, float | None]:
    if latency_ms < 0 or expiry_ms < 0:
        raise ValueError("latency and expiry must be non-negative")
    target = int(decision_t) + int(latency_ms)
    idx = int(np.searchsorted(times, target, side="left"))
    if idx >= len(times) or int(times[idx]) > target + int(expiry_ms):
        return None, None
    return int(times[idx]), float(opens[idx])


def modeled_return(entry_ref: float, exit_ref: float, scenario) -> float:
    buy = modeled_buy_fill(float(entry_ref), scenario)
    sell = modeled_sell_fill(float(exit_ref), scenario)
    sell *= 1 - scenario.sell_fee_bps / 10_000
    return float((sell / buy - 1) * 100)


def execution_decisions(episodes: pd.DataFrame) -> pd.DataFrame:
    work = episodes.copy()
    work["entry_decision_t"] = pd.to_numeric(work["entry_t"], errors="coerce")
    work["policy_exit_missing"] = work["exit_t"].isna()
    work["exit_decision_t"] = pd.to_numeric(work["exit_t"], errors="coerce")
    deadline = pd.to_numeric(work["hot_t"], errors="coerce") + EXIT_DEADLINE * MINUTE_MS
    missing = work["entered"] & work["exit_decision_t"].isna()
    work.loc[missing, "exit_decision_t"] = deadline.loc[missing]
    work["deadline_liquidation"] = missing
    return work


def fetch_paths(
    client: MassiveClient,
    decisions: pd.DataFrame,
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], dict]:
    paths = {}
    audit = {}
    entered = decisions.loc[decisions.entered].copy()
    keys = (
        entered.loc[:, ["trading_day", "ticker"]]
        .drop_duplicates()
        .sort_values(["trading_day", "ticker"])
    )
    for row in keys.itertuples(index=False):
        day_text = str(row.trading_day)
        ticker = str(row.ticker).upper()
        payload = client.second_bars_range(
            ticker,
            date.fromisoformat(day_text),
            date.fromisoformat(day_text),
            adjusted=False,
        )
        times, opens = observed_opens(payload)
        paths[(day_text, ticker)] = (times, opens)
        audit[f"{day_text}:{ticker}"] = {
            "rows": int(len(times)),
            "first_t": int(times[0]) if len(times) else None,
            "last_t": int(times[-1]) if len(times) else None,
        }
    return paths, audit


def reconstruct(
    decisions: pd.DataFrame,
    paths: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    *,
    latency_ms: int,
) -> pd.DataFrame:
    rows = []
    for record in decisions.to_dict("records"):
        out = dict(record)
        if not bool(record["entered"]):
            out.update(
                execution_status="cash",
                reconstructed_base_pct=0.0,
                reconstructed_stress_pct=0.0,
                entry_fill_t=None,
                exit_fill_t=None,
                entry_reference=None,
                exit_reference=None,
            )
            rows.append(out)
            continue

        key = (str(record["trading_day"]), str(record["ticker"]).upper())
        times, opens = paths.get(
            key, (np.array([], dtype=np.int64), np.array([], dtype=float))
        )
        entry_t = record.get("entry_decision_t")
        exit_t = record.get("exit_decision_t")
        if pd.isna(entry_t):
            out.update(
                execution_status="missing_policy_entry_time",
                reconstructed_base_pct=None,
                reconstructed_stress_pct=None,
                entry_fill_t=None,
                exit_fill_t=None,
                entry_reference=None,
                exit_reference=None,
            )
            rows.append(out)
            continue

        entry_fill_t, entry_ref = first_observed_open(
            times,
            opens,
            decision_t=int(entry_t),
            latency_ms=latency_ms,
        )
        if entry_ref is None:
            out.update(
                execution_status="entry_unavailable",
                reconstructed_base_pct=None,
                reconstructed_stress_pct=None,
                entry_fill_t=None,
                exit_fill_t=None,
                entry_reference=None,
                exit_reference=None,
            )
            rows.append(out)
            continue

        if pd.isna(exit_t):
            out.update(
                execution_status="missing_policy_exit_time",
                reconstructed_base_pct=None,
                reconstructed_stress_pct=None,
                entry_fill_t=entry_fill_t,
                exit_fill_t=None,
                entry_reference=entry_ref,
                exit_reference=None,
            )
            rows.append(out)
            continue

        exit_fill_t, exit_ref = first_observed_open(
            times,
            opens,
            decision_t=int(exit_t),
            latency_ms=latency_ms,
        )
        if exit_ref is None:
            out.update(
                execution_status="exit_unavailable",
                reconstructed_base_pct=None,
                reconstructed_stress_pct=None,
                entry_fill_t=entry_fill_t,
                exit_fill_t=None,
                entry_reference=entry_ref,
                exit_reference=None,
            )
            rows.append(out)
            continue

        out.update(
            execution_status="closed",
            reconstructed_base_pct=modeled_return(entry_ref, exit_ref, BASE),
            reconstructed_stress_pct=modeled_return(entry_ref, exit_ref, STRESS),
            entry_fill_t=entry_fill_t,
            exit_fill_t=exit_fill_t,
            entry_reference=entry_ref,
            exit_reference=exit_ref,
        )
        rows.append(out)
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame, source_unresolved: int) -> dict:
    entered = rows.loc[rows.entered]
    closed = entered.loc[entered.execution_status.eq("closed")]
    entry_available = entered.loc[
        ~entered.execution_status.eq("entry_unavailable")
        & ~entered.execution_status.eq("missing_policy_entry_time")
    ]
    entry_coverage = float(len(entry_available) / len(entered)) if len(entered) else 0.0
    exit_coverage = float(len(closed) / len(entry_available)) if len(entry_available) else 0.0
    resolution_rate = float(len(closed) / len(entered)) if len(entered) else 0.0
    old_resolution_rate = (
        float((len(entered) - source_unresolved) / len(entered)) if len(entered) else 0.0
    )
    resolution_uplift = resolution_rate - old_resolution_rate

    by_day = {}
    for day in CAL_DAYS:
        part = rows.loc[rows.trading_day.astype(str).eq(day)]
        traded = part.loc[part.entered]
        if traded.execution_status.ne("closed").any():
            by_day[day] = None
        else:
            by_day[day] = float(part.reconstructed_base_pct.fillna(0).mean())

    paired = closed.loc[~closed.unresolved.astype(bool)]
    paired_delta = None
    if len(paired):
        paired_delta = float(
            (
                pd.to_numeric(paired.reconstructed_base_pct, errors="coerce")
                - pd.to_numeric(paired.net_pct, errors="coerce")
            ).mean()
        )

    checks = {
        "entry_fill_coverage": entry_coverage >= MIN_ENTRY_FILL_COVERAGE,
        "exit_fill_coverage": exit_coverage >= MIN_EXIT_FILL_COVERAGE,
        "resolution_rate": resolution_rate >= MIN_RESOLUTION_RATE,
        "resolution_uplift": resolution_uplift >= MIN_RESOLUTION_UPLIFT,
    }
    return {
        "episodes": int(len(rows)),
        "entries": int(len(entered)),
        "source_request243_unresolved": int(source_unresolved),
        "statuses": dict(Counter(entered.execution_status.astype(str))),
        "entry_fill_coverage": entry_coverage,
        "exit_fill_coverage_conditional_on_entry": exit_coverage,
        "resolution_rate": resolution_rate,
        "request243_resolution_rate": old_resolution_rate,
        "resolution_uplift": resolution_uplift,
        "closed_trades": int(len(closed)),
        "closed_mean_base_pct": (
            float(pd.to_numeric(closed.reconstructed_base_pct).mean())
            if len(closed)
            else None
        ),
        "closed_mean_stress_pct": (
            float(pd.to_numeric(closed.reconstructed_stress_pct).mean())
            if len(closed)
            else None
        ),
        "closed_positive_rate": (
            float(pd.to_numeric(closed.reconstructed_base_pct).gt(0).mean())
            if len(closed)
            else None
        ),
        "closed_severe_loss_rate_le_minus2": (
            float(pd.to_numeric(closed.reconstructed_base_pct).le(-2).mean())
            if len(closed)
            else None
        ),
        "paired_source_resolved_count": int(len(paired)),
        "paired_mean_base_delta_vs_request243_pct": paired_delta,
        "deadline_liquidations_attempted": int(entered.deadline_liquidation.sum()),
        "by_day_equal_episode_mean_base_pct": by_day,
        "checks": checks,
        "structural_gate_pass": bool(all(checks.values())),
    }


def evaluate(
    fit_positions: Path,
    calibration_positions: Path,
    output: Path,
    rows_output: Path,
) -> int:
    fit_states = build_states(pd.read_parquet(fit_positions))
    cal_states = build_states(pd.read_parquet(calibration_positions))
    if sorted(cal_states.trading_day.astype(str).unique()) != CAL_DAYS:
        raise ValueError("calibration dates changed")

    _, student, _ = chronological_fit(fit_states)
    scored = rollout(cal_states, student, "full")
    source = episode_results(scored)
    decisions = execution_decisions(source)
    source_unresolved = int(decisions.loc[decisions.entered, "unresolved"].sum())

    client = MassiveClient(
        load_settings().massive_api_key,
        cache_dir=Path("data/cache/request244-second-bars"),
        request_interval_seconds=0.2,
    )
    paths, path_audit = fetch_paths(client, decisions)

    reports = {}
    frames = []
    for latency in LATENCIES_MS:
        rows = reconstruct(decisions, paths, latency_ms=latency)
        rows["latency_ms"] = latency
        reports[str(latency)] = summarize(rows, source_unresolved)
        frames.append(rows)

    primary = reports[str(PRIMARY_LATENCY_MS)]
    result = {
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "development_only": True,
        "policy_changed": False,
        "calibration_days": CAL_DAYS,
        "primary_latency_ms": PRIMARY_LATENCY_MS,
        "expiry_ms": EXPIRY_MS,
        "primary": primary,
        "latency_sensitivity": reports,
        "path_audit": path_audit,
        "api_stats": client.stats.to_dict(),
        "interpretation": (
            "Observed one-second aggregate opens are synthetic execution references, "
            "not brokerage fills. This request changes execution accounting only."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_parquet(rows_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument("--calibration-positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
