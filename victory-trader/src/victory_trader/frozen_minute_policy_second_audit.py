"""Request259: one-second execution-reference audit for frozen Request258."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings
from .full_hot_fixed_policy_value import attach_fixed_value, event_lookup
from .learned_pullback_entry import build_episode_states
from .massive_client import MassiveClient
from .minute_state_entry_controller import (
    MODEL_CALIBRATION_DAY,
    MODEL_FIT_DAYS,
    TEST_DAYS,
    fit_models,
    make_policy,
    minute_state_columns,
    pullback_rows,
)
from .recurrent_wait_entry_action_value import _base_return
from .rich_post_hot_state import (
    attach_dynamic_deltas,
    attach_minute_features,
    score_and_select_candidates,
)
from .second_execution_reconstruction import (
    first_observed_open,
    observed_opens,
)

REQUEST_ID = 259
LATENCIES_MS = (0, 1000, 2000, 5000)
PRIMARY_LATENCY_MS = 1000
EXPIRY_MS = 5000
RISK_CAP_MINUTES = 30
EVENT_HORIZONS = (1, 2, 3, 5)

MIN_ENTRY_COVERAGE = 0.80
MIN_EXIT_COVERAGE = 0.70
MIN_FULL_RESOLUTION = 0.60
MAX_SEVERE_RATE = 0.10
MIN_PAIRED_DELTA = -0.50


def frozen_request258(
    first_hot: pd.DataFrame,
    scan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    work = attach_fixed_value(work, scan)
    scored, train_threshold, test_threshold = score_and_select_candidates(
        work
    )
    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(
            [*MODEL_FIT_DAYS, MODEL_CALIBRATION_DAY]
        )
        & scored.candidate_probability.ge(train_threshold)
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_probability.ge(test_threshold)
    ].copy()

    train_states = build_episode_states(train_selected, scan)
    test_states = build_episode_states(test_selected, scan)
    combined = pd.concat(
        [
            train_states.assign(_split="train"),
            test_states.assign(_split="test"),
        ],
        ignore_index=True,
    )
    combined = attach_minute_features(combined, scan)
    combined = attach_dynamic_deltas(combined)
    train_states = combined.loc[combined._split.eq("train")].drop(
        columns="_split"
    )
    test_states = combined.loc[combined._split.eq("test")].drop(
        columns="_split"
    )
    train_pullback = pullback_rows(train_states)
    fit_pullback = train_pullback.loc[
        train_pullback.trading_day.astype(str).isin(MODEL_FIT_DAYS)
    ].copy()
    cal_pullback = train_pullback.loc[
        train_pullback.trading_day.astype(str).eq(MODEL_CALIBRATION_DAY)
    ].copy()
    columns = minute_state_columns(fit_pullback)
    fitted, _ = fit_models(fit_pullback, cal_pullback, columns)
    policy = make_policy(test_states, test_selected, fitted)
    return policy, test_selected


def anchor_times(
    scan_lookup: dict,
    *,
    trading_day: str,
    ticker: str,
    hot_t: int,
    entry_t: int,
) -> tuple[int, ...] | None:
    episode = scan_lookup.get((str(trading_day), str(ticker).upper()), [])
    cap_t = int(hot_t) + RISK_CAP_MINUTES * 60_000
    future = [
        (int(t), float(price))
        for t, price in episode
        if int(entry_t) < int(t) <= cap_t and float(price) > 0
    ]
    if len(future) < max(EVENT_HORIZONS):
        return None
    return tuple(int(future[h - 1][0]) for h in EVENT_HORIZONS)


def fetch_second_paths(
    policy: pd.DataFrame,
    client: MassiveClient,
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], dict]:
    entered = policy.loc[policy.entered.astype(bool)]
    keys = (
        entered.loc[:, ["trading_day", "ticker"]]
        .drop_duplicates()
        .sort_values(["trading_day", "ticker"])
    )
    paths = {}
    audit = {}
    for row in keys.itertuples(index=False):
        day_text = str(row.trading_day)
        ticker = str(row.ticker).upper()
        d = date.fromisoformat(day_text)
        payload = client.second_bars_range(
            ticker,
            d,
            d,
            adjusted=False,
        )
        times, opens = observed_opens(payload)
        paths[(day_text, ticker)] = (times, opens)
        audit[f"{day_text}|{ticker}"] = {
            "observed_second_opens": int(len(times)),
            "first_t": int(times[0]) if len(times) else None,
            "last_t": int(times[-1]) if len(times) else None,
        }
    return paths, audit


def reconstruct(
    policy: pd.DataFrame,
    scan_lookup: dict,
    paths: dict,
    *,
    latency_ms: int,
) -> pd.DataFrame:
    rows = []
    for raw in policy.to_dict("records"):
        out = dict(raw)
        if not bool(raw["entered"]):
            out.update(
                execution_status="cash",
                reconstructed_value_pct=0.0,
                entry_reference=None,
                exit_references=None,
            )
            rows.append(out)
            continue

        key = (
            str(raw["trading_day"]),
            str(raw["ticker"]).upper(),
        )
        times, opens = paths.get(
            key,
            (np.array([], dtype=np.int64), np.array([], dtype=float)),
        )
        entry_t = raw.get("entry_t")
        if entry_t is None or pd.isna(entry_t):
            out.update(
                execution_status="missing_policy_entry_time",
                reconstructed_value_pct=None,
                entry_reference=None,
                exit_references=None,
            )
            rows.append(out)
            continue

        anchors = anchor_times(
            scan_lookup,
            trading_day=str(raw["trading_day"]),
            ticker=str(raw["ticker"]),
            hot_t=int(raw["hot_t"]),
            entry_t=int(entry_t),
        )
        if anchors is None:
            out.update(
                execution_status="missing_policy_exit_anchors",
                reconstructed_value_pct=None,
                entry_reference=None,
                exit_references=None,
            )
            rows.append(out)
            continue

        _, entry_ref = first_observed_open(
            times,
            opens,
            decision_t=int(entry_t),
            latency_ms=latency_ms,
            expiry_ms=EXPIRY_MS,
        )
        if entry_ref is None:
            out.update(
                execution_status="entry_reference_unavailable",
                reconstructed_value_pct=None,
                entry_reference=None,
                exit_references=None,
            )
            rows.append(out)
            continue

        exit_refs = []
        missing_exit = False
        for anchor_t in anchors:
            _, exit_ref = first_observed_open(
                times,
                opens,
                decision_t=int(anchor_t),
                latency_ms=latency_ms,
                expiry_ms=EXPIRY_MS,
            )
            if exit_ref is None:
                missing_exit = True
                break
            exit_refs.append(float(exit_ref))

        if missing_exit:
            out.update(
                execution_status="exit_reference_unavailable",
                reconstructed_value_pct=None,
                entry_reference=float(entry_ref),
                exit_references=exit_refs,
            )
            rows.append(out)
            continue

        values = [
            _base_return(float(entry_ref), float(exit_ref))
            for exit_ref in exit_refs
        ]
        if not all(np.isfinite(v) for v in values):
            out.update(
                execution_status="invalid_reference_return",
                reconstructed_value_pct=None,
                entry_reference=float(entry_ref),
                exit_references=exit_refs,
            )
            rows.append(out)
            continue
        out.update(
            execution_status="resolved",
            reconstructed_value_pct=float(np.mean(values)),
            entry_reference=float(entry_ref),
            exit_references=exit_refs,
        )
        rows.append(out)
    return pd.DataFrame(rows)


def report(rows: pd.DataFrame) -> dict:
    entered = rows.loc[rows.entered.astype(bool)].copy()
    entry_available = entered.loc[
        ~entered.execution_status.isin(
            [
                "entry_reference_unavailable",
                "missing_policy_entry_time",
                "missing_policy_exit_anchors",
            ]
        )
    ]
    resolved = entered.loc[entered.execution_status.eq("resolved")].copy()
    entry_coverage = (
        float(len(entry_available) / len(entered)) if len(entered) else 0.0
    )
    exit_coverage = (
        float(len(resolved) / len(entry_available))
        if len(entry_available)
        else 0.0
    )
    resolution = (
        float(len(resolved) / len(entered)) if len(entered) else 0.0
    )
    reconstructed = pd.to_numeric(
        resolved.reconstructed_value_pct,
        errors="coerce",
    )
    minute = pd.to_numeric(resolved.value_pct, errors="coerce")
    paired_delta = (
        float((reconstructed - minute).mean()) if len(resolved) else None
    )

    complete_day_means = {}
    for day in TEST_DAYS:
        part = rows.loc[rows.trading_day.astype(str).eq(str(day))].copy()
        unresolved_entry = part.loc[
            part.entered.astype(bool)
            & ~part.execution_status.eq("resolved")
        ]
        if len(unresolved_entry):
            complete_day_means[str(day)] = None
        else:
            values = pd.to_numeric(
                part.reconstructed_value_pct,
                errors="coerce",
            ).fillna(0.0)
            complete_day_means[str(day)] = (
                float(values.mean()) if len(values) else None
            )

    checks = {
        "entry_reference_coverage": entry_coverage >= MIN_ENTRY_COVERAGE,
        "all_exit_anchor_coverage_given_entry": (
            exit_coverage >= MIN_EXIT_COVERAGE
        ),
        "full_trade_resolution": resolution >= MIN_FULL_RESOLUTION,
        "resolved_mean_positive": (
            len(reconstructed) > 0 and float(reconstructed.mean()) > 0
        ),
        "resolved_severe_loss_rate": (
            len(reconstructed) > 0
            and float(reconstructed.le(-2).mean()) <= MAX_SEVERE_RATE
        ),
        "paired_mean_delta": (
            paired_delta is not None
            and paired_delta >= MIN_PAIRED_DELTA
        ),
    }
    return {
        "candidate_episodes": int(len(rows)),
        "entries": int(len(entered)),
        "cash_episodes": int((~rows.entered.astype(bool)).sum()),
        "statuses": dict(Counter(entered.execution_status.astype(str))),
        "entry_reference_coverage": entry_coverage,
        "all_exit_anchor_coverage_given_entry_reference": exit_coverage,
        "full_trade_resolution": resolution,
        "resolved_trades": int(len(resolved)),
        "resolved_mean_base_pct": (
            float(reconstructed.mean()) if len(reconstructed) else None
        ),
        "resolved_positive_rate": (
            float(reconstructed.gt(0).mean()) if len(reconstructed) else None
        ),
        "resolved_severe_loss_rate_le_minus2": (
            float(reconstructed.le(-2).mean()) if len(reconstructed) else None
        ),
        "paired_mean_delta_vs_minute_pct": paired_delta,
        "complete_day_policy_means_pct": complete_day_means,
        "checks": checks,
        "execution_gate_pass": bool(all(checks.values())),
    }


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    policy, selected = frozen_request258(first_hot, scan)
    if len(policy) != len(selected):
        raise ValueError("Request259 frozen policy episode count changed")

    client = MassiveClient(
        load_settings().massive_api_key,
        cache_dir=Path("data/cache/request259-second-bars"),
        request_interval_seconds=0.05,
    )
    paths, path_audit = fetch_second_paths(policy, client)
    lookup = event_lookup(scan)

    reports = {}
    frames = []
    for latency in LATENCIES_MS:
        rows = reconstruct(
            policy,
            lookup,
            paths,
            latency_ms=latency,
        )
        rows["latency_ms"] = int(latency)
        reports[str(latency)] = report(rows)
        frames.append(rows)

    primary = reports[str(PRIMARY_LATENCY_MS)]
    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "policy_changed": False,
        "source_policy_request": 258,
        "test_days": list(TEST_DAYS),
        "primary_latency_ms": PRIMARY_LATENCY_MS,
        "expiry_ms": EXPIRY_MS,
        "event_horizons": list(EVENT_HORIZONS),
        "primary": primary,
        "latency_sensitivity": reports,
        "path_audit": path_audit,
        "api_stats": client.stats.to_dict(),
        "interpretation": (
            "Historical one-second aggregate opens are conservative execution "
            "references, not tick trades, NBBO, order-book observations, or "
            "brokerage fills. Missing aggregate opens remain unresolved."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.concat(frames, ignore_index=True).to_parquet(
        rows_output,
        index=False,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--first-hot", type=Path, required=True)
    p.add_argument("--opportunity-scan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows-output", type=Path, required=True)
    a = p.parse_args()
    return evaluate(
        a.first_hot,
        a.opportunity_scan,
        a.output,
        a.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
