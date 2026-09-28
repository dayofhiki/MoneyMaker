"""Request262: audit minute-reference alignment for frozen Request258."""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

from .frozen_minute_policy_second_audit import (
    EVENT_HORIZONS,
    anchor_times,
    frozen_request258,
)
from .full_hot_fixed_policy_value import event_lookup
from .recurrent_wait_entry_action_value import _base_return

REQUEST_ID = 262
MINUTE_MS = 60_000
MIN_COVERAGE = 0.70
MAX_SEVERE_RATE = 0.10
MIN_PAIRED_DELTA = -0.50


def build_reference_maps(scan: pd.DataFrame):
    required = {"trading_day", "ticker", "t", "bar_start_t", "o"}
    missing = required - set(scan.columns)
    if missing:
        raise ValueError(f"Request262 scan missing columns: {sorted(missing)}")

    frame = scan.loc[
        :, ["trading_day", "ticker", "t", "bar_start_t", "o"]
    ].copy()
    for column in ("t", "bar_start_t", "o"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame.t.notna()
        & frame.bar_start_t.notna()
        & frame.o.gt(0)
    ].copy()
    frame["t"] = frame.t.astype("int64")
    frame["bar_start_t"] = frame.bar_start_t.astype("int64")
    frame["lag_ms"] = frame.t - frame.bar_start_t

    legacy = {}
    causal = {}
    for row in frame.itertuples(index=False):
        key = (str(row.trading_day), str(row.ticker).upper())
        legacy[(key[0], key[1], int(row.t))] = float(row.o)
        causal[(key[0], key[1], int(row.bar_start_t))] = float(row.o)

    lag = pd.to_numeric(frame.lag_ms, errors="coerce")
    audit = {
        "rows": int(len(frame)),
        "exact_60s_shift_rate": float(lag.eq(MINUTE_MS).mean()) if len(frame) else None,
        "lag_ms_min": int(lag.min()) if len(frame) else None,
        "lag_ms_median": float(lag.median()) if len(frame) else None,
        "lag_ms_max": int(lag.max()) if len(frame) else None,
    }
    return legacy, causal, audit


def reconstruct_causal_minute(
    policy: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    legacy, causal, timing_audit = build_reference_maps(scan)
    lookup = event_lookup(scan)
    rows = []

    for raw in policy.to_dict("records"):
        out = dict(raw)
        if not bool(raw["entered"]):
            out.update(
                causal_status="cash",
                causal_minute_value_pct=0.0,
                legacy_entry_reference=None,
                causal_entry_reference=None,
                paired_entry_reference_delta_pct=None,
            )
            rows.append(out)
            continue

        day = str(raw["trading_day"])
        ticker = str(raw["ticker"]).upper()
        entry_t = int(raw["entry_t"])
        hot_t = int(raw["hot_t"])
        anchors = anchor_times(
            lookup,
            trading_day=day,
            ticker=ticker,
            hot_t=hot_t,
            entry_t=entry_t,
        )
        if anchors is None:
            out.update(
                causal_status="missing_policy_exit_anchors",
                causal_minute_value_pct=None,
                legacy_entry_reference=legacy.get((day, ticker, entry_t)),
                causal_entry_reference=causal.get((day, ticker, entry_t)),
                paired_entry_reference_delta_pct=None,
            )
            rows.append(out)
            continue

        legacy_entry = legacy.get((day, ticker, entry_t))
        causal_entry = causal.get((day, ticker, entry_t))
        if causal_entry is None:
            out.update(
                causal_status="causal_entry_reference_unavailable",
                causal_minute_value_pct=None,
                legacy_entry_reference=legacy_entry,
                causal_entry_reference=None,
                paired_entry_reference_delta_pct=None,
            )
            rows.append(out)
            continue

        exit_refs = [causal.get((day, ticker, int(t))) for t in anchors]
        if any(ref is None for ref in exit_refs):
            out.update(
                causal_status="causal_exit_reference_unavailable",
                causal_minute_value_pct=None,
                legacy_entry_reference=legacy_entry,
                causal_entry_reference=causal_entry,
                paired_entry_reference_delta_pct=(
                    (causal_entry / legacy_entry - 1.0) * 100.0
                    if legacy_entry is not None and legacy_entry > 0
                    else None
                ),
            )
            rows.append(out)
            continue

        values = [
            _base_return(float(causal_entry), float(ref))
            for ref in exit_refs
        ]
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Request262 causal minute return is non-finite")

        entry_delta = (
            (float(causal_entry) / float(legacy_entry) - 1.0) * 100.0
            if legacy_entry is not None and legacy_entry > 0
            else None
        )
        exit_deltas = []
        for anchor_t, causal_exit in zip(anchors, exit_refs):
            legacy_exit = legacy.get((day, ticker, int(anchor_t)))
            if legacy_exit is not None and legacy_exit > 0:
                exit_deltas.append(
                    (float(causal_exit) / float(legacy_exit) - 1.0) * 100.0
                )

        out.update(
            causal_status="resolved",
            causal_minute_value_pct=float(np.mean(values)),
            legacy_entry_reference=legacy_entry,
            causal_entry_reference=float(causal_entry),
            paired_entry_reference_delta_pct=entry_delta,
            mean_exit_reference_delta_pct=(
                float(np.mean(exit_deltas)) if exit_deltas else None
            ),
        )
        rows.append(out)

    result = pd.DataFrame(rows)
    result.attrs["timing_audit"] = timing_audit
    return result


def summarize(rows: pd.DataFrame) -> dict:
    entered = rows.loc[rows.entered.astype(bool)].copy()
    resolved = entered.loc[entered.causal_status.eq("resolved")].copy()
    coverage = float(len(resolved) / len(entered)) if len(entered) else 0.0

    causal = pd.to_numeric(
        resolved.causal_minute_value_pct,
        errors="coerce",
    )
    legacy = pd.to_numeric(resolved.value_pct, errors="coerce")
    delta = causal - legacy
    entry_delta = pd.to_numeric(
        resolved.paired_entry_reference_delta_pct,
        errors="coerce",
    )
    exit_delta = pd.to_numeric(
        resolved.mean_exit_reference_delta_pct,
        errors="coerce",
    )

    checks = {
        "causal_reference_coverage": coverage >= MIN_COVERAGE,
        "causal_mean_positive": len(causal) > 0 and float(causal.mean()) > 0,
        "causal_severe_loss_rate": (
            len(causal) > 0
            and float(causal.le(-2).mean()) <= MAX_SEVERE_RATE
        ),
        "paired_delta": (
            len(delta) > 0
            and float(delta.mean()) >= MIN_PAIRED_DELTA
        ),
    }
    return {
        "entries": int(len(entered)),
        "statuses": entered.causal_status.value_counts().to_dict(),
        "causal_reference_coverage": coverage,
        "resolved_trades": int(len(resolved)),
        "legacy_resolved_mean_pct": (
            float(legacy.mean()) if len(legacy) else None
        ),
        "causal_minute_mean_pct": (
            float(causal.mean()) if len(causal) else None
        ),
        "causal_minute_positive_rate": (
            float(causal.gt(0).mean()) if len(causal) else None
        ),
        "causal_minute_severe_loss_rate_le_minus2": (
            float(causal.le(-2).mean()) if len(causal) else None
        ),
        "paired_mean_delta_vs_legacy_pct": (
            float(delta.mean()) if len(delta) else None
        ),
        "mean_entry_reference_shift_pct": (
            float(entry_delta.mean()) if entry_delta.notna().any() else None
        ),
        "median_entry_reference_shift_pct": (
            float(entry_delta.median()) if entry_delta.notna().any() else None
        ),
        "mean_exit_reference_shift_pct": (
            float(exit_delta.mean()) if exit_delta.notna().any() else None
        ),
        "checks": checks,
        "economic_survival_pass": bool(all(checks.values())),
    }


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    policy, _ = frozen_request258(first_hot, scan)
    rows = reconstruct_causal_minute(policy, scan)
    timing = rows.attrs["timing_audit"]
    summary = summarize(rows)

    bug_confirmed = bool(
        timing["exact_60s_shift_rate"] is not None
        and timing["exact_60s_shift_rate"] >= 0.99
    )
    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "policy_changed": False,
        "source_policy_request": 258,
        "event_horizons": list(EVENT_HORIZONS),
        "minute_timestamp_audit": timing,
        "retroactive_reference_bug_confirmed": bug_confirmed,
        "economics": summary,
        "request258_economic_promotion_evidence_valid": bool(
            not bug_confirmed or summary["economic_survival_pass"]
        ),
        "interpretation": (
            "The broad scan intentionally shifts bar-start timestamps by +60s "
            "so completed minute OHLCV is causal. A legacy execution reference "
            "using that shifted row's unchanged open is therefore the open from "
            "the minute before the decision. The causal minute reference uses "
            "the next bar whose original bar_start_t equals the decision time."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    rows.to_parquet(rows_output, index=False)
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
