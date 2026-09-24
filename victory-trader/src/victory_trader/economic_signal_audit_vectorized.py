"""Vectorized Request-164 fixed-horizon audit.

Reuses stored Request-163 first-HOT rows and avoids building a Python price
lookup over the full market scan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .economic_signal_audit import (
    REQUEST_ID,
    _fixed_horizon_audit,
    _missing_audit,
    _signal_rows,
)
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS
from .hot_entry_economics import HORIZONS

MINUTE_MS = 60_000


def _add_fixed_horizon_returns(
    features: pd.DataFrame,
    scan_path: Path,
) -> pd.DataFrame:
    events = features.loc[:, ["trading_day", "ticker", "t"]].copy()
    events = events.reset_index(drop=True)
    events["_event_id"] = np.arange(len(events), dtype=np.int64)
    events["ticker"] = events["ticker"].astype(str).str.upper()

    needed_parts = []
    entry = events.loc[:, ["_event_id", "trading_day", "ticker", "t"]].copy()
    entry["target_t"] = pd.to_numeric(entry["t"], errors="raise").astype("int64") + MINUTE_MS
    entry["kind"] = "entry"
    needed_parts.append(entry.loc[:, ["_event_id", "trading_day", "ticker", "target_t", "kind"]])

    for horizon in HORIZONS:
        part = events.loc[:, ["_event_id", "trading_day", "ticker", "t"]].copy()
        part["target_t"] = (
            pd.to_numeric(part["t"], errors="raise").astype("int64")
            + (int(horizon) + 1) * MINUTE_MS
        )
        part["kind"] = f"exit_{int(horizon)}"
        needed_parts.append(
            part.loc[:, ["_event_id", "trading_day", "ticker", "target_t", "kind"]]
        )

    needed = pd.concat(needed_parts, ignore_index=True)

    scan = pd.read_parquet(
        scan_path,
        columns=["trading_day", "ticker", "t", "o"],
    )
    scan["ticker"] = scan["ticker"].astype(str).str.upper()
    scan["t"] = pd.to_numeric(scan["t"], errors="coerce").astype("Int64")
    scan["o"] = pd.to_numeric(scan["o"], errors="coerce")
    scan = scan.loc[scan["o"].gt(0)].copy()
    scan = scan.rename(columns={"t": "target_t", "o": "price"})
    scan = scan.drop_duplicates(
        ["trading_day", "ticker", "target_t"], keep="last"
    )

    matched = needed.merge(
        scan,
        on=["trading_day", "ticker", "target_t"],
        how="left",
        validate="many_to_one",
    )
    wide = matched.pivot(
        index="_event_id", columns="kind", values="price"
    )

    result = features.reset_index(drop=True).copy()
    result["_event_id"] = np.arange(len(result), dtype=np.int64)
    result["vectorized_entry_price"] = result["_event_id"].map(
        wide.get("entry", pd.Series(dtype=float))
    )

    base = next(
        scenario
        for scenario in DEFAULT_EXECUTION_SCENARIOS
        if scenario.name == "base"
    )
    entry_price = pd.to_numeric(
        result["vectorized_entry_price"], errors="coerce"
    )
    entry_half = np.maximum(
        entry_price * base.half_spread_bps / 10_000.0,
        base.min_half_spread_cents / 100.0,
    )
    entry_fill = (
        entry_price * (1.0 + base.slippage_bps / 10_000.0)
        + entry_half
    )

    for horizon in HORIZONS:
        exit_col = f"exit_{int(horizon)}"
        exit_price = result["_event_id"].map(
            wide.get(exit_col, pd.Series(dtype=float))
        )
        exit_price = pd.to_numeric(exit_price, errors="coerce")
        gross = (exit_price / entry_price - 1.0) * 100.0
        result[f"return_{int(horizon)}m_gross_pct"] = gross

        exit_half = np.maximum(
            exit_price * base.half_spread_bps / 10_000.0,
            base.min_half_spread_cents / 100.0,
        )
        exit_fill = (
            exit_price * (1.0 - base.slippage_bps / 10_000.0)
            - exit_half
        ).clip(lower=0)
        proceeds = exit_fill * (1.0 - base.sell_fee_bps / 10_000.0)
        net = (proceeds / entry_fill - 1.0) * 100.0
        net = net.where(entry_price.gt(0) & exit_price.gt(0))
        result[f"return_{int(horizon)}m_base_net_pct"] = net

    return result.drop(columns=["_event_id"])


def run(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_json: Path,
    signal_csv: Path,
    horizon_csv: Path,
) -> int:
    features = pd.read_parquet(first_hot_path)
    enriched = _add_fixed_horizon_returns(features, opportunity_scan_path)

    missing = _missing_audit(enriched)
    signals = _signal_rows(enriched)
    horizons = _fixed_horizon_audit(enriched)

    summary = {
        "schema_version": 3,
        "request_id": REQUEST_ID,
        "source_request": 163,
        "execution_mode": "stored_first_hot_vectorized_fixed_horizons",
        "opens_new_dates": False,
        "changes_model_or_threshold": False,
        "missing_label_audit": missing,
        "signals": signals,
        "fixed_horizon_audit": horizons,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(signals).to_csv(signal_csv, index=False)
    pd.DataFrame(
        [
            {k: v for k, v in row.items() if k != "by_day"}
            for row in horizons["horizons"]
        ]
    ).to_csv(horizon_csv, index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--signal-csv", type=Path, required=True)
    parser.add_argument("--horizon-csv", type=Path, required=True)
    args = parser.parse_args()
    return run(
        args.first_hot,
        args.opportunity_scan,
        args.output_json,
        args.signal_csv,
        args.horizon_csv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
