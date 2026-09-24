"""Fast Request-164 audit from the completed Request-163 first-HOT artifact.

This avoids replaying attention/HOT selection. It reuses the exact stored
first-HOT feature rows, joins mechanical fixed-horizon economic labels from the
stored Request-163 scan, and runs the same diagnostics as economic_signal_audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .economic_signal_audit import (
    REQUEST_ID,
    _fixed_horizon_audit,
    _missing_audit,
    _signal_rows,
)
from .hot_entry_economics import HORIZONS, label_hot_event_economics


def run(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_json: Path,
    signal_csv: Path,
    horizon_csv: Path,
) -> int:
    features = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)

    events = features.loc[:, ["trading_day", "ticker", "t"]].copy()
    labels = label_hot_event_economics(events, scan).rename(
        columns={"hot_t": "t"}
    )

    horizon_columns = ["trading_day", "ticker", "t"]
    for horizon in HORIZONS:
        for column in (
            f"return_{horizon}m_gross_pct",
            f"return_{horizon}m_base_net_pct",
        ):
            if column in labels.columns:
                horizon_columns.append(column)

    enriched = features.merge(
        labels.loc[:, horizon_columns],
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
    )

    missing = _missing_audit(enriched)
    signals = _signal_rows(enriched)
    horizons = _fixed_horizon_audit(enriched)

    summary = {
        "schema_version": 2,
        "request_id": REQUEST_ID,
        "source_request": 163,
        "execution_mode": "reuse_stored_first_hot",
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
