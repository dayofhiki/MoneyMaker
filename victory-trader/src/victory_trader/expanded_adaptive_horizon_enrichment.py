from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HORIZONS = (1, 2, 5, 10, 15, 30)
JOIN_KEYS = ("trading_day", "ticker", "t")


def return_column(horizon: int, scenario: str) -> str:
    if scenario == "gross":
        return f"buy_return_{horizon}m_pct"
    return f"buy_return_{horizon}m_{scenario}_net_return_pct"


def return_columns() -> tuple[str, ...]:
    return tuple(
        return_column(horizon, scenario)
        for horizon in HORIZONS
        for scenario in ("gross", "base", "stress")
    )


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["trading_day"] = result["trading_day"].astype(str)
    result["ticker"] = result["ticker"].astype(str)
    result["t"] = pd.to_numeric(result["t"], errors="raise").astype("int64")
    return result


def attach_labels(anchors: pd.DataFrame, state_labels: pd.DataFrame) -> pd.DataFrame:
    anchors = _normalize_keys(anchors)
    state_labels = _normalize_keys(state_labels)
    if state_labels.duplicated(list(JOIN_KEYS)).any():
        raise ValueError("duplicate state-panel keys in adaptive label source")

    renamed = state_labels.rename(
        columns={column: f"{column}__state" for column in return_columns()}
    )
    merged = anchors.merge(
        renamed,
        how="left",
        on=list(JOIN_KEYS),
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        examples = merged.loc[unmatched, list(JOIN_KEYS)].head(10).to_dict("records")
        raise ValueError(f"adaptive label join missed anchors: {examples}")
    merged = merged.drop(columns="_merge")

    for column in return_columns():
        state_column = f"{column}__state"
        state_value = pd.to_numeric(merged[state_column], errors="coerce")
        if column in anchors.columns:
            existing = pd.to_numeric(merged[column], errors="coerce")
            common = existing.notna() & state_value.notna()
            if common.any():
                delta = (existing.loc[common] - state_value.loc[common]).abs()
                if float(delta.max()) > 1e-9:
                    raise ValueError(
                        f"frozen label mismatch for {column}: max delta={float(delta.max())}"
                    )
        else:
            merged[column] = state_value
        merged = merged.drop(columns=state_column)
    return merged


def enrich_month(anchor_path: Path, state_path: Path) -> pd.DataFrame:
    anchors = pd.read_parquet(anchor_path)
    available = set(pq.ParquetFile(state_path).schema.names)
    required = set(JOIN_KEYS) | set(return_columns())
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"state panel {state_path} missing columns: {missing}")
    labels = pd.read_parquet(
        state_path,
        columns=[*JOIN_KEYS, *return_columns()],
    )
    return attach_labels(anchors, labels)


def coverage_row(frame: pd.DataFrame, month: str) -> dict[str, object]:
    row: dict[str, object] = {
        "month": month,
        "anchors": int(len(frame)),
    }
    for horizon in HORIZONS:
        base = pd.to_numeric(
            frame[return_column(horizon, "base")], errors="coerce"
        )
        row[f"base_{horizon}m_label_coverage"] = float(base.notna().mean())
    return row


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.expanded_adaptive_horizon_enrichment"
    )
    parser.add_argument("--anchor", action="append", type=_parse_labeled_path, required=True)
    parser.add_argument("--state", action="append", type=_parse_labeled_path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    anchors = dict(args.anchor)
    states = dict(args.state)
    if set(anchors) != set(states):
        raise ValueError(
            f"anchor/state month mismatch: anchors={sorted(anchors)}, states={sorted(states)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage: list[dict[str, object]] = []
    for month in sorted(anchors):
        enriched = enrich_month(anchors[month], states[month])
        output = args.output_dir / f"adaptive-anchors-{month}.parquet"
        enriched.to_parquet(output, index=False, compression="zstd")
        coverage.append(coverage_row(enriched, month))
        print(
            "adaptive label enrichment",
            {"month": month, "anchors": len(enriched), "output": str(output)},
            flush=True,
        )

    coverage_frame = pd.DataFrame(coverage)
    report = "\n".join(
        [
            "=== MoneyMaker Adaptive Multi-Horizon Label Enrichment v3.2 ===",
            "source=exact frozen v3.0 semantic anchors + original frozen state-panel outcomes",
            "horizons=1,2,5,10,15,30 minutes",
            "",
            coverage_frame.to_string(index=False),
        ]
    )
    print(report, flush=True)
    args.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    coverage_frame.to_csv(args.coverage_csv, index=False)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
