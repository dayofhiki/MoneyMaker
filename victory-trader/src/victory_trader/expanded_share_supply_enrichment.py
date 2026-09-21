from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings
from .expanded_episode_viability_rank import (
    _first_eligible_rows,
    read_model_panel,
)
from .expanded_execution_feasible_hurdle_ev import execution_feasible_mask
from .massive_client import MassiveClient


RAW_SUPPLY_COLUMNS = (
    "supply_weighted_shares_outstanding_prior",
    "supply_share_class_shares_outstanding_prior",
    "supply_provider_market_cap_prior",
)


def _positive_number(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or float(numeric) <= 0:
        return np.nan
    return float(numeric)


def enrich_anchors(
    anchors: pd.DataFrame,
    client: MassiveClient,
) -> pd.DataFrame:
    result = anchors.copy()
    result["supply_query_date"] = ""
    result["supply_query_success"] = 0.0
    result["supply_query_error"] = ""
    for column in RAW_SUPPLY_COLUMNS:
        result[column] = np.nan

    feasible = execution_feasible_mask(result)
    positions = np.flatnonzero(feasible.to_numpy(dtype=bool))

    for count, pos in enumerate(positions, start=1):
        row = result.iloc[pos]
        trading_day = pd.Timestamp(str(row["trading_day"])).date()
        query_day = trading_day - timedelta(days=1)
        if query_day >= trading_day:
            raise ValueError("share-supply query date is not strictly prior")

        ticker = str(row["ticker"]).upper()
        query_success = True
        error_type = ""
        payload: dict[str, object] = {}
        try:
            response = client.ticker_details(ticker, query_day)
            raw = response.get("results") or {}
            if isinstance(raw, dict):
                payload = raw
            else:
                query_success = False
                error_type = "missing_results"
        except Exception as exc:
            query_success = False
            error_type = type(exc).__name__

        index = result.index[pos]
        result.at[index, "supply_query_date"] = query_day.isoformat()
        result.at[index, "supply_query_success"] = float(query_success)
        result.at[index, "supply_query_error"] = error_type
        result.at[
            index,
            "supply_weighted_shares_outstanding_prior",
        ] = _positive_number(payload.get("weighted_shares_outstanding"))
        result.at[
            index,
            "supply_share_class_shares_outstanding_prior",
        ] = _positive_number(
            payload.get("share_class_shares_outstanding")
        )
        result.at[
            index,
            "supply_provider_market_cap_prior",
        ] = _positive_number(payload.get("market_cap"))

        if count % 250 == 0 or count == len(positions):
            print(
                "share-supply progress",
                {
                    "queried": count,
                    "total": len(positions),
                    "network_requests": client.stats.network_requests,
                    "cache_hits": client.stats.cache_hits,
                },
                flush=True,
            )

    return result


def coverage_row(frame: pd.DataFrame, month: str) -> dict[str, object]:
    feasible = frame.loc[execution_feasible_mask(frame)].copy()
    if feasible.empty:
        return {
            "month": month,
            "feasible_anchors": 0,
            "request_success": np.nan,
            "weighted_shares_coverage": np.nan,
            "share_class_coverage": np.nan,
            "provider_market_cap_coverage": np.nan,
            "strict_prior_query_dates": False,
        }

    query_date = pd.to_datetime(
        feasible["supply_query_date"], errors="coerce"
    )
    trading_day = pd.to_datetime(
        feasible["trading_day"], errors="coerce"
    )
    return {
        "month": month,
        "feasible_anchors": int(len(feasible)),
        "request_success": float(
            pd.to_numeric(
                feasible["supply_query_success"], errors="coerce"
            ).eq(1.0).mean()
        ),
        "weighted_shares_coverage": float(
            pd.to_numeric(
                feasible["supply_weighted_shares_outstanding_prior"],
                errors="coerce",
            ).notna().mean()
        ),
        "share_class_coverage": float(
            pd.to_numeric(
                feasible["supply_share_class_shares_outstanding_prior"],
                errors="coerce",
            ).notna().mean()
        ),
        "provider_market_cap_coverage": float(
            pd.to_numeric(
                feasible["supply_provider_market_cap_prior"],
                errors="coerce",
            ).notna().mean()
        ),
        "strict_prior_query_dates": bool(
            query_date.notna().all()
            and trading_day.notna().all()
            and query_date.lt(trading_day).all()
        ),
    }


def render_report(
    coverage: pd.DataFrame,
    client: MassiveClient,
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker Expanded-History Share-Supply Enrichment v2.7 ===",
            "source=Massive point-in-time ticker details at D-1 calendar day",
            "query_scope=execution-feasible first anchors only",
            "non-gated anchors retained without supply requests",
            "no current-date fallback",
            f"Massive_client_stats={client.stats.to_dict()}",
            "",
            coverage.to_string(index=False),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.expanded_share_supply_enrichment"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/massive-rest-v27-supply"),
    )
    args = parser.parse_args()

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=args.cache_dir,
        request_interval_seconds=0.05,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for month, path in args.dataset:
        panel = read_model_panel(path)
        anchors = _first_eligible_rows(panel)
        print(
            "enrich month",
            {
                "month": month,
                "first_anchors": len(anchors),
                "feasible": int(execution_feasible_mask(anchors).sum()),
            },
            flush=True,
        )
        enriched = enrich_anchors(anchors, client)
        enriched.to_parquet(
            args.output_dir / f"supply-anchors-{month}.parquet",
            index=False,
            compression="zstd",
        )
        rows.append(coverage_row(enriched, month))
        del panel, anchors, enriched

    coverage = pd.DataFrame(rows)
    report = render_report(coverage, client)
    print(report, flush=True)

    args.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.coverage_csv, index=False)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
