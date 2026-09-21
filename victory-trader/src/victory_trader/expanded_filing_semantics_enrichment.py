from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from victory_trader.config import load_settings
from victory_trader.massive_client import MassiveClient


LOOKBACK_DAYS = 30
RECENT_DAYS = 7

# Fixed before the v3.0 return evaluation.  These are economic-mechanism
# groupings, not categories chosen from January-March returns.
EQUITY_SUPPLY_TERTIARY = frozenset(
    {
        "public_offering",
        "private_placement",
        "underwriting_agreement",
        "acquisition_consideration_shares",
    }
)
OPERATING_CATALYST_TERTIARY = frozenset(
    {
        "clinical_trial_results",
        "preliminary_results",
        "quarterly_earnings",
        "guidance_issuance_or_update",
        "partnership_or_collaboration",
        "licensing_agreement",
        "strategic_initiative",
    }
)
ADVERSE_TERTIARY = frozenset(
    {
        "material_litigation",
        "executive_officer_departure",
        "cfo_departure",
        "director_departure",
    }
)

SEMANTIC_COLUMNS = (
    "filing_semantics_query_success",
    "filing_semantics_strict_prior",
    "filing_equity_supply_accessions_30d",
    "filing_equity_supply_accessions_7d",
    "filing_debt_accessions_30d",
    "filing_debt_accessions_7d",
    "filing_operating_catalyst_accessions_30d",
    "filing_operating_catalyst_accessions_7d",
    "filing_adverse_accessions_30d",
    "filing_adverse_accessions_7d",
    "filing_semantic_accessions_30d",
    "filing_latest_semantic_age_days",
)


def _filing_day(row: dict[str, object]) -> date | None:
    value = row.get("filing_date")
    if not value:
        return None
    return pd.Timestamp(str(value)).date()


def fetch_history(
    frames: dict[str, pd.DataFrame], client: MassiveClient
) -> tuple[dict[str, list[dict[str, object]]], set[str]]:
    tickers: set[str] = set()
    days: list[date] = []
    for frame in frames.values():
        tickers.update(frame["ticker"].astype(str).str.upper().unique())
        days.extend(pd.to_datetime(frame["trading_day"]).dt.date.unique())
    if not days:
        return {}, set()

    start = min(days) - timedelta(days=LOOKBACK_DAYS)
    end = max(days) - timedelta(days=1)
    rows = client.eight_k_disclosures_market(
        filing_date_gte=start, filing_date_lte=end
    )
    history: dict[str, list[dict[str, object]]] = defaultdict(list)
    for raw in rows:
        filing_day = _filing_day(raw)
        if filing_day is None:
            continue
        if not (start <= filing_day <= end):
            raise ValueError("8-K semantic global request date violation")
        for ticker_value in raw.get("tickers") or []:
            ticker = str(ticker_value).upper()
            if ticker not in tickers:
                continue
            item = dict(raw)
            item["_filing_day"] = filing_day
            history[ticker].append(item)
    for ticker in history:
        history[ticker].sort(key=lambda row: row["_filing_day"])
    return dict(history), tickers


def _accessions(
    rows: list[dict[str, object]],
    *,
    secondary: str | None = None,
    tertiary: frozenset[str] | None = None,
) -> set[str]:
    selected: set[str] = set()
    for row in rows:
        if secondary is not None and str(row.get("secondary_category") or "") != secondary:
            continue
        if tertiary is not None and str(row.get("tertiary_category") or "") not in tertiary:
            continue
        accession = str(row.get("accession_number") or "")
        if accession:
            selected.add(accession)
    return selected


def semantic_features(
    records: list[dict[str, object]], event_day: date
) -> dict[str, float]:
    start = event_day - timedelta(days=LOOKBACK_DAYS)
    prior = [row for row in records if start <= row["_filing_day"] < event_day]
    if any(row["_filing_day"] >= event_day for row in prior):
        raise ValueError("8-K semantic point-in-time violation")
    recent = [
        row for row in prior if (event_day - row["_filing_day"]).days <= RECENT_DAYS
    ]

    supply = _accessions(prior, tertiary=EQUITY_SUPPLY_TERTIARY)
    supply_recent = _accessions(recent, tertiary=EQUITY_SUPPLY_TERTIARY)
    debt = _accessions(prior, secondary="debt_activity")
    debt_recent = _accessions(recent, secondary="debt_activity")
    operating = _accessions(prior, tertiary=OPERATING_CATALYST_TERTIARY)
    operating_recent = _accessions(recent, tertiary=OPERATING_CATALYST_TERTIARY)
    adverse = _accessions(prior, tertiary=ADVERSE_TERTIARY)
    adverse_recent = _accessions(recent, tertiary=ADVERSE_TERTIARY)
    semantic_rows = [
        row
        for row in prior
        if str(row.get("secondary_category") or "") == "debt_activity"
        or str(row.get("tertiary_category") or "")
        in EQUITY_SUPPLY_TERTIARY | OPERATING_CATALYST_TERTIARY | ADVERSE_TERTIARY
    ]
    all_accessions = _accessions(semantic_rows)
    latest = max((row["_filing_day"] for row in semantic_rows), default=None)
    return {
        "filing_semantics_query_success": 1.0,
        "filing_semantics_strict_prior": 1.0,
        "filing_equity_supply_accessions_30d": float(len(supply)),
        "filing_equity_supply_accessions_7d": float(len(supply_recent)),
        "filing_debt_accessions_30d": float(len(debt)),
        "filing_debt_accessions_7d": float(len(debt_recent)),
        "filing_operating_catalyst_accessions_30d": float(len(operating)),
        "filing_operating_catalyst_accessions_7d": float(len(operating_recent)),
        "filing_adverse_accessions_30d": float(len(adverse)),
        "filing_adverse_accessions_7d": float(len(adverse_recent)),
        "filing_semantic_accessions_30d": float(len(all_accessions)),
        "filing_latest_semantic_age_days": (
            float((event_day - latest).days) if latest is not None else np.nan
        ),
    }


def enrich_frame(
    frame: pd.DataFrame,
    history: dict[str, list[dict[str, object]]],
    queried: set[str],
) -> pd.DataFrame:
    result = frame.copy()
    keys = result[["ticker", "trading_day"]].drop_duplicates().copy()
    rows: list[dict[str, object]] = []
    for item in keys.itertuples(index=False):
        ticker = str(item.ticker).upper()
        trading_day = str(item.trading_day)
        if ticker not in queried:
            raise ValueError(f"8-K semantics query missing for ticker: {ticker}")
        rows.append(
            {
                "ticker": ticker,
                "trading_day": trading_day,
                **semantic_features(
                    history.get(ticker, []), pd.Timestamp(trading_day).date()
                ),
            }
        )
    features = pd.DataFrame(rows)
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result["trading_day"] = result["trading_day"].astype(str)
    return result.merge(
        features, on=["ticker", "trading_day"], how="left", validate="many_to_one"
    )


def coverage_row(month: str, frame: pd.DataFrame) -> dict[str, object]:
    unique = frame.drop_duplicates(["ticker", "trading_day"])
    return {
        "month": month,
        "ticker_days": int(len(unique)),
        "query_success": float(unique["filing_semantics_query_success"].eq(1).mean()),
        "strict_prior": bool(unique["filing_semantics_strict_prior"].eq(1).all()),
        "semantic_30d_rate": float(unique["filing_semantic_accessions_30d"].gt(0).mean()),
        "equity_supply_30d_rate": float(unique["filing_equity_supply_accessions_30d"].gt(0).mean()),
        "debt_30d_rate": float(unique["filing_debt_accessions_30d"].gt(0).mean()),
        "operating_30d_rate": float(unique["filing_operating_catalyst_accessions_30d"].gt(0).mean()),
        "adverse_30d_rate": float(unique["filing_adverse_accessions_30d"].gt(0).mean()),
    }


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/massive-rest"))
    args = parser.parse_args()

    frames = {label: pd.read_parquet(path) for label, path in args.dataset}
    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=args.cache_dir,
        request_interval_seconds=0.05,
    )
    history, queried = fetch_history(frames, client)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = []
    for label, frame in frames.items():
        enriched = enrich_frame(frame, history, queried)
        enriched.to_parquet(
            args.output_dir / f"semantic-anchors-{label}.parquet",
            index=False,
            compression="zstd",
        )
        coverage.append(coverage_row(label, enriched))
    coverage_frame = pd.DataFrame(coverage)
    report = "\n".join(
        [
            "=== MoneyMaker strictly-prior 8-K filing semantics v3.0 ===",
            "rule=D-30 <= filing_date < D; accessions deduplicated per semantic bucket",
            f"client_stats={client.stats.to_dict()}",
            "",
            coverage_frame.to_string(index=False),
        ]
    )
    args.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    coverage_frame.to_csv(args.coverage_csv, index=False)
    args.report.write_text(report + "\n", encoding="utf-8")
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
