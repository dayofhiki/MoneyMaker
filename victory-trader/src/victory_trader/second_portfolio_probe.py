"""Development-only second-path account diagnostic for an already-open day."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from math import isfinite
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests

from .causal_second_execution import RiskRule, SecondBar
from .config import load_settings
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS
from .halts import HaltRecord, fetch_nasdaq_halts
from .market_calendar import regular_session_bounds
from .massive_client import MassiveClient
from .second_path_adapter import HaltInterval, adapt_second_bars
from .second_portfolio_replay import EntryAttempt, replay_portfolio

DEVELOPMENT_DAY = "2026-06-23"
MINUTE_MS = 60_000
RULE = RiskRule(stop_pct=3, take_pct=10, trail_retrace_pct=7)
LATENCIES_MS = (0, 1_000, 2_000, 5_000)


def build_attempts(
    positions: pd.DataFrame,
    scan: pd.DataFrame,
    *,
    day: str,
    session_close_t: int,
) -> tuple[list[EntryAttempt], dict[str, int]]:
    """Use only the exact previous completed minute close as sizing reference."""
    if day != DEVELOPMENT_DAY:
        raise ValueError("this diagnostic is restricted to the already-open development day")
    required_positions = {"trading_day", "ticker", "hot_t"}
    required_scan = {"trading_day", "ticker", "t", "c"}
    if not required_positions.issubset(positions.columns):
        raise ValueError("positions missing episode keys")
    if not required_scan.issubset(scan.columns):
        raise ValueError("scan missing minute close keys")
    episodes = positions.loc[
        positions["trading_day"].astype(str).eq(day),
        ["trading_day", "ticker", "hot_t"],
    ].drop_duplicates()
    episodes = episodes.sort_values(["hot_t", "ticker"], kind="stable")
    minute_rows = scan.loc[
        scan["trading_day"].astype(str).eq(day),
        ["ticker", "t", "c"],
    ].copy()
    minute_rows["ticker"] = minute_rows["ticker"].astype(str).str.upper()
    if minute_rows.duplicated(["ticker", "t"]).any():
        raise ValueError("duplicate scan minute prevents causal decision reference")
    closes = {
        (str(row.ticker), int(row.t)): float(row.c)
        for row in minute_rows.itertuples(index=False)
        if pd.notna(row.c)
    }
    attempts: list[EntryAttempt] = []
    missing = 0
    for row in episodes.itertuples(index=False):
        ticker = str(row.ticker).upper()
        decision_t = int(row.hot_t)
        reference = closes.get((ticker, decision_t - MINUTE_MS))
        if reference is None or not isfinite(reference) or reference <= 0:
            missing += 1
            continue
        if decision_t >= session_close_t:
            missing += 1
            continue
        attempts.append(EntryAttempt(ticker, decision_t, reference, session_close_t, RULE))
    return attempts, {
        "episodes": len(episodes),
        "causal_decision_refs": len(attempts),
        "missing_decision_refs": missing,
    }


def halt_intervals_for(
    ticker: str,
    records: list[HaltRecord],
) -> list[HaltInterval]:
    return [
        HaltInterval(record.halt_at_ms, record.resume_at_ms)
        for record in records if record.symbol == ticker
    ]


def summarize_scenarios(
    attempts: list[EntryAttempt],
    bars_by_ticker: dict[str, list[SecondBar]],
) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    for scenario in DEFAULT_EXECUTION_SCENARIOS:
        for latency in LATENCIES_MS:
            replay = replay_portfolio(
                attempts, bars_by_ticker, scenario,
                initial_cash=10_000.0,
                max_positions=5,
                risk_fraction=0.01,
                max_allocation_fraction=0.20,
                latency_ms=latency,
            )
            outputs.append({
                "scenario": scenario.name,
                "latency_ms": latency,
                "summary": replay.summary,
                "statuses": dict(Counter(str(row["status"]) for row in replay.records)),
            })
    return outputs


def run_probe(fresh_dir: Path, output: Path) -> dict[str, object]:
    day = date.fromisoformat(DEVELOPMENT_DAY)
    bounds = regular_session_bounds(day)
    if bounds is None:
        raise ValueError("missing official session bounds")
    open_t, close_t = (int(value.timestamp() * 1000) for value in bounds)
    position_files = list(fresh_dir.glob("*-positions.parquet"))
    scan_files = list(fresh_dir.glob("*-scan.parquet"))
    if len(position_files) != 1 or len(scan_files) != 1:
        raise ValueError("expected exactly one prepared position and scan file")
    positions = pd.read_parquet(position_files[0])
    scan = pd.read_parquet(scan_files[0])
    attempts, selection = build_attempts(
        positions, scan, day=DEVELOPMENT_DAY, session_close_t=close_t,
    )

    try:
        halt_records = fetch_nasdaq_halts(day)
        halt_status = "available"
    except (requests.RequestException, ElementTree.ParseError, ValueError) as exc:
        # Missing official feed is reported, never assumed to mean no halts.
        halt_records = []
        halt_status = f"unavailable:{type(exc).__name__}"

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive-second-attention"),
        request_interval_seconds=0.2,
    )
    bars_by_ticker: dict[str, list[SecondBar]] = {}
    data_audit: dict[str, object] = {}
    for ticker in sorted({attempt.ticker for attempt in attempts}):
        payload = client.second_bars_range(ticker, day, day, adjusted=False)
        raw = pd.DataFrame(payload.get("results") or [])
        if raw.empty:
            data_audit[ticker] = {"status": "no_second_bars"}
            continue
        try:
            adapted = adapt_second_bars(
                raw,
                session_open_t=open_t,
                session_close_t=close_t,
                halts=halt_intervals_for(ticker, halt_records),
            )
        except ValueError as exc:
            data_audit[ticker] = {
                "status": "rejected_second_path",
                "reason": str(exc),
            }
            continue
        bars_by_ticker[ticker] = list(adapted.bars)
        data_audit[ticker] = {
            "status": "adapted",
            "provider_rows": adapted.provider_rows,
            "session_rows": adapted.session_rows,
            "halt_overlap_rows": adapted.halt_overlap_rows,
        }

    report: dict[str, object] = {
        "development_only": True,
        "day": DEVELOPMENT_DAY,
        "policy": "all causal-reference HOT episodes; stop3/take10-half/trail7%-price",
        "halt_feed": halt_status,
        "selection": selection,
        "data_statuses": dict(Counter(str(item["status"]) for item in data_audit.values())),
        "data_audit": data_audit,
        "api_stats": client.stats.to_dict(),
        "scenarios": summarize_scenarios(attempts, bars_by_ticker),
        "interpretation": (
            "Seen-day implementation diagnostic only. Synthetic second-bar opens are "
            "not observed brokerage fills; missing/ambiguous paths and unknown halt "
            "coverage prohibit promotion or a profitability claim."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "day": DEVELOPMENT_DAY,
        "selection": selection,
        "halt_feed": halt_status,
        "data_statuses": report["data_statuses"],
        "scenarios": report["scenarios"],
    }, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_probe(args.fresh_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

