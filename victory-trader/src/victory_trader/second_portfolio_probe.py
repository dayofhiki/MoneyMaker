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
DEVELOPMENT_DAYS = (
    "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26", "2026-06-29",
)
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
    if day not in DEVELOPMENT_DAYS:
        raise ValueError("this diagnostic is restricted to already-open development days")
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
    missing_prior_minute = 0
    invalid_prior_close = 0
    after_session = 0
    for row in episodes.itertuples(index=False):
        ticker = str(row.ticker).upper()
        decision_t = int(row.hot_t)
        reference = closes.get((ticker, decision_t - MINUTE_MS))
        if reference is None:
            missing_prior_minute += 1
            continue
        if not isfinite(reference) or reference <= 0:
            invalid_prior_close += 1
            continue
        if decision_t >= session_close_t:
            after_session += 1
            continue
        attempts.append(EntryAttempt(ticker, decision_t, reference, session_close_t, RULE))
    return attempts, {
        "episodes": len(episodes),
        "causal_decision_refs": len(attempts),
        "missing_decision_refs": (
            missing_prior_minute + invalid_prior_close + after_session
        ),
        "missing_prior_minute": missing_prior_minute,
        "invalid_prior_close": invalid_prior_close,
        "after_session": after_session,
    }


def halt_intervals_for(
    ticker: str,
    records: list[HaltRecord],
) -> list[HaltInterval]:
    return [
        HaltInterval(record.halt_at_ms, record.resume_at_ms)
        for record in records if record.symbol == ticker
    ]


def load_offline_halts(path: Path, *, day: str) -> dict[str, list[HaltInterval]]:
    """Read an explicit halt manifest without treating omissions as official no-halts."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("day") != day or not isinstance(payload.get("intervals"), list):
        raise ValueError("offline halt manifest day or intervals is invalid")
    intervals: dict[str, list[HaltInterval]] = {}
    for row in payload["intervals"]:
        ticker = str(row["ticker"]).upper()
        start_t = int(row["start_t"])
        resume_t = row.get("resume_t")
        intervals.setdefault(ticker, []).append(HaltInterval(
            start_t, int(resume_t) if resume_t is not None else None,
        ))
    return intervals


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


def run_probe(
    fresh_dir: Path,
    output: Path,
    *,
    second_bars_dir: Path | None = None,
    halt_json: Path | None = None,
    day_text: str = DEVELOPMENT_DAY,
) -> dict[str, object]:
    if halt_json is not None and second_bars_dir is None:
        raise ValueError("offline halt manifest requires offline second bars")
    if day_text not in DEVELOPMENT_DAYS:
        raise ValueError("probe day must be already-open development data")
    day = date.fromisoformat(day_text)
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
        positions, scan, day=day_text, session_close_t=close_t,
    )

    halt_records: list[HaltRecord] = []
    offline_halts: dict[str, list[HaltInterval]] = {}
    client: MassiveClient | None = None
    if second_bars_dir is not None:
        source = "offline_parquet"
        if halt_json is None:
            halt_status = "unavailable:offline_no_halt_manifest"
        else:
            offline_halts = load_offline_halts(halt_json, day=day_text)
            halt_status = "user_supplied_unverified"
    else:
        source = "massive_rest"
        try:
            halt_records = fetch_nasdaq_halts(day)
            halt_status = "available"
        except (requests.RequestException, ElementTree.ParseError, ValueError) as exc:
            # Missing official feed is reported, never assumed to mean no halts.
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
        if second_bars_dir is not None:
            if not ticker or ".." in ticker or any(
                not (char.isalnum() or char in "._-") for char in ticker
            ):
                raise ValueError("unsafe ticker in offline second path")
            second_path = second_bars_dir / f"{day_text}-{ticker}-seconds.parquet"
            if not second_path.is_file():
                data_audit[ticker] = {"status": "missing_offline_second_file"}
                continue
            raw = pd.read_parquet(second_path)
            ticker_halts = offline_halts.get(ticker, [])
        else:
            assert client is not None
            payload = client.second_bars_range(ticker, day, day, adjusted=False)
            raw = pd.DataFrame(payload.get("results") or [])
            ticker_halts = halt_intervals_for(ticker, halt_records)
        if raw.empty:
            data_audit[ticker] = {"status": "no_second_bars"}
            continue
        try:
            adapted = adapt_second_bars(
                raw,
                session_open_t=open_t,
                session_close_t=close_t,
                halts=ticker_halts,
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
        "day": day_text,
        "second_source": source,
        "policy": "all causal-reference HOT episodes; stop3/take10-half/trail7%-price",
        "halt_feed": halt_status,
        "selection": selection,
        "data_statuses": dict(Counter(str(item["status"]) for item in data_audit.values())),
        "data_audit": data_audit,
        "observed_second_rows": {
            "provider": sum(int(item.get("provider_rows", 0)) for item in data_audit.values()),
            "regular_session": sum(int(item.get("session_rows", 0)) for item in data_audit.values()),
            "halt_overlap": sum(int(item.get("halt_overlap_rows", 0)) for item in data_audit.values()),
        },
        "api_stats": client.stats.to_dict() if client is not None else None,
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
        "day": day_text,
        "selection": selection,
        "halt_feed": halt_status,
        "data_statuses": report["data_statuses"],
        "observed_second_rows": report["observed_second_rows"],
        "scenarios": report["scenarios"],
    }, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--day", choices=DEVELOPMENT_DAYS, default=DEVELOPMENT_DAY)
    parser.add_argument("--second-bars-dir", type=Path)
    parser.add_argument("--halt-json", type=Path)
    args = parser.parse_args()
    run_probe(
        args.fresh_dir, args.output,
        second_bars_dir=args.second_bars_dir, halt_json=args.halt_json,
        day_text=args.day,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

