from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import load_settings
from .massive_client import MassiveClient


SAMPLE_PER_MONTH = 12
WINDOW_BEFORE_MS = 5_000
WINDOW_AFTER_MS = 1_000


def _valid_quotes(rows: list[dict]) -> list[dict]:
    valid: list[dict] = []
    for row in rows:
        bid = pd.to_numeric(pd.Series([row.get("bid_price")]), errors="coerce").iloc[0]
        ask = pd.to_numeric(pd.Series([row.get("ask_price")]), errors="coerce").iloc[0]
        ts = pd.to_numeric(pd.Series([row.get("sip_timestamp")]), errors="coerce").iloc[0]
        if (
            pd.notna(bid)
            and pd.notna(ask)
            and pd.notna(ts)
            and float(bid) > 0
            and float(ask) > 0
            and float(ask) >= float(bid)
        ):
            item = dict(row)
            item["bid_price"] = float(bid)
            item["ask_price"] = float(ask)
            item["sip_timestamp"] = int(ts)
            valid.append(item)
    return valid


def _quote_near(
    client: MassiveClient,
    ticker: str,
    timestamp_ms: int,
) -> dict[str, float | int | str | None]:
    target_ns = int(timestamp_ms) * 1_000_000
    start_ns = int(timestamp_ms - WINDOW_BEFORE_MS) * 1_000_000
    end_ns = int(timestamp_ms + WINDOW_AFTER_MS) * 1_000_000
    rows = client.quotes(
        ticker,
        timestamp_gte=start_ns,
        timestamp_lte=end_ns,
        limit=50_000,
        order="asc",
    )
    valid = _valid_quotes(rows)
    if not valid:
        return {
            "quote_found": False,
            "quote_side": None,
            "quote_age_ms": None,
            "bid": None,
            "ask": None,
            "mid": None,
            "spread_pct": None,
            "zero_move_crossing_return_pct": None,
        }

    before = [row for row in valid if int(row["sip_timestamp"]) <= target_ns]
    if before:
        chosen = max(before, key=lambda row: int(row["sip_timestamp"]))
        side = "before"
    else:
        chosen = min(valid, key=lambda row: int(row["sip_timestamp"]))
        side = "after"

    bid = float(chosen["bid_price"])
    ask = float(chosen["ask_price"])
    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid * 100.0 if mid > 0 else np.nan
    crossing = (bid / ask - 1.0) * 100.0 if ask > 0 else np.nan
    age_ms = (int(chosen["sip_timestamp"]) - target_ns) / 1_000_000.0

    return {
        "quote_found": True,
        "quote_side": side,
        "quote_age_ms": float(age_ms),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread_pct": float(spread_pct),
        "zero_move_crossing_return_pct": float(crossing),
    }


def _stratified_sample(trades: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month, group in trades.groupby("month", sort=True):
        group = group.sort_values(
            ["trading_day", "entry_t", "ticker"], kind="stable"
        ).reset_index(drop=True)
        if len(group) > SAMPLE_PER_MONTH:
            idx = np.linspace(0, len(group) - 1, SAMPLE_PER_MONTH, dtype=int)
            group = group.iloc[idx].copy()
        pieces.append(group)
    if not pieces:
        return trades.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def run_probe(
    trades: pd.DataFrame,
    client: MassiveClient,
) -> tuple[pd.DataFrame, str | None]:
    required = {
        "month",
        "trading_day",
        "ticker",
        "entry_t",
        "exit_t",
        "entry_reference_price",
        "exit_reference_price",
        "gross_return_pct",
        "base_net_return_pct",
    }
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"dynamic trades missing NBBO probe columns: {sorted(missing)}")

    sample = _stratified_sample(trades)
    rows: list[dict[str, object]] = []

    for _, trade in sample.iterrows():
        ticker = str(trade["ticker"]).upper()
        entry_t = int(trade["entry_t"])
        exit_t = int(trade["exit_t"])

        try:
            entry = _quote_near(client, ticker, entry_t)
            exit_quote = _quote_near(client, ticker, exit_t)
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {401, 403}:
                return pd.DataFrame(rows), f"NBBO REST authorization failed with HTTP {status}"
            raise

        nbbo_return = np.nan
        if entry["quote_found"] and exit_quote["quote_found"]:
            ask = float(entry["ask"])
            bid = float(exit_quote["bid"])
            if ask > 0:
                nbbo_return = (bid / ask - 1.0) * 100.0

        bar_gross = float(trade["gross_return_pct"])
        bar_base = float(trade["base_net_return_pct"])
        entry_reference = float(trade["entry_reference_price"])
        exit_reference = float(trade["exit_reference_price"])

        rows.append(
            {
                "month": str(trade["month"]),
                "trading_day": str(trade["trading_day"]),
                "ticker": ticker,
                "entry_t": entry_t,
                "exit_t": exit_t,
                "exit_reason": trade.get("exit_reason"),
                "entry_reference_price": entry_reference,
                "exit_reference_price": exit_reference,
                "bar_gross_return_pct": bar_gross,
                "bar_base_net_return_pct": bar_base,
                "entry_quote_found": bool(entry["quote_found"]),
                "entry_quote_side": entry["quote_side"],
                "entry_quote_age_ms": entry["quote_age_ms"],
                "entry_bid": entry["bid"],
                "entry_ask": entry["ask"],
                "entry_mid": entry["mid"],
                "entry_spread_pct": entry["spread_pct"],
                "entry_zero_move_crossing_return_pct": entry[
                    "zero_move_crossing_return_pct"
                ],
                "exit_quote_found": bool(exit_quote["quote_found"]),
                "exit_quote_side": exit_quote["quote_side"],
                "exit_quote_age_ms": exit_quote["quote_age_ms"],
                "exit_bid": exit_quote["bid"],
                "exit_ask": exit_quote["ask"],
                "exit_mid": exit_quote["mid"],
                "exit_spread_pct": exit_quote["spread_pct"],
                "nbbo_ask_to_bid_return_pct": nbbo_return,
                "nbbo_minus_bar_gross_pct": (
                    float(nbbo_return - bar_gross)
                    if np.isfinite(nbbo_return)
                    else np.nan
                ),
                "entry_mid_vs_bar_open_pct": (
                    (float(entry["mid"]) / entry_reference - 1.0) * 100.0
                    if entry["quote_found"] and entry_reference > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows), None


def _q(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else np.nan


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    buckets = pd.cut(
        pd.to_numeric(frame["entry_mid"], errors="coerce"),
        bins=[0, 2, 5, 10, np.inf],
        labels=["<$2", "$2-$5", "$5-$10", "$10+"],
        right=False,
    )
    work = frame.copy()
    work["price_bucket"] = buckets

    rows: list[dict[str, object]] = []
    groupers = [("overall", work)]
    groupers.extend(
        (f"month:{month}", group)
        for month, group in work.groupby("month", sort=True)
    )
    groupers.extend(
        (f"price:{bucket}", group)
        for bucket, group in work.groupby("price_bucket", observed=True, sort=True)
    )

    for label, group in groupers:
        both = group["entry_quote_found"].astype(bool) & group["exit_quote_found"].astype(bool)
        rows.append(
            {
                "group": label,
                "sampled_trades": int(len(group)),
                "entry_quote_coverage": float(group["entry_quote_found"].mean()),
                "exit_quote_coverage": float(group["exit_quote_found"].mean()),
                "both_quote_coverage": float(both.mean()),
                "median_entry_spread_pct": _q(group["entry_spread_pct"], 0.50),
                "p75_entry_spread_pct": _q(group["entry_spread_pct"], 0.75),
                "p90_entry_spread_pct": _q(group["entry_spread_pct"], 0.90),
                "median_exit_spread_pct": _q(group["exit_spread_pct"], 0.50),
                "median_zero_move_crossing_return_pct": _q(
                    group["entry_zero_move_crossing_return_pct"], 0.50
                ),
                "median_bar_gross_return_pct": _q(group["bar_gross_return_pct"], 0.50),
                "mean_bar_gross_return_pct": float(
                    pd.to_numeric(group["bar_gross_return_pct"], errors="coerce").mean()
                ),
                "mean_bar_base_net_return_pct": float(
                    pd.to_numeric(group["bar_base_net_return_pct"], errors="coerce").mean()
                ),
                "mean_nbbo_ask_to_bid_return_pct": float(
                    pd.to_numeric(
                        group.loc[both, "nbbo_ask_to_bid_return_pct"], errors="coerce"
                    ).mean()
                )
                if both.any()
                else np.nan,
                "median_nbbo_minus_bar_gross_pct": _q(
                    group.loc[both, "nbbo_minus_bar_gross_pct"], 0.50
                )
                if both.any()
                else np.nan,
                "median_abs_quote_age_ms": _q(
                    pd.concat(
                        [
                            pd.to_numeric(group["entry_quote_age_ms"], errors="coerce").abs(),
                            pd.to_numeric(group["exit_quote_age_ms"], errors="coerce").abs(),
                        ],
                        ignore_index=True,
                    ),
                    0.50,
                ),
            }
        )
    return pd.DataFrame(rows)


def render_report(
    details: pd.DataFrame,
    *,
    authorization_error: str | None,
    client: MassiveClient,
) -> str:
    if authorization_error is not None:
        return "\n".join(
            [
                "=== MoneyMaker Historical NBBO Probe ===",
                "status=UNAVAILABLE",
                f"reason={authorization_error}",
                f"Massive_client_stats={client.stats.to_dict()}",
                "No execution conclusion was drawn from missing quote entitlement.",
            ]
        )

    summary = _summary(details)
    return "\n".join(
        [
            "=== MoneyMaker Historical NBBO Probe ===",
            "source=Massive /v3/quotes/{ticker} historical NBBO",
            f"sampled_trades={len(details)}",
            "sampling=12 trades per held-out month, evenly spread through each month",
            "quote_selection=last valid SIP quote within 5s before target; otherwise first valid quote within 1s after target",
            "nbbo_trade_return=entry ask to exit bid; excludes commissions, fees and market impact",
            f"Massive_client_stats={client.stats.to_dict()}",
            "",
            "=== Coverage / spread summary ===",
            summary.to_string(index=False),
            "",
            "=== Sample details ===",
            details.to_string(index=False),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.execution_nbbo_probe"
    )
    parser.add_argument("trades_csv", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/massive-rest-nbbo"),
    )
    args = parser.parse_args()

    trades = pd.read_csv(args.trades_csv)
    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=args.cache_dir,
        request_interval_seconds=0.05,
    )
    details, authorization_error = run_probe(trades, client)
    report = render_report(
        details,
        authorization_error=authorization_error,
        client=client,
    )
    print(report)

    for path in (args.report, args.details_csv, args.summary_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    _summary(details).to_csv(args.summary_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
