from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import load_settings
from .massive_client import MassiveClient


SAMPLE_PER_MONTH = 20
DECISION_LOOKBACK_MS = 5_000
FRESH_DECISION_MS = 2_000
EXECUTION_AFTER_MS = 1_000
AUDIT_NOTIONAL = 1_000.0
MIN_MONTH_COVERAGE = 0.80
MIN_DEPTH_SUFFICIENCY = 0.70
POLICY = "raw_same_fit_cap1"


def _valid_quotes(rows: list[dict]) -> list[dict]:
    valid: list[dict] = []
    for row in rows:
        bid = pd.to_numeric(
            pd.Series([row.get("bid_price")]), errors="coerce"
        ).iloc[0]
        ask = pd.to_numeric(
            pd.Series([row.get("ask_price")]), errors="coerce"
        ).iloc[0]
        bid_size = pd.to_numeric(
            pd.Series([row.get("bid_size")]), errors="coerce"
        ).iloc[0]
        ask_size = pd.to_numeric(
            pd.Series([row.get("ask_size")]), errors="coerce"
        ).iloc[0]
        ts = pd.to_numeric(
            pd.Series([row.get("sip_timestamp")]), errors="coerce"
        ).iloc[0]
        if (
            pd.notna(bid)
            and pd.notna(ask)
            and pd.notna(bid_size)
            and pd.notna(ask_size)
            and pd.notna(ts)
            and float(bid) > 0
            and float(ask) > 0
            and float(ask) >= float(bid)
            and float(bid_size) > 0
            and float(ask_size) > 0
        ):
            item = dict(row)
            item["bid_price"] = float(bid)
            item["ask_price"] = float(ask)
            item["bid_size"] = float(bid_size)
            item["ask_size"] = float(ask_size)
            item["sip_timestamp"] = int(ts)
            valid.append(item)
    return valid


def _quote_metrics(
    row: dict | None,
    target_ms: int,
) -> dict[str, float | int | bool | None]:
    if row is None:
        return {
            "found": False,
            "age_ms": None,
            "bid": None,
            "ask": None,
            "bid_size": None,
            "ask_size": None,
            "mid": None,
            "spread_pct": None,
            "bid_depth_dollars": None,
            "ask_depth_dollars": None,
            "size_imbalance": None,
            "microprice": None,
            "microprice_mid_pct": None,
        }

    bid = float(row["bid_price"])
    ask = float(row["ask_price"])
    bid_size = float(row["bid_size"])
    ask_size = float(row["ask_size"])
    mid = (bid + ask) / 2.0
    total_size = bid_size + ask_size
    microprice = (
        (ask * bid_size + bid * ask_size) / total_size
        if total_size > 0
        else np.nan
    )
    return {
        "found": True,
        "age_ms": float(
            (int(row["sip_timestamp"]) - int(target_ms) * 1_000_000)
            / 1_000_000.0
        ),
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "mid": mid,
        "spread_pct": (
            float((ask - bid) / mid * 100.0) if mid > 0 else np.nan
        ),
        "bid_depth_dollars": bid * bid_size,
        "ask_depth_dollars": ask * ask_size,
        "size_imbalance": (
            float((bid_size - ask_size) / total_size)
            if total_size > 0
            else np.nan
        ),
        "microprice": float(microprice),
        "microprice_mid_pct": (
            float((microprice / mid - 1.0) * 100.0)
            if mid > 0 and np.isfinite(microprice)
            else np.nan
        ),
    }


def _last_prior_quote(
    client: MassiveClient,
    ticker: str,
    target_ms: int,
) -> dict[str, float | int | bool | None]:
    rows = client.quotes(
        ticker,
        timestamp_gte=(target_ms - DECISION_LOOKBACK_MS) * 1_000_000,
        timestamp_lte=target_ms * 1_000_000,
        limit=50_000,
        order="asc",
    )
    valid = [
        row
        for row in _valid_quotes(rows)
        if int(row["sip_timestamp"]) <= target_ms * 1_000_000
    ]
    chosen = (
        max(valid, key=lambda row: int(row["sip_timestamp"]))
        if valid
        else None
    )
    return _quote_metrics(chosen, target_ms)


def _first_after_quote(
    client: MassiveClient,
    ticker: str,
    target_ms: int,
) -> dict[str, float | int | bool | None]:
    rows = client.quotes(
        ticker,
        timestamp_gte=target_ms * 1_000_000,
        timestamp_lte=(target_ms + EXECUTION_AFTER_MS) * 1_000_000,
        limit=50_000,
        order="asc",
    )
    valid = [
        row
        for row in _valid_quotes(rows)
        if int(row["sip_timestamp"]) >= target_ms * 1_000_000
    ]
    chosen = (
        min(valid, key=lambda row: int(row["sip_timestamp"]))
        if valid
        else None
    )
    return _quote_metrics(chosen, target_ms)


def _sample_attempts(attempts: pd.DataFrame) -> pd.DataFrame:
    work = attempts.loc[attempts["policy"].astype(str).eq(POLICY)].copy()
    pieces: list[pd.DataFrame] = []
    for month, group in work.groupby("month", sort=True):
        group = group.sort_values(
            ["trading_day", "decision_t", "ticker"], kind="stable"
        ).reset_index(drop=True)
        if len(group) > SAMPLE_PER_MONTH:
            index = np.linspace(
                0, len(group) - 1, SAMPLE_PER_MONTH, dtype=int
            )
            group = group.iloc[index].copy()
        pieces.append(group)
    if not pieces:
        return work.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def _prefix(
    prefix: str,
    quote: dict[str, float | int | bool | None],
) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in quote.items()}


def run_probe(
    attempts: pd.DataFrame,
    client: MassiveClient,
) -> tuple[pd.DataFrame, str | None]:
    required = {
        "month",
        "policy",
        "trading_day",
        "ticker",
        "decision_t",
        "entry_price",
        "buy_return_10m_pct",
        "buy_return_10m_base_net_return_pct",
    }
    missing = required - set(attempts.columns)
    if missing:
        raise ValueError(
            f"attempt ledger missing NBBO probe columns: {sorted(missing)}"
        )

    sample = _sample_attempts(attempts)
    rows: list[dict[str, object]] = []

    for attempt in sample.to_dict("records"):
        ticker = str(attempt["ticker"]).upper()
        decision_t = int(attempt["decision_t"])
        decision_known_t = decision_t + 60_000
        entry_target_t = decision_known_t
        exit_target_t = decision_t + 11 * 60_000

        try:
            decision = _last_prior_quote(
                client, ticker, decision_known_t
            )
            entry = _first_after_quote(
                client, ticker, entry_target_t
            )
            exit_quote = _first_after_quote(
                client, ticker, exit_target_t
            )
        except requests.HTTPError as exc:
            status = getattr(
                getattr(exc, "response", None), "status_code", None
            )
            if status in {401, 403}:
                return (
                    pd.DataFrame(rows),
                    f"historical NBBO authorization failed with HTTP {status}",
                )
            raise

        decision_fresh = bool(
            decision["found"]
            and decision["age_ms"] is not None
            and -FRESH_DECISION_MS
            <= float(decision["age_ms"])
            <= 0.0
        )
        entry_depth_ok = bool(
            entry["found"]
            and entry["ask_depth_dollars"] is not None
            and float(entry["ask_depth_dollars"]) >= AUDIT_NOTIONAL
        )
        exit_depth_ok = bool(
            exit_quote["found"]
            and exit_quote["bid_depth_dollars"] is not None
            and float(exit_quote["bid_depth_dollars"]) >= AUDIT_NOTIONAL
        )
        both_execution = bool(entry["found"] and exit_quote["found"])
        top_book_return = np.nan
        if both_execution:
            ask = float(entry["ask"])
            bid = float(exit_quote["bid"])
            if ask > 0:
                top_book_return = (bid / ask - 1.0) * 100.0

        row: dict[str, object] = {
            "month": str(attempt["month"]),
            "trading_day": str(attempt["trading_day"]),
            "ticker": ticker,
            "decision_t": decision_t,
            "decision_known_t": decision_known_t,
            "entry_target_t": entry_target_t,
            "exit_target_t": exit_target_t,
            "decision_fresh_2s": decision_fresh,
            "entry_depth_1000_ok": entry_depth_ok,
            "exit_depth_1000_ok": exit_depth_ok,
            "both_depth_1000_ok": bool(entry_depth_ok and exit_depth_ok),
            "both_execution_quotes": both_execution,
            "top_book_ask_to_bid_return_pct": top_book_return,
            "bar_entry_price": attempt.get("entry_price"),
            "bar_gross_return_pct": attempt.get("buy_return_10m_pct"),
            "bar_base_net_return_pct": attempt.get(
                "buy_return_10m_base_net_return_pct"
            ),
            "legacy_evaluated": attempt.get("legacy_evaluated"),
            "evaluation_reason": attempt.get("evaluation_reason"),
        }
        row.update(_prefix("decision", decision))
        row.update(_prefix("entry", entry))
        row.update(_prefix("exit", exit_quote))
        rows.append(row)

    return pd.DataFrame(rows), None


def _quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else np.nan


def _summary_group(label: str, frame: pd.DataFrame) -> dict[str, object]:
    decision_found = frame["decision_found"].astype(bool)
    entry_found = frame["entry_found"].astype(bool)
    exit_found = frame["exit_found"].astype(bool)
    both = entry_found & exit_found

    return {
        "group": label,
        "sampled_attempts": int(len(frame)),
        "decision_quote_coverage": float(decision_found.mean()),
        "decision_fresh_2s_coverage": float(
            frame["decision_fresh_2s"].astype(bool).mean()
        ),
        "entry_quote_coverage": float(entry_found.mean()),
        "exit_quote_coverage": float(exit_found.mean()),
        "both_execution_quote_coverage": float(both.mean()),
        "both_depth_1000_rate_given_quotes": float(
            frame.loc[both, "both_depth_1000_ok"].astype(bool).mean()
        )
        if both.any()
        else np.nan,
        "median_decision_quote_age_ms": _quantile(
            pd.to_numeric(
                frame.loc[decision_found, "decision_age_ms"],
                errors="coerce",
            ).abs(),
            0.50,
        ),
        "median_decision_spread_pct": _quantile(
            frame.loc[decision_found, "decision_spread_pct"], 0.50
        ),
        "p75_decision_spread_pct": _quantile(
            frame.loc[decision_found, "decision_spread_pct"], 0.75
        ),
        "p90_decision_spread_pct": _quantile(
            frame.loc[decision_found, "decision_spread_pct"], 0.90
        ),
        "median_entry_quote_delay_ms": _quantile(
            frame.loc[entry_found, "entry_age_ms"], 0.50
        ),
        "median_exit_quote_delay_ms": _quantile(
            frame.loc[exit_found, "exit_age_ms"], 0.50
        ),
        "mean_top_book_return_pct": float(
            pd.to_numeric(
                frame.loc[both, "top_book_ask_to_bid_return_pct"],
                errors="coerce",
            ).mean()
        )
        if both.any()
        else np.nan,
        "mean_bar_gross_return_pct": float(
            pd.to_numeric(frame["bar_gross_return_pct"], errors="coerce").mean()
        ),
        "mean_bar_base_return_pct": float(
            pd.to_numeric(frame["bar_base_net_return_pct"], errors="coerce").mean()
        ),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = [_summary_group("overall", frame)]
    rows.extend(
        _summary_group(str(month), group)
        for month, group in frame.groupby("month", sort=True)
    )
    return pd.DataFrame(rows)


def feasibility(
    summary: pd.DataFrame,
    *,
    authorization_error: str | None,
) -> dict[str, bool]:
    authorization_ok = authorization_error is None
    if summary.empty or "group" not in summary.columns:
        months = pd.DataFrame()
    else:
        months = summary.loc[summary["group"].ne("overall")].copy()
    coverage_shape_ok = len(months) == 3
    decision_ok = bool(
        coverage_shape_ok
        and months["decision_fresh_2s_coverage"].ge(
            MIN_MONTH_COVERAGE
        ).all()
    )
    execution_ok = bool(
        coverage_shape_ok
        and months["entry_quote_coverage"].ge(MIN_MONTH_COVERAGE).all()
        and months["exit_quote_coverage"].ge(MIN_MONTH_COVERAGE).all()
    )
    depth_ok = bool(
        coverage_shape_ok
        and months["both_depth_1000_rate_given_quotes"].ge(
            MIN_DEPTH_SUFFICIENCY
        ).all()
    )
    checks = {
        "historical_quote_authorization": authorization_ok,
        "decision_fresh_2s_coverage_every_month": decision_ok,
        "entry_exit_quote_coverage_every_month": execution_ok,
        "top_book_depth_1000_every_month": depth_ok,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def render_report(
    details: pd.DataFrame,
    summary: pd.DataFrame,
    checks: dict[str, bool],
    *,
    authorization_error: str | None,
    client: MassiveClient,
) -> str:
    header = [
        "=== MoneyMaker Causal NBBO Liquidity Probe v1.9 ===",
        f"authorization_error={authorization_error}",
        f"client_stats={client.stats.to_dict()}",
        "sample=20 deterministic raw_same_fit BUY attempts per month",
        "decision_quote=latest valid NBBO in prior 5s only; no future fallback",
        "decision_fresh=age <=2s",
        "execution_quote=first valid NBBO within 1s after frozen target",
        "top_book_return=entry ask to exit bid; excludes extra slippage/fees/impact",
        "audit_notional=$1000 displayed top-of-book depth only",
        "NOTE=development data feasibility diagnostic; no fresh month",
        "",
    ]
    if authorization_error is not None:
        return "\n".join(
            header
            + [
                "=== Feasibility ===",
                pd.DataFrame(
                    [
                        {"criterion": key, "pass": value}
                        for key, value in checks.items()
                    ]
                ).to_string(index=False),
            ]
        )
    return "\n".join(
        header
        + [
            "=== Summary ===",
            summary.to_string(index=False),
            "",
            "=== Feasibility ===",
            pd.DataFrame(
                [
                    {"criterion": key, "pass": value}
                    for key, value in checks.items()
                ]
            ).to_string(index=False),
            "",
            "=== Details ===",
            details.to_string(index=False),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.causal_nbbo_liquidity_probe"
    )
    parser.add_argument("attempts_csv", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/massive-rest-causal-nbbo"),
    )
    args = parser.parse_args()

    attempts = pd.read_csv(args.attempts_csv)
    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=args.cache_dir,
        request_interval_seconds=0.05,
    )
    details, authorization_error = run_probe(attempts, client)
    summary = summarize(details)
    checks = feasibility(
        summary, authorization_error=authorization_error
    )
    report = render_report(
        details,
        summary,
        checks,
        authorization_error=authorization_error,
        client=client,
    )
    print(report)

    for path in (args.report, args.details_csv, args.summary_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
