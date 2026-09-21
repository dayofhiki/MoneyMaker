from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings
from .massive_client import MassiveClient


SAMPLE_PER_MONTH = 25
REQUEST_SUCCESS_MIN = 0.95
FIELD_COVERAGE_MIN = 0.80
SOURCE_POLICY = "feasible_earliest_15m_cap1"


def deterministic_sample(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.loc[trades["policy"].eq(SOURCE_POLICY)].copy()
    frame["trading_day"] = frame["trading_day"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = (
        frame.sort_values(
            ["month", "trading_day", "ticker", "t"],
            kind="stable",
        )
        .drop_duplicates(["month", "trading_day", "ticker"])
        .reset_index(drop=True)
    )

    samples: list[pd.DataFrame] = []
    for month, group in frame.groupby("month", sort=True):
        ordered = group.sort_values(
            ["trading_day", "ticker", "t"],
            kind="stable",
        ).reset_index(drop=True)
        if len(ordered) <= SAMPLE_PER_MONTH:
            chosen = ordered
        else:
            indexes = np.linspace(
                0,
                len(ordered) - 1,
                SAMPLE_PER_MONTH,
                dtype=int,
            )
            chosen = ordered.iloc[indexes].copy()
        samples.append(chosen)

    if not samples:
        return frame.iloc[0:0].copy()
    return pd.concat(samples, ignore_index=True)


def _positive_number(value: object) -> float:
    numeric = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(numeric) or float(numeric) <= 0:
        return np.nan
    return float(numeric)


def probe_rows(
    sample: pd.DataFrame,
    client: MassiveClient,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for item in sample.itertuples(index=False):
        trading_day = pd.Timestamp(str(item.trading_day)).date()
        query_day = trading_day - timedelta(days=1)
        if query_day >= trading_day:
            raise ValueError("reference query date is not strictly prior")

        ticker = str(item.ticker).upper()
        request_success = True
        error_type = ""
        payload: dict[str, object] = {}
        try:
            response = client.ticker_details(ticker, query_day)
            raw = response.get("results") or {}
            if isinstance(raw, dict):
                payload = raw
            else:
                request_success = False
                error_type = "missing_results"
        except Exception as exc:  # persist coverage failures; never backfill
            request_success = False
            error_type = type(exc).__name__

        share_class = _positive_number(
            payload.get("share_class_shares_outstanding")
        )
        weighted = _positive_number(
            payload.get("weighted_shares_outstanding")
        )
        market_cap = _positive_number(payload.get("market_cap"))
        previous_close = _positive_number(
            getattr(item, "previous_close", np.nan)
        )
        volume_5m = _positive_number(
            getattr(item, "volume_5m", np.nan)
        )
        dollar_volume_5m = _positive_number(
            getattr(item, "dollar_volume_5m", np.nan)
        )

        implied_market_cap = (
            weighted * previous_close
            if np.isfinite(weighted) and np.isfinite(previous_close)
            else np.nan
        )
        share_turnover = (
            volume_5m / share_class
            if np.isfinite(volume_5m) and np.isfinite(share_class)
            else np.nan
        )
        dollar_turnover = (
            dollar_volume_5m / implied_market_cap
            if np.isfinite(dollar_volume_5m)
            and np.isfinite(implied_market_cap)
            else np.nan
        )

        rows.append(
            {
                "month": str(item.month),
                "trading_day": trading_day.isoformat(),
                "ticker": ticker,
                "query_date": query_day.isoformat(),
                "request_success": float(request_success),
                "error_type": error_type,
                "share_class_shares_outstanding": share_class,
                "weighted_shares_outstanding": weighted,
                "provider_market_cap": market_cap,
                "implied_market_cap_prior": implied_market_cap,
                "volume_5m_share_turnover": share_turnover,
                "dollar_volume_5m_market_cap_turnover": dollar_turnover,
            }
        )

    return pd.DataFrame(rows)


def coverage_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for month, group in rows.groupby("month", sort=True):
        summaries.append(
            {
                "month": month,
                "sampled": int(len(group)),
                "request_success": float(
                    pd.to_numeric(
                        group["request_success"], errors="coerce"
                    ).eq(1.0).mean()
                ),
                "weighted_shares_coverage": float(
                    pd.to_numeric(
                        group["weighted_shares_outstanding"],
                        errors="coerce",
                    ).notna().mean()
                ),
                "share_class_coverage": float(
                    pd.to_numeric(
                        group["share_class_shares_outstanding"],
                        errors="coerce",
                    ).notna().mean()
                ),
                "provider_market_cap_coverage": float(
                    pd.to_numeric(
                        group["provider_market_cap"],
                        errors="coerce",
                    ).notna().mean()
                ),
                "implied_market_cap_coverage": float(
                    pd.to_numeric(
                        group["implied_market_cap_prior"],
                        errors="coerce",
                    ).notna().mean()
                ),
            }
        )
    return pd.DataFrame(summaries)


def success_check(rows: pd.DataFrame, summary: pd.DataFrame) -> dict[str, bool]:
    strict_prior = bool(
        (
            pd.to_datetime(rows["query_date"])
            < pd.to_datetime(rows["trading_day"])
        ).all()
    )
    checks = {
        "request_success_every_month_ge_95pct": bool(
            len(summary) == 3
            and summary["request_success"].ge(REQUEST_SUCCESS_MIN).all()
        ),
        "weighted_shares_every_month_ge_80pct": bool(
            len(summary) == 3
            and summary["weighted_shares_coverage"]
            .ge(FIELD_COVERAGE_MIN)
            .all()
        ),
        "share_class_every_month_ge_80pct": bool(
            len(summary) == 3
            and summary["share_class_coverage"]
            .ge(FIELD_COVERAGE_MIN)
            .all()
        ),
        "implied_market_cap_every_month_ge_80pct": bool(
            len(summary) == 3
            and summary["implied_market_cap_coverage"]
            .ge(FIELD_COVERAGE_MIN)
            .all()
        ),
        "all_query_dates_strictly_prior": strict_prior,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def render_report(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    checks: dict[str, bool],
    client: MassiveClient,
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker Point-in-Time Share-Supply Probe v0.1 ===",
            f"sampled_rows={len(rows)}",
            (
                "source=Massive ticker details at D-1 calendar day; "
                "no current fallback"
            ),
            f"Massive_client_stats={client.stats.to_dict()}",
            "",
            "=== Coverage ===",
            summary.to_string(index=False),
            "",
            "=== Pre-registered go/no-go ===",
            pd.DataFrame(
                [
                    {"criterion": key, "pass": value}
                    for key, value in checks.items()
                ]
            ).to_string(index=False),
            "",
            "=== Probe rows ===",
            rows.to_string(index=False),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.point_in_time_share_supply_probe"
    )
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/massive-rest"),
    )
    args = parser.parse_args()

    trades = pd.read_csv(args.trades_csv)
    sample = deterministic_sample(trades)

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=args.cache_dir,
        request_interval_seconds=0.05,
    )
    rows = probe_rows(sample, client)
    summary = coverage_summary(rows)
    checks = success_check(rows, summary)
    report = render_report(rows, summary, checks, client)
    print(report, flush=True)

    for path in (args.rows_csv, args.coverage_csv, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.rows_csv, index=False)
    summary.to_csv(args.coverage_csv, index=False)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
