from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = (5, 10, 15)
WINDOWS = (30, 60, 180)
EPISODE_KEYS = ["trading_day", "ticker"]


def _episode_oracle(
    frame: pd.DataFrame,
    *,
    horizon: int,
    max_minutes_since_cross: int,
) -> pd.DataFrame:
    base_col = f"buy_return_{horizon}m_base_net_return_pct"
    gross_col = f"buy_return_{horizon}m_pct"

    minutes = pd.to_numeric(
        frame["minutes_since_10pct_cross"], errors="coerce"
    )
    work = frame.loc[
        minutes.between(0, max_minutes_since_cross, inclusive="both")
        & pd.to_numeric(frame["entry_price"], errors="coerce").notna()
        & pd.to_numeric(frame[base_col], errors="coerce").notna()
    ].copy()

    if work.empty:
        return work

    work["_base"] = pd.to_numeric(work[base_col], errors="coerce")
    work["_gross"] = pd.to_numeric(work[gross_col], errors="coerce")
    work["_minutes"] = minutes.loc[work.index]

    idx = work.groupby(EPISODE_KEYS)["_base"].idxmax()
    best = work.loc[idx].copy()
    best = best.rename(
        columns={
            "_base": "oracle_best_base_pct",
            "_gross": "oracle_best_gross_pct",
            "_minutes": "oracle_entry_minute",
        }
    )
    return best


def evaluate_month(
    frame: pd.DataFrame,
    *,
    month: str,
) -> pd.DataFrame:
    episode_total = int(frame[EPISODE_KEYS].drop_duplicates().shape[0])
    rows: list[dict[str, object]] = []

    for horizon in HORIZONS:
        base_col = f"buy_return_{horizon}m_base_net_return_pct"
        for window in WINDOWS:
            best = _episode_oracle(
                frame,
                horizon=horizon,
                max_minutes_since_cross=window,
            )
            if best.empty:
                rows.append(
                    {
                        "month": month,
                        "horizon_min": horizon,
                        "window_min": window,
                        "episode_total": episode_total,
                        "evaluable_episodes": 0,
                    }
                )
                continue

            base = pd.to_numeric(
                best["oracle_best_base_pct"], errors="coerce"
            )
            gross = pd.to_numeric(
                best["oracle_best_gross_pct"], errors="coerce"
            )
            minute = pd.to_numeric(
                best["oracle_entry_minute"], errors="coerce"
            )

            # How many episodes contain at least one positive after-cost entry,
            # not just whether the single oracle maximum is positive.
            eligible = frame.loc[
                pd.to_numeric(
                    frame["minutes_since_10pct_cross"], errors="coerce"
                ).between(0, window, inclusive="both")
                & pd.to_numeric(frame[base_col], errors="coerce").notna()
            ].copy()
            episode_positive = (
                eligible.assign(
                    _positive=pd.to_numeric(
                        eligible[base_col], errors="coerce"
                    ).gt(0)
                )
                .groupby(EPISODE_KEYS)["_positive"]
                .any()
            )

            rows.append(
                {
                    "month": month,
                    "horizon_min": horizon,
                    "window_min": window,
                    "episode_total": episode_total,
                    "evaluable_episodes": int(len(best)),
                    "evaluable_episode_rate": float(
                        len(best) / episode_total
                    )
                    if episode_total
                    else np.nan,
                    "episodes_with_any_positive_base_rate": float(
                        episode_positive.mean()
                    )
                    if len(episode_positive)
                    else np.nan,
                    "oracle_best_base_mean_pct": float(base.mean()),
                    "oracle_best_base_median_pct": float(base.median()),
                    "oracle_best_base_p05_pct": float(base.quantile(0.05)),
                    "oracle_best_base_positive_rate": float((base > 0).mean()),
                    "oracle_best_gross_mean_pct": float(gross.mean()),
                    "oracle_entry_minute_median": float(minute.median()),
                    "oracle_entry_minute_p25": float(minute.quantile(0.25)),
                    "oracle_entry_minute_p75": float(minute.quantile(0.75)),
                }
            )

    return pd.DataFrame(rows)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (horizon, window), group in details.groupby(
        ["horizon_min", "window_min"], sort=True
    ):
        rows.append(
            {
                "horizon_min": int(horizon),
                "window_min": int(window),
                "months_tested": int(len(group)),
                "median_any_positive_episode_rate": float(
                    group["episodes_with_any_positive_base_rate"].median()
                ),
                "worst_any_positive_episode_rate": float(
                    group["episodes_with_any_positive_base_rate"].min()
                ),
                "median_oracle_base_mean_pct": float(
                    group["oracle_best_base_mean_pct"].median()
                ),
                "worst_oracle_base_mean_pct": float(
                    group["oracle_best_base_mean_pct"].min()
                ),
                "median_oracle_base_positive_rate": float(
                    group["oracle_best_base_positive_rate"].median()
                ),
                "median_oracle_entry_minute": float(
                    group["oracle_entry_minute_median"].median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "median_oracle_base_mean_pct",
            "median_any_positive_episode_rate",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)


def render_report(details: pd.DataFrame, summary: pd.DataFrame) -> str:
    return "\n".join(
        [
            "=== MoneyMaker State Opportunity Oracle ===",
            "purpose=non-tradable upper bound for whether profitable timing opportunities exist under the current base execution model",
            "oracle=for each ticker-day, retrospectively select the state with the best realized base-net fixed-horizon return",
            "windows=first 30m / 60m / 180m after initial +10% cross",
            "WARNING=this intentionally uses future returns and must never be used as a trading rule",
            "",
            "=== Cross-month opportunity ceiling ===",
            summary.to_string(index=False),
            "",
            "=== Month-by-month ===",
            details.to_string(index=False),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.state_oracle_diagnostic"
    )
    parser.add_argument(
        "--dataset", action="append", type=_parse_dataset, required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    pieces = []
    for label, path in args.dataset:
        frame = pd.read_parquet(path)
        pieces.append(evaluate_month(frame, month=label))
    details = pd.concat(pieces, ignore_index=True)
    summary = summarize(details)
    report = render_report(details, summary)
    print(report)

    for path in (args.report, args.details_csv, args.summary_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
