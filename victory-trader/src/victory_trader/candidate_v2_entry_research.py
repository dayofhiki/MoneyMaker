from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import load_event_dataset
from .candidate_v2_research import (
    V2Config,
    _fit_threshold_params,
    apply_v2_config,
    modeled_base_zero_return_cost_pct,
)


SIGNAL_CONFIG = V2Config("core3", 0.90, None)
COST_CEILING_PCT = 1.00
DELAYS = (0, 1, 2)
HORIZONS = (5, 10, 15)


def _delay_prefix(delay: int) -> str:
    return "" if delay == 0 else f"delay{delay}_"


def _entry_price_col(delay: int) -> str:
    return "entry_price" if delay == 0 else f"delay{delay}_entry_price"


def _return_col(delay: int, horizon: int, scenario: str | None = None) -> str:
    prefix = _delay_prefix(delay)
    base = f"{prefix}return_{horizon}m"
    if scenario is None:
        return f"{base}_pct"
    return f"{base}_{scenario}_net_return_pct"


def _alpha_liquidity_mask(scored: pd.DataFrame) -> pd.Series:
    return (
        scored["candidate_v2_liquid"].astype(bool)
        & pd.to_numeric(scored["candidate_v2_score"], errors="coerce").notna()
        & scored["candidate_v2_selected"].astype(bool)
    )


def _score_signal_only(
    frame: pd.DataFrame,
    params: dict[str, dict[str, object]],
) -> pd.DataFrame:
    scored = apply_v2_config(frame, params, SIGNAL_CONFIG)
    # With no cost ceiling, selected means alpha + liquidity + valid primary entry.
    # Rebuild the signal mask without relying on the primary entry for delayed variants.
    signal = pd.Series(False, index=scored.index)
    for threshold_key, threshold_params in params.items():
        threshold = float(threshold_key)
        mask = pd.to_numeric(scored["threshold_pct"], errors="coerce").eq(threshold)
        alpha_cut = float(threshold_params["alpha_score_cut"])
        signal.loc[mask] = (
            scored.loc[mask, "candidate_v2_liquid"].astype(bool)
            & pd.to_numeric(scored.loc[mask, "candidate_v2_score"], errors="coerce").ge(alpha_cut)
        )
    scored["candidate_v2_signal"] = signal
    return scored


def _evaluate_delay_month(
    scored: pd.DataFrame,
    *,
    month: str,
    delay: int,
) -> list[dict[str, object]]:
    entry_col = _entry_price_col(delay)
    if entry_col not in scored.columns:
        raise ValueError(f"dataset missing delayed entry column: {entry_col}")

    entry = pd.to_numeric(scored[entry_col], errors="coerce")
    entry_cost = entry.apply(modeled_base_zero_return_cost_pct)
    selected = (
        scored["candidate_v2_signal"].astype(bool)
        & entry.notna()
        & entry_cost.le(COST_CEILING_PCT)
    )

    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        gross_col = _return_col(delay, horizon)
        base_col = _return_col(delay, horizon, "base")
        stress_col = _return_col(delay, horizon, "stress")
        missing = [c for c in (gross_col, base_col, stress_col) if c not in scored.columns]
        if missing:
            raise ValueError(f"dataset missing delayed return columns: {missing}")

        gross = pd.to_numeric(scored[gross_col], errors="coerce")
        base = pd.to_numeric(scored[base_col], errors="coerce")
        stress = pd.to_numeric(scored[stress_col], errors="coerce")
        observed = selected & gross.notna()

        rows.append(
            {
                "month": month,
                "delay_min": delay,
                "horizon_min": horizon,
                "signal_n": int(scored["candidate_v2_signal"].sum()),
                "selected_entry_n": int(selected.sum()),
                "observed_n": int(observed.sum()),
                "observed_given_entry_rate": float(observed.sum() / selected.sum())
                if selected.any()
                else np.nan,
                "median_entry_price": float(entry.loc[selected].median())
                if selected.any()
                else np.nan,
                "median_modeled_zero_return_cost_pct": float(entry_cost.loc[selected].median())
                if selected.any()
                else np.nan,
                "gross_mean_pct": float(gross.loc[observed].mean())
                if observed.any()
                else np.nan,
                "gross_median_pct": float(gross.loc[observed].median())
                if observed.any()
                else np.nan,
                "base_mean_pct": float(base.loc[observed].mean())
                if observed.any()
                else np.nan,
                "base_median_pct": float(base.loc[observed].median())
                if observed.any()
                else np.nan,
                "base_positive_rate": float((base.loc[observed] > 0).mean())
                if observed.any()
                else np.nan,
                "base_p05_pct": float(base.loc[observed].quantile(0.05))
                if observed.any()
                else np.nan,
                "stress_mean_pct": float(stress.loc[observed].mean())
                if observed.any()
                else np.nan,
            }
        )
    return rows


def run_entry_timing_research(
    monthly_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if len(monthly_frames) < 3:
        raise ValueError("entry timing research requires at least three development months")

    rows: list[dict[str, object]] = []
    for holdout_month, holdout in monthly_frames.items():
        train = pd.concat(
            [frame for month, frame in monthly_frames.items() if month != holdout_month],
            ignore_index=True,
        )
        params = _fit_threshold_params(train, SIGNAL_CONFIG)
        scored = _score_signal_only(holdout, params)
        for delay in DELAYS:
            rows.extend(
                _evaluate_delay_month(
                    scored,
                    month=holdout_month,
                    delay=delay,
                )
            )
    return pd.DataFrame(rows)


def summarize_entry_timing(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (delay, horizon), group in details.groupby(
        ["delay_min", "horizon_min"], sort=True
    ):
        valid = group.loc[group["observed_n"].ge(40)].copy()
        rows.append(
            {
                "delay_min": int(delay),
                "horizon_min": int(horizon),
                "months_tested": int(len(valid)),
                "min_observed_n": int(valid["observed_n"].min()) if len(valid) else 0,
                "median_selected_entry_n": float(valid["selected_entry_n"].median())
                if len(valid)
                else np.nan,
                "median_observed_given_entry_rate": float(
                    valid["observed_given_entry_rate"].median()
                )
                if len(valid)
                else np.nan,
                "gross_positive_months": int((valid["gross_mean_pct"] > 0).sum()),
                "base_positive_months": int((valid["base_mean_pct"] > 0).sum()),
                "median_gross_mean_pct": float(valid["gross_mean_pct"].median())
                if len(valid)
                else np.nan,
                "median_base_mean_pct": float(valid["base_mean_pct"].median())
                if len(valid)
                else np.nan,
                "worst_base_mean_pct": float(valid["base_mean_pct"].min())
                if len(valid)
                else np.nan,
                "median_base_positive_rate": float(valid["base_positive_rate"].median())
                if len(valid)
                else np.nan,
                "median_base_p05_pct": float(valid["base_p05_pct"].median())
                if len(valid)
                else np.nan,
                "median_stress_mean_pct": float(valid["stress_mean_pct"].median())
                if len(valid)
                else np.nan,
                "median_entry_price": float(valid["median_entry_price"].median())
                if len(valid)
                else np.nan,
                "median_modeled_zero_return_cost_pct": float(
                    valid["median_modeled_zero_return_cost_pct"].median()
                )
                if len(valid)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "base_positive_months",
            "median_base_mean_pct",
            "worst_base_mean_pct",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def render_report(details: pd.DataFrame, summary: pd.DataFrame) -> str:
    best = summary.iloc[0] if len(summary) else None
    lines = [
        "=== MoneyMaker Candidate v2 Entry Timing Research ===",
        "signal=core3 top10% with liquidity gate, fit leave-one-month-out",
        "entry_cost_gate=delay-specific modeled base zero-return friction <= 1.00%",
        "delays=0(next-minute open),1(one additional minute wait),2(two additional minute wait)",
        "horizons=5m,10m,15m measured from each delayed entry",
        "",
        "=== Delay summary across held-out months ===",
        summary.to_string(index=False),
        "",
    ]
    if best is not None:
        lines += [
            "=== Best delay/horizon month-by-month ===",
            details.loc[
                details["delay_min"].eq(best["delay_min"])
                & details["horizon_min"].eq(best["horizon_min"])
            ].to_string(index=False),
        ]
    return "\n".join(lines)


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.candidate_v2_entry_research"
    )
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    frames = {label: load_event_dataset(path) for label, path in args.dataset}
    details = run_entry_timing_research(frames)
    summary = summarize_entry_timing(details)
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
