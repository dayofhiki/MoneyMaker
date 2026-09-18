from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import load_event_dataset
from .candidate_v2_research import (
    V2Config,
    _fit_threshold_params,
    apply_v2_config,
)
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS, net_round_trip_return_pct


WINNER_CONFIG = V2Config("core3", 0.90, 1.00)
BARRIERS = ("tp2_sl1", "tp3_sl2", "tp5_sl3", "tp10_sl5")
RESOLVED_STATUSES = {"take_profit", "stop_loss", "stop_gap", "timeout", "session_close"}
BASE_SCENARIO = next(s for s in DEFAULT_EXECUTION_SCENARIOS if s.name == "base")
STRESS_SCENARIO = next(s for s in DEFAULT_EXECUTION_SCENARIOS if s.name == "stress")


def _stop_pct(key: str) -> float:
    match = re.search(r"_sl([0-9p]+)$", key)
    if not match:
        raise ValueError(f"cannot parse stop from barrier {key}")
    return float(match.group(1).replace("p", "."))


def _net_from_gross(
    entry_price: pd.Series,
    gross_return: pd.Series,
    scenario,
) -> pd.Series:
    values = []
    for price, gross in zip(entry_price, gross_return, strict=False):
        if pd.isna(price) or pd.isna(gross) or float(price) <= 0:
            values.append(np.nan)
        else:
            values.append(
                net_round_trip_return_pct(float(price), float(gross), scenario)
            )
    return pd.Series(values, index=entry_price.index, dtype=float)


def _evaluate_barrier_group(
    frame: pd.DataFrame,
    key: str,
) -> dict[str, float | int | str]:
    status_col = f"{key}_status"
    return_col = f"{key}_exit_return_pct"
    if status_col not in frame.columns or return_col not in frame.columns:
        raise ValueError(f"dataset missing barrier columns for {key}")

    statuses = frame[status_col].astype("string")
    resolved = statuses.isin(RESOLVED_STATUSES)
    ambiguous = statuses.eq("ambiguous")
    evaluable = resolved | ambiguous
    stop = _stop_pct(key)

    gross = pd.to_numeric(frame[return_col], errors="coerce")
    conservative_gross = gross.copy()
    conservative_gross.loc[ambiguous] = -stop

    base_col = f"{key}_base_net_return_pct"
    stress_col = f"{key}_stress_net_return_pct"

    if base_col in frame.columns:
        base = pd.to_numeric(frame[base_col], errors="coerce")
    else:
        base = _net_from_gross(
            pd.to_numeric(frame["entry_price"], errors="coerce"),
            gross,
            BASE_SCENARIO,
        )
    if stress_col in frame.columns:
        stress = pd.to_numeric(frame[stress_col], errors="coerce")
    else:
        stress = _net_from_gross(
            pd.to_numeric(frame["entry_price"], errors="coerce"),
            gross,
            STRESS_SCENARIO,
        )

    conservative_base = base.copy()
    conservative_stress = stress.copy()
    if ambiguous.any():
        amb_entry = pd.to_numeric(frame.loc[ambiguous, "entry_price"], errors="coerce")
        amb_gross = pd.Series(-stop, index=amb_entry.index, dtype=float)
        conservative_base.loc[ambiguous] = _net_from_gross(
            amb_entry, amb_gross, BASE_SCENARIO
        )
        conservative_stress.loc[ambiguous] = _net_from_gross(
            amb_entry, amb_gross, STRESS_SCENARIO
        )

    cg = conservative_gross.loc[evaluable].dropna()
    cb = conservative_base.loc[evaluable].dropna()
    cs = conservative_stress.loc[evaluable].dropna()

    return {
        "barrier": key,
        "selected_n": int(len(frame)),
        "evaluable_n": int(evaluable.sum()),
        "evaluable_rate": float(evaluable.mean()) if len(frame) else np.nan,
        "ambiguous_rate": float(ambiguous.mean()) if len(frame) else np.nan,
        "take_profit_rate": float((statuses.loc[evaluable] == "take_profit").mean())
        if evaluable.any()
        else np.nan,
        "stop_or_gap_rate": float(
            statuses.loc[evaluable].isin(["stop_loss", "stop_gap"]).mean()
        )
        if evaluable.any()
        else np.nan,
        "timeout_or_close_rate": float(
            statuses.loc[evaluable].isin(["timeout", "session_close"]).mean()
        )
        if evaluable.any()
        else np.nan,
        "conservative_gross_mean_pct": float(cg.mean()) if len(cg) else np.nan,
        "conservative_base_mean_pct": float(cb.mean()) if len(cb) else np.nan,
        "conservative_base_median_pct": float(cb.median()) if len(cb) else np.nan,
        "conservative_base_positive_rate": float((cb > 0).mean()) if len(cb) else np.nan,
        "conservative_base_p05_pct": float(cb.quantile(0.05)) if len(cb) else np.nan,
        "conservative_stress_mean_pct": float(cs.mean()) if len(cs) else np.nan,
    }


def run_exit_research(monthly_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if len(monthly_frames) < 3:
        raise ValueError("exit research requires at least three development months")

    rows: list[dict[str, object]] = []
    for holdout_month, holdout in monthly_frames.items():
        train = pd.concat(
            [frame for month, frame in monthly_frames.items() if month != holdout_month],
            ignore_index=True,
        )
        params = _fit_threshold_params(train, WINNER_CONFIG)
        scored = apply_v2_config(holdout, params, WINNER_CONFIG)
        executable = pd.to_numeric(scored["entry_price"], errors="coerce").notna()
        selected = scored.loc[
            executable & scored["candidate_v2_selected"].astype(bool)
        ].copy()

        for key in BARRIERS:
            result = _evaluate_barrier_group(selected, key)
            result["month"] = holdout_month
            result["config"] = WINNER_CONFIG.key
            rows.append(result)
    return pd.DataFrame(rows)


def summarize_exits(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for barrier, group in details.groupby("barrier", sort=False):
        valid = group.loc[group["evaluable_n"].ge(40)].copy()
        rows.append(
            {
                "barrier": barrier,
                "months_tested": int(len(valid)),
                "min_evaluable_n": int(valid["evaluable_n"].min()) if len(valid) else 0,
                "median_evaluable_rate": float(valid["evaluable_rate"].median())
                if len(valid)
                else np.nan,
                "gross_positive_months": int(
                    (valid["conservative_gross_mean_pct"] > 0).sum()
                ),
                "base_positive_months": int(
                    (valid["conservative_base_mean_pct"] > 0).sum()
                ),
                "median_gross_mean_pct": float(
                    valid["conservative_gross_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
                "median_base_mean_pct": float(
                    valid["conservative_base_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
                "worst_base_mean_pct": float(
                    valid["conservative_base_mean_pct"].min()
                )
                if len(valid)
                else np.nan,
                "median_base_positive_rate": float(
                    valid["conservative_base_positive_rate"].median()
                )
                if len(valid)
                else np.nan,
                "median_base_p05_pct": float(
                    valid["conservative_base_p05_pct"].median()
                )
                if len(valid)
                else np.nan,
                "median_stress_mean_pct": float(
                    valid["conservative_stress_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
                "median_ambiguous_rate": float(valid["ambiguous_rate"].median())
                if len(valid)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["base_positive_months", "median_base_mean_pct", "worst_base_mean_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def render_report(details: pd.DataFrame, summary: pd.DataFrame) -> str:
    best = summary.iloc[0] if len(summary) else None
    lines = [
        "=== MoneyMaker Candidate v2 Exit Research ===",
        f"candidate_config={WINNER_CONFIG.key}",
        "method=leave-one-month-out candidate selection; barrier outcomes evaluated only on each held-out month",
        "barriers=tp2/sl1,tp3/sl2,tp5/sl3,tp10/sl5; max path horizon inherited from dataset",
        "ambiguity_policy=ambiguous OHLC minute counted conservatively as stop loss",
        "unresolved missing/session-close outcomes excluded and reported through evaluable_rate",
        "WARNING=halt annotations are still unavailable; positive barrier results would require halt/LULD repair before promotion",
        "",
        "=== Barrier summary across held-out months ===",
        summary.to_string(index=False),
        "",
    ]
    if best is not None:
        lines += [
            "=== Best barrier month-by-month ===",
            details.loc[details["barrier"].eq(best["barrier"])].to_string(index=False),
        ]
    return "\n".join(lines)


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.candidate_v2_exit_research"
    )
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    frames = {label: load_event_dataset(path) for label, path in args.dataset}
    details = run_exit_research(frames)
    summary = summarize_exits(details)
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
