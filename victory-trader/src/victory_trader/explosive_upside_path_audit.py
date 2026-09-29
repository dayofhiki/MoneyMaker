"""Request289: explosive-upside path audit.

This is a hindsight diagnostic only.  It does not change any trading action.

For every causal first-2%-pullback entry on May5-8, measure how much upside was
actually available over fixed 10/20/30/60 minute windows and how much adverse
movement had to be endured before the later peak.  Compare all pullback entries
with the exact outer-fold stage-1 top-20 shortlist, then compare the shortlist's
future upside with the frozen -2.5% / 2% trailing / 10-minute exit policy.

Future highs/lows are labels for diagnosis, never model features or fill
guarantees.  Minute opens are also reported as a stricter ordered reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .causal_minute_controller_rebuild import causal_execution_scan
from .downstream_aligned_candidate import (
    attach_downstream_target,
    candidate_columns,
)
from .full_hot_fixed_policy_value import attach_fixed_value
from .joint_pullback_admission import (
    FIXED_MAX_HOLD_MINUTES,
    FIXED_STOP_LOSS_PCT,
    FIXED_TRAIL_PCT,
)
from .lagged_minute_context import CROSSFIT_DAYS
from .pullback_trailing_exit import (
    apply_exit_rule,
    build_pullback_episodes,
)
from .two_stage_risk_veto import (
    attach_ticker_history,
    fold_stage1,
)

REQUEST_ID = 289
HORIZONS_MINUTES = (10, 20, 30, 60)
EXPLOSIVE_THRESHOLDS_PCT = (5.0, 10.0, 20.0)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def market_path_lookup(
    scan: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    columns = ["trading_day", "ticker", "t", "o"]
    if "h" in scan.columns:
        columns.append("h")
    if "l" in scan.columns:
        columns.append("l")

    work = scan.loc[:, columns].copy()
    for name in ("t", "o", "h", "l"):
        if name in work:
            work[name] = _numeric(work[name])
    work = work.loc[
        work.t.notna() & work.o.gt(0)
    ].copy()
    if "h" not in work:
        work["h"] = work["o"]
    if "l" not in work:
        work["l"] = work["o"]
    work["h"] = work["h"].where(work["h"].gt(0), work["o"])
    work["l"] = work["l"].where(work["l"].gt(0), work["o"])
    work["ticker"] = work.ticker.astype(str).str.upper()
    work["trading_day"] = work.trading_day.astype(str)
    work = work.sort_values(["trading_day", "ticker", "t"])

    return {
        (str(day), str(ticker).upper()): part.reset_index(drop=True)
        for (day, ticker), part in work.groupby(
            ["trading_day", "ticker"],
            sort=False,
        )
    }


def excursion_for_horizon(
    path: pd.DataFrame,
    *,
    entry_t: int,
    entry_price: float,
    horizon_minutes: int,
) -> dict:
    deadline = int(entry_t) + int(horizon_minutes) * 60_000
    future = path.loc[
        path.t.gt(int(entry_t)) & path.t.le(deadline)
    ].copy()
    later = path.loc[path.t.ge(deadline)].copy()
    complete = bool(len(later))
    result = {
        "complete": complete,
        "observed_rows": int(len(future)),
        "mfe_open_pct": None,
        "mfe_high_pct": None,
        "mae_low_pct": None,
        "time_to_open_mfe_minutes": None,
        "time_to_high_mfe_minutes": None,
        "pre_high_peak_mae_low_pct": None,
    }
    for threshold in EXPLOSIVE_THRESHOLDS_PCT:
        tag = int(threshold)
        result[f"first_high_hit_{tag}_t"] = None
        result[f"first_high_hit_{tag}_minutes"] = None

    if not complete or future.empty or entry_price <= 0:
        return result

    future["open_return_pct"] = (
        future["o"].astype(float) / float(entry_price) - 1.0
    ) * 100.0
    future["high_return_pct"] = (
        future["h"].astype(float) / float(entry_price) - 1.0
    ) * 100.0
    future["low_return_pct"] = (
        future["l"].astype(float) / float(entry_price) - 1.0
    ) * 100.0

    open_index = future.open_return_pct.idxmax()
    high_index = future.high_return_pct.idxmax()
    open_row = future.loc[open_index]
    high_row = future.loc[high_index]

    result["mfe_open_pct"] = float(max(0.0, open_row.open_return_pct))
    result["mfe_high_pct"] = float(max(0.0, high_row.high_return_pct))
    result["mae_low_pct"] = float(min(0.0, future.low_return_pct.min()))
    result["time_to_open_mfe_minutes"] = float(
        (int(open_row.t) - int(entry_t)) / 60_000
    )
    result["time_to_high_mfe_minutes"] = float(
        (int(high_row.t) - int(entry_t)) / 60_000
    )

    before_peak = future.loc[future.t.lt(int(high_row.t))]
    result["pre_high_peak_mae_low_pct"] = (
        float(min(0.0, before_peak.low_return_pct.min()))
        if len(before_peak)
        else 0.0
    )

    for threshold in EXPLOSIVE_THRESHOLDS_PCT:
        tag = int(threshold)
        hits = future.loc[
            future.high_return_pct.ge(float(threshold))
        ]
        if len(hits):
            hit_t = int(hits.iloc[0].t)
            result[f"first_high_hit_{tag}_t"] = hit_t
            result[f"first_high_hit_{tag}_minutes"] = float(
                (hit_t - int(entry_t)) / 60_000
            )
    return result


def build_entry_audit(
    episodes: pd.DataFrame,
    policy: pd.DataFrame,
    stage1: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    rows = episodes.merge(
        stage1.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "prehot_predicted_value_pct",
            "prehot_threshold",
            "selected_prehot",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="one_to_one",
    ).merge(
        policy.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "resolved",
            "trade_return_pct",
            "exit_t",
            "exit_reason",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="one_to_one",
    )
    rows["selected_prehot"] = (
        rows.selected_prehot.fillna(False).astype(bool)
    )
    lookup = market_path_lookup(scan)
    records = []
    for raw in rows.to_dict("records"):
        record = {
            key: value
            for key, value in raw.items()
            if key != "path"
        }
        if not bool(raw["entered"]):
            for horizon in HORIZONS_MINUTES:
                prefix = f"h{horizon}"
                metrics = excursion_for_horizon(
                    pd.DataFrame(columns=["t", "o", "h", "l"]),
                    entry_t=0,
                    entry_price=1.0,
                    horizon_minutes=horizon,
                )
                for name, value in metrics.items():
                    record[f"{prefix}_{name}"] = value
            records.append(record)
            continue

        day = str(raw["trading_day"])
        ticker = str(raw["ticker"]).upper()
        path = lookup.get(
            (day, ticker),
            pd.DataFrame(columns=["t", "o", "h", "l"]),
        )
        entry_t = int(raw["entry_t"])
        entry_price = float(raw["entry_price"])
        for horizon in HORIZONS_MINUTES:
            prefix = f"h{horizon}"
            metrics = excursion_for_horizon(
                path,
                entry_t=entry_t,
                entry_price=entry_price,
                horizon_minutes=horizon,
            )
            for name, value in metrics.items():
                record[f"{prefix}_{name}"] = value
        records.append(record)
    return pd.DataFrame(records)


def population_metrics(
    rows: pd.DataFrame,
    *,
    selected_only: bool,
) -> dict:
    work = rows.loc[rows.entered.astype(bool)].copy()
    if selected_only:
        work = work.loc[
            work.selected_prehot.fillna(False).astype(bool)
        ].copy()

    result = {
        "entries": int(len(work)),
        "resolved_frozen_policy": int(
            work.resolved.fillna(False).astype(bool).sum()
        ),
        "horizons": {},
    }
    for horizon in HORIZONS_MINUTES:
        prefix = f"h{horizon}"
        complete = work.loc[
            work[f"{prefix}_complete"].fillna(False).astype(bool)
        ].copy()
        mfe_high = _numeric(complete[f"{prefix}_mfe_high_pct"])
        mfe_open = _numeric(complete[f"{prefix}_mfe_open_pct"])
        mae = _numeric(complete[f"{prefix}_mae_low_pct"])
        pre_peak_mae = _numeric(
            complete[f"{prefix}_pre_high_peak_mae_low_pct"]
        )
        horizon_report = {
            "complete_rows": int(len(complete)),
            "coverage": (
                float(len(complete) / len(work))
                if len(work)
                else 0.0
            ),
            "mean_mfe_open_pct": (
                float(mfe_open.mean()) if len(mfe_open) else None
            ),
            "median_mfe_open_pct": (
                float(mfe_open.median()) if len(mfe_open) else None
            ),
            "mean_mfe_high_pct": (
                float(mfe_high.mean()) if len(mfe_high) else None
            ),
            "median_mfe_high_pct": (
                float(mfe_high.median()) if len(mfe_high) else None
            ),
            "mean_mae_low_pct": (
                float(mae.mean()) if len(mae) else None
            ),
            "median_pre_peak_mae_low_pct": (
                float(pre_peak_mae.median())
                if len(pre_peak_mae)
                else None
            ),
        }
        for threshold in EXPLOSIVE_THRESHOLDS_PCT:
            tag = int(threshold)
            rate = (
                float(mfe_high.ge(threshold).mean())
                if len(mfe_high)
                else None
            )
            horizon_report[f"potential_ge_plus{tag}_rate"] = rate
            horizon_report[f"potential_ge_plus{tag}_rows"] = (
                int(mfe_high.ge(threshold).sum())
                if len(mfe_high)
                else 0
            )
        result["horizons"][str(horizon)] = horizon_report
    return result


def shortlist_uplift(
    all_metrics: dict,
    shortlist_metrics: dict,
) -> dict:
    out = {}
    for horizon in HORIZONS_MINUTES:
        all_h = all_metrics["horizons"][str(horizon)]
        sel_h = shortlist_metrics["horizons"][str(horizon)]
        row = {}
        for threshold in EXPLOSIVE_THRESHOLDS_PCT:
            tag = int(threshold)
            key = f"potential_ge_plus{tag}_rate"
            left = all_h.get(key)
            right = sel_h.get(key)
            row[f"plus{tag}_rate_uplift_pp"] = (
                float((right - left) * 100.0)
                if left is not None and right is not None
                else None
            )
            row[f"plus{tag}_rate_ratio"] = (
                float(right / left)
                if left is not None
                and right is not None
                and left > 0
                else None
            )
        out[str(horizon)] = row
    return out


def frozen_policy_capture(rows: pd.DataFrame) -> dict:
    work = rows.loc[
        rows.entered.astype(bool)
        & rows.selected_prehot.fillna(False).astype(bool)
        & rows.resolved.fillna(False).astype(bool)
        & rows.h60_complete.fillna(False).astype(bool)
        & _numeric(rows.trade_return_pct).notna()
    ].copy()
    actual = _numeric(work.trade_return_pct)
    mfe = _numeric(work.h60_mfe_high_pct)
    report = {
        "resolved_shortlist_h60_rows": int(len(work)),
        "frozen_policy_trade_mean_pct": (
            float(actual.mean()) if len(actual) else None
        ),
        "mean_60m_mfe_high_pct": (
            float(mfe.mean()) if len(mfe) else None
        ),
        "mean_left_on_table_vs_60m_high_pp": (
            float((mfe - actual).mean())
            if len(work)
            else None
        ),
        "exit_reasons": {
            str(reason): int(len(part))
            for reason, part in work.groupby(
                work.exit_reason.astype(str),
                sort=True,
            )
        },
        "explosive_groups": {},
    }

    for threshold in EXPLOSIVE_THRESHOLDS_PCT:
        tag = int(threshold)
        group = work.loc[mfe.ge(float(threshold))].copy()
        group_return = _numeric(group.trade_return_pct)
        hit_t = _numeric(group[f"h60_first_high_hit_{tag}_t"])
        exit_t = _numeric(group.exit_t)
        valid_time = hit_t.notna() & exit_t.notna()
        exited_before_hit = (
            exit_t.loc[valid_time] < hit_t.loc[valid_time]
        )
        pre_peak_mae = _numeric(
            group.h60_pre_high_peak_mae_low_pct
        )
        time_to_peak = _numeric(
            group.h60_time_to_high_mfe_minutes
        )
        report["explosive_groups"][f"plus{tag}"] = {
            "rows": int(len(group)),
            "frozen_policy_trade_mean_pct": (
                float(group_return.mean())
                if len(group_return)
                else None
            ),
            "mean_60m_mfe_high_pct": (
                float(_numeric(group.h60_mfe_high_pct).mean())
                if len(group)
                else None
            ),
            "median_pre_peak_mae_low_pct": (
                float(pre_peak_mae.median())
                if len(pre_peak_mae)
                else None
            ),
            "median_minutes_to_60m_peak": (
                float(time_to_peak.median())
                if len(time_to_peak)
                else None
            ),
            "exited_before_first_threshold_hit_rate": (
                float(exited_before_hit.mean())
                if len(exited_before_hit)
                else None
            ),
            "policy_realized_ge_threshold_rate": (
                float(group_return.ge(float(threshold)).mean())
                if len(group_return)
                else None
            ),
        }
    return report


def evaluate(
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    raw_scan = pd.read_parquet(scan_path)
    raw_scan = raw_scan.loc[
        raw_scan.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    causal_scan = causal_execution_scan(raw_scan)

    first_hot = pd.read_parquet(first_hot_path).drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    first_hot = attach_fixed_value(first_hot, causal_scan)

    episodes = build_pullback_episodes(first_hot, causal_scan)
    policy = apply_exit_rule(
        episodes,
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name="request289_frozen_policy",
    )
    labeled = attach_downstream_target(first_hot, policy)
    candidates, ticker_extra = attach_ticker_history(
        labeled,
        raw_scan,
    )
    prehot_columns = tuple(
        dict.fromkeys([
            *candidate_columns(candidates),
            *ticker_extra,
        ])
    )

    stage1_parts = []
    fold_support = {}
    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train = candidates.loc[
            ~candidates.trading_day.astype(str).eq(held_day)
        ].copy()
        held = candidates.loc[
            candidates.trading_day.astype(str).eq(held_day)
        ].copy()
        scored, threshold = fold_stage1(
            train,
            held,
            prehot_columns,
            20262700 + fold_index * 10,
        )
        scored["fold_holdout_day"] = held_day
        stage1_parts.append(scored)
        fold_support[str(held_day)] = {
            "candidate_rows": int(len(held)),
            "prehot_threshold": float(threshold),
            "shortlisted_candidates": int(
                scored.selected_prehot.astype(bool).sum()
            ),
        }
    stage1 = pd.concat(stage1_parts, ignore_index=True)

    audit = build_entry_audit(
        episodes,
        policy,
        stage1,
        raw_scan,
    )
    all_entries = population_metrics(
        audit,
        selected_only=False,
    )
    shortlist_entries = population_metrics(
        audit,
        selected_only=True,
    )
    uplift = shortlist_uplift(
        all_entries,
        shortlist_entries,
    )
    capture = frozen_policy_capture(audit)

    selected_entered = audit.loc[
        audit.entered.astype(bool)
        & audit.selected_prehot.fillna(False).astype(bool)
    ].copy()
    h60 = selected_entered.loc[
        selected_entered.h60_complete.fillna(False).astype(bool)
    ].copy()
    plus10_rate = (
        float(_numeric(h60.h60_mfe_high_pct).ge(10.0).mean())
        if len(h60)
        else None
    )
    plus10_uplift = uplift["60"]["plus10_rate_uplift_pp"]
    plus10_group = capture["explosive_groups"]["plus10"]
    exit_before_plus10 = plus10_group[
        "exited_before_first_threshold_hit_rate"
    ]

    diagnosis = {
        "shortlist_contains_plus10_opportunities": (
            plus10_rate is not None and plus10_rate > 0
        ),
        "shortlist_enriches_plus10_vs_all_entries": (
            plus10_uplift is not None and plus10_uplift > 0
        ),
        "frozen_exit_often_leaves_before_plus10": (
            exit_before_plus10 is not None
            and exit_before_plus10 >= 0.50
        ),
    }

    if (
        diagnosis["shortlist_contains_plus10_opportunities"]
        and diagnosis["frozen_exit_often_leaves_before_plus10"]
    ):
        next_boundary = (
            "exit/hold logic is a primary bottleneck: model continuation value "
            "on the explosive subset before changing candidate selection"
        )
    elif diagnosis["shortlist_enriches_plus10_vs_all_entries"]:
        next_boundary = (
            "selection has some explosive-upside signal but capture is unclear; "
            "study path states around continuation/re-acceleration"
        )
    else:
        next_boundary = (
            "stage1 selection does not sufficiently enrich explosive upside; "
            "learn a dedicated crack-opportunity target from future excursion "
            "labels before revisiting exits"
        )

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "diagnostic_only": True,
        "future_information_usage": (
            "future minute highs/lows/opens are outcome labels only and are "
            "never available to the causal selector or entry rule"
        ),
        "fixed_entry": "first causal 2% pullback",
        "frozen_exit_policy": {
            "hard_stop_loss_pct": FIXED_STOP_LOSS_PCT,
            "trailing_drawdown_pct": FIXED_TRAIL_PCT,
            "max_hold_minutes": FIXED_MAX_HOLD_MINUTES,
        },
        "horizons_minutes": list(HORIZONS_MINUTES),
        "explosive_thresholds_pct": list(
            EXPLOSIVE_THRESHOLDS_PCT
        ),
        "fold_support": fold_support,
        "all_pullback_entries": all_entries,
        "stage1_shortlist_entries": shortlist_entries,
        "shortlist_explosive_uplift": uplift,
        "frozen_policy_capture": capture,
        "diagnosis": diagnosis,
        "next_boundary": next_boundary,
        "interpretation": (
            "Minute-bar highs are excursion ceilings, not executable fill "
            "guarantees. Open-path MFE is reported separately as a stricter "
            "ordered reference. Pre-peak MAE excludes the peak minute because "
            "minute OHLC does not reveal intraminute high/low ordering."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    audit.to_parquet(rows_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
