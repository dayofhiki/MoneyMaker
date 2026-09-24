"""Fast, vectorized final evaluation for Request 165.

Scientific contract is identical to causal_entry_executability.py. This module
reuses the completed fresh day shards and vectorizes exact-minute economic
labeling instead of iterating over the full scan price lookup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .causal_entry_executability import (
    EXECUTION_FEATURES,
    EXEC_MODEL_KWARGS,
    FRESH_DAILY_ENTRY_RATE,
    FRESH_DAYS,
    FRESH_EXEC_AUC,
    FRESH_MIN_BUYS_PER_DAY,
    FRESH_POOLED_ENTRY_RATE,
    MAX_ORACLE_MEAN_REGRESSION_PP,
    MAX_POSITIVE_RATE_REGRESSION_PP,
    REQUEST_ID,
    _fit_exec_model,
    _safe_auc,
    _subset_metrics,
)
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS
from .extended_history_economic_opportunity import (
    _fit_extended_policy_selector,
)
from .hierarchical_attention_runtime import (
    add_stage2_scores,
    build_learned_runtime_trace,
    fit_stage2,
)
from .hot_economic_opportunity import ENTRY_FEATURES, _attach_entry_context

MINUTE_MS = 60_000


def _first_hot_features(
    scored_candidates: pd.DataFrame,
    scan: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    enriched = _attach_entry_context(scored_candidates, scan)
    trace, audit = build_learned_runtime_trace(enriched, enriched)
    first = (
        trace.loc[trace["state"].astype(str).eq("hot")]
        .sort_values(["trading_day", "ticker", "t"], kind="stable")
        .groupby(["trading_day", "ticker"], sort=False)
        .head(1)
        .loc[:, ["trading_day", "ticker", "t"]]
    )
    features = first.merge(
        enriched,
        on=["trading_day", "ticker", "t"],
        how="inner",
        validate="one_to_one",
    )
    return features, audit


def _vectorized_economics(
    features: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    events = features.reset_index(drop=True).copy()
    events["_event_id"] = np.arange(len(events), dtype=np.int64)
    events["ticker"] = events["ticker"].astype(str).str.upper()

    scan_prices = scan.loc[:, ["trading_day", "ticker", "t", "o"]].copy()
    scan_prices["ticker"] = scan_prices["ticker"].astype(str).str.upper()
    scan_prices["t"] = pd.to_numeric(
        scan_prices["t"], errors="coerce"
    ).astype("Int64")
    scan_prices["o"] = pd.to_numeric(scan_prices["o"], errors="coerce")
    scan_prices = scan_prices.loc[scan_prices["o"].gt(0)].drop_duplicates(
        ["trading_day", "ticker", "t"], keep="last"
    )
    scan_prices = scan_prices.rename(columns={"t": "target_t", "o": "price"})

    pieces = []
    for offset in range(1, 32):
        q = events.loc[:, ["_event_id", "trading_day", "ticker", "t"]].copy()
        q["target_t"] = (
            pd.to_numeric(q["t"], errors="raise").astype("int64")
            + offset * MINUTE_MS
        )
        q["offset"] = offset
        pieces.append(
            q.loc[
                :,
                ["_event_id", "trading_day", "ticker", "target_t", "offset"],
            ]
        )
    lookup = pd.concat(pieces, ignore_index=True).merge(
        scan_prices,
        on=["trading_day", "ticker", "target_t"],
        how="left",
        validate="many_to_one",
    )
    wide = lookup.pivot(index="_event_id", columns="offset", values="price")

    entry = events["_event_id"].map(wide.get(1, pd.Series(dtype=float)))
    events["entry_price"] = pd.to_numeric(entry, errors="coerce")
    events["entry_reference_available"] = events["entry_price"].gt(0)

    base = next(
        item for item in DEFAULT_EXECUTION_SCENARIOS if item.name == "base"
    )
    entry_price = events["entry_price"]
    entry_half = np.maximum(
        entry_price * base.half_spread_bps / 10_000.0,
        base.min_half_spread_cents / 100.0,
    )
    buy_fill = (
        entry_price * (1.0 + base.slippage_bps / 10_000.0) + entry_half
    )

    net_columns = []
    for minute in range(1, 31):
        exit_price = events["_event_id"].map(
            wide.get(minute + 1, pd.Series(dtype=float))
        )
        exit_price = pd.to_numeric(exit_price, errors="coerce")
        exit_half = np.maximum(
            exit_price * base.half_spread_bps / 10_000.0,
            base.min_half_spread_cents / 100.0,
        )
        sell_fill = (
            exit_price * (1.0 - base.slippage_bps / 10_000.0) - exit_half
        ).clip(lower=0)
        proceeds = sell_fill * (1.0 - base.sell_fee_bps / 10_000.0)
        net = (proceeds / buy_fill - 1.0) * 100.0
        net = net.where(entry_price.gt(0) & exit_price.gt(0))
        column = f"_net_{minute}"
        events[column] = net
        net_columns.append(column)

        if minute in {1, 2, 5, 10, 15, 30}:
            events[f"return_{minute}m_base_net_pct"] = net

    net_frame = events.loc[:, net_columns]
    events["oracle_best_base_pct"] = net_frame.max(axis=1, skipna=True)
    no_future = net_frame.notna().sum(axis=1).eq(0)
    events.loc[no_future, "oracle_best_base_pct"] = np.nan

    values = net_frame.to_numpy(dtype=float)
    all_nan = np.isnan(values).all(axis=1)
    safe = np.where(np.isnan(values), -np.inf, values)
    best_index = np.argmax(safe, axis=1) + 1
    events["oracle_best_minute"] = best_index.astype(float)
    events.loc[all_nan, "oracle_best_minute"] = np.nan
    events["oracle_base_positive"] = (
        events["oracle_best_base_pct"].gt(0).astype("boolean")
    )
    events.loc[
        events["oracle_best_base_pct"].isna(), "oracle_base_positive"
    ] = pd.NA

    return events.drop(columns=["_event_id", *net_columns])


def evaluate_dir(
    fresh_dir: Path,
    history_first_hot_path: Path,
    stage2_candidates_path: Path,
    output_path: Path,
) -> int:
    candidate_paths = sorted(fresh_dir.glob("*-candidates.parquet"))
    scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    audit_paths = sorted(fresh_dir.glob("*-audit.json"))
    if len(candidate_paths) != 5 or len(scan_paths) != 5:
        raise ValueError(
            "request 165 fast evaluation requires all five fresh shards"
        )

    fresh_candidates = pd.concat(
        [pd.read_parquet(path) for path in candidate_paths],
        ignore_index=True,
    )
    scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    history = pd.read_parquet(history_first_hot_path)
    stage2 = pd.read_parquet(stage2_candidates_path)

    minute_model, second_model = fit_stage2(stage2)
    fresh_scored = add_stage2_scores(
        fresh_candidates, minute_model, second_model
    )
    fresh, runtime_audit = _first_hot_features(fresh_scored, scan)
    fresh = _vectorized_economics(fresh, scan)

    economic_model, economic_gate = _fit_extended_policy_selector(history)
    exec_model, exec_cal = _fit_exec_model(history)

    econ_p = economic_model.predict_proba(
        fresh.loc[:, ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    exec_p = exec_model.predict_proba(
        fresh.loc[:, EXECUTION_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    fresh["economic_probability"] = econ_p
    fresh["execution_probability"] = exec_p

    economic_ok = fresh["economic_probability"].ge(float(economic_gate))
    execution_ok = fresh["execution_probability"].ge(
        float(exec_cal["threshold"])
    )
    fresh["action"] = np.where(
        economic_ok & execution_ok,
        "BUY",
        np.where(economic_ok, "WAIT", "ABSTAIN"),
    )

    baseline = _subset_metrics(fresh, economic_ok)
    candidate = _subset_metrics(fresh, fresh["action"].eq("BUY"))
    exec_auc = _safe_auc(
        fresh["entry_reference_available"].fillna(False).astype(int),
        exec_p,
    )

    by_day = {}
    support_ok = True
    daily_entry_ok = True
    for day in FRESH_DAYS:
        part = fresh.loc[fresh["trading_day"].astype(str).eq(day)].copy()
        econ_mask = part["economic_probability"].ge(float(economic_gate))
        buy_mask = part["action"].eq("BUY")
        metrics = {
            "first_hot_rows": int(len(part)),
            "actions": {
                key: int(value)
                for key, value in part["action"].value_counts().to_dict().items()
            },
            "economic_only": _subset_metrics(part, econ_mask),
            "buy": _subset_metrics(part, buy_mask),
        }
        by_day[day] = metrics
        if metrics["buy"]["rows"] < FRESH_MIN_BUYS_PER_DAY:
            support_ok = False
        rate = metrics["buy"]["entry_reference_rate"]
        if rate is None or rate < FRESH_DAILY_ENTRY_RATE:
            daily_entry_ok = False

    bp = baseline["oracle_positive_rate"]
    cp = candidate["oracle_positive_rate"]
    bm = baseline["oracle_mean_pct"]
    cm = candidate["oracle_mean_pct"]
    economic_nonregression = bool(
        bp is not None
        and cp is not None
        and cp >= bp - MAX_POSITIVE_RATE_REGRESSION_PP
        and bm is not None
        and cm is not None
        and cm >= bm - MAX_ORACLE_MEAN_REGRESSION_PP
    )
    gate_pass = bool(
        support_ok
        and daily_entry_ok
        and candidate["entry_reference_rate"] is not None
        and candidate["entry_reference_rate"] >= FRESH_POOLED_ENTRY_RATE
        and exec_auc is not None
        and exec_auc >= FRESH_EXEC_AUC
        and economic_nonregression
    )

    summary = {
        "schema_version": 2,
        "request_id": REQUEST_ID,
        "execution_mode": "parallel_fresh_shards_vectorized_economics",
        "fresh_days": FRESH_DAYS,
        "attention_changed": False,
        "economic_model_changed": False,
        "economic_gate": float(economic_gate),
        "execution_features": list(EXECUTION_FEATURES),
        "execution_model_kwargs": EXEC_MODEL_KWARGS,
        "execution_calibration": exec_cal,
        "fresh_execution_auc": exec_auc,
        "economic_only": baseline,
        "buy": candidate,
        "by_day": by_day,
        "economic_nonregression": economic_nonregression,
        "promotion_gate_pass": gate_pass,
        "runtime_audit": runtime_audit,
        "fresh_prepare_audits": [
            json.loads(path.read_text(encoding="utf-8"))
            for path in audit_paths
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--history-first-hot", type=Path, required=True)
    parser.add_argument("--stage2-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate_dir(
        args.fresh_dir,
        args.history_first_hot,
        args.stage2_candidates,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
