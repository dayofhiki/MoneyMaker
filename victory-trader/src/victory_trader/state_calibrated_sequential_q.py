from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import state_fitted_sequential_q as base
from .state_multi_source_value import EXTERNAL_FEATURES, _coverage_row
from .state_rank_turn import day_cluster_bootstrap
from .state_sequence_enrichment import SEQUENCE_FEATURES
from .state_value_model import (
    BOOLEAN_FEATURES,
    LOG_FEATURES,
    RAW_FEATURES,
    _eligible,
)

CALIBRATION_FRACTION = 0.20
OPTIMISM_MULTIPLIER = 1.645
MIN_CALIBRATION_DAYS = 5
POLICY = "calibrated_sequential_q_cap1"

MODEL_INPUT_COLUMNS = tuple(
    dict.fromkeys(
        [
            *base.EPISODE_KEYS,
            "t",
            "c",
            "previous_close",
            "entry_price",
            "active_minute_fraction_15m",
            *RAW_FEATURES,
            *BOOLEAN_FEATURES,
            *LOG_FEATURES,
            *SEQUENCE_FEATURES,
            *EXTERNAL_FEATURES,
            base.GROSS_COLUMN,
            base.TARGET_COLUMN,
            base.STRESS_COLUMN,
            "eight_k_query_complete",
            "short_interest_query_complete",
            "short_ratio_latest_prior",
            "short_interest_latest",
        ]
    )
)


def read_model_panel(path: str | Path) -> pd.DataFrame:
    """Read only columns consumed by the frozen model and its audit outputs."""
    available = set(pd.read_parquet(path, columns=[]).columns)
    # Some parquet engines return an empty column list for columns=[]; inspect
    # metadata without materializing row data when pyarrow is available.
    if not available:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema.names)
    columns = [column for column in MODEL_INPUT_COLUMNS if column in available]
    return pd.read_parquet(path, columns=columns)


@dataclass(frozen=True)
class AdjustedModel:
    fitted: base.QModel
    correction: float


def predict(frame: pd.DataFrame, model: AdjustedModel) -> np.ndarray:
    return base._predict_q(frame, model.fitted) - model.correction


def checkpoints(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    out = frame.loc[_eligible(frame)].copy()
    out["_time"] = pd.to_numeric(out["t"], errors="coerce")
    out = out.dropna(subset=["_time"])
    first = out.groupby(base.EPISODE_KEYS)["_time"].transform("min")
    out["_offset"] = (out["_time"] - first) / base.MINUTE_MS
    return {
        offset: out.loc[out["_offset"].eq(offset)]
        .sort_values("_time", kind="stable")
        .drop_duplicates(base.EPISODE_KEYS, keep="last")
        .drop(columns=["_time", "_offset"])
        .reset_index(drop=True)
        for offset in base.CHECKPOINT_OFFSETS
    }


def split_fit_calibration(frame: pd.DataFrame):
    days = sorted(frame["trading_day"].astype(str).unique())
    cut = int(len(days) * (1 - CALIBRATION_FRACTION))
    if cut < 6 or len(days) - cut < MIN_CALIBRATION_DAYS:
        raise ValueError("insufficient independent fit/calibration days")
    fit_days = set(days[:cut])
    mask = frame["trading_day"].astype(str).isin(fit_days)
    return frame.loc[mask].copy(), frame.loc[~mask].copy()


def optimism_allowance(prediction, actual, days):
    p = np.asarray(prediction, dtype=float)
    y = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy()
    d = pd.Series(np.asarray(days).astype(str))
    selected = np.isfinite(p) & np.isfinite(y) & (p > 0)
    errors = pd.DataFrame({"day": d[selected], "error": p[selected] - y[selected]})
    daily = errors.groupby("day")["error"].mean()
    enough = len(daily) >= MIN_CALIBRATION_DAYS
    mean = float(daily.mean()) if len(daily) else np.nan
    se = float(daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 else np.nan
    correction = max(0.0, mean + OPTIMISM_MULTIPLIER * se) if enough else np.inf
    return correction, {
        "selected_rows": int(selected.sum()),
        "calibration_days": int(len(daily)),
        "mean_selected_optimism_pct": mean,
        "day_standard_error_pct": se,
        "correction_pct": correction,
        "disabled_insufficient_days": not enough,
    }


def calibrate(fitted, frame, target, name, records):
    correction, diagnostic = optimism_allowance(
        base._predict_q(frame, fitted), target, frame["trading_day"]
    )
    records.append({"model": name, **diagnostic})
    return AdjustedModel(fitted, correction)


def aligned_values(destination, source, values):
    source_index = pd.MultiIndex.from_frame(source[base.EPISODE_KEYS].astype(str))
    dest_index = pd.MultiIndex.from_frame(destination[base.EPISODE_KEYS].astype(str))
    series = pd.Series(np.asarray(values, dtype=float), index=source_index)
    # Missing checkpoint means SKIP=0. Existing missing evaluation labels remain NaN.
    aligned = series.reindex(dest_index)
    missing = ~dest_index.isin(source_index)
    result = aligned.to_numpy(dtype=float, copy=True)
    result[missing] = 0.0
    return pd.Series(result, index=destination.index, dtype=float)


def terminal_values(frame, model):
    q = predict(frame, model)
    actual = pd.to_numeric(frame[base.TARGET_COLUMN], errors="coerce").to_numpy()
    return np.where(q > 0, actual, 0.0)


def mid_values(stage5, stage10, buy5, wait5, buy10):
    buy = predict(stage5, buy5)
    wait = predict(stage5, wait5)
    later = aligned_values(stage5, stage10, terminal_values(stage10, buy10)).to_numpy()
    actual = pd.to_numeric(stage5[base.TARGET_COLUMN], errors="coerce").to_numpy()
    choose_wait = (wait > 0) & (wait >= buy)
    choose_buy = (buy > 0) & (buy > wait)
    return np.where(choose_wait, later, np.where(choose_buy, actual, 0.0))


def train_chain(fit, calibration=None):
    partitions = base.split_training_days(fit)
    cp = checkpoints(fit)
    cal = checkpoints(calibration) if calibration is not None else None
    records = []

    def subset(stage, part):
        return cp[stage].loc[cp[stage]["trading_day"].astype(str).isin(part)].copy()

    def finish(fitted, stage, target, name):
        if cal is None:
            return AdjustedModel(fitted, 0.0)
        return calibrate(fitted, cal[stage], target, name, records)

    a10 = subset(10, partitions[0])
    q10 = base._fit_q_model(a10, a10[base.TARGET_COLUMN], seed=2026092710)
    buy10 = finish(q10, 10, cal[10][base.TARGET_COLUMN] if cal else None, "q10_buy")
    b5, b10 = subset(5, partitions[1]), subset(10, partitions[1])
    q5b = base._fit_q_model(b5, b5[base.TARGET_COLUMN], seed=2026092705)
    buy5 = finish(q5b, 5, cal[5][base.TARGET_COLUMN] if cal else None, "q5_buy")
    w5 = aligned_values(b5, b10, terminal_values(b10, buy10))
    q5w = base._fit_q_model(b5, w5, seed=2026092755)
    calw5 = aligned_values(cal[5], cal[10], terminal_values(cal[10], buy10)) if cal else None
    wait5 = finish(q5w, 5, calw5, "q5_wait")
    c0, c5, c10 = [subset(s, partitions[2]) for s in (0, 5, 10)]
    q0b = base._fit_q_model(c0, c0[base.TARGET_COLUMN], seed=2026092700)
    buy0 = finish(q0b, 0, cal[0][base.TARGET_COLUMN] if cal else None, "q0_buy")
    w0 = aligned_values(c0, c5, mid_values(c5, c10, buy5, wait5, buy10))
    q0w = base._fit_q_model(c0, w0, seed=2026092750)
    calw0 = aligned_values(cal[0], cal[5], mid_values(cal[5], cal[10], buy5, wait5, buy10)) if cal else None
    wait0 = finish(q0w, 0, calw0, "q0_wait")
    return {0: (buy0, wait0), 5: (buy5, wait5), 10: (buy10, None)}, records


def attempt_diagnostic(row, stage, trade):
    """Post-decision observation only; never a feature or action gate."""
    values = {key: pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
              for key in ("entry_price", base.GROSS_COLUMN, base.TARGET_COLUMN, base.STRESS_COLUMN)}
    entry = values["entry_price"]
    flags = {
        "entry_missing": bool(pd.isna(entry)),
        "entry_nonpositive": bool(pd.notna(entry) and entry <= 0),
        "entry_nonfinite": bool(pd.notna(entry) and not np.isfinite(entry)),
        "gross_missing": bool(pd.isna(values[base.GROSS_COLUMN])),
        "base_missing": bool(pd.isna(values[base.TARGET_COLUMN])),
        "stress_missing": bool(pd.isna(values[base.STRESS_COLUMN])),
        "return_nonfinite": any(pd.notna(values[k]) and not np.isfinite(values[k])
                                for k in (base.GROSS_COLUMN, base.TARGET_COLUMN, base.STRESS_COLUMN)),
    }
    # Mutually exclusive reason with full overlapping flags retained.
    if flags["entry_missing"]:
        reason = "entry_missing"
    elif flags["entry_nonpositive"] or flags["entry_nonfinite"]:
        reason = "entry_invalid"
    elif flags["gross_missing"]:
        reason = "gross_label_missing"
    elif flags["base_missing"] or flags["stress_missing"]:
        reason = "scenario_label_missing"
    elif flags["return_nonfinite"]:
        reason = "return_nonfinite"
    else:
        reason = "evaluated"
    return {"trading_day": row.get("trading_day"), "ticker": row.get("ticker"),
            "decision_t": row.get("t"), "checkpoint_min": stage,
            "q_buy": row.get("_buy", 0.0), "q_wait": row.get("_wait", np.nan),
            "evaluation_reason": reason, "legacy_evaluated": trade is not None,
            **flags, **values}


def select_policy(cp, models, *, attempt_records=None):
    # Batch inference once per model, not once per ticker-day.
    maps = {}
    for stage, frame in cp.items():
        buy_model, wait_model = models[stage]
        scored = frame.copy()
        scored["_buy"] = predict(frame, buy_model)
        scored["_wait"] = predict(frame, wait_model) if wait_model else -np.inf
        maps[stage] = base._episode_map(scored)
    paths = {"episodes": len(maps[0]), "buy_at_0": 0, "buy_at_5": 0,
             "buy_at_10": 0, "skip": 0, "missing_checkpoint": 0,
             "buy_attempts": 0, "unevaluated_attempts": 0}
    trades = []
    for key in maps[0]:
        for stage in (0, 5, 10):
            row = maps[stage].get(key)
            if row is None:
                paths["missing_checkpoint"] += 1
                break
            action = base._choose_mid_action(row["_buy"], row["_wait"])
            if action == "SKIP":
                paths["skip"] += 1
                break
            if action == "BUY":
                paths["buy_attempts"] += 1
                trade = base._attempt_trade(row, decision_score=row["_buy"])
                if attempt_records is not None:
                    attempt_records.append(attempt_diagnostic(row, stage, trade))
                if trade is None:
                    paths["unevaluated_attempts"] += 1
                else:
                    paths[f"buy_at_{stage}"] += 1
                    trades.append(trade)
                break
    return pd.DataFrame(trades), paths


def evaluation_folds(
    monthly,
    mode,
    *,
    evaluation_min_month=None,
    evaluation_max_month=None,
):
    """Forward audit uses only earlier months, with two training months minimum."""
    labels = sorted(monthly)
    for month in labels:
        if evaluation_min_month is not None and month < evaluation_min_month:
            continue
        if evaluation_max_month is not None and month > evaluation_max_month:
            continue
        training = [m for m in labels if m != month and (mode == "lomo" or m < month)]
        if mode == "forward" and len(training) < 2:
            continue
        holdout = monthly[month]
        if mode == "forward":
            last_training_day = max(monthly[m]["trading_day"].astype(str).max() for m in training)
            if last_training_day >= holdout["trading_day"].astype(str).min():
                raise ValueError("forward training overlaps or follows evaluation dates")
        yield month, training, holdout


def policy_bootstrap(trades):
    selected = trades.loc[trades["policy"].eq(POLICY)] if "policy" in trades else pd.DataFrame()
    daily = (selected.assign(_base=pd.to_numeric(selected["realized_base_net_return_pct"], errors="coerce"))
             .groupby("trading_day")["_base"].mean().dropna()) if not selected.empty else pd.Series(dtype=float)
    if len(daily) < 2:
        return {"days": len(daily), "day_balanced_mean_pct": float(daily.mean()),
                "ci_low_pct": np.nan, "ci_high_pct": np.nan,
                "bootstrap_available": False,
                "reason": "fewer than two independent evaluated trading days"}
    return {**day_cluster_bootstrap(trades, policy=POLICY, samples=10000),
            "bootstrap_available": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=base._parse_dataset, required=True)
    parser.add_argument("--evaluation", choices=("lomo", "forward"), default="lomo")
    parser.add_argument("--evaluation-min-month")
    parser.add_argument("--evaluation-max-month")
    parser.add_argument("--attempts-csv", type=Path)
    for name in ("report", "details", "trades", "diagnostics", "comparisons", "coverage", "paths", "calibration"):
        parser.add_argument("--" + name + ("" if name == "report" else "-csv"), type=Path, required=True)
    args = parser.parse_args()
    monthly = {label: read_model_panel(path) for label, path in args.dataset}
    details, trades, comparisons, coverage, paths, calibration_rows, diagnostics = [], [], [], [], [], [], []
    attempts, provenance = [], []
    for month, training, holdout in evaluation_folds(
        monthly,
        args.evaluation,
        evaluation_min_month=args.evaluation_min_month,
        evaluation_max_month=args.evaluation_max_month,
    ):
        train = pd.concat([monthly[m] for m in training], ignore_index=True)
        fit, cal = split_fit_calibration(train)
        provenance.append({"month": month, "mode": args.evaluation,
                           "fit_start": fit.trading_day.min(), "fit_end": fit.trading_day.max(),
                           "cal_start": cal.trading_day.min(), "cal_end": cal.trading_day.max(),
                           "eval_start": holdout.trading_day.min(), "eval_end": holdout.trading_day.max()})
        # Do not retain the full concat alongside its disjoint copied splits.
        del train
        gc.collect()
        corrected, records = train_chain(fit, cal)
        raw, _ = train_chain(fit)
        # Q models retain only fitted estimators, not the full state panels.
        del fit, cal
        gc.collect()
        cp = checkpoints(holdout)
        earliest = []
        for _, r in cp[0].iterrows():
            trade = base._attempt_trade(r, decision_score=0.0)
            earliest.append(trade)
            attempts.append({"month": month, "policy": "earliest_eligible_10m_cap1",
                             **attempt_diagnostic(r, 0, trade)})
        baseline = pd.DataFrame([r for r in earliest if r is not None])
        adjusted_attempts, raw_attempts = [], []
        adjusted, path = select_policy(cp, corrected, attempt_records=adjusted_attempts)
        raw_trades, raw_paths = select_policy(cp, raw, attempt_records=raw_attempts)
        for name, records_for_policy, counts in ((POLICY, adjusted_attempts, path),
                                                ("raw_same_fit_cap1", raw_attempts, raw_paths)):
            assert len(records_for_policy) == counts["buy_attempts"]
            assert sum(not r["legacy_evaluated"] for r in records_for_policy) == counts["unevaluated_attempts"]
            attempts.extend({"month": month, "policy": name, **r} for r in records_for_policy)
        coverage.append(_coverage_row(holdout, month))
        for policy, selected, counts in (("earliest_eligible_10m_cap1", baseline, {}),
                                         ("raw_same_fit_cap1", raw_trades, raw_paths),
                                         (POLICY, adjusted, path)):
            details.append(base._metrics(selected, month=month, policy=policy))
            paths.append({"month": month, "policy": policy, **counts})
            if not selected.empty:
                selected = selected.copy()
                selected["month"], selected["policy"] = month, policy
                selected["modeled_friction_pct"] = selected.realized_gross_return_pct - selected.realized_base_net_return_pct
                trades.append(selected)
        comparisons.append(base.common_episode_comparison(baseline, adjusted, month=month))
        calibration_rows.extend({"month": month, **r} for r in records)
        p = predict(cp[0], corrected[0][0])
        y = pd.to_numeric(cp[0][base.TARGET_COLUMN], errors="coerce")
        finite = np.isfinite(p)
        diagnostics.append({"month": month, "q0_buy_spearman": base._spearman(p[finite], y.loc[finite]) if finite.any() else np.nan})
        print(f"Finished holdout {month}", flush=True)
        print(pd.DataFrame(details[-3:]).to_string(index=False), flush=True)
        print(pd.DataFrame(records).to_string(index=False), flush=True)
        del cp, corrected, raw
        gc.collect()
    frames = {"details": pd.DataFrame(details), "trades": pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
              "diagnostics": pd.DataFrame(diagnostics), "comparisons": pd.DataFrame(comparisons),
              "coverage": pd.DataFrame(coverage), "paths": pd.DataFrame(paths), "calibration": pd.DataFrame(calibration_rows)}
    boot = policy_bootstrap(frames["trades"])
    report = "=== Selected-tail calibrated sequential Q v1.8 ===\nAdaptive development only; no fresh month\n"
    report += f"evaluation={args.evaluation}; forward is also previously seen development, not fresh validation\n"
    report += "\n=== Split provenance ===\n" + pd.DataFrame(provenance).to_string(index=False) + "\n"
    attempt_frame = pd.DataFrame(attempts)
    if not attempt_frame.empty:
        summary = attempt_frame.groupby(["month", "policy", "evaluation_reason"], dropna=False).size().rename("attempts").reset_index()
        report += "\n=== Attempt evaluation reasons (post-decision only) ===\n" + summary.to_string(index=False) + "\n"
        if attempt_frame["entry_nonfinite"].any() or attempt_frame["return_nonfinite"].any():
            report += "DATA QUALITY FAILURE: nonfinite values; no performance promotion allowed.\n"
    for name in ("details", "calibration", "paths", "comparisons", "coverage", "diagnostics"):
        report += f"\n=== {name} ===\n" + frames[name].to_string(index=False) + "\n"
    report += "\n=== Day-cluster bootstrap ===\n" + str(boot) + "\n"
    report += "\nNo promotion without checking every preregistered criterion. Zero trades is not a profitable strategy.\n"
    print(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    if args.attempts_csv is not None:
        args.attempts_csv.parent.mkdir(parents=True, exist_ok=True)
        attempt_frame.to_csv(args.attempts_csv, index=False)
    for name, frame in frames.items():
        path = getattr(args, name + "_csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
