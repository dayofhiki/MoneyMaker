"""Request 165: causal entry-executability gate on the validated attention path.

Economic opportunity and attention remain frozen. A separate classifier learns
whether the exact next executable minute bar exists. The calibration threshold
is chosen only to reach a 90% observed-entry rate on already-opened calibration
history while retaining the largest support. Fresh dates are June 15,16,17,18,22.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .config import load_settings, require_flatfile_credentials
from .economic_signal_audit_vectorized import _add_fixed_horizon_returns
from .extended_history_economic_opportunity import (
    EXTENDED_CAL_DAYS,
    EXTENDED_FIT_DAYS,
    _fit_extended_policy_selector,
    _fit_frozen_attention,
)
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .fresh_validated_attention_economic_opportunity import _build_frozen_active
from .hierarchical_attention_runtime import add_stage2_scores, fit_stage2
from .hot_economic_opportunity import ENTRY_FEATURES, _first_hot_feature_rows
from .market_calendar import is_us_equity_trading_day
from .massive_client import MassiveClient
from .multi_day import daterange
from .second_path_attention_probe import SECOND_CACHE_DIR
from .targeted_second_hot_reranker import add_second_features
from .temporal_active_admission import HORIZON, add_cross_within_horizon_targets

REQUEST_ID = 165
FRESH_DAYS = [
    "2026-06-15",
    "2026-06-16",
    "2026-06-17",
    "2026-06-18",
    "2026-06-22",
]
EXECUTION_FEATURES = tuple(ENTRY_FEATURES)
EXEC_MODEL_KWARGS = {
    "learning_rate": 0.05,
    "max_iter": 160,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 60,
    "l2_regularization": 2.0,
    "random_state": 20261065,
}
CAL_TARGET_ENTRY_RATE = 0.90
CAL_MIN_SELECTED = 200
FRESH_MIN_BUYS_PER_DAY = 25
FRESH_POOLED_ENTRY_RATE = 0.90
FRESH_DAILY_ENTRY_RATE = 0.85
FRESH_EXEC_AUC = 0.55
MAX_POSITIVE_RATE_REGRESSION_PP = 0.02
MAX_ORACLE_MEAN_REGRESSION_PP = 0.25


def _fit_exec_model(history: pd.DataFrame):
    day = history["trading_day"].astype(str)
    fit = history.loc[day.isin(EXTENDED_FIT_DAYS)].copy()
    cal = history.loc[day.isin(EXTENDED_CAL_DAYS)].copy()

    y_fit = fit["entry_reference_available"].fillna(False).astype(int)
    y_cal = cal["entry_reference_available"].fillna(False).astype(int)
    if fit.empty or cal.empty or y_fit.nunique() < 2 or y_cal.nunique() < 2:
        raise ValueError("request 165 executability fit/cal needs both classes")

    model = HistGradientBoostingClassifier(**EXEC_MODEL_KWARGS)
    model.fit(
        fit.loc[:, EXECUTION_FEATURES].replace([np.inf, -np.inf], np.nan),
        y_fit,
    )
    cal_p = model.predict_proba(
        cal.loc[:, EXECUTION_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]

    candidates = np.unique(cal_p)
    candidates.sort()
    best = None
    for threshold in candidates:
        selected = cal_p >= float(threshold)
        count = int(selected.sum())
        if count < CAL_MIN_SELECTED:
            continue
        rate = float(y_cal.to_numpy()[selected].mean())
        if rate >= CAL_TARGET_ENTRY_RATE:
            if best is None or count > best["selected_rows"]:
                best = {
                    "threshold": float(threshold),
                    "selected_rows": count,
                    "entry_rate": rate,
                }
    if best is None:
        raise ValueError(
            "request 165 calibration cannot reach 90% entry observability "
            f"with >= {CAL_MIN_SELECTED} rows"
        )
    return model, best


def _safe_auc(y: pd.Series, score: np.ndarray) -> float | None:
    yy = pd.to_numeric(y, errors="coerce")
    valid = yy.notna() & np.isfinite(score)
    yy = yy.loc[valid].astype(int)
    ss = np.asarray(score)[valid.to_numpy()]
    if yy.nunique() < 2:
        return None
    return float(roc_auc_score(yy, ss))


def prepare_fresh(
    start: date,
    end: date,
    candidates_output: Path,
    scan_output: Path,
    audit_output: Path,
) -> int:
    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key),
        FLATFILE_CACHE_DIR,
    )
    scan_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=0.0,
    )
    second_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR,
        request_interval_seconds=0.0,
    )
    market_model, active_model = _fit_frozen_attention(
        store, scan_client, start, end
    )

    wanted = set(FRESH_DAYS)
    active_parts: list[pd.DataFrame] = []
    scan_parts: list[pd.DataFrame] = []

    for day in daterange(start, end):
        text = day.isoformat()
        if text not in wanted or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        rows = add_cross_within_horizon_targets(scan, horizons=(HORIZON,))
        active = _build_frozen_active(rows, market_model, active_model)
        active_parts.append(active)
        scan_parts.append(scan)

    if not active_parts:
        raise ValueError("request 165 fresh Active population is empty")

    active = pd.concat(active_parts, ignore_index=True)
    enriched, second_audit = add_second_features(active, second_client)
    scan = pd.concat(scan_parts, ignore_index=True)
    found = sorted(scan["trading_day"].astype(str).unique())
    if found != FRESH_DAYS:
        raise ValueError(f"request 165 expected {FRESH_DAYS}, found {found}")

    candidates_output.parent.mkdir(parents=True, exist_ok=True)
    scan_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(candidates_output, index=False, compression="zstd")
    scan.to_parquet(scan_output, index=False, compression="zstd")

    audit = {
        "request_id": REQUEST_ID,
        "fresh_days": FRESH_DAYS,
        "candidate_rows": int(len(enriched)),
        "scan_rows": int(len(scan)),
        "flatfile_stats": store.stats.to_dict(),
        "second_audit": second_audit,
        "second_client_stats": second_client.stats.to_dict(),
    }
    audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def _subset_metrics(frame: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    part = frame.loc[mask].copy()
    entry = part["entry_reference_available"].fillna(False).astype(bool)
    oracle = pd.to_numeric(part["oracle_best_base_pct"], errors="coerce")
    labeled = oracle.notna()
    result = {
        "rows": int(len(part)),
        "entry_reference_rate": float(entry.mean()) if len(part) else None,
        "economic_label_coverage": float(labeled.mean()) if len(part) else None,
        "labeled_rows": int(labeled.sum()),
        "oracle_mean_pct": (
            float(oracle.loc[labeled].mean()) if labeled.any() else None
        ),
        "oracle_positive_rate": (
            float(oracle.loc[labeled].gt(0).mean()) if labeled.any() else None
        ),
    }
    horizon_rows = {}
    for horizon in (1, 2, 5, 10, 15, 30):
        column = f"return_{horizon}m_base_net_pct"
        values = pd.to_numeric(part.get(column), errors="coerce")
        valid = values.notna()
        horizon_rows[str(horizon)] = {
            "rows": int(valid.sum()),
            "coverage": float(valid.mean()) if len(part) else None,
            "mean_net_pct": (
                float(values.loc[valid].mean()) if valid.any() else None
            ),
            "positive_rate": (
                float(values.loc[valid].gt(0).mean()) if valid.any() else None
            ),
        }
    result["fixed_horizons"] = horizon_rows
    return result


def evaluate(
    history_first_hot_path: Path,
    stage2_candidates_path: Path,
    fresh_candidates_path: Path,
    fresh_scan_path: Path,
    fresh_prepare_audit_path: Path,
    output_path: Path,
) -> int:
    history = pd.read_parquet(history_first_hot_path)
    stage2 = pd.read_parquet(stage2_candidates_path)
    fresh_candidates = pd.read_parquet(fresh_candidates_path)

    minute_model, second_model = fit_stage2(stage2)
    fresh_scored = add_stage2_scores(
        fresh_candidates, minute_model, second_model
    )
    scan = pd.read_parquet(fresh_scan_path)
    fresh, runtime_audit = _first_hot_feature_rows(fresh_scored, scan)
    fresh = _add_fixed_horizon_returns(fresh, fresh_scan_path)

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

    by_day: dict[str, object] = {}
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
        buy_rows = metrics["buy"]["rows"]
        entry_rate = metrics["buy"]["entry_reference_rate"]
        if buy_rows < FRESH_MIN_BUYS_PER_DAY:
            support_ok = False
        if entry_rate is None or entry_rate < FRESH_DAILY_ENTRY_RATE:
            daily_entry_ok = False

    baseline_pos = baseline["oracle_positive_rate"]
    candidate_pos = candidate["oracle_positive_rate"]
    baseline_mean = baseline["oracle_mean_pct"]
    candidate_mean = candidate["oracle_mean_pct"]

    economic_nonregression = bool(
        baseline_pos is not None
        and candidate_pos is not None
        and candidate_pos >= baseline_pos - MAX_POSITIVE_RATE_REGRESSION_PP
        and baseline_mean is not None
        and candidate_mean is not None
        and candidate_mean >= baseline_mean - MAX_ORACLE_MEAN_REGRESSION_PP
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
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "fresh_days": FRESH_DAYS,
        "attention_changed": False,
        "economic_model_changed": False,
        "economic_gate": float(economic_gate),
        "execution_features": list(EXECUTION_FEATURES),
        "execution_model_kwargs": EXEC_MODEL_KWARGS,
        "execution_calibration": exec_cal,
        "fresh_execution_auc": exec_auc,
        "action_semantics": {
            "BUY": "economic opportunity passes and execution gate passes",
            "WAIT": "economic opportunity passes but execution gate fails",
            "ABSTAIN": "economic opportunity gate fails",
        },
        "economic_only": baseline,
        "buy": candidate,
        "by_day": by_day,
        "economic_nonregression": economic_nonregression,
        "promotion_gate_pass": gate_pass,
        "runtime_audit": runtime_audit,
        "fresh_prepare_audit": json.loads(
            fresh_prepare_audit_path.read_text(encoding="utf-8")
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("start", type=date.fromisoformat)
    prep.add_argument("end", type=date.fromisoformat)
    prep.add_argument("--candidates-output", type=Path, required=True)
    prep.add_argument("--scan-output", type=Path, required=True)
    prep.add_argument("--audit-output", type=Path, required=True)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--history-first-hot", type=Path, required=True)
    ev.add_argument("--stage2-candidates", type=Path, required=True)
    ev.add_argument("--fresh-candidates", type=Path, required=True)
    ev.add_argument("--fresh-scan", type=Path, required=True)
    ev.add_argument("--fresh-prepare-audit", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        return prepare_fresh(
            args.start,
            args.end,
            args.candidates_output,
            args.scan_output,
            args.audit_output,
        )
    return evaluate(
        args.history_first_hot,
        args.stage2_candidates,
        args.fresh_candidates,
        args.fresh_scan,
        args.fresh_prepare_audit,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
