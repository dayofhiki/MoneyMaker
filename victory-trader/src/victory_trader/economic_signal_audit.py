"""Request 164: economic signal audit and fixed-horizon return bridge.

Uses the frozen Request-163 prepared populations. No new dates are opened and
no model/threshold is changed. The goal is diagnostic:
1) explain missing economic labels;
2) identify which causal ENTRY_FEATURES carry stable economic signal;
3) test the already-frozen opportunity selector against fixed, causal exit
   horizons instead of only hindsight best-within-30m opportunity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .extended_history_economic_opportunity import (
    EXTENDED_CAL_DAYS,
    EXTENDED_FIT_DAYS,
    FRESH_EVAL_DAYS,
    _fit_extended_policy_selector,
)
from .hierarchical_attention_runtime import (
    add_stage2_scores,
    build_learned_runtime_trace,
    fit_stage2,
)
from .hot_economic_opportunity import ENTRY_FEATURES, _attach_entry_context
from .hot_entry_economics import (
    HORIZONS,
    first_hot_events,
    label_first_hot_economics,
)

REQUEST_ID = 164


def _safe_auc(y: pd.Series, score: pd.Series) -> float | None:
    yv = pd.to_numeric(y, errors="coerce")
    sv = pd.to_numeric(score, errors="coerce")
    valid = yv.notna() & sv.notna()
    yv = yv.loc[valid].astype(int)
    sv = sv.loc[valid].astype(float)
    if len(yv) < 20 or yv.nunique() < 2 or sv.nunique() < 2:
        return None
    return float(roc_auc_score(yv, sv))


def _safe_spearman(a: pd.Series, b: pd.Series) -> float | None:
    av = pd.to_numeric(a, errors="coerce")
    bv = pd.to_numeric(b, errors="coerce")
    valid = av.notna() & bv.notna()
    if int(valid.sum()) < 20:
        return None
    value = av.loc[valid].corr(bv.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _build_first_hot(
    stage2_candidates: pd.DataFrame,
    opportunity_candidates: pd.DataFrame,
    opportunity_scan: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    minute_model, second_model = fit_stage2(stage2_candidates)
    scored = add_stage2_scores(
        opportunity_candidates,
        minute_model,
        second_model,
    )
    enriched = _attach_entry_context(scored, opportunity_scan)
    trace, runtime_audit = build_learned_runtime_trace(enriched, enriched)

    first_hot = first_hot_events(trace).loc[
        :, ["trading_day", "ticker", "t"]
    ].copy()
    features = first_hot.merge(
        enriched,
        on=["trading_day", "ticker", "t"],
        how="inner",
        validate="one_to_one",
    )

    labels = label_first_hot_economics(trace, opportunity_scan).rename(
        columns={"hot_t": "t"}
    )
    economic_columns = [
        "trading_day",
        "ticker",
        "t",
        "entry_reference_available",
        "entry_price",
        "oracle_best_base_pct",
        "oracle_best_minute",
        "oracle_base_positive",
    ]
    for horizon in HORIZONS:
        for column in (
            f"return_{horizon}m_gross_pct",
            f"return_{horizon}m_base_net_pct",
        ):
            if column in labels.columns:
                economic_columns.append(column)

    features = features.merge(
        labels.loc[:, economic_columns],
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
    )
    return features, runtime_audit


def _missing_reason(row: pd.Series) -> str:
    label = pd.to_numeric(
        pd.Series([row.get("oracle_best_base_pct")]), errors="coerce"
    ).iloc[0]
    if pd.notna(label):
        return "labeled"

    entry = bool(row.get("entry_reference_available", False))
    to_close = pd.to_numeric(
        pd.Series([row.get("minutes_to_close")]), errors="coerce"
    ).iloc[0]

    if not entry:
        if pd.notna(to_close) and float(to_close) <= 1.0:
            return "near_close_no_next_minute_entry"
        return "mid_session_no_exact_next_minute_entry"

    if pd.notna(to_close) and float(to_close) <= 31.0:
        return "near_close_no_future_exit"
    return "mid_session_no_future_exit_bar"


def _missing_audit(features: pd.DataFrame) -> dict[str, object]:
    fresh = features.loc[
        features["trading_day"].astype(str).isin(FRESH_EVAL_DAYS)
    ].copy()
    fresh["missing_reason"] = fresh.apply(_missing_reason, axis=1)
    counts = fresh["missing_reason"].value_counts(dropna=False).to_dict()
    total = len(fresh)
    rows = []
    for reason, count in counts.items():
        rows.append(
            {
                "reason": str(reason),
                "rows": int(count),
                "share": float(count / total) if total else None,
            }
        )
    rows.sort(key=lambda x: x["rows"], reverse=True)

    by_day = {}
    for day in FRESH_EVAL_DAYS:
        part = fresh.loc[fresh["trading_day"].astype(str).eq(day)]
        vc = part["missing_reason"].value_counts().to_dict()
        by_day[day] = {
            "first_hot_rows": int(len(part)),
            "labeled_rows": int(vc.get("labeled", 0)),
            "label_coverage": (
                float(vc.get("labeled", 0) / len(part)) if len(part) else None
            ),
            "reason_counts": {str(k): int(v) for k, v in vc.items()},
        }
    return {"pooled": rows, "by_day": by_day}


def _signal_rows(features: pd.DataFrame) -> list[dict[str, object]]:
    development_days = set(EXTENDED_FIT_DAYS + EXTENDED_CAL_DAYS)
    dev = features.loc[
        features["trading_day"].astype(str).isin(development_days)
    ].copy()
    fresh = features.loc[
        features["trading_day"].astype(str).isin(FRESH_EVAL_DAYS)
    ].copy()

    rows: list[dict[str, object]] = []
    for feature in ENTRY_FEATURES:
        if feature not in features.columns:
            continue
        dtarget = pd.to_numeric(dev["oracle_best_base_pct"], errors="coerce")
        dscore = pd.to_numeric(dev[feature], errors="coerce")
        dvalid = dtarget.notna() & dscore.notna()
        dy = dtarget.loc[dvalid].gt(0).astype(int)
        dscore = dscore.loc[dvalid]

        ftarget = pd.to_numeric(fresh["oracle_best_base_pct"], errors="coerce")
        fscore = pd.to_numeric(fresh[feature], errors="coerce")
        fvalid = ftarget.notna() & fscore.notna()
        fy = ftarget.loc[fvalid].gt(0).astype(int)
        fscore = fscore.loc[fvalid]

        dev_auc = _safe_auc(dy, dscore)
        fresh_auc = _safe_auc(fy, fscore)
        direction = None
        if dev_auc is not None:
            direction = "higher" if dev_auc >= 0.5 else "lower"
        fresh_aligned_auc = None
        if fresh_auc is not None and direction is not None:
            fresh_aligned_auc = (
                fresh_auc if direction == "higher" else 1.0 - fresh_auc
            )

        dev_pos = dscore.loc[dy.eq(1)]
        dev_neg = dscore.loc[dy.eq(0)]
        fresh_pos = fscore.loc[fy.eq(1)]
        fresh_neg = fscore.loc[fy.eq(0)]

        rows.append(
            {
                "feature": feature,
                "direction_from_development": direction,
                "development_rows": int(len(dscore)),
                "development_auc_aligned": (
                    max(dev_auc, 1.0 - dev_auc)
                    if dev_auc is not None
                    else None
                ),
                "development_spearman": _safe_spearman(
                    dtarget.loc[dvalid], dscore
                ),
                "development_positive_median": (
                    float(dev_pos.median()) if len(dev_pos) else None
                ),
                "development_negative_median": (
                    float(dev_neg.median()) if len(dev_neg) else None
                ),
                "fresh_rows": int(len(fscore)),
                "fresh_auc_aligned_to_development": fresh_aligned_auc,
                "fresh_raw_spearman": _safe_spearman(
                    ftarget.loc[fvalid], fscore
                ),
                "fresh_positive_median": (
                    float(fresh_pos.median()) if len(fresh_pos) else None
                ),
                "fresh_negative_median": (
                    float(fresh_neg.median()) if len(fresh_neg) else None
                ),
                "direction_consistent": (
                    None
                    if direction is None or fresh_auc is None
                    else (
                        fresh_auc >= 0.5
                        if direction == "higher"
                        else fresh_auc <= 0.5
                    )
                ),
            }
        )

    rows.sort(
        key=lambda x: (
            x["development_auc_aligned"]
            if x["development_auc_aligned"] is not None
            else -1
        ),
        reverse=True,
    )
    return rows


def _fixed_horizon_audit(features: pd.DataFrame) -> dict[str, object]:
    classifier, gate = _fit_extended_policy_selector(features)
    fresh = features.loc[
        features["trading_day"].astype(str).isin(FRESH_EVAL_DAYS)
    ].copy()
    x = fresh[ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan)
    fresh["opportunity_probability"] = classifier.predict_proba(x)[:, 1]
    fresh["selected"] = fresh["opportunity_probability"].ge(gate)

    oracle = pd.to_numeric(fresh["oracle_best_base_pct"], errors="coerce")
    oracle_valid = oracle.notna()
    y = oracle.loc[oracle_valid].gt(0).astype(int)
    auc = _safe_auc(
        y,
        fresh.loc[oracle_valid, "opportunity_probability"],
    )

    results = []
    for horizon in HORIZONS:
        column = f"return_{horizon}m_base_net_pct"
        if column not in fresh.columns:
            continue
        values = pd.to_numeric(fresh[column], errors="coerce")
        available = values.notna()
        selected_available = available & fresh["selected"]

        all_values = values.loc[available]
        selected_values = values.loc[selected_available]

        positive_days = 0
        nonlower_days = 0
        day_rows = {}
        for day in FRESH_EVAL_DAYS:
            day_mask = fresh["trading_day"].astype(str).eq(day)
            day_all = values.loc[day_mask & available]
            day_selected = values.loc[
                day_mask & selected_available
            ]
            all_mean = float(day_all.mean()) if len(day_all) else None
            sel_mean = (
                float(day_selected.mean()) if len(day_selected) else None
            )
            if sel_mean is not None and sel_mean > 0:
                positive_days += 1
            if (
                sel_mean is not None
                and all_mean is not None
                and sel_mean >= all_mean
            ):
                nonlower_days += 1
            day_rows[day] = {
                "all_rows": int(len(day_all)),
                "selected_rows": int(len(day_selected)),
                "all_mean_net_pct": all_mean,
                "selected_mean_net_pct": sel_mean,
                "selected_positive_rate": (
                    float(day_selected.gt(0).mean())
                    if len(day_selected)
                    else None
                ),
            }

        results.append(
            {
                "horizon_minutes": int(horizon),
                "all_available_rows": int(len(all_values)),
                "all_coverage": float(available.mean()),
                "all_mean_net_pct": (
                    float(all_values.mean()) if len(all_values) else None
                ),
                "all_median_net_pct": (
                    float(all_values.median()) if len(all_values) else None
                ),
                "all_positive_rate": (
                    float(all_values.gt(0).mean())
                    if len(all_values)
                    else None
                ),
                "selected_available_rows": int(len(selected_values)),
                "selected_coverage_within_selected": (
                    float(
                        selected_available.sum()
                        / max(1, int(fresh["selected"].sum()))
                    )
                ),
                "selected_mean_net_pct": (
                    float(selected_values.mean())
                    if len(selected_values)
                    else None
                ),
                "selected_median_net_pct": (
                    float(selected_values.median())
                    if len(selected_values)
                    else None
                ),
                "selected_positive_rate": (
                    float(selected_values.gt(0).mean())
                    if len(selected_values)
                    else None
                ),
                "selected_minus_all_mean_pp": (
                    float(selected_values.mean() - all_values.mean())
                    if len(selected_values) and len(all_values)
                    else None
                ),
                "selected_positive_mean_days": positive_days,
                "selected_nonlower_mean_days": nonlower_days,
                "by_day": day_rows,
            }
        )

    return {
        "selector_gate": float(gate),
        "fresh_oracle_classifier_auc_reproduced": auc,
        "fresh_first_hot_rows": int(len(fresh)),
        "fresh_selected_rows": int(fresh["selected"].sum()),
        "horizons": results,
    }


def run(
    stage2_candidates_path: Path,
    opportunity_candidates_path: Path,
    opportunity_scan_path: Path,
    output_json: Path,
    signal_csv: Path,
    horizon_csv: Path,
) -> int:
    stage2 = pd.read_parquet(stage2_candidates_path)
    opportunity = pd.read_parquet(opportunity_candidates_path)
    scan = pd.read_parquet(opportunity_scan_path)

    features, runtime_audit = _build_first_hot(stage2, opportunity, scan)
    missing = _missing_audit(features)
    signals = _signal_rows(features)
    horizons = _fixed_horizon_audit(features)

    summary = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "source_request": 163,
        "opens_new_dates": False,
        "changes_model_or_threshold": False,
        "purpose": (
            "diagnose economic signal, missing labels, and fixed-horizon "
            "BASE net returns under the frozen Request-163 selector"
        ),
        "runtime_audit": runtime_audit,
        "missing_label_audit": missing,
        "signals": signals,
        "fixed_horizon_audit": horizons,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    signal_csv.parent.mkdir(parents=True, exist_ok=True)
    horizon_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(signals).to_csv(signal_csv, index=False)
    pd.DataFrame(
        [
            {k: v for k, v in row.items() if k != "by_day"}
            for row in horizons["horizons"]
        ]
    ).to_csv(horizon_csv, index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-candidates", type=Path, required=True)
    parser.add_argument("--opportunity-candidates", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--signal-csv", type=Path, required=True)
    parser.add_argument("--horizon-csv", type=Path, required=True)
    args = parser.parse_args()
    return run(
        args.stage2_candidates,
        args.opportunity_candidates,
        args.opportunity_scan,
        args.output_json,
        args.signal_csv,
        args.horizon_csv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
