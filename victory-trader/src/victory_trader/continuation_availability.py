"""Request240: continuation availability as a separate tradability head.

Request239 showed that removing the consecutive-opportunity constraint does not
repair the persistent 61.7% target coverage. Request240 treats that missingness
as a potential economic state rather than silently discarding it.

For every first-HOT episode:
- continuation_available = at least two evaluable Request232 WATCH values;
- this binary label is defined for every first-HOT row;
- a separate causal classifier predicts continuation availability;
- conditional economics are still measured only where WATCH values exist.

This asks whether "will this HOT remain executable enough to keep watching?" is
a distinct learnable dimension.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .extended_history_economic_opportunity import EXTENDED_FIT_DAYS, EXTENDED_CAL_DAYS
from .fresh_validated_attention_economic_opportunity import FRESH_EVAL_DAYS
from .hot_economic_opportunity import ENTRY_FEATURES

REQUEST_ID = 240
SOURCE_RUN = 36251239308
WATCH_COLUMNS = [
    f"watch_event_{i}_multi_event_value_pct"
    for i in range(1, 6)
]

MIN_AUC = 0.70
MIN_AUC_GAIN_VS_RUNNER = 0.05
MIN_SELECTED_AVAILABILITY_RATE = 0.80
MIN_POSITIVE_AUC_DAYS = 4

def attach_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    values = result[WATCH_COLUMNS].apply(pd.to_numeric, errors="coerce")
    count = values.notna().sum(axis=1)
    result["continuation_available"] = count.ge(2).astype(int)
    result["available_watch_count"] = count.astype(int)

    def top2(row):
        arr = row.dropna().to_numpy(dtype=float)
        if len(arr) < 2:
            return np.nan
        return float(np.mean(np.sort(arr)[-2:]))

    result["conditional_top2_value_pct"] = values.apply(top2, axis=1)
    return result

def _feature_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

def _usable_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        c for c in ENTRY_FEATURES
        if c in frame.columns
        and pd.to_numeric(frame[c], errors="coerce").notna().any()
    )

def _safe_auc(actual: pd.Series, score: pd.Series) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = a.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    y = a.loc[valid].astype(int)
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y, s.loc[valid].astype(float)))

def train(frame: pd.DataFrame):
    day = frame["trading_day"].astype(str)
    fit = frame.loc[day.isin(EXTENDED_FIT_DAYS)].copy()
    cal = frame.loc[day.isin(EXTENDED_CAL_DAYS)].copy()
    columns = _usable_columns(fit)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=20261311,
    )
    model.fit(
        _feature_frame(fit, columns),
        fit["continuation_available"].astype(int),
    )
    cal_prob = model.predict_proba(_feature_frame(cal, columns))[:, 1]
    gate = float(np.quantile(cal_prob, 0.75))
    return model, columns, gate

def evaluate(scored_hot_path: Path, output_path: Path, rows_output_path: Path) -> int:
    frame = attach_labels(pd.read_parquet(scored_hot_path))
    model, columns, gate = train(frame)
    frame["continuation_probability"] = model.predict_proba(
        _feature_frame(frame, columns)
    )[:, 1]
    frame["continuation_selected"] = frame[
        "continuation_probability"
    ].ge(gate)

    fresh = frame.loc[
        frame["trading_day"].astype(str).isin(FRESH_EVAL_DAYS)
    ].copy()
    target = fresh["continuation_available"].astype(int)
    auc = _safe_auc(target, fresh["continuation_probability"])
    runner_auc = _safe_auc(target, fresh["second_rerank_probability"])

    selected = fresh.loc[
        fresh["continuation_selected"].fillna(False).astype(bool)
    ].copy()
    selected_avail = float(
        selected["continuation_available"].mean()
    ) if len(selected) else None
    selected_rate = float(len(selected) / len(fresh)) if len(fresh) else None

    available_selected = selected.loc[
        selected["continuation_available"].eq(1)
    ].copy()
    conditional = pd.to_numeric(
        available_selected["conditional_top2_value_pct"],
        errors="coerce",
    ).dropna()

    by_day = {}
    positive_auc_days = 0
    for day in FRESH_EVAL_DAYS:
        part = fresh.loc[
            fresh["trading_day"].astype(str).eq(day)
        ].copy()
        day_auc = _safe_auc(
            part["continuation_available"],
            part["continuation_probability"],
        )
        if day_auc is not None and day_auc > 0.5:
            positive_auc_days += 1
        chosen = part.loc[
            part["continuation_selected"].fillna(False).astype(bool)
        ]
        by_day[day] = {
            "rows": int(len(part)),
            "availability_rate": float(
                part["continuation_available"].mean()
            ) if len(part) else None,
            "auc": day_auc,
            "selected_rows": int(len(chosen)),
            "selected_availability_rate": float(
                chosen["continuation_available"].mean()
            ) if len(chosen) else None,
        }

    auc_gain = (
        float(auc) - float(runner_auc)
        if auc is not None and runner_auc is not None
        else None
    )
    gate_pass = bool(
        auc is not None
        and auc >= MIN_AUC
        and auc_gain is not None
        and auc_gain >= MIN_AUC_GAIN_VS_RUNNER
        and selected_avail is not None
        and selected_avail >= MIN_SELECTED_AVAILABILITY_RATE
        and positive_auc_days >= MIN_POSITIVE_AUC_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "source_run": SOURCE_RUN,
        "opens_new_dates": False,
        "development_only": True,
        "promotion_eligible": False,
        "label": {
            "continuation_available": (
                "at least two evaluable WATCH multi-event values"
            ),
            "coverage": 1.0,
            "missing_return_zero_filled": False,
        },
        "fresh_development": {
            "rows": int(len(fresh)),
            "population_availability_rate": float(target.mean()),
            "candidate_auc": auc,
            "runner_auc": runner_auc,
            "auc_gain_vs_runner": auc_gain,
            "selected_rows": int(len(selected)),
            "selected_rate": selected_rate,
            "selected_availability_rate": selected_avail,
            "selected_available_conditional_mean_pct": (
                float(conditional.mean()) if len(conditional) else None
            ),
            "selected_available_conditional_positive_rate": (
                float(conditional.gt(0).mean()) if len(conditional) else None
            ),
            "positive_auc_days": positive_auc_days,
            "by_day": by_day,
        },
        "development_gate": {
            "min_auc": MIN_AUC,
            "min_auc_gain_vs_runner": MIN_AUC_GAIN_VS_RUNNER,
            "min_selected_availability_rate": MIN_SELECTED_AVAILABILITY_RATE,
            "min_positive_auc_days": MIN_POSITIVE_AUC_DAYS,
            "pass": gate_pass,
        },
        "development_gate_pass": gate_pass,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze continuation availability as a separate attention/risk "
            "head and combine it with opportunity value in the next entry "
            "timing experiment."
        ),
        "next_boundary_if_fail": (
            "the 38% unresolved population is not reliably predictable "
            "from current first-HOT state; inspect liquidity/executability "
            "raw features before redesigning timing."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    frame.to_parquet(rows_output_path, index=False, compression="zstd")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-hot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(args.scored_hot, args.output, args.rows_output)

if __name__ == "__main__":
    raise SystemExit(main())
