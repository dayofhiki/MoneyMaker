"""Request249: full-HOT fixed-policy economic value without hindsight entry search."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from .attention_replay import MINUTE_MS
from .extended_history_economic_opportunity import EXTENDED_CAL_DAYS, EXTENDED_FIT_DAYS
from .fresh_validated_attention_economic_opportunity import FRESH_EVAL_DAYS
from .hierarchical_attention_runtime import add_stage2_scores, fit_stage2
from .hot_economic_opportunity import ENTRY_FEATURES, _first_hot_feature_rows, _safe_spearman
from .recurrent_wait_entry_action_value import _base_return

REQUEST_ID = 249
WATCH_WINDOW_MINUTES = 5
RISK_CAP_MINUTES = 30
EVENT_HORIZONS = (1, 2, 3, 5)
MIN_FRESH_SPEARMAN = 0.08
MIN_FRESH_AUC = 0.58
MIN_SELECTED_RATE = 0.10
MAX_SELECTED_RATE = 0.40
MIN_POSITIVE_RATE_GAIN = 0.08
MIN_POSITIVE_MEAN_DAYS = 4


@dataclass(frozen=True)
class Model:
    regressor: HistGradientBoostingRegressor
    classifier: HistGradientBoostingClassifier
    columns: tuple[str, ...]
    offset: float
    probability_gate: float


def event_lookup(scan: pd.DataFrame) -> dict[tuple[str, str], list[tuple[int, float]]]:
    work = scan.loc[:, ["trading_day", "ticker", "t", "o"]].copy()
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work["o"] = pd.to_numeric(work["o"], errors="coerce")
    work = work.loc[work.t.notna() & work.o.gt(0)].sort_values(
        ["trading_day", "ticker", "t"]
    )
    out: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in work.itertuples(index=False):
        key = (str(row.trading_day), str(row.ticker).upper())
        out.setdefault(key, []).append((int(row.t), float(row.o)))
    return out


def attach_fixed_value(first_hot: pd.DataFrame, scan: pd.DataFrame) -> pd.DataFrame:
    result = first_hot.copy()
    result["fixed_first_watch_value_pct"] = np.nan
    result["fixed_entry_elapsed_minutes"] = np.nan
    lookup = event_lookup(scan)
    for idx, row in result.iterrows():
        hot_t = int(row["t"])
        episode = lookup.get(
            (str(row["trading_day"]), str(row["ticker"]).upper()),
            [],
        )
        watch = [
            (t, p)
            for t, p in episode
            if hot_t < t <= hot_t + WATCH_WINDOW_MINUTES * MINUTE_MS
        ]
        if not watch:
            continue
        entry_t, entry_open = watch[0]
        future = [
            (t, p)
            for t, p in episode
            if entry_t < t <= hot_t + RISK_CAP_MINUTES * MINUTE_MS
        ]
        if len(future) < max(EVENT_HORIZONS):
            continue
        values = []
        complete = True
        for horizon in EVENT_HORIZONS:
            exit_t, exit_open = future[horizon - 1]
            if exit_t > hot_t + RISK_CAP_MINUTES * MINUTE_MS:
                complete = False
                break
            value = _base_return(entry_open, exit_open)
            if not np.isfinite(value):
                complete = False
                break
            values.append(float(value))
        if not complete:
            continue
        result.at[idx, "fixed_first_watch_value_pct"] = float(np.mean(values))
        result.at[idx, "fixed_entry_elapsed_minutes"] = float(
            (entry_t - hot_t) / MINUTE_MS
        )
    return result


def usable_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        c
        for c in ENTRY_FEATURES
        if c in frame
        and pd.to_numeric(frame[c], errors="coerce").notna().any()
    )


def xframe(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return (
        frame.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def train(frame: pd.DataFrame) -> Model:
    day = frame.trading_day.astype(str)
    fit = frame.loc[day.isin(EXTENDED_FIT_DAYS)].copy()
    cal = frame.loc[day.isin(EXTENDED_CAL_DAYS)].copy()
    fit = fit.loc[pd.to_numeric(fit.fixed_first_watch_value_pct, errors="coerce").notna()]
    cal = cal.loc[pd.to_numeric(cal.fixed_first_watch_value_pct, errors="coerce").notna()]
    columns = usable_columns(fit)
    y = fit.fixed_first_watch_value_pct.to_numpy(float)
    low, high = np.quantile(y, [0.005, 0.995])
    reg = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=20261349,
    )
    reg.fit(xframe(fit, columns), np.clip(y, low, high))
    cls = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=20261350,
    )
    cls.fit(xframe(fit, columns), (y > 0).astype(int))
    cal_y = cal.fixed_first_watch_value_pct.to_numpy(float)
    cal_x = xframe(cal, columns)
    offset = float(np.mean(cal_y - reg.predict(cal_x)))
    gate = float(np.quantile(cls.predict_proba(cal_x)[:, 1], 0.75))
    return Model(reg, cls, columns, offset, gate)


def evaluate_fresh(frame: pd.DataFrame, model: Model) -> dict:
    fresh = frame.loc[
        frame.trading_day.astype(str).isin(FRESH_EVAL_DAYS)
    ].copy()
    target = pd.to_numeric(fresh.fixed_first_watch_value_pct, errors="coerce")
    coverage = float(target.notna().mean()) if len(fresh) else None
    fresh = fresh.loc[target.notna()].copy()
    target = fresh.fixed_first_watch_value_pct.astype(float)
    x = xframe(fresh, model.columns)
    pred = model.regressor.predict(x) + model.offset
    prob = model.classifier.predict_proba(x)[:, 1]
    fresh["predicted_fixed_value_pct"] = pred
    fresh["positive_probability"] = prob
    fresh["selected"] = prob >= model.probability_gate
    auc = (
        float(roc_auc_score(target.gt(0).astype(int), prob))
        if target.gt(0).nunique() == 2
        else None
    )
    spearman = _safe_spearman(target, pd.Series(pred, index=fresh.index))
    selected = fresh.loc[fresh.selected]
    selected_target = selected.fixed_first_watch_value_pct.astype(float)
    pop_positive = float(target.gt(0).mean()) if len(target) else None
    sel_positive = (
        float(selected_target.gt(0).mean()) if len(selected_target) else None
    )
    positive_gain = (
        sel_positive - pop_positive
        if sel_positive is not None and pop_positive is not None
        else None
    )
    day_means = {}
    positive_days = 0
    for day in FRESH_EVAL_DAYS:
        chosen = selected.loc[selected.trading_day.astype(str).eq(day)]
        mean = (
            float(chosen.fixed_first_watch_value_pct.mean())
            if len(chosen)
            else None
        )
        day_means[str(day)] = mean
        positive_days += int(mean is not None and mean > 0)
    selected_rate = float(len(selected) / len(fresh)) if len(fresh) else None
    selected_mean = (
        float(selected_target.mean()) if len(selected_target) else None
    )
    checks = {
        "fresh_spearman": spearman is not None and spearman >= MIN_FRESH_SPEARMAN,
        "fresh_auc": auc is not None and auc >= MIN_FRESH_AUC,
        "selected_rate": (
            selected_rate is not None
            and MIN_SELECTED_RATE <= selected_rate <= MAX_SELECTED_RATE
        ),
        "selected_mean_positive": (
            selected_mean is not None and selected_mean > 0
        ),
        "positive_rate_gain": (
            positive_gain is not None and positive_gain >= MIN_POSITIVE_RATE_GAIN
        ),
        "positive_selected_mean_days": positive_days >= MIN_POSITIVE_MEAN_DAYS,
    }
    return {
        "rows": int(len(fresh)),
        "coverage": coverage,
        "spearman": spearman,
        "auc": auc,
        "population_mean_pct": float(target.mean()) if len(target) else None,
        "population_positive_rate": pop_positive,
        "selected_rows": int(len(selected)),
        "selected_rate": selected_rate,
        "selected_mean_pct": selected_mean,
        "selected_positive_rate": sel_positive,
        "selected_positive_rate_gain": positive_gain,
        "positive_selected_mean_days": positive_days,
        "selected_mean_by_day": day_means,
        "checks": checks,
        "signal_gate_pass": bool(all(checks.values())),
    }


def evaluate(stage2_path: Path, candidates_path: Path, scan_path: Path, output: Path) -> int:
    stage2 = pd.read_parquet(stage2_path)
    candidates = pd.read_parquet(candidates_path)
    scan = pd.read_parquet(scan_path)
    minute_model, second_model = fit_stage2(stage2)
    scored = add_stage2_scores(candidates, minute_model, second_model)
    first_hot, runtime_audit = _first_hot_feature_rows(scored, scan)
    first_hot = attach_fixed_value(first_hot, scan)
    model = train(first_hot)
    fresh = evaluate_fresh(first_hot, model)
    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "population": "full validated-attention first-HOT before BUY selection",
        "target": {
            "entry": "first observed WATCH state only",
            "event_horizons": list(EVENT_HORIZONS),
            "future_best_search": False,
        },
        "support": {
            "first_hot_rows": int(len(first_hot)),
            "fit_labeled_rows": int(
                first_hot.loc[
                    first_hot.trading_day.astype(str).isin(EXTENDED_FIT_DAYS),
                    "fixed_first_watch_value_pct",
                ].notna().sum()
            ),
            "calibration_labeled_rows": int(
                first_hot.loc[
                    first_hot.trading_day.astype(str).isin(EXTENDED_CAL_DAYS),
                    "fixed_first_watch_value_pct",
                ].notna().sum()
            ),
        },
        "feature_count": len(model.columns),
        "calibration_probability_gate": model.probability_gate,
        "runtime_audit": runtime_audit,
        "fresh": fresh,
        "signal_gate_pass": fresh["signal_gate_pass"],
        "interpretation": (
            "Minute aggregate opens are historical research execution references, "
            "not actual brokerage fills. No WATCH or exit horizon is chosen by hindsight."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-candidates", type=Path, required=True)
    p.add_argument("--opportunity-candidates", type=Path, required=True)
    p.add_argument("--opportunity-scan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    return evaluate(
        a.stage2_candidates,
        a.opportunity_candidates,
        a.opportunity_scan,
        a.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
