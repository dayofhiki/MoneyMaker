"""Request 141: development economic-opportunity gate for HOT promotions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hot_entry_economics import label_hot_event_economics, promotion_hot_events
from .massive_client import MassiveClient

REQUEST_ID = 141
FIT_DAYS = ["2026-04-30", "2026-05-01", "2026-05-04"]
CAL_DAYS = ["2026-05-05", "2026-05-06"]
EVAL_DAYS = ["2026-05-07", "2026-05-08", "2026-05-11", "2026-05-12", "2026-05-13"]
ALL_DAYS = FIT_DAYS + CAL_DAYS + EVAL_DAYS

FEATURES = (
    "learned_hot_score",
    "log_learned_hot_score",
    "attention_score",
    "hot_rank",
    "score_margin_to_hot_floor",
    "promotion_index",
    "minutes_since_previous_promotion",
    "is_repromotion",
    "minutes_from_regular_open",
)

RANDOM_SEED = 20261049


@dataclass(frozen=True)
class HurdleGate:
    classifier: HistGradientBoostingClassifier
    platt: LogisticRegression
    positive_model: HistGradientBoostingRegressor
    loss_model: HistGradientBoostingRegressor
    positive_offset: float
    loss_offset: float
    positive_winsor: tuple[float, float]
    loss_winsor: tuple[float, float]


def _event_features(trace: pd.DataFrame) -> pd.DataFrame:
    events = promotion_hot_events(trace)
    if events.empty:
        return events

    hot = trace.loc[trace["state"].astype(str).eq("hot")].copy()
    hot["learned_hot_score"] = pd.to_numeric(
        hot["learned_hot_score"], errors="coerce"
    )
    hot["hot_rank"] = hot.groupby(
        ["trading_day", "t"], sort=False
    )["learned_hot_score"].rank(method="first", ascending=False)
    hot_floor = hot.groupby(
        ["trading_day", "t"], sort=False
    )["learned_hot_score"].transform("min")
    hot["score_margin_to_hot_floor"] = hot["learned_hot_score"] - hot_floor

    context = hot.loc[
        :,
        [
            "trading_day",
            "ticker",
            "t",
            "hot_rank",
            "score_margin_to_hot_floor",
        ],
    ]
    events = events.merge(
        context,
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
    )

    score = pd.to_numeric(events["learned_hot_score"], errors="coerce")
    events["log_learned_hot_score"] = np.log(score.clip(lower=1e-12))
    times = pd.to_datetime(
        pd.to_numeric(events["t"], errors="coerce"),
        unit="ms",
        utc=True,
    ).dt.tz_convert("America/New_York")
    events["minutes_from_regular_open"] = (
        (times.dt.hour * 60 + times.dt.minute) - (9 * 60 + 30)
    ).astype(float)
    events["is_repromotion"] = (
        events["is_repromotion"].fillna(False).astype(bool).astype(float)
    )
    return events


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reindex(columns=FEATURES).apply(pd.to_numeric, errors="coerce")


def _load_block(
    trace_path: Path,
    days: list[str],
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
) -> pd.DataFrame:
    trace = pd.read_parquet(trace_path)
    trace = trace.loc[trace["trading_day"].astype(str).isin(days)].copy()
    events = _event_features(trace)

    scans: list[pd.DataFrame] = []
    for day_text in days:
        scan, _ = build_flatfile_scan_day(
            store,
            scan_client,
            date.fromisoformat(day_text),
        )
        scans.append(scan)
    scan = pd.concat(scans, ignore_index=True)
    labeled = label_hot_event_economics(events, scan)
    labeled["oracle_best_base_pct"] = pd.to_numeric(
        labeled["oracle_best_base_pct"], errors="coerce"
    )
    return labeled


def _winsor_fit(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
) -> tuple[HistGradientBoostingRegressor, tuple[float, float]]:
    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(x, y.clip(lower=low, upper=high))
    return model, (float(low), float(high))


def fit_hurdle_gate(fit: pd.DataFrame, calibration: pd.DataFrame) -> HurdleGate:
    fit = fit.loc[fit["oracle_best_base_pct"].notna()].copy()
    calibration = calibration.loc[
        calibration["oracle_best_base_pct"].notna()
    ].copy()
    if len(fit) < 1000 or len(calibration) < 500:
        raise ValueError("insufficient HOT promotion rows for hurdle gate")

    x_fit = _feature_frame(fit)
    y_fit = fit["oracle_best_base_pct"].astype(float)
    binary_fit = y_fit.gt(0).astype(int)
    if binary_fit.nunique() < 2:
        raise ValueError("fit block lacks both opportunity classes")

    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=2.0,
        random_state=RANDOM_SEED,
    )
    classifier.fit(x_fit, binary_fit)

    x_cal = _feature_frame(calibration)
    y_cal = calibration["oracle_best_base_pct"].astype(float)
    binary_cal = y_cal.gt(0).astype(int)
    if len(calibration) < 500 or binary_cal.nunique() < 2:
        raise ValueError("calibration block lacks support")

    raw = classifier.predict_proba(x_cal)[:, 1]
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    platt.fit(logits, binary_cal)

    pos_fit = y_fit.gt(0)
    loss_fit = y_fit.le(0)
    if int(pos_fit.sum()) < 200 or int(loss_fit.sum()) < 200:
        raise ValueError("fit block lacks magnitude support")

    positive_model, positive_winsor = _winsor_fit(
        x_fit.loc[pos_fit],
        y_fit.loc[pos_fit],
        seed=RANDOM_SEED + 1,
    )
    loss_target = -y_fit.loc[loss_fit]
    loss_model, loss_winsor = _winsor_fit(
        x_fit.loc[loss_fit],
        loss_target,
        seed=RANDOM_SEED + 2,
    )

    cal_pos = y_cal.gt(0)
    cal_loss = y_cal.le(0)
    if int(cal_pos.sum()) < 100 or int(cal_loss.sum()) < 100:
        raise ValueError("calibration block lacks magnitude classes")

    pred_pos = positive_model.predict(x_cal.loc[cal_pos])
    pred_loss = loss_model.predict(x_cal.loc[cal_loss])
    positive_offset = float(
        y_cal.loc[cal_pos].mean() - float(np.mean(pred_pos))
    )
    loss_offset = float(
        (-y_cal.loc[cal_loss]).mean() - float(np.mean(pred_loss))
    )

    return HurdleGate(
        classifier=classifier,
        platt=platt,
        positive_model=positive_model,
        loss_model=loss_model,
        positive_offset=positive_offset,
        loss_offset=loss_offset,
        positive_winsor=positive_winsor,
        loss_winsor=loss_winsor,
    )


def score_gate(frame: pd.DataFrame, gate: HurdleGate) -> pd.DataFrame:
    result = frame.copy()
    x = _feature_frame(result)

    raw = gate.classifier.predict_proba(x)[:, 1]
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    probability = gate.platt.predict_proba(logits)[:, 1]

    positive = np.maximum(
        0.0,
        gate.positive_model.predict(x) + gate.positive_offset,
    )
    loss = np.maximum(
        0.0,
        gate.loss_model.predict(x) + gate.loss_offset,
    )
    expected = probability * positive - (1.0 - probability) * loss

    result["predicted_opportunity_probability"] = probability
    result["predicted_positive_magnitude_pct"] = positive
    result["predicted_loss_magnitude_pct"] = loss
    result["predicted_oracle_ev_pct"] = expected
    result["economic_gate_positive"] = expected > 0.0
    return result


def _safe_auc(y: pd.Series, score: pd.Series) -> float | None:
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y.astype(int), score.astype(float)))


def _metrics(scored: pd.DataFrame) -> dict[str, object]:
    valid = scored["oracle_best_base_pct"].notna()
    frame = scored.loc[valid].copy()
    actual = frame["oracle_best_base_pct"].astype(float)
    y = actual.gt(0).astype(int)
    p = frame["predicted_opportunity_probability"].astype(float)
    ev = frame["predicted_oracle_ev_pct"].astype(float)
    hot = pd.to_numeric(frame["learned_hot_score"], errors="coerce")

    selected = frame.loc[frame["economic_gate_positive"].astype(bool)].copy()
    selected_actual = selected["oracle_best_base_pct"].astype(float)

    by_day: dict[str, object] = {}
    positive_days = 0
    min_selected_day = None
    for day in EVAL_DAYS:
        group = selected.loc[selected["trading_day"].astype(str).eq(day)]
        values = pd.to_numeric(group["oracle_best_base_pct"], errors="coerce").dropna()
        mean = float(values.mean()) if len(values) else None
        if mean is not None and mean > 0:
            positive_days += 1
        count = int(len(values))
        min_selected_day = count if min_selected_day is None else min(min_selected_day, count)
        by_day[day] = {
            "selected": count,
            "oracle_base_mean_pct": mean,
            "oracle_base_positive_rate": (
                float(values.gt(0).mean()) if len(values) else None
            ),
        }

    baseline_auc = _safe_auc(y.loc[hot.notna()], hot.loc[hot.notna()])
    model_auc = _safe_auc(y, p)
    baseline_spearman = (
        float(hot.corr(actual, method="spearman")) if hot.notna().sum() >= 20 else None
    )
    ev_spearman = float(ev.corr(actual, method="spearman"))

    selected_rate = float(len(selected) / len(frame)) if len(frame) else None
    selected_mean = (
        float(selected_actual.mean()) if len(selected_actual) else None
    )
    selected_positive = (
        float(selected_actual.gt(0).mean()) if len(selected_actual) else None
    )
    population_mean = float(actual.mean()) if len(actual) else None
    population_positive = float(y.mean()) if len(y) else None

    bridge_pass = bool(
        len(frame) >= 1000
        and model_auc is not None
        and baseline_auc is not None
        and model_auc > 0.5
        and model_auc > baseline_auc
        and baseline_spearman is not None
        and ev_spearman > baseline_spearman
        and len(selected) >= 100
        and selected_rate is not None
        and selected_rate >= 0.05
        and selected_mean is not None
        and selected_mean > 0
        and population_mean is not None
        and selected_mean > population_mean
        and selected_positive is not None
        and population_positive is not None
        and selected_positive > population_positive
        and positive_days >= 4
        and min_selected_day is not None
        and min_selected_day >= 20
    )

    return {
        "evaluation_rows": int(len(frame)),
        "opportunity_positive_rate": population_positive,
        "population_oracle_mean_pct": population_mean,
        "baseline_hot_score_auc": baseline_auc,
        "model_opportunity_auc": model_auc,
        "model_brier": float(brier_score_loss(y, p)),
        "model_log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "baseline_hot_score_spearman_oracle": baseline_spearman,
        "predicted_ev_spearman_oracle": ev_spearman,
        "selected_rows": int(len(selected)),
        "selected_rate": selected_rate,
        "selected_oracle_mean_pct": selected_mean,
        "selected_oracle_median_pct": (
            float(selected_actual.median()) if len(selected_actual) else None
        ),
        "selected_oracle_positive_rate": selected_positive,
        "positive_selected_days": positive_days,
        "minimum_selected_rows_per_day": min_selected_day,
        "by_day": by_day,
        "development_bridge_pass": bridge_pass,
    }


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    request137_trace: Path,
    request138_trace: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    block137 = _load_block(
        request137_trace,
        FIT_DAYS + CAL_DAYS,
        store,
        scan_client,
    )
    block138 = _load_block(
        request138_trace,
        EVAL_DAYS,
        store,
        scan_client,
    )

    fit = block137.loc[
        block137["trading_day"].astype(str).isin(FIT_DAYS)
    ].copy()
    calibration = block137.loc[
        block137["trading_day"].astype(str).isin(CAL_DAYS)
    ].copy()
    evaluation = block138.loc[
        block138["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()

    gate = fit_hurdle_gate(fit, calibration)
    scored = score_gate(evaluation, gate)
    metrics = _metrics(scored)

    summary = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "fit_days": FIT_DAYS,
        "calibration_days": CAL_DAYS,
        "evaluation_days": EVAL_DAYS,
        "features": FEATURES,
        "positive_winsor": gate.positive_winsor,
        "loss_winsor": gate.loss_winsor,
        "positive_offset": gate.positive_offset,
        "loss_offset": gate.loss_offset,
        "metrics": metrics,
    }
    return scored, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Develop an economic opportunity gate for HOT promotions."
    )
    parser.add_argument("--request137-trace", type=Path, required=True)
    parser.add_argument("--request138-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key), FLATFILE_CACHE_DIR
    )
    scan_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive"),
        request_interval_seconds=0.0,
    )
    output, summary = run_probe(
        store,
        scan_client,
        args.request137_trace,
        args.request138_trace,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
