"""Request 140: first-HOT economic-opportunity learnability.

The validated attention hierarchy is frozen. This experiment asks whether the
strictly causal state available at first HOT can identify episodes that contain
some BASE-positive exit opportunity within the following 30 minutes.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import active_transport_integrated_hierarchy as transport
from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .config import load_settings, require_flatfile_credentials
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hierarchical_attention_runtime import (
    STAGE2_FIT_END,
    STAGE2_FIT_START,
    add_stage2_scores,
    build_learned_runtime_trace,
    fit_stage2,
)
from .hot_entry_economics import label_first_hot_economics
from .market_calendar import is_us_equity_trading_day, regular_session_bounds
from .market_wide_focus_hazard import (
    FIT_END,
    HAZARD_COLUMNS,
    build_market_hazard_rows,
    fit_market_hazard,
    select_learned_focus,
)
from .massive_client import MassiveClient
from .multi_day import daterange
from .second_path_attention_probe import SECOND_CACHE_DIR
from .targeted_second_hot_reranker import SECOND_RERANK_FEATURES, add_second_features

REQUEST_ID = 140

FIT_DAYS = [
    "2026-04-30",
    "2026-05-01",
    "2026-05-04",
    "2026-05-05",
    "2026-05-06",
]
CAL_DAYS = [
    "2026-05-07",
    "2026-05-08",
    "2026-05-11",
    "2026-05-12",
    "2026-05-13",
]
EVAL_DAYS = [
    "2026-05-14",
    "2026-05-15",
    "2026-05-18",
    "2026-05-19",
    "2026-05-20",
]
OPPORTUNITY_DAYS = FIT_DAYS + CAL_DAYS + EVAL_DAYS

ENTRY_FEATURES = [
    *SECOND_RERANK_FEATURES,
    "minute_rerank_probability",
    "second_rerank_probability",
    "log_current_price",
    "minutes_since_open",
    "minutes_to_close",
    "hot_candidate_rank",
    "transport_desired_now_numeric",
    "promotion_index",
    "minutes_since_previous_hot_promotion",
]

CLASSIFIER_KWARGS = {
    "learning_rate": 0.05,
    "max_iter": 160,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 60,
    "l2_regularization": 2.0,
    "random_state": 20261050,
}
REGRESSOR_KWARGS = {
    "learning_rate": 0.05,
    "max_iter": 180,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 60,
    "l2_regularization": 2.0,
    "random_state": 20261051,
}


def _safe_auc(y: pd.Series, score: pd.Series) -> float | None:
    yv = pd.to_numeric(y, errors="coerce")
    sv = pd.to_numeric(score, errors="coerce")
    valid = yv.notna() & sv.notna()
    yv = yv.loc[valid].astype(int)
    sv = sv.loc[valid].astype(float)
    if yv.nunique() < 2:
        return None
    return float(roc_auc_score(yv, sv))


def _safe_ap(y: pd.Series, score: pd.Series) -> float | None:
    yv = pd.to_numeric(y, errors="coerce")
    sv = pd.to_numeric(score, errors="coerce")
    valid = yv.notna() & sv.notna()
    yv = yv.loc[valid].astype(int)
    sv = sv.loc[valid].astype(float)
    if int(yv.sum()) == 0:
        return None
    return float(average_precision_score(yv, sv))


def _safe_brier(y: pd.Series, score: pd.Series) -> float | None:
    yv = pd.to_numeric(y, errors="coerce")
    sv = pd.to_numeric(score, errors="coerce")
    valid = yv.notna() & sv.notna()
    if not valid.any():
        return None
    return float(brier_score_loss(yv.loc[valid].astype(int), sv.loc[valid].astype(float)))


def _safe_spearman(actual: pd.Series, predicted: pd.Series) -> float | None:
    av = pd.to_numeric(actual, errors="coerce")
    pv = pd.to_numeric(predicted, errors="coerce")
    valid = av.notna() & pv.notna()
    if int(valid.sum()) < 3:
        return None
    value = av.loc[valid].corr(pv.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _attach_entry_context(
    candidates: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    result = candidates.copy()
    result = result.sort_values(
        ["trading_day", "ticker", "t"], kind="stable"
    ).copy()
    hot_candidate = (
        pd.to_numeric(result["second_rerank_probability"], errors="coerce")
        .notna()
    )
    result["promotion_index"] = np.nan
    result["minutes_since_previous_hot_promotion"] = np.nan
    # Promotion history is causal. These fields are populated for candidate
    # rows from prior HOT membership after runtime trace construction; first-HOT
    # rows therefore receive index 1 and no previous-promotion spacing.
    result["hot_candidate_rank"] = result.groupby(
        ["trading_day", "t"], sort=False
    )["second_rerank_probability"].rank(method="first", ascending=False)
    result["transport_desired_now_numeric"] = (
        result["transport_desired_now"].fillna(False).astype(int)
    )

    prices = scan.loc[:, ["trading_day", "ticker", "t", "c"]].copy()
    prices = prices.rename(columns={"c": "current_price"})
    result = result.merge(
        prices,
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
    )
    result["log_current_price"] = np.log(
        pd.to_numeric(result["current_price"], errors="coerce").where(
            pd.to_numeric(result["current_price"], errors="coerce").gt(0)
        )
    )

    since: list[float] = []
    remaining: list[float] = []
    bounds_by_day: dict[str, tuple[int, int]] = {}
    for day_text in result["trading_day"].astype(str).unique():
        bounds = regular_session_bounds(date.fromisoformat(day_text))
        if bounds is None:
            raise ValueError(f"missing regular session bounds for {day_text}")
        bounds_by_day[day_text] = (
            int(bounds[0].timestamp() * 1000),
            int(bounds[1].timestamp() * 1000),
        )
    for row in result.itertuples(index=False):
        open_ms, close_ms = bounds_by_day[str(row.trading_day)]
        timestamp = int(row.t)
        since.append((timestamp - open_ms) / 60_000.0)
        remaining.append((close_ms - timestamp) / 60_000.0)
    result["minutes_since_open"] = since
    result["minutes_to_close"] = remaining
    return result


def _first_hot_feature_rows(
    scored_candidates: pd.DataFrame,
    scan: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    enriched = _attach_entry_context(scored_candidates, scan)
    trace, runtime_audit = build_learned_runtime_trace(enriched, enriched)
    first_hot = (
        trace.loc[trace["state"].astype(str).eq("hot")]
        .sort_values(["trading_day", "ticker", "t"], kind="stable")
        .groupby(["trading_day", "ticker"], sort=False)
        .head(1)
        .loc[:, ["trading_day", "ticker", "t"]]
        .copy()
    )
    features = first_hot.merge(
        enriched,
        on=["trading_day", "ticker", "t"],
        how="inner",
        validate="one_to_one",
    )

    labels = label_first_hot_economics(trace, scan)
    labels = labels.rename(columns={"hot_t": "t"})
    label_columns = [
        "trading_day",
        "ticker",
        "t",
        "entry_reference_available",
        "entry_price",
        "oracle_best_base_pct",
        "oracle_best_minute",
        "oracle_base_positive",
    ]
    features = features.merge(
        labels.loc[:, label_columns],
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
    )
    return features, runtime_audit


def _fit_models(
    frame: pd.DataFrame,
) -> tuple[
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    float,
    float,
    float,
    float,
]:
    fit = frame.loc[frame["trading_day"].astype(str).isin(FIT_DAYS)].copy()
    cal = frame.loc[frame["trading_day"].astype(str).isin(CAL_DAYS)].copy()
    fit_target = pd.to_numeric(fit["oracle_best_base_pct"], errors="coerce")
    cal_target = pd.to_numeric(cal["oracle_best_base_pct"], errors="coerce")
    fit_valid = fit_target.notna()
    cal_valid = cal_target.notna()
    fit = fit.loc[fit_valid].copy()
    cal = cal.loc[cal_valid].copy()
    fit_target = pd.to_numeric(fit["oracle_best_base_pct"], errors="coerce")
    cal_target = pd.to_numeric(cal["oracle_best_base_pct"], errors="coerce")
    if len(fit) < 1500 or len(cal) < 1500:
        raise ValueError(
            f"request 140 requires >=1500 labeled rows in fit/cal, got {len(fit)}/{len(cal)}"
        )
    y_fit = fit_target.gt(0).astype(int)
    if y_fit.nunique() < 2:
        raise ValueError("fit opportunity labels need both classes")

    classifier = HistGradientBoostingClassifier(**CLASSIFIER_KWARGS)
    classifier.fit(
        fit[ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan),
        y_fit,
    )

    low, high = np.quantile(fit_target.to_numpy(dtype=float), [0.005, 0.995])
    regressor = HistGradientBoostingRegressor(**REGRESSOR_KWARGS)
    regressor.fit(
        fit[ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan),
        fit_target.clip(lower=low, upper=high),
    )
    cal_prediction = regressor.predict(
        cal[ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan)
    )
    offset = float(np.mean(cal_target.to_numpy(dtype=float) - cal_prediction))

    cal_probability = classifier.predict_proba(
        cal[ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    probability_gate = float(np.quantile(cal_probability, 0.75))
    return classifier, regressor, offset, probability_gate, float(low), float(high)


def _evaluate(
    frame: pd.DataFrame,
    classifier: HistGradientBoostingClassifier,
    regressor: HistGradientBoostingRegressor,
    offset: float,
    probability_gate: float,
) -> dict[str, object]:
    evaluation = frame.loc[
        frame["trading_day"].astype(str).isin(EVAL_DAYS)
    ].copy()
    target = pd.to_numeric(evaluation["oracle_best_base_pct"], errors="coerce")
    valid = target.notna()
    evaluation = evaluation.loc[valid].copy()
    target = pd.to_numeric(evaluation["oracle_best_base_pct"], errors="coerce")
    y = target.gt(0).astype(int)

    x = evaluation[ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan)
    probability = classifier.predict_proba(x)[:, 1]
    predicted_value = regressor.predict(x) + offset
    evaluation["opportunity_probability"] = probability
    evaluation["predicted_oracle_base_pct"] = predicted_value
    evaluation["selected_top_quartile"] = probability >= probability_gate

    auc = _safe_auc(y, evaluation["opportunity_probability"])
    ap = _safe_ap(y, evaluation["opportunity_probability"])
    brier = _safe_brier(y, evaluation["opportunity_probability"])
    spearman = _safe_spearman(
        evaluation["oracle_best_base_pct"],
        evaluation["predicted_oracle_base_pct"],
    )

    selected = evaluation.loc[evaluation["selected_top_quartile"]].copy()
    all_mean = float(target.mean()) if len(evaluation) else None
    all_positive = float(y.mean()) if len(evaluation) else None
    selected_target = pd.to_numeric(
        selected["oracle_best_base_pct"], errors="coerce"
    )
    selected_positive = selected_target.gt(0)

    by_day: dict[str, object] = {}
    auc_positive_days = 0
    spearman_positive_days = 0
    selected_mean_nonlower_days = 0
    support_ok = True
    coverage_ok = True
    for day in EVAL_DAYS:
        source = frame.loc[frame["trading_day"].astype(str).eq(day)].copy()
        source_labeled = pd.to_numeric(
            source["oracle_best_base_pct"], errors="coerce"
        ).notna()
        day_eval = evaluation.loc[
            evaluation["trading_day"].astype(str).eq(day)
        ].copy()
        if len(day_eval) < 100:
            support_ok = False
        coverage = float(source_labeled.mean()) if len(source) else 0.0
        if coverage < 0.95:
            coverage_ok = False

        day_target = pd.to_numeric(day_eval["oracle_best_base_pct"], errors="coerce")
        day_y = day_target.gt(0).astype(int)
        day_auc = _safe_auc(day_y, day_eval["opportunity_probability"])
        day_spearman = _safe_spearman(
            day_eval["oracle_best_base_pct"],
            day_eval["predicted_oracle_base_pct"],
        )
        if day_auc is not None and day_auc > 0.5:
            auc_positive_days += 1
        if day_spearman is not None and day_spearman > 0:
            spearman_positive_days += 1

        day_selected = day_eval.loc[day_eval["selected_top_quartile"]]
        day_selected_target = pd.to_numeric(
            day_selected["oracle_best_base_pct"], errors="coerce"
        )
        day_all_mean = float(day_target.mean()) if len(day_eval) else None
        day_selected_mean = (
            float(day_selected_target.mean()) if len(day_selected) else None
        )
        if (
            day_all_mean is not None
            and day_selected_mean is not None
            and day_selected_mean >= day_all_mean
        ):
            selected_mean_nonlower_days += 1

        by_day[day] = {
            "first_hot_rows": int(len(source)),
            "labeled_rows": int(len(day_eval)),
            "label_coverage": coverage,
            "positive_rate": float(day_y.mean()) if len(day_eval) else None,
            "auc": day_auc,
            "spearman": day_spearman,
            "all_oracle_mean_pct": day_all_mean,
            "selected_rows": int(len(day_selected)),
            "selected_oracle_mean_pct": day_selected_mean,
            "selected_positive_rate": (
                float(day_selected_target.gt(0).mean())
                if len(day_selected)
                else None
            ),
        }

    selected_mean = (
        float(selected_target.mean()) if len(selected) else None
    )
    selected_positive_rate = (
        float(selected_positive.mean()) if len(selected) else None
    )
    gate = bool(
        support_ok
        and coverage_ok
        and auc is not None
        and auc >= 0.55
        and auc_positive_days >= 4
        and spearman is not None
        and spearman >= 0.05
        and spearman_positive_days >= 4
        and selected_mean is not None
        and all_mean is not None
        and selected_mean > 0
        and selected_mean > all_mean
        and selected_positive_rate is not None
        and all_positive is not None
        and selected_positive_rate >= all_positive + 0.05
        and selected_mean_nonlower_days >= 4
    )
    return {
        "eval_days": EVAL_DAYS,
        "labeled_rows": int(len(evaluation)),
        "opportunity_positive_rate": all_positive,
        "oracle_base_mean_pct": all_mean,
        "classifier_auc": auc,
        "classifier_average_precision": ap,
        "classifier_brier": brier,
        "value_spearman": spearman,
        "calibration_probability_p75_gate": probability_gate,
        "selected_rows": int(len(selected)),
        "selected_rate": (
            float(len(selected) / len(evaluation)) if len(evaluation) else None
        ),
        "selected_oracle_base_mean_pct": selected_mean,
        "selected_positive_rate": selected_positive_rate,
        "auc_above_random_days": auc_positive_days,
        "positive_spearman_days": spearman_positive_days,
        "selected_mean_nonlower_days": selected_mean_nonlower_days,
        "by_day": by_day,
        "promotion_gate_pass": gate,
    }


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    second_client: MassiveClient,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fit_rows: list[pd.DataFrame] = []
    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text > FIT_END or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        rows = build_market_hazard_rows(scan)
        fit_rows.append(rows.loc[:, HAZARD_COLUMNS].copy())
    if not fit_rows:
        raise ValueError("market-hazard fit rows are empty")
    market_model = fit_market_hazard(pd.concat(fit_rows, ignore_index=True))

    candidate_parts: list[pd.DataFrame] = []
    scan_parts: list[pd.DataFrame] = []
    needed_days = set(OPPORTUNITY_DAYS)
    for day in daterange(start, end):
        day_text = day.isoformat()
        in_stage2_fit = STAGE2_FIT_START <= day_text <= STAGE2_FIT_END
        in_opportunity = day_text in needed_days
        if (not in_stage2_fit and not in_opportunity) or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        rows = build_market_hazard_rows(scan).loc[:, HAZARD_COLUMNS]
        scored, _, _ = select_learned_focus(rows, market_model)
        active_rows, _ = transport.active_observation_rows_with_state(scored)
        candidate_parts.append(active_rows)
        if in_opportunity:
            scan_parts.append(scan)

    candidates, second_audit = add_second_features(
        pd.concat(candidate_parts, ignore_index=True),
        second_client,
    )
    minute_model, second_model = fit_stage2(candidates)
    scored = add_stage2_scores(candidates, minute_model, second_model)
    opportunity_candidates = scored.loc[
        scored["trading_day"].astype(str).isin(OPPORTUNITY_DAYS)
    ].copy()
    opportunity_scan = pd.concat(scan_parts, ignore_index=True)
    found = sorted(opportunity_scan["trading_day"].astype(str).unique())
    if found != OPPORTUNITY_DAYS:
        raise ValueError(f"expected opportunity days {OPPORTUNITY_DAYS}, found {found}")

    features, runtime_audit = _first_hot_feature_rows(
        opportunity_candidates,
        opportunity_scan,
    )
    missing = set(ENTRY_FEATURES) - set(features.columns)
    if missing:
        raise ValueError(f"first-HOT features missing: {sorted(missing)}")

    classifier, regressor, offset, probability_gate, low, high = _fit_models(
        features
    )
    evaluation = _evaluate(
        features,
        classifier,
        regressor,
        offset,
        probability_gate,
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "fit_days": FIT_DAYS,
        "calibration_days": CAL_DAYS,
        "eval_days": EVAL_DAYS,
        "entry_features": ENTRY_FEATURES,
        "regression_fit_winsor_low_pct": low,
        "regression_fit_winsor_high_pct": high,
        "regression_calibration_offset_pct": offset,
        "second_audit": second_audit,
        "runtime_audit": runtime_audit,
        "evaluation": evaluation,
        "flatfile_stats": store.stats.to_dict(),
        "second_client_stats": second_client.stats.to_dict(),
    }
    return features, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test first-HOT economic-opportunity learnability."
    )
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
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
    second_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR,
        request_interval_seconds=0.0,
    )
    output, summary = run_probe(
        store,
        scan_client,
        second_client,
        args.start,
        args.end,
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
