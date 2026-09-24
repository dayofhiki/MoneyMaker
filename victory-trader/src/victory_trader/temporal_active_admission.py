"""Request 155: temporal Active admission with multi-minute crossing horizons.

Development-only on already-opened request-151/152 sessions.  No later date is
opened and no production policy is promoted by this request.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .actionable_active_handoff import (
    add_actionable_priority,
    fit_observability_model,
    mark_focus_eligibility,
)
from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
from .attention_replay import MINUTE_MS
from .balanced_retention_active_allocator import (
    add_balanced_retention_value,
    add_transport_survival_probability,
    build_transport_survival_rows,
    fit_transport_survival_model,
)
from .config import load_settings, require_flatfile_credentials
from .direct_active_admission import EVAL_DAYS, _variant_metrics
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .hierarchical_attention_runtime import MODEL_KWARGS
from .market_calendar import is_us_equity_trading_day, regular_session_bounds
from .market_wide_focus_hazard import (
    BASELINE_FEATURES,
    FIT_END,
    HAZARD_COLUMNS,
    build_market_hazard_frame,
    build_market_hazard_rows,
    fit_market_hazard,
    select_learned_focus,
)
from .massive_client import MassiveClient
from .multi_day import daterange

REQUEST_ID = 155
HORIZONS = (1, 2, 3, 5)
PRIMARY_HORIZON = 3
TARGET_RETENTION = 0.95
MATERIAL_GAIN = 0.01


def add_cross_within_horizon_targets(
    scan: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Add nullable wall-clock crossing-within-H-minute targets.

    A positive is known as soon as a future runner crossing occurs inside the
    requested horizon.  A negative is assigned only when the whole horizon fits
    inside the regular session.  Rows too near the close remain unlabeled if no
    crossing occurs.
    """

    base = build_market_hazard_frame(scan).copy()
    crossings = scan.loc[
        scan["runner_cross_now"].fillna(False).astype(bool),
        ["trading_day", "ticker", "t"],
    ].copy()
    crossings["trading_day"] = crossings["trading_day"].astype(str)
    crossings["ticker"] = crossings["ticker"].astype(str).str.upper()
    base["trading_day"] = base["trading_day"].astype(str)
    base["ticker"] = base["ticker"].astype(str).str.upper()

    close_by_day: dict[str, int] = {}
    for day_text in base["trading_day"].unique():
        bounds = regular_session_bounds(date.fromisoformat(str(day_text)))
        if bounds is None:
            raise ValueError(f"missing session bounds for {day_text}")
        close_by_day[str(day_text)] = int(bounds[1].timestamp() * 1000)

    crossing_index = pd.MultiIndex.from_frame(
        crossings.loc[:, ["trading_day", "ticker", "t"]]
    )
    day_values = base["trading_day"].astype(str).to_numpy()
    ticker_values = base["ticker"].astype(str).to_numpy()
    t_values = pd.to_numeric(base["t"], errors="raise").astype("int64").to_numpy()

    for horizon in horizons:
        positive = np.zeros(len(base), dtype=bool)
        for minute in range(1, horizon + 1):
            future_keys = pd.MultiIndex.from_arrays(
                [
                    day_values,
                    ticker_values,
                    t_values + minute * MINUTE_MS,
                ],
                names=["trading_day", "ticker", "t"],
            )
            positive |= future_keys.isin(crossing_index)

        close_ms = base["trading_day"].map(close_by_day).astype("int64").to_numpy()
        full_horizon = (t_values + horizon * MINUTE_MS) <= close_ms
        target = np.full(len(base), np.nan, dtype=float)
        target[positive] = 1.0
        target[full_horizon & ~positive] = 0.0
        base[f"target_cross_within_{horizon}m"] = target

    return base


def fit_temporal_admission_model(
    rows: pd.DataFrame,
    *,
    horizon: int,
) -> HistGradientBoostingClassifier:
    target_column = f"target_cross_within_{horizon}m"
    fit = rows.loc[rows["trading_day"].astype(str).le(FIT_END)].copy()
    target = pd.to_numeric(fit[target_column], errors="coerce")
    fit = fit.loc[target.notna()].copy()
    y = pd.to_numeric(fit[target_column], errors="raise").astype(int)
    if fit.empty or y.nunique() < 2 or int(y.sum()) == 0:
        raise ValueError(f"temporal {horizon}m fit needs both classes")
    model = HistGradientBoostingClassifier(**MODEL_KWARGS)
    model.fit(
        fit[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan),
        y,
    )
    return model


def _focus_rows(
    target_rows: pd.DataFrame,
    market_model: HistGradientBoostingClassifier,
) -> pd.DataFrame:
    scored, focus, _ = select_learned_focus(
        target_rows.loc[:, HAZARD_COLUMNS].copy(),
        market_model,
    )
    del scored
    keys = focus.loc[:, ["trading_day", "ticker", "t"]].copy()
    return target_rows.merge(
        keys,
        on=["trading_day", "ticker", "t"],
        how="inner",
        validate="one_to_one",
    )


def add_temporal_probabilities(
    scored: pd.DataFrame,
    models: dict[int, HistGradientBoostingClassifier],
) -> pd.DataFrame:
    result = scored.copy()
    features = result[BASELINE_FEATURES].replace([np.inf, -np.inf], np.nan)
    for horizon, model in models.items():
        result[f"cross_within_{horizon}m_probability"] = model.predict_proba(
            features
        )[:, 1]
    return result


def _apply_temporal_score(scored: pd.DataFrame, horizon: int) -> pd.DataFrame:
    result = scored.copy()
    score = pd.to_numeric(
        result[f"cross_within_{horizon}m_probability"],
        errors="coerce",
    ).clip(lower=0.0, upper=1.0)
    survival = pd.to_numeric(
        result["transport_survival_probability"], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    result["active_priority"] = score
    result["balanced_retention_value"] = score * survival
    return result


def _post_additions(metrics: dict[str, object], cap: str = "8") -> float | None:
    audit = metrics[cap]["transport_audit"]
    value = audit["mean_post_initial_subscription_additions_per_decision"]
    return float(value) if value is not None else None


def _diagnose(
    legacy: dict[str, object],
    primary: dict[str, object],
) -> dict[str, object]:
    legacy8 = legacy["8"]["active_retention_given_focus"]
    primary8 = primary["8"]["active_retention_given_focus"]
    primary20 = primary["20"]["active_retention_given_focus"]
    if legacy8 is None or primary8 is None or primary20 is None:
        return {"diagnosis": "insufficient_support"}

    gain8 = float(primary8 - legacy8)
    legacy_churn = _post_additions(legacy)
    primary_churn = _post_additions(primary)
    nonhigher_churn = (
        legacy_churn is not None
        and primary_churn is not None
        and primary_churn <= legacy_churn
    )

    if primary8 >= TARGET_RETENTION and gain8 >= MATERIAL_GAIN and nonhigher_churn:
        diagnosis = "temporal_policy_candidate"
    elif primary20 >= TARGET_RETENTION and primary8 < TARGET_RETENTION:
        diagnosis = "temporal_ranking_candidate_allocator_mismatch"
    elif gain8 >= MATERIAL_GAIN:
        diagnosis = "partial_temporal_gain"
    else:
        diagnosis = "temporal_target_insufficient"

    return {
        "diagnosis": diagnosis,
        "primary_cap8_gain_vs_legacy": gain8,
        "legacy_cap8_mean_post_initial_additions": legacy_churn,
        "primary_cap8_mean_post_initial_additions": primary_churn,
        "primary_nonhigher_churn": nonhigher_churn,
    }


def run_probe(
    store: MassiveFlatFileStore,
    scan_client: MassiveClient,
    start: date,
    end: date,
) -> dict[str, object]:
    fit_hazard: list[pd.DataFrame] = []
    fit_causal: list[pd.DataFrame] = []
    fit_temporal: list[pd.DataFrame] = []
    fit_survival: list[pd.DataFrame] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text > FIT_END or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        fit_hazard.append(
            build_market_hazard_rows(scan).loc[:, HAZARD_COLUMNS].copy()
        )
        fit_causal.append(build_market_hazard_frame(scan))
        fit_temporal.append(add_cross_within_horizon_targets(scan))
        fit_survival.append(build_transport_survival_rows(scan))

    if not fit_hazard or not fit_temporal:
        raise ValueError("request 155 fit population is empty")

    market_model = fit_market_hazard(pd.concat(fit_hazard, ignore_index=True))
    observability_model = fit_observability_model(
        pd.concat(fit_causal, ignore_index=True)
    )
    temporal_fit = pd.concat(fit_temporal, ignore_index=True)
    focus_temporal_fit = _focus_rows(temporal_fit, market_model)
    temporal_models = {
        horizon: fit_temporal_admission_model(
            focus_temporal_fit,
            horizon=horizon,
        )
        for horizon in HORIZONS
    }
    survival_model = fit_transport_survival_model(
        pd.concat(fit_survival, ignore_index=True)
    )

    eval_scans: list[pd.DataFrame] = []
    eval_focus: list[pd.DataFrame] = []
    eval_scored: list[pd.DataFrame] = []

    for day in daterange(start, end):
        day_text = day.isoformat()
        if day_text not in EVAL_DAYS or not is_us_equity_trading_day(day):
            continue
        scan, _ = build_flatfile_scan_day(store, scan_client, day)
        causal = build_market_hazard_frame(scan).copy()
        scored, focus, _ = select_learned_focus(
            causal.loc[:, HAZARD_COLUMNS].copy(),
            market_model,
        )
        scored = add_actionable_priority(scored, observability_model)
        scored = add_temporal_probabilities(scored, temporal_models)
        scored = mark_focus_eligibility(scored, focus)
        scored = add_transport_survival_probability(scored, survival_model)
        scored = add_balanced_retention_value(scored)
        eval_scans.append(scan)
        eval_focus.append(focus)
        eval_scored.append(scored)

    scan = pd.concat(eval_scans, ignore_index=True)
    focus = pd.concat(eval_focus, ignore_index=True)
    scored = pd.concat(eval_scored, ignore_index=True)
    found = sorted(scan["trading_day"].astype(str).unique())
    if found != EVAL_DAYS:
        raise ValueError(f"expected request 155 eval days {EVAL_DAYS}, found {found}")

    legacy = _variant_metrics(scored, focus, scan)
    horizon_metrics: dict[str, object] = {}
    for horizon in HORIZONS:
        variant = _apply_temporal_score(scored, horizon)
        horizon_metrics[str(horizon)] = _variant_metrics(variant, focus, scan)

    primary = horizon_metrics[str(PRIMARY_HORIZON)]
    diagnosis = _diagnose(legacy, primary)

    return {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "diagnostic_only": True,
        "eval_days": EVAL_DAYS,
        "horizons_minutes": list(HORIZONS),
        "primary_horizon_minutes": PRIMARY_HORIZON,
        "training_population": "learned Focus-60 only",
        "target_retention": TARGET_RETENTION,
        "material_gain_threshold": MATERIAL_GAIN,
        "legacy": legacy,
        "temporal_horizons": horizon_metrics,
        **diagnosis,
        "interpretation_rule": {
            "temporal_policy_candidate": (
                "3m cap8 reaches 95%, gains >=1pp over legacy cap8, and does not increase mean churn"
            ),
            "temporal_ranking_candidate_allocator_mismatch": (
                "3m cap20 reaches 95% but cap8 does not; ranking improves but bounded allocator needs redesign"
            ),
            "partial_temporal_gain": (
                "3m cap8 gains >=1pp but remains below 95%"
            ),
            "temporal_target_insufficient": (
                "3m cap8 gains <1pp; horizon target alone does not solve Active memory"
            ),
        },
        "flatfile_stats": store.stats.to_dict(),
        "scan_client_stats": scan_client.stats.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

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
    summary = run_probe(store, scan_client, args.start, args.end)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
