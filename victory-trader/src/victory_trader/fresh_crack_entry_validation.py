"""Request296: untouched five-trading-day validation of frozen crack + entry.

Development is frozen at Request295.  This module opens 2026-06-15 through
2026-06-19 exactly once, reconstructs the same validated upstream first-HOT
population, trains crack/entry models only from May5-8 development artifacts,
and applies one frozen fresh policy without threshold tuning.

This validates setup selection and entry timing only.  HOLD/EXIT remains outside
this request, so the fixed-horizon utility is an entry-quality diagnostic, not
realized trading P&L.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .attention_flatfile_replay import (
    FLATFILE_CACHE_DIR,
    build_flatfile_scan_day,
)
from .causal_minute_controller_rebuild import causal_execution_scan
from .config import load_settings, require_flatfile_credentials
from .downstream_aligned_candidate import candidate_columns
from .extended_history_economic_opportunity import _fit_frozen_attention
from .fixed_horizon_fitted_entry import build_fixed_horizon_states
from .flatfiles import MassiveFlatFilesClient, MassiveFlatFileStore
from .fresh_validated_attention_economic_opportunity import _build_frozen_active
from .hierarchical_active_features import HORIZON
from .hierarchical_attention_runtime import add_stage2_scores, fit_stage2
from .hierarchical_crack_entry_controller import (
    EVENT_FEATURES,
    EXPLOSIVE_LEVELS,
    SETUP_SELECTION_FRACTION,
    _fit,
    _numeric,
    _predict,
    _safe_spearman,
    _usable,
    attach_crack_targets,
)
from .hot_economic_opportunity import _first_hot_feature_rows
from .local_turn_ablation import _policy_metrics
from .local_turn_entry import (
    TURN_FEATURES,
    _fit_models,
    _policy,
    _score,
    attach_local_turn_features,
)
from .market_calendar import is_us_equity_trading_day
from .massive_client import MassiveClient
from .multi_day import daterange
from .prehot_context_ablation import usable_context_columns
from .second_path_attention_probe import SECOND_CACHE_DIR
from .targeted_second_hot_reranker import add_second_features
from .temporal_active_admission import add_cross_within_horizon_targets
from .two_stage_risk_veto import attach_ticker_history

REQUEST_ID = 296
FRESH_DAYS = (
    "2026-06-15",
    "2026-06-16",
    "2026-06-17",
    "2026-06-18",
    "2026-06-22",
)

# Frozen from the four outer-fold Request294/295 action gates.
FROZEN_UTILITY_THRESHOLD = -2.00886730522822
FROZEN_TURN_THRESHOLD = 0.12298633907918213
FROZEN_OVERRIDE_THRESHOLD = np.inf

SETUP_MODEL_SEED = 20263600
ENTRY_MODEL_SEED = 20263650


def _prepare_fresh_first_hot(
    stage2_candidates_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    settings = load_settings()
    access_key, secret_key = require_flatfile_credentials(settings)
    store = MassiveFlatFileStore(
        MassiveFlatFilesClient(access_key, secret_key),
        FLATFILE_CACHE_DIR / "request296",
    )
    scan_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=Path("data/cache/massive/request296"),
        request_interval_seconds=0.0,
    )
    second_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR / "request296-upstream",
        request_interval_seconds=0.02,
    )

    # Attention fit is frozen to the historical January boundary.
    market_model, active_model = _fit_frozen_attention(
        store,
        scan_client,
        date(2026, 1, 2),
        date(2026, 1, 16),
    )

    candidate_parts: list[pd.DataFrame] = []
    scan_parts: list[pd.DataFrame] = []
    day_audit: dict[str, dict] = {}

    for day in daterange(date(2026, 6, 15), date(2026, 6, 22)):
        if not is_us_equity_trading_day(day):
            continue
        scan, audit = build_flatfile_scan_day(store, scan_client, day)
        rows = add_cross_within_horizon_targets(
            scan,
            horizons=(HORIZON,),
        )
        active = _build_frozen_active(
            rows,
            market_model,
            active_model,
        )
        candidate_parts.append(active)
        scan_parts.append(scan)
        day_audit[day.isoformat()] = {
            "scan_rows": int(len(scan)),
            "active_rows_before_second": int(len(active)),
            "scan_audit": audit,
        }

    candidates = pd.concat(candidate_parts, ignore_index=True)
    fresh_scan = pd.concat(scan_parts, ignore_index=True)
    enriched, second_audit = add_second_features(
        candidates,
        second_client,
    )

    stage2_fit = pd.read_parquet(stage2_candidates_path)
    minute_model, second_model = fit_stage2(stage2_fit)
    scored = add_stage2_scores(
        enriched,
        minute_model,
        second_model,
    )
    first_hot, runtime_audit = _first_hot_feature_rows(
        scored,
        fresh_scan,
    )
    first_hot["ticker"] = first_hot.ticker.astype(str).str.upper()

    found = tuple(sorted(first_hot.trading_day.astype(str).unique()))
    if found != FRESH_DAYS:
        raise ValueError(
            f"Request296 expected fresh days {FRESH_DAYS}, found {found}"
        )

    audit = {
        "fresh_days": list(FRESH_DAYS),
        "active_rows": int(len(enriched)),
        "first_hot_rows": int(len(first_hot)),
        "first_hot_by_day": {
            day: int(first_hot.trading_day.astype(str).eq(day).sum())
            for day in FRESH_DAYS
        },
        "day_scan": day_audit,
        "upstream_second_audit": second_audit,
        "runtime_audit": runtime_audit,
        "flatfile_stats": store.stats.to_dict(),
        "second_client_stats": second_client.stats.to_dict(),
    }
    return first_hot, fresh_scan, audit


def _setup_feature_contract(
    development: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    base_columns = candidate_columns(development)
    context_names = tuple(
        name
        for name in development.columns
        if name.startswith("prehot_lag")
        or name.startswith("prehot_delta")
    )
    ticker_extra = usable_context_columns(
        development,
        context_names,
    )
    rich_columns = _usable(
        development,
        [*base_columns, *ticker_extra],
    )
    return base_columns, rich_columns


def _train_frozen_setup(
    development: pd.DataFrame,
    rich_columns: tuple[str, ...],
):
    fit = development.loc[
        development.crack_complete_60m.astype(bool)
        & _numeric(development.crack_mfe_open_60m_pct).notna()
    ].copy()
    model = _fit(
        fit,
        rich_columns,
        "crack_mfe_open_60m_pct",
        SETUP_MODEL_SEED,
        log_target=True,
    )
    training_score = _predict(fit, model)
    threshold = float(
        np.quantile(
            training_score,
            1.0 - SETUP_SELECTION_FRACTION,
        )
    )
    return model, threshold


def _setup_metrics(frame: pd.DataFrame) -> dict:
    target = _numeric(frame.crack_mfe_open_60m_pct)
    score = _numeric(frame.rich_setup_score_pct)
    valid = target.notna() & score.notna()
    work = frame.loc[valid].copy()
    target = _numeric(work.crack_mfe_open_60m_pct)
    score = _numeric(work.rich_setup_score_pct)
    selected = work.loc[
        work.selected_crack.fillna(False).astype(bool)
    ].copy()
    selected_target = _numeric(
        selected.crack_mfe_open_60m_pct
    )

    result = {
        "rows": int(len(work)),
        "selected_rows": int(len(selected)),
        "selected_rate": (
            float(len(selected) / len(work)) if len(work) else None
        ),
        "spearman": _safe_spearman(target, score),
        "population_mean_mfe_pct": (
            float(target.mean()) if len(target) else None
        ),
        "selected_mean_mfe_pct": (
            float(selected_target.mean()) if len(selected_target) else None
        ),
        "levels": {},
        "by_day": {},
    }
    for threshold in EXPLOSIVE_LEVELS:
        tag = int(threshold)
        pop_rate = (
            float(target.ge(threshold).mean()) if len(target) else None
        )
        selected_rate = (
            float(selected_target.ge(threshold).mean())
            if len(selected_target)
            else None
        )
        ratio = (
            float(selected_rate / pop_rate)
            if pop_rate is not None
            and selected_rate is not None
            and pop_rate > 0
            else None
        )
        result["levels"][f"plus{tag}"] = {
            "population_rate": pop_rate,
            "selected_rate": selected_rate,
            "rate_ratio": ratio,
        }

    for day in FRESH_DAYS:
        part = work.loc[
            work.trading_day.astype(str).eq(day)
        ]
        chosen = part.loc[
            part.selected_crack.fillna(False).astype(bool)
        ]
        y = _numeric(part.crack_mfe_open_60m_pct)
        cy = _numeric(chosen.crack_mfe_open_60m_pct)
        result["by_day"][day] = {
            "rows": int(len(part)),
            "selected_rows": int(len(chosen)),
            "population_mean_mfe_pct": (
                float(y.mean()) if len(y) else None
            ),
            "selected_mean_mfe_pct": (
                float(cy.mean()) if len(cy) else None
            ),
            "population_plus10_rate": (
                float(y.ge(10.0).mean()) if len(y) else None
            ),
            "selected_plus10_rate": (
                float(cy.ge(10.0).mean()) if len(cy) else None
            ),
        }
    return result


def _train_frozen_entry(
    development_states: pd.DataFrame,
    rich_columns: tuple[str, ...],
):
    states = attach_local_turn_features(development_states)
    second_columns = _usable(
        states,
        [
            name
            for name in states.columns
            if name.startswith("current_")
        ],
    )
    entry_columns = _usable(
        states,
        [
            *rich_columns,
            "rich_setup_score_pct",
            *second_columns,
            *EVENT_FEATURES,
            *TURN_FEATURES,
        ],
    )
    turn_columns = _usable(
        states,
        [
            "rich_setup_score_pct",
            *second_columns,
            *EVENT_FEATURES,
            *TURN_FEATURES,
        ],
    )
    entry_model, turn_model = _fit_models(
        states,
        entry_columns,
        turn_columns,
        ENTRY_MODEL_SEED,
    )
    return entry_model, turn_model, entry_columns, turn_columns


def _fresh_policy_metrics(
    decisions: pd.DataFrame,
) -> dict:
    metrics = _policy_metrics(decisions)
    entered = decisions.loc[
        decisions.action.eq("ENTER")
    ].copy()
    metrics["plus5_rate"] = (
        float(
            _numeric(entered.chosen_mfe_fixed_pct)
            .ge(5.0)
            .mean()
        )
        if len(entered)
        else None
    )
    metrics["immediate_plus5_same_entered_rate"] = (
        float(
            _numeric(entered.immediate_mfe_fixed_pct)
            .ge(5.0)
            .mean()
        )
        if len(entered)
        else None
    )
    return metrics


def evaluate(
    stage2_candidates_path: Path,
    development_setup_path: Path,
    development_states_path: Path,
    output_path: Path,
    first_hot_output: Path,
    states_output: Path,
    decisions_output: Path,
) -> int:
    development_setup = pd.read_parquet(
        development_setup_path
    )
    development_states = pd.read_parquet(
        development_states_path
    )

    base_columns, rich_columns = _setup_feature_contract(
        development_setup
    )
    setup_model, setup_threshold = _train_frozen_setup(
        development_setup,
        rich_columns,
    )
    (
        entry_model,
        turn_model,
        entry_columns,
        turn_columns,
    ) = _train_frozen_entry(
        development_states,
        rich_columns,
    )

    fresh_first_hot, fresh_scan, upstream_audit = (
        _prepare_fresh_first_hot(stage2_candidates_path)
    )
    fresh_enriched, _ = attach_ticker_history(
        fresh_first_hot,
        fresh_scan,
    )
    causal_scan = causal_execution_scan(fresh_scan)
    fresh_labeled = attach_crack_targets(
        fresh_enriched,
        causal_scan,
    )
    missing_setup = [
        name
        for name in rich_columns
        if name not in fresh_labeled.columns
    ]
    if missing_setup:
        raise ValueError(
            "Request296 fresh setup missing frozen columns: "
            + ",".join(missing_setup)
        )

    fresh_labeled["rich_setup_score_pct"] = _predict(
        fresh_labeled,
        setup_model,
    )
    fresh_labeled["setup_threshold"] = setup_threshold
    fresh_labeled["selected_crack"] = (
        fresh_labeled.rich_setup_score_pct >= setup_threshold
    )
    setup = _setup_metrics(fresh_labeled)

    selected = fresh_labeled.loc[
        fresh_labeled.selected_crack.astype(bool)
    ].copy()
    raw_states, _, event_audit = build_fixed_horizon_states(
        selected,
        fresh_first_hot,
        causal_scan,
    )
    fresh_states = attach_local_turn_features(raw_states)

    missing_entry = [
        name
        for name in entry_columns
        if name not in fresh_states.columns
    ]
    missing_turn = [
        name
        for name in turn_columns
        if name not in fresh_states.columns
    ]
    if missing_entry or missing_turn:
        raise ValueError(
            "Request296 fresh entry feature mismatch: "
            f"entry={missing_entry}, turn={missing_turn}"
        )

    scored = _score(
        fresh_states,
        entry_model,
        turn_model,
    )
    decisions = _policy(
        scored,
        utility_threshold=FROZEN_UTILITY_THRESHOLD,
        turn_threshold=FROZEN_TURN_THRESHOLD,
        override_threshold=FROZEN_OVERRIDE_THRESHOLD,
    )
    policy = _fresh_policy_metrics(decisions)

    plus5 = setup["levels"]["plus5"]["rate_ratio"]
    plus10 = setup["levels"]["plus10"]["rate_ratio"]
    entry_plus10 = policy.get("plus10_rate")
    immediate_plus10 = policy.get(
        "immediate_plus10_same_entered_rate"
    )
    checks = {
        "fresh_days_complete": (
            tuple(
                sorted(
                    fresh_labeled.trading_day.astype(str).unique()
                )
            )
            == FRESH_DAYS
        ),
        "setup_selected_support": setup["selected_rows"] >= 30,
        "setup_plus5_enrichment": (
            plus5 is not None and plus5 > 1.0
        ),
        "setup_plus10_enrichment": (
            plus10 is not None and plus10 > 1.0
        ),
        "entry_episode_support": policy["episodes"] >= 20,
        "entry_rate_nontrivial": (
            0.20 <= policy["entry_rate"] <= 0.95
        ),
        "entry_beats_immediate_mean": (
            policy["utility_gain_vs_immediate_pp"] > 0
        ),
        "entry_beats_immediate_days": (
            policy["beat_immediate_days"] >= 3
        ),
        "entry_price_improves": (
            policy["price_improvement_pct"] is not None
            and policy["price_improvement_pct"] > 0
        ),
        "entry_mfe_nonlower": (
            policy["mfe_gain_pp"] is not None
            and policy["mfe_gain_pp"] >= 0
        ),
        "entry_plus10_nonlower": (
            entry_plus10 is not None
            and immediate_plus10 is not None
            and entry_plus10 >= immediate_plus10
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "opens_new_dates": True,
        "untouched_validation": True,
        "fresh_days": list(FRESH_DAYS),
        "development_dates_used_for_fitting": [
            "2026-05-05",
            "2026-05-06",
            "2026-05-07",
            "2026-05-08",
        ],
        "fresh_used_for_tuning": False,
        "frozen_policy": {
            "setup_selection_fraction": SETUP_SELECTION_FRACTION,
            "setup_threshold_from_all_development": setup_threshold,
            "utility_threshold": FROZEN_UTILITY_THRESHOLD,
            "turn_threshold": FROZEN_TURN_THRESHOLD,
            "strong_value_override": "disabled",
        },
        "feature_counts": {
            "setup_base": len(base_columns),
            "setup_rich": len(rich_columns),
            "entry": len(entry_columns),
            "turn": len(turn_columns),
        },
        "upstream_audit": upstream_audit,
        "event_audit": event_audit,
        "setup_validation": setup,
        "entry_validation": policy,
        "checks": checks,
        "validation_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "This is a frozen setup+entry validation only. Fixed-horizon "
            "entry utility and MFE are research diagnostics; no claim about "
            "realized P&L is made until HOLD/EXIT is built and validated."
        ),
        "next_boundary_if_pass": (
            "freeze setup+entry and build causal second-resolution HOLD/EXIT "
            "continuation without revisiting June15-19 thresholds"
        ),
        "next_boundary_if_fail": (
            "do not retune on June15-19; use the failure only to identify "
            "which frozen stage failed and return to development data or "
            "reserve a different future block for a later validation"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_hot_output.parent.mkdir(parents=True, exist_ok=True)
    states_output.parent.mkdir(parents=True, exist_ok=True)
    decisions_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fresh_labeled.to_parquet(first_hot_output, index=False)
    scored.to_parquet(states_output, index=False)
    decisions.to_parquet(decisions_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2-candidates",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--development-setup",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--development-states",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--first-hot-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--states-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--decisions-output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.stage2_candidates,
        args.development_setup,
        args.development_states,
        args.output,
        args.first_hot_output,
        args.states_output,
        args.decisions_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
