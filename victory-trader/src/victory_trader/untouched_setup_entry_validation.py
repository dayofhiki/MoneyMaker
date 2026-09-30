"""Request296: untouched validation of frozen crack selection + local-turn entry.

Fresh window: 2026-06-15 through 2026-06-19.

No model family, feature, setup selection fraction, action threshold, or entry
policy may be changed using this fresh window.  Upstream attention and HOT
construction are reconstructed from the frozen Request163 architecture and its
saved stage2 training population.  Crack and entry models are trained only on
the already-opened May5-8 development data.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .attention_flatfile_replay import FLATFILE_CACHE_DIR, build_flatfile_scan_day
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
    SETUP_SELECTION_FRACTION,
    _fit,
    _numeric,
    _predict,
    _safe_spearman,
    _usable,
    attach_crack_targets,
)
from .hot_economic_opportunity import _first_hot_feature_rows
from .lagged_minute_context import CROSSFIT_DAYS
from .local_turn_entry import (
    TURN_FEATURES,
    _fit_models as fit_turn_models,
    _policy as turn_policy,
    _score as score_turn,
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
FRESH_DAYS = [
    "2026-06-15",
    "2026-06-16",
    "2026-06-17",
    "2026-06-18",
    "2026-06-22",
]
ATTENTION_FIT_START = date(2026, 1, 2)
ATTENTION_FIT_END = date(2026, 6, 12)
FRESH_START = date(2026, 6, 15)
FRESH_END = date(2026, 6, 22)
CRACK_SEED = 20263600
ENTRY_SEED = 20263650


def _frozen_action_thresholds(request294: dict) -> dict[str, float]:
    folds = request294.get("outer_folds", {})
    if sorted(folds) != sorted(CROSSFIT_DAYS):
        raise ValueError(
            "Request296 needs the exact four Request294 development folds"
        )
    utility = []
    turn = []
    for day in CROSSFIT_DAYS:
        selected = folds[day]["selected"]
        utility.append(float(selected["utility_threshold"]))
        if selected.get("turn_threshold") is None:
            raise ValueError("Request294 fold unexpectedly disabled turn gate")
        turn.append(float(selected["turn_threshold"]))
    return {
        "utility_threshold": float(np.median(utility)),
        "turn_threshold": float(np.median(turn)),
        # Frozen by Request295 because the override appeared in only one fold.
        "override_threshold": float("inf"),
    }


def _build_fresh_first_hot(
    stage2_candidates: pd.DataFrame,
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

    market_model, active_model = _fit_frozen_attention(
        store,
        scan_client,
        ATTENTION_FIT_START,
        ATTENTION_FIT_END,
    )
    minute_model, second_model = fit_stage2(stage2_candidates)

    active_parts: list[pd.DataFrame] = []
    scan_parts: list[pd.DataFrame] = []
    day_audit: dict[str, dict] = {}

    for day in daterange(FRESH_START, FRESH_END):
        day_text = day.isoformat()
        if not is_us_equity_trading_day(day):
            continue
        scan, scan_audit = build_flatfile_scan_day(store, scan_client, day)
        rows = add_cross_within_horizon_targets(scan, horizons=(HORIZON,))
        active = _build_frozen_active(rows, market_model, active_model)
        active_parts.append(active)
        scan_parts.append(scan)
        day_audit[day_text] = {
            "scan_rows": int(len(scan)),
            "active_rows_before_second": int(len(active)),
            "scan_audit": scan_audit,
        }

    found_days = sorted(day_audit)
    if found_days != FRESH_DAYS:
        raise ValueError(
            f"Request296 expected fresh days {FRESH_DAYS}, found {found_days}"
        )
    if not active_parts or not scan_parts:
        raise ValueError("Request296 fresh upstream population is empty")

    active = pd.concat(active_parts, ignore_index=True)
    scan = pd.concat(scan_parts, ignore_index=True)
    enriched, second_audit = add_second_features(active, second_client)
    scored = add_stage2_scores(enriched, minute_model, second_model)
    first_hot, runtime_audit = _first_hot_feature_rows(scored, scan)
    first_hot["ticker"] = first_hot.ticker.astype(str).str.upper()

    audit = {
        "fresh_days": FRESH_DAYS,
        "active_rows": int(len(active)),
        "enriched_active_rows": int(len(enriched)),
        "first_hot_rows": int(len(first_hot)),
        "days": day_audit,
        "upstream_second_audit": second_audit,
        "runtime_audit": runtime_audit,
        "flatfile_stats": store.stats.to_dict(),
        "scan_client_stats": scan_client.stats.to_dict(),
        "second_client_stats": second_client.stats.to_dict(),
    }
    return first_hot, scan, audit


def _fit_frozen_crack_selector(
    dev_first_hot: pd.DataFrame,
    dev_scan: pd.DataFrame,
) -> tuple[object, tuple[str, ...], float, dict]:
    dev_first_hot = dev_first_hot.loc[
        dev_first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    dev_scan = dev_scan.loc[
        dev_scan.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    dev_causal = causal_execution_scan(dev_scan)
    candidates, ticker_extra = attach_ticker_history(
        dev_first_hot,
        dev_scan,
    )
    ticker_extra = usable_context_columns(candidates, ticker_extra)
    base_columns = candidate_columns(candidates)
    rich_columns = _usable(candidates, [*base_columns, *ticker_extra])
    labeled = attach_crack_targets(candidates, dev_causal)
    train = labeled.loc[
        labeled.crack_complete_60m.astype(bool)
        & _numeric(labeled.crack_mfe_open_60m_pct).notna()
    ].copy()
    if len(train) < 800:
        raise ValueError(
            f"Request296 crack training support too small: {len(train)}"
        )

    model = _fit(
        train,
        rich_columns,
        "crack_mfe_open_60m_pct",
        CRACK_SEED,
        log_target=True,
        episode_weighting=False,
    )
    train_score = _predict(train, model)
    threshold = float(
        np.quantile(train_score, 1.0 - SETUP_SELECTION_FRACTION)
    )
    audit = {
        "training_rows": int(len(train)),
        "feature_count": len(rich_columns),
        "selection_fraction_rule": SETUP_SELECTION_FRACTION,
        "training_score_threshold": threshold,
        "training_score_mean": float(np.mean(train_score)),
    }
    return model, rich_columns, threshold, audit


def _score_fresh_crack(
    fresh_first_hot: pd.DataFrame,
    fresh_scan: pd.DataFrame,
    model,
    columns: tuple[str, ...],
    threshold: float,
) -> pd.DataFrame:
    causal = causal_execution_scan(fresh_scan)
    candidates, ticker_extra = attach_ticker_history(
        fresh_first_hot,
        fresh_scan,
    )
    missing = [name for name in columns if name not in candidates]
    if missing:
        raise ValueError(
            f"Request296 fresh crack features missing: {missing[:20]}"
        )
    scored = attach_crack_targets(candidates, causal)
    scored["rich_setup_score_pct"] = _predict(scored, model)
    scored["setup_threshold"] = float(threshold)
    scored["selected_crack"] = (
        _numeric(scored.rich_setup_score_pct).ge(float(threshold))
    )
    return scored


def _crack_report(scored: pd.DataFrame) -> dict:
    valid = scored.loc[
        scored.crack_complete_60m.astype(bool)
        & _numeric(scored.crack_mfe_open_60m_pct).notna()
    ].copy()
    selected = valid.loc[
        valid.selected_crack.fillna(False).astype(bool)
    ].copy()
    target = _numeric(valid.crack_mfe_open_60m_pct)
    selected_target = _numeric(selected.crack_mfe_open_60m_pct)
    levels = {}
    for threshold in (5.0, 10.0, 20.0):
        population_rate = float(target.ge(threshold).mean())
        selected_rate = (
            float(selected_target.ge(threshold).mean())
            if len(selected)
            else None
        )
        levels[f"plus{int(threshold)}"] = {
            "population_rate": population_rate,
            "selected_rate": selected_rate,
            "rate_ratio": (
                float(selected_rate / population_rate)
                if selected_rate is not None and population_rate > 0
                else None
            ),
            "population_count": int(target.ge(threshold).sum()),
            "selected_count": int(selected_target.ge(threshold).sum()),
        }

    by_day = {}
    for day in FRESH_DAYS:
        part = valid.loc[valid.trading_day.astype(str).eq(day)]
        chosen = part.loc[
            part.selected_crack.fillna(False).astype(bool)
        ]
        y = _numeric(part.crack_mfe_open_60m_pct)
        cy = _numeric(chosen.crack_mfe_open_60m_pct)
        by_day[day] = {
            "rows": int(len(part)),
            "selected_rows": int(len(chosen)),
            "population_mean_mfe_pct": (
                float(y.mean()) if len(y) else None
            ),
            "selected_mean_mfe_pct": (
                float(cy.mean()) if len(cy) else None
            ),
            "plus5_population_rate": (
                float(y.ge(5.0).mean()) if len(y) else None
            ),
            "plus5_selected_rate": (
                float(cy.ge(5.0).mean()) if len(cy) else None
            ),
            "plus10_population_rate": (
                float(y.ge(10.0).mean()) if len(y) else None
            ),
            "plus10_selected_rate": (
                float(cy.ge(10.0).mean()) if len(cy) else None
            ),
        }

    return {
        "rows": int(len(valid)),
        "selected_rows": int(len(selected)),
        "selected_rate": (
            float(len(selected) / len(valid)) if len(valid) else None
        ),
        "score_spearman": _safe_spearman(
            target,
            _numeric(valid.rich_setup_score_pct),
        ),
        "population_mean_mfe_pct": (
            float(target.mean()) if len(target) else None
        ),
        "selected_mean_mfe_pct": (
            float(selected_target.mean()) if len(selected_target) else None
        ),
        "levels": levels,
        "by_day": by_day,
    }


def _fit_frozen_entry(
    dev_states_path: Path,
    dev_first_hot: pd.DataFrame,
    dev_scan: pd.DataFrame,
):
    dev_states = attach_local_turn_features(
        pd.read_parquet(dev_states_path)
    )
    dev_first_hot = dev_first_hot.loc[
        dev_first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    dev_scan = dev_scan.loc[
        dev_scan.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()

    candidates, ticker_extra = attach_ticker_history(
        dev_first_hot,
        dev_scan,
    )
    ticker_extra = usable_context_columns(candidates, ticker_extra)
    base_columns = candidate_columns(candidates)
    static = _usable(dev_states, [*base_columns, *ticker_extra])
    seconds = _usable(
        dev_states,
        [
            name for name in dev_states.columns
            if name.startswith("current_")
        ],
    )
    entry_columns = _usable(
        dev_states,
        [
            *static,
            "rich_setup_score_pct",
            *seconds,
            "elapsed_from_hot_s",
            "active_event_index",
            "last_close_vs_hot_pct",
            "running_low_vs_hot_pct",
            "running_high_vs_hot_pct",
            "drawdown_from_active_high_pct",
            "bounce_from_active_low_pct",
            "seconds_since_active_high",
            "seconds_since_active_low",
            "last_active_return_pct",
            "inter_event_gap_s",
            *TURN_FEATURES,
        ],
    )
    turn_columns = _usable(
        dev_states,
        [
            "rich_setup_score_pct",
            *seconds,
            "elapsed_from_hot_s",
            "active_event_index",
            "last_close_vs_hot_pct",
            "running_low_vs_hot_pct",
            "running_high_vs_hot_pct",
            "drawdown_from_active_high_pct",
            "bounce_from_active_low_pct",
            "seconds_since_active_high",
            "seconds_since_active_low",
            "last_active_return_pct",
            "inter_event_gap_s",
            *TURN_FEATURES,
        ],
    )
    entry_model, turn_model = fit_turn_models(
        dev_states,
        entry_columns,
        turn_columns,
        ENTRY_SEED,
    )
    return entry_model, turn_model, entry_columns, turn_columns


def _policy_report(decisions: pd.DataFrame) -> dict:
    if decisions.empty:
        return {"episodes": 0, "by_day": {}}

    entered = decisions.loc[decisions.action.eq("ENTER")].copy()
    utility = _numeric(decisions.realized_utility_pct)
    immediate = _numeric(decisions.immediate_utility_pct)
    beat_days = 0
    by_day = {}

    for day in FRESH_DAYS:
        part = decisions.loc[
            decisions.trading_day.astype(str).eq(day)
        ].copy()
        pu = _numeric(part.realized_utility_pct)
        pi = _numeric(part.immediate_utility_pct)
        decision_mean = float(pu.mean()) if len(part) else None
        immediate_mean = float(pi.mean()) if len(part) else None
        gain = (
            float(decision_mean - immediate_mean)
            if decision_mean is not None and immediate_mean is not None
            else None
        )
        if gain is not None and gain > 0:
            beat_days += 1
        by_day[day] = {
            "episodes": int(len(part)),
            "entry_rate": (
                float(part.action.eq("ENTER").mean())
                if len(part)
                else None
            ),
            "decision_utility_pct": decision_mean,
            "immediate_utility_pct": immediate_mean,
            "gain_pp": gain,
        }

    result = {
        "episodes": int(len(decisions)),
        "entered": int(len(entered)),
        "entry_rate": float(len(entered) / len(decisions)),
        "skipped": int(decisions.action.eq("SKIP").sum()),
        "mean_decision_utility_pct": float(utility.mean()),
        "mean_immediate_utility_pct": float(immediate.mean()),
        "utility_gain_vs_immediate_pp": float(
            (utility - immediate).mean()
        ),
        "beat_immediate_days": int(beat_days),
        "by_day": by_day,
        "mean_delay_s": (
            float(_numeric(entered.delay_s).mean())
            if len(entered)
            else None
        ),
        "median_delay_s": (
            float(_numeric(entered.delay_s).median())
            if len(entered)
            else None
        ),
        "mean_entry_price_improvement_pct": (
            float(
                _numeric(
                    entered.entry_price_improvement_pct
                ).mean()
            )
            if len(entered)
            else None
        ),
        "mean_chosen_mfe_pct": (
            float(_numeric(entered.chosen_mfe_fixed_pct).mean())
            if len(entered)
            else None
        ),
        "mean_immediate_mfe_same_entered_pct": (
            float(
                _numeric(
                    entered.immediate_mfe_fixed_pct
                ).mean()
            )
            if len(entered)
            else None
        ),
        "mfe_gain_same_entered_pp": (
            float(
                (
                    _numeric(entered.chosen_mfe_fixed_pct)
                    - _numeric(
                        entered.immediate_mfe_fixed_pct
                    )
                ).mean()
            )
            if len(entered)
            else None
        ),
        "plus10_chosen_rate": (
            float(
                _numeric(entered.chosen_mfe_fixed_pct)
                .ge(10.0)
                .mean()
            )
            if len(entered)
            else None
        ),
        "plus10_immediate_same_entered_rate": (
            float(
                _numeric(entered.immediate_mfe_fixed_pct)
                .ge(10.0)
                .mean()
            )
            if len(entered)
            else None
        ),
    }
    return result


def evaluate(
    stage2_path: Path,
    dev_first_hot_path: Path,
    dev_scan_path: Path,
    dev_states_path: Path,
    request294_path: Path,
    output_path: Path,
    fresh_first_hot_output: Path,
    fresh_setup_output: Path,
    fresh_states_output: Path,
    fresh_decisions_output: Path,
) -> int:
    stage2_candidates = pd.read_parquet(stage2_path)
    dev_first_hot = pd.read_parquet(dev_first_hot_path)
    dev_first_hot["ticker"] = dev_first_hot.ticker.astype(str).str.upper()
    dev_scan = pd.read_parquet(dev_scan_path)
    request294 = json.loads(
        request294_path.read_text(encoding="utf-8")
    )
    action_thresholds = _frozen_action_thresholds(request294)

    # Fresh market data are opened here once. Everything above is frozen from
    # development artifacts or fixed code.
    fresh_first_hot, fresh_scan, upstream_audit = _build_fresh_first_hot(
        stage2_candidates
    )

    crack_model, crack_columns, crack_threshold, crack_fit_audit = (
        _fit_frozen_crack_selector(dev_first_hot, dev_scan)
    )
    fresh_setup = _score_fresh_crack(
        fresh_first_hot,
        fresh_scan,
        crack_model,
        crack_columns,
        crack_threshold,
    )
    crack = _crack_report(fresh_setup)

    selected = fresh_setup.loc[
        fresh_setup.selected_crack.fillna(False).astype(bool)
    ].copy()
    fresh_causal = causal_execution_scan(fresh_scan)
    fresh_states, _, fresh_event_audit = build_fixed_horizon_states(
        selected,
        fresh_first_hot,
        fresh_causal,
    )
    fresh_states = attach_local_turn_features(fresh_states)

    entry_model, turn_model, entry_columns, turn_columns = _fit_frozen_entry(
        dev_states_path,
        dev_first_hot,
        dev_scan,
    )
    missing_entry = [
        name for name in entry_columns if name not in fresh_states
    ]
    missing_turn = [
        name for name in turn_columns if name not in fresh_states
    ]
    if missing_entry or missing_turn:
        raise ValueError(
            "Request296 fresh entry feature mismatch: "
            f"entry={missing_entry[:20]} turn={missing_turn[:20]}"
        )

    fresh_scored = score_turn(
        fresh_states,
        entry_model,
        turn_model,
    )
    fresh_decisions = turn_policy(
        fresh_scored,
        utility_threshold=action_thresholds["utility_threshold"],
        turn_threshold=action_thresholds["turn_threshold"],
        override_threshold=action_thresholds["override_threshold"],
    )
    policy = _policy_report(fresh_decisions)

    plus5_ratio = crack["levels"]["plus5"]["rate_ratio"]
    plus10_ratio = crack["levels"]["plus10"]["rate_ratio"]
    checks = {
        "fresh_days_exact": sorted(
            fresh_first_hot.trading_day.astype(str).unique()
        ) == FRESH_DAYS,
        "crack_plus5_enrichment": (
            plus5_ratio is not None and plus5_ratio > 1.0
        ),
        "crack_plus10_enrichment": (
            plus10_ratio is not None and plus10_ratio > 1.0
        ),
        "entry_beats_immediate_mean": (
            policy.get("utility_gain_vs_immediate_pp") is not None
            and policy["utility_gain_vs_immediate_pp"] > 0
        ),
        "entry_beats_immediate_3_of_5_days": (
            policy.get("beat_immediate_days", 0) >= 3
        ),
        "entry_price_improves": (
            policy.get("mean_entry_price_improvement_pct") is not None
            and policy["mean_entry_price_improvement_pct"] > 0
        ),
        "entry_mfe_nonlower": (
            policy.get("mfe_gain_same_entered_pp") is not None
            and policy["mfe_gain_same_entered_pp"] >= 0
        ),
        "entry_plus10_nonlower": (
            policy.get("plus10_chosen_rate") is not None
            and policy.get("plus10_immediate_same_entered_rate") is not None
            and policy["plus10_chosen_rate"]
            >= policy["plus10_immediate_same_entered_rate"]
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "fresh_validation": True,
        "fresh_days": FRESH_DAYS,
        "development_days": list(CROSSFIT_DAYS),
        "fresh_data_used_for_tuning": False,
        "upstream_source": (
            "Request163 frozen attention architecture + saved Request163 "
            "stage2 training population"
        ),
        "frozen_policy": {
            "setup_selection_fraction_rule": SETUP_SELECTION_FRACTION,
            "crack_training_seed": CRACK_SEED,
            "entry_training_seed": ENTRY_SEED,
            "action_thresholds_from_request294_fold_medians": action_thresholds,
            "strong_value_override": "disabled",
        },
        "upstream_audit": upstream_audit,
        "crack_fit_audit": crack_fit_audit,
        "crack_validation": crack,
        "entry_event_audit": fresh_event_audit,
        "entry_feature_counts": {
            "entry": len(entry_columns),
            "turn": len(turn_columns),
        },
        "entry_validation": policy,
        "checks": checks,
        "fresh_gate_pass": bool(all(checks.values())),
        "interpretation_boundary": (
            "This validates selection and entry only. HOLD/EXIT remains "
            "unfrozen, so these utility/MFE results are not realized trading P&L."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fresh_first_hot_output.parent.mkdir(parents=True, exist_ok=True)
    fresh_setup_output.parent.mkdir(parents=True, exist_ok=True)
    fresh_states_output.parent.mkdir(parents=True, exist_ok=True)
    fresh_decisions_output.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fresh_first_hot.to_parquet(fresh_first_hot_output, index=False)
    fresh_setup.to_parquet(fresh_setup_output, index=False)
    fresh_scored.to_parquet(fresh_states_output, index=False)
    fresh_decisions.to_parquet(fresh_decisions_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-candidates", type=Path, required=True)
    parser.add_argument("--dev-first-hot", type=Path, required=True)
    parser.add_argument("--dev-scan", type=Path, required=True)
    parser.add_argument("--dev-states", type=Path, required=True)
    parser.add_argument("--request294-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-first-hot-output", type=Path, required=True)
    parser.add_argument("--fresh-setup-output", type=Path, required=True)
    parser.add_argument("--fresh-states-output", type=Path, required=True)
    parser.add_argument("--fresh-decisions-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.stage2_candidates,
        args.dev_first_hot,
        args.dev_scan,
        args.dev_states,
        args.request294_json,
        args.output,
        args.fresh_first_hot_output,
        args.fresh_setup_output,
        args.fresh_states_output,
        args.fresh_decisions_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
