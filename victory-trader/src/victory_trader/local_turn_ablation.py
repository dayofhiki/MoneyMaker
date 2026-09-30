"""Request295: matched ablation of the first positive entry-timing edge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .direct_attractive_entry import (
    _calibrate_threshold as calibrate_direct,
    _policy as direct_policy,
    signal_report as direct_signal_report,
)
from .downstream_aligned_candidate import candidate_columns
from .hierarchical_crack_entry_controller import (
    EVENT_FEATURES,
    _fit,
    _numeric,
    _predict,
    _usable,
)
from .lagged_minute_context import CROSSFIT_DAYS
from .local_turn_entry import (
    TURN_FEATURES,
    _calibrate as calibrate_turn,
    _fit_models as fit_turn_models,
    _policy as turn_policy,
    _score as score_turn,
    attach_local_turn_features,
)
from .prehot_context_ablation import usable_context_columns
from .two_stage_risk_veto import attach_ticker_history

REQUEST_ID = 295


def _direct_crossfit(
    states: pd.DataFrame,
    columns: tuple[str, ...],
    seed_base: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scored_parts = []
    decision_parts = []
    folds = {}
    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train_days = [day for day in CROSSFIT_DAYS if day != held_day]
        calibration_day = train_days[-1]
        inner_days = train_days[:-1]

        inner = states.loc[
            states.trading_day.astype(str).isin(inner_days)
        ].copy()
        calibration = states.loc[
            states.trading_day.astype(str).eq(calibration_day)
        ].copy()
        outer = states.loc[
            states.trading_day.astype(str).isin(train_days)
        ].copy()
        held = states.loc[
            states.trading_day.astype(str).eq(held_day)
        ].copy()

        inner_model = _fit(
            inner,
            columns,
            "entry_utility_fixed_pct",
            seed_base + fold_index * 50,
            log_target=False,
            episode_weighting=True,
        )
        threshold, cal_report = calibrate_direct(
            inner,
            calibration,
            inner_model,
        )
        model = _fit(
            outer,
            columns,
            "entry_utility_fixed_pct",
            seed_base + 25 + fold_index * 50,
            log_target=False,
            episode_weighting=True,
        )
        held["predicted_enter_utility_pct"] = _predict(held, model)
        held["direct_threshold"] = threshold
        held["held_out_day"] = held_day
        scored_parts.append(held)

        decisions = direct_policy(held, threshold)
        decisions["held_out_day"] = held_day
        decision_parts.append(decisions)
        folds[held_day] = {
            "inner_fit_days": inner_days,
            "calibration_day": calibration_day,
            "threshold": threshold,
            "calibration": cal_report["selected"],
        }

    return (
        pd.concat(scored_parts, ignore_index=True),
        pd.concat(decision_parts, ignore_index=True),
        folds,
    )


def _turn_crossfit(
    states: pd.DataFrame,
    entry_columns: tuple[str, ...],
    turn_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scored_parts = []
    decision_parts = []
    folds = {}
    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train_days = [day for day in CROSSFIT_DAYS if day != held_day]
        calibration_day = train_days[-1]
        inner_days = train_days[:-1]
        inner = states.loc[
            states.trading_day.astype(str).isin(inner_days)
        ].copy()
        calibration = states.loc[
            states.trading_day.astype(str).eq(calibration_day)
        ].copy()
        outer = states.loc[
            states.trading_day.astype(str).isin(train_days)
        ].copy()
        held = states.loc[
            states.trading_day.astype(str).eq(held_day)
        ].copy()

        inner_entry, inner_turn = fit_turn_models(
            inner,
            entry_columns,
            turn_columns,
            20263500 + fold_index * 50,
        )
        inner_scored = score_turn(inner, inner_entry, inner_turn)
        cal_scored = score_turn(calibration, inner_entry, inner_turn)
        selected, _ = calibrate_turn(inner_scored, cal_scored)

        entry_model, turn_model = fit_turn_models(
            outer,
            entry_columns,
            turn_columns,
            20263525 + fold_index * 50,
        )
        held_scored = score_turn(held, entry_model, turn_model)
        held_scored["held_out_day"] = held_day
        scored_parts.append(held_scored)

        turn_threshold = (
            -np.inf
            if selected["turn_threshold"] is None
            else float(selected["turn_threshold"])
        )
        override_threshold = (
            np.inf
            if selected["override_threshold"] is None
            else float(selected["override_threshold"])
        )
        decisions = turn_policy(
            held_scored,
            utility_threshold=float(selected["utility_threshold"]),
            turn_threshold=turn_threshold,
            override_threshold=override_threshold,
        )
        decisions["held_out_day"] = held_day
        decision_parts.append(decisions)
        folds[held_day] = {
            "inner_fit_days": inner_days,
            "calibration_day": calibration_day,
            "selected": selected,
        }

    return (
        pd.concat(scored_parts, ignore_index=True),
        pd.concat(decision_parts, ignore_index=True),
        folds,
    )


def _policy_metrics(decisions: pd.DataFrame) -> dict:
    entered = decisions.loc[decisions.action.eq("ENTER")].copy()
    utility = _numeric(decisions.realized_utility_pct)
    immediate = _numeric(decisions.immediate_utility_pct)
    by_day = {}
    beat_days = 0
    for day, part in decisions.groupby(
        decisions.trading_day.astype(str), sort=True
    ):
        u = float(_numeric(part.realized_utility_pct).mean())
        imm = float(_numeric(part.immediate_utility_pct).mean())
        if u > imm:
            beat_days += 1
        by_day[str(day)] = {
            "decision_utility_pct": u,
            "immediate_utility_pct": imm,
            "gain_pp": float(u - imm),
            "episodes": int(len(part)),
            "entry_rate": float(part.action.eq("ENTER").mean()),
        }
    plus10 = (
        float(_numeric(entered.chosen_mfe_fixed_pct).ge(10.0).mean())
        if len(entered)
        else None
    )
    immediate_plus10 = (
        float(_numeric(entered.immediate_mfe_fixed_pct).ge(10.0).mean())
        if len(entered)
        else None
    )
    return {
        "episodes": int(len(decisions)),
        "entered": int(len(entered)),
        "entry_rate": float(len(entered) / len(decisions)),
        "mean_decision_utility_pct": float(utility.mean()),
        "mean_immediate_utility_pct": float(immediate.mean()),
        "utility_gain_vs_immediate_pp": float((utility - immediate).mean()),
        "beat_immediate_days": int(beat_days),
        "by_day": by_day,
        "mean_delay_s": (
            float(_numeric(entered.delay_s).mean()) if len(entered) else None
        ),
        "median_delay_s": (
            float(_numeric(entered.delay_s).median()) if len(entered) else None
        ),
        "price_improvement_pct": (
            float(_numeric(entered.entry_price_improvement_pct).mean())
            if len(entered)
            else None
        ),
        "mfe_gain_pp": (
            float(
                (
                    _numeric(entered.chosen_mfe_fixed_pct)
                    - _numeric(entered.immediate_mfe_fixed_pct)
                ).mean()
            )
            if len(entered)
            else None
        ),
        "plus10_rate": plus10,
        "immediate_plus10_same_entered_rate": immediate_plus10,
    }


def evaluate(
    state_path: Path,
    first_hot_path: Path,
    scan_path: Path,
    request294_path: Path,
    output_path: Path,
    decisions_output: Path,
) -> int:
    states = attach_local_turn_features(pd.read_parquet(state_path))
    request294 = json.loads(request294_path.read_text(encoding="utf-8"))

    first_hot = pd.read_parquet(first_hot_path)
    first_hot = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    raw_scan = pd.read_parquet(scan_path)
    raw_scan = raw_scan.loc[
        raw_scan.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()

    candidates, ticker_extra = attach_ticker_history(first_hot, raw_scan)
    ticker_extra = usable_context_columns(candidates, ticker_extra)
    base_candidate = candidate_columns(candidates)
    static = _usable(states, [*base_candidate, *ticker_extra])
    seconds = _usable(
        states,
        [name for name in states.columns if name.startswith("current_")],
    )
    base_columns = _usable(
        states,
        [
            *static,
            "rich_setup_score_pct",
            *seconds,
            *EVENT_FEATURES,
        ],
    )
    enriched_columns = _usable(
        states,
        [*base_columns, *TURN_FEATURES],
    )
    turn_columns = _usable(
        states,
        [
            "rich_setup_score_pct",
            *seconds,
            *EVENT_FEATURES,
            *TURN_FEATURES,
        ],
    )

    base_scored, base_decisions, base_folds = _direct_crossfit(
        states, base_columns, 20263400
    )
    enriched_scored, enriched_decisions, enriched_folds = _direct_crossfit(
        states, enriched_columns, 20263500
    )
    full_scored, full_decisions, full_folds = _turn_crossfit(
        states, enriched_columns, turn_columns
    )

    variants = {
        "base_direct": {
            "signal": direct_signal_report(base_scored),
            "policy": _policy_metrics(base_decisions),
            "folds": base_folds,
        },
        "turn_enriched_direct": {
            "signal": direct_signal_report(enriched_scored),
            "policy": _policy_metrics(enriched_decisions),
            "folds": enriched_folds,
        },
        "full_turn_gate": {
            "signal": direct_signal_report(full_scored),
            "policy": _policy_metrics(full_decisions),
            "folds": full_folds,
        },
    }

    r294_utility = (
        request294.get("policy_metrics", {})
        .get("mean_decision_utility_pct")
    )
    full_utility = variants["full_turn_gate"]["policy"][
        "mean_decision_utility_pct"
    ]
    checks = {
        "reproduces_request294": (
            r294_utility is not None
            and abs(float(r294_utility) - float(full_utility)) < 1e-9
        ),
        "enriched_signal_beats_base": (
            variants["turn_enriched_direct"]["signal"]["utility_spearman"]
            > variants["base_direct"]["signal"]["utility_spearman"]
        ),
        "full_beats_base_utility": (
            variants["full_turn_gate"]["policy"][
                "mean_decision_utility_pct"
            ]
            > variants["base_direct"]["policy"][
                "mean_decision_utility_pct"
            ]
        ),
        "full_beats_enriched_direct_utility": (
            variants["full_turn_gate"]["policy"][
                "mean_decision_utility_pct"
            ]
            > variants["turn_enriched_direct"]["policy"][
                "mean_decision_utility_pct"
            ]
        ),
        "full_beats_immediate_majority_days": (
            variants["full_turn_gate"]["policy"]["beat_immediate_days"] >= 3
        ),
        "full_price_improves": (
            variants["full_turn_gate"]["policy"]["price_improvement_pct"] > 0
        ),
        "full_mfe_nonlower": (
            variants["full_turn_gate"]["policy"]["mfe_gain_pp"] >= 0
        ),
    }

    rows = []
    for name, decisions in (
        ("base_direct", base_decisions),
        ("turn_enriched_direct", enriched_decisions),
        ("full_turn_gate", full_decisions),
    ):
        part = decisions.copy()
        part["variant"] = name
        rows.append(part)

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "purpose": (
            "ablate whether Request294's first positive entry-timing edge "
            "came from turn features improving direct value, from the "
            "separate turn gate, or both"
        ),
        "feature_counts": {
            "base_direct": len(base_columns),
            "turn_enriched_direct": len(enriched_columns),
            "turn_gate": len(turn_columns),
        },
        "variants": variants,
        "checks": checks,
        "research_gate_pass": bool(all(checks.values())),
        "next_boundary": (
            "if full turn gate is stable across held-out days, freeze this "
            "entry architecture before untouched-date validation; otherwise "
            "keep development dates closed and simplify the unstable component"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.concat(rows, ignore_index=True).to_parquet(
        decisions_output, index=False
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-oof", type=Path, required=True)
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--request294-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.state_oof,
        args.first_hot,
        args.opportunity_scan,
        args.request294_json,
        args.output,
        args.decisions_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
