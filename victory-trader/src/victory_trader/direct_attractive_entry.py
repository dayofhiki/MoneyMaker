"""Request293: direct attractive-entry stopping.

Request292 showed that a separate WAIT-value model was essentially random, while
the entries that were actually taken were cheaper and retained more upside.

Request293 removes the WAIT model. At every causal active-second state it asks
only whether ENTER now has enough predicted fixed-horizon value. The first state
whose predicted value clears a nested-calibrated threshold is entered; otherwise
observation continues naturally. If no state clears the threshold, the setup is
skipped.

The setup population and HOT+60m fixed-terminal labels are inherited from
Request292. No new dates are opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .downstream_aligned_candidate import candidate_columns
from .fixed_horizon_fitted_entry import (
    MIN_CAL_ENTRY_RATE,
)
from .hierarchical_crack_entry_controller import (
    EVENT_FEATURES,
    _fit,
    _numeric,
    _predict,
    _safe_auc,
    _safe_spearman,
    _usable,
)
from .lagged_minute_context import CROSSFIT_DAYS
from .prehot_context_ablation import usable_context_columns
from .two_stage_risk_veto import attach_ticker_history

REQUEST_ID = 293
THRESHOLD_QUANTILES = (0.20, 0.35, 0.50, 0.65, 0.80)
MIN_ENTRY_RATE = 0.35
MAX_ENTRY_RATE = 0.95


def _policy(
    scored: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    keys = ["trading_day", "ticker", "hot_t"]
    for _, episode in scored.groupby(keys, sort=False):
        ordered = episode.sort_values("decision_t", kind="stable")
        if ordered.empty:
            continue
        immediate = ordered.iloc[0]
        qualified = ordered.loc[
            _numeric(ordered.predicted_enter_utility_pct).ge(float(threshold))
        ]
        if qualified.empty:
            rows.append(
                {
                    "trading_day": str(immediate.trading_day),
                    "ticker": str(immediate.ticker).upper(),
                    "hot_t": int(immediate.hot_t),
                    "action": "SKIP",
                    "delay_s": np.nan,
                    "immediate_price": float(immediate.execution_price),
                    "chosen_price": np.nan,
                    "entry_price_improvement_pct": np.nan,
                    "realized_utility_pct": 0.0,
                    "chosen_mfe_fixed_pct": 0.0,
                    "chosen_mae_fixed_pct": 0.0,
                    "immediate_utility_pct": float(
                        immediate.entry_utility_fixed_pct
                    ),
                    "immediate_mfe_fixed_pct": float(
                        immediate.entry_mfe_fixed_pct
                    ),
                    "immediate_mae_fixed_pct": float(
                        immediate.entry_mae_fixed_pct
                    ),
                }
            )
            continue

        chosen = qualified.iloc[0]
        rows.append(
            {
                "trading_day": str(chosen.trading_day),
                "ticker": str(chosen.ticker).upper(),
                "hot_t": int(chosen.hot_t),
                "action": "ENTER",
                "delay_s": float(
                    (int(chosen.decision_t) - int(immediate.decision_t)) / 1000.0
                ),
                "immediate_price": float(immediate.execution_price),
                "chosen_price": float(chosen.execution_price),
                "entry_price_improvement_pct": float(
                    (
                        float(immediate.execution_price)
                        / float(chosen.execution_price)
                        - 1.0
                    )
                    * 100.0
                ),
                "realized_utility_pct": float(chosen.entry_utility_fixed_pct),
                "chosen_mfe_fixed_pct": float(chosen.entry_mfe_fixed_pct),
                "chosen_mae_fixed_pct": float(chosen.entry_mae_fixed_pct),
                "immediate_utility_pct": float(
                    immediate.entry_utility_fixed_pct
                ),
                "immediate_mfe_fixed_pct": float(
                    immediate.entry_mfe_fixed_pct
                ),
                "immediate_mae_fixed_pct": float(
                    immediate.entry_mae_fixed_pct
                ),
            }
        )
    return pd.DataFrame(rows)


def _objective(decisions: pd.DataFrame) -> tuple[float, float, float]:
    if decisions.empty:
        return -np.inf, 0.0, np.inf
    entry_rate = float(decisions.action.eq("ENTER").mean())
    mean_utility = float(_numeric(decisions.realized_utility_pct).mean())
    entered = decisions.loc[decisions.action.eq("ENTER")]
    delay = (
        float(_numeric(entered.delay_s).mean())
        if len(entered)
        else np.inf
    )
    return mean_utility, entry_rate, delay


def _calibrate_threshold(
    inner_fit: pd.DataFrame,
    calibration: pd.DataFrame,
    model,
) -> tuple[float, dict]:
    fit_score = pd.Series(
        _predict(inner_fit, model),
        index=inner_fit.index,
        dtype=float,
    )
    candidates = [float(fit_score.min()) - 1e-6]
    candidates.extend(
        float(fit_score.quantile(q))
        for q in THRESHOLD_QUANTILES
    )
    cal = calibration.copy()
    cal["predicted_enter_utility_pct"] = _predict(cal, model)

    reports = []
    for threshold in candidates:
        decisions = _policy(cal, threshold)
        utility, entry_rate, delay = _objective(decisions)
        reports.append(
            {
                "threshold": float(threshold),
                "mean_decision_utility_pct": utility,
                "entry_rate": entry_rate,
                "mean_delay_s": delay,
                "eligible": bool(
                    MIN_CAL_ENTRY_RATE <= entry_rate <= MAX_ENTRY_RATE
                ),
            }
        )
    eligible = [row for row in reports if row["eligible"]]
    pool = eligible if eligible else reports
    best = max(
        pool,
        key=lambda row: (
            row["mean_decision_utility_pct"],
            -row["mean_delay_s"],
            row["entry_rate"],
        ),
    )
    return float(best["threshold"]), {
        "selected": best,
        "grid": reports,
    }


def crossfit_direct_stopping(
    states: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scored_parts = []
    decision_parts = []
    folds = {}

    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train_days = [
            day for day in CROSSFIT_DAYS if day != held_day
        ]
        calibration_day = train_days[-1]
        inner_fit_days = train_days[:-1]

        inner_fit = states.loc[
            states.trading_day.astype(str).isin(inner_fit_days)
        ].copy()
        calibration = states.loc[
            states.trading_day.astype(str).eq(calibration_day)
        ].copy()
        outer_train = states.loc[
            states.trading_day.astype(str).isin(train_days)
        ].copy()
        held = states.loc[
            states.trading_day.astype(str).eq(held_day)
        ].copy()

        inner_model = _fit(
            inner_fit,
            columns,
            "entry_utility_fixed_pct",
            20263400 + fold_index * 50,
            log_target=False,
            episode_weighting=True,
        )
        threshold, cal_report = _calibrate_threshold(
            inner_fit,
            calibration,
            inner_model,
        )

        model = _fit(
            outer_train,
            columns,
            "entry_utility_fixed_pct",
            20263425 + fold_index * 50,
            log_target=False,
            episode_weighting=True,
        )
        held["predicted_enter_utility_pct"] = _predict(held, model)
        held["direct_threshold"] = threshold
        held["held_out_day"] = held_day
        scored_parts.append(held)

        decisions = _policy(held, threshold)
        decisions["held_out_day"] = held_day
        decision_parts.append(decisions)

        folds[held_day] = {
            "inner_fit_days": inner_fit_days,
            "calibration_day": calibration_day,
            "outer_train_days": train_days,
            "threshold": threshold,
            "calibration": cal_report["selected"],
        }

    return (
        pd.concat(scored_parts, ignore_index=True),
        pd.concat(decision_parts, ignore_index=True),
        folds,
    )


def signal_report(scored: pd.DataFrame) -> dict:
    actual = _numeric(scored.entry_utility_fixed_pct)
    predicted = _numeric(scored.predicted_enter_utility_pct)
    valid = actual.notna() & predicted.notna()
    actual = actual.loc[valid]
    predicted = predicted.loc[valid]
    return {
        "rows": int(valid.sum()),
        "utility_spearman": _safe_spearman(actual, predicted),
        "positive_utility_rate": float(actual.gt(0).mean()),
        "positive_utility_auc": _safe_auc(
            actual.gt(0).astype(int),
            predicted,
        ),
    }


def policy_report(
    decisions: pd.DataFrame,
    request292: dict,
) -> dict:
    entered = decisions.loc[decisions.action.eq("ENTER")].copy()
    utility = _numeric(decisions.realized_utility_pct)
    immediate = _numeric(decisions.immediate_utility_pct)

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
        "mean_delay_s_entered": (
            float(_numeric(entered.delay_s).mean())
            if len(entered)
            else None
        ),
        "median_delay_s_entered": (
            float(_numeric(entered.delay_s).median())
            if len(entered)
            else None
        ),
        "mean_entry_price_improvement_pct": (
            float(_numeric(entered.entry_price_improvement_pct).mean())
            if len(entered)
            else None
        ),
        "mean_chosen_mfe_pct": (
            float(_numeric(entered.chosen_mfe_fixed_pct).mean())
            if len(entered)
            else None
        ),
        "mean_immediate_mfe_same_entered_pct": (
            float(_numeric(entered.immediate_mfe_fixed_pct).mean())
            if len(entered)
            else None
        ),
        "mfe_gain_same_entered_pp": (
            float(
                (
                    _numeric(entered.chosen_mfe_fixed_pct)
                    - _numeric(entered.immediate_mfe_fixed_pct)
                ).mean()
            )
            if len(entered)
            else None
        ),
        "mean_chosen_mae_pct": (
            float(_numeric(entered.chosen_mae_fixed_pct).mean())
            if len(entered)
            else None
        ),
        "levels": {},
    }
    for threshold in (5.0, 10.0, 20.0):
        tag = int(threshold)
        result["levels"][f"plus{tag}"] = {
            "chosen_rate": (
                float(_numeric(entered.chosen_mfe_fixed_pct).ge(threshold).mean())
                if len(entered)
                else None
            ),
            "immediate_same_entered_rate": (
                float(
                    _numeric(entered.immediate_mfe_fixed_pct)
                    .ge(threshold)
                    .mean()
                )
                if len(entered)
                else None
            ),
        }

    old_policy = (
        request292.get("fitted_stopping", {})
        .get("policy", {})
    )
    result["request292_comparison"] = {
        "mean_decision_utility_pct": old_policy.get(
            "mean_decision_utility_pct"
        ),
        "entry_rate": old_policy.get("entry_rate"),
        "mean_entry_price_improvement_pct": old_policy.get(
            "mean_entry_price_improvement_pct"
        ),
        "mfe_gain_same_entered_pp": old_policy.get(
            "mfe_gain_same_entered_pp"
        ),
    }
    result["fixed_2pct"] = old_policy.get("fixed_2pct", {})
    return result


def evaluate(
    state_path: Path,
    first_hot_path: Path,
    scan_path: Path,
    request292_path: Path,
    output_path: Path,
    scored_output: Path,
    decisions_output: Path,
) -> int:
    states = pd.read_parquet(state_path)
    request292 = json.loads(
        request292_path.read_text(encoding="utf-8")
    )

    first_hot = pd.read_parquet(first_hot_path)
    first_hot = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    raw_scan = pd.read_parquet(scan_path)
    raw_scan = raw_scan.loc[
        raw_scan.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()

    candidates, ticker_extra = attach_ticker_history(
        first_hot,
        raw_scan,
    )
    ticker_extra = usable_context_columns(candidates, ticker_extra)
    base_columns = candidate_columns(candidates)
    rich_columns = _usable(
        states,
        [*base_columns, *ticker_extra],
    )
    second_columns = _usable(
        states,
        [
            name for name in states.columns
            if name.startswith("current_")
        ],
    )
    columns = _usable(
        states,
        [
            *rich_columns,
            "rich_setup_score_pct",
            *second_columns,
            *EVENT_FEATURES,
        ],
    )

    scored, decisions, folds = crossfit_direct_stopping(
        states,
        columns,
    )
    signal = signal_report(scored)
    policy = policy_report(decisions, request292)
    fixed_cash = policy.get("fixed_2pct", {}).get(
        "cash_adjusted_mean_utility_pct"
    )

    checks = {
        "direct_signal_positive": (
            signal["utility_spearman"] is not None
            and signal["utility_spearman"] > 0
        ),
        "direct_positive_auc": (
            signal["positive_utility_auc"] is not None
            and signal["positive_utility_auc"] > 0.55
        ),
        "entry_rate": (
            MIN_ENTRY_RATE <= policy["entry_rate"] <= MAX_ENTRY_RATE
        ),
        "utility_beats_immediate": (
            policy["utility_gain_vs_immediate_pp"] > 0
        ),
        "price_improves": (
            policy["mean_entry_price_improvement_pct"] is not None
            and policy["mean_entry_price_improvement_pct"] > 0
        ),
        "mfe_nonlower": (
            policy["mfe_gain_same_entered_pp"] is not None
            and policy["mfe_gain_same_entered_pp"] >= 0
        ),
        "beats_fixed_2pct": (
            fixed_cash is None
            or policy["mean_decision_utility_pct"] > fixed_cash
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "setup_frozen_from_request291": True,
        "state_labels_reused_from_request292": True,
        "policy": (
            "at each active second ENTER at the first state whose predicted "
            "current fixed-horizon utility clears the nested-calibrated "
            "threshold; otherwise continue observing; SKIP if never cleared"
        ),
        "feature_count": len(columns),
        "outer_folds": folds,
        "signal": signal,
        "policy_metrics": policy,
        "checks": checks,
        "research_gate_pass": bool(all(checks.values())),
        "next_boundary": (
            "if direct stopping beats immediate, freeze setup+entry and move "
            "to second-resolution HOLD/EXIT continuation; if not, inspect "
            "direct utility representation and local turning features without "
            "reintroducing a separate WAIT-value model"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_output.parent.mkdir(parents=True, exist_ok=True)
    decisions_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    scored.to_parquet(scored_output, index=False)
    decisions.to_parquet(decisions_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-oof", type=Path, required=True)
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--request292-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.state_oof,
        args.first_hot,
        args.opportunity_scan,
        args.request292_json,
        args.output,
        args.scored_output,
        args.decisions_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
