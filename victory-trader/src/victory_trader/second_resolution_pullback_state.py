"""Request286: causal completed-second context at the original pullback.

Request285 showed that one extra minute improves winner classification but
destroys entry economics. This diagnostic keeps the original first causal 2%
pullback decision, entry and downstream outcome frozen, then asks whether
completed one-second aggregates already available before that decision recover
the missing information without waiting.

Second features consume only completed one-second aggregates ending at
decision_t - 1 second. No post-decision second is a feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .causal_minute_controller_rebuild import causal_execution_scan
from .downstream_aligned_candidate import (
    attach_downstream_target,
    candidate_columns,
)
from .full_hot_fixed_policy_value import attach_fixed_value
from .joint_pullback_admission import (
    FIXED_MAX_HOLD_MINUTES,
    FIXED_STOP_LOSS_PCT,
    FIXED_TRAIL_PCT,
    feature_x,
)
from .lagged_minute_context import CROSSFIT_DAYS, fit_triplet
from .pullback_trailing_exit import (
    apply_exit_rule,
    build_pullback_episodes,
)
from .rich_post_hot_state import (
    CURRENT_SECOND_FEATURES,
    attach_second_features,
)
from .two_stage_risk_veto import (
    attach_ticker_history,
    fold_stage1,
    prepare_pullback_states,
)

REQUEST_ID = 286
MIN_CONDITIONED_RESOLVED = 30
MIN_SECOND_COVERAGE = 0.80
MIN_RICH_VALUE_SPEARMAN = 0.20
MIN_RICH_POSITIVE_AUC = 0.60
MIN_RICH_SEVERE_AUC = 0.60
MIN_METRIC_GAIN = 0.05
MIN_GAINED_METRICS = 2


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def usable_second_columns(
    frame: pd.DataFrame,
) -> tuple[str, ...]:
    return tuple(
        name
        for name in CURRENT_SECOND_FEATURES
        if name in frame
        and _numeric(frame[name]).notna().any()
    )


def _safe_spearman(
    actual: pd.Series,
    predicted: pd.Series,
) -> float | None:
    actual = _numeric(actual)
    predicted = _numeric(predicted)
    valid = actual.notna() & predicted.notna()
    if int(valid.sum()) < 20:
        return None
    value = actual.loc[valid].corr(
        predicted.loc[valid],
        method="spearman",
    )
    return None if pd.isna(value) else float(value)


def representation_metrics(
    rows: pd.DataFrame,
    prefix: str,
) -> dict:
    work = rows.loc[
        rows.selected_prehot.fillna(False).astype(bool)
        & rows.resolved.astype(bool)
        & _numeric(rows.trade_return_pct).notna()
    ].copy()
    actual = _numeric(work.trade_return_pct)
    positive = actual.gt(0).astype(int)
    severe = actual.le(-2.0).astype(int)

    def auc(target: pd.Series, column: str) -> float | None:
        if len(work) < 20 or target.nunique() != 2:
            return None
        return float(
            roc_auc_score(
                target,
                _numeric(work[column]),
            )
        )

    return {
        "resolved_shortlist_rows": int(len(work)),
        "actual_trade_mean_pct": (
            float(actual.mean()) if len(actual) else None
        ),
        "actual_positive_rate": (
            float(positive.mean()) if len(positive) else None
        ),
        "actual_severe_rate_le_minus2": (
            float(severe.mean()) if len(severe) else None
        ),
        "value_spearman": _safe_spearman(
            actual,
            work[f"{prefix}_predicted_return_pct"],
        ),
        "positive_auc": auc(
            positive,
            f"{prefix}_positive_probability",
        ),
        "severe_auc": auc(
            severe,
            f"{prefix}_severe_probability",
        ),
    }


def metric_gains(
    baseline: dict,
    rich: dict,
) -> dict:
    out = {}
    for name in (
        "value_spearman",
        "positive_auc",
        "severe_auc",
    ):
        left = baseline.get(name)
        right = rich.get(name)
        out[name] = (
            float(right - left)
            if left is not None and right is not None
            else None
        )
    return out


def evaluate(
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    raw_scan = pd.read_parquet(scan_path)
    raw_scan = raw_scan.loc[
        raw_scan.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    causal_scan = causal_execution_scan(raw_scan)

    first_hot = pd.read_parquet(first_hot_path).drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    first_hot = attach_fixed_value(first_hot, causal_scan)

    policy = apply_exit_rule(
        build_pullback_episodes(first_hot, causal_scan),
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name="request286_frozen_h0",
    )
    labeled = attach_downstream_target(first_hot, policy)
    candidates, ticker_extra = attach_ticker_history(
        labeled,
        raw_scan,
    )
    prehot_columns = tuple(
        dict.fromkeys([
            *candidate_columns(candidates),
            *ticker_extra,
        ])
    )
    states, baseline_columns = prepare_pullback_states(
        candidates,
        ticker_extra,
        raw_scan,
        causal_scan,
        policy,
    )

    states, second_audit = attach_second_features(states)
    second_columns = usable_second_columns(states)
    rich_columns = tuple(
        dict.fromkeys([
            *baseline_columns,
            *second_columns,
        ])
    )

    if second_columns:
        coverage = float(
            states.loc[:, second_columns].notna().any(axis=1).mean()
        )
    else:
        coverage = 0.0

    parts = []
    fold_support = {}
    for fold_index, held_day in enumerate(CROSSFIT_DAYS):
        train = states.loc[
            ~states.trading_day.astype(str).eq(held_day)
        ].copy()
        held = states.loc[
            states.trading_day.astype(str).eq(held_day)
        ].copy()

        train_candidates = candidates.loc[
            ~candidates.trading_day.astype(str).eq(held_day)
        ].copy()
        held_candidates = candidates.loc[
            candidates.trading_day.astype(str).eq(held_day)
        ].copy()
        prehot, _ = fold_stage1(
            train_candidates,
            held_candidates,
            prehot_columns,
            20262000 + fold_index * 10,
        )
        held = held.merge(
            prehot.loc[:, [
                "trading_day",
                "ticker",
                "hot_t",
                "selected_prehot",
            ]],
            on=["trading_day", "ticker", "hot_t"],
            how="left",
            validate="one_to_one",
        )

        base_reg, base_pos, base_sev = fit_triplet(
            train,
            baseline_columns,
            20262100 + fold_index * 10,
        )
        rich_reg, rich_pos, rich_sev = fit_triplet(
            train,
            rich_columns,
            20262200 + fold_index * 10,
        )

        base_x = feature_x(held, baseline_columns)
        rich_x = feature_x(held, rich_columns)
        held["base_predicted_return_pct"] = base_reg.predict(base_x)
        held["base_positive_probability"] = (
            base_pos.predict_proba(base_x)[:, 1]
        )
        held["base_severe_probability"] = (
            base_sev.predict_proba(base_x)[:, 1]
        )
        held["rich_predicted_return_pct"] = rich_reg.predict(rich_x)
        held["rich_positive_probability"] = (
            rich_pos.predict_proba(rich_x)[:, 1]
        )
        held["rich_severe_probability"] = (
            rich_sev.predict_proba(rich_x)[:, 1]
        )
        held["fold_holdout_day"] = held_day
        parts.append(held)

        fold_support[str(held_day)] = {
            "train_pullback_states": int(len(train)),
            "held_pullback_states": int(len(held)),
            "held_shortlisted_pullbacks": int(
                held.selected_prehot.fillna(False).astype(bool).sum()
            ),
        }

    oof = pd.concat(parts, ignore_index=True)
    baseline = representation_metrics(oof, "base")
    rich = representation_metrics(oof, "rich")
    gains = metric_gains(baseline, rich)
    gained = sum(
        value is not None and value >= MIN_METRIC_GAIN
        for value in gains.values()
    )

    checks = {
        "conditioned_support": (
            rich["resolved_shortlist_rows"]
            >= MIN_CONDITIONED_RESOLVED
        ),
        "second_feature_coverage": coverage >= MIN_SECOND_COVERAGE,
        "rich_value_spearman": (
            rich["value_spearman"] is not None
            and rich["value_spearman"] >= MIN_RICH_VALUE_SPEARMAN
        ),
        "rich_positive_auc": (
            rich["positive_auc"] is not None
            and rich["positive_auc"] >= MIN_RICH_POSITIVE_AUC
        ),
        "rich_severe_auc": (
            rich["severe_auc"] is not None
            and rich["severe_auc"] >= MIN_RICH_SEVERE_AUC
        ),
        "multiple_metric_gains": gained >= MIN_GAINED_METRICS,
    }
    gate = bool(all(checks.values()))

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "fixed_policy": {
            "entry": "original first causal 2% pullback",
            "hard_stop_loss_pct": FIXED_STOP_LOSS_PCT,
            "trailing_drawdown_pct": FIXED_TRAIL_PCT,
            "max_hold_minutes": FIXED_MAX_HOLD_MINUTES,
            "missing_exact_deadline": "unresolved",
        },
        "feature_counts": {
            "baseline": len(baseline_columns),
            "completed_second_extra": len(second_columns),
            "rich_total": len(rich_columns),
        },
        "second_feature_coverage": coverage,
        "second_data_audit": second_audit,
        "fold_support": fold_support,
        "baseline_conditioned": baseline,
        "second_enriched_conditioned": rich,
        "second_minus_baseline": gains,
        "metrics_improved_by_at_least_0_05": int(gained),
        "checks": checks,
        "second_information_gate_pass": gate,
        "timing_contract": (
            "second features consume only completed one-second aggregates "
            "from the trailing 60 seconds ending at decision_t - 1 second; "
            "entry time and downstream outcome are unchanged"
        ),
        "next_boundary": (
            "build a sub-minute causal ENTER/WAIT recheck because second "
            "context restores enough information at the original pullback"
            if gate
            else (
                "completed pre-decision seconds are insufficient; next test "
                "fixed 5/15/30-second WAIT windows before any recurrent policy"
            )
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    oof.to_parquet(rows_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
