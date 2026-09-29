"""Request271: recurrent one-step causal HOLD/EXIT controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .causal_continuation_horizon import (
    MODEL_FIT_DAYS,
    attach_continuation_targets,
    fit_horizon_model,
)
from .causal_hold_exit_controller import (
    _base_record,
    policy_metrics,
    static_policy_rows,
)
from .causal_hold_exit_signal import build_position_states, path_columns
from .causal_minute_controller_rebuild import causal_execution_scan
from .full_hot_fixed_policy_value import attach_fixed_value
from .learned_pullback_entry import CONTROLLER_TRAIN_DAYS, TEST_DAYS, controller_x
from .rich_post_hot_state import score_and_select_candidates

REQUEST_ID = 271
HORIZON = 1
HOLD_PROBABILITY_THRESHOLD = 0.50
MIN_TEST_EPISODES = 80
MIN_POSITIVE_RATE = 0.40
MAX_SEVERE_RATE = 0.15
MIN_POSITIVE_MEAN_DAYS = 5
MIN_GAIN_VS_BEST_STATIC = 0.10


def dynamic_policy_rows(
    states: pd.DataFrame,
    reg,
    cls,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    work = states.copy()
    x = controller_x(work, columns)
    work["predicted_continuation_advantage_pct"] = reg.predict(x)
    work["continuation_probability"] = cls.predict_proba(x)[:, 1]
    rows = []
    for _, group in work.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        ordered = group.sort_values("state_t").reset_index(drop=True)
        if ordered.empty:
            continue
        chosen = ordered.iloc[-1]
        exit_reason = "terminal_cap"
        for position, (_, state) in enumerate(ordered.iterrows()):
            is_terminal = position == len(ordered) - 1
            if is_terminal:
                chosen = state
                exit_reason = "terminal_cap"
                break
            hold = (
                float(state.predicted_continuation_advantage_pct) > 0.0
                and float(state.continuation_probability)
                >= HOLD_PROBABILITY_THRESHOLD
            )
            if not hold:
                chosen = state
                exit_reason = "model_exit"
                break
        record = _base_record(ordered, chosen, "dynamic_one_step")
        record["exit_reason"] = exit_reason
        record["predicted_continuation_advantage_pct_at_exit"] = float(
            chosen.predicted_continuation_advantage_pct
        )
        record["continuation_probability_at_exit"] = float(
            chosen.continuation_probability
        )
        rows.append(record)
    return pd.DataFrame(rows)


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    causal_scan = causal_execution_scan(scan)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, causal_scan)
    scored, train_threshold, test_threshold = score_and_select_candidates(
        first_hot
    )

    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(CONTROLLER_TRAIN_DAYS)
        & scored.candidate_probability.ge(train_threshold)
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_probability.ge(test_threshold)
    ].copy()

    train_states = attach_continuation_targets(
        build_position_states(train_selected, causal_scan)
    )
    test_states = attach_continuation_targets(
        build_position_states(test_selected, causal_scan)
    )
    fit_states = train_states.loc[
        train_states.trading_day.astype(str).isin(MODEL_FIT_DAYS)
    ].copy()
    columns = path_columns(fit_states)
    reg, cls, fit_support = fit_horizon_model(
        fit_states,
        columns,
        HORIZON,
    )

    dynamic_rows = dynamic_policy_rows(
        test_states,
        reg,
        cls,
        columns,
    )
    static_rows = {
        mode: static_policy_rows(test_states, mode)
        for mode in ("first", "fifth", "terminal")
    }
    dynamic = policy_metrics(dynamic_rows)
    statics = {
        mode: policy_metrics(rows)
        for mode, rows in static_rows.items()
    }
    best_static_mean = max(
        report["mean_pct"]
        for report in statics.values()
        if report["mean_pct"] is not None
    )
    gain_vs_best_static = dynamic["mean_pct"] - best_static_mean

    checks = {
        "test_position_episodes": dynamic["episodes"] >= MIN_TEST_EPISODES,
        "dynamic_mean_positive": dynamic["mean_pct"] > 0,
        "day_balanced_mean_positive": (
            dynamic["day_balanced_mean_pct"] > 0
        ),
        "positive_rate": (
            dynamic["positive_rate"] >= MIN_POSITIVE_RATE
        ),
        "severe_loss_rate": (
            dynamic["severe_loss_rate_le_minus2"]
            <= MAX_SEVERE_RATE
        ),
        "positive_mean_days": (
            dynamic["positive_mean_days"] >= MIN_POSITIVE_MEAN_DAYS
        ),
        "gain_vs_best_static": (
            gain_vs_best_static >= MIN_GAIN_VS_BEST_STATIC
        ),
    }

    all_rows = [dynamic_rows]
    for mode, rows in static_rows.items():
        tagged = rows.copy()
        tagged["policy"] = mode
        all_rows.append(tagged)

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "causal_reference_contract": True,
        "train_candidate_threshold_top20": train_threshold,
        "test_candidate_threshold_top5": test_threshold,
        "horizon_states": HORIZON,
        "hold_probability_threshold": HOLD_PROBABILITY_THRESHOLD,
        "support": {
            "train_candidate_episodes": int(len(train_selected)),
            "test_candidate_episodes": int(len(test_selected)),
            "fit_position_states": int(len(fit_states)),
            "test_position_states": int(len(test_states)),
            "feature_count": len(columns),
            "fit_target": fit_support,
        },
        "policies": {
            "dynamic_one_step_hold_exit": dynamic,
            "static_first_exit": statics["first"],
            "static_fifth_exit": statics["fifth"],
            "static_terminal_exit": statics["terminal"],
        },
        "best_static_mean_pct": best_static_mean,
        "mean_gain_vs_best_static_pct": gain_vs_best_static,
        "checks": checks,
        "economic_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "The one-state continuation horizon was selected only on May8 in "
            "Request270. Request271 applies it recurrently to top-5% candidates "
            "with fixed 0/0.50 thresholds. All prices are causal."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.concat(all_rows, ignore_index=True).to_parquet(
        rows_output,
        index=False,
    )
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
