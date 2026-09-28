"""Request263: rebuild Request258 using causal minute-open references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .full_hot_fixed_policy_value import attach_fixed_value
from .learned_pullback_entry import (
    baseline_policy,
    build_episode_states,
    policy_metrics,
)
from .minute_state_entry_controller import (
    MAX_SEVERE_RATE,
    MIN_CAL_PULLBACK_ROWS,
    MIN_ENTRIES,
    MIN_ENTRY_RATE,
    MIN_FIT_PULLBACK_ROWS,
    MIN_GAIN_VS_FIXED_PULLBACK,
    MIN_POSITIVE_MEAN_DAYS,
    MIN_RESOLUTION_OR_CASH,
    MIN_TEST_PULLBACK_ROWS,
    MAX_ENTRY_RATE,
    MODEL_CALIBRATION_DAY,
    MODEL_FIT_DAYS,
    TEST_DAYS,
    fit_models,
    make_policy,
    minute_state_columns,
    signal_report,
)
from .rich_post_hot_state import (
    TEST_CANDIDATE_FRACTION,
    TRAIN_CANDIDATE_FRACTION,
    attach_dynamic_deltas,
    attach_minute_features,
    pullback_rows,
    score_and_select_candidates,
)

REQUEST_ID = 263
MINUTE_MS = 60_000


def causal_execution_scan(scan: pd.DataFrame) -> pd.DataFrame:
    required = {"trading_day", "ticker", "bar_start_t", "o"}
    missing = required - set(scan.columns)
    if missing:
        raise ValueError(
            f"Request263 scan missing columns: {sorted(missing)}"
        )
    result = scan.loc[
        :, ["trading_day", "ticker", "bar_start_t", "o"]
    ].copy()
    result["bar_start_t"] = pd.to_numeric(
        result["bar_start_t"],
        errors="coerce",
    )
    result["o"] = pd.to_numeric(result["o"], errors="coerce")
    result = result.loc[
        result.bar_start_t.notna() & result.o.gt(0)
    ].copy()
    result["t"] = result.bar_start_t.astype("int64")
    result = result.drop(columns="bar_start_t")
    duplicate = result.duplicated(
        ["trading_day", "ticker", "t"],
        keep=False,
    )
    if duplicate.any():
        raise ValueError("Request263 causal execution scan has duplicates")
    return result.sort_values(
        ["trading_day", "ticker", "t"],
        kind="stable",
    ).reset_index(drop=True)


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
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

    scored, train_threshold, test_threshold = (
        score_and_select_candidates(first_hot)
    )
    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(
            [*MODEL_FIT_DAYS, MODEL_CALIBRATION_DAY]
        )
        & scored.candidate_probability.ge(train_threshold)
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_probability.ge(test_threshold)
    ].copy()

    train_states = build_episode_states(train_selected, causal_scan)
    test_states = build_episode_states(test_selected, causal_scan)
    combined = pd.concat(
        [
            train_states.assign(_split="train"),
            test_states.assign(_split="test"),
        ],
        ignore_index=True,
    )
    combined = attach_minute_features(combined, scan)
    combined = attach_dynamic_deltas(combined)
    train_states = combined.loc[
        combined._split.eq("train")
    ].drop(columns="_split")
    test_states = combined.loc[
        combined._split.eq("test")
    ].drop(columns="_split")

    train_pullback = pullback_rows(train_states)
    test_pullback = pullback_rows(test_states)
    fit_pullback = train_pullback.loc[
        train_pullback.trading_day.astype(str).isin(MODEL_FIT_DAYS)
    ].copy()
    cal_pullback = train_pullback.loc[
        train_pullback.trading_day.astype(str).eq(
            MODEL_CALIBRATION_DAY
        )
    ].copy()

    columns = minute_state_columns(fit_pullback)
    fitted, model_support = fit_models(
        fit_pullback,
        cal_pullback,
        columns,
    )
    signal = signal_report(test_pullback, fitted)

    policy_rows = make_policy(
        test_states,
        test_selected,
        fitted,
    )
    immediate_rows = baseline_policy(
        test_selected,
        causal_scan,
        "immediate",
    )
    fixed_rows = baseline_policy(
        test_selected,
        causal_scan,
        "pullback_2",
    )

    candidate_count = int(len(test_selected))
    policy = policy_metrics(policy_rows, candidate_count)
    immediate = policy_metrics(immediate_rows, candidate_count)
    fixed = policy_metrics(fixed_rows, candidate_count)

    policy_mean = policy["mean_pct"]
    fixed_mean = fixed["mean_pct"]
    gain = (
        policy_mean - fixed_mean
        if policy_mean is not None and fixed_mean is not None
        else None
    )

    checks = {
        "fit_pullback_rows": (
            len(fit_pullback) >= MIN_FIT_PULLBACK_ROWS
        ),
        "calibration_pullback_rows": (
            len(cal_pullback) >= MIN_CAL_PULLBACK_ROWS
        ),
        "test_pullback_rows": (
            len(test_pullback) >= MIN_TEST_PULLBACK_ROWS
        ),
        "entries": policy["entries"] >= MIN_ENTRIES,
        "entry_rate": (
            MIN_ENTRY_RATE
            <= policy["entry_rate"]
            <= MAX_ENTRY_RATE
        ),
        "resolution_or_cash_rate": (
            policy["resolution_or_cash_rate"]
            >= MIN_RESOLUTION_OR_CASH
        ),
        "mean_positive": (
            policy_mean is not None and policy_mean > 0
        ),
        "day_balanced_mean_positive": (
            policy["day_balanced_mean_pct"] is not None
            and policy["day_balanced_mean_pct"] > 0
        ),
        "positive_mean_days": (
            policy["positive_mean_days"]
            >= MIN_POSITIVE_MEAN_DAYS
        ),
        "severe_loss_rate": (
            policy["severe_loss_rate_le_minus2"] is not None
            and policy["severe_loss_rate_le_minus2"]
            <= MAX_SEVERE_RATE
        ),
        "gain_vs_fixed_pullback": (
            gain is not None
            and gain >= MIN_GAIN_VS_FIXED_PULLBACK
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "retroactive_reference_removed": True,
        "training_candidate_fraction": TRAIN_CANDIDATE_FRACTION,
        "test_candidate_fraction": TEST_CANDIDATE_FRACTION,
        "train_candidate_threshold": train_threshold,
        "test_candidate_threshold": test_threshold,
        "support": {
            "causal_execution_rows": int(len(causal_scan)),
            "causal_fixed_target_rows": int(
                pd.to_numeric(
                    first_hot.fixed_first_watch_value_pct,
                    errors="coerce",
                ).notna().sum()
            ),
            "train_candidate_episodes": int(len(train_selected)),
            "test_candidate_episodes": candidate_count,
            "fit_pullback_rows": int(len(fit_pullback)),
            "calibration_pullback_rows": int(len(cal_pullback)),
            "test_pullback_rows": int(len(test_pullback)),
        },
        "model_support": model_support,
        "development_signal": signal,
        "development_policies": {
            "immediate": immediate,
            "fixed_2pct_pullback": fixed,
            "causal_minute_state_controller": policy,
        },
        "mean_gain_vs_fixed_2pct_pullback_pct": gain,
        "checks": checks,
        "economic_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "Every state price and economic target is now tied to the open "
            "of a minute beginning at the decision timestamp, never the open "
            "of the already-completed minute. May11-20 is development-only."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
