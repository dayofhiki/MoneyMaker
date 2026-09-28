"""Request264: rich-state diagnostic under corrected causal minute targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .causal_minute_controller_rebuild import causal_execution_scan
from .full_hot_fixed_policy_value import attach_fixed_value
from .learned_pullback_entry import (
    CONTROLLER_TRAIN_DAYS,
    TEST_DAYS,
    build_episode_states,
    controller_columns,
)
from .rich_post_hot_state import (
    CURRENT_MINUTE_FEATURES,
    CURRENT_SECOND_FEATURES,
    DELTA_MINUTE_FEATURES,
    DELTA_SECOND_FEATURES,
    MIN_POSITIVE_AUC_GAIN,
    MIN_RICH_POSITIVE_AUC,
    MIN_RICH_SECOND_COVERAGE,
    MIN_RICH_SEVERE_AUC,
    MIN_RICH_VALUE_SPEARMAN,
    MIN_TEST_PULLBACK_ROWS,
    MIN_TRAIN_PULLBACK_ROWS,
    MIN_VALUE_SPEARMAN_GAIN,
    attach_dynamic_deltas,
    attach_minute_features,
    attach_second_features,
    fit_representation,
    pullback_rows,
    representation_report,
    score_and_select_candidates,
    usable,
)

REQUEST_ID = 264


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
        scored.trading_day.astype(str).isin(CONTROLLER_TRAIN_DAYS)
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
    combined, second_audit = attach_second_features(combined)
    combined = attach_dynamic_deltas(combined)
    train_states = combined.loc[
        combined._split.eq("train")
    ].drop(columns="_split")
    test_states = combined.loc[
        combined._split.eq("test")
    ].drop(columns="_split")

    train_pullback = pullback_rows(train_states)
    test_pullback = pullback_rows(test_states)

    baseline_columns = controller_columns(train_pullback)
    minute_columns = usable(
        train_pullback,
        [
            *baseline_columns,
            *CURRENT_MINUTE_FEATURES,
            *DELTA_MINUTE_FEATURES,
        ],
    )
    rich_columns = usable(
        train_pullback,
        [
            *minute_columns,
            *CURRENT_SECOND_FEATURES,
            *DELTA_SECOND_FEATURES,
        ],
    )

    baseline = fit_representation(
        train_pullback,
        baseline_columns,
        20261381,
    )
    minute = fit_representation(
        train_pullback,
        minute_columns,
        20261384,
    )
    rich = fit_representation(
        train_pullback,
        rich_columns,
        20261387,
    )
    reports = {
        "baseline": representation_report(
            test_pullback,
            baseline,
        ),
        "minute": representation_report(
            test_pullback,
            minute,
        ),
        "rich": representation_report(
            test_pullback,
            rich,
        ),
    }

    baseline_report = reports["baseline"]
    rich_report = reports["rich"]
    value_gain = (
        rich_report["value_spearman"]
        - baseline_report["value_spearman"]
        if rich_report["value_spearman"] is not None
        and baseline_report["value_spearman"] is not None
        else None
    )
    positive_gain = (
        rich_report["positive_auc"]
        - baseline_report["positive_auc"]
        if rich_report["positive_auc"] is not None
        and baseline_report["positive_auc"] is not None
        else None
    )

    checks = {
        "train_pullback_rows": (
            len(train_pullback) >= MIN_TRAIN_PULLBACK_ROWS
        ),
        "test_pullback_rows": (
            len(test_pullback) >= MIN_TEST_PULLBACK_ROWS
        ),
        "rich_second_coverage": (
            second_audit["any_second_coverage"]
            >= MIN_RICH_SECOND_COVERAGE
        ),
        "rich_value_spearman": (
            rich_report["value_spearman"] is not None
            and rich_report["value_spearman"]
            >= MIN_RICH_VALUE_SPEARMAN
        ),
        "value_spearman_gain": (
            value_gain is not None
            and value_gain >= MIN_VALUE_SPEARMAN_GAIN
        ),
        "rich_positive_auc": (
            rich_report["positive_auc"] is not None
            and rich_report["positive_auc"]
            >= MIN_RICH_POSITIVE_AUC
        ),
        "positive_auc_gain": (
            positive_gain is not None
            and positive_gain >= MIN_POSITIVE_AUC_GAIN
        ),
        "rich_severe_auc": (
            rich_report["severe_auc"] is not None
            and rich_report["severe_auc"]
            >= MIN_RICH_SEVERE_AUC
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "causal_reference_contract": True,
        "train_candidate_threshold": train_threshold,
        "test_candidate_threshold": test_threshold,
        "support": {
            "train_candidate_episodes": int(len(train_selected)),
            "test_candidate_episodes": int(len(test_selected)),
            "train_states": int(len(train_states)),
            "test_states": int(len(test_states)),
            "train_pullback_rows": int(len(train_pullback)),
            "test_pullback_rows": int(len(test_pullback)),
        },
        "second_data_audit": second_audit,
        "representations": reports,
        "rich_minus_baseline": {
            "value_spearman": value_gain,
            "positive_auc": positive_gain,
            "severe_auc": (
                rich_report["severe_auc"]
                - baseline_report["severe_auc"]
                if rich_report["severe_auc"] is not None
                and baseline_report["severe_auc"] is not None
                else None
            ),
        },
        "checks": checks,
        "representation_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "Targets and path prices use causal decision-time minute opens. "
            "Second features consume completed seconds only. May11-20 remains "
            "development-only."
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
