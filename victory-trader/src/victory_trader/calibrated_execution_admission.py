"""Request261: May8-calibrated execution admission for Request258."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .execution_aware_minute_controller import (
    attach_causal_second_features,
    execution_columns,
    fit_execution_model,
    label_execution_support,
    load_second_cache,
)
from .frozen_minute_policy_second_audit import (
    PRIMARY_LATENCY_MS,
    reconstruct,
    report as second_report,
)
from .full_hot_fixed_policy_value import attach_fixed_value, event_lookup
from .learned_pullback_entry import (
    baseline_policy,
    build_episode_states,
    controller_x,
    policy_metrics,
)
from .minute_state_entry_controller import (
    MODEL_CALIBRATION_DAY,
    MODEL_FIT_DAYS,
    POSITIVE_PROBABILITY_THRESHOLD,
    SEVERE_PROBABILITY_THRESHOLD,
    TEST_DAYS,
    fit_models,
    minute_state_columns,
    pullback_mask,
    pullback_rows,
)
from .rich_post_hot_state import (
    attach_dynamic_deltas,
    attach_minute_features,
    score_and_select_candidates,
)

REQUEST_ID = 261
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
MIN_CAL_ADMITTED_ROWS = 15
MIN_CAL_PRECISION = 0.55
MIN_CAL_RECALL = 0.25
MIN_CAL_ADMITTED_RATE = 0.05
MAX_CAL_ADMITTED_RATE = 0.40
MIN_TEST_AUC = 0.65
MIN_ENTRIES = 10
MIN_MINUTE_GAIN = 0.10
MIN_SECOND_ENTRY_COVERAGE = 0.60
MIN_SECOND_RESOLUTION = 0.30
MAX_SECOND_SEVERE = 0.15


def threshold_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    y = frame.execution_supported.astype(int).to_numpy()
    admitted = probability >= float(threshold)
    admitted_rows = int(np.sum(admitted))
    positives = int(np.sum(y))
    precision = (
        float(np.mean(y[admitted]))
        if admitted_rows
        else None
    )
    recall = (
        float(np.sum(y[admitted]) / positives)
        if positives
        else None
    )
    admitted_rate = (
        float(np.mean(admitted))
        if len(admitted)
        else 0.0
    )
    checks = {
        "min_admitted_rows": admitted_rows >= MIN_CAL_ADMITTED_ROWS,
        "precision": precision is not None and precision >= MIN_CAL_PRECISION,
        "recall": recall is not None and recall >= MIN_CAL_RECALL,
        "admitted_rate": (
            MIN_CAL_ADMITTED_RATE
            <= admitted_rate
            <= MAX_CAL_ADMITTED_RATE
        ),
    }
    return {
        "threshold": float(threshold),
        "rows": int(len(frame)),
        "positive_rate": float(np.mean(y)) if len(y) else None,
        "admitted_rows": admitted_rows,
        "admitted_rate": admitted_rate,
        "precision": precision,
        "recall": recall,
        "checks": checks,
        "passes": bool(all(checks.values())),
    }


def choose_threshold(
    calibration: pd.DataFrame,
    model,
    columns: tuple[str, ...],
) -> tuple[float | None, list[dict]]:
    probability = model.predict_proba(
        controller_x(calibration, columns)
    )[:, 1]
    rows = [
        threshold_metrics(calibration, probability, threshold)
        for threshold in THRESHOLDS
    ]
    passing = [row for row in rows if row["passes"]]
    if not passing:
        return None, rows
    passing.sort(
        key=lambda row: (
            -(row["recall"] or 0.0),
            -(row["precision"] or 0.0),
            -float(row["threshold"]),
        )
    )
    return float(passing[0]["threshold"]), rows


def test_signal(
    frame: pd.DataFrame,
    model,
    columns: tuple[str, ...],
    threshold: float,
) -> dict:
    y = frame.execution_supported.astype(int)
    probability = model.predict_proba(
        controller_x(frame, columns)
    )[:, 1]
    auc = (
        float(roc_auc_score(y, probability))
        if len(frame) >= 20 and y.nunique() == 2
        else None
    )
    row = threshold_metrics(frame, probability, threshold)
    row["auc"] = auc
    return row


def make_overlay_policy(
    states: pd.DataFrame,
    selected: pd.DataFrame,
    economic,
    execution_model,
    execution_columns_: tuple[str, ...],
    threshold: float,
) -> pd.DataFrame:
    work = states.copy()
    econ_x = controller_x(work, economic.columns)
    work["predicted_value_pct"] = (
        economic.value.predict(econ_x) + economic.value_offset
    )
    work["positive_probability"] = (
        economic.positive.predict_proba(econ_x)[:, 1]
    )
    work["severe_probability"] = (
        economic.severe.predict_proba(econ_x)[:, 1]
    )
    work["execution_probability"] = execution_model.predict_proba(
        controller_x(work, execution_columns_)
    )[:, 1]
    groups = {
        (str(k[0]), str(k[1]).upper(), int(k[2])): g.sort_values("state_t")
        for k, g in work.groupby(
            ["trading_day", "ticker", "hot_t"],
            sort=False,
        )
    }
    records = []
    for raw in selected.itertuples(index=False):
        key = (
            str(raw.trading_day),
            str(raw.ticker).upper(),
            int(raw.t),
        )
        group = groups.get(key)
        chosen = None
        if group is not None:
            for _, state in group.loc[pullback_mask(group)].iterrows():
                if (
                    float(state.predicted_value_pct) > 0
                    and float(state.positive_probability)
                    >= POSITIVE_PROBABILITY_THRESHOLD
                    and float(state.severe_probability)
                    < SEVERE_PROBABILITY_THRESHOLD
                    and float(state.execution_probability)
                    >= float(threshold)
                ):
                    chosen = state
                    break
        records.append(
            {
                "trading_day": key[0],
                "ticker": key[1],
                "hot_t": key[2],
                "entered": chosen is not None,
                "entry_t": (
                    int(chosen.state_t) if chosen is not None else None
                ),
                "entry_price": (
                    float(chosen.state_price)
                    if chosen is not None
                    else None
                ),
                "value_pct": (
                    float(chosen.enter_value_pct)
                    if chosen is not None
                    and pd.notna(chosen.enter_value_pct)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, scan)
    scored, train_threshold, test_threshold = score_and_select_candidates(
        first_hot
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

    train_states = build_episode_states(train_selected, scan)
    test_states = build_episode_states(test_selected, scan)
    combined = pd.concat(
        [
            train_states.assign(_split="train"),
            test_states.assign(_split="test"),
        ],
        ignore_index=True,
    )
    combined = attach_minute_features(combined, scan)
    frames, paths, api_stats = load_second_cache(combined)
    combined = attach_causal_second_features(combined, frames)
    combined = attach_dynamic_deltas(combined)
    combined = label_execution_support(combined, paths)
    train_states = combined.loc[combined._split.eq("train")].drop(
        columns="_split"
    ).copy()
    test_states = combined.loc[combined._split.eq("test")].drop(
        columns="_split"
    ).copy()

    fit_exec = train_states.loc[
        train_states.trading_day.astype(str).isin(MODEL_FIT_DAYS)
    ].copy()
    cal_exec = train_states.loc[
        train_states.trading_day.astype(str).eq(MODEL_CALIBRATION_DAY)
    ].copy()
    exec_columns = execution_columns(fit_exec)
    exec_model, exec_support = fit_execution_model(
        fit_exec,
        exec_columns,
    )
    chosen_threshold, calibration_table = choose_threshold(
        cal_exec,
        exec_model,
        exec_columns,
    )

    train_pullback = pullback_rows(train_states)
    test_pullback = pullback_rows(test_states)
    fit_pullback = train_pullback.loc[
        train_pullback.trading_day.astype(str).isin(MODEL_FIT_DAYS)
    ].copy()
    cal_pullback = train_pullback.loc[
        train_pullback.trading_day.astype(str).eq(MODEL_CALIBRATION_DAY)
    ].copy()
    econ_columns = minute_state_columns(fit_pullback)
    economic, econ_support = fit_models(
        fit_pullback,
        cal_pullback,
        econ_columns,
    )

    if chosen_threshold is None:
        result = {
            "request_id": REQUEST_ID,
            "development_only": True,
            "opens_new_dates": False,
            "promotion_eligible": False,
            "threshold_calibration_pass": False,
            "threshold_calibration": calibration_table,
            "execution_model_support": exec_support,
            "api_stats": api_stats,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0

    signal = test_signal(
        test_pullback,
        exec_model,
        exec_columns,
        chosen_threshold,
    )
    overlay = make_overlay_policy(
        test_states,
        test_selected,
        economic,
        exec_model,
        exec_columns,
        chosen_threshold,
    )
    fixed = baseline_policy(test_selected, scan, "pullback_2")
    candidate_count = int(len(test_selected))
    overlay_minute = policy_metrics(overlay, candidate_count)
    fixed_minute = policy_metrics(fixed, candidate_count)
    minute_gain = (
        overlay_minute["mean_pct"] - fixed_minute["mean_pct"]
        if overlay_minute["mean_pct"] is not None
        and fixed_minute["mean_pct"] is not None
        else None
    )

    second_rows = reconstruct(
        overlay,
        event_lookup(scan),
        paths,
        latency_ms=PRIMARY_LATENCY_MS,
    )
    second = second_report(second_rows)

    checks = {
        "threshold_calibration": True,
        "test_execution_auc": (
            signal["auc"] is not None
            and signal["auc"] >= MIN_TEST_AUC
        ),
        "entries": overlay_minute["entries"] >= MIN_ENTRIES,
        "minute_mean_positive": (
            overlay_minute["mean_pct"] is not None
            and overlay_minute["mean_pct"] > 0
        ),
        "minute_gain_vs_fixed": (
            minute_gain is not None
            and minute_gain >= MIN_MINUTE_GAIN
        ),
        "second_entry_coverage": (
            second["entry_reference_coverage"]
            >= MIN_SECOND_ENTRY_COVERAGE
        ),
        "second_full_resolution": (
            second["full_trade_resolution"]
            >= MIN_SECOND_RESOLUTION
        ),
        "second_resolved_mean_positive": (
            second["resolved_mean_base_pct"] is not None
            and second["resolved_mean_base_pct"] > 0
        ),
        "second_severe_rate": (
            second["resolved_severe_loss_rate_le_minus2"] is not None
            and second["resolved_severe_loss_rate_le_minus2"]
            <= MAX_SECOND_SEVERE
        ),
    }
    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "chosen_execution_threshold": chosen_threshold,
        "threshold_calibration_pass": True,
        "threshold_calibration": calibration_table,
        "execution_model_support": exec_support,
        "economic_model_support": econ_support,
        "test_execution_signal": signal,
        "minute_policy": overlay_minute,
        "fixed_2pct_pullback": fixed_minute,
        "minute_gain_vs_fixed_2pct_pullback_pct": minute_gain,
        "second_reference": second,
        "checks": checks,
        "joint_gate_pass": bool(all(checks.values())),
        "api_stats": api_stats,
        "interpretation": (
            "Threshold selection uses May8 only. May11-20 remains development "
            "evidence. Historical one-second aggregates are execution references, "
            "not brokerage fills."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    second_rows.to_parquet(rows_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--first-hot", type=Path, required=True)
    p.add_argument("--opportunity-scan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows-output", type=Path, required=True)
    a = p.parse_args()
    return evaluate(
        a.first_hot,
        a.opportunity_scan,
        a.output,
        a.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
