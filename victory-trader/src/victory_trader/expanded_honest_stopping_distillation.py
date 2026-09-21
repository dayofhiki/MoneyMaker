from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .expanded_fitted_optimal_stopping import (
    BOOTSTRAP_SAMPLES,
    MAX_HOLD_MINUTES,
    MinuteStoppingModel,
    _attach_exit_now_base,
    _bootstrap_daily,
    _first_later_base_return,
    _forced_30m_base_return,
    _month_of,
    _policy_key,
    _state_groups_by_month,
    build_optimal_trajectories,
    matched_difference,
    score_stopping_rows,
    summarize_policy,
    train_fitted_stopping,
)
from .expanded_opportunity_shared_q import (
    predict_opportunity,
    train_opportunity_model,
)
from .expanded_path_aware_continuation import build_continuation_rows
from .expanded_path_transition_continuation import (
    predict_transition_continuation,
    train_transition_model,
    transition_feature_frame,
)
from .expanded_recurrent_path_policy import (
    account_replay,
    build_always_hold_comparator,
    build_recurrent_trajectories,
)
from .expanded_supply_hurdle_ev import load_fold

BOOTSTRAP_SEED = 20261047


def build_honest_teacher_targets(
    calibration_rows: pd.DataFrame,
    teacher_models: dict[int, MinuteStoppingModel],
    states: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fit-only teacher continuation honestly on later calibration."""
    frame = score_stopping_rows(
        calibration_rows,
        teacher_models,
    ).reset_index(drop=True)
    groups = _state_groups_by_month(states)
    held = pd.to_numeric(frame["minutes_held"], errors="coerce")

    row_by_key: dict[tuple[str, str, int], int] = {}
    for idx, row in frame.iterrows():
        minute = held.iloc[idx]
        if pd.notna(minute) and float(minute).is_integer():
            row_by_key[_policy_key(row, int(minute))] = int(idx)

    teacher_value = pd.Series(np.nan, index=frame.index, dtype=float)
    hold_value = pd.Series(np.nan, index=frame.index, dtype=float)
    honest_target = pd.Series(np.nan, index=frame.index, dtype=float)
    diagnostics: list[dict[str, object]] = []

    for minute in range(MAX_HOLD_MINUTES - 1, 0, -1):
        minute_idx = frame.index[held.eq(minute)]
        if not len(minute_idx):
            continue

        for idx in minute_idx:
            row = frame.loc[idx]
            exit_now = pd.to_numeric(
                pd.Series([row.get("exit_now_base_return_pct")]),
                errors="coerce",
            ).iloc[0]

            if minute == MAX_HOLD_MINUTES - 1:
                downstream = _forced_30m_base_return(row, groups)
            else:
                next_idx = row_by_key.get(_policy_key(row, minute + 1))
                if (
                    next_idx is not None
                    and pd.notna(teacher_value.loc[next_idx])
                ):
                    downstream = float(teacher_value.loc[next_idx])
                else:
                    timestamp = pd.to_numeric(
                        pd.Series([row.get("hold_reference_t")]),
                        errors="coerce",
                    ).iloc[0]
                    downstream = (
                        _first_later_base_return(
                            row,
                            groups,
                            int(timestamp),
                        )
                        if pd.notna(timestamp)
                        else np.nan
                    )

            hold_value.loc[idx] = downstream
            if pd.notna(exit_now) and pd.notna(downstream):
                honest_target.loc[idx] = float(downstream) - float(exit_now)

            predicted = pd.to_numeric(
                pd.Series(
                    [row.get("predicted_stopping_advantage_pct")]
                ),
                errors="coerce",
            ).iloc[0]

            if pd.notna(exit_now):
                if (
                    pd.notna(predicted)
                    and np.isfinite(float(predicted))
                    and float(predicted) > 0
                    and pd.notna(downstream)
                ):
                    teacher_value.loc[idx] = float(downstream)
                else:
                    teacher_value.loc[idx] = float(exit_now)
            else:
                timestamp = pd.to_numeric(
                    pd.Series([row.get("exit_reference_t")]),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(timestamp):
                    teacher_value.loc[idx] = _first_later_base_return(
                        row,
                        groups,
                        int(timestamp),
                    )

        minute_targets = pd.to_numeric(
            honest_target.loc[minute_idx],
            errors="coerce",
        ).dropna()
        minute_teacher = pd.to_numeric(
            teacher_value.loc[minute_idx],
            errors="coerce",
        ).dropna()
        minute_predictions = pd.to_numeric(
            frame.loc[
                minute_idx,
                "predicted_stopping_advantage_pct",
            ],
            errors="coerce",
        ).dropna()
        diagnostics.append(
            {
                "minute": minute,
                "candidate_rows": int(len(minute_idx)),
                "honest_target_rows": int(len(minute_targets)),
                "honest_target_mean_pct": (
                    float(minute_targets.mean())
                    if len(minute_targets)
                    else np.nan
                ),
                "teacher_prediction_mean_pct": (
                    float(minute_predictions.mean())
                    if len(minute_predictions)
                    else np.nan
                ),
                "teacher_predicted_hold_rate": (
                    float(minute_predictions.gt(0).mean())
                    if len(minute_predictions)
                    else np.nan
                ),
                "teacher_realized_value_mean_pct": (
                    float(minute_teacher.mean())
                    if len(minute_teacher)
                    else np.nan
                ),
            }
        )

    frame["teacher_realized_policy_value_pct"] = teacher_value
    frame["honest_hold_downstream_value_pct"] = hold_value
    frame["honest_hold_advantage_pct"] = honest_target
    return (
        frame,
        pd.DataFrame(diagnostics).sort_values("minute"),
    )


def train_honest_student(
    honest_rows: pd.DataFrame,
) -> tuple[dict[int, MinuteStoppingModel], pd.DataFrame]:
    features = transition_feature_frame(honest_rows)
    held = pd.to_numeric(honest_rows["minutes_held"], errors="coerce")
    models: dict[int, MinuteStoppingModel] = {}
    diagnostics: list[dict[str, object]] = []

    for minute in range(MAX_HOLD_MINUTES - 1, 0, -1):
        minute_idx = honest_rows.index[held.eq(minute)]
        target = pd.to_numeric(
            honest_rows.loc[minute_idx, "honest_hold_advantage_pct"],
            errors="coerce",
        )
        train_idx = minute_idx[target.notna().to_numpy()]
        if len(train_idx) < 150:
            raise ValueError(
                f"minute {minute} has only {len(train_idx)} honest targets"
            )

        y = pd.to_numeric(
            honest_rows.loc[
                train_idx,
                "honest_hold_advantage_pct",
            ],
            errors="coerce",
        ).astype(float)
        low, high = np.quantile(y.to_numpy(), [0.005, 0.995])
        y_fit = y.clip(lower=low, upper=high)

        x_minute = features.loc[train_idx]
        columns = tuple(
            column
            for column in x_minute
            if x_minute[column].notna().any()
        )
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=75,
            l2_regularization=2.0,
            random_state=20261047 + minute,
        )
        model.fit(x_minute.loc[:, columns], y_fit)
        fitted = MinuteStoppingModel(minute, model, columns)
        models[minute] = fitted

        prediction = model.predict(x_minute.loc[:, columns])
        diagnostics.append(
            {
                "minute": minute,
                "target_rows": int(len(train_idx)),
                "target_mean_pct": float(y.mean()),
                "winsor_low_pct": float(low),
                "winsor_high_pct": float(high),
                "prediction_mean_pct": float(np.mean(prediction)),
                "predicted_hold_rate": float(np.mean(prediction > 0)),
            }
        )

    missing = sorted(set(range(1, MAX_HOLD_MINUTES)) - set(models))
    if missing:
        raise ValueError(f"missing honest student minutes: {missing}")

    return models, pd.DataFrame(diagnostics).sort_values("minute")


def run_fold(
    anchor_paths: dict[str, Path],
    state_paths: dict[str, Path],
    evaluation_month: str,
):
    fit, calibration, evaluation, provenance = load_fold(
        anchor_paths,
        evaluation_month,
    )
    opportunity = train_opportunity_model(fit, calibration)

    partitions = []
    for name, frame in (
        ("fit", fit),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        item = frame.copy()
        item["_partition"] = name
        item["opportunity_probability"] = predict_opportunity(
            item,
            opportunity,
        )
        partitions.append(item)
    anchors = pd.concat(partitions, ignore_index=True)

    states: dict[str, pd.DataFrame] = {}
    sequence_parts = []
    for month, group in anchors.groupby(_month_of(anchors), sort=True):
        if month not in state_paths:
            raise ValueError(f"missing state panel for {month}")
        state = pd.read_parquet(state_paths[month])
        states[month] = state
        sequence_parts.append(build_continuation_rows(state, group))

    sequence = pd.concat(sequence_parts, ignore_index=True)
    fit_rows = sequence.loc[
        sequence["_partition"].eq("fit")
    ].reset_index(drop=True)
    cal_rows = sequence.loc[
        sequence["_partition"].eq("calibration")
    ].reset_index(drop=True)
    eval_rows = sequence.loc[
        sequence["_partition"].eq("evaluation")
    ].reset_index(drop=True)

    teacher_models, teacher_fit_diagnostics, _ = train_fitted_stopping(
        fit_rows,
        states,
    )
    honest_rows, teacher_honest_diagnostics = (
        build_honest_teacher_targets(
            cal_rows,
            teacher_models,
            states,
        )
    )
    student_models, student_diagnostics = train_honest_student(
        honest_rows,
    )

    eval_student_scored = score_stopping_rows(
        eval_rows,
        student_models,
    )
    eval_teacher_scored = score_stopping_rows(
        eval_rows,
        teacher_models,
    )

    eval_anchors = anchors.loc[
        anchors["_partition"].eq("evaluation")
    ].reset_index(drop=True)
    eval_state = states[evaluation_month]

    primary, decisions = build_optimal_trajectories(
        eval_anchors,
        eval_student_scored,
        eval_state,
        evaluation_month,
    )
    v40, _ = build_optimal_trajectories(
        eval_anchors,
        eval_teacher_scored,
        eval_state,
        evaluation_month,
    )

    transition = train_transition_model(fit_rows, cal_rows)
    v38_scored = eval_rows.copy()
    v38_scored["hold_probability"] = (
        predict_transition_continuation(v38_scored, transition)
    )
    v38, _ = build_recurrent_trajectories(
        eval_anchors,
        v38_scored,
        eval_state,
        evaluation_month,
    )
    hold30 = build_always_hold_comparator(
        primary,
        eval_state,
        evaluation_month,
    )

    summaries = pd.concat(
        [
            summarize_policy(
                primary,
                "honest_stopping_distillation_v41",
                evaluation_month,
            ),
            summarize_policy(v38, "recurrent_v38", evaluation_month),
            summarize_policy(
                v40,
                "fitted_optimal_stopping_v40",
                evaluation_month,
            ),
            summarize_policy(
                hold30,
                "always_hold_30m",
                evaluation_month,
            ),
        ],
        ignore_index=True,
    )

    _, diff_v38 = matched_difference(
        primary,
        v38,
        "recurrent_v38",
        evaluation_month,
    )
    _, diff_v40 = matched_difference(
        primary,
        v40,
        "fitted_optimal_stopping_v40",
        evaluation_month,
    )
    _, diff_hold30 = matched_difference(
        primary,
        hold30,
        "always_hold_30m",
        evaluation_month,
    )
    differences = pd.concat(
        [diff_v38, diff_v40, diff_hold30],
        ignore_index=True,
    )

    completed = primary.loc[primary["status"].eq("completed")]
    days, day_mean, low, high = _bootstrap_daily(
        completed,
        "base_net_return_pct",
    )
    bootstrap = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "days": days,
                "day_balanced_base_mean_pct": day_mean,
                "ci_low_pct": low,
                "ci_high_pct": high,
            }
        ]
    )

    teacher_cal_parts = []
    cal_anchors = anchors.loc[
        anchors["_partition"].eq("calibration")
    ].reset_index(drop=True)
    for month in sorted(set(_month_of(cal_anchors))):
        month_anchors = cal_anchors.loc[
            _month_of(cal_anchors).eq(month)
        ].reset_index(drop=True)
        month_rows = honest_rows.loc[
            _month_of(honest_rows).eq(month)
        ].reset_index(drop=True)
        teacher_traj, _ = build_optimal_trajectories(
            month_anchors,
            month_rows,
            states[month],
            month,
        )
        teacher_cal_parts.append(
            summarize_policy(
                teacher_traj,
                "v40_teacher_on_honest_calibration",
                month,
            )
        )
    teacher_calibration_summary = (
        pd.concat(teacher_cal_parts, ignore_index=True)
        if teacher_cal_parts
        else pd.DataFrame()
    )

    accounts = pd.concat(
        [
            account_replay(
                primary,
                eval_state,
                "honest_stopping_distillation_v41",
                evaluation_month,
            ),
            account_replay(
                v38,
                eval_state,
                "recurrent_v38",
                evaluation_month,
            ),
            account_replay(
                v40,
                eval_state,
                "fitted_optimal_stopping_v40",
                evaluation_month,
            ),
            account_replay(
                hold30,
                eval_state,
                "always_hold_30m",
                evaluation_month,
            ),
        ],
        ignore_index=True,
    )

    p = summaries.loc[
        summaries["policy"].eq("honest_stopping_distillation_v41")
    ].iloc[0]
    b38 = summaries.loc[
        summaries["policy"].eq("recurrent_v38")
    ].iloc[0]
    b40 = summaries.loc[
        summaries["policy"].eq("fitted_optimal_stopping_v40")
    ].iloc[0]
    d38 = diff_v38.iloc[0]
    boot = bootstrap.iloc[0]
    base_account = accounts.loc[
        accounts["policy"].eq("honest_stopping_distillation_v41")
        & accounts["scenario"].eq("base")
    ]
    if len(base_account) != 1:
        raise AssertionError("expected one v4.1 BASE account row")
    account = base_account.iloc[0]

    success = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "at_least_100_completed": bool(p["completed"] >= 100),
                "valid_entry_coverage_at_least_95pct": bool(
                    p["valid_entry_coverage"] >= 0.95
                ),
                "completion_at_least_90pct": bool(
                    p["completion_coverage"] >= 0.90
                ),
                "base_mean_positive": bool(
                    p["base_net_return_pct_mean"] > 0
                ),
                "day_base_mean_positive": bool(
                    p["day_balanced_base_mean_pct"] > 0
                ),
                "base_bootstrap_low_positive": bool(
                    boot["ci_low_pct"] > 0
                ),
                "beats_v38_day_mean": bool(
                    p["day_balanced_base_mean_pct"]
                    > b38["day_balanced_base_mean_pct"]
                ),
                "v38_difference_positive_robust": bool(
                    d38["day_balanced_difference_pct"] > 0
                    and d38["ci_low_pct"] > 0
                ),
                "beats_v40_day_mean": bool(
                    p["day_balanced_base_mean_pct"]
                    > b40["day_balanced_base_mean_pct"]
                ),
                "stress_not_worse_than_v38": bool(
                    p["stress_net_return_pct_mean"]
                    >= b38["stress_net_return_pct_mean"]
                ),
                "base_account_positive_no_unresolved": bool(
                    account["marked_return_pct"] > 0
                    and int(account["unresolved"]) == 0
                ),
            }
        ]
    )
    success["all_frozen_checks_pass"] = success.iloc[:, 1:].all(axis=1)

    exit_reasons = (
        primary.groupby(
            ["month", "exit_reason"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    return (
        primary,
        decisions,
        summaries,
        differences,
        bootstrap,
        exit_reasons,
        teacher_calibration_summary,
        teacher_fit_diagnostics,
        teacher_honest_diagnostics,
        student_diagnostics,
        accounts,
        provenance,
        success,
    )


def render_report(outputs: tuple[pd.DataFrame, ...]) -> str:
    (
        _primary,
        _decisions,
        summaries,
        differences,
        bootstrap,
        exit_reasons,
        teacher_calibration_summary,
        teacher_fit_diagnostics,
        teacher_honest_diagnostics,
        student_diagnostics,
        accounts,
        provenance,
        success,
    ) = outputs
    lines = [
        "=== MoneyMaker Honest Stopping Distillation v4.1 ===",
        "fit-only v4.0 teacher; honest realized teacher continuation on later calibration",
        "student HOLD iff predicted honest downstream BASE advantage > 0",
        "frozen entry/state/lags/costs/30m cap; April 2026+ sealed",
    ]
    for title, frame in (
        ("Policy summaries", summaries),
        ("Matched differences", differences),
        ("Primary day bootstrap", bootstrap),
        ("Primary exit reasons", exit_reasons),
        ("Teacher calibration trajectories", teacher_calibration_summary),
        ("Teacher fit diagnostics", teacher_fit_diagnostics),
        ("Honest teacher target diagnostics", teacher_honest_diagnostics),
        ("Student diagnostics", student_diagnostics),
        ("Account replay", accounts),
        ("Provenance", provenance),
        ("Frozen checks", success),
    ):
        lines.extend(("", f"=== {title} ===", frame.to_string(index=False)))
    return "\n".join(lines)


def _parse(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-month", required=True)
    parser.add_argument("--anchor", action="append", type=_parse, required=True)
    parser.add_argument("--state", action="append", type=_parse, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trajectories-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--summaries-csv", type=Path, required=True)
    parser.add_argument("--differences-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-csv", type=Path, required=True)
    parser.add_argument("--exit-reasons-csv", type=Path, required=True)
    parser.add_argument("--teacher-calibration-csv", type=Path, required=True)
    parser.add_argument("--teacher-fit-csv", type=Path, required=True)
    parser.add_argument("--teacher-honest-csv", type=Path, required=True)
    parser.add_argument("--student-diagnostics-csv", type=Path, required=True)
    parser.add_argument("--accounts-csv", type=Path, required=True)
    parser.add_argument("--provenance-csv", type=Path, required=True)
    parser.add_argument("--success-csv", type=Path, required=True)
    args = parser.parse_args()

    outputs = run_fold(
        dict(args.anchor),
        dict(args.state),
        args.evaluation_month,
    )
    report = render_report(outputs)
    print(report, flush=True)

    paths = (
        args.trajectories_csv,
        args.decisions_csv,
        args.summaries_csv,
        args.differences_csv,
        args.bootstrap_csv,
        args.exit_reasons_csv,
        args.teacher_calibration_csv,
        args.teacher_fit_csv,
        args.teacher_honest_csv,
        args.student_diagnostics_csv,
        args.accounts_csv,
        args.provenance_csv,
        args.success_csv,
    )
    for path, frame in zip(paths, outputs, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
