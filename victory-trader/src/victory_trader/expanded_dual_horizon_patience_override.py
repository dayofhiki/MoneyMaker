from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .expanded_fitted_optimal_stopping import (
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
from .expanded_honest_stopping_distillation import (
    build_honest_teacher_targets,
    train_honest_student,
)
from .expanded_opportunity_shared_q import (
    predict_opportunity,
    train_opportunity_model,
)
from .expanded_path_aware_continuation import (
    HOLD_THRESHOLD,
    MAX_HOLD_MINUTES,
    build_continuation_rows,
)
from .expanded_path_transition_continuation import (
    predict_transition_continuation,
    train_transition_model,
)
from .expanded_recurrent_path_policy import (
    account_replay,
    build_always_hold_comparator,
    build_recurrent_trajectories,
)
from .expanded_remaining_option_observability import (
    add_remaining_option_labels,
    apply_excess_target,
    fit_minute_baselines,
    predict_remaining_value,
    train_remaining_value_model,
)
from .expanded_supply_hurdle_ev import load_fold

RANDOM_SEED = 20261048


@dataclass(frozen=True)
class OverrideModel:
    model: HistGradientBoostingRegressor
    winsor_low_pct: float
    winsor_high_pct: float
    training_rows: int


def split_calibration_days(
    calibration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    days = sorted(calibration["trading_day"].astype(str).unique().tolist())
    split = len(days) // 2
    first = days[:split]
    second = days[split:]
    if len(first) < 5 or len(second) < 5:
        raise ValueError(
            f"v4.2 calibration split requires >=5 days per half, got "
            f"{len(first)}/{len(second)}"
        )
    day_series = calibration["trading_day"].astype(str)
    cal_a = calibration.loc[day_series.isin(first)].reset_index(drop=True)
    cal_b = calibration.loc[day_series.isin(second)].reset_index(drop=True)
    return cal_a, cal_b, first, second


def override_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "short_hold_probability": pd.to_numeric(
                frame["short_hold_probability"],
                errors="coerce",
            ),
            "predicted_excess_option_value_pct": pd.to_numeric(
                frame["predicted_excess_option_value_pct"],
                errors="coerce",
            ),
        },
        index=frame.index,
    )


def build_v38_patience_targets(
    rows: pd.DataFrame,
    states: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _attach_exit_now_base(rows).reset_index(drop=True)
    groups = _state_groups_by_month(states)
    held = pd.to_numeric(frame["minutes_held"], errors="coerce")

    row_by_key: dict[tuple[str, str, int], int] = {}
    for idx, row in frame.iterrows():
        minute = held.iloc[idx]
        if pd.notna(minute) and float(minute).is_integer():
            row_by_key[_policy_key(row, int(minute))] = int(idx)

    teacher_value = pd.Series(np.nan, index=frame.index, dtype=float)
    forced_hold_value = pd.Series(np.nan, index=frame.index, dtype=float)
    target = pd.Series(np.nan, index=frame.index, dtype=float)

    for minute in range(MAX_HOLD_MINUTES - 1, 0, -1):
        minute_idx = frame.index[held.eq(minute)]
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

            forced_hold_value.loc[idx] = downstream
            short = pd.to_numeric(
                pd.Series([row.get("short_hold_probability")]),
                errors="coerce",
            ).iloc[0]
            teacher_holds = bool(
                pd.notna(short)
                and np.isfinite(float(short))
                and float(short) > HOLD_THRESHOLD
            )

            if pd.notna(exit_now):
                if teacher_holds and pd.notna(downstream):
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

            gate = pd.to_numeric(
                pd.Series([row.get("opportunity_probability")]),
                errors="coerce",
            ).iloc[0]
            if (
                pd.notna(exit_now)
                and pd.notna(downstream)
                and pd.notna(gate)
                and float(gate) > 0.5
                and not teacher_holds
            ):
                target.loc[idx] = float(downstream) - float(exit_now)

    frame["v38_teacher_realized_value_pct"] = teacher_value
    frame["forced_one_minute_value_pct"] = forced_hold_value
    frame["patience_override_advantage_pct"] = target

    valid = frame.loc[
        pd.to_numeric(
            frame["patience_override_advantage_pct"],
            errors="coerce",
        ).notna()
    ].copy()
    target_values = pd.to_numeric(
        valid["patience_override_advantage_pct"],
        errors="coerce",
    )
    diag = pd.DataFrame(
        [
            {
                "candidate_rows": int(len(frame)),
                "valid_override_target_rows": int(len(valid)),
                "target_mean_pct": (
                    float(target_values.mean()) if len(valid) else np.nan
                ),
                "target_positive_rate": (
                    float(target_values.gt(0).mean()) if len(valid) else np.nan
                ),
                "short_probability_mean": (
                    float(
                        pd.to_numeric(
                            valid["short_hold_probability"],
                            errors="coerce",
                        ).mean()
                    )
                    if len(valid)
                    else np.nan
                ),
                "remaining_score_mean_pct": (
                    float(
                        pd.to_numeric(
                            valid["predicted_excess_option_value_pct"],
                            errors="coerce",
                        ).mean()
                    )
                    if len(valid)
                    else np.nan
                ),
            }
        ]
    )
    return frame, diag


def train_override_model(
    honest: pd.DataFrame,
) -> OverrideModel:
    target = pd.to_numeric(
        honest["patience_override_advantage_pct"],
        errors="coerce",
    )
    x = override_feature_frame(honest)
    valid = target.notna() & x.notna().all(axis=1)
    train = honest.loc[valid].copy()
    if len(train) < 500:
        raise ValueError(
            f"v4.2 override requires >=500 valid rows, got {len(train)}"
        )

    y = pd.to_numeric(
        train["patience_override_advantage_pct"],
        errors="coerce",
    ).astype(float)
    low, high = np.quantile(y.to_numpy(), [0.005, 0.995])
    y_fit = y.clip(lower=low, upper=high)
    x_fit = override_feature_frame(train)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=100,
        l2_regularization=2.0,
        random_state=RANDOM_SEED,
    )
    model.fit(x_fit, y_fit)
    return OverrideModel(
        model=model,
        winsor_low_pct=float(low),
        winsor_high_pct=float(high),
        training_rows=int(len(train)),
    )


def predict_override(
    frame: pd.DataFrame,
    fitted: OverrideModel,
) -> np.ndarray:
    x = override_feature_frame(frame)
    result = np.full(len(frame), np.nan, dtype=float)
    valid = x.notna().all(axis=1)
    if valid.any():
        result[valid.to_numpy()] = fitted.model.predict(x.loc[valid])
    return result


def _build_sequence(
    anchors: pd.DataFrame,
    states: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    parts = []
    for month, group in anchors.groupby(_month_of(anchors), sort=True):
        if month not in states:
            raise ValueError(f"missing state panel for {month}")
        parts.append(build_continuation_rows(states[month], group))
    return pd.concat(parts, ignore_index=True)


def _decision_sources(
    decisions: pd.DataFrame,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    if decisions.empty:
        return decisions.copy()
    keys = ["trading_day", "ticker", "minutes_held"]
    cols = keys + [
        "short_hold_probability",
        "predicted_excess_option_value_pct",
        "predicted_patience_override_pct",
    ]
    merged = decisions.merge(
        scored.loc[:, cols],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    short = pd.to_numeric(
        merged["short_hold_probability"],
        errors="coerce",
    )
    override = pd.to_numeric(
        merged["predicted_patience_override_pct"],
        errors="coerce",
    )
    merged["decision_source"] = np.select(
        [
            merged["action"].eq("HOLD") & short.gt(HOLD_THRESHOLD),
            merged["action"].eq("HOLD")
            & short.le(HOLD_THRESHOLD)
            & override.gt(0),
            merged["action"].eq("EXIT"),
            merged["action"].eq("EXIT_PENDING"),
        ],
        [
            "short_term_hold",
            "patience_override_hold",
            "dual_horizon_exit",
            "gap_exit_pending",
        ],
        default="other",
    )
    return merged


def run_fold(
    anchor_paths: dict[str, Path],
    state_paths: dict[str, Path],
    evaluation_month: str,
):
    fit, calibration, evaluation, provenance = load_fold(
        anchor_paths,
        evaluation_month,
    )
    cal_a, cal_b, cal_a_days, cal_b_days = split_calibration_days(
        calibration
    )

    months = sorted(
        set(_month_of(pd.concat([fit, calibration, evaluation], ignore_index=True)))
    )
    states = {
        month: pd.read_parquet(state_paths[month])
        for month in months
    }

    # Partial-history models used only to manufacture honest Calibration-B
    # patience targets.
    partial_opp = train_opportunity_model(fit, cal_a)
    partial_anchor_parts = []
    for name, frame in (
        ("fit", fit),
        ("calibration_a", cal_a),
        ("calibration_b", cal_b),
    ):
        item = frame.copy()
        item["_partial_partition"] = name
        item["opportunity_probability"] = predict_opportunity(
            item,
            partial_opp,
        )
        partial_anchor_parts.append(item)
    partial_anchors = pd.concat(partial_anchor_parts, ignore_index=True)
    partial_sequence = _build_sequence(partial_anchors, states)
    partial_sequence = add_remaining_option_labels(
        partial_sequence,
        states,
    )

    pfit = partial_sequence.loc[
        partial_sequence["_partial_partition"].eq("fit")
    ].reset_index(drop=True)
    pcal_a = partial_sequence.loc[
        partial_sequence["_partial_partition"].eq("calibration_a")
    ].reset_index(drop=True)
    pcal_b = partial_sequence.loc[
        partial_sequence["_partial_partition"].eq("calibration_b")
    ].reset_index(drop=True)

    baselines = fit_minute_baselines(pfit)
    pfit = apply_excess_target(pfit, baselines)
    pcal_a = apply_excess_target(pcal_a, baselines)
    pcal_b = apply_excess_target(pcal_b, baselines)

    partial_short = train_transition_model(pfit, pcal_a)
    partial_medium = train_remaining_value_model(pfit, pcal_a)
    pcal_b["short_hold_probability"] = (
        predict_transition_continuation(pcal_b, partial_short)
    )
    pcal_b["predicted_excess_option_value_pct"] = (
        predict_remaining_value(pcal_b, partial_medium)
    )

    honest_b, honest_diag = build_v38_patience_targets(
        pcal_b,
        states,
    )
    override_model = train_override_model(honest_b)
    honest_valid = honest_b.loc[
        pd.to_numeric(
            honest_b["patience_override_advantage_pct"],
            errors="coerce",
        ).notna()
    ].copy()
    honest_valid["override_fit_prediction_pct"] = predict_override(
        honest_valid,
        override_model,
    )
    override_training_diag = pd.DataFrame(
        [
            {
                "evaluation_month": evaluation_month,
                "calibration_a_days": len(cal_a_days),
                "calibration_b_days": len(cal_b_days),
                "calibration_a_start": cal_a_days[0],
                "calibration_a_end": cal_a_days[-1],
                "calibration_b_start": cal_b_days[0],
                "calibration_b_end": cal_b_days[-1],
                "training_rows": override_model.training_rows,
                "target_mean_pct": float(
                    pd.to_numeric(
                        honest_valid[
                            "patience_override_advantage_pct"
                        ],
                        errors="coerce",
                    ).mean()
                ),
                "target_positive_rate": float(
                    pd.to_numeric(
                        honest_valid[
                            "patience_override_advantage_pct"
                        ],
                        errors="coerce",
                    ).gt(0).mean()
                ),
                "winsor_low_pct": override_model.winsor_low_pct,
                "winsor_high_pct": override_model.winsor_high_pct,
                "prediction_mean_pct": float(
                    pd.to_numeric(
                        honest_valid["override_fit_prediction_pct"],
                        errors="coerce",
                    ).mean()
                ),
                "prediction_positive_rate": float(
                    pd.to_numeric(
                        honest_valid["override_fit_prediction_pct"],
                        errors="coerce",
                    ).gt(0).mean()
                ),
            }
        ]
    )

    # Full pre-evaluation upstream models.
    full_opp = train_opportunity_model(fit, calibration)
    anchor_parts = []
    for name, frame in (
        ("fit", fit),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        item = frame.copy()
        item["_partition"] = name
        item["opportunity_probability"] = predict_opportunity(
            item,
            full_opp,
        )
        anchor_parts.append(item)
    anchors = pd.concat(anchor_parts, ignore_index=True)
    sequence = _build_sequence(anchors, states)
    sequence = add_remaining_option_labels(sequence, states)

    fit_rows = sequence.loc[
        sequence["_partition"].eq("fit")
    ].reset_index(drop=True)
    cal_rows = sequence.loc[
        sequence["_partition"].eq("calibration")
    ].reset_index(drop=True)
    eval_rows = sequence.loc[
        sequence["_partition"].eq("evaluation")
    ].reset_index(drop=True)

    full_baselines = fit_minute_baselines(fit_rows)
    fit_rows = apply_excess_target(fit_rows, full_baselines)
    cal_rows = apply_excess_target(cal_rows, full_baselines)
    eval_rows = apply_excess_target(eval_rows, full_baselines)

    full_short = train_transition_model(fit_rows, cal_rows)
    full_medium = train_remaining_value_model(fit_rows, cal_rows)
    eval_rows["short_hold_probability"] = (
        predict_transition_continuation(eval_rows, full_short)
    )
    eval_rows["predicted_excess_option_value_pct"] = (
        predict_remaining_value(eval_rows, full_medium)
    )
    eval_rows["predicted_patience_override_pct"] = predict_override(
        eval_rows,
        override_model,
    )

    short_hold = pd.to_numeric(
        eval_rows["short_hold_probability"],
        errors="coerce",
    ).gt(HOLD_THRESHOLD)
    patience_hold = (
        ~short_hold
        & pd.to_numeric(
            eval_rows["predicted_patience_override_pct"],
            errors="coerce",
        ).gt(0)
    )
    eval_rows["hold_probability"] = np.where(
        short_hold | patience_hold,
        1.0,
        0.0,
    )

    eval_anchors = anchors.loc[
        anchors["_partition"].eq("evaluation")
    ].reset_index(drop=True)
    eval_state = states[evaluation_month]

    primary, decisions = build_recurrent_trajectories(
        eval_anchors,
        eval_rows,
        eval_state,
        evaluation_month,
    )
    decisions = _decision_sources(decisions, eval_rows)

    v38_scored = eval_rows.copy()
    v38_scored["hold_probability"] = v38_scored[
        "short_hold_probability"
    ]
    v38, _ = build_recurrent_trajectories(
        eval_anchors,
        v38_scored,
        eval_state,
        evaluation_month,
    )

    # Reproduce v4.1 as a diagnostic comparator.
    teacher_models, _teacher_fit_diag, _ = train_fitted_stopping(
        fit_rows,
        states,
    )
    honest_rows, _teacher_honest_diag = build_honest_teacher_targets(
        cal_rows,
        teacher_models,
        states,
    )
    student_models, _student_diag = train_honest_student(honest_rows)
    v41_scored = score_stopping_rows(eval_rows, student_models)
    v41, _ = build_optimal_trajectories(
        eval_anchors,
        v41_scored,
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
                "dual_horizon_patience_override_v42",
                evaluation_month,
            ),
            summarize_policy(v38, "recurrent_v38", evaluation_month),
            summarize_policy(
                v41,
                "honest_stopping_distillation_v41",
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
    _, diff_v41 = matched_difference(
        primary,
        v41,
        "honest_stopping_distillation_v41",
        evaluation_month,
    )
    differences = pd.concat([diff_v38, diff_v41], ignore_index=True)

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

    decision_diag = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "reached_decisions": int(len(decisions)),
                "short_term_hold_decisions": int(
                    decisions["decision_source"]
                    .eq("short_term_hold")
                    .sum()
                )
                if len(decisions)
                else 0,
                "patience_override_hold_decisions": int(
                    decisions["decision_source"]
                    .eq("patience_override_hold")
                    .sum()
                )
                if len(decisions)
                else 0,
                "dual_horizon_exit_decisions": int(
                    decisions["decision_source"]
                    .eq("dual_horizon_exit")
                    .sum()
                )
                if len(decisions)
                else 0,
                "patience_override_fraction": (
                    float(
                        decisions["decision_source"]
                        .eq("patience_override_hold")
                        .mean()
                    )
                    if len(decisions)
                    else np.nan
                ),
            }
        ]
    )

    accounts = pd.concat(
        [
            account_replay(
                primary,
                eval_state,
                "dual_horizon_patience_override_v42",
                evaluation_month,
            ),
            account_replay(
                v38,
                eval_state,
                "recurrent_v38",
                evaluation_month,
            ),
            account_replay(
                v41,
                eval_state,
                "honest_stopping_distillation_v41",
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
        summaries["policy"].eq("dual_horizon_patience_override_v42")
    ].iloc[0]
    b38 = summaries.loc[
        summaries["policy"].eq("recurrent_v38")
    ].iloc[0]
    d38 = diff_v38.iloc[0]
    boot = bootstrap.iloc[0]
    base_account = accounts.loc[
        accounts["policy"].eq("dual_horizon_patience_override_v42")
        & accounts["scenario"].eq("base")
    ]
    if len(base_account) != 1:
        raise AssertionError("expected one v4.2 BASE account row")
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
                "stress_not_worse_than_v38": bool(
                    p["stress_net_return_pct_mean"]
                    >= b38["stress_net_return_pct_mean"]
                ),
                "base_account_above_start_no_unresolved": bool(
                    account["ending_stale_marked_equity"] > 10_000.0
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
        decision_diag,
        exit_reasons,
        honest_diag,
        override_training_diag,
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
        decision_diag,
        exit_reasons,
        honest_diag,
        override_training_diag,
        accounts,
        provenance,
        success,
    ) = outputs
    lines = [
        "=== MoneyMaker Dual-Horizon Patience Override v4.2 ===",
        "v3.8 remains default short-horizon controller",
        "override only teacher EXIT states using v3.7 probability + v3.9 excess remaining value",
        "honest override target from chronological Calibration-B",
        "frozen costs/0.5 boundary/zero boundary/30m cap; April 2026+ sealed",
    ]
    for title, frame in (
        ("Policy summaries", summaries),
        ("Matched differences", differences),
        ("Primary day bootstrap", bootstrap),
        ("Reached-decision diagnostics", decision_diag),
        ("Primary exit reasons", exit_reasons),
        ("Honest target diagnostics", honest_diag),
        ("Override training diagnostics", override_training_diag),
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
    parser.add_argument("--decision-diagnostics-csv", type=Path, required=True)
    parser.add_argument("--exit-reasons-csv", type=Path, required=True)
    parser.add_argument("--honest-target-csv", type=Path, required=True)
    parser.add_argument("--override-training-csv", type=Path, required=True)
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
        args.decision_diagnostics_csv,
        args.exit_reasons_csv,
        args.honest_target_csv,
        args.override_training_csv,
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
