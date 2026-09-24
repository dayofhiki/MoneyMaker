"""Request 166: BUY-gated POSITION value observability.

No new market dates are opened. The Request-165 economic + executability BUY
bridge is frozen and used to define entries. The position feature/model family
is intentionally kept identical to corrected Request 145 so this experiment
isolates whether the validated entry gate fixes the old coverage/stability
failure before composing a recurrent HOLD/EXIT trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .causal_entry_executability import _fit_exec_model
from .config import load_settings
from .extended_history_economic_opportunity import (
    EXTENDED_CAL_DAYS,
    EXTENDED_FIT_DAYS,
    _fit_extended_policy_selector,
)
from .fresh_validated_attention_economic_opportunity import FRESH_EVAL_DAYS
from .massive_client import MassiveClient
from .second_path_attention_probe import SECOND_CACHE_DIR
from .selected_hot_position_value_observability import (
    MODEL_FEATURES,
    apply_excess_target,
    build_position_rows,
    evaluate_observability,
    fit_minute_baselines,
    predict_hold,
    predict_option,
    train_hold_model,
    train_option_model,
)

REQUEST_ID = 166

FIT_DAYS = list(EXTENDED_FIT_DAYS)
CAL_DAYS = list(EXTENDED_CAL_DAYS)
EVAL_DAYS = list(FRESH_EVAL_DAYS)

PHASE_DAYS = {
    "fit": FIT_DAYS,
    "calibration": CAL_DAYS,
    "evaluation": EVAL_DAYS,
}


def _score_buy_actions(first_hot: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    economic_model, economic_gate = _fit_extended_policy_selector(first_hot)
    exec_model, exec_cal = _fit_exec_model(first_hot)

    frame = first_hot.copy()
    economic_features = frame.reindex(
        columns=[
            "attention_score",
            "attention_rank",
            "return_from_previous_close_pct",
            "minute_body_return_pct",
            "minute_range_pct",
            "log_minute_volume",
            "log_minute_transactions",
            "minute_return_1m_pct",
            "return_accel_1m_pct",
            "volume_ratio_prev1",
            "transactions_ratio_prev1",
            "focus_age_minutes",
            "watch_run_age_minutes",
            "stage1_hazard_probability",
            "active_seconds_60",
            "active_seconds_10",
            "last_activity_age_s",
            "sec_last10_return_pct",
            "sec_prev10_return_pct",
            "sec_accel_10_pct",
            "sec_first30_return_pct",
            "sec_last30_return_pct",
            "sec_accel_30_pct",
            "sec_realized_vol_pct",
            "sec_positive_fraction",
            "sec_max_drawdown_pct",
            "sec_max_runup_pct",
            "sec_volume_last10_share",
            "sec_transactions_last10_share",
            "sec_volume_burst_10",
            "sec_transactions_burst_10",
            "minute_rerank_probability",
            "second_rerank_probability",
            "log_current_price",
            "minutes_since_open",
            "minutes_to_close",
            "hot_candidate_rank",
            "transport_desired_now_numeric",
        ]
    ).replace([np.inf, -np.inf], np.nan)

    # Both Request 163 and Request 165 use the exact same ENTRY_FEATURES
    # contract. Keeping this explicit avoids any future-label columns.
    econ_p = economic_model.predict_proba(economic_features)[:, 1]
    exec_p = exec_model.predict_proba(economic_features)[:, 1]
    frame["economic_probability"] = econ_p
    frame["execution_probability"] = exec_p
    frame["entry_action"] = np.where(
        (econ_p >= float(economic_gate))
        & (exec_p >= float(exec_cal["threshold"])),
        "BUY",
        np.where(econ_p >= float(economic_gate), "WAIT", "ABSTAIN"),
    )
    return frame, {
        "economic_gate": float(economic_gate),
        "execution_gate": float(exec_cal["threshold"]),
        "execution_calibration_entry_rate": float(exec_cal["entry_rate"]),
    }


def prepare_phase(
    phase: str,
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    audit_path: Path,
) -> int:
    days = PHASE_DAYS[phase]
    first_hot = pd.read_parquet(first_hot_path)
    scored, gates = _score_buy_actions(first_hot)
    scan = pd.read_parquet(scan_path)

    settings = load_settings()
    second_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR,
        request_interval_seconds=0.0,
    )

    pieces: list[pd.DataFrame] = []
    by_day: dict[str, object] = {}
    for day in days:
        anchors = scored.loc[
            scored["trading_day"].astype(str).eq(day)
            & scored["entry_action"].astype(str).eq("BUY"),
            ["trading_day", "ticker", "t"],
        ].copy()
        scan_day = scan.loc[
            scan["trading_day"].astype(str).eq(day)
        ].copy()
        paths, audit = build_position_rows(
            anchors,
            scan_day,
            second_client,
        )
        path_anchor_count = (
            int(
                paths.loc[:, ["trading_day", "ticker", "hot_t"]]
                .drop_duplicates()
                .shape[0]
            )
            if len(paths)
            else 0
        )
        by_day[day] = {
            "buy_anchors": int(len(anchors)),
            "path_anchors": path_anchor_count,
            "anchor_path_coverage": (
                float(path_anchor_count / len(anchors))
                if len(anchors)
                else None
            ),
            "position_rows": int(len(paths)),
            "path_audit": audit,
        }
        if not paths.empty:
            pieces.append(paths)

    if not pieces:
        raise ValueError(f"request 166 {phase} produced no position rows")

    frame = pd.concat(pieces, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False, compression="zstd")

    total_anchors = sum(int(item["buy_anchors"]) for item in by_day.values())
    total_path_anchors = sum(
        int(item["path_anchors"]) for item in by_day.values()
    )
    audit = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "phase": phase,
        "days": days,
        "gates": gates,
        "buy_anchors": total_anchors,
        "path_anchors": total_path_anchors,
        "anchor_path_coverage": (
            float(total_path_anchors / total_anchors)
            if total_anchors
            else None
        ),
        "position_rows": int(len(frame)),
        "by_day": by_day,
        "second_client_stats": second_client.stats.to_dict(),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    evaluation_path: Path,
    fit_audit_path: Path,
    calibration_audit_path: Path,
    evaluation_audit_path: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit = pd.read_parquet(fit_path)
    calibration = pd.read_parquet(calibration_path)
    evaluation = pd.read_parquet(evaluation_path)

    baselines = fit_minute_baselines(fit)
    fit = apply_excess_target(fit, baselines)
    calibration = apply_excess_target(calibration, baselines)
    evaluation = apply_excess_target(evaluation, baselines)

    hold_model = train_hold_model(fit, calibration)
    option_model = train_option_model(fit, calibration)
    evaluation["hold_probability"] = predict_hold(evaluation, hold_model)
    evaluation["predicted_excess_option_value_pct"] = predict_option(
        evaluation, option_model
    )

    observed = evaluate_observability(
        evaluation,
        label="request165_buy_gated",
    )

    fit_audit = json.loads(fit_audit_path.read_text(encoding="utf-8"))
    cal_audit = json.loads(
        calibration_audit_path.read_text(encoding="utf-8")
    )
    eval_audit = json.loads(
        evaluation_audit_path.read_text(encoding="utf-8")
    )

    coverage = eval_audit["anchor_path_coverage"]
    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "fit_days": FIT_DAYS,
        "calibration_days": CAL_DAYS,
        "evaluation_days": EVAL_DAYS,
        "entry_policy": "request165_economic_plus_executability_BUY",
        "position_model_changed_from_request145": False,
        "model_features": list(MODEL_FEATURES),
        "fit_audit": fit_audit,
        "calibration_audit": cal_audit,
        "evaluation_audit": eval_audit,
        "option_winsor_low_pct": option_model.winsor_low,
        "option_winsor_high_pct": option_model.winsor_high,
        "option_calibration_offset_pct": option_model.offset,
        "position_observability": observed,
        "promotion_gate_pass": bool(
            observed["bridge_pass"]
            and coverage is not None
            and float(coverage) >= 0.90
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    evaluation.to_parquet(
        scored_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("phase", choices=sorted(PHASE_DAYS))
    prep.add_argument("--first-hot", type=Path, required=True)
    prep.add_argument("--scan", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--audit", type=Path, required=True)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--fit", type=Path, required=True)
    ev.add_argument("--calibration", type=Path, required=True)
    ev.add_argument("--evaluation", type=Path, required=True)
    ev.add_argument("--fit-audit", type=Path, required=True)
    ev.add_argument("--calibration-audit", type=Path, required=True)
    ev.add_argument("--evaluation-audit", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--scored-output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        return prepare_phase(
            args.phase,
            args.first_hot,
            args.scan,
            args.output,
            args.audit,
        )
    return evaluate(
        args.fit,
        args.calibration,
        args.evaluation,
        args.fit_audit,
        args.calibration_audit,
        args.evaluation_audit,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
