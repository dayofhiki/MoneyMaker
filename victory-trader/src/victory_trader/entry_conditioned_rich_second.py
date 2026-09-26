"""Request 220: entry-conditioned rich-second POSITION state.

Freeze Request219's five-event remaining-option target and development dates.
Change only the observable state: add the full Request171 one-second feature
family and causal deltas from the entry observation to the current observation.

This tests whether "how microstructure changed since I entered" is more useful
than either the coarse current state or unanchored recent sequence history.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_settings
from .event_time_position_replay import EVENT_FEATURES
from .massive_client import MassiveClient
from .multi_event_option_value import (
    attach_option_rank_target,
    attach_option_value_target,
    option_diagnostics,
    predict_target_rank,
    train_target_rank_model,
    _prepare_event_frames,
)
from .rich_second_position_value import (
    RICH_SECOND_FEATURES,
    rich_second_features,
)
from .second_path_attention_probe import (
    SECOND_CACHE_DIR,
    _second_frame,
)

REQUEST_ID = 220

ENTRY_DELTA_FEATURES = tuple(
    f"entry_delta_{source}" for source in RICH_SECOND_FEATURES
)
RICH_CURRENT_FEATURES = tuple(
    dict.fromkeys([*EVENT_FEATURES, *RICH_SECOND_FEATURES])
)
ENTRY_CONDITIONED_FEATURES = tuple(
    dict.fromkeys(
        [
            *EVENT_FEATURES,
            *RICH_SECOND_FEATURES,
            *ENTRY_DELTA_FEATURES,
        ]
    )
)

MIN_FRESH_RICH_COVERAGE = 0.85
MIN_CALIBRATION_GAIN = 0.015
MIN_FRESH_SPEARMAN = 0.05
MIN_FRESH_GAIN = 0.02
MIN_GOOD_DAYS = 3
MIN_TOP10_LIFT_PCT = 0.30
MIN_TOP10_POSITIVE_RATE_LIFT = 0.03
MIN_SAME_AGE_MEDIAN_SPEARMAN = 0.03


def enrich_positions(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
) -> int:
    """Overwrite/add Request171 rich-second fields using only completed seconds."""
    frame = pd.read_parquet(input_path).copy()
    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR,
        request_interval_seconds=0.0,
    )

    cache: dict[tuple[str, str], pd.DataFrame] = {}
    records: list[dict[str, float]] = []
    columns = ["trading_day", "ticker", "state_t"]
    for row in frame.loc[:, columns].to_dict("records"):
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        key = (day, ticker)
        if key not in cache:
            payload = client.second_bars_range(
                ticker,
                date.fromisoformat(day),
                date.fromisoformat(day),
                adjusted=False,
            )
            cache[key] = _second_frame(payload)
        records.append(
            rich_second_features(
                cache[key],
                int(row["state_t"]),
            )
        )

    features = pd.DataFrame(records, index=frame.index)
    for source in RICH_SECOND_FEATURES:
        frame[source] = pd.to_numeric(
            features[source], errors="coerce"
        )

    any_coverage = float(
        frame.loc[:, RICH_SECOND_FEATURES]
        .notna()
        .any(axis=1)
        .mean()
    )
    coverage = {
        source: float(frame[source].notna().mean())
        for source in RICH_SECOND_FEATURES
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        output_path,
        index=False,
        compression="zstd",
    )
    audit = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "rows": int(len(frame)),
        "ticker_days": int(len(cache)),
        "any_rich_second_feature_coverage": any_coverage,
        "feature_coverage": coverage,
        "client_stats": client.stats.to_dict(),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def attach_entry_conditioned_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Attach current minus first-observed rich-second state per episode."""
    result = frame.copy()
    for feature in ENTRY_DELTA_FEATURES:
        result[feature] = np.nan

    keys = ["trading_day", "ticker", "hot_t"]
    for _, group in result.groupby(keys, sort=False):
        ordered = group.sort_values("state_t", kind="stable")
        if ordered.empty:
            continue
        entry = ordered.iloc[0]
        for source in RICH_SECOND_FEATURES:
            anchor = pd.to_numeric(
                pd.Series([entry.get(source)]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(anchor):
                continue
            values = pd.to_numeric(
                ordered[source], errors="coerce"
            )
            result.loc[
                ordered.index,
                f"entry_delta_{source}",
            ] = values - float(anchor)
    return result


def _coverage(frame: pd.DataFrame) -> float:
    available = [
        source
        for source in RICH_SECOND_FEATURES
        if source in frame.columns
    ]
    if not available:
        return 0.0
    return float(
        frame.loc[:, available].notna().any(axis=1).mean()
    )


def _select_candidate_on_calibration(
    diagnostics: dict[str, dict[str, object]],
) -> str:
    candidates = ["rich_current", "entry_conditioned"]
    return max(
        candidates,
        key=lambda name: (
            float(diagnostics[name]["spearman"])
            if diagnostics[name].get("spearman") is not None
            else -np.inf
        ),
    )


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    states_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )
    history_scan = pd.read_parquet(history_scan_path)

    fresh_position_paths = sorted(
        fresh_dir.glob("*-positions.parquet")
    )
    fresh_scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    if len(fresh_position_paths) != 5 or len(fresh_scan_paths) != 5:
        raise ValueError(
            "request 220 requires five enriched Request178 shards"
        )
    fresh_positions = pd.concat(
        [
            pd.read_parquet(path)
            for path in fresh_position_paths
        ],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in fresh_scan_paths],
        ignore_index=True,
    )

    fit, calibration, fresh = _prepare_event_frames(
        fit_positions,
        calibration_positions,
        fresh_positions,
        history_scan,
        fresh_scan,
    )

    fit = attach_option_rank_target(
        attach_option_value_target(fit)
    )
    calibration = attach_option_rank_target(
        attach_option_value_target(calibration)
    )
    fresh = attach_option_rank_target(
        attach_option_value_target(fresh)
    )

    fit = attach_entry_conditioned_features(fit)
    calibration = attach_entry_conditioned_features(calibration)
    fresh = attach_entry_conditioned_features(fresh)

    baseline_model = train_target_rank_model(
        fit, EVENT_FEATURES
    )
    rich_model = train_target_rank_model(
        fit, RICH_CURRENT_FEATURES
    )
    entry_model = train_target_rank_model(
        fit, ENTRY_CONDITIONED_FEATURES
    )

    for frame in (calibration, fresh):
        frame["baseline_score"] = predict_target_rank(
            frame, baseline_model
        )
        frame["rich_current_score"] = predict_target_rank(
            frame, rich_model
        )
        frame["entry_conditioned_score"] = predict_target_rank(
            frame, entry_model
        )

    calibration_diag = {
        "baseline": option_diagnostics(
            calibration, "baseline_score"
        ),
        "rich_current": option_diagnostics(
            calibration, "rich_current_score"
        ),
        "entry_conditioned": option_diagnostics(
            calibration, "entry_conditioned_score"
        ),
    }
    selected = _select_candidate_on_calibration(
        calibration_diag
    )
    fresh_diag = {
        "baseline": option_diagnostics(
            fresh, "baseline_score"
        ),
        "rich_current": option_diagnostics(
            fresh, "rich_current_score"
        ),
        "entry_conditioned": option_diagnostics(
            fresh, "entry_conditioned_score"
        ),
    }
    selected_diag = fresh_diag[selected]

    cal_base = calibration_diag["baseline"].get("spearman")
    cal_selected = calibration_diag[selected].get("spearman")
    fresh_base = fresh_diag["baseline"].get("spearman")
    fresh_selected = selected_diag.get("spearman")
    cal_gain = (
        float(cal_selected) - float(cal_base)
        if cal_selected is not None and cal_base is not None
        else None
    )
    fresh_gain = (
        float(fresh_selected) - float(fresh_base)
        if fresh_selected is not None and fresh_base is not None
        else None
    )

    top10 = selected_diag["top10_by_day"]
    fresh_coverage = _coverage(fresh)
    gate = bool(
        fresh_coverage >= MIN_FRESH_RICH_COVERAGE
        and cal_gain is not None
        and cal_gain >= MIN_CALIBRATION_GAIN
        and fresh_selected is not None
        and float(fresh_selected) >= MIN_FRESH_SPEARMAN
        and fresh_gain is not None
        and fresh_gain >= MIN_FRESH_GAIN
        and int(selected_diag["good_days"]) >= MIN_GOOD_DAYS
        and top10.get("day_balanced_mean_lift_pct") is not None
        and float(top10["day_balanced_mean_lift_pct"])
        >= MIN_TOP10_LIFT_PCT
        and top10.get("day_balanced_positive_rate_lift")
        is not None
        and float(top10["day_balanced_positive_rate_lift"])
        >= MIN_TOP10_POSITIVE_RATE_LIFT
        and selected_diag.get("same_age_median_spearman")
        is not None
        and float(selected_diag["same_age_median_spearman"])
        >= MIN_SAME_AGE_MEDIAN_SPEARMAN
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "target_changed_from_request219": False,
        "target": "Request219 five-event remaining-option upper bound",
        "intervention": (
            "full Request171 rich-second current state plus causal "
            "current-minus-entry rich-second deltas"
        ),
        "rich_second_features": list(RICH_SECOND_FEATURES),
        "entry_delta_features": list(ENTRY_DELTA_FEATURES),
        "fresh_rich_second_coverage": fresh_coverage,
        "feature_counts": {
            "baseline": len(baseline_model.feature_columns),
            "rich_current": len(rich_model.feature_columns),
            "entry_conditioned": len(entry_model.feature_columns),
        },
        "calibration": calibration_diag,
        "selected_candidate_on_calibration": selected,
        "calibration_selected_minus_baseline_spearman": cal_gain,
        "fresh": fresh_diag,
        "fresh_selected_candidate": selected_diag,
        "fresh_selected_minus_baseline_spearman": fresh_gain,
        "frozen_gate": {
            "min_fresh_rich_coverage": MIN_FRESH_RICH_COVERAGE,
            "min_calibration_gain": MIN_CALIBRATION_GAIN,
            "min_fresh_spearman": MIN_FRESH_SPEARMAN,
            "min_fresh_gain": MIN_FRESH_GAIN,
            "min_good_days": MIN_GOOD_DAYS,
            "min_top10_day_balanced_lift_pct": MIN_TOP10_LIFT_PCT,
            "min_top10_positive_rate_lift": (
                MIN_TOP10_POSITIVE_RATE_LIFT
            ),
            "min_same_age_median_spearman": (
                MIN_SAME_AGE_MEDIAN_SPEARMAN
            ),
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze the entry-conditioned rich-second state and replace "
            "Request219's hindsight maximum with a causal fitted "
            "continuation-value recursion before recurrent policy replay"
        ),
        "next_boundary_if_fail": (
            "do not add more indicator families; test whether predictive "
            "information is concentrated around entry/pullback transition "
            "events rather than every POSITION observation"
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    states_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fresh.to_parquet(
        states_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    enrich = sub.add_parser("enrich")
    enrich.add_argument("--input", type=Path, required=True)
    enrich.add_argument("--output", type=Path, required=True)
    enrich.add_argument("--audit", type=Path, required=True)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--fit-positions", type=Path, required=True)
    ev.add_argument(
        "--calibration-positions", type=Path, required=True
    )
    ev.add_argument("--history-scan", type=Path, required=True)
    ev.add_argument("--fresh-dir", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--states-output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "enrich":
        return enrich_positions(
            args.input,
            args.output,
            args.audit,
        )
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.states_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
