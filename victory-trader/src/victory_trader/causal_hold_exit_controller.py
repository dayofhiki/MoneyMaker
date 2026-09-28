"""Request268: recurrent causal HOLD/EXIT controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .causal_hold_exit_signal import (
    POSITION_CAP_MINUTES,
    build_position_states,
    fit_models,
    path_columns,
)
from .causal_minute_controller_rebuild import causal_execution_scan
from .full_hot_fixed_policy_value import attach_fixed_value
from .learned_pullback_entry import (
    CONTROLLER_TRAIN_DAYS,
    TEST_DAYS,
    controller_x,
)
from .rich_post_hot_state import score_and_select_candidates

REQUEST_ID = 268
HOLD_PROBABILITY_THRESHOLD = 0.50
MIN_TRAIN_STATES = 3000
MIN_TEST_STATES = 5000
MIN_EPISODE_COVERAGE = 0.80
MIN_POSITIVE_RATE = 0.40
MAX_SEVERE_RATE = 0.15
MIN_POSITIVE_MEAN_DAYS = 5
MIN_GAIN_VS_BEST_STATIC = 0.10


def _episode_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.trading_day.astype(str)
        + "|"
        + frame.ticker.astype(str).str.upper()
        + "|"
        + frame.hot_t.astype("int64").astype(str)
    )


def _base_record(group: pd.DataFrame, chosen: pd.Series, name: str) -> dict:
    first = group.iloc[0]
    return {
        "trading_day": str(first.trading_day),
        "ticker": str(first.ticker).upper(),
        "hot_t": int(first.hot_t),
        "entry_t": int(first.entry_t),
        "entry_price": float(first.entry_price),
        "policy": name,
        "exit_t": int(chosen.state_t),
        "exit_price": float(chosen.state_price),
        "return_pct": float(chosen.exit_now_pct),
        "exit_state_number": int(chosen.events_since_entry),
        "available_position_states": int(len(group)),
    }


def static_policy_rows(
    states: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    rows = []
    for _, group in states.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        ordered = group.sort_values("state_t").reset_index(drop=True)
        if ordered.empty:
            continue
        if mode == "first":
            chosen = ordered.iloc[0]
        elif mode == "fifth":
            chosen = ordered.iloc[min(4, len(ordered) - 1)]
        elif mode == "terminal":
            chosen = ordered.iloc[-1]
        else:
            raise ValueError(f"unknown Request268 static mode: {mode}")
        rows.append(_base_record(ordered, chosen, mode))
    return pd.DataFrame(rows)


def dynamic_policy_rows(
    states: pd.DataFrame,
    reg,
    cls,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    work = states.copy()
    x = controller_x(work, columns)
    work["predicted_hold_advantage_pct"] = reg.predict(x)
    work["hold_probability"] = cls.predict_proba(x)[:, 1]
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
        for _, state in ordered.iterrows():
            hold = (
                float(state.predicted_hold_advantage_pct) > 0.0
                and float(state.hold_probability)
                >= HOLD_PROBABILITY_THRESHOLD
            )
            if not hold:
                chosen = state
                exit_reason = "model_exit"
                break
        record = _base_record(ordered, chosen, "dynamic")
        record["exit_reason"] = exit_reason
        record["predicted_hold_advantage_pct_at_exit"] = float(
            chosen.predicted_hold_advantage_pct
        )
        record["hold_probability_at_exit"] = float(
            chosen.hold_probability
        )
        rows.append(record)
    return pd.DataFrame(rows)


def policy_metrics(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {
            "episodes": 0,
            "mean_pct": None,
            "median_pct": None,
            "positive_rate": None,
            "severe_loss_rate_le_minus2": None,
            "day_balanced_mean_pct": None,
            "positive_mean_days": 0,
            "mean_exit_state_number": None,
        }
    ret = pd.to_numeric(rows.return_pct, errors="coerce")
    valid = ret.notna()
    work = rows.loc[valid].copy()
    ret = pd.to_numeric(work.return_pct, errors="coerce")
    daily = work.groupby(
        work.trading_day.astype(str),
        sort=True,
    ).return_pct.mean()
    return {
        "episodes": int(len(work)),
        "mean_pct": float(ret.mean()),
        "median_pct": float(ret.median()),
        "positive_rate": float(ret.gt(0).mean()),
        "severe_loss_rate_le_minus2": float(ret.le(-2).mean()),
        "day_balanced_mean_pct": float(daily.mean()) if len(daily) else None,
        "positive_mean_days": int((daily > 0).sum()),
        "mean_exit_state_number": float(
            pd.to_numeric(
                work.exit_state_number,
                errors="coerce",
            ).mean()
        ),
        "by_day": {
            str(day): {
                "episodes": int(len(part)),
                "mean_pct": float(
                    pd.to_numeric(part.return_pct, errors="coerce").mean()
                ),
            }
            for day, part in work.groupby(
                work.trading_day.astype(str),
                sort=True,
            )
        },
    }


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
    scored, train_threshold, _ = score_and_select_candidates(first_hot)

    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(CONTROLLER_TRAIN_DAYS)
        & scored.candidate_probability.ge(train_threshold)
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_probability.ge(train_threshold)
    ].copy()

    train_states = build_position_states(train_selected, causal_scan)
    test_states = build_position_states(test_selected, causal_scan)
    columns = path_columns(train_states)
    reg, cls = fit_models(train_states, columns, 20261400)

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
    static_means = [
        report["mean_pct"]
        for report in statics.values()
        if report["mean_pct"] is not None
    ]
    best_static_mean = max(static_means) if static_means else None
    gain_vs_best_static = (
        dynamic["mean_pct"] - best_static_mean
        if dynamic["mean_pct"] is not None
        and best_static_mean is not None
        else None
    )

    test_episode_count = int(
        test_states.assign(_key=_episode_key(test_states))["_key"].nunique()
    ) if not test_states.empty else 0
    candidate_count = int(len(test_selected))
    episode_coverage = (
        float(test_episode_count / candidate_count)
        if candidate_count
        else 0.0
    )
    checks = {
        "train_position_states": len(train_states) >= MIN_TRAIN_STATES,
        "test_position_states": len(test_states) >= MIN_TEST_STATES,
        "test_position_episode_coverage": (
            episode_coverage >= MIN_EPISODE_COVERAGE
        ),
        "dynamic_mean_positive": (
            dynamic["mean_pct"] is not None
            and dynamic["mean_pct"] > 0
        ),
        "day_balanced_mean_positive": (
            dynamic["day_balanced_mean_pct"] is not None
            and dynamic["day_balanced_mean_pct"] > 0
        ),
        "dynamic_positive_rate": (
            dynamic["positive_rate"] is not None
            and dynamic["positive_rate"] >= MIN_POSITIVE_RATE
        ),
        "dynamic_severe_loss_rate": (
            dynamic["severe_loss_rate_le_minus2"] is not None
            and dynamic["severe_loss_rate_le_minus2"]
            <= MAX_SEVERE_RATE
        ),
        "positive_mean_days": (
            dynamic["positive_mean_days"] >= MIN_POSITIVE_MEAN_DAYS
        ),
        "gain_vs_best_static": (
            gain_vs_best_static is not None
            and gain_vs_best_static >= MIN_GAIN_VS_BEST_STATIC
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
        "candidate_threshold_top20": train_threshold,
        "hold_probability_threshold": HOLD_PROBABILITY_THRESHOLD,
        "position_cap_minutes": POSITION_CAP_MINUTES,
        "support": {
            "train_candidate_episodes": int(len(train_selected)),
            "test_candidate_episodes": candidate_count,
            "train_position_states": int(len(train_states)),
            "test_position_states": int(len(test_states)),
            "test_position_episodes": test_episode_count,
            "test_position_episode_coverage": episode_coverage,
            "feature_count": len(columns),
        },
        "policies": {
            "dynamic_hold_exit": dynamic,
            "static_first_exit": statics["first"],
            "static_fifth_exit": statics["fifth"],
            "static_terminal_exit": statics["terminal"],
        },
        "best_static_mean_pct": best_static_mean,
        "mean_gain_vs_best_static_pct": gain_vs_best_static,
        "checks": checks,
        "economic_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "The future-best exit appears only in Request267 training labels. "
            "Request268 decisions use current causal path features and fixed "
            "0/0.50 regression/classification thresholds only. May11-20 is "
            "development-only."
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
