"""Request 204: recurrent causal watch-to-entry controller.

The ticker entering HOT starts a watch episode. At each 1-10 minute causal
checkpoint the controller chooses ENTER_NOW, WAIT or DROP. Request-178 stays
sealed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .causal_second_risk_compatible_entry import (
    MODEL_SEED,
    SecondStore,
    _summarize,
    first_admitted_baseline,
    prepare_states,
)
from .future_cost_cover_state_observability import _day_weights
from .shadow_entry_repricing_decomposition import (
    EPISODE_KEYS,
    FEATURES,
)

REQUEST_ID = 204
RULE_NAME = "runner"
PATH_SEED_NOW = MODEL_SEED + 100
PATH_SEED_WAIT = MODEL_SEED + 101
BASE_SEED_NOW = MODEL_SEED + 102
BASE_SEED_WAIT = MODEL_SEED + 103
MIN_TRADES = 30
MIN_CLOSED_COVERAGE = 0.90
MIN_POSITIVE_DAYS = 6
MIN_MATCHED_UPLIFT_PCT = 0.50

WATCH_PATH_FEATURES = (
    "watch_location_in_running_range",
    "watch_peak_age_minutes",
    "watch_trough_age_minutes",
    "watch_drawdown_change_1m",
    "watch_recovery_change_1m",
    "watch_attention_change_1m",
    "watch_attention_change_from_first",
    "watch_log_volume_change_1m",
    "watch_log_volume_change_from_first",
    "watch_price_change_from_first_pct",
)


@dataclass(frozen=True)
class ValueHead:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    low: float
    high: float


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def add_watch_path_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add only current/past episode-path features."""

    work = frame.sort_values(
        EPISODE_KEYS + ["state_t"],
        kind="stable",
    ).copy()

    move = _numeric(work, "entry_to_current_close_pct")
    held = _numeric(work, "minutes_held")
    drawdown = _numeric(work, "drawdown_from_peak_pct")
    recovery = _numeric(work, "recovery_from_trough_pct")
    attention = _numeric(work, "attention_score")
    log_volume = _numeric(work, "log_minute_volume")
    log_close = _numeric(work, "log_current_close")

    output_parts: list[pd.DataFrame] = []
    for _, group in work.groupby(EPISODE_KEYS, sort=False):
        idx = group.index
        gmove = move.loc[idx].to_numpy(dtype=float)
        gheld = held.loc[idx].to_numpy(dtype=float)

        # The HOT admission point is a virtual move=0 at held=0. This prevents
        # an episode that immediately pulls back from pretending its minute-1
        # state was the historical peak.
        peak_value = 0.0
        trough_value = 0.0
        peak_at = 0.0
        trough_at = 0.0
        peak_age: list[float] = []
        trough_age: list[float] = []
        locations: list[float] = []
        for value, minute in zip(gmove, gheld, strict=True):
            if np.isfinite(value):
                if value >= peak_value:
                    peak_value = float(value)
                    peak_at = float(minute)
                if value <= trough_value:
                    trough_value = float(value)
                    trough_at = float(minute)
            width = peak_value - trough_value
            locations.append(
                float((value - trough_value) / width)
                if np.isfinite(value) and width > 0
                else np.nan
            )
            peak_age.append(
                float(minute - peak_at)
                if np.isfinite(minute)
                else np.nan
            )
            trough_age.append(
                float(minute - trough_at)
                if np.isfinite(minute)
                else np.nan
            )

        part = pd.DataFrame(index=idx)
        part["watch_location_in_running_range"] = locations
        part["watch_peak_age_minutes"] = peak_age
        part["watch_trough_age_minutes"] = trough_age

        for source, name in (
            (drawdown.loc[idx], "watch_drawdown_change_1m"),
            (recovery.loc[idx], "watch_recovery_change_1m"),
            (attention.loc[idx], "watch_attention_change_1m"),
            (log_volume.loc[idx], "watch_log_volume_change_1m"),
        ):
            part[name] = source.diff().to_numpy(dtype=float)

        first_attention = attention.loc[idx].iloc[0]
        first_volume = log_volume.loc[idx].iloc[0]
        first_log_close = log_close.loc[idx].iloc[0]
        part["watch_attention_change_from_first"] = (
            attention.loc[idx] - first_attention
        ).to_numpy(dtype=float)
        part["watch_log_volume_change_from_first"] = (
            log_volume.loc[idx] - first_volume
        ).to_numpy(dtype=float)
        part["watch_price_change_from_first_pct"] = (
            (np.exp(log_close.loc[idx]) / np.exp(first_log_close) - 1.0)
            * 100.0
            if np.isfinite(first_log_close)
            else np.nan
        )
        output_parts.append(part)

    features = (
        pd.concat(output_parts).sort_index()
        if output_parts
        else pd.DataFrame(index=work.index)
    )
    for column in WATCH_PATH_FEATURES:
        work[column] = features.reindex(work.index)[column]
    return work.sort_index()


def add_action_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach realized enter-now and hindsight best-later training values."""

    work = frame.sort_values(
        EPISODE_KEYS + ["state_t"],
        kind="stable",
    ).copy()
    realized = _numeric(work, "policy_net_return_pct").where(
        work["replay_status"].astype(str).eq("closed")
    )
    work["target_enter_now_pct"] = realized
    work["target_wait_best_later_pct"] = np.nan

    for _, group in work.groupby(EPISODE_KEYS, sort=False):
        idx = group.index
        values = realized.loc[idx].to_numpy(dtype=float)
        later = np.full(len(values), np.nan, dtype=float)
        running_best = np.nan
        for position in range(len(values) - 1, -1, -1):
            later[position] = running_best
            value = values[position]
            if np.isfinite(value):
                running_best = (
                    float(value)
                    if not np.isfinite(running_best)
                    else max(float(running_best), float(value))
                )
        work.loc[idx, "target_wait_best_later_pct"] = later

    return work.sort_index()


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def train_head(
    frame: pd.DataFrame,
    *,
    target_column: str,
    columns: tuple[str, ...],
    seed: int,
) -> ValueHead:
    target = _numeric(frame, target_column)
    valid = target.notna()
    train = frame.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 500:
        raise ValueError(
            f"request 204 insufficient {target_column} support: {len(train)}"
        )
    usable = tuple(
        column
        for column in columns
        if column in train.columns
        and _numeric(train, column).notna().any()
    )
    if not usable:
        raise ValueError("request 204 has no usable causal features")
    low, high = np.quantile(
        y.to_numpy(dtype=float),
        [0.005, 0.995],
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _feature_frame(train, usable),
        y.clip(lower=low, upper=high),
        sample_weight=_day_weights(train),
    )
    return ValueHead(
        model=model,
        columns=usable,
        low=float(low),
        high=float(high),
    )


def predict_head(
    frame: pd.DataFrame,
    head: ValueHead,
) -> np.ndarray:
    return head.model.predict(
        _feature_frame(frame, head.columns)
    )


def fit_controller(
    fit: pd.DataFrame,
    *,
    include_watch_path: bool,
) -> tuple[ValueHead, ValueHead]:
    columns = tuple(
        dict.fromkeys(
            [
                *FEATURES,
                *(
                    WATCH_PATH_FEATURES
                    if include_watch_path
                    else ()
                ),
            ]
        )
    )
    seed_now = (
        PATH_SEED_NOW
        if include_watch_path
        else BASE_SEED_NOW
    )
    seed_wait = (
        PATH_SEED_WAIT
        if include_watch_path
        else BASE_SEED_WAIT
    )
    now_head = train_head(
        fit,
        target_column="target_enter_now_pct",
        columns=columns,
        seed=seed_now,
    )
    wait_head = train_head(
        fit,
        target_column="target_wait_best_later_pct",
        columns=columns,
        seed=seed_wait,
    )
    return now_head, wait_head


def score_controller(
    frame: pd.DataFrame,
    now_head: ValueHead,
    wait_head: ValueHead,
    *,
    prefix: str,
) -> pd.DataFrame:
    scored = frame.copy()
    scored[f"{prefix}_pred_enter_now_pct"] = predict_head(
        scored,
        now_head,
    )
    scored[f"{prefix}_pred_wait_pct"] = predict_head(
        scored,
        wait_head,
    )
    return scored


def run_controller(
    frame: pd.DataFrame,
    *,
    prefix: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Execute the frozen ENTER / WAIT / DROP state machine."""

    chosen: list[pd.Series] = []
    action_rows: list[dict[str, object]] = []
    retries = 0
    entered = 0
    dropped = 0
    no_action = 0

    for key, group in frame.sort_values(
        EPISODE_KEYS + ["state_t"],
        kind="stable",
    ).groupby(EPISODE_KEYS, sort=False):
        rows = list(group.iterrows())
        episode_finished = False
        for position, (_, row) in enumerate(rows):
            now = pd.to_numeric(
                pd.Series(
                    [row.get(f"{prefix}_pred_enter_now_pct")]
                ),
                errors="coerce",
            ).iloc[0]
            wait = pd.to_numeric(
                pd.Series(
                    [row.get(f"{prefix}_pred_wait_pct")]
                ),
                errors="coerce",
            ).iloc[0]
            final = position == len(rows) - 1

            action = "WAIT"
            if final:
                action = (
                    "ENTER_NOW"
                    if pd.notna(now) and float(now) > 0.0
                    else "DROP"
                )
            elif (
                pd.notna(now)
                and pd.notna(wait)
                and float(now) <= 0.0
                and float(wait) <= 0.0
            ):
                action = "DROP"
            elif (
                pd.notna(now)
                and float(now) > 0.0
                and (
                    pd.isna(wait)
                    or float(now) >= float(wait)
                )
            ):
                action = "ENTER_NOW"

            action_rows.append(
                {
                    "trading_day": str(row["trading_day"]),
                    "ticker": str(row["ticker"]),
                    "hot_t": int(row["hot_t"]),
                    "state_t": int(row["state_t"]),
                    "minutes_held": float(row["minutes_held"]),
                    "pred_enter_now_pct": (
                        float(now) if pd.notna(now) else np.nan
                    ),
                    "pred_wait_pct": (
                        float(wait) if pd.notna(wait) else np.nan
                    ),
                    "action": action,
                }
            )

            if action == "DROP":
                dropped += 1
                episode_finished = True
                break
            if action != "ENTER_NOW":
                continue

            if str(row["replay_status"]) == "entry_unavailable":
                retries += 1
                continue

            chosen.append(row)
            entered += 1
            episode_finished = True
            break

        if not episode_finished:
            no_action += 1

    selected = (
        pd.DataFrame(chosen).reset_index(drop=True)
        if chosen
        else frame.iloc[0:0].copy()
    )
    actions = pd.DataFrame(action_rows)
    minute_counts = (
        pd.to_numeric(
            selected.get("minutes_held"),
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    total_entries = int(minute_counts.sum())
    max_minute_share = (
        float(minute_counts.max() / total_entries)
        if total_entries
        else None
    )
    audit = {
        "episodes": int(
            frame.loc[:, EPISODE_KEYS]
            .drop_duplicates()
            .shape[0]
        ),
        "entered_episodes": int(entered),
        "dropped_episodes": int(dropped),
        "no_action_episodes": int(no_action),
        "entry_unavailable_retries": int(retries),
        "mean_chosen_entry_minute": (
            float(
                pd.to_numeric(
                    selected["minutes_held"],
                    errors="coerce",
                ).mean()
            )
            if len(selected)
            else None
        ),
        "chosen_entry_minute_counts": {
            str(int(k)): int(v)
            for k, v in minute_counts.items()
        },
        "max_single_minute_share": max_minute_share,
        "unique_chosen_minutes": int(len(minute_counts)),
        "action_counts": (
            {
                str(k): int(v)
                for k, v in actions["action"]
                .value_counts()
                .sort_index()
                .items()
            }
            if not actions.empty
            else {}
        ),
    }
    return selected, audit


def oracle_best_entry(frame: pd.DataFrame) -> pd.DataFrame:
    """Non-executable ceiling: hindsight best closed runner-policy state."""

    closed = frame.loc[
        frame["replay_status"].astype(str).eq("closed")
        & _numeric(frame, "policy_net_return_pct").notna()
    ].copy()
    if closed.empty:
        return closed
    closed["_value"] = _numeric(
        closed,
        "policy_net_return_pct",
    )
    ordered = closed.sort_values(
        EPISODE_KEYS + ["_value", "state_t"],
        ascending=[True, True, True, False, True],
        kind="stable",
    )
    return (
        ordered.groupby(
            EPISODE_KEYS,
            sort=False,
            as_index=False,
        )
        .head(1)
        .drop(columns=["_value"])
        .reset_index(drop=True)
    )


def matched_improvement(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, object]:
    left = baseline.loc[
        :,
        [*EPISODE_KEYS, "policy_net_return_pct", "replay_status"],
    ].rename(
        columns={
            "policy_net_return_pct": "baseline_return",
            "replay_status": "baseline_status",
        }
    )
    right = candidate.loc[
        :,
        [*EPISODE_KEYS, "policy_net_return_pct", "replay_status"],
    ].rename(
        columns={
            "policy_net_return_pct": "candidate_return",
            "replay_status": "candidate_status",
        }
    )
    merged = left.merge(
        right,
        on=EPISODE_KEYS,
        how="inner",
        validate="one_to_one",
    )
    base = pd.to_numeric(
        merged["baseline_return"],
        errors="coerce",
    )
    cand = pd.to_numeric(
        merged["candidate_return"],
        errors="coerce",
    )
    valid = (
        merged["baseline_status"].astype(str).eq("closed")
        & merged["candidate_status"].astype(str).eq("closed")
        & base.notna()
        & cand.notna()
    )
    matched = merged.loc[valid].copy()
    if matched.empty:
        return {
            "matched_closed_episodes": 0,
            "mean_improvement_pct": None,
            "day_balanced_improvement_pct": None,
        }
    matched["_delta"] = (
        cand.loc[valid].to_numpy(dtype=float)
        - base.loc[valid].to_numpy(dtype=float)
    )
    daily = matched.groupby(
        matched["trading_day"].astype(str),
        sort=True,
    )["_delta"].mean()
    return {
        "matched_closed_episodes": int(len(matched)),
        "mean_improvement_pct": float(
            matched["_delta"].mean()
        ),
        "day_balanced_improvement_pct": float(
            daily.mean()
        ),
        "by_day_improvement_pct": {
            str(k): float(v)
            for k, v in daily.items()
        },
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_path)
    cal_positions = pd.read_parquet(calibration_path)
    scan = pd.read_parquet(history_scan_path)

    fit_days = set(
        fit_positions["trading_day"].astype(str)
    )
    cal_days = set(
        cal_positions["trading_day"].astype(str)
    )
    fit_scan = scan.loc[
        scan["trading_day"].astype(str).isin(fit_days)
    ].copy()
    cal_scan = scan.loc[
        scan["trading_day"].astype(str).isin(cal_days)
    ].copy()

    store = SecondStore()
    _, fit_by_rule = prepare_states(
        fit_positions,
        fit_scan,
        store,
    )
    _, cal_by_rule = prepare_states(
        cal_positions,
        cal_scan,
        store,
    )

    fit = add_action_targets(
        add_watch_path_features(
            fit_by_rule[RULE_NAME]
        )
    )
    cal = add_action_targets(
        add_watch_path_features(
            cal_by_rule[RULE_NAME]
        )
    )

    base_now, base_wait = fit_controller(
        fit,
        include_watch_path=False,
    )
    path_now, path_wait = fit_controller(
        fit,
        include_watch_path=True,
    )

    scored = score_controller(
        cal,
        base_now,
        base_wait,
        prefix="base",
    )
    scored = score_controller(
        scored,
        path_now,
        path_wait,
        prefix="path",
    )

    baseline, baseline_audit = first_admitted_baseline(
        scored
    )
    base_selected, base_audit = run_controller(
        scored,
        prefix="base",
    )
    path_selected, path_audit = run_controller(
        scored,
        prefix="path",
    )
    oracle = oracle_best_entry(scored)

    baseline_summary = _summarize(baseline)
    base_summary = _summarize(base_selected)
    path_summary = _summarize(path_selected)
    oracle_summary = _summarize(oracle)

    matched = matched_improvement(
        baseline,
        path_selected,
    )

    path_day = path_summary[
        "day_balanced_net_return_pct"
    ]
    base_day = base_summary[
        "day_balanced_net_return_pct"
    ]
    baseline_severe = baseline_summary[
        "severe_loss_rate"
    ]
    path_severe = path_summary[
        "severe_loss_rate"
    ]
    ci_low = path_summary[
        "daily_bootstrap"
    ]["ci_low_pct"]
    matched_day = matched[
        "day_balanced_improvement_pct"
    ]

    pass_checks = {
        "at_least_30_admitted": bool(
            len(path_selected) >= MIN_TRADES
        ),
        "closed_coverage_at_least_90pct": bool(
            path_summary["closed_coverage"] is not None
            and float(path_summary["closed_coverage"])
            >= MIN_CLOSED_COVERAGE
        ),
        "positive_day_balanced_return": bool(
            path_day is not None
            and float(path_day) > 0.0
        ),
        "at_least_6_positive_days": bool(
            int(path_summary["positive_days"])
            >= MIN_POSITIVE_DAYS
        ),
        "bootstrap_lower_above_zero": bool(
            ci_low is not None
            and float(ci_low) > 0.0
        ),
        "matched_uplift_at_least_half_point": bool(
            matched_day is not None
            and float(matched_day)
            >= MIN_MATCHED_UPLIFT_PCT
        ),
        "severe_loss_not_worse": bool(
            path_severe is not None
            and baseline_severe is not None
            and float(path_severe)
            <= float(baseline_severe)
        ),
        "not_one_fixed_minute": bool(
            int(path_audit["unique_chosen_minutes"])
            >= 2
            and (
                path_audit["max_single_minute_share"] is not None
                and float(
                    path_audit["max_single_minute_share"]
                ) < 1.0
            )
        ),
        "path_not_worse_than_base_controller": bool(
            path_day is not None
            and base_day is not None
            and float(path_day) >= float(base_day)
        ),
    }
    pass_checks["all_pass"] = all(
        pass_checks.values()
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "fresh_evaluated": False,
        "promotion_eligible": False,
        "rule": RULE_NAME,
        "watch_minutes": [1, 10],
        "watch_path_features": list(
            WATCH_PATH_FEATURES
        ),
        "baseline": {
            "audit": baseline_audit,
            "summary": baseline_summary,
        },
        "base_feature_controller": {
            "audit": base_audit,
            "summary": base_summary,
        },
        "path_controller": {
            "audit": path_audit,
            "summary": path_summary,
            "matched_vs_baseline": matched,
        },
        "oracle_best_entry_non_executable": {
            "summary": oracle_summary,
        },
        "development_checks": pass_checks,
        "development_signal_pass": bool(
            pass_checks["all_pass"]
        ),
        "halt_feed_by_day": dict(
            sorted(store.halt_status.items())
        ),
        "second_client_stats": (
            store.client.stats.to_dict()
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--history-scan",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
