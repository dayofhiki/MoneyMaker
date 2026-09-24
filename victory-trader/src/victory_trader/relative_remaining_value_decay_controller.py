"""Request 188: relative remaining-value decay controller.

Uses only within-position changes in the frozen Request-178 consensus
remaining-value score. Absolute score levels and thresholds are ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS

REQUEST_ID = 188
MAX_HOLD_MINUTES = 30
BOOTSTRAP_SEED = 20261088
BOOTSTRAP_SAMPLES = 10_000
SEVERE_LOSS_PCT = -5.0
MIN_START_COVERAGE = 0.85
MIN_COMPLETION_COVERAGE = 0.90


def _safe_spearman(
    actual: pd.Series,
    predicted: pd.Series,
) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna()
    if int(valid.sum()) < 20:
        return None
    value = a.loc[valid].corr(p.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def add_trend_columns(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored.copy()
    frame["_minute"] = pd.to_numeric(
        frame["minutes_held"], errors="coerce"
    )
    frame = frame.loc[
        frame["_minute"].notna()
        & frame["_minute"].mod(1).eq(0)
    ].copy()
    frame["_minute"] = frame["_minute"].astype(int)
    frame = frame.sort_values(
        EPISODE_KEYS + ["_minute"],
        kind="stable",
    ).reset_index(drop=True)

    grouped = frame.groupby(EPISODE_KEYS, sort=False)
    frame["previous_consensus_hurdle_ev_pct"] = grouped[
        "consensus_hurdle_ev_pct"
    ].shift()
    frame["score_delta_pct"] = (
        pd.to_numeric(
            frame["consensus_hurdle_ev_pct"], errors="coerce"
        )
        - pd.to_numeric(
            frame["previous_consensus_hurdle_ev_pct"], errors="coerce"
        )
    )
    frame["previous_excess_remaining_option_value_pct"] = grouped[
        "excess_remaining_option_value_pct"
    ].shift()
    frame["actual_excess_value_delta_pct"] = (
        pd.to_numeric(
            frame["excess_remaining_option_value_pct"], errors="coerce"
        )
        - pd.to_numeric(
            frame["previous_excess_remaining_option_value_pct"],
            errors="coerce",
        )
    )
    return frame


def trend_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    delta = pd.to_numeric(frame["score_delta_pct"], errors="coerce")
    actual_delta = pd.to_numeric(
        frame["actual_excess_value_delta_pct"], errors="coerce"
    )
    hold_adv = pd.to_numeric(
        frame["hold_advantage_1m_pct"], errors="coerce"
    )
    remaining = pd.to_numeric(
        frame["remaining_option_value_pct"], errors="coerce"
    )
    valid = delta.notna()
    rising = valid & delta.gt(0)
    nonrising = valid & delta.le(0)

    def _mean(series: pd.Series, mask: pd.Series) -> float | None:
        values = pd.to_numeric(series.loc[mask], errors="coerce").dropna()
        return float(values.mean()) if len(values) else None

    by_day: dict[str, object] = {}
    for day in FRESH_DAYS:
        mask = frame["trading_day"].astype(str).eq(day)
        d = delta.loc[mask]
        ad = actual_delta.loc[mask]
        local_valid = d.notna()
        local_rising = local_valid & d.gt(0)
        local_nonrising = local_valid & d.le(0)
        h = hold_adv.loc[mask]
        rem = remaining.loc[mask]
        by_day[day] = {
            "pairs": int(local_valid.sum()),
            "delta_spearman": _safe_spearman(
                ad.loc[local_valid],
                d.loc[local_valid],
            ),
            "rising_hold_advantage_mean_pct": _mean(h, local_rising),
            "nonrising_hold_advantage_mean_pct": _mean(
                h, local_nonrising
            ),
            "rising_remaining_value_mean_pct": _mean(rem, local_rising),
            "nonrising_remaining_value_mean_pct": _mean(
                rem, local_nonrising
            ),
        }

    return {
        "pairs": int(valid.sum()),
        "score_delta_vs_actual_excess_delta_spearman": _safe_spearman(
            actual_delta.loc[valid],
            delta.loc[valid],
        ),
        "rising_rows": int(rising.sum()),
        "nonrising_rows": int(nonrising.sum()),
        "rising_hold_advantage_mean_pct": _mean(hold_adv, rising),
        "nonrising_hold_advantage_mean_pct": _mean(
            hold_adv, nonrising
        ),
        "rising_remaining_value_mean_pct": _mean(remaining, rising),
        "nonrising_remaining_value_mean_pct": _mean(
            remaining, nonrising
        ),
        "by_day": by_day,
    }


def build_trajectories(
    frame: pd.DataFrame,
    *,
    policy: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    total = int(
        frame.loc[:, EPISODE_KEYS].drop_duplicates().shape[0]
    )
    records: list[dict[str, object]] = []

    for _, group in frame.groupby(EPISODE_KEYS, sort=False):
        by_minute = {
            int(row["_minute"]): row
            for _, row in group.sort_values(
                "_minute", kind="stable"
            ).iterrows()
        }
        first = by_minute.get(1)
        if first is None:
            continue
        first_exit = pd.to_numeric(
            pd.Series([first.get("exit_now_base_return_pct")]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(first_exit):
            continue

        if policy == "minute1_exit":
            records.append(
                {
                    "trading_day": str(first["trading_day"]),
                    "ticker": str(first["ticker"]).upper(),
                    "hot_t": int(first["hot_t"]),
                    "policy": policy,
                    "status": "completed",
                    "exit_reason": "minute1_exit",
                    "base_net_return_pct": float(first_exit),
                    "minutes_held": 1.0,
                }
            )
            continue

        previous_score = pd.to_numeric(
            pd.Series([first.get("consensus_hurdle_ev_pct")]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(previous_score):
            records.append(
                {
                    "trading_day": str(first["trading_day"]),
                    "ticker": str(first["ticker"]).upper(),
                    "hot_t": int(first["hot_t"]),
                    "policy": policy,
                    "status": "completed",
                    "exit_reason": "unscoreable_minute1_exit",
                    "base_net_return_pct": float(first_exit),
                    "minutes_held": 1.0,
                }
            )
            continue

        minute = 1
        realized = np.nan
        status = "unresolved"
        reason = "unknown"
        exit_minute = np.nan

        while minute < MAX_HOLD_MINUTES:
            row = by_minute.get(minute)
            if row is None:
                status = "unresolved"
                reason = "missing_reached_state"
                break

            current_exit = pd.to_numeric(
                pd.Series([row.get("exit_now_base_return_pct")]),
                errors="coerce",
            ).iloc[0]
            current_score = pd.to_numeric(
                pd.Series([row.get("consensus_hurdle_ev_pct")]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(current_exit):
                status = "unresolved"
                reason = "missing_current_exit"
                break

            if minute >= 2:
                if (
                    pd.isna(current_score)
                    or not np.isfinite(float(current_score))
                ):
                    status = "completed"
                    reason = "unscoreable_exit"
                    realized = float(current_exit)
                    exit_minute = float(minute)
                    break
                if float(current_score) <= float(previous_score):
                    status = "completed"
                    reason = "relative_value_decay_exit"
                    realized = float(current_exit)
                    exit_minute = float(minute)
                    break

            if minute == MAX_HOLD_MINUTES - 1:
                next_return = pd.to_numeric(
                    pd.Series(
                        [row.get("next_minute_base_return_pct")]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(next_return):
                    status = "completed"
                    reason = "forced_30m_cap"
                    realized = float(next_return)
                    exit_minute = float(MAX_HOLD_MINUTES)
                else:
                    status = "unresolved"
                    reason = "missing_forced_cap_exit"
                break

            next_row = by_minute.get(minute + 1)
            if next_row is None:
                next_return = pd.to_numeric(
                    pd.Series(
                        [row.get("next_minute_base_return_pct")]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(next_return):
                    status = "completed"
                    reason = "missing_state_next_minute_exit"
                    realized = float(next_return)
                    exit_minute = float(minute + 1)
                else:
                    status = "unresolved"
                    reason = "missing_state_and_exit"
                break

            if pd.notna(current_score):
                previous_score = float(current_score)
            minute += 1

        records.append(
            {
                "trading_day": str(first["trading_day"]),
                "ticker": str(first["ticker"]).upper(),
                "hot_t": int(first["hot_t"]),
                "policy": policy,
                "status": status,
                "exit_reason": reason,
                "base_net_return_pct": (
                    float(realized) if pd.notna(realized) else np.nan
                ),
                "minutes_held": exit_minute,
            }
        )

    trajectories = pd.DataFrame(records)
    started = int(len(trajectories))
    return trajectories, {
        "total_position_episodes": total,
        "started_episodes": started,
        "start_state_coverage": (
            float(started / total) if total else None
        ),
    }


def _bootstrap_daily(
    frame: pd.DataFrame,
    column: str,
    seed: int,
) -> dict[str, object]:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna()
    daily = (
        pd.DataFrame(
            {
                "day": frame.loc[valid, "trading_day"].astype(str),
                "value": values.loc[valid].to_numpy(dtype=float),
            }
        )
        .groupby("day", sort=True)["value"]
        .mean()
        .to_numpy(dtype=float)
    )
    if len(daily) < 2:
        return {
            "days": int(len(daily)),
            "samples": BOOTSTRAP_SAMPLES,
            "mean_pct": float(daily.mean()) if len(daily) else None,
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(
        0,
        len(daily),
        size=(BOOTSTRAP_SAMPLES, len(daily)),
    )
    draws = daily[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(daily)),
        "samples": BOOTSTRAP_SAMPLES,
        "mean_pct": float(daily.mean()),
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
    }


def summarize(
    trajectories: pd.DataFrame,
    policy: str,
) -> dict[str, object]:
    completed = trajectories.loc[
        trajectories["status"].eq("completed")
    ].copy()
    values = pd.to_numeric(
        completed["base_net_return_pct"], errors="coerce"
    )
    valid = values.notna()
    completed = completed.loc[valid].copy()
    values = values.loc[valid]
    held = pd.to_numeric(
        completed["minutes_held"], errors="coerce"
    )
    return {
        "policy": policy,
        "started": int(len(trajectories)),
        "completed": int(len(completed)),
        "completion_coverage": (
            float(len(completed) / len(trajectories))
            if len(trajectories)
            else None
        ),
        "base_mean_pct": float(values.mean()) if len(values) else None,
        "day_balanced_base_mean_pct": (
            _bootstrap_daily(
                completed,
                "base_net_return_pct",
                BOOTSTRAP_SEED,
            )["mean_pct"]
            if len(completed)
            else None
        ),
        "base_median_pct": (
            float(values.median()) if len(values) else None
        ),
        "base_p05_pct": (
            float(values.quantile(0.05)) if len(values) else None
        ),
        "positive_rate": (
            float(values.gt(0).mean()) if len(values) else None
        ),
        "severe_loss_rate": (
            float(values.le(SEVERE_LOSS_PCT).mean())
            if len(values)
            else None
        ),
        "mean_hold_minutes": (
            float(held.mean()) if held.notna().any() else None
        ),
        "median_hold_minutes": (
            float(held.median()) if held.notna().any() else None
        ),
        "p90_hold_minutes": (
            float(held.quantile(0.90)) if held.notna().any() else None
        ),
        "exit_reasons": {
            str(key): int(value)
            for key, value in trajectories["exit_reason"]
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "day_bootstrap": _bootstrap_daily(
            completed,
            "base_net_return_pct",
            BOOTSTRAP_SEED,
        ),
    }


def matched_difference(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
) -> dict[str, object]:
    left = candidate.loc[
        candidate["status"].eq("completed"),
        EPISODE_KEYS + ["base_net_return_pct"],
    ].copy()
    right = comparator.loc[
        comparator["status"].eq("completed"),
        EPISODE_KEYS + ["base_net_return_pct"],
    ].copy()
    merged = left.merge(
        right,
        on=EPISODE_KEYS,
        suffixes=("_candidate", "_comparator"),
        validate="one_to_one",
    )
    merged["difference_pct"] = (
        pd.to_numeric(
            merged["base_net_return_pct_candidate"],
            errors="coerce",
        )
        - pd.to_numeric(
            merged["base_net_return_pct_comparator"],
            errors="coerce",
        )
    )
    bootstrap = _bootstrap_daily(
        merged,
        "difference_pct",
        BOOTSTRAP_SEED + 1,
    )
    return {
        "matched_episodes": int(len(merged)),
        "mean_difference_pct": (
            float(merged["difference_pct"].mean())
            if len(merged)
            else None
        ),
        "day_balanced_difference_pct": bootstrap["mean_pct"],
        "bootstrap": bootstrap,
    }


def evaluate(
    scored_path: Path,
    output_path: Path,
    trajectories_output_path: Path,
) -> int:
    scored = pd.read_parquet(scored_path)
    found = sorted(scored["trading_day"].astype(str).unique())
    if found != list(FRESH_DAYS):
        raise ValueError(f"request 188 expected {FRESH_DAYS}, found {found}")

    frame = add_trend_columns(scored)
    trend = trend_diagnostics(frame)
    candidate, candidate_coverage = build_trajectories(
        frame,
        policy="relative_value_decay",
    )
    comparator, comparator_coverage = build_trajectories(
        frame,
        policy="minute1_exit",
    )
    candidate_summary = summarize(
        candidate, "relative_value_decay"
    )
    comparator_summary = summarize(
        comparator, "minute1_exit"
    )
    difference = matched_difference(candidate, comparator)

    c_day = candidate_summary["day_balanced_base_mean_pct"]
    c_boot_low = candidate_summary["day_bootstrap"]["ci_low_pct"]
    diff_day = difference["day_balanced_difference_pct"]
    diff_low = difference["bootstrap"]["ci_low_pct"]
    c_severe = candidate_summary["severe_loss_rate"]
    b_severe = comparator_summary["severe_loss_rate"]
    rising_remaining = trend["rising_remaining_value_mean_pct"]
    nonrising_remaining = trend["nonrising_remaining_value_mean_pct"]
    delta_spearman = trend[
        "score_delta_vs_actual_excess_delta_spearman"
    ]

    gate = bool(
        candidate_coverage["start_state_coverage"] is not None
        and float(candidate_coverage["start_state_coverage"])
        >= MIN_START_COVERAGE
        and candidate_summary["completion_coverage"] is not None
        and float(candidate_summary["completion_coverage"])
        >= MIN_COMPLETION_COVERAGE
        and delta_spearman is not None
        and float(delta_spearman) > 0
        and rising_remaining is not None
        and nonrising_remaining is not None
        and float(nonrising_remaining) < float(rising_remaining)
        and c_day is not None
        and float(c_day) > 0
        and c_boot_low is not None
        and float(c_boot_low) > 0
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
        and c_severe is not None
        and b_severe is not None
        and float(c_severe) <= float(b_severe)
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "absolute_score_threshold_used": False,
        "trend_diagnostics": trend,
        "candidate_coverage": candidate_coverage,
        "comparator_coverage": comparator_coverage,
        "candidate": candidate_summary,
        "comparator": comparator_summary,
        "matched_candidate_minus_minute1": difference,
        "frozen_gate": {
            "min_start_coverage": MIN_START_COVERAGE,
            "min_completion_coverage": MIN_COMPLETION_COVERAGE,
            "delta_spearman_must_be_positive": True,
            "nonrising_remaining_value_must_be_lower": True,
            "candidate_day_balanced_base_must_be_positive": True,
            "candidate_bootstrap_low_must_be_positive": True,
            "matched_difference_must_be_positive": True,
            "matched_bootstrap_low_must_be_positive": True,
            "severe_loss_no_worse_than_minute1": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.concat([candidate, comparator], ignore_index=True).to_parquet(
        trajectories_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectories-output", type=Path, required=True
    )
    args = parser.parse_args()
    return evaluate(
        args.scored,
        args.output,
        args.trajectories_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
