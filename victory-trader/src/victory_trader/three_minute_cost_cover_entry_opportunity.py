"""Request 185: three-minute cost-cover entry opportunity bridge.

The target is a short-horizon hindsight opportunity label, not an executable
exit rule. The three-minute horizon is inherited from the established runner
attention horizon rather than selected from fresh outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .decomposed_cost_aware_entry_surplus import (
    EPISODE_KEYS,
    FRESH_DAYS,
    build_historical_states,
    predict_component,
    train_component,
)
from .selected_hot_position_value_observability import _base_return

REQUEST_ID = 185
ENTER_SEED = 20261088
WAIT_SEED = 20261089
BOOTSTRAP_SEED = 20261085
BOOTSTRAP_SAMPLES = 10_000
MIN_ENTRY_COVERAGE = 0.90
MIN_SECOND_COVERAGE = 0.90
MIN_SPEARMAN = 0.10
MIN_SELECTED_RATE = 0.10
MAX_SELECTED_RATE = 0.70
MIN_SELECTED_POSITIVE_RATE = 0.60
MIN_UPLIFT_PCT = 0.25
MIN_POSITIVE_DAYS = 4


def build_three_minute_labels(position_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, group in position_rows.groupby(EPISODE_KEYS, sort=False):
        work = group.copy()
        work["_minute"] = pd.to_numeric(
            work["minutes_held"], errors="coerce"
        )
        work = work.loc[
            work["_minute"].notna() & work["_minute"].mod(1).eq(0)
        ].copy()
        work["_minute"] = work["_minute"].astype(int)
        by_minute = {
            int(row["_minute"]): row
            for _, row in work.sort_values("_minute", kind="stable").iterrows()
        }
        row1 = by_minute.get(1)
        if row1 is None:
            continue

        entry = pd.to_numeric(
            pd.Series([row1.get("entry_open")]), errors="coerce"
        ).iloc[0]
        minute1_open = pd.to_numeric(
            pd.Series([row1.get("exit_reference_open")]), errors="coerce"
        ).iloc[0]

        enter_values: list[tuple[int, float]] = []
        if pd.notna(entry) and float(entry) > 0:
            for minute in (1, 2, 3):
                row = by_minute.get(minute)
                if row is None:
                    continue
                exit_price = pd.to_numeric(
                    pd.Series([row.get("exit_reference_open")]),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(exit_price) and float(exit_price) > 0:
                    value = _base_return(float(entry), float(exit_price))
                    if pd.notna(value):
                        enter_values.append((minute, float(value)))

        wait_values: list[tuple[int, float]] = []
        if pd.notna(minute1_open) and float(minute1_open) > 0:
            for minute in (2, 3, 4):
                row = by_minute.get(minute)
                if row is None:
                    continue
                exit_price = pd.to_numeric(
                    pd.Series([row.get("exit_reference_open")]),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(exit_price) and float(exit_price) > 0:
                    value = _base_return(
                        float(minute1_open), float(exit_price)
                    )
                    if pd.notna(value):
                        wait_values.append((minute - 1, float(value)))

        enter_best = (
            max(enter_values, key=lambda item: item[1])
            if enter_values
            else None
        )
        wait_best = (
            max(wait_values, key=lambda item: item[1])
            if wait_values
            else None
        )
        records.append(
            {
                "trading_day": str(row1["trading_day"]),
                "ticker": str(row1["ticker"]).upper(),
                "hot_t": int(row1["hot_t"]),
                "entry_open": (
                    float(entry) if pd.notna(entry) else np.nan
                ),
                "enter_3m_base_pct": (
                    float(enter_best[1]) if enter_best is not None else np.nan
                ),
                "enter_3m_best_minute": (
                    int(enter_best[0]) if enter_best is not None else np.nan
                ),
                "wait_3m_base_pct": (
                    float(wait_best[1]) if wait_best is not None else np.nan
                ),
                "wait_3m_best_minute": (
                    int(wait_best[0]) if wait_best is not None else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def _safe_spearman(
    actual: pd.Series,
    predicted: pd.Series,
) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = y.notna() & p.notna()
    if int(valid.sum()) < 20:
        return None
    value = y.loc[valid].corr(p.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def _day_balanced(
    frame: pd.DataFrame,
    column: str,
) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return None
    daily = (
        pd.DataFrame(
            {
                "day": frame.loc[valid, "trading_day"].astype(str),
                "value": values.loc[valid].to_numpy(dtype=float),
            }
        )
        .groupby("day", sort=True)["value"]
        .mean()
    )
    return float(daily.mean()) if len(daily) else None


def _bootstrap_selected(
    selected: pd.DataFrame,
) -> dict[str, float | int | None]:
    value = pd.to_numeric(
        selected["selected_actual_opportunity_pct"], errors="coerce"
    )
    valid = value.notna()
    daily = (
        pd.DataFrame(
            {
                "day": selected.loc[valid, "trading_day"].astype(str),
                "value": value.loc[valid].to_numpy(dtype=float),
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
    rng = np.random.default_rng(BOOTSTRAP_SEED)
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


def choose_action(frame: pd.DataFrame) -> pd.Series:
    enter = pd.to_numeric(
        frame["predicted_enter_3m_base_pct"], errors="coerce"
    )
    wait = pd.to_numeric(
        frame["predicted_wait_3m_base_pct"], errors="coerce"
    )
    action = pd.Series("ABSTAIN", index=frame.index, dtype=object)
    positive = pd.concat([enter, wait], axis=1).max(axis=1).gt(0)
    wait_wins = positive & wait.ge(enter)
    enter_wins = positive & enter.gt(wait)
    action.loc[wait_wins] = "WAIT_1M"
    action.loc[enter_wins] = "ENTER_NOW"
    return action


def _group_summary(frame: pd.DataFrame, column: str) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, group in frame.groupby(column, sort=True, observed=True):
        selected = group.loc[group["action"].ne("ABSTAIN")]
        values = pd.to_numeric(
            selected["selected_actual_opportunity_pct"], errors="coerce"
        )
        output[str(key)] = {
            "rows": int(len(group)),
            "selected_rows": int(len(selected)),
            "selected_rate": (
                float(len(selected) / len(group)) if len(group) else None
            ),
            "selected_mean_pct": (
                float(values.mean()) if values.notna().any() else None
            ),
            "selected_positive_rate": (
                float(values.dropna().gt(0).mean())
                if values.notna().any()
                else None
            ),
        }
    return output


def evaluate(
    history_first_hot_path: Path,
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_entry_states_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    first_hot = pd.read_parquet(history_first_hot_path)
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(calibration_positions_path)

    fit_labels = build_three_minute_labels(fit_positions)
    cal_labels = build_three_minute_labels(calibration_positions)
    fit = build_historical_states(first_hot, fit_labels)
    calibration = build_historical_states(first_hot, cal_labels)

    enter_model = train_component(
        fit,
        calibration,
        target_column="enter_3m_base_pct",
        seed=ENTER_SEED,
    )
    wait_model = train_component(
        fit,
        calibration,
        target_column="wait_3m_base_pct",
        seed=WAIT_SEED,
    )

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(position_paths) != len(FRESH_DAYS):
        raise ValueError("request 185 requires complete request 178 positions")
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_labels = build_three_minute_labels(fresh_positions)

    fresh_features = pd.read_parquet(fresh_entry_states_path)
    keep = [
        column
        for column in fresh_features.columns
        if column not in {
            "entry_open",
            "enter_gross_pct",
            "enter_drag_pct",
            "enter_now_base_pct",
            "wait_gross_pct",
            "wait_drag_pct",
            "wait_1m_base_pct",
        }
    ]
    fresh = fresh_features.loc[:, keep].merge(
        fresh_labels,
        on=EPISODE_KEYS,
        how="left",
        validate="one_to_one",
    )

    fresh["predicted_enter_3m_base_pct"] = predict_component(
        fresh, enter_model
    )
    fresh["predicted_wait_3m_base_pct"] = predict_component(
        fresh, wait_model
    )
    fresh["action"] = choose_action(fresh)

    enter_actual = pd.to_numeric(
        fresh["enter_3m_base_pct"], errors="coerce"
    )
    wait_actual = pd.to_numeric(
        fresh["wait_3m_base_pct"], errors="coerce"
    )
    selected_actual = pd.Series(np.nan, index=fresh.index, dtype=float)
    selected_actual.loc[fresh["action"].eq("ENTER_NOW")] = (
        enter_actual.loc[fresh["action"].eq("ENTER_NOW")]
    )
    selected_actual.loc[fresh["action"].eq("WAIT_1M")] = (
        wait_actual.loc[fresh["action"].eq("WAIT_1M")]
    )
    fresh["selected_actual_opportunity_pct"] = selected_actual
    fresh["all_buy_best_3m_opportunity_pct"] = pd.concat(
        [enter_actual, wait_actual], axis=1
    ).max(axis=1, skipna=True)

    covered = fresh["entry_state_available"].fillna(False)
    second = pd.to_numeric(
        fresh.get("active_seconds_60"), errors="coerce"
    ).fillna(0).gt(0)
    entry_coverage = float(covered.mean()) if len(fresh) else None
    second_coverage = (
        float(second.loc[covered].mean()) if int(covered.sum()) else None
    )

    selected = fresh.loc[
        covered
        & fresh["action"].ne("ABSTAIN")
        & fresh["selected_actual_opportunity_pct"].notna()
    ].copy()
    selected_values = pd.to_numeric(
        selected["selected_actual_opportunity_pct"], errors="coerce"
    )
    selected_rate = (
        float(len(selected) / int(covered.sum()))
        if int(covered.sum())
        else None
    )
    selected_mean = (
        float(selected_values.mean()) if len(selected_values) else None
    )
    selected_day = _day_balanced(
        selected, "selected_actual_opportunity_pct"
    )
    selected_positive = (
        float(selected_values.gt(0).mean())
        if len(selected_values)
        else None
    )

    all_values = pd.to_numeric(
        fresh.loc[covered, "all_buy_best_3m_opportunity_pct"],
        errors="coerce",
    )
    all_mean = float(all_values.mean()) if all_values.notna().any() else None
    all_day = _day_balanced(
        fresh.loc[covered], "all_buy_best_3m_opportunity_pct"
    )
    uplift = (
        float(selected_mean - all_mean)
        if selected_mean is not None and all_mean is not None
        else None
    )
    bootstrap = _bootstrap_selected(selected)

    by_day: dict[str, object] = {}
    positive_days = 0
    for day in FRESH_DAYS:
        part = fresh.loc[
            fresh["trading_day"].astype(str).eq(day)
        ].copy()
        chosen = part.loc[
            part["entry_state_available"].fillna(False)
            & part["action"].ne("ABSTAIN")
            & part["selected_actual_opportunity_pct"].notna()
        ]
        values = pd.to_numeric(
            chosen["selected_actual_opportunity_pct"], errors="coerce"
        )
        mean = float(values.mean()) if len(values) else None
        good = bool(mean is not None and mean > 0)
        positive_days += int(good)
        by_day[day] = {
            "rows": int(len(part)),
            "selected_rows": int(len(chosen)),
            "selected_mean_pct": mean,
            "selected_positive_rate": (
                float(values.gt(0).mean()) if len(values) else None
            ),
            "positive_selected_opportunity_day": good,
        }

    spearman = {
        "enter_3m": _safe_spearman(
            fresh["enter_3m_base_pct"],
            fresh["predicted_enter_3m_base_pct"],
        ),
        "wait_3m": _safe_spearman(
            fresh["wait_3m_base_pct"],
            fresh["predicted_wait_3m_base_pct"],
        ),
    }

    action_counts = {
        str(key): int(value)
        for key, value in fresh.loc[covered, "action"]
        .value_counts()
        .to_dict()
        .items()
    }

    current_price = pd.to_numeric(
        fresh.get("current_price"), errors="coerce"
    )
    fresh["entry_price_bin"] = pd.cut(
        current_price,
        bins=(-np.inf, 1.0, 2.0, 5.0, 10.0, np.inf),
        labels=("<1", "1-2", "2-5", "5-10", ">=10"),
        right=False,
    ).astype(str)

    gate = bool(
        entry_coverage is not None
        and entry_coverage >= MIN_ENTRY_COVERAGE
        and second_coverage is not None
        and second_coverage >= MIN_SECOND_COVERAGE
        and spearman["enter_3m"] is not None
        and float(spearman["enter_3m"]) >= MIN_SPEARMAN
        and spearman["wait_3m"] is not None
        and float(spearman["wait_3m"]) >= MIN_SPEARMAN
        and selected_rate is not None
        and MIN_SELECTED_RATE <= selected_rate <= MAX_SELECTED_RATE
        and selected_day is not None
        and selected_day > 0
        and selected_positive is not None
        and selected_positive >= MIN_SELECTED_POSITIVE_RATE
        and uplift is not None
        and uplift >= MIN_UPLIFT_PCT
        and bootstrap["ci_low_pct"] is not None
        and float(bootstrap["ci_low_pct"]) > 0
        and positive_days >= MIN_POSITIVE_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "target_semantics": (
            "hindsight best BASE opportunity within inherited 3-minute "
            "runner horizon; not executable return"
        ),
        "feature_count_enter": len(enter_model.columns),
        "feature_count_wait": len(wait_model.columns),
        "historical_entry_state_coverage": {
            "fit": float(fit["entry_state_available"].mean()) if len(fit) else None,
            "calibration": (
                float(calibration["entry_state_available"].mean())
                if len(calibration)
                else None
            ),
        },
        "fresh_entry_state_coverage": entry_coverage,
        "fresh_second_feature_coverage": second_coverage,
        "opportunity_spearman": spearman,
        "action_counts": action_counts,
        "selected_rate": selected_rate,
        "selected_opportunity_mean_pct": selected_mean,
        "selected_day_balanced_opportunity_pct": selected_day,
        "selected_positive_opportunity_rate": selected_positive,
        "all_buy_best_opportunity_mean_pct": all_mean,
        "all_buy_best_day_balanced_opportunity_pct": all_day,
        "selected_minus_all_buy_mean_uplift_pct": uplift,
        "selected_day_bootstrap": bootstrap,
        "positive_selected_opportunity_days": int(positive_days),
        "by_day": by_day,
        "by_action": _group_summary(fresh, "action"),
        "by_entry_price_bin": _group_summary(fresh, "entry_price_bin"),
        "model_diagnostics": {
            "enter": {
                "offset_pct": enter_model.offset,
                "winsor_low_pct": enter_model.winsor_low,
                "winsor_high_pct": enter_model.winsor_high,
            },
            "wait": {
                "offset_pct": wait_model.offset,
                "winsor_low_pct": wait_model.winsor_low,
                "winsor_high_pct": wait_model.winsor_high,
            },
        },
        "frozen_gate": {
            "min_entry_coverage": MIN_ENTRY_COVERAGE,
            "min_second_coverage": MIN_SECOND_COVERAGE,
            "min_spearman_each": MIN_SPEARMAN,
            "min_selected_rate": MIN_SELECTED_RATE,
            "max_selected_rate": MAX_SELECTED_RATE,
            "min_selected_positive_rate": MIN_SELECTED_POSITIVE_RATE,
            "min_selected_minus_all_buy_uplift_pct": MIN_UPLIFT_PCT,
            "bootstrap_low_must_be_positive": True,
            "min_positive_days": MIN_POSITIVE_DAYS,
        },
        "development_bridge_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fresh.to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-first-hot", type=Path, required=True)
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument("--calibration-positions", type=Path, required=True)
    parser.add_argument("--fresh-entry-states", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.history_first_hot,
        args.fit_positions,
        args.calibration_positions,
        args.fresh_entry_states,
        args.fresh_dir,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
