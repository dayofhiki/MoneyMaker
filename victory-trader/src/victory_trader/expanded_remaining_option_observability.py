from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from .execution_costs import modeled_buy_fill, modeled_sell_fill
from .expanded_opportunity_shared_q import (
    predict_opportunity,
    train_opportunity_model,
)
from .expanded_path_aware_continuation import (
    BASE_SCENARIO,
    KEYS,
    MAX_HOLD_MINUTES,
    MINUTE_MS,
    build_continuation_rows,
)
from .expanded_path_transition_continuation import transition_feature_frame
from .expanded_supply_hurdle_ev import load_fold

BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 20261045


def _positive(value: object) -> bool:
    return bool(
        pd.notna(value)
        and np.isfinite(float(value))
        and float(value) > 0
    )


def _month_of(frame: pd.DataFrame) -> pd.Series:
    return frame["trading_day"].astype(str).str.slice(0, 7)


def _normalize_state(state: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + ["t", "o"])
    if not required.issubset(state.columns):
        raise ValueError("state panel requires trading_day/ticker/t/o")
    result = state.copy()
    result["trading_day"] = result["trading_day"].astype(str)
    result["ticker"] = result["ticker"].astype(str)
    result["t"] = pd.to_numeric(result["t"], errors="coerce")
    result["o"] = pd.to_numeric(result["o"], errors="coerce")
    if result.duplicated(KEYS + ["t"]).any():
        raise ValueError("state panel must be unique by ticker-day/t")
    return result.sort_values(KEYS + ["t"], kind="stable").reset_index(drop=True)


def _state_groups(
    state: pd.DataFrame,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    result: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (day, ticker), group in _normalize_state(state).groupby(KEYS, sort=False):
        result[(str(day), str(ticker))] = (
            pd.to_numeric(group["t"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(group["o"], errors="coerce").to_numpy(dtype=float),
        )
    return result


def _exact_open(
    group: tuple[np.ndarray, np.ndarray] | None,
    timestamp: int,
) -> float:
    if group is None:
        return np.nan
    times, opens = group
    where = np.flatnonzero(times == timestamp)
    if not len(where):
        return np.nan
    value = opens[int(where[0])]
    return float(value) if _positive(value) else np.nan


def _observed_future_opens(
    group: tuple[np.ndarray, np.ndarray] | None,
    *,
    after_t: int,
    through_t: int,
) -> tuple[np.ndarray, np.ndarray]:
    if group is None:
        return np.array([], dtype=float), np.array([], dtype=float)
    times, opens = group
    mask = (
        (times > after_t)
        & (times <= through_t)
        & np.isfinite(opens)
        & (opens > 0)
    )
    return times[mask], opens[mask]


def _base_return(entry_open: float, exit_open: float) -> float:
    if not (_positive(entry_open) and _positive(exit_open)):
        return np.nan
    buy = modeled_buy_fill(float(entry_open), BASE_SCENARIO)
    sell = modeled_sell_fill(float(exit_open), BASE_SCENARIO)
    sell *= 1.0 - BASE_SCENARIO.sell_fee_bps / 10_000.0
    return (sell / buy - 1.0) * 100.0


def oracle_entry_ceiling(
    anchors: pd.DataFrame,
    state: pd.DataFrame,
    month: str,
) -> pd.DataFrame:
    groups = _state_groups(state)
    selected = anchors.loc[
        pd.to_numeric(
            anchors["opportunity_probability"],
            errors="coerce",
        ).gt(0.5)
    ].copy()
    selected["trading_day"] = selected["trading_day"].astype(str)
    selected["ticker"] = selected["ticker"].astype(str)
    selected["t"] = pd.to_numeric(selected["t"], errors="coerce")

    records: list[dict[str, object]] = []
    for row in selected.sort_values(
        ["trading_day", "t", "ticker"],
        kind="stable",
    ).to_dict("records"):
        day = str(row["trading_day"])
        ticker = str(row["ticker"])
        group = groups.get((day, ticker))
        anchor_t = int(row["t"])
        entry_t = anchor_t + MINUTE_MS
        entry_open = _exact_open(group, entry_t)
        record: dict[str, object] = {
            "month": month,
            "trading_day": day,
            "ticker": ticker,
            "anchor_t": anchor_t,
            "entry_t": entry_t,
            "entry_open": entry_open,
            "status": "missing_entry",
            "oracle_best_base_pct": np.nan,
            "oracle_best_holding_min": np.nan,
            "observed_exit_count": 0,
            "exit_slot_count": MAX_HOLD_MINUTES,
            "exit_observed_fraction": 0.0,
        }
        if not _positive(entry_open):
            records.append(record)
            continue

        times, opens = _observed_future_opens(
            group,
            after_t=entry_t,
            through_t=entry_t + MAX_HOLD_MINUTES * MINUTE_MS,
        )
        record["observed_exit_count"] = int(len(opens))
        record["exit_observed_fraction"] = float(
            len(opens) / MAX_HOLD_MINUTES
        )
        if not len(opens):
            record["status"] = "unresolved_no_future_open"
            records.append(record)
            continue

        returns = np.array(
            [_base_return(float(entry_open), float(value)) for value in opens],
            dtype=float,
        )
        best = int(np.nanargmax(returns))
        record.update(
            status="evaluated",
            oracle_best_base_pct=float(returns[best]),
            oracle_best_holding_min=float(
                (times[best] - entry_t) / MINUTE_MS
            ),
        )
        records.append(record)
    return pd.DataFrame(records)


def add_remaining_option_labels(
    rows: pd.DataFrame,
    state_by_month: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result_parts: list[pd.DataFrame] = []
    for month, month_rows in rows.groupby(_month_of(rows), sort=True):
        if month not in state_by_month:
            raise ValueError(f"missing state panel for {month}")
        groups = _state_groups(state_by_month[month])
        frame = month_rows.copy()
        option_values: list[float] = []
        best_future_values: list[float] = []
        current_values: list[float] = []
        observed_counts: list[int] = []
        slot_counts: list[int] = []
        observed_fractions: list[float] = []

        for row in frame.to_dict("records"):
            key = (str(row["trading_day"]), str(row["ticker"]))
            group = groups.get(key)
            entry_open = pd.to_numeric(
                pd.Series([row.get("entry_open")]),
                errors="coerce",
            ).iloc[0]
            current_open = pd.to_numeric(
                pd.Series([row.get("exit_open")]),
                errors="coerce",
            ).iloc[0]
            held = pd.to_numeric(
                pd.Series([row.get("minutes_held")]),
                errors="coerce",
            ).iloc[0]
            entry_t = pd.to_numeric(
                pd.Series([row.get("entry_reference_t")]),
                errors="coerce",
            ).iloc[0]
            current_t = pd.to_numeric(
                pd.Series([row.get("exit_reference_t")]),
                errors="coerce",
            ).iloc[0]

            if not (
                _positive(entry_open)
                and _positive(current_open)
                and pd.notna(held)
                and pd.notna(entry_t)
                and pd.notna(current_t)
            ):
                current_values.append(np.nan)
                best_future_values.append(np.nan)
                option_values.append(np.nan)
                observed_counts.append(0)
                slots = (
                    max(0, MAX_HOLD_MINUTES - int(held))
                    if pd.notna(held)
                    else 0
                )
                slot_counts.append(slots)
                observed_fractions.append(np.nan if not slots else 0.0)
                continue

            held_int = int(held)
            slots = max(0, MAX_HOLD_MINUTES - held_int)
            current_base = _base_return(
                float(entry_open),
                float(current_open),
            )
            times, opens = _observed_future_opens(
                group,
                after_t=int(current_t),
                through_t=int(entry_t)
                + MAX_HOLD_MINUTES * MINUTE_MS,
            )
            observed_counts.append(int(len(opens)))
            slot_counts.append(slots)
            observed_fractions.append(
                float(len(opens) / slots) if slots else np.nan
            )
            current_values.append(current_base)

            if not slots or not len(opens):
                best_future_values.append(np.nan)
                option_values.append(np.nan)
                continue

            future_returns = np.array(
                [
                    _base_return(float(entry_open), float(value))
                    for value in opens
                ],
                dtype=float,
            )
            best_future = float(np.nanmax(future_returns))
            best_future_values.append(best_future)
            option_values.append(best_future - current_base)

        frame["exit_now_base_return_pct"] = current_values
        frame["best_future_base_return_pct"] = best_future_values
        frame["remaining_option_value_pct"] = option_values
        frame["future_observed_opens"] = observed_counts
        frame["future_slot_count"] = slot_counts
        frame["future_observed_fraction"] = observed_fractions
        result_parts.append(frame)

    return pd.concat(result_parts, ignore_index=True)


def fit_minute_baselines(fit: pd.DataFrame) -> dict[int, float]:
    target = pd.to_numeric(
        fit["remaining_option_value_pct"],
        errors="coerce",
    )
    held = pd.to_numeric(fit["minutes_held"], errors="coerce")
    valid = target.notna() & held.notna()
    grouped = pd.DataFrame(
        {
            "minute": held.loc[valid].astype(int),
            "target": target.loc[valid],
        }
    ).groupby("minute")["target"].mean()
    return {int(index): float(value) for index, value in grouped.items()}


def apply_excess_target(
    frame: pd.DataFrame,
    baselines: dict[int, float],
) -> pd.DataFrame:
    result = frame.copy()
    held = pd.to_numeric(result["minutes_held"], errors="coerce")
    baseline = held.map(
        lambda value: baselines.get(int(value), np.nan)
        if pd.notna(value)
        else np.nan
    )
    result["fit_minute_baseline_pct"] = baseline
    result["excess_remaining_option_value_pct"] = (
        pd.to_numeric(
            result["remaining_option_value_pct"],
            errors="coerce",
        )
        - baseline
    )
    return result


def train_remaining_value_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> tuple[HistGradientBoostingRegressor, tuple[str, ...], float]:
    target = pd.to_numeric(
        fit["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].copy()
    if len(train) < 100:
        raise ValueError("remaining-value fit requires at least 100 rows")

    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
    y = y.clip(lower=low, upper=high)

    x = transition_feature_frame(train)
    columns = tuple(column for column in x if x[column].notna().any())
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=RANDOM_SEED,
    )
    model.fit(x.loc[:, columns], y)

    cal_target = pd.to_numeric(
        calibration["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    cal_valid = cal_target.notna()
    if cal_valid.sum() < 100:
        raise ValueError("remaining-value calibration requires at least 100 rows")
    raw = model.predict(
        transition_feature_frame(
            calibration.loc[cal_valid]
        ).reindex(columns=columns)
    )
    offset = float(
        np.mean(cal_target.loc[cal_valid].to_numpy(dtype=float) - raw)
    )
    return model, columns, offset


def predict_remaining_value(
    frame: pd.DataFrame,
    fitted: tuple[HistGradientBoostingRegressor, tuple[str, ...], float],
) -> np.ndarray:
    model, columns, offset = fitted
    raw = model.predict(
        transition_feature_frame(frame).reindex(columns=columns)
    )
    return raw + offset


def _safe_corr(
    x: pd.Series,
    y: pd.Series,
    kind: str,
) -> float:
    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    if kind == "spearman":
        return float(spearmanr(pair["x"], pair["y"]).statistic)
    return float(pearsonr(pair["x"], pair["y"]).statistic)


def _bootstrap_daily(
    frame: pd.DataFrame,
    column: str,
) -> tuple[int, float, float, float]:
    values = pd.to_numeric(frame[column], errors="coerce")
    daily = pd.DataFrame(
        {
            "day": frame["trading_day"].astype(str),
            "value": values,
        }
    ).dropna().groupby("day")["value"].mean()
    if len(daily) < 2:
        return (
            int(len(daily)),
            float(daily.mean()) if len(daily) else np.nan,
            np.nan,
            np.nan,
        )
    arr = daily.to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    means = arr[
        rng.integers(
            0,
            len(arr),
            size=(BOOTSTRAP_SAMPLES, len(arr)),
        )
    ].mean(axis=1)
    return (
        int(len(arr)),
        float(arr.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def oracle_diagnostics(
    oracle: pd.DataFrame,
    month: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_entries = oracle.loc[
        pd.to_numeric(oracle["entry_open"], errors="coerce").gt(0)
    ]
    evaluated = oracle.loc[oracle["status"].eq("evaluated")].copy()
    values = pd.to_numeric(
        evaluated["oracle_best_base_pct"],
        errors="coerce",
    ).dropna()
    days, day_mean, low, high = _bootstrap_daily(
        evaluated,
        "oracle_best_base_pct",
    )
    summary = pd.DataFrame(
        [
            {
                "month": month,
                "gated_attempts": int(len(oracle)),
                "valid_entries": int(len(valid_entries)),
                "oracle_evaluable": int(len(evaluated)),
                "oracle_coverage": (
                    float(len(evaluated) / len(valid_entries))
                    if len(valid_entries)
                    else np.nan
                ),
                "oracle_base_mean_pct": (
                    float(values.mean()) if len(values) else np.nan
                ),
                "oracle_day_balanced_base_mean_pct": day_mean,
                "oracle_base_median_pct": (
                    float(values.median()) if len(values) else np.nan
                ),
                "oracle_base_p05_pct": (
                    float(values.quantile(0.05)) if len(values) else np.nan
                ),
                "oracle_base_positive_rate": (
                    float(values.gt(0).mean()) if len(values) else np.nan
                ),
                "oracle_best_hold_mean_min": float(
                    pd.to_numeric(
                        evaluated["oracle_best_holding_min"],
                        errors="coerce",
                    ).mean()
                )
                if len(evaluated)
                else np.nan,
                "mean_exit_observed_fraction": float(
                    pd.to_numeric(
                        evaluated["exit_observed_fraction"],
                        errors="coerce",
                    ).mean()
                )
                if len(evaluated)
                else np.nan,
            }
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "month": month,
                "days": days,
                "day_balanced_oracle_base_mean_pct": day_mean,
                "ci_low_pct": low,
                "ci_high_pct": high,
            }
        ]
    )
    return summary, bootstrap


def observability_diagnostics(
    scored: pd.DataFrame,
    month: str,
    calibration_offset: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gate = pd.to_numeric(
        scored["opportunity_probability"],
        errors="coerce",
    ).gt(0.5)
    current_valid = pd.to_numeric(
        scored["exit_now_base_return_pct"],
        errors="coerce",
    ).notna()
    pool = scored.loc[gate & current_valid].copy()
    target = pd.to_numeric(
        pool["excess_remaining_option_value_pct"],
        errors="coerce",
    )
    evaluable = pool.loc[target.notna()].copy()
    selected = evaluable.loc[
        pd.to_numeric(
            evaluable["predicted_excess_option_value_pct"],
            errors="coerce",
        ).gt(0)
    ].copy()

    per_minute = []
    for minute, group in evaluable.groupby(
        pd.to_numeric(
            evaluable["minutes_held"],
            errors="coerce",
        ).astype("Int64"),
        dropna=True,
    ):
        if len(group) < 20:
            continue
        corr = _safe_corr(
            group["predicted_excess_option_value_pct"],
            group["excess_remaining_option_value_pct"],
            "spearman",
        )
        per_minute.append(
            {
                "month": month,
                "minutes_held": int(minute),
                "rows": int(len(group)),
                "spearman": corr,
            }
        )
    minute_frame = pd.DataFrame(per_minute)
    valid_minute_corr = (
        pd.to_numeric(minute_frame["spearman"], errors="coerce").dropna()
        if len(minute_frame)
        else pd.Series(dtype=float)
    )

    days, selected_day_mean, low, high = _bootstrap_daily(
        selected,
        "excess_remaining_option_value_pct",
    )
    summary = pd.DataFrame(
        [
            {
                "month": month,
                "candidate_valid_exit_rows": int(len(pool)),
                "target_evaluable_rows": int(len(evaluable)),
                "target_coverage": (
                    float(len(evaluable) / len(pool))
                    if len(pool)
                    else np.nan
                ),
                "mean_future_observed_fraction": float(
                    pd.to_numeric(
                        evaluable["future_observed_fraction"],
                        errors="coerce",
                    ).mean()
                )
                if len(evaluable)
                else np.nan,
                "pearson": _safe_corr(
                    evaluable["predicted_excess_option_value_pct"],
                    evaluable["excess_remaining_option_value_pct"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    evaluable["predicted_excess_option_value_pct"],
                    evaluable["excess_remaining_option_value_pct"],
                    "spearman",
                ),
                "eligible_minute_groups": int(len(valid_minute_corr)),
                "median_minute_spearman": (
                    float(valid_minute_corr.median())
                    if len(valid_minute_corr)
                    else np.nan
                ),
                "positive_minute_spearman_fraction": (
                    float(valid_minute_corr.gt(0).mean())
                    if len(valid_minute_corr)
                    else np.nan
                ),
                "prediction_mean_pct": float(
                    pd.to_numeric(
                        evaluable["predicted_excess_option_value_pct"],
                        errors="coerce",
                    ).mean()
                )
                if len(evaluable)
                else np.nan,
                "realized_excess_mean_pct": float(
                    pd.to_numeric(
                        evaluable["excess_remaining_option_value_pct"],
                        errors="coerce",
                    ).mean()
                )
                if len(evaluable)
                else np.nan,
                "calibration_offset_pct": calibration_offset,
                "selected_rows": int(len(selected)),
                "selected_rate": (
                    float(len(selected) / len(evaluable))
                    if len(evaluable)
                    else np.nan
                ),
                "selected_remaining_option_mean_pct": float(
                    pd.to_numeric(
                        selected["remaining_option_value_pct"],
                        errors="coerce",
                    ).mean()
                )
                if len(selected)
                else np.nan,
                "selected_excess_mean_pct": float(
                    pd.to_numeric(
                        selected["excess_remaining_option_value_pct"],
                        errors="coerce",
                    ).mean()
                )
                if len(selected)
                else np.nan,
                "selected_day_balanced_excess_mean_pct": selected_day_mean,
            }
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "month": month,
                "selected_days": days,
                "selected_day_balanced_excess_mean_pct": selected_day_mean,
                "ci_low_pct": low,
                "ci_high_pct": high,
            }
        ]
    )
    return summary, minute_frame, bootstrap


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

    baselines = fit_minute_baselines(fit_rows)
    fit_rows = apply_excess_target(fit_rows, baselines)
    cal_rows = apply_excess_target(cal_rows, baselines)
    eval_rows = apply_excess_target(eval_rows, baselines)

    fitted = train_remaining_value_model(fit_rows, cal_rows)
    eval_rows["predicted_excess_option_value_pct"] = (
        predict_remaining_value(eval_rows, fitted)
    )

    eval_anchors = anchors.loc[
        anchors["_partition"].eq("evaluation")
    ].reset_index(drop=True)
    oracle = oracle_entry_ceiling(
        eval_anchors,
        states[evaluation_month],
        evaluation_month,
    )
    oracle_summary, oracle_bootstrap = oracle_diagnostics(
        oracle,
        evaluation_month,
    )

    _, _, offset = fitted
    observability, minute_corr, selected_bootstrap = (
        observability_diagnostics(
            eval_rows,
            evaluation_month,
            offset,
        )
    )

    o = oracle_summary.iloc[0]
    ob = oracle_bootstrap.iloc[0]
    v = observability.iloc[0]
    sb = selected_bootstrap.iloc[0]
    success = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "oracle_coverage_at_least_90pct": bool(
                    o["oracle_coverage"] >= 0.90
                ),
                "oracle_day_mean_positive": bool(
                    o["oracle_day_balanced_base_mean_pct"] > 0
                ),
                "oracle_bootstrap_low_positive": bool(
                    ob["ci_low_pct"] > 0
                ),
                "target_coverage_at_least_90pct": bool(
                    v["target_coverage"] >= 0.90
                ),
                "global_spearman_positive": bool(
                    v["spearman"] > 0
                ),
                "median_minute_spearman_positive": bool(
                    v["median_minute_spearman"] > 0
                ),
                "selected_day_excess_positive": bool(
                    v["selected_day_balanced_excess_mean_pct"] > 0
                ),
                "selected_bootstrap_low_positive": bool(
                    sb["ci_low_pct"] > 0
                ),
            }
        ]
    )
    success["all_frozen_checks_pass"] = success.iloc[:, 1:].all(axis=1)

    baseline_frame = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "minutes_held": minute,
                "fit_baseline_pct": value,
            }
            for minute, value in sorted(baselines.items())
        ]
    )
    return (
        oracle,
        oracle_summary,
        oracle_bootstrap,
        eval_rows,
        observability,
        minute_corr,
        selected_bootstrap,
        baseline_frame,
        provenance,
        success,
    )


def render_report(outputs: tuple[pd.DataFrame, ...]) -> str:
    (
        _oracle,
        oracle_summary,
        oracle_bootstrap,
        _scored,
        observability,
        minute_corr,
        selected_bootstrap,
        _baselines,
        provenance,
        success,
    ) = outputs
    lines = [
        "=== MoneyMaker Remaining-Option Observability v3.9 ===",
        "diagnostic only: hindsight best-exit ceiling plus causal remaining-value observability",
        "features/gate/lags frozen from v3.7; BASE full-round-trip oracle labels",
        "held-minute baseline learned from fit only; April 2026+ sealed",
    ]
    for title, frame in (
        ("Oracle entry ceiling", oracle_summary),
        ("Oracle day bootstrap", oracle_bootstrap),
        ("Remaining-value observability", observability),
        ("Per-minute Spearman", minute_corr),
        ("Selected-excess bootstrap", selected_bootstrap),
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
    parser.add_argument("--oracle-csv", type=Path, required=True)
    parser.add_argument("--oracle-summary-csv", type=Path, required=True)
    parser.add_argument("--oracle-bootstrap-csv", type=Path, required=True)
    parser.add_argument("--scored-csv", type=Path, required=True)
    parser.add_argument("--observability-csv", type=Path, required=True)
    parser.add_argument("--minute-corr-csv", type=Path, required=True)
    parser.add_argument("--selected-bootstrap-csv", type=Path, required=True)
    parser.add_argument("--baselines-csv", type=Path, required=True)
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
        args.oracle_csv,
        args.oracle_summary_csv,
        args.oracle_bootstrap_csv,
        args.scored_csv,
        args.observability_csv,
        args.minute_corr_csv,
        args.selected_bootstrap_csv,
        args.baselines_csv,
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
