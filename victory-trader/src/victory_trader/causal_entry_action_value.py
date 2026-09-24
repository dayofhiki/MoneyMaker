"""Request 182: causal ENTER / WAIT / ABSTAIN entry-value bridge.

This development-only experiment adds an entry-timing/abstention layer after
the frozen BUY gate. Runtime inputs come only from the original HOT timestamp.
Future POSITION rows are used only to construct historical action-value labels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .fresh_consensus_hurdle_value import FRESH_DAYS
from .market_calendar import regular_session_bounds
from .second_path_attention_probe import BASELINE_FEATURES, _annotate_scan
from .selected_hot_position_value_observability import _base_return

REQUEST_ID = 182
ENTER_SEED = 20261082
WAIT_SEED = 20261083
BOOTSTRAP_SEED = 20261082
BOOTSTRAP_SAMPLES = 10_000
SEVERE_LOSS_PCT = -5.0
MIN_ENTRY_STATE_COVERAGE = 0.90
MIN_ACTION_VALUE_COVERAGE = 0.90
MIN_EXECUTED_RATE = 0.10
MAX_EXECUTED_RATE = 0.80

EPISODE_KEYS = ["trading_day", "ticker", "hot_t"]
ENTRY_FEATURES = tuple(
    dict.fromkeys(
        [
            *BASELINE_FEATURES,
            "log_current_price",
            "minutes_since_open",
            "minutes_to_close",
        ]
    )
)


@dataclass(frozen=True)
class ValueModel:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def _day_weights(frame: pd.DataFrame) -> np.ndarray:
    day = frame["trading_day"].astype(str)
    counts = day.map(day.value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(dtype=float)
    mean = float(np.mean(weights))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("invalid equal-day weights")
    return weights / mean


def build_action_labels(position_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, group in position_rows.groupby(EPISODE_KEYS, sort=False):
        work = group.copy()
        work["_minute"] = pd.to_numeric(
            work["minutes_held"],
            errors="coerce",
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

        enter_value = pd.to_numeric(
            pd.Series([row1.get("exit_now_base_return_pct")]),
            errors="coerce",
        ).iloc[0]
        minute1_open = pd.to_numeric(
            pd.Series([row1.get("exit_reference_open")]),
            errors="coerce",
        ).iloc[0]

        wait_value = np.nan
        row2 = by_minute.get(2)
        if row2 is not None and pd.notna(minute1_open):
            minute2_open = pd.to_numeric(
                pd.Series([row2.get("exit_reference_open")]),
                errors="coerce",
            ).iloc[0]
            if (
                pd.notna(minute2_open)
                and float(minute1_open) > 0
                and float(minute2_open) > 0
            ):
                wait_value = _base_return(
                    float(minute1_open),
                    float(minute2_open),
                )

        entry_open = pd.to_numeric(
            pd.Series([row1.get("entry_open")]),
            errors="coerce",
        ).iloc[0]
        records.append(
            {
                "trading_day": str(row1["trading_day"]),
                "ticker": str(row1["ticker"]).upper(),
                "hot_t": int(row1["hot_t"]),
                "entry_open": (
                    float(entry_open) if pd.notna(entry_open) else np.nan
                ),
                "enter_now_base_pct": (
                    float(enter_value) if pd.notna(enter_value) else np.nan
                ),
                "wait_1m_base_pct": (
                    float(wait_value) if pd.notna(wait_value) else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def build_entry_states(
    scan: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    if labels.empty:
        return labels.copy()

    ticker_days = labels.loc[:, ["trading_day", "ticker"]].copy()
    ticker_days["trading_day"] = ticker_days["trading_day"].astype(str)
    ticker_days["ticker"] = ticker_days["ticker"].astype(str).str.upper()
    ticker_days = ticker_days.drop_duplicates()

    source = scan.copy()
    source["trading_day"] = source["trading_day"].astype(str)
    source["ticker"] = source["ticker"].astype(str).str.upper()

    selected = source.merge(
        ticker_days,
        on=["trading_day", "ticker"],
        how="inner",
        validate="many_to_one",
    )
    annotated = _annotate_scan(selected)

    ranks = source.loc[
        :, ["trading_day", "ticker", "t", "attention_score"]
    ].copy()
    ranks["attention_rank"] = ranks.groupby(
        ["trading_day", "t"],
        sort=False,
    )["attention_score"].rank(method="first", ascending=False)
    annotated = annotated.merge(
        ranks.loc[
            :, ["trading_day", "ticker", "t", "attention_rank"]
        ],
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
    )

    keys = labels.loc[:, EPISODE_KEYS + ["entry_open"]].copy()
    keys = keys.rename(columns={"hot_t": "t"})
    states = keys.merge(
        annotated,
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
        indicator="_entry_snapshot_merge",
    )
    states["entry_state_available"] = states[
        "_entry_snapshot_merge"
    ].eq("both")
    states = states.drop(columns=["_entry_snapshot_merge"])
    states = states.rename(columns={"t": "hot_t"})

    current = pd.to_numeric(states.get("c"), errors="coerce")
    states["log_current_price"] = np.log(current.where(current.gt(0)))

    bounds: dict[str, tuple[int, int]] = {}
    for day_text in states["trading_day"].astype(str).unique():
        session = regular_session_bounds(date.fromisoformat(day_text))
        if session is None:
            raise ValueError(f"missing regular session bounds for {day_text}")
        bounds[day_text] = (
            int(session[0].timestamp() * 1000),
            int(session[1].timestamp() * 1000),
        )

    since: list[float] = []
    remaining: list[float] = []
    for row in states.itertuples(index=False):
        open_ms, close_ms = bounds[str(row.trading_day)]
        timestamp = int(row.hot_t)
        since.append((timestamp - open_ms) / 60_000.0)
        remaining.append((close_ms - timestamp) / 60_000.0)
    states["minutes_since_open"] = since
    states["minutes_to_close"] = remaining

    return states.merge(
        labels.drop(columns=["entry_open"]),
        on=EPISODE_KEYS,
        how="left",
        validate="one_to_one",
    )


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def train_value_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    target_column: str,
    seed: int,
) -> ValueModel:
    target = pd.to_numeric(fit[target_column], errors="coerce")
    valid = target.notna() & fit["entry_state_available"].fillna(False)
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 500:
        raise ValueError(
            f"request 182 {target_column} needs >=500 fit rows, got {len(train)}"
        )

    columns = tuple(
        column
        for column in ENTRY_FEATURES
        if column in train.columns
        and pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    if not columns:
        raise ValueError("request 182 has no usable entry features")

    low, high = np.quantile(y.to_numpy(dtype=float), [0.005, 0.995])
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
        _feature_frame(train, columns),
        y.clip(lower=low, upper=high),
        sample_weight=_day_weights(train),
    )

    cal_target = pd.to_numeric(
        calibration[target_column],
        errors="coerce",
    )
    cal_valid = (
        cal_target.notna()
        & calibration["entry_state_available"].fillna(False)
    )
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 250:
        raise ValueError(
            f"request 182 {target_column} needs >=250 calibration rows, "
            f"got {len(cal)}"
        )
    raw = model.predict(_feature_frame(cal, columns))
    residual = cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    offset = float(np.average(residual, weights=_day_weights(cal)))
    return ValueModel(
        model=model,
        columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_value(frame: pd.DataFrame, fitted: ValueModel) -> np.ndarray:
    return (
        fitted.model.predict(_feature_frame(frame, fitted.columns))
        + fitted.offset
    )


def _safe_spearman(actual: pd.Series, predicted: pd.Series) -> float | None:
    y = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = y.notna() & p.notna()
    if int(valid.sum()) < 20:
        return None
    value = y.loc[valid].corr(p.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def choose_action(frame: pd.DataFrame) -> pd.Series:
    enter = pd.to_numeric(
        frame["predicted_enter_now_base_pct"],
        errors="coerce",
    )
    wait = pd.to_numeric(
        frame["predicted_wait_1m_base_pct"],
        errors="coerce",
    )

    # Conservative tie-breaking: ABSTAIN at non-positive maxima; WAIT beats
    # ENTER_NOW on an exact positive tie.
    action = pd.Series("ABSTAIN", index=frame.index, dtype=object)
    positive = pd.concat([enter, wait], axis=1).max(axis=1).gt(0)
    wait_wins = positive & wait.ge(enter)
    enter_wins = positive & enter.gt(wait)
    action.loc[wait_wins] = "WAIT_1M"
    action.loc[enter_wins] = "ENTER_NOW"
    return action


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


def _bootstrap_difference(
    frame: pd.DataFrame,
) -> dict[str, float | int | None]:
    values = pd.to_numeric(frame["policy_minus_current_pct"], errors="coerce")
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


def evaluate_policy(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    result = frame.copy()
    result["action"] = choose_action(result)

    enter_realized = pd.to_numeric(
        result["enter_now_base_pct"],
        errors="coerce",
    )
    wait_realized = pd.to_numeric(
        result["wait_1m_base_pct"],
        errors="coerce",
    )
    realized = pd.Series(np.nan, index=result.index, dtype=float)
    realized.loc[result["action"].eq("ABSTAIN")] = 0.0
    realized.loc[result["action"].eq("ENTER_NOW")] = enter_realized.loc[
        result["action"].eq("ENTER_NOW")
    ]
    realized.loc[result["action"].eq("WAIT_1M")] = wait_realized.loc[
        result["action"].eq("WAIT_1M")
    ]
    result["policy_realized_base_pct"] = realized
    result["action_value_resolved"] = realized.notna()
    result["executed"] = result["action"].ne("ABSTAIN")

    comparator = enter_realized
    result["current_enter_now_base_pct"] = comparator
    matched = realized.notna() & comparator.notna()
    result["policy_minus_current_pct"] = np.where(
        matched,
        realized - comparator,
        np.nan,
    )

    covered = result["entry_state_available"].fillna(False)
    resolved = result["action_value_resolved"] & covered
    executed_resolved = resolved & result["executed"]
    trade_values = pd.to_numeric(
        result.loc[executed_resolved, "policy_realized_base_pct"],
        errors="coerce",
    )
    comparator_values = pd.to_numeric(
        result.loc[matched, "current_enter_now_base_pct"],
        errors="coerce",
    )

    action_counts = result.loc[covered, "action"].value_counts().to_dict()
    action_rates = {
        action: float(count / int(covered.sum()))
        for action, count in action_counts.items()
    } if int(covered.sum()) else {}

    metrics = {
        "rows": int(len(result)),
        "entry_state_rows": int(covered.sum()),
        "entry_state_coverage": (
            float(covered.mean()) if len(result) else None
        ),
        "action_value_resolved_rows": int(resolved.sum()),
        "action_value_coverage": (
            float(resolved.sum() / covered.sum())
            if int(covered.sum())
            else None
        ),
        "action_counts": {
            str(key): int(value) for key, value in action_counts.items()
        },
        "action_rates": action_rates,
        "executed_rows": int(executed_resolved.sum()),
        "executed_rate": (
            float(executed_resolved.sum() / covered.sum())
            if int(covered.sum())
            else None
        ),
        "executed_base_mean_pct": (
            float(trade_values.mean()) if len(trade_values) else None
        ),
        "executed_day_balanced_base_mean_pct": _day_balanced(
            result.loc[executed_resolved],
            "policy_realized_base_pct",
        ),
        "executed_positive_rate": (
            float(trade_values.gt(0).mean()) if len(trade_values) else None
        ),
        "executed_severe_loss_rate": (
            float(trade_values.le(SEVERE_LOSS_PCT).mean())
            if len(trade_values)
            else None
        ),
        "all_opportunity_policy_value_mean_pct": (
            float(
                pd.to_numeric(
                    result.loc[resolved, "policy_realized_base_pct"],
                    errors="coerce",
                ).mean()
            )
            if int(resolved.sum())
            else None
        ),
        "all_opportunity_day_balanced_policy_value_pct": _day_balanced(
            result.loc[resolved],
            "policy_realized_base_pct",
        ),
        "comparator_enter_now_mean_pct": (
            float(comparator_values.mean())
            if len(comparator_values)
            else None
        ),
        "comparator_enter_now_day_balanced_pct": _day_balanced(
            result.loc[matched],
            "current_enter_now_base_pct",
        ),
        "comparator_severe_loss_rate": (
            float(comparator_values.le(SEVERE_LOSS_PCT).mean())
            if len(comparator_values)
            else None
        ),
        "matched_rows": int(matched.sum()),
        "matched_policy_minus_current_mean_pct": (
            float(
                pd.to_numeric(
                    result.loc[matched, "policy_minus_current_pct"],
                    errors="coerce",
                ).mean()
            )
            if int(matched.sum())
            else None
        ),
        "matched_policy_minus_current_day_balanced_pct": _day_balanced(
            result.loc[matched],
            "policy_minus_current_pct",
        ),
        "matched_difference_bootstrap": _bootstrap_difference(
            result.loc[matched]
        ),
    }
    return result, metrics


def _group_policy_metrics(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, group in frame.groupby(column, sort=True, observed=True):
        _, metrics = evaluate_policy(group)
        output[str(key)] = metrics
    return output


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_path)
    cal_positions = pd.read_parquet(calibration_path)
    history_scan = pd.read_parquet(history_scan_path)

    fit_labels = build_action_labels(fit_positions)
    cal_labels = build_action_labels(cal_positions)
    historical_labels = pd.concat(
        [fit_labels, cal_labels],
        ignore_index=True,
    )
    historical_states = build_entry_states(
        history_scan,
        historical_labels,
    )
    fit_days = set(fit_labels["trading_day"].astype(str))
    cal_days = set(cal_labels["trading_day"].astype(str))
    fit = historical_states.loc[
        historical_states["trading_day"].astype(str).isin(fit_days)
    ].copy()
    calibration = historical_states.loc[
        historical_states["trading_day"].astype(str).isin(cal_days)
    ].copy()

    enter_model = train_value_model(
        fit,
        calibration,
        target_column="enter_now_base_pct",
        seed=ENTER_SEED,
    )
    wait_model = train_value_model(
        fit,
        calibration,
        target_column="wait_1m_base_pct",
        seed=WAIT_SEED,
    )

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError("request 182 requires complete request 178 shards")
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    fresh_labels = build_action_labels(fresh_positions)
    fresh = build_entry_states(fresh_scan, fresh_labels)
    if sorted(fresh["trading_day"].astype(str).unique()) != list(FRESH_DAYS):
        raise ValueError("request 182 fresh block differs from request 178")

    fresh["predicted_enter_now_base_pct"] = predict_value(
        fresh,
        enter_model,
    )
    fresh["predicted_wait_1m_base_pct"] = predict_value(
        fresh,
        wait_model,
    )
    scored, policy = evaluate_policy(fresh)

    entry_open = pd.to_numeric(scored["entry_open"], errors="coerce")
    scored["entry_price_bin"] = pd.cut(
        entry_open,
        bins=(-np.inf, 1.0, 2.0, 5.0, 10.0, np.inf),
        labels=("<1", "1-2", "2-5", "5-10", ">=10"),
        right=False,
    ).astype(str)

    enter_spearman = _safe_spearman(
        scored["enter_now_base_pct"],
        scored["predicted_enter_now_base_pct"],
    )
    wait_spearman = _safe_spearman(
        scored["wait_1m_base_pct"],
        scored["predicted_wait_1m_base_pct"],
    )

    executed_rate = policy["executed_rate"]
    diff = policy["matched_policy_minus_current_day_balanced_pct"]
    diff_low = policy["matched_difference_bootstrap"]["ci_low_pct"]
    trade_day = policy["executed_day_balanced_base_mean_pct"]
    all_day = policy["all_opportunity_day_balanced_policy_value_pct"]
    severe = policy["executed_severe_loss_rate"]
    comparator_severe = policy["comparator_severe_loss_rate"]

    development_gate = bool(
        policy["entry_state_coverage"] is not None
        and float(policy["entry_state_coverage"]) >= MIN_ENTRY_STATE_COVERAGE
        and policy["action_value_coverage"] is not None
        and float(policy["action_value_coverage"]) >= MIN_ACTION_VALUE_COVERAGE
        and executed_rate is not None
        and MIN_EXECUTED_RATE <= float(executed_rate) <= MAX_EXECUTED_RATE
        and trade_day is not None
        and float(trade_day) > 0
        and all_day is not None
        and float(all_day) > 0
        and diff is not None
        and float(diff) > 0
        and diff_low is not None
        and float(diff_low) > 0
        and severe is not None
        and comparator_severe is not None
        and float(severe) <= float(comparator_severe)
        and enter_spearman is not None
        and float(enter_spearman) > 0
        and wait_spearman is not None
        and float(wait_spearman) > 0
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "upstream_buy_policy_changed": False,
        "entry_features": list(enter_model.columns),
        "model_diagnostics": {
            "enter_offset_pct": enter_model.offset,
            "enter_winsor_low_pct": enter_model.winsor_low,
            "enter_winsor_high_pct": enter_model.winsor_high,
            "wait_offset_pct": wait_model.offset,
            "wait_winsor_low_pct": wait_model.winsor_low,
            "wait_winsor_high_pct": wait_model.winsor_high,
            "enter_value_spearman": enter_spearman,
            "wait_value_spearman": wait_spearman,
        },
        "policy": policy,
        "by_action": _group_policy_metrics(scored, "action"),
        "by_entry_price_bin": _group_policy_metrics(
            scored, "entry_price_bin"
        ),
        "frozen_gate": {
            "min_entry_state_coverage": MIN_ENTRY_STATE_COVERAGE,
            "min_action_value_coverage": MIN_ACTION_VALUE_COVERAGE,
            "min_executed_rate": MIN_EXECUTED_RATE,
            "max_executed_rate": MAX_EXECUTED_RATE,
            "executed_day_balanced_base_must_be_positive": True,
            "all_opportunity_day_balanced_value_must_be_positive": True,
            "matched_difference_must_be_positive": True,
            "matched_bootstrap_low_must_be_positive": True,
            "severe_loss_no_worse_than_current": True,
            "both_action_value_spearman_must_be_positive": True,
        },
        "development_gate_pass": development_gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scored.to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--history-scan", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
