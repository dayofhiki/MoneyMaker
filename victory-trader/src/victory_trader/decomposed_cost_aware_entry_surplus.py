"""Request 184: decomposed cost-aware entry surplus.

Development-only experiment on already-opened dates. Entry economics are split
into raw price movement and modeled execution drag, with causal one-second
features attached at the original HOT timestamp.
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

from .causal_entry_action_value import (
    BOOTSTRAP_SAMPLES,
    EPISODE_KEYS,
    ENTRY_FEATURES,
    SEVERE_LOSS_PCT,
    _bootstrap_difference,
    _day_balanced,
    evaluate_policy,
)
from .config import load_settings
from .execution_costs import (
    DEFAULT_EXECUTION_SCENARIOS,
    net_round_trip_return_pct,
)
from .fresh_consensus_hurdle_value import FRESH_DAYS
from .market_calendar import regular_session_bounds
from .massive_client import MassiveClient
from .second_path_attention_probe import (
    SECOND_CACHE_DIR,
    SECOND_FEATURES,
    _annotate_scan,
    _second_frame,
    second_path_features,
)
from .selected_hot_position_value_observability import _base_return

REQUEST_ID = 184
SEEDS = {
    "enter_gross_pct": 20261084,
    "enter_drag_pct": 20261085,
    "wait_gross_pct": 20261086,
    "wait_drag_pct": 20261087,
}
MIN_FIT_ROWS = 300
MIN_CAL_ROWS = 150
MIN_ENTRY_COVERAGE = 0.90
MIN_SECOND_COVERAGE = 0.90
MIN_EXECUTED_RATE = 0.03
MAX_EXECUTED_RATE = 0.40
MIN_POSITIVE_DAYS = 3
BASE_SCENARIO = next(
    item for item in DEFAULT_EXECUTION_SCENARIOS if item.name == "base"
)
DECOMPOSED_FEATURES = tuple(
    dict.fromkeys(
        [
            *ENTRY_FEATURES,
            *SECOND_FEATURES,
            "base_zero_move_cost_proxy_pct",
        ]
    )
)


@dataclass(frozen=True)
class ComponentModel:
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
        raise ValueError("request 184 invalid equal-day weights")
    return weights / mean


def _gross_return(entry: float, exit_price: float) -> float:
    if not (
        np.isfinite(float(entry))
        and np.isfinite(float(exit_price))
        and float(entry) > 0
        and float(exit_price) > 0
    ):
        return np.nan
    return (float(exit_price) / float(entry) - 1.0) * 100.0


def build_decomposed_labels(position_rows: pd.DataFrame) -> pd.DataFrame:
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
        minute1 = pd.to_numeric(
            pd.Series([row1.get("exit_reference_open")]), errors="coerce"
        ).iloc[0]
        enter_base = pd.to_numeric(
            pd.Series([row1.get("exit_now_base_return_pct")]), errors="coerce"
        ).iloc[0]

        enter_gross = np.nan
        enter_drag = np.nan
        if (
            pd.notna(entry)
            and pd.notna(minute1)
            and pd.notna(enter_base)
            and float(entry) > 0
            and float(minute1) > 0
        ):
            enter_gross = _gross_return(float(entry), float(minute1))
            enter_drag = float(enter_gross) - float(enter_base)

        wait_gross = np.nan
        wait_base = np.nan
        wait_drag = np.nan
        row2 = by_minute.get(2)
        if row2 is not None and pd.notna(minute1) and float(minute1) > 0:
            minute2 = pd.to_numeric(
                pd.Series([row2.get("exit_reference_open")]), errors="coerce"
            ).iloc[0]
            if pd.notna(minute2) and float(minute2) > 0:
                wait_gross = _gross_return(float(minute1), float(minute2))
                wait_base = _base_return(float(minute1), float(minute2))
                if pd.notna(wait_base):
                    wait_drag = float(wait_gross) - float(wait_base)

        records.append(
            {
                "trading_day": str(row1["trading_day"]),
                "ticker": str(row1["ticker"]).upper(),
                "hot_t": int(row1["hot_t"]),
                "entry_open": float(entry) if pd.notna(entry) else np.nan,
                "enter_gross_pct": (
                    float(enter_gross) if pd.notna(enter_gross) else np.nan
                ),
                "enter_drag_pct": (
                    float(enter_drag) if pd.notna(enter_drag) else np.nan
                ),
                "enter_now_base_pct": (
                    float(enter_base) if pd.notna(enter_base) else np.nan
                ),
                "wait_gross_pct": (
                    float(wait_gross) if pd.notna(wait_gross) else np.nan
                ),
                "wait_drag_pct": (
                    float(wait_drag) if pd.notna(wait_drag) else np.nan
                ),
                "wait_1m_base_pct": (
                    float(wait_base) if pd.notna(wait_base) else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def _add_cost_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "current_price" in result:
        current = pd.to_numeric(result["current_price"], errors="coerce")
    else:
        current = np.exp(
            pd.to_numeric(result.get("log_current_price"), errors="coerce")
        )
        result["current_price"] = current
    proxy = []
    for value in current:
        if pd.notna(value) and np.isfinite(float(value)) and float(value) > 0:
            net = net_round_trip_return_pct(
                float(value),
                0.0,
                BASE_SCENARIO,
            )
            proxy.append(float(-net))
        else:
            proxy.append(np.nan)
    result["base_zero_move_cost_proxy_pct"] = proxy
    return result


def build_historical_states(
    first_hot: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    features = first_hot.copy()
    features["trading_day"] = features["trading_day"].astype(str)
    features["ticker"] = features["ticker"].astype(str).str.upper()
    features = features.rename(columns={"t": "hot_t"})
    features = features.drop_duplicates(EPISODE_KEYS, keep="first")

    states = labels.merge(
        features,
        on=EPISODE_KEYS,
        how="left",
        validate="one_to_one",
        indicator="_feature_merge",
        suffixes=("", "_feature"),
    )
    states["entry_state_available"] = states["_feature_merge"].eq("both")
    states = states.drop(columns=["_feature_merge"])
    return _add_cost_proxy(states)


def _session_clock(day_text: str, timestamp_ms: int) -> tuple[float, float]:
    session = regular_session_bounds(date.fromisoformat(day_text))
    if session is None:
        raise ValueError(f"request 184 missing session bounds for {day_text}")
    open_ms = int(session[0].timestamp() * 1000)
    close_ms = int(session[1].timestamp() * 1000)
    return (
        float((timestamp_ms - open_ms) / 60_000.0),
        float((close_ms - timestamp_ms) / 60_000.0),
    )


def build_fresh_states(
    scan: pd.DataFrame,
    labels: pd.DataFrame,
    client: MassiveClient,
) -> pd.DataFrame:
    if labels.empty:
        return labels.copy()

    source = scan.copy()
    source["trading_day"] = source["trading_day"].astype(str)
    source["ticker"] = source["ticker"].astype(str).str.upper()

    ticker_days = labels.loc[:, ["trading_day", "ticker"]].drop_duplicates()
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
        ["trading_day", "t"], sort=False
    )["attention_score"].rank(method="first", ascending=False)
    annotated = annotated.merge(
        ranks.loc[:, ["trading_day", "ticker", "t", "attention_rank"]],
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
    )

    keys = labels.loc[:, EPISODE_KEYS].rename(columns={"hot_t": "t"})
    states = keys.merge(
        annotated,
        on=["trading_day", "ticker", "t"],
        how="left",
        validate="one_to_one",
        indicator="_feature_merge",
    )
    states["entry_state_available"] = states["_feature_merge"].eq("both")
    states = states.drop(columns=["_feature_merge"])

    second_cache: dict[tuple[str, str], pd.DataFrame] = {}
    second_records: list[dict[str, float]] = []
    for row in states.itertuples(index=False):
        day_text = str(row.trading_day)
        ticker = str(row.ticker).upper()
        key = (day_text, ticker)
        if key not in second_cache:
            payload = client.second_bars_range(
                ticker,
                date.fromisoformat(day_text),
                date.fromisoformat(day_text),
                adjusted=False,
            )
            second_cache[key] = _second_frame(payload)
        second_records.append(
            second_path_features(second_cache[key], int(row.t))
        )
    states = pd.concat(
        [states.reset_index(drop=True), pd.DataFrame(second_records)],
        axis=1,
    )

    current = pd.to_numeric(states.get("c"), errors="coerce")
    states["current_price"] = current
    states["log_current_price"] = np.log(current.where(current.gt(0)))
    since: list[float] = []
    remaining: list[float] = []
    for row in states.itertuples(index=False):
        a, b = _session_clock(str(row.trading_day), int(row.t))
        since.append(a)
        remaining.append(b)
    states["minutes_since_open"] = since
    states["minutes_to_close"] = remaining
    states = states.rename(columns={"t": "hot_t"})

    states = states.merge(
        labels,
        on=EPISODE_KEYS,
        how="left",
        validate="one_to_one",
    )
    return _add_cost_proxy(states)


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric, errors="coerce"
    )


def train_component(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    target_column: str,
    seed: int,
) -> ComponentModel:
    target = pd.to_numeric(fit[target_column], errors="coerce")
    valid = target.notna() & fit["entry_state_available"].fillna(False)
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < MIN_FIT_ROWS:
        raise ValueError(
            f"request 184 {target_column} needs >= {MIN_FIT_ROWS} fit rows, "
            f"got {len(train)}"
        )

    columns = tuple(
        column
        for column in DECOMPOSED_FEATURES
        if column in train.columns
        and pd.to_numeric(train[column], errors="coerce").notna().any()
    )
    if not columns:
        raise ValueError("request 184 has no usable causal entry features")

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
        calibration[target_column], errors="coerce"
    )
    cal_valid = (
        cal_target.notna()
        & calibration["entry_state_available"].fillna(False)
    )
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < MIN_CAL_ROWS:
        raise ValueError(
            f"request 184 {target_column} needs >= {MIN_CAL_ROWS} "
            f"calibration rows, got {len(cal)}"
        )
    raw = model.predict(_feature_frame(cal, columns))
    residual = cal_target.loc[cal_valid].to_numpy(dtype=float) - raw
    offset = float(np.average(residual, weights=_day_weights(cal)))
    return ComponentModel(
        model=model,
        columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_component(
    frame: pd.DataFrame,
    fitted: ComponentModel,
) -> np.ndarray:
    return (
        fitted.model.predict(_feature_frame(frame, fitted.columns))
        + fitted.offset
    )


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


def _group_metrics(frame: pd.DataFrame, column: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, group in frame.groupby(column, sort=True, observed=True):
        _, metrics = evaluate_policy(group)
        result[str(key)] = metrics
    return result


def evaluate(
    history_first_hot_path: Path,
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    first_hot = pd.read_parquet(history_first_hot_path)
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(calibration_positions_path)

    fit_labels = build_decomposed_labels(fit_positions)
    cal_labels = build_decomposed_labels(calibration_positions)
    fit = build_historical_states(first_hot, fit_labels)
    calibration = build_historical_states(first_hot, cal_labels)

    models = {
        target: train_component(
            fit,
            calibration,
            target_column=target,
            seed=SEEDS[target],
        )
        for target in SEEDS
    }

    position_paths = sorted(fresh_dir.glob("*-positions.parquet"))
    scan_paths = sorted(fresh_dir.glob("*-scan.parquet"))
    if (
        len(position_paths) != len(FRESH_DAYS)
        or len(scan_paths) != len(FRESH_DAYS)
    ):
        raise ValueError("request 184 requires complete request 178 shards")
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in position_paths],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [pd.read_parquet(path) for path in scan_paths],
        ignore_index=True,
    )
    fresh_labels = build_decomposed_labels(fresh_positions)

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR,
        request_interval_seconds=0.0,
    )
    fresh = build_fresh_states(fresh_scan, fresh_labels, client)
    if sorted(fresh["trading_day"].astype(str).unique()) != list(FRESH_DAYS):
        raise ValueError("request 184 fresh block differs from request 178")

    for target, model in models.items():
        fresh[f"predicted_{target}"] = predict_component(fresh, model)

    fresh["predicted_enter_drag_pct"] = pd.to_numeric(
        fresh["predicted_enter_drag_pct"], errors="coerce"
    ).clip(lower=0.0)
    fresh["predicted_wait_drag_pct"] = pd.to_numeric(
        fresh["predicted_wait_drag_pct"], errors="coerce"
    ).clip(lower=0.0)
    fresh["predicted_enter_now_base_pct"] = (
        pd.to_numeric(fresh["predicted_enter_gross_pct"], errors="coerce")
        - pd.to_numeric(fresh["predicted_enter_drag_pct"], errors="coerce")
    )
    fresh["predicted_wait_1m_base_pct"] = (
        pd.to_numeric(fresh["predicted_wait_gross_pct"], errors="coerce")
        - pd.to_numeric(fresh["predicted_wait_drag_pct"], errors="coerce")
    )

    scored, policy = evaluate_policy(fresh)
    current = pd.to_numeric(scored["current_price"], errors="coerce")
    scored["entry_price_bin"] = pd.cut(
        current,
        bins=(-np.inf, 1.0, 2.0, 5.0, 10.0, np.inf),
        labels=("<1", "1-2", "2-5", "5-10", ">=10"),
        right=False,
    ).astype(str)

    component_spearman = {
        target: _safe_spearman(
            scored[target],
            scored[f"predicted_{target}"],
        )
        for target in SEEDS
    }
    net_spearman = {
        "enter_now": _safe_spearman(
            scored["enter_now_base_pct"],
            scored["predicted_enter_now_base_pct"],
        ),
        "wait_1m": _safe_spearman(
            scored["wait_1m_base_pct"],
            scored["predicted_wait_1m_base_pct"],
        ),
    }

    covered = scored["entry_state_available"].fillna(False)
    second = pd.to_numeric(
        scored.get("active_seconds_60"), errors="coerce"
    ).fillna(0).gt(0)
    second_coverage = (
        float(second.loc[covered].mean()) if int(covered.sum()) else None
    )

    executed_resolved = (
        scored["executed"]
        & scored["action_value_resolved"]
        & covered
    )
    positive_days = 0
    by_day: dict[str, object] = {}
    for day in FRESH_DAYS:
        part = scored.loc[
            scored["trading_day"].astype(str).eq(day)
        ].copy()
        mask = (
            part["executed"]
            & part["action_value_resolved"]
            & part["entry_state_available"].fillna(False)
        )
        values = pd.to_numeric(
            part.loc[mask, "policy_realized_base_pct"], errors="coerce"
        )
        mean = float(values.mean()) if len(values) else None
        good = bool(mean is not None and mean > 0)
        positive_days += int(good)
        by_day[day] = {
            "rows": int(len(part)),
            "executed_rows": int(mask.sum()),
            "executed_base_mean_pct": mean,
            "positive_return_day": good,
        }

    entry_coverage = float(covered.mean()) if len(scored) else None
    executed_rate = policy.get("executed_rate")
    executed_day = policy.get("executed_day_balanced_base_mean_pct")
    all_day = policy.get("all_opportunity_day_balanced_policy_value_pct")
    diff_day = policy.get("matched_policy_minus_current_day_balanced_pct")
    diff_low = policy.get("matched_difference_bootstrap", {}).get("ci_low_pct")
    severe = policy.get("executed_severe_loss_rate")
    comparator_severe = policy.get("comparator_severe_loss_rate")

    gate = bool(
        entry_coverage is not None
        and entry_coverage >= MIN_ENTRY_COVERAGE
        and second_coverage is not None
        and second_coverage >= MIN_SECOND_COVERAGE
        and net_spearman["enter_now"] is not None
        and float(net_spearman["enter_now"]) > 0
        and net_spearman["wait_1m"] is not None
        and float(net_spearman["wait_1m"]) > 0
        and executed_rate is not None
        and MIN_EXECUTED_RATE <= float(executed_rate) <= MAX_EXECUTED_RATE
        and executed_day is not None
        and float(executed_day) > 0
        and all_day is not None
        and float(all_day) > 0
        and diff_day is not None
        and float(diff_day) > 0
        and diff_low is not None
        and float(diff_low) > 0
        and severe is not None
        and comparator_severe is not None
        and float(severe) <= float(comparator_severe)
        and positive_days >= MIN_POSITIVE_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "representation_change": (
            "request182 causal entry features + one-second entry snapshot + "
            "BASE zero-move cost proxy"
        ),
        "economic_decomposition": (
            "predicted gross movement minus predicted nonnegative execution drag"
        ),
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
        "component_spearman": component_spearman,
        "net_value_spearman": net_spearman,
        "model_diagnostics": {
            target: {
                "feature_count": len(model.columns),
                "offset_pct": model.offset,
                "winsor_low_pct": model.winsor_low,
                "winsor_high_pct": model.winsor_high,
            }
            for target, model in models.items()
        },
        "policy": policy,
        "positive_return_days": int(positive_days),
        "by_day": by_day,
        "by_action": _group_metrics(scored, "action"),
        "by_entry_price_bin": _group_metrics(scored, "entry_price_bin"),
        "second_client_stats": client.stats.to_dict(),
        "frozen_gate": {
            "min_entry_coverage": MIN_ENTRY_COVERAGE,
            "min_second_coverage": MIN_SECOND_COVERAGE,
            "both_net_spearman_positive": True,
            "min_executed_rate": MIN_EXECUTED_RATE,
            "max_executed_rate": MAX_EXECUTED_RATE,
            "executed_day_balanced_base_must_be_positive": True,
            "all_opportunity_day_balanced_value_must_be_positive": True,
            "matched_difference_must_be_positive": True,
            "matched_bootstrap_low_must_be_positive": True,
            "severe_loss_no_worse_than_current": True,
            "min_positive_days": MIN_POSITIVE_DAYS,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "development_gate_pass": gate,
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
    parser.add_argument("--history-first-hot", type=Path, required=True)
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument("--calibration-positions", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.history_first_hot,
        args.fit_positions,
        args.calibration_positions,
        args.fresh_dir,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
