"""Request 195: shadow-entry repricing decomposition.

Separates the remaining attainable gross move from modeled execution drag at
causal 1-10 minute shadow states. Fresh Request-178 dates are development-only.
The chosen future exit is an oracle label and never an executable-policy claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .attention_replay import MINUTE_MS
from .decomposed_cost_aware_entry_surplus import (
    FRESH_DAYS,
    _gross_return,
)
from .execution_costs import (
    DEFAULT_EXECUTION_SCENARIOS,
    net_round_trip_return_pct,
)
from .future_cost_cover_state_observability import (
    FEATURES as BASE_FEATURES,
    _day_weights,
    _safe_spearman,
)
from .market_regime_position_value import (
    attach_market_regime,
    build_market_regime,
)
from .selected_hot_position_value_observability import (
    _base_return,
    _open_map,
)

REQUEST_ID = 195
EPISODE_KEYS = ["trading_day", "ticker", "hot_t"]
SHADOW_MIN_MINUTE = 1
SHADOW_MAX_MINUTE = 10
RESEARCH_CAP_MINUTES = 30
GROSS_SEED = 20261098
DRAG_SEED = 20261099

MIN_GROSS_SPEARMAN = 0.15
MIN_DRAG_SPEARMAN = 0.50
MIN_NET_SPEARMAN = 0.15
MIN_POSITIVE_NET_DAYS = 4
MIN_FIRST_RATE = 0.10
MAX_FIRST_RATE = 0.70
MIN_FIRST_POSITIVE_RATE = 0.60
MIN_FIRST_UPLIFT_PCT = 0.25
MIN_POSITIVE_FIRST_DAYS = 4

BASE_SCENARIO = next(
    item
    for item in DEFAULT_EXECUTION_SCENARIOS
    if item.name == "base"
)

EXTRA_FEATURES = (
    "shadow_move_1m_pct",
    "shadow_move_3m_pct",
    "shadow_move_5m_pct",
    "shadow_accel_1m_pct",
    "shadow_running_range_width_pct",
    "shadow_abs_move_consumed_pct",
    "shadow_zero_move_cost_proxy_pct",
)
FEATURES = tuple(
    dict.fromkeys([*BASE_FEATURES, *EXTRA_FEATURES])
)


@dataclass(frozen=True)
class Head:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    offset: float
    winsor_low: float
    winsor_high: float


def add_shadow_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_minute"] = pd.to_numeric(
        result["minutes_held"],
        errors="coerce",
    )
    result["_move"] = pd.to_numeric(
        result["entry_to_current_close_pct"],
        errors="coerce",
    )
    result = result.sort_values(
        EPISODE_KEYS + ["_minute"],
        kind="stable",
    )

    grouped = result.groupby(
        EPISODE_KEYS,
        sort=False,
    )["_move"]
    result["shadow_move_1m_pct"] = (
        result["_move"] - grouped.shift(1)
    )
    result["shadow_move_3m_pct"] = (
        result["_move"] - grouped.shift(3)
    )
    result["shadow_move_5m_pct"] = (
        result["_move"] - grouped.shift(5)
    )

    move1 = result["shadow_move_1m_pct"]
    result["shadow_accel_1m_pct"] = (
        move1
        - move1.groupby(
            [
                result["trading_day"],
                result["ticker"],
                result["hot_t"],
            ],
            sort=False,
        ).shift(1)
    )

    running_max = pd.to_numeric(
        result["running_max_return_pct"],
        errors="coerce",
    )
    running_min = pd.to_numeric(
        result["running_min_return_pct"],
        errors="coerce",
    )
    result["shadow_running_range_width_pct"] = (
        running_max - running_min
    )
    result["shadow_abs_move_consumed_pct"] = (
        result["_move"].abs()
    )

    log_close = pd.to_numeric(
        result["log_current_close"],
        errors="coerce",
    )
    prices = np.exp(log_close)
    proxy: list[float] = []
    for value in prices:
        if (
            pd.notna(value)
            and np.isfinite(float(value))
            and float(value) > 0
        ):
            net = net_round_trip_return_pct(
                float(value),
                0.0,
                BASE_SCENARIO,
            )
            proxy.append(float(-net))
        else:
            proxy.append(np.nan)
    result["shadow_zero_move_cost_proxy_pct"] = proxy

    return result.drop(
        columns=["_minute", "_move"]
    ).sort_index()


def build_labels(
    positions: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    frame = positions.copy()
    held = pd.to_numeric(
        frame["minutes_held"],
        errors="coerce",
    )
    frame = frame.loc[
        held.notna()
        & held.ge(SHADOW_MIN_MINUTE)
        & held.le(SHADOW_MAX_MINUTE)
    ].copy()
    if frame.empty:
        return frame

    opens = _open_map(scan)
    groups: dict[
        tuple[str, str],
        list[tuple[int, float]],
    ] = {}
    for (day, ticker, timestamp), price in opens.items():
        key = (str(day), str(ticker).upper())
        groups.setdefault(key, []).append(
            (int(timestamp), float(price))
        )
    for key in groups:
        groups[key].sort()

    entry_prices: list[float] = []
    best_exit_prices: list[float] = []
    best_gross: list[float] = []
    best_drag: list[float] = []
    best_base: list[float] = []
    exit_counts: list[int] = []

    for row in frame.to_dict("records"):
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        hot_t = int(row["hot_t"])
        state_t = int(row["state_t"])
        delayed_entry = opens.get(
            (day, ticker, state_t),
            np.nan,
        )
        if (
            pd.isna(delayed_entry)
            or not np.isfinite(float(delayed_entry))
            or float(delayed_entry) <= 0
        ):
            entry_prices.append(np.nan)
            best_exit_prices.append(np.nan)
            best_gross.append(np.nan)
            best_drag.append(np.nan)
            best_base.append(np.nan)
            exit_counts.append(0)
            continue

        cap_t = hot_t + RESEARCH_CAP_MINUTES * MINUTE_MS
        candidates: list[
            tuple[float, float, float, float]
        ] = []
        for timestamp, exit_price in groups.get(
            (day, ticker),
            [],
        ):
            if not (
                state_t < int(timestamp) <= cap_t
            ):
                continue
            base = _base_return(
                float(delayed_entry),
                float(exit_price),
            )
            gross = _gross_return(
                float(delayed_entry),
                float(exit_price),
            )
            if (
                pd.isna(base)
                or pd.isna(gross)
                or not np.isfinite(float(base))
                or not np.isfinite(float(gross))
            ):
                continue
            drag = float(gross) - float(base)
            candidates.append(
                (
                    float(base),
                    float(gross),
                    float(drag),
                    float(exit_price),
                )
            )

        entry_prices.append(float(delayed_entry))
        exit_counts.append(len(candidates))
        if not candidates:
            best_exit_prices.append(np.nan)
            best_gross.append(np.nan)
            best_drag.append(np.nan)
            best_base.append(np.nan)
            continue

        chosen = max(
            candidates,
            key=lambda item: item[0],
        )
        base, gross, drag, exit_price = chosen
        best_exit_prices.append(exit_price)
        best_gross.append(gross)
        best_drag.append(drag)
        best_base.append(base)

    frame["delayed_entry_reference_open"] = entry_prices
    frame["delayed_future_exit_count"] = exit_counts
    frame["oracle_best_exit_open"] = best_exit_prices
    frame["remaining_best_gross_pct"] = best_gross
    frame["remaining_best_drag_pct"] = best_drag
    frame["remaining_best_base_pct"] = best_base
    return frame


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(
        columns=columns
    ).apply(
        pd.to_numeric,
        errors="coerce",
    )


def train_head(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    target_column: str,
    seed: int,
) -> Head:
    target = pd.to_numeric(
        fit[target_column],
        errors="coerce",
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 800:
        raise ValueError(
            f"request 195 {target_column} insufficient fit support"
        )

    columns = tuple(
        column
        for column in FEATURES
        if column in train.columns
        and pd.to_numeric(
            train[column],
            errors="coerce",
        ).notna().any()
    )
    if not columns:
        raise ValueError(
            "request 195 has no causal repricing features"
        )

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
        _feature_frame(train, columns),
        y.clip(lower=low, upper=high),
        sample_weight=_day_weights(train),
    )

    cal_target = pd.to_numeric(
        calibration[target_column],
        errors="coerce",
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    if len(cal) < 400:
        raise ValueError(
            f"request 195 {target_column} insufficient calibration support"
        )

    raw = model.predict(
        _feature_frame(cal, columns)
    )
    residual = (
        cal_target.loc[cal_valid].to_numpy(
            dtype=float
        )
        - raw
    )
    offset = float(
        np.average(
            residual,
            weights=_day_weights(cal),
        )
    )
    return Head(
        model=model,
        columns=columns,
        offset=offset,
        winsor_low=float(low),
        winsor_high=float(high),
    )


def predict_head(
    frame: pd.DataFrame,
    head: Head,
) -> np.ndarray:
    return (
        head.model.predict(
            _feature_frame(
                frame,
                head.columns,
            )
        )
        + head.offset
    )


def score(
    frame: pd.DataFrame,
    gross_head: Head,
    drag_head: Head,
) -> pd.DataFrame:
    result = frame.copy()
    result["predicted_remaining_best_gross_pct"] = (
        predict_head(result, gross_head)
    )
    result["predicted_remaining_best_drag_pct"] = (
        np.maximum(
            predict_head(result, drag_head),
            0.0,
        )
    )
    result["predicted_new_entry_surplus_pct"] = (
        pd.to_numeric(
            result[
                "predicted_remaining_best_gross_pct"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            result[
                "predicted_remaining_best_drag_pct"
            ],
            errors="coerce",
        )
    )
    result["shadow_qualifies"] = (
        result["predicted_new_entry_surplus_pct"]
        .gt(0)
    )
    return result


def state_metrics(
    scored: pd.DataFrame,
) -> dict[str, object]:
    actual_gross = pd.to_numeric(
        scored["remaining_best_gross_pct"],
        errors="coerce",
    )
    actual_drag = pd.to_numeric(
        scored["remaining_best_drag_pct"],
        errors="coerce",
    )
    actual_base = pd.to_numeric(
        scored["remaining_best_base_pct"],
        errors="coerce",
    )
    pred_gross = pd.to_numeric(
        scored[
            "predicted_remaining_best_gross_pct"
        ],
        errors="coerce",
    )
    pred_drag = pd.to_numeric(
        scored[
            "predicted_remaining_best_drag_pct"
        ],
        errors="coerce",
    )
    pred_net = pd.to_numeric(
        scored[
            "predicted_new_entry_surplus_pct"
        ],
        errors="coerce",
    )
    valid = (
        actual_gross.notna()
        & actual_drag.notna()
        & actual_base.notna()
        & pred_gross.notna()
        & pred_drag.notna()
        & pred_net.notna()
    )
    part = scored.loc[valid].copy()
    actual_base = actual_base.loc[valid]
    pred_net = pred_net.loc[valid]

    selected = part.loc[
        part["shadow_qualifies"].fillna(False)
    ].copy()
    selected_values = pd.to_numeric(
        selected["remaining_best_base_pct"],
        errors="coerce",
    )
    all_mean = (
        float(actual_base.mean())
        if len(actual_base)
        else None
    )
    selected_mean = (
        float(selected_values.mean())
        if len(selected_values)
        else None
    )

    by_day: dict[str, object] = {}
    positive_spearman_days = 0
    for day in FRESH_DAYS:
        mask = (
            part["trading_day"]
            .astype(str)
            .eq(day)
        )
        day_part = part.loc[mask]
        corr = _safe_spearman(
            day_part["remaining_best_base_pct"],
            day_part[
                "predicted_new_entry_surplus_pct"
            ],
        )
        if corr is not None and corr > 0:
            positive_spearman_days += 1
        by_day[day] = {
            "rows": int(len(day_part)),
            "net_surplus_spearman": corr,
        }

    return {
        "rows": int(len(scored)),
        "evaluable_rows": int(valid.sum()),
        "gross_spearman": _safe_spearman(
            actual_gross.loc[valid],
            pred_gross.loc[valid],
        ),
        "drag_spearman": _safe_spearman(
            actual_drag.loc[valid],
            pred_drag.loc[valid],
        ),
        "net_surplus_spearman": _safe_spearman(
            actual_base,
            pred_net,
        ),
        "positive_net_spearman_days": int(
            positive_spearman_days
        ),
        "selected_rows": int(len(selected)),
        "selected_rate": (
            float(len(selected) / len(part))
            if len(part)
            else None
        ),
        "selected_positive_opportunity_rate": (
            float(selected_values.gt(0).mean())
            if len(selected_values)
            else None
        ),
        "selected_actual_base_mean_pct": selected_mean,
        "all_shadow_actual_base_mean_pct": all_mean,
        "selected_minus_all_uplift_pct": (
            float(selected_mean - all_mean)
            if selected_mean is not None
            and all_mean is not None
            else None
        ),
        "by_day": by_day,
    }


def first_qualifying_bridge(
    scored: pd.DataFrame,
) -> dict[str, object]:
    total_episodes = int(
        scored.loc[:, EPISODE_KEYS]
        .drop_duplicates()
        .shape[0]
    )
    rows: list[dict[str, object]] = []

    for _, group in scored.groupby(
        EPISODE_KEYS,
        sort=False,
    ):
        work = group.copy()
        work["_minute"] = pd.to_numeric(
            work["minutes_held"],
            errors="coerce",
        )
        work = work.sort_values(
            "_minute",
            kind="stable",
        )
        qualified = work.loc[
            work["shadow_qualifies"]
            .fillna(False)
        ]
        if qualified.empty:
            continue
        row = qualified.iloc[0]
        rows.append(
            {
                "trading_day": str(
                    row["trading_day"]
                ),
                "ticker": str(
                    row["ticker"]
                ).upper(),
                "hot_t": int(row["hot_t"]),
                "minute": float(
                    row["_minute"]
                ),
                "actual_base_pct": float(
                    row["remaining_best_base_pct"]
                ),
            }
        )

    selected = pd.DataFrame(rows)
    values = (
        pd.to_numeric(
            selected["actual_base_pct"],
            errors="coerce",
        )
        if len(selected)
        else pd.Series(dtype=float)
    )
    minutes = (
        pd.to_numeric(
            selected["minute"],
            errors="coerce",
        )
        if len(selected)
        else pd.Series(dtype=float)
    )
    all_values = pd.to_numeric(
        scored["remaining_best_base_pct"],
        errors="coerce",
    ).dropna()

    selected_mean = (
        float(values.mean())
        if len(values)
        else None
    )
    all_mean = (
        float(all_values.mean())
        if len(all_values)
        else None
    )

    by_day: dict[str, object] = {}
    positive_days = 0
    for day in FRESH_DAYS:
        part = (
            selected.loc[
                selected["trading_day"]
                .astype(str)
                .eq(day)
            ]
            if len(selected)
            else selected
        )
        day_values = (
            pd.to_numeric(
                part["actual_base_pct"],
                errors="coerce",
            )
            if len(part)
            else pd.Series(dtype=float)
        )
        mean = (
            float(day_values.mean())
            if len(day_values)
            else None
        )
        if mean is not None and mean > 0:
            positive_days += 1
        by_day[day] = {
            "qualified_episodes": int(
                len(part)
            ),
            "actual_base_mean_pct": mean,
            "positive_rate": (
                float(
                    day_values.gt(0).mean()
                )
                if len(day_values)
                else None
            ),
        }

    return {
        "total_episodes": total_episodes,
        "qualified_episodes": int(
            len(selected)
        ),
        "qualified_episode_rate": (
            float(
                len(selected)
                / total_episodes
            )
            if total_episodes
            else None
        ),
        "first_qualifying_minute_mean": (
            float(minutes.mean())
            if len(minutes)
            else None
        ),
        "first_qualifying_minute_median": (
            float(minutes.median())
            if len(minutes)
            else None
        ),
        "first_qualifying_minute_p90": (
            float(
                minutes.quantile(0.90)
            )
            if len(minutes)
            else None
        ),
        "actual_base_mean_pct": selected_mean,
        "positive_opportunity_rate": (
            float(values.gt(0).mean())
            if len(values)
            else None
        ),
        "all_shadow_actual_base_mean_pct": all_mean,
        "selected_minus_all_uplift_pct": (
            float(selected_mean - all_mean)
            if selected_mean is not None
            and all_mean is not None
            else None
        ),
        "positive_mean_days": int(
            positive_days
        ),
        "by_day": by_day,
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    fresh_dir: Path,
    output_path: Path,
    scored_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(
        fit_path
    )
    calibration_positions = (
        pd.read_parquet(
            calibration_path
        )
    )
    history_scan = pd.read_parquet(
        history_scan_path
    )

    historical_days = set(
        fit_positions["trading_day"]
        .astype(str)
    ) | set(
        calibration_positions[
            "trading_day"
        ].astype(str)
    )
    hist_scan = history_scan.loc[
        history_scan["trading_day"]
        .astype(str)
        .isin(historical_days)
    ].copy()
    hist_regime = build_market_regime(
        hist_scan
    )

    fit_positions = attach_market_regime(
        add_shadow_features(
            fit_positions
        ),
        hist_regime,
    )
    calibration_positions = (
        attach_market_regime(
            add_shadow_features(
                calibration_positions
            ),
            hist_regime,
        )
    )
    fit = build_labels(
        fit_positions,
        hist_scan,
    )
    calibration = build_labels(
        calibration_positions,
        hist_scan,
    )

    gross_head = train_head(
        fit,
        calibration,
        target_column=(
            "remaining_best_gross_pct"
        ),
        seed=GROSS_SEED,
    )
    drag_head = train_head(
        fit,
        calibration,
        target_column=(
            "remaining_best_drag_pct"
        ),
        seed=DRAG_SEED,
    )

    position_paths = sorted(
        fresh_dir.glob(
            "*-positions.parquet"
        )
    )
    scan_paths = sorted(
        fresh_dir.glob("*-scan.parquet")
    )
    if (
        len(position_paths)
        != len(FRESH_DAYS)
        or len(scan_paths)
        != len(FRESH_DAYS)
    ):
        raise ValueError(
            "request 195 requires complete request 178 shards"
        )

    fresh_positions = pd.concat(
        [
            pd.read_parquet(path)
            for path in position_paths
        ],
        ignore_index=True,
    )
    fresh_scan = pd.concat(
        [
            pd.read_parquet(path)
            for path in scan_paths
        ],
        ignore_index=True,
    )
    found = sorted(
        fresh_positions[
            "trading_day"
        ].astype(str).unique()
    )
    if found != list(FRESH_DAYS):
        raise ValueError(
            f"request 195 expected {FRESH_DAYS}, found {found}"
        )

    fresh_regime = build_market_regime(
        fresh_scan
    )
    fresh_positions = attach_market_regime(
        add_shadow_features(
            fresh_positions
        ),
        fresh_regime,
    )
    fresh = build_labels(
        fresh_positions,
        fresh_scan,
    )
    scored = score(
        fresh,
        gross_head,
        drag_head,
    )

    state = state_metrics(scored)
    bridge = first_qualifying_bridge(
        scored
    )

    gate = bool(
        state["gross_spearman"]
        is not None
        and float(
            state["gross_spearman"]
        )
        >= MIN_GROSS_SPEARMAN
        and state["drag_spearman"]
        is not None
        and float(
            state["drag_spearman"]
        )
        >= MIN_DRAG_SPEARMAN
        and state["net_surplus_spearman"]
        is not None
        and float(
            state[
                "net_surplus_spearman"
            ]
        )
        >= MIN_NET_SPEARMAN
        and int(
            state[
                "positive_net_spearman_days"
            ]
        )
        >= MIN_POSITIVE_NET_DAYS
        and bridge[
            "qualified_episode_rate"
        ]
        is not None
        and MIN_FIRST_RATE
        <= float(
            bridge[
                "qualified_episode_rate"
            ]
        )
        <= MAX_FIRST_RATE
        and bridge[
            "positive_opportunity_rate"
        ]
        is not None
        and float(
            bridge[
                "positive_opportunity_rate"
            ]
        )
        >= MIN_FIRST_POSITIVE_RATE
        and bridge[
            "actual_base_mean_pct"
        ]
        is not None
        and float(
            bridge[
                "actual_base_mean_pct"
            ]
        )
        > 0
        and bridge[
            "selected_minus_all_uplift_pct"
        ]
        is not None
        and float(
            bridge[
                "selected_minus_all_uplift_pct"
            ]
        )
        >= MIN_FIRST_UPLIFT_PCT
        and int(
            bridge["positive_mean_days"]
        )
        >= MIN_POSITIVE_FIRST_DAYS
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "oracle_opportunity_only": True,
        "state_metrics": state,
        "episode_bridge": bridge,
        "model": {
            "gross_seed": GROSS_SEED,
            "drag_seed": DRAG_SEED,
            "gross_columns": list(
                gross_head.columns
            ),
            "drag_columns": list(
                drag_head.columns
            ),
            "gross_offset": (
                gross_head.offset
            ),
            "drag_offset": (
                drag_head.offset
            ),
        },
        "frozen_gate": {
            "min_gross_spearman": MIN_GROSS_SPEARMAN,
            "min_drag_spearman": MIN_DRAG_SPEARMAN,
            "min_net_spearman": MIN_NET_SPEARMAN,
            "min_positive_net_days": MIN_POSITIVE_NET_DAYS,
            "first_rate_range": [
                MIN_FIRST_RATE,
                MAX_FIRST_RATE,
            ],
            "min_first_positive_rate": MIN_FIRST_POSITIVE_RATE,
            "min_first_uplift_pct": MIN_FIRST_UPLIFT_PCT,
            "min_positive_first_days": MIN_POSITIVE_FIRST_DAYS,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    scored_output_path.parent.mkdir(
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
    scored.to_parquet(
        scored_output_path,
        index=False,
        compression="zstd",
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
        "--fresh-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--scored-output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.fresh_dir,
        args.output,
        args.scored_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
