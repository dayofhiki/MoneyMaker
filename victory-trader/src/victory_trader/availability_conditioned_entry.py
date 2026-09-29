"""Request241: availability-conditioned event-time entry timing.

Request240 established that continuation/executability is a separate and highly
learnable dimension. Request241 reproduces that idea on the older Request171 /
Request178 entry-timing population, then uses the availability probability as a
frozen episode-level input to a new ENTER-vs-WAIT classifier.

Key changes from Request227:
* WATCH decisions follow actually observed executable states, not exact clock
  minutes;
* continuation availability and entry timing are separate heads;
* the timing target is classificatory: ENTER now only when the current
  multi-event value is positive and no later WATCH state in the five-state
  window is better;
* no raw predicted-return > 0 boundary is used.

June 23-29 is an already-opened development block. It is never used to fit or
choose thresholds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .recurrent_wait_entry_action_value import _base_return
from .state_action_risk import SEVERE_LOSS_PCT
from .wait_reobserve_entry_confirmation import WAIT_STATE_FEATURES, _day_weights

REQUEST_ID = 241
AVAIL_SEED = 20261312
ENTRY_SEED = 20261313
WATCH_EVENTS = 5
EVENT_HORIZONS = (1, 2, 3, 5)

MIN_AVAIL_AUC = 0.75
MIN_SELECTED_AVAILABILITY = 0.85
MIN_ENTRY_AUC = 0.58
MIN_EXECUTION_RATE = 0.03
MAX_EXECUTION_RATE = 0.40
MIN_TRADES = 20
MIN_POSITIVE_DAYS = 3

TRANSITION_FEATURES = (
    "watch_event_index",
    "elapsed_minutes_since_hot",
    "gap_from_previous_event_minutes",
    "event_price_change_pct",
    "event_return_from_start_pct",
    "event_drawdown_from_peak_pct",
    "event_rebound_from_trough_pct",
    "event_attention_change",
    "event_rank_change",
)
TIMING_FEATURES = tuple(
    dict.fromkeys(
        [
            *WAIT_STATE_FEATURES,
            *TRANSITION_FEATURES,
            "episode_availability_probability",
        ]
    )
)


@dataclass(frozen=True)
class ClassifierHead:
    model: HistGradientBoostingClassifier
    columns: tuple[str, ...]


def _number(value: object) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return (
        float(parsed)
        if pd.notna(parsed) and np.isfinite(float(parsed))
        else np.nan
    )


def _pct_change(current: float, previous: float) -> float:
    if not (
        np.isfinite(current)
        and np.isfinite(previous)
        and previous > 0
    ):
        return np.nan
    return float((current / previous - 1.0) * 100.0)


def build_event_watch_states(
    position_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build first five actually observed executable WATCH states."""
    records: list[dict[str, object]] = []

    for keys, group in position_rows.groupby(EPISODE_KEYS, sort=False):
        work = group.copy()
        work["_state_t"] = pd.to_numeric(
            work["state_t"], errors="coerce"
        )
        work["_entry_open"] = pd.to_numeric(
            work["exit_reference_open"], errors="coerce"
        )
        work = work.loc[
            work["_state_t"].notna()
            & work["_entry_open"].gt(0)
            & work["_state_t"].gt(int(keys[2]))
        ].copy()
        work = (
            work.sort_values("_state_t", kind="stable")
            .drop_duplicates("_state_t", keep="first")
        )
        if work.empty:
            continue

        watch = work.head(WATCH_EVENTS)
        first_price = np.nan
        peak = np.nan
        trough = np.nan
        previous_price = np.nan
        previous_t = np.nan
        previous_attention = np.nan
        previous_rank = np.nan

        for event_index, (_, row) in enumerate(
            watch.iterrows(),
            start=1,
        ):
            state_t = int(row["_state_t"])
            entry_open = float(row["_entry_open"])
            log_close = _number(row.get("log_current_close"))
            current_price = (
                float(np.exp(log_close))
                if np.isfinite(log_close)
                else entry_open
            )
            elapsed = float(
                (state_t - int(keys[2])) / 60_000.0
            )

            if not np.isfinite(first_price):
                first_price = current_price
            peak = (
                current_price
                if not np.isfinite(peak)
                else max(peak, current_price)
            )
            trough = (
                current_price
                if not np.isfinite(trough)
                else min(trough, current_price)
            )

            attention = _number(row.get("attention_score"))
            rank = _number(row.get("attention_rank"))

            record: dict[str, object] = {
                "trading_day": str(keys[0]),
                "ticker": str(keys[1]).upper(),
                "hot_t": int(keys[2]),
                "state_t": state_t,
                "watch_event_index": event_index,
                "elapsed_minutes_since_hot": elapsed,
                "entry_reference_open": entry_open,
                "current_price": current_price,
                "gap_from_previous_event_minutes": (
                    float((state_t - previous_t) / 60_000.0)
                    if np.isfinite(previous_t)
                    else np.nan
                ),
                "event_price_change_pct": _pct_change(
                    current_price,
                    previous_price,
                ),
                "event_return_from_start_pct": _pct_change(
                    current_price,
                    first_price,
                ),
                "event_drawdown_from_peak_pct": _pct_change(
                    current_price,
                    peak,
                ),
                "event_rebound_from_trough_pct": _pct_change(
                    current_price,
                    trough,
                ),
                "event_attention_change": (
                    attention - previous_attention
                    if np.isfinite(attention)
                    and np.isfinite(previous_attention)
                    else np.nan
                ),
                "event_rank_change": (
                    rank - previous_rank
                    if np.isfinite(rank)
                    and np.isfinite(previous_rank)
                    else np.nan
                ),
            }
            for column in WAIT_STATE_FEATURES:
                record[column] = row.get(column, np.nan)

            future = work.loc[
                work["_state_t"].gt(state_t)
            ].sort_values("_state_t", kind="stable")
            values: list[float] = []
            complete = len(future) >= max(EVENT_HORIZONS)
            if complete:
                for horizon in EVENT_HORIZONS:
                    exit_open = float(
                        future.iloc[horizon - 1]["_entry_open"]
                    )
                    value = _base_return(
                        entry_open,
                        exit_open,
                    )
                    if not np.isfinite(value):
                        complete = False
                        break
                    record[
                        f"entry_event{horizon}_base_pct"
                    ] = float(value)
                    values.append(float(value))
            if complete:
                record["entry_multi_event_value_pct"] = float(
                    np.mean(values)
                )
                record["entry_event3_realized_base_pct"] = float(
                    record["entry_event3_base_pct"]
                )
            else:
                record["entry_multi_event_value_pct"] = np.nan
                record["entry_event3_realized_base_pct"] = np.nan

            records.append(record)
            previous_price = current_price
            previous_t = float(state_t)
            previous_attention = attention
            previous_rank = rank

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result["episode_continuation_available"] = 0
    result["enter_now_best_positive"] = np.nan

    for _, group in result.groupby(EPISODE_KEYS, sort=False):
        ordered = group.sort_values(
            "watch_event_index",
            kind="stable",
        )
        values = pd.to_numeric(
            ordered["entry_multi_event_value_pct"],
            errors="coerce",
        )
        availability = int(values.notna().sum() >= 2)
        result.loc[
            ordered.index,
            "episode_continuation_available",
        ] = availability

        for idx in ordered.index:
            event_index = int(
                result.at[idx, "watch_event_index"]
            )
            current = _number(
                result.at[idx, "entry_multi_event_value_pct"]
            )
            if not np.isfinite(current):
                continue
            later = ordered.loc[
                pd.to_numeric(
                    ordered["watch_event_index"],
                    errors="coerce",
                ).gt(event_index),
                "entry_multi_event_value_pct",
            ]
            later_values = pd.to_numeric(
                later,
                errors="coerce",
            ).dropna()
            best_later = (
                max(0.0, float(later_values.max()))
                if len(later_values)
                else 0.0
            )
            result.at[idx, "enter_now_best_positive"] = int(
                current > 0
                and current >= best_later
            )

    return result


def _feature_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)


def _usable_columns(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        column
        for column in candidates
        if column in frame.columns
        and pd.to_numeric(
            frame[column], errors="coerce"
        ).notna().any()
    )


def _fit_classifier(
    frame: pd.DataFrame,
    target_column: str,
    candidates: tuple[str, ...],
    seed: int,
) -> ClassifierHead:
    target = pd.to_numeric(
        frame[target_column],
        errors="coerce",
    )
    train = frame.loc[target.notna()].copy()
    y = pd.to_numeric(
        train[target_column],
        errors="raise",
    ).astype(int)
    if len(train) < 250:
        raise ValueError(
            f"request241 needs >=250 rows for {target_column}, "
            f"got {len(train)}"
        )
    if y.nunique() < 2:
        raise ValueError(
            f"request241 {target_column} needs both classes"
        )
    columns = _usable_columns(train, candidates)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _feature_frame(train, columns),
        y,
        sample_weight=_day_weights(train),
    )
    return ClassifierHead(model=model, columns=columns)


def _predict(
    frame: pd.DataFrame,
    fitted: ClassifierHead,
) -> np.ndarray:
    return fitted.model.predict_proba(
        _feature_frame(frame, fitted.columns)
    )[:, 1]


def _safe_auc(
    actual: pd.Series,
    score: pd.Series,
) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = a.notna() & s.notna()
    if int(valid.sum()) < 20:
        return None
    y = a.loc[valid].astype(int)
    if y.nunique() < 2:
        return None
    return float(
        roc_auc_score(
            y,
            s.loc[valid].astype(float),
        )
    )


def _episode_first(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values(
            [*EPISODE_KEYS, "watch_event_index"],
            kind="stable",
        )
        .groupby(EPISODE_KEYS, sort=False)
        .head(1)
        .copy()
    )


def _availability_oof(
    fit_states: pd.DataFrame,
) -> tuple[pd.DataFrame, ClassifierHead]:
    first = _episode_first(fit_states)
    first["episode_availability_probability"] = np.nan
    feature_candidates = tuple(
        dict.fromkeys(
            [
                *WAIT_STATE_FEATURES,
                *TRANSITION_FEATURES,
            ]
        )
    )
    days = sorted(
        first["trading_day"].astype(str).unique()
    )
    for held_day in days:
        train = first.loc[
            ~first["trading_day"].astype(str).eq(held_day)
        ].copy()
        held = first.loc[
            first["trading_day"].astype(str).eq(held_day)
        ].copy()
        fitted = _fit_classifier(
            train,
            "episode_continuation_available",
            feature_candidates,
            AVAIL_SEED,
        )
        first.loc[
            held.index,
            "episode_availability_probability",
        ] = _predict(held, fitted)

    final = _fit_classifier(
        first,
        "episode_continuation_available",
        feature_candidates,
        AVAIL_SEED,
    )
    return first, final


def _merge_episode_probability(
    states: pd.DataFrame,
    first_scores: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        *EPISODE_KEYS,
        "episode_availability_probability",
    ]
    return states.drop(
        columns=["episode_availability_probability"],
        errors="ignore",
    ).merge(
        first_scores.loc[:, columns],
        on=EPISODE_KEYS,
        how="left",
        validate="many_to_one",
    )


def _score_episode_probability(
    states: pd.DataFrame,
    fitted: ClassifierHead,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = _episode_first(states)
    first["episode_availability_probability"] = _predict(
        first,
        fitted,
    )
    return (
        _merge_episode_probability(states, first),
        first,
    )


def _portfolio_metrics(
    universe: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, object]:
    base = universe.loc[:, EPISODE_KEYS].drop_duplicates().copy()
    if trades.empty:
        base["realized_base_return_pct"] = 0.0
        base["executed"] = False
    else:
        selected = trades.loc[
            :,
            EPISODE_KEYS + ["realized_base_return_pct"],
        ].copy()
        base = base.merge(
            selected,
            on=EPISODE_KEYS,
            how="left",
            validate="one_to_one",
        )
        base["executed"] = base[
            "realized_base_return_pct"
        ].notna()
        base["realized_base_return_pct"] = pd.to_numeric(
            base["realized_base_return_pct"],
            errors="coerce",
        ).fillna(0.0)

    daily = base.groupby(
        base["trading_day"].astype(str),
        sort=True,
    )["realized_base_return_pct"].mean()
    executed = base.loc[base["executed"]].copy()
    values = pd.to_numeric(
        executed["realized_base_return_pct"],
        errors="coerce",
    )

    return {
        "episodes": int(len(base)),
        "executed": int(base["executed"].sum()),
        "execution_rate": float(base["executed"].mean()),
        "portfolio_mean_pct": float(
            base["realized_base_return_pct"].mean()
        ),
        "portfolio_day_balanced_mean_pct": float(
            daily.mean()
        ),
        "positive_portfolio_days": int(
            (daily > 0).sum()
        ),
        "trade_mean_pct": (
            float(values.mean()) if len(values) else None
        ),
        "trade_positive_rate": (
            float(values.gt(0).mean())
            if len(values) else None
        ),
        "trade_severe_loss_rate": (
            float(values.le(SEVERE_LOSS_PCT).mean())
            if len(values) else None
        ),
        "by_day": {
            str(day): float(value)
            for day, value in daily.items()
        },
    }


def _runtime_policy(
    states: pd.DataFrame,
    availability_gate: float,
    entry_gate: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for keys, group in states.groupby(EPISODE_KEYS, sort=False):
        ordered = group.sort_values(
            "watch_event_index",
            kind="stable",
        )
        episode_probability = _number(
            ordered.iloc[0].get(
                "episode_availability_probability"
            )
        )
        if (
            not np.isfinite(episode_probability)
            or episode_probability < availability_gate
        ):
            continue

        chosen = None
        for _, row in ordered.iterrows():
            probability = _number(
                row.get("enter_now_probability")
            )
            if (
                np.isfinite(probability)
                and probability >= entry_gate
            ):
                chosen = row
                break
        if chosen is None:
            continue

        realized = _number(
            chosen.get("entry_event3_realized_base_pct")
        )
        if not np.isfinite(realized):
            continue
        records.append(
            {
                "trading_day": str(keys[0]),
                "ticker": str(keys[1]).upper(),
                "hot_t": int(keys[2]),
                "watch_event_index": int(
                    chosen["watch_event_index"]
                ),
                "elapsed_minutes_since_hot": float(
                    chosen["elapsed_minutes_since_hot"]
                ),
                "episode_availability_probability": (
                    episode_probability
                ),
                "enter_now_probability": float(
                    chosen["enter_now_probability"]
                ),
                "realized_base_return_pct": realized,
                "chosen_multi_event_value_pct": _number(
                    chosen.get(
                        "entry_multi_event_value_pct"
                    )
                ),
            }
        )
    return pd.DataFrame(records)


def _first_event_baseline(
    states: pd.DataFrame,
    availability_gate: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for keys, group in states.groupby(EPISODE_KEYS, sort=False):
        ordered = group.sort_values(
            "watch_event_index",
            kind="stable",
        )
        first = ordered.iloc[0]
        probability = _number(
            first.get("episode_availability_probability")
        )
        if (
            not np.isfinite(probability)
            or probability < availability_gate
        ):
            continue
        realized = _number(
            first.get("entry_event3_realized_base_pct")
        )
        if not np.isfinite(realized):
            continue
        records.append(
            {
                "trading_day": str(keys[0]),
                "ticker": str(keys[1]).upper(),
                "hot_t": int(keys[2]),
                "realized_base_return_pct": realized,
            }
        )
    return pd.DataFrame(records)


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_positions_path)
    calibration_positions = pd.read_parquet(
        calibration_positions_path
    )
    fresh_paths = sorted(
        fresh_dir.glob("*-positions.parquet")
    )
    if len(fresh_paths) != len(FRESH_DAYS):
        raise ValueError(
            "request241 requires complete Request178 position shards"
        )
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in fresh_paths],
        ignore_index=True,
    )

    fit_states = build_event_watch_states(fit_positions)
    cal_states = build_event_watch_states(
        calibration_positions
    )
    fresh_states = build_event_watch_states(fresh_positions)

    fit_first_oof, availability_model = _availability_oof(
        fit_states
    )
    fit_states = _merge_episode_probability(
        fit_states,
        fit_first_oof,
    )
    cal_states, cal_first = _score_episode_probability(
        cal_states,
        availability_model,
    )
    fresh_states, fresh_first = _score_episode_probability(
        fresh_states,
        availability_model,
    )

    availability_gate = float(
        np.quantile(
            pd.to_numeric(
                cal_first[
                    "episode_availability_probability"
                ],
                errors="coerce",
            ).dropna(),
            0.75,
        )
    )

    timing_model = _fit_classifier(
        fit_states,
        "enter_now_best_positive",
        TIMING_FEATURES,
        ENTRY_SEED,
    )
    cal_states["enter_now_probability"] = _predict(
        cal_states,
        timing_model,
    )
    fresh_states["enter_now_probability"] = _predict(
        fresh_states,
        timing_model,
    )

    cal_admitted = cal_states.loc[
        pd.to_numeric(
            cal_states["episode_availability_probability"],
            errors="coerce",
        ).ge(availability_gate)
        & pd.to_numeric(
            cal_states["enter_now_best_positive"],
            errors="coerce",
        ).notna()
    ].copy()
    if len(cal_admitted) < 100:
        raise ValueError(
            "request241 needs >=100 admitted calibration timing rows"
        )
    entry_gate = float(
        np.quantile(
            pd.to_numeric(
                cal_admitted["enter_now_probability"],
                errors="coerce",
            ).dropna(),
            0.75,
        )
    )

    availability_auc = _safe_auc(
        fresh_first["episode_continuation_available"],
        fresh_first["episode_availability_probability"],
    )
    runner_auc = (
        _safe_auc(
            fresh_first["episode_continuation_available"],
            fresh_first["second_rerank_probability"],
        )
        if "second_rerank_probability" in fresh_first.columns
        else None
    )
    admitted_first = fresh_first.loc[
        pd.to_numeric(
            fresh_first[
                "episode_availability_probability"
            ],
            errors="coerce",
        ).ge(availability_gate)
    ].copy()
    selected_availability = (
        float(
            admitted_first[
                "episode_continuation_available"
            ].mean()
        )
        if len(admitted_first)
        else None
    )

    timing_eval = fresh_states.loc[
        pd.to_numeric(
            fresh_states[
                "episode_availability_probability"
            ],
            errors="coerce",
        ).ge(availability_gate)
    ].copy()
    entry_auc = _safe_auc(
        timing_eval["enter_now_best_positive"],
        timing_eval["enter_now_probability"],
    )

    candidate_trades = _runtime_policy(
        fresh_states,
        availability_gate,
        entry_gate,
    )
    baseline_trades = _first_event_baseline(
        fresh_states,
        availability_gate,
    )
    universe = fresh_states.loc[
        :, EPISODE_KEYS
    ].drop_duplicates()

    candidate_metrics = _portfolio_metrics(
        universe,
        candidate_trades,
    )
    baseline_metrics = _portfolio_metrics(
        universe,
        baseline_trades,
    )

    candidate_day = candidate_metrics[
        "portfolio_day_balanced_mean_pct"
    ]
    baseline_day = baseline_metrics[
        "portfolio_day_balanced_mean_pct"
    ]
    execution_rate = candidate_metrics["execution_rate"]
    trade_mean = candidate_metrics["trade_mean_pct"]
    candidate_severe = candidate_metrics[
        "trade_severe_loss_rate"
    ]
    baseline_severe = baseline_metrics[
        "trade_severe_loss_rate"
    ]

    gate = bool(
        availability_auc is not None
        and availability_auc >= MIN_AVAIL_AUC
        and selected_availability is not None
        and selected_availability
        >= MIN_SELECTED_AVAILABILITY
        and entry_auc is not None
        and entry_auc >= MIN_ENTRY_AUC
        and MIN_EXECUTION_RATE
        <= execution_rate
        <= MAX_EXECUTION_RATE
        and candidate_metrics["executed"] >= MIN_TRADES
        and trade_mean is not None
        and trade_mean > 0
        and candidate_day > 0
        and candidate_day > baseline_day
        and candidate_metrics["positive_portfolio_days"]
        >= MIN_POSITIVE_DAYS
        and candidate_severe is not None
        and baseline_severe is not None
        and candidate_severe <= baseline_severe
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "development_only": True,
        "promotion_eligible": False,
        "architecture": {
            "stage1": (
                "episode continuation/executability classifier"
            ),
            "stage2": (
                "event-time ENTER-now-best-positive classifier"
            ),
            "watch_clock": (
                "first five actually observed executable states"
            ),
            "raw_return_zero_boundary_used": False,
        },
        "labels": {
            "availability": (
                "at least two evaluable multi-event entry states "
                "inside the first five observed WATCH states"
            ),
            "timing": (
                "current multi-event value > 0 and no later WATCH "
                "state in the five-state window has greater value"
            ),
            "realized_economic_outcome": (
                "BASE return at the third subsequently observed "
                "executable state after chosen entry"
            ),
        },
        "thresholds": {
            "availability_probability_gate": availability_gate,
            "entry_probability_gate": entry_gate,
            "both_from_calibration_only": True,
        },
        "fresh_development": {
            "availability_auc": availability_auc,
            "runner_availability_auc": runner_auc,
            "availability_auc_gain_vs_runner": (
                float(availability_auc - runner_auc)
                if availability_auc is not None
                and runner_auc is not None
                else None
            ),
            "admitted_episode_count": int(
                len(admitted_first)
            ),
            "admitted_availability_rate": (
                selected_availability
            ),
            "entry_timing_auc": entry_auc,
            "candidate": candidate_metrics,
            "first_event_baseline": baseline_metrics,
        },
        "frozen_gate": {
            "min_availability_auc": MIN_AVAIL_AUC,
            "min_admitted_availability_rate": (
                MIN_SELECTED_AVAILABILITY
            ),
            "min_entry_auc": MIN_ENTRY_AUC,
            "min_execution_rate": MIN_EXECUTION_RATE,
            "max_execution_rate": MAX_EXECUTION_RATE,
            "min_trades": MIN_TRADES,
            "candidate_trade_mean_must_be_positive": True,
            "candidate_portfolio_day_mean_must_be_positive": True,
            "candidate_must_beat_first_event_baseline": True,
            "min_positive_portfolio_days": (
                MIN_POSITIVE_DAYS
            ),
            "severe_loss_no_worse_than_baseline": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
        "next_boundary_if_pass": (
            "freeze availability-conditioned event-time entry timing, "
            "then integrate a causal HOLD/EXIT controller and validate "
            "the full trajectory on a genuinely new holdout block."
        ),
        "next_boundary_if_fail": (
            "separate whether the failure is availability transfer or "
            "entry-timing economics; do not return to global HOT "
            "admission thresholds."
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    rows_output_path.parent.mkdir(
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
    pd.concat(
        [
            candidate_trades.assign(
                policy="request241"
            ),
            baseline_trades.assign(
                policy="availability_first_event"
            ),
        ],
        ignore_index=True,
    ).to_parquet(
        rows_output_path,
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
        "--fit-positions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration-positions",
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
        "--rows-output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.fresh_dir,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
