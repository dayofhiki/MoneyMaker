"""Request248: tradable positive-opportunity ceiling and learnability audit.

No future-best state is used as a supervised target. Every admitted state receives
its own forced-entry value under the frozen Request247 position policy. Episode
maxima are diagnostic oracle summaries only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from .causal_tradability_admission import (
    TRAIN_DAYS,
    fetch_second_paths,
    fit_admission,
    label_tradability,
    probability,
)
from .chronological_action_value import (
    CAL_DAYS,
    CAUSAL_SOURCE_FEATURES,
    EXIT_DEADLINE,
    FIT_DAYS,
    KEYS,
    PATH_FEATURES,
    build_states,
    features,
)
from .config import load_settings
from .massive_client import MassiveClient
from .second_execution_reconstruction import (
    EXPIRY_MS,
    PRIMARY_LATENCY_MS,
    first_observed_open,
    modeled_return,
)
from .tradability_conditioned_entry_value import (
    ADMISSION_THRESHOLD,
    VALUE_TRAIN_DAYS,
)
from .tradability_conditioned_position_value import (
    build_position_targets,
    fit_position_model,
)

REQUEST_ID = 248
LEARN_DAYS = VALUE_TRAIN_DAYS
TEST_DAYS = CAL_DAYS
MINUTE_MS = 60_000
MIN_VALUE_ROWS = 150


def economic_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        c
        for c in dict.fromkeys(
            [*CAUSAL_SOURCE_FEATURES, *PATH_FEATURES, "tradability_probability"]
        )
        if c in frame and pd.to_numeric(frame[c], errors="coerce").notna().any()
    )


def frozen_exit_time(
    part: pd.DataFrame,
    entry_index: int,
    predicted_hold: pd.Series,
) -> int:
    later = part.loc[part.index > entry_index]
    for idx, row in later.iterrows():
        if bool(row["terminal"]):
            return int(row["state_t"])
        if float(predicted_hold.loc[idx]) > 0:
            continue
        return int(row["state_t"])
    return int(part.iloc[0].hot_t) + EXIT_DEADLINE * MINUTE_MS


def forced_state_values(
    states: pd.DataFrame,
    admission_probs: np.ndarray,
    position_model,
    position_columns: tuple[str, ...],
    paths: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    work = states.reset_index(drop=True).copy()
    work["tradability_probability"] = admission_probs
    work["predicted_hold_advantage"] = position_model.predict(
        features(work, position_columns)
    )
    predicted_hold = work["predicted_hold_advantage"]
    rows = []
    for key, part in work.groupby(KEYS, sort=False):
        part = part.sort_values("state_t")
        day, ticker, _ = key
        path = paths.get(
            (str(day), str(ticker).upper()),
            (np.array([], dtype=np.int64), np.array([], dtype=float)),
        )
        for idx, row in part.iterrows():
            if (
                not bool(row["can_enter"])
                or bool(row["terminal"])
                or float(row["tradability_probability"]) < ADMISSION_THRESHOLD
            ):
                continue
            entry_t = int(row["state_t"])
            exit_t = frozen_exit_time(part, idx, predicted_hold)
            times, opens = path
            entry_fill_t, entry_ref = first_observed_open(
                times,
                opens,
                decision_t=entry_t,
                latency_ms=PRIMARY_LATENCY_MS,
                expiry_ms=EXPIRY_MS,
            )
            exit_fill_t = None
            exit_ref = None
            if entry_ref is not None:
                exit_fill_t, exit_ref = first_observed_open(
                    times,
                    opens,
                    decision_t=exit_t,
                    latency_ms=PRIMARY_LATENCY_MS,
                    expiry_ms=EXPIRY_MS,
                )
            record = row.to_dict()
            record.update(
                forced_exit_t=exit_t,
                entry_fill_t=entry_fill_t,
                exit_fill_t=exit_fill_t,
                forced_base_return=np.nan,
                forced_stress_return=np.nan,
                forced_status="closed",
            )
            if entry_ref is None:
                record["forced_status"] = "entry_unavailable"
            elif exit_ref is None:
                record["forced_status"] = "exit_unavailable"
            else:
                record["forced_base_return"] = modeled_return(
                    float(entry_ref), float(exit_ref), __import__(
                        "victory_trader.chronological_action_value",
                        fromlist=["BASE"],
                    ).BASE
                )
                record["forced_stress_return"] = modeled_return(
                    float(entry_ref), float(exit_ref), __import__(
                        "victory_trader.chronological_action_value",
                        fromlist=["STRESS"],
                    ).STRESS
                )
            rows.append(record)
    return pd.DataFrame(rows)


def fit_economic_value(train: pd.DataFrame):
    valid = np.isfinite(pd.to_numeric(train.forced_base_return, errors="coerce"))
    fitted = train.loc[valid].copy()
    if len(fitted) < MIN_VALUE_ROWS:
        raise ValueError(f"only {len(fitted)} finite economic targets")
    columns = economic_columns(fitted)
    y = fitted.forced_base_return.to_numpy(float)
    low, high = np.quantile(y, [0.005, 0.995])
    episode_sizes = fitted.groupby(KEYS)["state_t"].transform("size").to_numpy(float)
    episodes_per_day = fitted[KEYS].drop_duplicates().groupby("trading_day").size()
    weights = (
        1 / episode_sizes / fitted.trading_day.map(episodes_per_day).to_numpy(float)
    )
    weights /= weights.mean()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=140,
        max_leaf_nodes=7,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261348,
    )
    model.fit(
        features(fitted, columns),
        np.clip(y, low, high),
        sample_weight=weights,
    )
    return model, columns, {
        "rows": int(len(fitted)),
        "episodes": int(len(fitted[KEYS].drop_duplicates())),
        "mean_pct": float(np.mean(y)),
        "positive_rate": float(np.mean(y > 0)),
        "winsor_low": float(low),
        "winsor_high": float(high),
    }


def learnability_report(
    frame: pd.DataFrame,
    model,
    columns: tuple[str, ...],
) -> dict:
    valid = np.isfinite(pd.to_numeric(frame.forced_base_return, errors="coerce"))
    scored = frame.loc[valid].copy()
    if not len(scored):
        return {"resolved_states": 0}
    realized = scored.forced_base_return.to_numpy(float)
    predicted = model.predict(features(scored, columns))
    labels = realized > 0
    rho = spearmanr(predicted, realized).statistic if len(scored) >= 2 else np.nan
    auc = (
        float(roc_auc_score(labels.astype(int), predicted))
        if len(np.unique(labels)) == 2
        else None
    )
    ap = (
        float(average_precision_score(labels.astype(int), predicted))
        if np.any(labels)
        else None
    )
    cutoff = float(np.quantile(predicted, 0.80))
    top = predicted >= cutoff
    return {
        "resolved_states": int(len(scored)),
        "realized_mean_pct": float(np.mean(realized)),
        "realized_positive_rate": float(np.mean(labels)),
        "spearman": float(rho) if np.isfinite(rho) else None,
        "positive_auc": auc,
        "positive_average_precision": ap,
        "top_predicted_quintile_states": int(np.sum(top)),
        "top_predicted_quintile_mean_pct": float(np.mean(realized[top])),
        "top_predicted_quintile_positive_rate": float(np.mean(labels[top])),
        "prediction_mean": float(np.mean(predicted)),
        "prediction_std": float(np.std(predicted)),
    }


def opportunity_ceiling(frame: pd.DataFrame, all_episodes: pd.DataFrame) -> dict:
    resolved = frame.loc[
        np.isfinite(pd.to_numeric(frame.forced_base_return, errors="coerce"))
    ].copy()
    admitted_episode_keys = frame[KEYS].drop_duplicates()
    resolved_episode_keys = resolved[KEYS].drop_duplicates()
    episode_rows = []
    for key, part in resolved.groupby(KEYS, sort=False):
        values = part.forced_base_return.to_numpy(float)
        episode_rows.append(
            dict(zip(KEYS, key))
            | {
                "resolved_states": int(len(part)),
                "best_return_pct": float(np.max(values)),
                "mean_return_pct": float(np.mean(values)),
                "positive_states": int(np.sum(values > 0)),
                "has_positive": bool(np.any(values > 0)),
            }
        )
    episodes = pd.DataFrame(episode_rows)
    total_all = len(all_episodes[KEYS].drop_duplicates())
    total_admitted = len(admitted_episode_keys)
    total_resolved = len(resolved_episode_keys)
    if len(episodes):
        positive_rate_resolved = float(episodes.has_positive.mean())
        best_mean = float(episodes.best_return_pct.mean())
        best_positive_rate = float(episodes.best_return_pct.gt(0).mean())
    else:
        positive_rate_resolved = best_mean = best_positive_rate = None
    return {
        "all_candidate_episodes": int(total_all),
        "episodes_with_admitted_state": int(total_admitted),
        "episodes_with_resolved_admitted_state": int(total_resolved),
        "admitted_episode_coverage": (
            float(total_admitted / total_all) if total_all else 0.0
        ),
        "resolved_episode_coverage": (
            float(total_resolved / total_admitted) if total_admitted else 0.0
        ),
        "positive_opportunity_rate_among_resolved_admitted_episodes": (
            positive_rate_resolved
        ),
        "diagnostic_oracle_best_return_mean_pct": best_mean,
        "diagnostic_oracle_best_positive_rate": best_positive_rate,
        "resolved_state_mean_pct": (
            float(resolved.forced_base_return.mean()) if len(resolved) else None
        ),
        "resolved_state_positive_rate": (
            float(resolved.forced_base_return.gt(0).mean()) if len(resolved) else None
        ),
    }


def diagnose(ceiling: dict, learnability: dict) -> str:
    opp_rate = (
        ceiling.get("positive_opportunity_rate_among_resolved_admitted_episodes")
        or 0.0
    )
    oracle_mean = ceiling.get("diagnostic_oracle_best_return_mean_pct")
    oracle_mean = oracle_mean if oracle_mean is not None else -np.inf
    if opp_rate < 0.30 or oracle_mean <= 0:
        return "candidate_population_bottleneck"
    auc = learnability.get("positive_auc")
    rho = learnability.get("spearman")
    if (auc is None or auc < 0.60) and (rho is None or rho < 0.10):
        return "economic_representation_bottleneck"
    return "controller_composition_or_joint_value_bottleneck"


def evaluate(
    fit_positions: Path,
    calibration_positions: Path,
    output: Path,
    rows_output: Path,
) -> int:
    fit_states = build_states(pd.read_parquet(fit_positions))
    test_states = build_states(pd.read_parquet(calibration_positions))
    if sorted(fit_states.trading_day.astype(str).unique()) != FIT_DAYS:
        raise ValueError("fit dates changed")
    if sorted(test_states.trading_day.astype(str).unique()) != TEST_DAYS:
        raise ValueError("test dates changed")

    all_states = pd.concat([fit_states, test_states], ignore_index=True)
    client = MassiveClient(
        load_settings().massive_api_key,
        cache_dir=Path("data/cache/request248-second-bars"),
        request_interval_seconds=0.05,
    )
    paths = fetch_second_paths(all_states, client)
    labeled = label_tradability(all_states, paths)

    early_labeled = labeled.loc[
        labeled.trading_day.astype(str).isin(TRAIN_DAYS)
    ].copy()
    admission_model, admission_columns = fit_admission(early_labeled)

    value_states = fit_states.loc[
        fit_states.trading_day.astype(str).isin(LEARN_DAYS)
    ].reset_index(drop=True)
    train_admission_probs = probability(
        admission_model, admission_columns, value_states
    )
    position_targets = build_position_targets(
        value_states, train_admission_probs, paths
    )
    position_model, position_columns, position_support = fit_position_model(
        position_targets
    )

    train_forced = forced_state_values(
        value_states,
        train_admission_probs,
        position_model,
        position_columns,
        paths,
    )
    economic_model, economic_model_columns, economic_support = fit_economic_value(
        train_forced
    )

    test_admission_probs = probability(
        admission_model, admission_columns, test_states
    )
    test_forced = forced_state_values(
        test_states,
        test_admission_probs,
        position_model,
        position_columns,
        paths,
    )

    train_learnability = learnability_report(
        train_forced, economic_model, economic_model_columns
    )
    test_learnability = learnability_report(
        test_forced, economic_model, economic_model_columns
    )
    ceiling = opportunity_ceiling(test_forced, test_states)
    bottleneck = diagnose(ceiling, test_learnability)

    result = {
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "development_only": True,
        "admission_threshold": ADMISSION_THRESHOLD,
        "position_model_support": position_support,
        "economic_model_support": economic_support,
        "train_learnability": train_learnability,
        "test_learnability": test_learnability,
        "test_opportunity_ceiling": ceiling,
        "diagnosed_bottleneck": bottleneck,
        "api_stats": client.stats.to_dict(),
        "interpretation": (
            "Episode maxima are diagnostic oracle summaries only. The supervised "
            "model is trained on each state's own forced-policy return, never on a "
            "future-best return or best future state."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.concat(
        [
            train_forced.assign(block="train"),
            test_forced.assign(block="test"),
        ],
        ignore_index=True,
    ).to_parquet(rows_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument("--calibration-positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
