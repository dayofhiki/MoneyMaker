"""Request247: one-step HOLD versus EXIT value on frozen Request246 entries.

The entry policy is consumed from the authoritative Request246 artifact and must
not change. The position target compares EXIT now with waiting exactly to the
next observed state. No best-future exit search is allowed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .causal_tradability_admission import (
    TRAIN_DAYS,
    execution_report,
    fetch_second_paths,
    fit_admission,
    label_tradability,
    probability,
)
from .chronological_action_value import (
    BASE,
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
from .execution_costs import modeled_sell_fill
from .massive_client import MassiveClient
from .second_execution_reconstruction import (
    EXPIRY_MS,
    PRIMARY_LATENCY_MS,
    first_observed_open,
    reconstruct,
)

REQUEST_ID = 247
ADMISSION_THRESHOLD = 0.60
VALUE_TRAIN_DAYS = ["2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08"]
TEST_DAYS = CAL_DAYS
MINUTE_MS = 60_000
MIN_TARGET_ROWS = 300
MIN_CLOSED_RESOLUTION = 0.70
MIN_CLOSED_POSITIVE_RATE = 0.50
HOLD_THRESHOLD = 0.0


def sell_continuation_pct(now_ref: float, next_ref: float, scenario=BASE) -> float:
    now_sell = modeled_sell_fill(float(now_ref), scenario)
    next_sell = modeled_sell_fill(float(next_ref), scenario)
    if not (np.isfinite(now_sell) and np.isfinite(next_sell) and now_sell > 0):
        return np.nan
    return float((next_sell / now_sell - 1) * 100)


def one_step_targets(
    states: pd.DataFrame,
    admission_probs: np.ndarray,
    paths: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    work = states.reset_index(drop=True).copy()
    work["tradability_probability"] = admission_probs
    rows = []
    for key, part in work.groupby(KEYS, sort=False):
        part = part.sort_values("state_t")
        ids = list(part.index)
        day, ticker, _ = key
        path = paths.get(
            (str(day), str(ticker).upper()),
            (np.array([], dtype=np.int64), np.array([], dtype=float)),
        )
        times, opens = path
        for pos in range(len(ids) - 1):
            idx = ids[pos]
            next_idx = ids[pos + 1]
            row = work.loc[idx]
            if bool(row["terminal"]):
                continue
            _, now_ref = first_observed_open(
                times,
                opens,
                decision_t=int(row["state_t"]),
                latency_ms=PRIMARY_LATENCY_MS,
                expiry_ms=EXPIRY_MS,
            )
            _, next_ref = first_observed_open(
                times,
                opens,
                decision_t=int(work.at[next_idx, "state_t"]),
                latency_ms=PRIMARY_LATENCY_MS,
                expiry_ms=EXPIRY_MS,
            )
            out = row.to_dict()
            out["next_state_t"] = int(work.at[next_idx, "state_t"])
            out["hold_one_step_value"] = (
                sell_continuation_pct(now_ref, next_ref)
                if now_ref is not None and next_ref is not None
                else np.nan
            )
            rows.append(out)
    return pd.DataFrame(rows)


def position_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        c
        for c in dict.fromkeys(
            [*CAUSAL_SOURCE_FEATURES, *PATH_FEATURES, "tradability_probability"]
        )
        if c in frame and pd.to_numeric(frame[c], errors="coerce").notna().any()
    )


def fit_position_model(targets: pd.DataFrame):
    if sorted(targets.trading_day.astype(str).unique()) != VALUE_TRAIN_DAYS:
        raise ValueError("position-value training dates changed")
    valid = np.isfinite(pd.to_numeric(targets.hold_one_step_value, errors="coerce"))
    train = targets.loc[valid].copy()
    if len(train) < MIN_TARGET_ROWS:
        raise ValueError(f"hold_one_step_value: only {len(train)} finite targets")
    columns = position_columns(train)
    y = train.hold_one_step_value.to_numpy(float)
    low, high = np.quantile(y, [0.005, 0.995])
    sizes = train.groupby(KEYS)["state_t"].transform("size").to_numpy(float)
    ep_per_day = train[KEYS].drop_duplicates().groupby("trading_day").size()
    weights = 1 / sizes / train.trading_day.map(ep_per_day).to_numpy(float)
    weights /= weights.mean()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=140,
        max_leaf_nodes=7,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261347,
    )
    model.fit(features(train, columns), np.clip(y, low, high), sample_weight=weights)
    support = {
        "rows": int(len(train)),
        "episodes": int(len(train[KEYS].drop_duplicates())),
        "mean_pct": float(np.mean(y)),
        "positive_rate": float(np.mean(y > 0)),
        "winsor_low": float(low),
        "winsor_high": float(high),
    }
    return model, columns, support


def frozen_request246_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_parquet(path)
    if "policy" not in rows:
        raise ValueError("Request246 artifact is missing policy column")
    rows = rows.loc[rows.policy.astype(str).eq("learned_enter_wait")].copy()
    if len(rows) == 0:
        raise ValueError("Request246 learned_enter_wait rows missing")
    if sorted(rows.trading_day.astype(str).unique()) != TEST_DAYS:
        raise ValueError("Request246 test dates changed")
    if rows.duplicated(KEYS).any():
        raise ValueError("duplicate Request246 episode rows")
    return rows.reset_index(drop=True)


def make_position_decisions(
    test_states: pd.DataFrame,
    admission_probs: np.ndarray,
    frozen_entries: pd.DataFrame,
    model,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    work = test_states.reset_index(drop=True).copy()
    work["tradability_probability"] = admission_probs
    work["predicted_hold_one_step_value"] = model.predict(features(work, columns))
    frozen = frozen_entries.set_index(KEYS)
    rows = []
    for key, part in work.groupby(KEYS, sort=False):
        part = part.sort_values("state_t")
        if key not in frozen.index:
            raise ValueError(f"frozen Request246 episode missing: {key}")
        base = frozen.loc[key]
        entered = bool(base["entered"])
        record = {
            "trading_day": key[0],
            "ticker": key[1],
            "hot_t": key[2],
            "entered": entered,
            "unresolved": False,
            "net_pct": np.nan,
            "stress_pct": np.nan,
            "entry_t": None,
            "exit_t": None,
            "wait_actions": int(base.get("wait_actions", 0)),
            "hold_actions": 0,
            "entry_decision_t": None,
            "exit_decision_t": None,
            "deadline_liquidation": False,
            "policy_exit_missing": False,
        }
        if not entered:
            rows.append(record)
            continue
        entry_t = int(base["entry_decision_t"])
        record["entry_t"] = entry_t
        record["entry_decision_t"] = entry_t
        later = part.loc[part.state_t.astype(np.int64) > entry_t]
        exit_t = None
        holds = 0
        for _, row in later.iterrows():
            if bool(row["terminal"]):
                exit_t = int(row["state_t"])
                break
            if float(row["tradability_probability"]) < ADMISSION_THRESHOLD:
                holds += 1
                continue
            if float(row["predicted_hold_one_step_value"]) > HOLD_THRESHOLD:
                holds += 1
                continue
            exit_t = int(row["state_t"])
            break
        if exit_t is None:
            record["policy_exit_missing"] = True
            record["deadline_liquidation"] = True
            exit_t = int(key[2]) + EXIT_DEADLINE * MINUTE_MS
        record["exit_t"] = exit_t
        record["exit_decision_t"] = exit_t
        record["hold_actions"] = holds
        rows.append(record)
    return pd.DataFrame(rows)


def paired_delta(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    left = candidate.loc[
        candidate.execution_status.eq("closed"),
        [*KEYS, "reconstructed_base_pct"],
    ].rename(columns={"reconstructed_base_pct": "candidate_pct"})
    right = baseline.loc[
        baseline.execution_status.eq("closed"),
        [*KEYS, "reconstructed_base_pct"],
    ].rename(columns={"reconstructed_base_pct": "baseline_pct"})
    paired = left.merge(right, on=KEYS, how="inner")
    return {
        "paired_closed_episodes": int(len(paired)),
        "mean_delta_pct": (
            float((paired.candidate_pct - paired.baseline_pct).mean())
            if len(paired)
            else None
        ),
    }


def entry_signature(rows: pd.DataFrame) -> list[tuple]:
    entered = rows.loc[rows.entered].copy()
    return sorted(
        (
            str(r.trading_day),
            str(r.ticker),
            int(r.hot_t),
            int(r.entry_decision_t),
        )
        for r in entered.itertuples(index=False)
    )


def evaluate(
    fit_positions: Path,
    calibration_positions: Path,
    request246_rows: Path,
    output: Path,
    rows_output: Path,
) -> int:
    fit_states = build_states(pd.read_parquet(fit_positions))
    test_states = build_states(pd.read_parquet(calibration_positions))
    if sorted(fit_states.trading_day.astype(str).unique()) != FIT_DAYS:
        raise ValueError("fit dates changed")
    if sorted(test_states.trading_day.astype(str).unique()) != TEST_DAYS:
        raise ValueError("test dates changed")

    frozen = frozen_request246_rows(request246_rows)
    all_states = pd.concat([fit_states, test_states], ignore_index=True)
    client = MassiveClient(
        load_settings().massive_api_key,
        cache_dir=Path("data/cache/request247-second-bars"),
        request_interval_seconds=0.05,
    )
    paths = fetch_second_paths(all_states, client)
    labeled = label_tradability(all_states, paths)
    early = labeled.loc[
        labeled.trading_day.astype(str).isin(TRAIN_DAYS)
    ].copy()
    admission_model, admission_columns = fit_admission(early)

    value_states = fit_states.loc[
        fit_states.trading_day.astype(str).isin(VALUE_TRAIN_DAYS)
    ].reset_index(drop=True)
    value_probs = probability(admission_model, admission_columns, value_states)
    targets = one_step_targets(value_states, value_probs, paths)
    model, columns, support = fit_position_model(targets)

    test_probs = probability(admission_model, admission_columns, test_states)
    decisions = make_position_decisions(
        test_states, test_probs, frozen, model, columns
    )
    candidate = reconstruct(decisions, paths, latency_ms=PRIMARY_LATENCY_MS)
    baseline = frozen.copy()
    candidate_report = execution_report(candidate)
    baseline_report = execution_report(baseline)
    paired = paired_delta(candidate, baseline)

    same_entries = entry_signature(decisions) == entry_signature(frozen)
    candidate_mean = candidate_report["closed_mean_base_pct"]
    baseline_mean = baseline_report["closed_mean_base_pct"]
    candidate_severe = candidate_report["closed_severe_loss_rate_le_minus2"]
    baseline_severe = baseline_report["closed_severe_loss_rate_le_minus2"]
    checks = {
        "position_target_support": support["rows"] >= MIN_TARGET_ROWS,
        "entry_times_identical_to_request246": same_entries,
        "closed_resolution_rate": (
            candidate_report["closed_resolution_rate"] >= MIN_CLOSED_RESOLUTION
        ),
        "closed_mean_beats_request246": (
            candidate_mean is not None
            and baseline_mean is not None
            and candidate_mean > baseline_mean
        ),
        "paired_mean_delta_positive": (paired["mean_delta_pct"] or 0) > 0,
        "closed_base_mean_positive": (
            candidate_mean is not None and candidate_mean > 0
        ),
        "closed_positive_rate": (
            (candidate_report["closed_positive_rate"] or 0)
            >= MIN_CLOSED_POSITIVE_RATE
        ),
        "severe_loss_rate_not_worse": (
            candidate_severe is not None
            and baseline_severe is not None
            and candidate_severe <= baseline_severe
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "development_only": True,
        "admission_threshold": ADMISSION_THRESHOLD,
        "admission_training_days": TRAIN_DAYS,
        "position_value_training_days": VALUE_TRAIN_DAYS,
        "test_days": TEST_DAYS,
        "position_feature_count": len(columns),
        "support": support,
        "training_targets": {
            "rows_total": int(len(targets)),
            "finite": int(np.isfinite(targets.hold_one_step_value).sum()),
            "mean_pct": float(
                pd.to_numeric(targets.hold_one_step_value, errors="coerce").mean()
            ),
            "positive_rate": float(
                pd.to_numeric(targets.hold_one_step_value, errors="coerce")
                .dropna()
                .gt(0)
                .mean()
            ),
        },
        "test": {
            "request246_frozen_baseline": baseline_report,
            "one_step_position_value": candidate_report,
            "paired_comparison": paired,
        },
        "checks": checks,
        "economic_gate_pass": bool(all(checks.values())),
        "api_stats": client.stats.to_dict(),
        "interpretation": (
            "Request246 entry times are frozen. This request changes HOLD/EXIT only. "
            "One-second aggregate opens are conservative historical execution "
            "references, not actual brokerage fills."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    pd.concat(
        [
            baseline.assign(policy="request246_frozen"),
            candidate.assign(policy="one_step_position_value"),
        ],
        ignore_index=True,
    ).to_parquet(rows_output, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fit-positions", type=Path, required=True)
    p.add_argument("--calibration-positions", type=Path, required=True)
    p.add_argument("--request246-rows", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows-output", type=Path, required=True)
    a = p.parse_args()
    return evaluate(
        a.fit_positions,
        a.calibration_positions,
        a.request246_rows,
        a.output,
        a.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
