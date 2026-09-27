"""Request245: causal tradability admission before Request243 ENTER actions.

The classifier may learn only from information already observable at a decision
state. One-second bars after the decision are labels for supervised research,
never features. May11-20 is a locked test block and never chooses the threshold.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .chronological_action_value import (
    CAL_DAYS,
    CAUSAL_SOURCE_FEATURES,
    EXIT_DEADLINE,
    FIT_DAYS,
    KEYS,
    PATH_FEATURES,
    build_states,
    chronological_fit,
    episode_results,
    features,
    rollout,
)
from .config import load_settings
from .massive_client import MassiveClient
from .second_execution_reconstruction import (
    EXPIRY_MS,
    PRIMARY_LATENCY_MS,
    execution_decisions,
    first_observed_open,
    observed_opens,
    reconstruct,
)

REQUEST_ID = 245
TRAIN_DAYS = FIT_DAYS[:3]
THRESHOLD_DAYS = FIT_DAYS[3:]
TEST_DAYS = CAL_DAYS
ANCHOR_MINUTES = (1, 2, 3, 4, 5)
MIN_CONTINUATION_ANCHORS = 3
THRESHOLD_GRID = tuple(round(x, 2) for x in np.arange(0.05, 1.0, 0.05))
MIN_VALIDATION_PRECISION = 0.80
MIN_VALIDATION_RECALL = 0.20
MIN_VALIDATION_ADMITTED = 30
MIN_TEST_PRECISION = 0.75
MIN_TEST_ADMITTED = 30
MIN_OVERLAY_ENTRIES = 20
MIN_OVERLAY_ENTRY_FILL_COVERAGE = 0.70
MIN_OVERLAY_RESOLUTION_UPLIFT = 0.15
MINUTE_MS = 60_000


def fetch_second_paths(
    states: pd.DataFrame,
    client: MassiveClient,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    eligible = states.loc[states.can_enter, ["trading_day", "ticker"]].drop_duplicates()
    paths: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for row in eligible.sort_values(["trading_day", "ticker"]).itertuples(index=False):
        day_text = str(row.trading_day)
        ticker = str(row.ticker).upper()
        payload = client.second_bars_range(
            ticker,
            date.fromisoformat(day_text),
            date.fromisoformat(day_text),
            adjusted=False,
        )
        paths[(day_text, ticker)] = observed_opens(payload)
    return paths


def label_tradability(
    states: pd.DataFrame,
    paths: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for record in states.loc[states.can_enter].to_dict("records"):
        key = (str(record["trading_day"]), str(record["ticker"]).upper())
        times, opens = paths.get(
            key, (np.array([], dtype=np.int64), np.array([], dtype=float))
        )
        state_t = int(record["state_t"])
        _, entry_ref = first_observed_open(
            times,
            opens,
            decision_t=state_t,
            latency_ms=PRIMARY_LATENCY_MS,
            expiry_ms=EXPIRY_MS,
        )
        anchor_hits = 0
        for minute in ANCHOR_MINUTES:
            _, ref = first_observed_open(
                times,
                opens,
                decision_t=state_t + minute * MINUTE_MS,
                latency_ms=PRIMARY_LATENCY_MS,
                expiry_ms=EXPIRY_MS,
            )
            anchor_hits += int(ref is not None)
        out = dict(record)
        out["entry_available"] = bool(entry_ref is not None)
        out["continuation_anchor_hits"] = int(anchor_hits)
        out["continuation_available"] = bool(
            anchor_hits >= MIN_CONTINUATION_ANCHORS
        )
        out["tradable_label"] = bool(
            out["entry_available"] and out["continuation_available"]
        )
        rows.append(out)
    return pd.DataFrame(rows)


def model_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        c
        for c in dict.fromkeys([*CAUSAL_SOURCE_FEATURES, *PATH_FEATURES])
        if c in frame and pd.to_numeric(frame[c], errors="coerce").notna().any()
    )


def fit_admission(train: pd.DataFrame):
    if sorted(train.trading_day.astype(str).unique()) != TRAIN_DAYS:
        raise ValueError("tradability training dates changed")
    y = train.tradable_label.astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("tradability training block needs both classes")
    columns = model_columns(train)
    episode_sizes = train.groupby(KEYS)["state_t"].transform("size").to_numpy(float)
    episodes_per_day = train[KEYS].drop_duplicates().groupby("trading_day").size()
    weights = 1 / episode_sizes / train.trading_day.map(episodes_per_day).to_numpy(float)
    weights /= weights.mean()
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=60,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261345,
    )
    model.fit(features(train, columns), y, sample_weight=weights)
    return model, columns


def probability(model, columns, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features(frame, columns))[:, 1]


def threshold_metrics(
    frame: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    y = frame.tradable_label.astype(bool).to_numpy()
    admitted = probs >= threshold
    tp = int(np.sum(admitted & y))
    count = int(np.sum(admitted))
    positives = int(np.sum(y))
    precision = float(tp / count) if count else None
    recall = float(tp / positives) if positives else None
    return {
        "threshold": float(threshold),
        "admitted": count,
        "true_tradable": positives,
        "precision": precision,
        "recall": recall,
    }


def choose_threshold(frame: pd.DataFrame, probs: np.ndarray):
    if sorted(frame.trading_day.astype(str).unique()) != THRESHOLD_DAYS:
        raise ValueError("threshold calibration dates changed")
    table = []
    chosen = None
    for threshold in THRESHOLD_GRID:
        row = threshold_metrics(frame, probs, threshold)
        table.append(row)
        if (
            chosen is None
            and row["admitted"] >= MIN_VALIDATION_ADMITTED
            and (row["precision"] or 0) >= MIN_VALIDATION_PRECISION
            and (row["recall"] or 0) >= MIN_VALIDATION_RECALL
        ):
            chosen = float(threshold)
    return chosen, table


def classification_report(
    frame: pd.DataFrame,
    probs: np.ndarray,
    threshold: float | None,
) -> dict:
    y = frame.tradable_label.astype(int).to_numpy()
    auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else None
    ap = float(average_precision_score(y, probs)) if np.any(y) else None
    base_rate = float(np.mean(y)) if len(y) else 0.0
    report = {
        "states": int(len(frame)),
        "tradable_rate": base_rate,
        "roc_auc": auc,
        "average_precision": ap,
        "entry_available_rate": float(frame.entry_available.mean()) if len(frame) else 0.0,
        "continuation_available_rate": (
            float(frame.continuation_available.mean()) if len(frame) else 0.0
        ),
    }
    if threshold is not None:
        metrics = threshold_metrics(frame, probs, threshold)
        admitted = probs >= threshold
        report["threshold_metrics"] = metrics
        report["admitted_entry_available_rate"] = (
            float(frame.loc[admitted, "entry_available"].mean())
            if np.any(admitted)
            else None
        )
        report["admitted_continuation_available_rate"] = (
            float(frame.loc[admitted, "continuation_available"].mean())
            if np.any(admitted)
            else None
        )
    else:
        report["threshold_metrics"] = None
        report["admitted_entry_available_rate"] = None
        report["admitted_continuation_available_rate"] = None
    return report


def gated_policy_decisions(
    scored: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    work = scored.reset_index(drop=True).copy()
    work["tradability_probability"] = probs
    rows = []
    for key, part in work.groupby(KEYS, sort=False):
        part = part.sort_values("state_t")
        entry_index = None
        blocked_enters = 0
        wait_actions = 0
        for idx, row in part.iterrows():
            if not bool(row["can_enter"]) or bool(row["terminal"]):
                continue
            action = str(row["flat_action"])
            if action == "ENTER":
                if float(row["tradability_probability"]) >= threshold:
                    entry_index = idx
                    break
                blocked_enters += 1
                wait_actions += 1
                continue
            if action == "WAIT":
                wait_actions += 1
                continue
            if action == "ABSTAIN":
                break

        trading_day, ticker, hot_t = key
        base = {
            "trading_day": trading_day,
            "ticker": ticker,
            "hot_t": hot_t,
            "entered": entry_index is not None,
            "unresolved": False,
            "net_pct": np.nan,
            "stress_pct": np.nan,
            "entry_t": None,
            "exit_t": None,
            "wait_actions": wait_actions,
            "hold_actions": 0,
            "blocked_enter_actions": blocked_enters,
            "entry_tradability_probability": None,
            "entry_decision_t": None,
            "exit_decision_t": None,
            "deadline_liquidation": False,
            "policy_exit_missing": False,
        }
        if entry_index is None:
            rows.append(base)
            continue

        entry_row = work.loc[entry_index]
        base["entry_t"] = int(entry_row["state_t"])
        base["entry_decision_t"] = int(entry_row["state_t"])
        base["entry_tradability_probability"] = float(
            entry_row["tradability_probability"]
        )

        later = part.loc[part.index > entry_index]
        exits = later.loc[later.held_action.astype(str).eq("EXIT")]
        if len(exits):
            exit_index = exits.index[0]
            base["exit_t"] = int(work.at[exit_index, "state_t"])
            base["exit_decision_t"] = int(work.at[exit_index, "state_t"])
            between = later.loc[later.index < exit_index]
            base["hold_actions"] = int(
                between.held_action.astype(str).eq("HOLD").sum()
            )
        else:
            base["policy_exit_missing"] = True
            base["deadline_liquidation"] = True
            base["exit_decision_t"] = int(hot_t) + EXIT_DEADLINE * MINUTE_MS
            base["hold_actions"] = int(
                later.held_action.astype(str).eq("HOLD").sum()
            )
        rows.append(base)
    return pd.DataFrame(rows)


def execution_report(rows: pd.DataFrame) -> dict:
    entered = rows.loc[rows.entered]
    closed = entered.loc[entered.execution_status.eq("closed")]
    entry_filled = entered.loc[
        ~entered.execution_status.isin(
            ["entry_unavailable", "missing_policy_entry_time"]
        )
    ]
    return {
        "episodes": int(len(rows)),
        "entries": int(len(entered)),
        "statuses": dict(Counter(entered.execution_status.astype(str))),
        "entry_fill_coverage": (
            float(len(entry_filled) / len(entered)) if len(entered) else 0.0
        ),
        "closed_resolution_rate": (
            float(len(closed) / len(entered)) if len(entered) else 0.0
        ),
        "closed_trades": int(len(closed)),
        "closed_mean_base_pct": (
            float(pd.to_numeric(closed.reconstructed_base_pct).mean())
            if len(closed)
            else None
        ),
        "closed_mean_stress_pct": (
            float(pd.to_numeric(closed.reconstructed_stress_pct).mean())
            if len(closed)
            else None
        ),
        "closed_positive_rate": (
            float(pd.to_numeric(closed.reconstructed_base_pct).gt(0).mean())
            if len(closed)
            else None
        ),
        "closed_severe_loss_rate_le_minus2": (
            float(pd.to_numeric(closed.reconstructed_base_pct).le(-2).mean())
            if len(closed)
            else None
        ),
        "blocked_enter_actions": (
            int(rows.get("blocked_enter_actions", pd.Series(dtype=int)).sum())
            if "blocked_enter_actions" in rows
            else 0
        ),
    }


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
        cache_dir=Path("data/cache/request245-second-bars"),
        request_interval_seconds=0.05,
    )
    paths = fetch_second_paths(all_states, client)
    labeled = label_tradability(all_states, paths)
    train = labeled.loc[labeled.trading_day.astype(str).isin(TRAIN_DAYS)].copy()
    threshold_block = labeled.loc[
        labeled.trading_day.astype(str).isin(THRESHOLD_DAYS)
    ].copy()
    test = labeled.loc[labeled.trading_day.astype(str).isin(TEST_DAYS)].copy()

    model, columns = fit_admission(train)
    threshold_probs = probability(model, columns, threshold_block)
    threshold, threshold_table = choose_threshold(threshold_block, threshold_probs)
    test_probs = probability(model, columns, test)

    _teacher, student, _ = chronological_fit(fit_states)
    scored = rollout(test_states, student, "full")
    scored_probs = probability(model, columns, scored)
    base_decisions = execution_decisions(episode_results(scored))
    base_rows = reconstruct(base_decisions, paths, latency_ms=PRIMARY_LATENCY_MS)
    base_report = execution_report(base_rows)

    gated_rows = pd.DataFrame()
    gated_report = {
        "episodes": int(len(base_rows)),
        "entries": 0,
        "statuses": {},
        "entry_fill_coverage": 0.0,
        "closed_resolution_rate": 0.0,
        "closed_trades": 0,
        "closed_mean_base_pct": None,
        "closed_mean_stress_pct": None,
        "closed_positive_rate": None,
        "closed_severe_loss_rate_le_minus2": None,
        "blocked_enter_actions": 0,
    }
    if threshold is not None:
        gated_decisions = gated_policy_decisions(scored, scored_probs, threshold)
        gated_rows = reconstruct(
            gated_decisions, paths, latency_ms=PRIMARY_LATENCY_MS
        )
        gated_report = execution_report(gated_rows)

    test_report = classification_report(test, test_probs, threshold)
    resolution_uplift = (
        gated_report["closed_resolution_rate"] - base_report["closed_resolution_rate"]
    )
    threshold_metrics_test = test_report.get("threshold_metrics") or {}
    checks = {
        "threshold_found": threshold is not None,
        "test_tradable_precision": (
            (threshold_metrics_test.get("precision") or 0) >= MIN_TEST_PRECISION
        ),
        "test_admitted_states": (
            int(threshold_metrics_test.get("admitted") or 0) >= MIN_TEST_ADMITTED
        ),
        "overlay_entries": gated_report["entries"] >= MIN_OVERLAY_ENTRIES,
        "overlay_entry_fill_coverage": (
            gated_report["entry_fill_coverage"] >= MIN_OVERLAY_ENTRY_FILL_COVERAGE
        ),
        "overlay_closed_resolution_uplift": (
            resolution_uplift >= MIN_OVERLAY_RESOLUTION_UPLIFT
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "development_only": True,
        "training_days": TRAIN_DAYS,
        "threshold_days": THRESHOLD_DAYS,
        "test_days": TEST_DAYS,
        "label_contract": {
            "latency_ms": PRIMARY_LATENCY_MS,
            "expiry_ms": EXPIRY_MS,
            "continuation_anchor_minutes": ANCHOR_MINUTES,
            "min_continuation_anchors": MIN_CONTINUATION_ANCHORS,
        },
        "feature_count": len(columns),
        "training_tradable_rate": float(train.tradable_label.mean()),
        "threshold_block": classification_report(
            threshold_block, threshold_probs, threshold
        ),
        "chosen_threshold": threshold,
        "threshold_search": threshold_table,
        "test": test_report,
        "controller": {
            "request244_equivalent_base": base_report,
            "tradability_gated": gated_report,
            "closed_resolution_uplift": resolution_uplift,
        },
        "checks": checks,
        "structural_gate_pass": bool(all(checks.values())),
        "api_stats": client.stats.to_dict(),
        "interpretation": (
            "This is an execution-admission diagnostic, not a profitability or live "
            "readiness claim. Future one-second bars create labels only; May11-20 "
            "never selects the classifier threshold."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    row_frames = [base_rows.assign(policy="base")]
    if len(gated_rows):
        row_frames.append(gated_rows.assign(policy="tradability_gated"))
    pd.concat(row_frames, ignore_index=True).to_parquet(rows_output, index=False)
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
