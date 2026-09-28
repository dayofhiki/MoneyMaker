"""Request255: 2% pullback trigger with learned local ENTER-vs-WAIT gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .learned_pullback_entry import (
    CANDIDATE_FRACTION,
    CONTROLLER_TRAIN_DAYS,
    TEST_DAYS,
    _weights,
    baseline_policy,
    build_episode_states,
    controller_columns,
    controller_x,
    fit_candidate_model,
    policy_metrics,
    score_candidates,
)

REQUEST_ID = 255
PULLBACK_TRIGGER_PCT = 2.0
ADVANTAGE_PROBABILITY_THRESHOLD = 0.50
MIN_TRAIN_ADVANTAGE_ROWS = 200
MIN_TEST_PULLBACK_STATES = 80
MIN_PULLBACK_AUC = 0.55
MIN_ENTRIES = 20
MIN_ENTRY_RATE = 0.10
MAX_ENTRY_RATE = 0.45
MIN_RESOLUTION_OR_CASH = 0.95
MIN_POSITIVE_MEAN_DAYS = 5
MAX_SEVERE_RATE = 0.08
MIN_GAIN_VS_FIXED_PULLBACK = 0.10


def fit_advantage_classifier(
    states: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[HistGradientBoostingClassifier, dict]:
    target = pd.to_numeric(
        states.enter_minus_wait_pct,
        errors="coerce",
    )
    train = states.loc[target.notna()].copy()
    y = pd.to_numeric(
        train.enter_minus_wait_pct,
        errors="coerce",
    ).gt(0).astype(int)
    if len(train) < MIN_TRAIN_ADVANTAGE_ROWS:
        raise ValueError(
            f"Request255 advantage support only {len(train)}"
        )
    if y.nunique() != 2:
        raise ValueError("Request255 advantage target needs both classes")
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261358,
    )
    model.fit(
        controller_x(train, columns),
        y,
        sample_weight=_weights(train),
    )
    return model, {
        "rows": int(len(train)),
        "positive_rate": float(y.mean()),
    }


def pullback_mask(states: pd.DataFrame) -> pd.Series:
    drawdown = pd.to_numeric(
        states.drawdown_from_running_high_pct,
        errors="coerce",
    )
    return drawdown.le(-PULLBACK_TRIGGER_PCT)


def pullback_signal_report(
    states: pd.DataFrame,
    model: HistGradientBoostingClassifier,
    columns: tuple[str, ...],
) -> dict:
    work = states.loc[pullback_mask(states)].copy()
    target = pd.to_numeric(
        work.enter_minus_wait_pct,
        errors="coerce",
    )
    work = work.loc[target.notna()].copy()
    target = pd.to_numeric(
        work.enter_minus_wait_pct,
        errors="coerce",
    )
    if len(work):
        probability = model.predict_proba(
            controller_x(work, columns)
        )[:, 1]
    else:
        probability = np.array([], dtype=float)
    y = target.gt(0).astype(int)
    auc = (
        float(roc_auc_score(y, probability))
        if len(work) >= 20 and y.nunique() == 2
        else None
    )
    return {
        "rows": int(len(work)),
        "positive_rate": float(y.mean()) if len(y) else None,
        "auc": auc,
        "mean_advantage_pct": (
            float(target.mean()) if len(target) else None
        ),
    }


def make_policy(
    states: pd.DataFrame,
    selected: pd.DataFrame,
    model: HistGradientBoostingClassifier,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    scored = states.copy()
    if len(scored):
        scored["advantage_probability"] = model.predict_proba(
            controller_x(scored, columns)
        )[:, 1]
    else:
        scored["advantage_probability"] = pd.Series(dtype=float)

    groups = {
        (
            str(key[0]),
            str(key[1]).upper(),
            int(key[2]),
        ): part.sort_values("state_t")
        for key, part in scored.groupby(
            ["trading_day", "ticker", "hot_t"],
            sort=False,
        )
    }

    records = []
    for row in selected.itertuples(index=False):
        key = (
            str(row.trading_day),
            str(row.ticker).upper(),
            int(row.t),
        )
        group = groups.get(key)
        chosen = None
        if group is not None:
            eligible = group.loc[pullback_mask(group)].copy()
            for _, state in eligible.iterrows():
                if (
                    float(state.advantage_probability)
                    >= ADVANTAGE_PROBABILITY_THRESHOLD
                ):
                    chosen = state
                    break
        records.append(
            {
                "trading_day": key[0],
                "ticker": key[1],
                "hot_t": key[2],
                "entered": chosen is not None,
                "entry_t": (
                    int(chosen.state_t)
                    if chosen is not None
                    else None
                ),
                "entry_price": (
                    float(chosen.state_price)
                    if chosen is not None
                    else None
                ),
                "value_pct": (
                    float(chosen.enter_value_pct)
                    if chosen is not None
                    and pd.notna(chosen.enter_value_pct)
                    else np.nan
                ),
                "advantage_probability": (
                    float(chosen.advantage_probability)
                    if chosen is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(records)


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    from .full_hot_fixed_policy_value import attach_fixed_value

    first_hot = attach_fixed_value(first_hot, scan)

    candidate_model, candidate_columns = fit_candidate_model(first_hot)
    scored, candidate_threshold = score_candidates(
        first_hot,
        candidate_model,
        candidate_columns,
    )
    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(CONTROLLER_TRAIN_DAYS)
        & scored.candidate_selected
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_selected
    ].copy()

    train_states = build_episode_states(train_selected, scan)
    test_states = build_episode_states(test_selected, scan)
    columns = controller_columns(train_states)

    model, train_support = fit_advantage_classifier(
        train_states,
        columns,
    )
    signal = pullback_signal_report(
        test_states,
        model,
        columns,
    )

    policy_rows = make_policy(
        test_states,
        test_selected,
        model,
        columns,
    )
    immediate_rows = baseline_policy(
        test_selected,
        scan,
        "immediate",
    )
    fixed_rows = baseline_policy(
        test_selected,
        scan,
        "pullback_2",
    )

    candidate_count = int(len(test_selected))
    policy_report = policy_metrics(policy_rows, candidate_count)
    immediate_report = policy_metrics(
        immediate_rows,
        candidate_count,
    )
    fixed_report = policy_metrics(
        fixed_rows,
        candidate_count,
    )
    policy_mean = policy_report["mean_pct"]
    fixed_mean = fixed_report["mean_pct"]
    gain_vs_fixed = (
        policy_mean - fixed_mean
        if policy_mean is not None and fixed_mean is not None
        else None
    )

    checks = {
        "train_advantage_rows": (
            train_support["rows"] >= MIN_TRAIN_ADVANTAGE_ROWS
        ),
        "test_pullback_states": (
            signal["rows"] >= MIN_TEST_PULLBACK_STATES
        ),
        "pullback_advantage_auc": (
            signal["auc"] is not None
            and signal["auc"] >= MIN_PULLBACK_AUC
        ),
        "entries": policy_report["entries"] >= MIN_ENTRIES,
        "entry_rate": (
            MIN_ENTRY_RATE
            <= policy_report["entry_rate"]
            <= MAX_ENTRY_RATE
        ),
        "resolution_or_cash_rate": (
            policy_report["resolution_or_cash_rate"]
            >= MIN_RESOLUTION_OR_CASH
        ),
        "mean_positive": (
            policy_mean is not None and policy_mean > 0
        ),
        "day_balanced_mean_positive": (
            policy_report["day_balanced_mean_pct"] is not None
            and policy_report["day_balanced_mean_pct"] > 0
        ),
        "positive_mean_days": (
            policy_report["positive_mean_days"]
            >= MIN_POSITIVE_MEAN_DAYS
        ),
        "severe_loss_rate": (
            policy_report["severe_loss_rate_le_minus2"] is not None
            and policy_report["severe_loss_rate_le_minus2"]
            <= MAX_SEVERE_RATE
        ),
        "gain_vs_fixed_pullback": (
            gain_vs_fixed is not None
            and gain_vs_fixed >= MIN_GAIN_VS_FIXED_PULLBACK
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "candidate_fraction": CANDIDATE_FRACTION,
        "candidate_threshold": candidate_threshold,
        "pullback_trigger_pct": PULLBACK_TRIGGER_PCT,
        "advantage_probability_threshold": (
            ADVANTAGE_PROBABILITY_THRESHOLD
        ),
        "selected_support": {
            "train_episodes": int(len(train_selected)),
            "test_episodes": candidate_count,
            "train_states": int(len(train_states)),
            "test_states": int(len(test_states)),
            "train_pullback_states": int(
                pullback_mask(train_states).sum()
            ),
            "test_pullback_states": int(
                pullback_mask(test_states).sum()
            ),
        },
        "train_advantage_support": train_support,
        "development_pullback_signal": signal,
        "development_policies": {
            "immediate": immediate_report,
            "fixed_2pct_pullback": fixed_report,
            "pullback_advantage_gate": policy_report,
        },
        "mean_gain_vs_fixed_2pct_pullback_pct": gain_vs_fixed,
        "checks": checks,
        "economic_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "Development-only. The 2% trigger and 0.5 classifier threshold "
            "are fixed before this run. May11-20 has been seen previously, "
            "so this is not fresh validation."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
