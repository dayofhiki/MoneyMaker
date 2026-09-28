"""Request256: fixed three-opportunity WAIT option after a 2% pullback."""

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
from .pullback_advantage_gate import (
    ADVANTAGE_PROBABILITY_THRESHOLD,
    PULLBACK_TRIGGER_PCT,
    make_policy as make_one_step_policy,
    pullback_mask,
)

REQUEST_ID = 256
WAIT_OPTION_HORIZON = 3
MIN_TRAIN_TARGET_ROWS = 180
MIN_TEST_PULLBACK_STATES = 80
MIN_PULLBACK_AUC = 0.55
MIN_ENTRIES = 20
MIN_ENTRY_RATE = 0.10
MAX_ENTRY_RATE = 0.45
MIN_RESOLUTION_OR_CASH = 0.95
MIN_POSITIVE_MEAN_DAYS = 5
MAX_SEVERE_RATE = 0.08
MIN_GAIN_VS_FIXED_PULLBACK = 0.10


def attach_multistep_target(states: pd.DataFrame) -> pd.DataFrame:
    work = states.copy()
    work["wait_option_value_pct"] = np.nan
    work["enter_minus_wait_option_pct"] = np.nan

    for _, group in work.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        ordered = group.sort_values("state_t")
        indices = list(ordered.index)
        for position, index in enumerate(indices):
            enter_value = pd.to_numeric(
                pd.Series([work.at[index, "enter_value_pct"]]),
                errors="coerce",
            ).iloc[0]
            future_values = []
            unresolved = False
            for step in range(1, WAIT_OPTION_HORIZON + 1):
                future_position = position + step
                if future_position >= len(indices):
                    future_values.append(0.0)
                    continue
                future_index = indices[future_position]
                future_value = pd.to_numeric(
                    pd.Series(
                        [work.at[future_index, "enter_value_pct"]]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.isna(future_value):
                    unresolved = True
                    break
                future_values.append(float(future_value))
            if unresolved or pd.isna(enter_value):
                continue
            wait_value = float(np.mean(future_values))
            work.at[index, "wait_option_value_pct"] = wait_value
            work.at[index, "enter_minus_wait_option_pct"] = float(
                enter_value - wait_value
            )
    return work


def fit_classifier(
    states: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[HistGradientBoostingClassifier, dict]:
    target = pd.to_numeric(
        states.enter_minus_wait_option_pct,
        errors="coerce",
    )
    train = states.loc[target.notna()].copy()
    y = pd.to_numeric(
        train.enter_minus_wait_option_pct,
        errors="coerce",
    ).gt(0).astype(int)
    if len(train) < MIN_TRAIN_TARGET_ROWS:
        raise ValueError(
            f"Request256 target support only {len(train)}"
        )
    if y.nunique() != 2:
        raise ValueError("Request256 target needs both classes")
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261359,
    )
    model.fit(
        controller_x(train, columns),
        y,
        sample_weight=_weights(train),
    )
    return model, {
        "rows": int(len(train)),
        "positive_rate": float(y.mean()),
        "mean_advantage_pct": float(
            pd.to_numeric(
                train.enter_minus_wait_option_pct,
                errors="coerce",
            ).mean()
        ),
    }


def signal_report(
    states: pd.DataFrame,
    model: HistGradientBoostingClassifier,
    columns: tuple[str, ...],
) -> dict:
    pullback = states.loc[pullback_mask(states)].copy()
    target = pd.to_numeric(
        pullback.enter_minus_wait_option_pct,
        errors="coerce",
    )
    pullback = pullback.loc[target.notna()].copy()
    target = pd.to_numeric(
        pullback.enter_minus_wait_option_pct,
        errors="coerce",
    )
    probability = (
        model.predict_proba(controller_x(pullback, columns))[:, 1]
        if len(pullback)
        else np.array([], dtype=float)
    )
    y = target.gt(0).astype(int)
    auc = (
        float(roc_auc_score(y, probability))
        if len(pullback) >= 20 and y.nunique() == 2
        else None
    )
    return {
        "rows": int(len(pullback)),
        "positive_rate": float(y.mean()) if len(y) else None,
        "mean_advantage_pct": (
            float(target.mean()) if len(target) else None
        ),
        "auc": auc,
    }


def make_policy(
    states: pd.DataFrame,
    selected: pd.DataFrame,
    model: HistGradientBoostingClassifier,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    scored = states.copy()
    if len(scored):
        scored["wait_option_probability"] = model.predict_proba(
            controller_x(scored, columns)
        )[:, 1]
    else:
        scored["wait_option_probability"] = pd.Series(dtype=float)

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
                    float(state.wait_option_probability)
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
            }
        )
    return pd.DataFrame(records)


def fit_one_step_model_for_comparator(
    train_states: pd.DataFrame,
    columns: tuple[str, ...],
) -> HistGradientBoostingClassifier:
    target = pd.to_numeric(
        train_states.enter_minus_wait_pct,
        errors="coerce",
    )
    train = train_states.loc[target.notna()].copy()
    y = pd.to_numeric(
        train.enter_minus_wait_pct,
        errors="coerce",
    ).gt(0).astype(int)
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
    return model


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
) -> int:
    from .full_hot_fixed_policy_value import attach_fixed_value

    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
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

    train_states = attach_multistep_target(
        build_episode_states(train_selected, scan)
    )
    test_states = attach_multistep_target(
        build_episode_states(test_selected, scan)
    )
    columns = controller_columns(train_states)
    model, train_support = fit_classifier(train_states, columns)
    signal = signal_report(test_states, model, columns)

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
    one_step_model = fit_one_step_model_for_comparator(
        train_states,
        columns,
    )
    one_step_rows = make_one_step_policy(
        test_states,
        test_selected,
        one_step_model,
        columns,
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
    one_step_report = policy_metrics(
        one_step_rows,
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
        "train_target_rows": (
            train_support["rows"] >= MIN_TRAIN_TARGET_ROWS
        ),
        "test_pullback_states": (
            signal["rows"] >= MIN_TEST_PULLBACK_STATES
        ),
        "pullback_auc": (
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
        "wait_option_horizon_states": WAIT_OPTION_HORIZON,
        "probability_threshold": ADVANTAGE_PROBABILITY_THRESHOLD,
        "support": {
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
        "train_target_support": train_support,
        "development_signal": signal,
        "development_policies": {
            "immediate": immediate_report,
            "fixed_2pct_pullback": fixed_report,
            "one_step_advantage_gate": one_step_report,
            "three_opportunity_wait_gate": policy_report,
        },
        "mean_gain_vs_fixed_2pct_pullback_pct": gain_vs_fixed,
        "checks": checks,
        "economic_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "The WAIT option is a fixed mean over the next three decision "
            "opportunities, not a hindsight maximum. Development-only."
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
