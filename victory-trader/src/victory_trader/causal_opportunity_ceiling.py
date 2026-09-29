"""Request265: causal full-HOT opportunity ceiling and candidate signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score

from .causal_minute_controller_rebuild import causal_execution_scan
from .full_hot_fixed_policy_value import (
    attach_fixed_value,
    usable_columns,
    xframe,
)
from .learned_pullback_entry import (
    CANDIDATE_MODEL_DAYS,
    TEST_DAYS,
    build_episode_states,
)

REQUEST_ID = 265
MIN_ORACLE_MEAN = 0.50
MIN_ANY_POSITIVE_RATE = 0.50
MIN_SPEARMAN = 0.08
MIN_AUC = 0.58


def fit_candidate_models(first_hot: pd.DataFrame):
    fit = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CANDIDATE_MODEL_DAYS)
    ].copy()
    target = pd.to_numeric(
        fit.fixed_first_watch_value_pct,
        errors="coerce",
    )
    fit = fit.loc[target.notna()].copy()
    target = pd.to_numeric(
        fit.fixed_first_watch_value_pct,
        errors="coerce",
    ).to_numpy(float)
    columns = usable_columns(fit)
    if len(fit) < 500:
        raise ValueError("Request265 candidate support too small")
    positive = target > 0
    if len(np.unique(positive)) != 2:
        raise ValueError("Request265 candidate classes missing")

    low, high = np.quantile(target, [0.005, 0.995])
    reg = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261390,
    )
    cls = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261391,
    )
    x = xframe(fit, columns)
    reg.fit(x, np.clip(target, low, high))
    cls.fit(x, positive.astype(int))
    return reg, cls, columns, {
        "rows": int(len(fit)),
        "mean_pct": float(np.mean(target)),
        "positive_rate": float(np.mean(positive)),
    }


def candidate_report(
    first_hot: pd.DataFrame,
    reg,
    cls,
    columns: tuple[str, ...],
) -> dict:
    test = first_hot.loc[
        first_hot.trading_day.astype(str).isin(TEST_DAYS)
    ].copy()
    target = pd.to_numeric(
        test.fixed_first_watch_value_pct,
        errors="coerce",
    )
    test = test.loc[target.notna()].copy()
    target = pd.to_numeric(
        test.fixed_first_watch_value_pct,
        errors="coerce",
    )
    x = xframe(test, columns)
    pred = reg.predict(x)
    prob = cls.predict_proba(x)[:, 1]
    spearman = pd.Series(pred).corr(
        pd.Series(target.to_numpy(float)),
        method="spearman",
    )
    auc = (
        float(roc_auc_score(target.gt(0).astype(int), prob))
        if target.gt(0).nunique() == 2
        else None
    )
    q80 = float(np.quantile(prob, 0.80))
    top = target.loc[prob >= q80]
    return {
        "rows": int(len(test)),
        "population_mean_pct": float(target.mean()),
        "population_positive_rate": float(target.gt(0).mean()),
        "value_spearman": (
            None if pd.isna(spearman) else float(spearman)
        ),
        "positive_auc": auc,
        "top20_rows": int(len(top)),
        "top20_mean_pct": float(top.mean()) if len(top) else None,
        "top20_positive_rate": (
            float(top.gt(0).mean()) if len(top) else None
        ),
    }


def opportunity_report(
    states: pd.DataFrame,
    episode_count: int,
) -> dict:
    episodes = []
    keys = ["trading_day", "ticker", "hot_t"]
    for key, group in states.groupby(keys, sort=False):
        ordered = group.sort_values("state_t")
        values = pd.to_numeric(
            ordered.enter_value_pct,
            errors="coerce",
        )
        finite = ordered.loc[values.notna()].copy()
        if finite.empty:
            episodes.append(
                {
                    "key": key,
                    "first": np.nan,
                    "pullback": np.nan,
                    "pullback_entered": False,
                    "oracle": np.nan,
                    "any_positive": False,
                }
            )
            continue

        first = float(finite.iloc[0].enter_value_pct)
        pullback_rows = ordered.loc[
            pd.to_numeric(
                ordered.drawdown_from_running_high_pct,
                errors="coerce",
            ).le(-2.0)
        ]
        if len(pullback_rows):
            pb = pullback_rows.iloc[0]
            pullback_entered = True
            pullback = pd.to_numeric(
                pd.Series([pb.enter_value_pct]),
                errors="coerce",
            ).iloc[0]
        else:
            pullback_entered = False
            pullback = 0.0
        oracle = float(
            pd.to_numeric(
                finite.enter_value_pct,
                errors="coerce",
            ).max()
        )
        episodes.append(
            {
                "key": key,
                "first": first,
                "pullback": (
                    float(pullback) if pd.notna(pullback) else np.nan
                ),
                "pullback_entered": pullback_entered,
                "oracle": oracle,
                "any_positive": oracle > 0,
            }
        )

    frame = pd.DataFrame(episodes)
    first = pd.to_numeric(frame["first"], errors="coerce")
    pullback = pd.to_numeric(frame["pullback"], errors="coerce")
    oracle = pd.to_numeric(frame["oracle"], errors="coerce")
    return {
        "requested_episodes": int(episode_count),
        "episodes_with_states": int(len(frame)),
        "first_state_coverage": float(first.notna().mean()) if len(frame) else 0.0,
        "first_state_mean_pct": float(first.mean()) if first.notna().any() else None,
        "first_state_positive_rate": (
            float(first.gt(0).mean()) if first.notna().any() else None
        ),
        "fixed_2pct_pullback_entry_rate": (
            float(frame.pullback_entered.mean()) if len(frame) else 0.0
        ),
        "fixed_2pct_pullback_resolution_or_cash_rate": (
            float(pullback.notna().mean()) if len(frame) else 0.0
        ),
        "fixed_2pct_pullback_mean_pct": (
            float(pullback.mean()) if pullback.notna().any() else None
        ),
        "oracle_coverage": float(oracle.notna().mean()) if len(frame) else 0.0,
        "oracle_best_state_mean_pct": (
            float(oracle.mean()) if oracle.notna().any() else None
        ),
        "oracle_best_state_positive_rate": (
            float(oracle.gt(0).mean()) if oracle.notna().any() else None
        ),
        "any_positive_state_rate": (
            float(frame.any_positive.mean()) if len(frame) else 0.0
        ),
        "oracle_mean_improvement_vs_first_pct": (
            float(oracle.mean() - first.mean())
            if oracle.notna().any() and first.notna().any()
            else None
        ),
    }


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    causal_scan = causal_execution_scan(scan)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, causal_scan)

    reg, cls, columns, fit_support = fit_candidate_models(first_hot)
    candidate = candidate_report(first_hot, reg, cls, columns)

    test_hot = first_hot.loc[
        first_hot.trading_day.astype(str).isin(TEST_DAYS)
    ].copy()
    states = build_episode_states(test_hot, causal_scan)
    opportunity = opportunity_report(states, len(test_hot))

    opportunity_pass = bool(
        opportunity["oracle_best_state_mean_pct"] is not None
        and opportunity["oracle_best_state_mean_pct"]
        >= MIN_ORACLE_MEAN
        and opportunity["any_positive_state_rate"]
        >= MIN_ANY_POSITIVE_RATE
    )
    signal_pass = bool(
        candidate["value_spearman"] is not None
        and candidate["value_spearman"] >= MIN_SPEARMAN
        and candidate["positive_auc"] is not None
        and candidate["positive_auc"] >= MIN_AUC
    )

    if opportunity_pass and not signal_pass:
        diagnosis = "opportunity_exists_representation_fails"
    elif not opportunity_pass:
        diagnosis = "causal_opportunity_ceiling_too_low"
    else:
        diagnosis = "candidate_signal_and_opportunity_exist"

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "causal_reference_contract": True,
        "fit_support": fit_support,
        "candidate_signal": candidate,
        "opportunity_ceiling": opportunity,
        "opportunity_gate_pass": opportunity_pass,
        "candidate_signal_gate_pass": signal_pass,
        "diagnosis": diagnosis,
        "interpretation": (
            "Oracle best-state metrics are diagnostic ceilings only and are "
            "not deployable policy results. All economic references are causal "
            "decision-time minute opens."
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
