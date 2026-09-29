"""Request285: fixed post-pullback recheck learnability diagnostic.

Requests283-284 showed that the first causal 2% pullback is not reliably
separable inside the stage-1 shortlist. This request asks whether the problem
becomes more learnable after one, two, or three additional observed states.

No recurrent policy is created. For each fixed horizon 0/1/2/3 after the first
2% pullback, the candidate is treated as entering exactly at that state and is
then managed by the same frozen downstream exit rule. The model never chooses
the best later entry state by hindsight.
"""
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
from .causal_pullback_entry_audit import RISK_CAP_MINUTES
from .downstream_aligned_candidate import (
    attach_downstream_target,
    candidate_columns,
)
from .full_hot_fixed_policy_value import attach_fixed_value, event_lookup
from .joint_pullback_admission import (
    FIXED_MAX_HOLD_MINUTES,
    FIXED_STOP_LOSS_PCT,
    FIXED_TRAIL_PCT,
    feature_x,
)
from .lagged_minute_context import CROSSFIT_DAYS
from .joint_pullback_admission import day_weights
from .learned_pullback_entry import (
    build_episode_states,
    controller_columns,
)
from .prehot_context_ablation import usable_context_columns
from .pullback_trailing_exit import apply_exit_rule, build_pullback_episodes
from .rich_post_hot_state import (
    CURRENT_MINUTE_FEATURES,
    attach_minute_features,
)
from .two_stage_risk_veto import (
    attach_ticker_history,
    fold_stage1,
)

REQUEST_ID = 285
RECHECK_HORIZONS = (0, 1, 2, 3)
MIN_CONDITIONED_RESOLVED = 20
MIN_VALUE_SPEARMAN = 0.15
MIN_POSITIVE_AUC = 0.60
MIN_SEVERE_AUC = 0.65
MIN_SIGNAL_GAIN = 0.05


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_recheck_states(
    candidates: pd.DataFrame,
    causal_scan: pd.DataFrame,
) -> pd.DataFrame:
    source = candidates.copy()
    source["candidate_probability"] = 0.5
    states = build_episode_states(source, causal_scan)
    if states.empty:
        return states

    rows = []
    for _, group in states.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        ordered = group.sort_values("state_t").reset_index(drop=True)
        drawdown = _numeric(
            ordered.drawdown_from_running_high_pct
        )
        trigger_positions = np.flatnonzero(
            drawdown.le(-2.0).to_numpy()
        )
        if len(trigger_positions) == 0:
            continue
        trigger = int(trigger_positions[0])
        for horizon in RECHECK_HORIZONS:
            position = trigger + int(horizon)
            if position >= len(ordered):
                continue
            row = ordered.iloc[position].to_dict()
            row["recheck_horizon_events"] = int(horizon)
            row["entry_t"] = int(row["state_t"])
            row["entry_price"] = float(row["state_price"])
            rows.append(row)
    return pd.DataFrame(rows)


def delayed_policy(
    recheck_states: pd.DataFrame,
    causal_scan: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    selected = recheck_states.loc[
        recheck_states.recheck_horizon_events.eq(int(horizon))
    ].copy()
    lookup = event_lookup(causal_scan)
    episodes = []
    for row in selected.itertuples(index=False):
        day = str(row.trading_day)
        ticker = str(row.ticker).upper()
        hot_t = int(row.hot_t)
        entry_t = int(row.entry_t)
        entry_price = float(row.entry_price)
        cap = hot_t + RISK_CAP_MINUTES * 60_000
        path = [
            (int(t), float(price))
            for t, price in lookup.get((day, ticker), [])
            if entry_t < int(t) <= cap and float(price) > 0
        ]
        episodes.append({
            "trading_day": day,
            "ticker": ticker,
            "hot_t": hot_t,
            "entered": True,
            "entry_t": entry_t,
            "entry_price": entry_price,
            "path": path,
        })
    if not episodes:
        return pd.DataFrame()
    policy = apply_exit_rule(
        pd.DataFrame(episodes),
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name=f"request285_recheck_h{int(horizon)}",
    )
    policy["recheck_horizon_events"] = int(horizon)
    return policy


def attach_policy_labels(
    states: pd.DataFrame,
    policy: pd.DataFrame,
) -> pd.DataFrame:
    if states.empty or policy.empty:
        return states.iloc[0:0].copy()
    return states.merge(
        policy.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "recheck_horizon_events",
            "resolved",
            "trade_return_pct",
            "exit_t",
            "exit_reason",
        ]],
        on=[
            "trading_day",
            "ticker",
            "hot_t",
            "recheck_horizon_events",
        ],
        how="inner",
        validate="one_to_one",
    )


def representation_columns(
    states: pd.DataFrame,
    ticker_extra: tuple[str, ...],
) -> tuple[str, ...]:
    base = tuple(
        name
        for name in controller_columns(states)
        if name != "candidate_probability"
    )
    ticker = usable_context_columns(states, ticker_extra)
    minute = tuple(
        name
        for name in CURRENT_MINUTE_FEATURES
        if name in states
        and _numeric(states[name]).notna().any()
    )
    return tuple(dict.fromkeys([*base, *ticker, *minute]))


def fit_recheck_triplet(
    train: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
):
    """Smaller fixed-capacity diagnostic model for shrinking delayed support."""
    target = _numeric(train.trade_return_pct)
    fit = train.loc[
        train.resolved.astype(bool) & target.notna()
    ].copy()
    y = _numeric(fit.trade_return_pct).to_numpy(float)
    if len(fit) < 60:
        raise ValueError(
            f"Request285 delayed fold has only {len(fit)} training rows"
        )
    positive = (y > 0).astype(int)
    severe = (y <= -2.0).astype(int)
    if np.unique(positive).size != 2:
        raise ValueError("Request285 positive target lacks both classes")
    if np.unique(severe).size != 2:
        raise ValueError("Request285 severe target lacks both classes")
    low, high = np.quantile(y, [0.01, 0.99])
    weights = day_weights(fit)
    kwargs = {
        "learning_rate": 0.04,
        "max_iter": 160,
        "max_leaf_nodes": 11,
        "min_samples_leaf": 15,
        "l2_regularization": 3.0,
        "early_stopping": False,
    }
    reg = HistGradientBoostingRegressor(
        **kwargs,
        random_state=seed,
    )
    pos = HistGradientBoostingClassifier(
        **kwargs,
        random_state=seed + 1,
    )
    sev = HistGradientBoostingClassifier(
        **kwargs,
        random_state=seed + 2,
    )
    x = feature_x(fit, columns)
    reg.fit(
        x,
        np.clip(y, low, high),
        sample_weight=weights,
    )
    pos.fit(
        x,
        positive,
        sample_weight=weights,
    )
    sev.fit(
        x,
        severe,
        sample_weight=weights,
    )
    return reg, pos, sev


def _safe_spearman(
    actual: pd.Series,
    predicted: pd.Series,
) -> float | None:
    a = _numeric(actual)
    p = _numeric(predicted)
    valid = a.notna() & p.notna()
    if int(valid.sum()) < 20:
        return None
    value = a.loc[valid].corr(p.loc[valid], method="spearman")
    return None if pd.isna(value) else float(value)


def horizon_metrics(
    frame: pd.DataFrame,
    prefix: str,
) -> dict:
    work = frame.loc[
        frame.selected_prehot.fillna(False).astype(bool)
        & frame.resolved.astype(bool)
        & _numeric(frame.trade_return_pct).notna()
    ].copy()
    actual = _numeric(work.trade_return_pct)
    positive = actual.gt(0).astype(int)
    severe = actual.le(-2.0).astype(int)
    pos_score = _numeric(work[f"{prefix}_positive_probability"])
    sev_score = _numeric(work[f"{prefix}_severe_probability"])

    positive_auc = (
        float(roc_auc_score(positive, pos_score))
        if len(work) >= 20 and positive.nunique() == 2
        else None
    )
    severe_auc = (
        float(roc_auc_score(severe, sev_score))
        if len(work) >= 20 and severe.nunique() == 2
        else None
    )
    return {
        "resolved_shortlist_rows": int(len(work)),
        "trade_mean_pct": (
            float(actual.mean()) if len(actual) else None
        ),
        "trade_positive_rate": (
            float(positive.mean()) if len(positive) else None
        ),
        "trade_severe_rate_le_minus2": (
            float(severe.mean()) if len(severe) else None
        ),
        "value_spearman": _safe_spearman(
            actual,
            work[f"{prefix}_predicted_return_pct"],
        ),
        "positive_auc": positive_auc,
        "severe_auc": severe_auc,
    }


def evaluate(
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    raw_scan = pd.read_parquet(scan_path)
    raw_scan = raw_scan.loc[
        raw_scan.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    causal_scan = causal_execution_scan(raw_scan)

    first_hot = pd.read_parquet(first_hot_path).drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = first_hot.loc[
        first_hot.trading_day.astype(str).isin(CROSSFIT_DAYS)
    ].copy()
    first_hot = attach_fixed_value(first_hot, causal_scan)

    first_policy = apply_exit_rule(
        build_pullback_episodes(first_hot, causal_scan),
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name="request285_h0_reference",
    )
    labeled = attach_downstream_target(first_hot, first_policy)
    candidates, ticker_extra = attach_ticker_history(
        labeled,
        raw_scan,
    )
    prehot_columns = tuple(
        dict.fromkeys([
            *candidate_columns(candidates),
            *ticker_extra,
        ])
    )

    recheck = build_recheck_states(candidates, causal_scan)
    recheck = attach_minute_features(recheck, raw_scan)
    columns = representation_columns(recheck, ticker_extra)

    labeled_by_horizon = {}
    for horizon in RECHECK_HORIZONS:
        states_h = recheck.loc[
            recheck.recheck_horizon_events.eq(horizon)
        ].copy()
        policy_h = delayed_policy(
            recheck,
            causal_scan,
            horizon,
        )
        labeled_by_horizon[horizon] = attach_policy_labels(
            states_h,
            policy_h,
        )

    oof_parts = []
    metrics = {}
    for horizon in RECHECK_HORIZONS:
        rows = labeled_by_horizon[horizon]
        fold_parts = []
        for fold_index, held_day in enumerate(CROSSFIT_DAYS):
            train = rows.loc[
                ~rows.trading_day.astype(str).eq(held_day)
            ].copy()
            held = rows.loc[
                rows.trading_day.astype(str).eq(held_day)
            ].copy()

            outer_train_candidates = candidates.loc[
                ~candidates.trading_day.astype(str).eq(held_day)
            ].copy()
            held_candidates = candidates.loc[
                candidates.trading_day.astype(str).eq(held_day)
            ].copy()
            prehot, _ = fold_stage1(
                outer_train_candidates,
                held_candidates,
                prehot_columns,
                20261800 + fold_index * 10,
            )
            held = held.merge(
                prehot.loc[:, [
                    "trading_day",
                    "ticker",
                    "hot_t",
                    "selected_prehot",
                ]],
                on=["trading_day", "ticker", "hot_t"],
                how="left",
                validate="one_to_one",
            )
            if train.empty or held.empty:
                continue

            reg, positive, severe = fit_recheck_triplet(
                train,
                columns,
                20261900 + horizon * 100 + fold_index * 10,
            )
            x = feature_x(held, columns)
            prefix = f"h{horizon}"
            held[f"{prefix}_predicted_return_pct"] = reg.predict(x)
            held[f"{prefix}_positive_probability"] = (
                positive.predict_proba(x)[:, 1]
            )
            held[f"{prefix}_severe_probability"] = (
                severe.predict_proba(x)[:, 1]
            )
            fold_parts.append(held)

        if fold_parts:
            oof = pd.concat(fold_parts, ignore_index=True)
        else:
            oof = rows.iloc[0:0].copy()
        prefix = f"h{horizon}"
        metrics[str(horizon)] = horizon_metrics(oof, prefix)
        oof_parts.append(oof)

    baseline = metrics["0"]
    comparisons = {}
    best_horizon = None
    best_score = -999.0
    for horizon in RECHECK_HORIZONS[1:]:
        current = metrics[str(horizon)]
        signal_values = [
            value
            for value in [
                current["value_spearman"],
                current["positive_auc"],
                current["severe_auc"],
            ]
            if value is not None
        ]
        score = float(np.mean(signal_values)) if signal_values else -999.0
        if score > best_score:
            best_score = score
            best_horizon = int(horizon)

        comparisons[str(horizon)] = {
            "trade_mean_gain_vs_h0_pct": (
                float(
                    current["trade_mean_pct"]
                    - baseline["trade_mean_pct"]
                )
                if current["trade_mean_pct"] is not None
                and baseline["trade_mean_pct"] is not None
                else None
            ),
            "value_spearman_gain_vs_h0": (
                float(
                    current["value_spearman"]
                    - baseline["value_spearman"]
                )
                if current["value_spearman"] is not None
                and baseline["value_spearman"] is not None
                else None
            ),
            "positive_auc_gain_vs_h0": (
                float(
                    current["positive_auc"]
                    - baseline["positive_auc"]
                )
                if current["positive_auc"] is not None
                and baseline["positive_auc"] is not None
                else None
            ),
            "severe_auc_gain_vs_h0": (
                float(
                    current["severe_auc"]
                    - baseline["severe_auc"]
                )
                if current["severe_auc"] is not None
                and baseline["severe_auc"] is not None
                else None
            ),
        }

    best = (
        metrics[str(best_horizon)]
        if best_horizon is not None
        else None
    )
    best_comparison = (
        comparisons[str(best_horizon)]
        if best_horizon is not None
        else None
    )
    signal_gain = False
    if best_comparison is not None:
        gains = [
            best_comparison["value_spearman_gain_vs_h0"],
            best_comparison["positive_auc_gain_vs_h0"],
            best_comparison["severe_auc_gain_vs_h0"],
        ]
        signal_gain = any(
            value is not None and value >= MIN_SIGNAL_GAIN
            for value in gains
        )

    checks = {
        "best_support": (
            best is not None
            and best["resolved_shortlist_rows"]
            >= MIN_CONDITIONED_RESOLVED
        ),
        "best_value_spearman": (
            best is not None
            and best["value_spearman"] is not None
            and best["value_spearman"] >= MIN_VALUE_SPEARMAN
        ),
        "best_positive_auc": (
            best is not None
            and best["positive_auc"] is not None
            and best["positive_auc"] >= MIN_POSITIVE_AUC
        ),
        "best_severe_auc": (
            best is not None
            and best["severe_auc"] is not None
            and best["severe_auc"] >= MIN_SEVERE_AUC
        ),
        "signal_gain_vs_first_pullback": signal_gain,
    }
    delayed_state_signal_pass = bool(all(checks.values()))

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "recheck_horizons_events": list(RECHECK_HORIZONS),
        "feature_count": len(columns),
        "fixed_policy": {
            "entry": (
                "fixed state at first causal 2% pullback plus "
                "0/1/2/3 observed events"
            ),
            "hard_stop_loss_pct": FIXED_STOP_LOSS_PCT,
            "trailing_drawdown_pct": FIXED_TRAIL_PCT,
            "max_hold_minutes": FIXED_MAX_HOLD_MINUTES,
            "missing_exact_deadline": "unresolved",
            "future_best_state_search": False,
        },
        "horizon_metrics": metrics,
        "comparisons_vs_h0": comparisons,
        "best_delayed_horizon_events": best_horizon,
        "checks": checks,
        "delayed_state_signal_pass": delayed_state_signal_pass,
        "timing_contract": (
            "each horizon is fixed before evaluation; features use only path "
            "prefix through that recheck state plus strictly completed minute "
            "context at that state; later states are labels only"
        ),
        "next_boundary": (
            "build ENTER/WAIT/REJECT using repeated causal rechecks because "
            "additional observed states materially improve separability"
            if delayed_state_signal_pass
            else (
                "additional minute states do not make the shortlisted "
                "pullback sufficiently learnable; revisit the observation "
                "resolution/state source before adding recurrent actions"
            )
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if oof_parts:
        pd.concat(oof_parts, ignore_index=True).to_parquet(
            rows_output,
            index=False,
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
