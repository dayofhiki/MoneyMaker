"""Request277: strictly lagged completed-minute context at pullback admission.

Request275 found useful ranking signal but no positive admitted subset, while
Request276 showed that price-only rebound confirmation made outcomes worse.
This diagnostic asks whether causal context from the minute that has fully
completed *before* the decision helps distinguish recoverable pullbacks from
continued deterioration.

No new market dates are opened. May5-8 are used only in leave-one-day-out
cross-fitting. The downstream entry/exit contract remains frozen from
Request275. May11-20 and June15-19 are untouched.
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
from .full_hot_fixed_policy_value import attach_fixed_value
from .joint_pullback_admission import (
    FIXED_MAX_HOLD_MINUTES,
    FIXED_STOP_LOSS_PCT,
    FIXED_TRAIL_PCT,
    build_pullback_state_rows,
    day_weights,
    feature_columns,
    feature_x,
)
from .minute_reference_alignment_audit import build_reference_maps
from .pullback_trailing_exit import apply_exit_rule, build_pullback_episodes
from .rich_post_hot_state import (
    CURRENT_MINUTE_FEATURES,
    attach_minute_features,
    score_and_select_candidates,
)

REQUEST_ID = 277
CROSSFIT_DAYS = (
    "2026-05-05",
    "2026-05-06",
    "2026-05-07",
    "2026-05-08",
)
MIN_TRAIN_ROWS = 150
MIN_OOF_ROWS = 180
MIN_TIMING_SHIFT_RATE = 0.99
MIN_VALUE_SPEARMAN = 0.20
MIN_VALUE_GAIN = 0.05
MIN_POSITIVE_AUC = 0.60
MIN_POSITIVE_AUC_GAIN = 0.02
MIN_SEVERE_AUC = 0.65
MIN_SEVERE_AUC_GAIN = 0.03


def fit_triplet(
    train: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
):
    target = pd.to_numeric(
        train.trade_return_pct,
        errors="coerce",
    )
    train = train.loc[
        train.resolved.astype(bool) & target.notna()
    ].copy()
    y = pd.to_numeric(
        train.trade_return_pct,
        errors="coerce",
    ).to_numpy(float)
    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Request277 fold has only {len(train)} training rows"
        )

    positive = (y > 0).astype(int)
    severe = (y <= -2.0).astype(int)
    if np.unique(positive).size != 2:
        raise ValueError("Request277 positive target lacks both classes")
    if np.unique(severe).size != 2:
        raise ValueError("Request277 severe target lacks both classes")

    low, high = np.quantile(y, [0.01, 0.99])
    weights = day_weights(train)
    kwargs = {
        "learning_rate": 0.04,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 25,
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
    x = feature_x(train, columns)
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


def predict_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    prefix: str,
    seed: int,
) -> pd.DataFrame:
    reg, pos, sev = fit_triplet(
        train,
        columns,
        seed,
    )
    out = test.copy()
    x = feature_x(out, columns)
    out[f"{prefix}_predicted_return_pct"] = reg.predict(x)
    out[f"{prefix}_positive_probability"] = (
        pos.predict_proba(x)[:, 1]
    )
    out[f"{prefix}_severe_probability"] = (
        sev.predict_proba(x)[:, 1]
    )
    return out


def safe_auc(
    target: pd.Series,
    score: pd.Series,
) -> float | None:
    y = pd.to_numeric(target, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.notna() & s.notna()
    y = y.loc[valid].astype(int)
    s = s.loc[valid].astype(float)
    if len(y) < 20 or y.nunique() != 2:
        return None
    return float(roc_auc_score(y, s))


def representation_metrics(
    rows: pd.DataFrame,
    prefix: str,
) -> dict:
    actual = pd.to_numeric(
        rows.trade_return_pct,
        errors="coerce",
    )
    valid = (
        rows.resolved.astype(bool)
        & actual.notna()
    )
    frame = rows.loc[valid].copy()
    actual = pd.to_numeric(
        frame.trade_return_pct,
        errors="coerce",
    )
    predicted = pd.to_numeric(
        frame[f"{prefix}_predicted_return_pct"],
        errors="coerce",
    )
    spearman = actual.corr(
        predicted,
        method="spearman",
    )
    positive_auc = safe_auc(
        actual.gt(0).astype(int),
        frame[f"{prefix}_positive_probability"],
    )
    severe_auc = safe_auc(
        actual.le(-2.0).astype(int),
        frame[f"{prefix}_severe_probability"],
    )

    by_day = {}
    for day, part in frame.groupby(
        frame.trading_day.astype(str),
        sort=True,
    ):
        day_actual = pd.to_numeric(
            part.trade_return_pct,
            errors="coerce",
        )
        day_pred = pd.to_numeric(
            part[f"{prefix}_predicted_return_pct"],
            errors="coerce",
        )
        day_spearman = day_actual.corr(
            day_pred,
            method="spearman",
        )
        by_day[str(day)] = {
            "rows": int(len(part)),
            "actual_mean_pct": float(
                day_actual.mean()
            ),
            "value_spearman": (
                None
                if pd.isna(day_spearman)
                else float(day_spearman)
            ),
            "positive_rate": float(
                day_actual.gt(0).mean()
            ),
            "severe_rate_le_minus2": float(
                day_actual.le(-2.0).mean()
            ),
        }

    return {
        "rows": int(len(frame)),
        "actual_mean_pct": (
            float(actual.mean())
            if len(actual)
            else None
        ),
        "actual_positive_rate": (
            float(actual.gt(0).mean())
            if len(actual)
            else None
        ),
        "actual_severe_rate_le_minus2": (
            float(actual.le(-2.0).mean())
            if len(actual)
            else None
        ),
        "value_spearman": (
            None
            if pd.isna(spearman)
            else float(spearman)
        ),
        "positive_auc": positive_auc,
        "severe_auc": severe_auc,
        "by_day": by_day,
    }


def evaluate(
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    raw_scan = pd.read_parquet(scan_path)
    _, _, timing = build_reference_maps(raw_scan)
    causal_scan = causal_execution_scan(raw_scan)

    first_hot = pd.read_parquet(first_hot_path).drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(
        first_hot,
        causal_scan,
    )
    scored, _, _ = score_and_select_candidates(first_hot)
    population = scored.loc[
        scored.trading_day.astype(str).isin(
            CROSSFIT_DAYS
        )
    ].copy()

    states = build_pullback_state_rows(
        population,
        causal_scan,
    )
    episodes = build_pullback_episodes(
        population,
        causal_scan,
    )
    outcomes = apply_exit_rule(
        episodes,
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name="fixed_downstream_teacher",
    )
    states = states.merge(
        outcomes.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "resolved",
            "trade_return_pct",
            "exit_t",
            "exit_reason",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="one_to_one",
    )
    if states.resolved.isna().any():
        raise ValueError(
            "Request277 state/outcome merge lost pullback rows"
        )

    baseline_columns = feature_columns(states)
    states["state_t"] = pd.to_numeric(
        states.entry_t,
        errors="raise",
    ).astype("int64")
    enriched = attach_minute_features(
        states,
        raw_scan,
    )
    minute_columns = tuple(
        name
        for name in CURRENT_MINUTE_FEATURES
        if name in enriched
        and pd.to_numeric(
            enriched[name],
            errors="coerce",
        ).notna().any()
    )
    enriched_columns = tuple(
        dict.fromkeys([
            *baseline_columns,
            *minute_columns,
        ])
    )

    oof_parts = []
    fold_support = {}
    for fold_index, held_day in enumerate(
        CROSSFIT_DAYS
    ):
        train = enriched.loc[
            ~enriched.trading_day.astype(str).eq(
                held_day
            )
        ].copy()
        held = enriched.loc[
            enriched.trading_day.astype(str).eq(
                held_day
            )
        ].copy()
        held = held.loc[
            held.resolved.astype(bool)
            & pd.to_numeric(
                held.trade_return_pct,
                errors="coerce",
            ).notna()
        ].copy()

        baseline_pred = predict_fold(
            train,
            held,
            baseline_columns,
            prefix="baseline",
            seed=20261377 + fold_index * 10,
        )
        enriched_pred = predict_fold(
            train,
            held,
            enriched_columns,
            prefix="enriched",
            seed=20261417 + fold_index * 10,
        )
        columns_to_add = [
            "enriched_predicted_return_pct",
            "enriched_positive_probability",
            "enriched_severe_probability",
        ]
        baseline_pred = baseline_pred.merge(
            enriched_pred.loc[
                :,
                [
                    "trading_day",
                    "ticker",
                    "hot_t",
                    *columns_to_add,
                ],
            ],
            on=["trading_day", "ticker", "hot_t"],
            how="left",
            validate="one_to_one",
        )
        baseline_pred["held_out_day"] = held_day
        oof_parts.append(baseline_pred)
        fold_support[held_day] = {
            "train_rows": int(
                (
                    train.resolved.astype(bool)
                    & pd.to_numeric(
                        train.trade_return_pct,
                        errors="coerce",
                    ).notna()
                ).sum()
            ),
            "held_out_rows": int(len(held)),
        }

    oof = pd.concat(
        oof_parts,
        ignore_index=True,
    )
    baseline = representation_metrics(
        oof,
        "baseline",
    )
    enriched_report = representation_metrics(
        oof,
        "enriched",
    )

    value_gain = (
        enriched_report["value_spearman"]
        - baseline["value_spearman"]
        if enriched_report["value_spearman"]
        is not None
        and baseline["value_spearman"] is not None
        else None
    )
    positive_gain = (
        enriched_report["positive_auc"]
        - baseline["positive_auc"]
        if enriched_report["positive_auc"] is not None
        and baseline["positive_auc"] is not None
        else None
    )
    severe_gain = (
        enriched_report["severe_auc"]
        - baseline["severe_auc"]
        if enriched_report["severe_auc"] is not None
        and baseline["severe_auc"] is not None
        else None
    )

    checks = {
        "timing_shift_contract": (
            timing["exact_60s_shift_rate"] is not None
            and timing["exact_60s_shift_rate"]
            >= MIN_TIMING_SHIFT_RATE
        ),
        "oof_support": (
            enriched_report["rows"] >= MIN_OOF_ROWS
        ),
        "enriched_value_spearman": (
            enriched_report["value_spearman"] is not None
            and enriched_report["value_spearman"]
            >= MIN_VALUE_SPEARMAN
        ),
        "value_spearman_gain": (
            value_gain is not None
            and value_gain >= MIN_VALUE_GAIN
        ),
        "enriched_positive_auc": (
            enriched_report["positive_auc"] is not None
            and enriched_report["positive_auc"]
            >= MIN_POSITIVE_AUC
        ),
        "positive_auc_gain": (
            positive_gain is not None
            and positive_gain >= MIN_POSITIVE_AUC_GAIN
        ),
        "enriched_severe_auc": (
            enriched_report["severe_auc"] is not None
            and enriched_report["severe_auc"]
            >= MIN_SEVERE_AUC
        ),
        "severe_auc_gain": (
            severe_gain is not None
            and severe_gain >= MIN_SEVERE_AUC_GAIN
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(CROSSFIT_DAYS),
        "downstream_policy": {
            "entry": "first causal 2% pullback",
            "hard_stop_loss_pct": FIXED_STOP_LOSS_PCT,
            "trailing_drawdown_pct": FIXED_TRAIL_PCT,
            "max_hold_minutes": FIXED_MAX_HOLD_MINUTES,
            "missing_deadline": "unresolved",
        },
        "timing_audit": timing,
        "causal_feature_contract": (
            "execution price is the open at bar_start_t == decision_t; "
            "minute context joins raw scan.t == decision_t, which is the "
            "bar whose bar_start_t is decision_t-60s and is therefore "
            "fully completed before the decision"
        ),
        "support": {
            "candidate_episodes": int(len(population)),
            "pullback_states": int(len(states)),
            "resolved_pullback_states": int(
                (
                    states.resolved.astype(bool)
                    & pd.to_numeric(
                        states.trade_return_pct,
                        errors="coerce",
                    ).notna()
                ).sum()
            ),
            "folds": fold_support,
        },
        "baseline_features": list(
            baseline_columns
        ),
        "added_completed_minute_features": list(
            minute_columns
        ),
        "baseline_oof": baseline,
        "enriched_oof": enriched_report,
        "enriched_minus_baseline": {
            "value_spearman": value_gain,
            "positive_auc": positive_gain,
            "severe_auc": severe_gain,
        },
        "checks": checks,
        "representation_gate_pass": bool(
            all(checks.values())
        ),
        "interpretation": (
            "All predictions are out-of-fold by trading day. "
            "No May11-20 or June15-19 outcome is accessed. "
            "Minute OHLCV context is taken only from the fully "
            "completed minute before the execution timestamp."
        ),
        "next_boundary": (
            "build cross-fitted economic admission from enriched context"
            if all(checks.values())
            else (
                "redesign candidate-level state before adding more "
                "entry/exit rules; keep June15-19 unopened"
            )
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    rows_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    oof.to_parquet(
        rows_output,
        index=False,
    )
    print(
        json.dumps(
            result,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first-hot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--opportunity-scan",
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
        args.first_hot,
        args.opportunity_scan,
        args.output,
        args.rows_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
