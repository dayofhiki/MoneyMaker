"""Request275: joint candidate/pullback-state admission with fixed causal exits.

The downstream policy is frozen a priori. The model only decides whether a
causal 2% pullback state is worth admitting; rejected candidates remain cash.
May5-7 fit the model, May8 calibrates one admission threshold, and May11-20 is
development-only evaluation. June15-19 remains unopened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from .attention_replay import MINUTE_MS
from .causal_minute_controller_rebuild import causal_execution_scan
from .causal_pullback_entry_audit import ENTRY_WINDOW_MINUTES
from .full_hot_fixed_policy_value import attach_fixed_value, event_lookup, usable_columns
from .learned_pullback_entry import PATH_FEATURES, _pct
from .pullback_trailing_exit import apply_exit_rule, build_pullback_episodes
from .rich_post_hot_state import score_and_select_candidates

REQUEST_ID = 275
FIT_DAYS = ("2026-05-05", "2026-05-06", "2026-05-07")
CAL_DAY = "2026-05-08"
TEST_DAYS = (
    "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14",
    "2026-05-15", "2026-05-18", "2026-05-19", "2026-05-20",
)
FIXED_STOP_LOSS_PCT = -2.5
FIXED_TRAIL_PCT = 2.0
FIXED_MAX_HOLD_MINUTES = 10
ADMISSION_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.40)

MIN_FIT_ROWS = 120
MIN_CAL_ADMISSIONS = 10
MIN_CAL_RESOLUTION = 0.90
MAX_CAL_SEVERE = 0.20
MIN_TEST_ADMISSIONS = 20
MIN_TEST_RESOLUTION = 0.90
MIN_TEST_POSITIVE_DAYS = 5
MIN_TEST_POSITIVE_RATE = 0.45
MAX_TEST_SEVERE = 0.15


def build_pullback_state_rows(
    scored: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    """Return exactly the first causal 2% pullback state for each episode.

    Only prefix information through the trigger state is materialized. No
    downstream return/label is computed in this function.
    """
    lookup = event_lookup(scan)
    static_columns = usable_columns(scored)
    rows: list[dict[str, object]] = []

    for raw in scored.itertuples(index=False):
        day = str(raw.trading_day)
        ticker = str(raw.ticker).upper()
        hot_t = int(raw.t)
        limit_t = hot_t + ENTRY_WINDOW_MINUTES * MINUTE_MS
        events = [
            (int(t), float(price))
            for t, price in lookup.get((day, ticker), [])
            if hot_t < int(t) <= limit_t and float(price) > 0
        ]
        if len(events) < 2:
            continue

        first_price = float(events[0][1])
        running_high = first_price
        running_low = first_price
        high_index = 0
        low_index = 0
        path_abs_move = 0.0
        prices: list[float] = []
        base = {
            column: getattr(raw, column)
            for column in static_columns
            if hasattr(raw, column)
        }

        for event_index, (state_t, price) in enumerate(events):
            prices.append(float(price))
            if event_index:
                move = _pct(float(price), float(prices[event_index - 1]))
                if np.isfinite(move):
                    path_abs_move += abs(float(move))
            if float(price) >= running_high:
                running_high = float(price)
                high_index = event_index
            if float(price) <= running_low:
                running_low = float(price)
                low_index = event_index

            drawdown = _pct(float(price), running_high)
            if (
                event_index == 0
                or not np.isfinite(drawdown)
                or drawdown > -2.0
            ):
                continue

            current_return = _pct(float(price), first_price)
            high_return = _pct(running_high, first_price)
            low_return = _pct(running_low, first_price)
            recovery = _pct(float(price), running_low)
            path_range = _pct(running_high, running_low)
            range_den = max(running_high - running_low, 1e-12)
            recovery_fraction = float(
                (float(price) - running_low) / range_den
            )
            last_return = (
                _pct(float(price), float(prices[event_index - 1]))
                if event_index >= 1
                else 0.0
            )
            two_return = (
                _pct(float(price), float(prices[event_index - 2]))
                if event_index >= 2
                else last_return
            )
            displacement = (
                abs(float(current_return))
                if np.isfinite(current_return)
                else np.nan
            )
            efficiency = (
                float(displacement / path_abs_move)
                if np.isfinite(displacement) and path_abs_move > 1e-12
                else 1.0
            )
            rows.append({
                **base,
                "trading_day": day,
                "ticker": ticker,
                "hot_t": hot_t,
                "entry_t": int(state_t),
                "entry_price": float(price),
                "candidate_probability": float(raw.candidate_probability),
                "elapsed_minutes": float(
                    (int(state_t) - hot_t) / MINUTE_MS
                ),
                "event_index": int(event_index),
                "current_return_from_first_pct": current_return,
                "running_high_return_from_first_pct": high_return,
                "running_low_return_from_first_pct": low_return,
                "drawdown_from_running_high_pct": drawdown,
                "recovery_from_running_low_pct": recovery,
                "recovery_fraction_of_range": recovery_fraction,
                "last_event_return_pct": last_return,
                "two_event_return_pct": two_return,
                "path_range_pct": path_range,
                "path_efficiency": efficiency,
                "events_since_high": int(event_index - high_index),
                "events_since_low": int(event_index - low_index),
            })
            break

    return pd.DataFrame(rows)


def feature_columns(rows: pd.DataFrame) -> tuple[str, ...]:
    static = usable_columns(rows)
    return tuple(
        name
        for name in dict.fromkeys([*static, *PATH_FEATURES])
        if name in rows
        and pd.to_numeric(rows[name], errors="coerce").notna().any()
    )


def feature_x(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return (
        frame.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def day_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.trading_day.astype(str).map(
        frame.trading_day.astype(str).value_counts()
    ).astype(float)
    weights = 1.0 / counts.to_numpy(float)
    return weights / float(np.mean(weights))


def fit_models(
    rows: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    dict,
]:
    target = pd.to_numeric(rows.trade_return_pct, errors="coerce")
    train = rows.loc[
        rows.trading_day.astype(str).isin(FIT_DAYS)
        & rows.resolved.astype(bool)
        & target.notna()
    ].copy()
    y = pd.to_numeric(
        train.trade_return_pct,
        errors="coerce",
    ).to_numpy(float)
    if len(train) < MIN_FIT_ROWS:
        raise ValueError(
            f"Request275 finite fit support only {len(train)}"
        )
    if np.unique(y > 0).size != 2:
        raise ValueError(
            "Request275 fit needs positive and non-positive trades"
        )

    low, high = np.quantile(y, [0.01, 0.99])
    weights = day_weights(train)
    reg = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=20261375,
    )
    cls = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=20261376,
    )
    x = feature_x(train, columns)
    reg.fit(
        x,
        np.clip(y, low, high),
        sample_weight=weights,
    )
    cls.fit(
        x,
        (y > 0).astype(int),
        sample_weight=weights,
    )
    return reg, cls, {
        "rows": int(len(train)),
        "mean_trade_return_pct": float(np.mean(y)),
        "positive_rate": float(np.mean(y > 0)),
        "severe_loss_rate_le_minus2": float(np.mean(y <= -2.0)),
        "winsor": [float(low), float(high)],
    }


def attach_predictions(
    rows: pd.DataFrame,
    reg: HistGradientBoostingRegressor,
    cls: HistGradientBoostingClassifier,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    out = rows.copy()
    x = feature_x(out, columns)
    out["predicted_return_pct"] = reg.predict(x)
    out["positive_probability"] = cls.predict_proba(x)[:, 1]
    return out


def prediction_diagnostics(rows: pd.DataFrame) -> dict:
    actual = pd.to_numeric(
        rows.trade_return_pct,
        errors="coerce",
    )
    valid = rows.resolved.astype(bool) & actual.notna()
    frame = rows.loc[valid].copy()
    actual = pd.to_numeric(
        frame.trade_return_pct,
        errors="coerce",
    )
    if len(frame) < 20:
        return {
            "rows": int(len(frame)),
            "value_spearman": None,
            "positive_auc": None,
        }
    pred = pd.to_numeric(
        frame.predicted_return_pct,
        errors="coerce",
    )
    prob = pd.to_numeric(
        frame.positive_probability,
        errors="coerce",
    )
    spearman = actual.corr(pred, method="spearman")
    y = actual.gt(0).astype(int)
    auc = (
        float(roc_auc_score(y, prob))
        if y.nunique() == 2
        else None
    )
    return {
        "rows": int(len(frame)),
        "value_spearman": (
            None if pd.isna(spearman) else float(spearman)
        ),
        "positive_auc": auc,
        "actual_mean_pct": float(actual.mean()),
        "actual_positive_rate": float(y.mean()),
    }


def decision_metrics(
    policy_rows: pd.DataFrame,
    predicted_rows: pd.DataFrame,
    *,
    probability_threshold: float | None,
    require_positive_value: bool,
) -> dict:
    """Evaluate rejected/no-pullback episodes as cash."""
    decisions = policy_rows.copy()
    admission = predicted_rows.loc[:, [
        "trading_day",
        "ticker",
        "hot_t",
        "positive_probability",
        "predicted_return_pct",
    ]].copy()

    if probability_threshold is None:
        admission["admitted"] = True
    else:
        admission["admitted"] = (
            admission.positive_probability.ge(
                float(probability_threshold)
            )
        )
        if require_positive_value:
            admission["admitted"] &= (
                admission.predicted_return_pct.gt(0)
            )

    decisions = decisions.merge(
        admission.loc[:, [
            "trading_day",
            "ticker",
            "hot_t",
            "admitted",
        ]],
        on=["trading_day", "ticker", "hot_t"],
        how="left",
        validate="one_to_one",
    )
    decisions["admitted"] = (
        decisions.admitted.fillna(False).astype(bool)
    )
    decisions["decision_resolved"] = (
        ~decisions.admitted
        | decisions.resolved.astype(bool)
    )
    decisions["decision_return_pct"] = 0.0
    trade_mask = (
        decisions.admitted
        & decisions.resolved.astype(bool)
    )
    decisions.loc[
        trade_mask,
        "decision_return_pct",
    ] = pd.to_numeric(
        decisions.loc[trade_mask, "trade_return_pct"],
        errors="coerce",
    )
    decisions.loc[
        ~decisions.decision_resolved,
        "decision_return_pct",
    ] = np.nan

    resolved = decisions.loc[
        decisions.decision_resolved
    ].copy()
    trade = decisions.loc[trade_mask].copy()
    daily = resolved.groupby(
        resolved.trading_day.astype(str),
        sort=True,
    ).decision_return_pct.mean()

    return {
        "candidate_episodes": int(len(decisions)),
        "pullback_triggers": int(
            decisions.entered.astype(bool).sum()
        ),
        "admissions": int(decisions.admitted.sum()),
        "admission_rate_of_candidates": (
            float(decisions.admitted.mean())
            if len(decisions)
            else 0.0
        ),
        "resolution_or_cash_rate": (
            float(decisions.decision_resolved.mean())
            if len(decisions)
            else 0.0
        ),
        "candidate_mean_pct": (
            float(
                pd.to_numeric(
                    resolved.decision_return_pct,
                    errors="coerce",
                ).mean()
            )
            if len(resolved)
            else None
        ),
        "day_balanced_candidate_mean_pct": (
            float(daily.mean())
            if len(daily)
            else None
        ),
        "positive_candidate_mean_days": int(
            (daily > 0).sum()
        ),
        "trade_count": int(len(trade)),
        "trade_mean_pct": (
            float(
                pd.to_numeric(
                    trade.trade_return_pct,
                    errors="coerce",
                ).mean()
            )
            if len(trade)
            else None
        ),
        "trade_positive_rate": (
            float(
                pd.to_numeric(
                    trade.trade_return_pct,
                    errors="coerce",
                ).gt(0).mean()
            )
            if len(trade)
            else None
        ),
        "trade_severe_loss_rate_le_minus2": (
            float(
                pd.to_numeric(
                    trade.trade_return_pct,
                    errors="coerce",
                ).le(-2).mean()
            )
            if len(trade)
            else None
        ),
        "by_day": {
            str(day): {
                "candidates": int(len(part)),
                "admissions": int(part.admitted.sum()),
                "candidate_mean_pct": float(
                    pd.to_numeric(
                        part.loc[
                            part.decision_resolved,
                            "decision_return_pct",
                        ],
                        errors="coerce",
                    ).mean()
                ),
            }
            for day, part in decisions.groupby(
                decisions.trading_day.astype(str),
                sort=True,
            )
        },
    }


def subset_policy(
    rows: pd.DataFrame,
    days: tuple[str, ...],
) -> pd.DataFrame:
    return rows.loc[
        rows.trading_day.astype(str).isin(days)
    ].copy()


def subset_predictions(
    rows: pd.DataFrame,
    days: tuple[str, ...],
) -> pd.DataFrame:
    return rows.loc[
        rows.trading_day.astype(str).isin(days)
    ].copy()


def evaluate(
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    scan = causal_execution_scan(
        pd.read_parquet(scan_path)
    )
    first_hot = pd.read_parquet(first_hot_path).drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, scan)
    scored, _, _ = score_and_select_candidates(first_hot)

    used_days = (*FIT_DAYS, CAL_DAY, *TEST_DAYS)
    population = scored.loc[
        scored.trading_day.astype(str).isin(used_days)
    ].copy()

    state_rows = build_pullback_state_rows(
        population,
        scan,
    )
    episodes = build_pullback_episodes(
        population,
        scan,
    )
    policy = apply_exit_rule(
        episodes,
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name="fixed_downstream_teacher",
    )
    state_rows = state_rows.merge(
        policy.loc[:, [
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
    if state_rows.resolved.isna().any():
        raise ValueError(
            "Request275 state/policy merge lost pullback outcomes"
        )

    columns = feature_columns(state_rows)
    reg, cls, fit_report = fit_models(
        state_rows,
        columns,
    )
    predicted = attach_predictions(
        state_rows,
        reg,
        cls,
        columns,
    )

    cal_pred = subset_predictions(
        predicted,
        (CAL_DAY,),
    )
    cal_policy = subset_policy(
        policy,
        (CAL_DAY,),
    )
    cal_diag = prediction_diagnostics(cal_pred)
    baseline_cal = decision_metrics(
        cal_policy,
        cal_pred,
        probability_threshold=None,
        require_positive_value=False,
    )

    calibration_grid = []
    passing = []
    for fraction in ADMISSION_FRACTIONS:
        if cal_pred.empty:
            continue
        threshold = float(
            np.quantile(
                cal_pred.positive_probability.to_numpy(float),
                1.0 - float(fraction),
            )
        )
        metrics = decision_metrics(
            cal_policy,
            cal_pred,
            probability_threshold=threshold,
            require_positive_value=True,
        )
        gain = (
            metrics["candidate_mean_pct"]
            - baseline_cal["candidate_mean_pct"]
            if metrics["candidate_mean_pct"] is not None
            and baseline_cal["candidate_mean_pct"] is not None
            else None
        )
        checks = {
            "min_admissions": (
                metrics["admissions"]
                >= MIN_CAL_ADMISSIONS
            ),
            "resolution": (
                metrics["resolution_or_cash_rate"]
                >= MIN_CAL_RESOLUTION
            ),
            "candidate_mean_positive": (
                metrics["candidate_mean_pct"] is not None
                and metrics["candidate_mean_pct"] > 0
            ),
            "gain_vs_admit_all_positive": (
                gain is not None and gain > 0
            ),
            "trade_mean_positive": (
                metrics["trade_mean_pct"] is not None
                and metrics["trade_mean_pct"] > 0
            ),
            "severe_loss": (
                metrics[
                    "trade_severe_loss_rate_le_minus2"
                ] is not None
                and metrics[
                    "trade_severe_loss_rate_le_minus2"
                ] <= MAX_CAL_SEVERE
            ),
        }
        row = {
            "fraction": float(fraction),
            "probability_threshold": threshold,
            "require_positive_predicted_value": True,
            "metrics": metrics,
            "candidate_mean_gain_vs_admit_all_pct": gain,
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
        calibration_grid.append(row)
        if row["passes"]:
            passing.append(row)

    chosen = None
    if passing:
        passing.sort(
            key=lambda row: (
                -(
                    row["metrics"]["candidate_mean_pct"]
                    or -999.0
                ),
                -row["metrics"]["admissions"],
                row["fraction"],
            )
        )
        chosen = passing[0]

    test_diag = None
    baseline_test = None
    test_metrics = None
    test_checks = None
    if chosen is not None:
        test_pred = subset_predictions(
            predicted,
            TEST_DAYS,
        )
        test_policy = subset_policy(
            policy,
            TEST_DAYS,
        )
        test_diag = prediction_diagnostics(test_pred)
        baseline_test = decision_metrics(
            test_policy,
            test_pred,
            probability_threshold=None,
            require_positive_value=False,
        )
        test_metrics = decision_metrics(
            test_policy,
            test_pred,
            probability_threshold=float(
                chosen["probability_threshold"]
            ),
            require_positive_value=True,
        )
        test_checks = {
            "min_admissions": (
                test_metrics["admissions"]
                >= MIN_TEST_ADMISSIONS
            ),
            "resolution": (
                test_metrics["resolution_or_cash_rate"]
                >= MIN_TEST_RESOLUTION
            ),
            "candidate_mean_positive": (
                test_metrics["candidate_mean_pct"]
                is not None
                and test_metrics["candidate_mean_pct"] > 0
            ),
            "day_balanced_mean_positive": (
                test_metrics[
                    "day_balanced_candidate_mean_pct"
                ] is not None
                and test_metrics[
                    "day_balanced_candidate_mean_pct"
                ] > 0
            ),
            "positive_days": (
                test_metrics[
                    "positive_candidate_mean_days"
                ] >= MIN_TEST_POSITIVE_DAYS
            ),
            "trade_positive_rate": (
                test_metrics[
                    "trade_positive_rate"
                ] is not None
                and test_metrics[
                    "trade_positive_rate"
                ] >= MIN_TEST_POSITIVE_RATE
            ),
            "severe_loss": (
                test_metrics[
                    "trade_severe_loss_rate_le_minus2"
                ] is not None
                and test_metrics[
                    "trade_severe_loss_rate_le_minus2"
                ] <= MAX_TEST_SEVERE
            ),
            "gain_vs_admit_all": (
                baseline_test[
                    "candidate_mean_pct"
                ] is not None
                and test_metrics[
                    "candidate_mean_pct"
                ] is not None
                and test_metrics[
                    "candidate_mean_pct"
                ] > baseline_test[
                    "candidate_mean_pct"
                ]
            ),
        }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "fit_days": list(FIT_DAYS),
        "calibration_day": CAL_DAY,
        "development_test_days": list(TEST_DAYS),
        "fixed_downstream_policy": {
            "entry": (
                "first causal 2% pullback within "
                "10 clock minutes after HOT"
            ),
            "stop_loss_pct": FIXED_STOP_LOSS_PCT,
            "trailing_drawdown_pct": FIXED_TRAIL_PCT,
            "max_hold_minutes": (
                FIXED_MAX_HOLD_MINUTES
            ),
            "missing_deadline": "unresolved",
        },
        "population": (
            "all scored full-HOT candidates on already-opened "
            "dates; no candidate-probability prefilter"
        ),
        "features": list(columns),
        "fit_report": fit_report,
        "calibration_prediction_diagnostics": cal_diag,
        "calibration_admit_all_baseline": baseline_cal,
        "calibration_grid": calibration_grid,
        "calibration_gate_pass": chosen is not None,
        "chosen_admission": chosen,
        "development_prediction_diagnostics": test_diag,
        "development_admit_all_baseline": baseline_test,
        "development_metrics": test_metrics,
        "development_checks": test_checks,
        "development_gate_pass": bool(
            test_checks
            and all(test_checks.values())
        ),
        "interpretation": (
            "The admission model sees only pre-HOT candidate "
            "features plus the causal post-HOT price path through "
            "the first 2% pullback. Labels are realized returns "
            "from one fixed downstream policy, never future-best "
            "exits. Rejected and no-pullback candidates are cash; "
            "admitted missing-deadline outcomes remain unresolved."
        ),
        "next_boundary": (
            "learn recurrent ENTER/WAIT then HOLD/EXIT around "
            "the admitted pullback state"
            if test_checks
            and all(test_checks.values())
            else (
                "add causal completed-minute state features or "
                "redesign the pullback trigger; do not tune on "
                "June15-19"
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
    predicted.to_parquet(
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
