"""Request278: risk-first causal pullback admission.

Request277 showed that strictly lagged completed-minute context materially
improves severe-loss discrimination and return ranking, but not direct
positive-trade classification. This experiment uses those two strengths in a
small preregistered two-stage policy:

1. veto pullback states with high predicted severe-loss risk;
2. among survivors, admit only states with high predicted return rank.

May5-8 are used only through leave-one-day-out predictions for policy
selection. May11-20 is evaluated only if the cross-fitted economic gate
passes. June15-19 remains unopened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .causal_minute_controller_rebuild import causal_execution_scan
from .full_hot_fixed_policy_value import attach_fixed_value
from .joint_pullback_admission import (
    FIXED_MAX_HOLD_MINUTES,
    FIXED_STOP_LOSS_PCT,
    FIXED_TRAIL_PCT,
    build_pullback_state_rows,
    feature_columns,
    feature_x,
)
from .lagged_minute_context import (
    CROSSFIT_DAYS,
    fit_triplet,
)
from .pullback_trailing_exit import (
    apply_exit_rule,
    build_pullback_episodes,
)
from .rich_post_hot_state import (
    CURRENT_MINUTE_FEATURES,
    attach_minute_features,
    score_and_select_candidates,
)

REQUEST_ID = 278
TEST_DAYS = (
    "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14",
    "2026-05-15", "2026-05-18", "2026-05-19", "2026-05-20",
)
RISK_REJECT_FRACTIONS = (0.30, 0.50)
VALUE_RETAIN_FRACTIONS = (0.30, 0.50)

MIN_OOF_ADMISSIONS = 25
MIN_OOF_RESOLUTION = 0.95
MIN_OOF_POSITIVE_DAYS = 3
MIN_OOF_TRADE_POSITIVE_RATE = 0.40
MAX_OOF_SEVERE_RATE = 0.20
MIN_OOF_GAIN_VS_RAW = 0.10

MIN_TEST_ADMISSIONS = 20
MIN_TEST_RESOLUTION = 0.95
MIN_TEST_POSITIVE_DAYS = 5
MIN_TEST_TRADE_POSITIVE_RATE = 0.40
MAX_TEST_SEVERE_RATE = 0.15
MIN_TEST_GAIN_VS_RAW = 0.10


def prepare(
    first_hot: pd.DataFrame,
    raw_scan: pd.DataFrame,
    days: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    causal_scan = causal_execution_scan(raw_scan)
    frame = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    frame = attach_fixed_value(frame, causal_scan)
    scored, _, _ = score_and_select_candidates(frame)
    population = scored.loc[
        scored.trading_day.astype(str).isin(days)
    ].copy()

    states = build_pullback_state_rows(
        population,
        causal_scan,
    )
    episodes = build_pullback_episodes(
        population,
        causal_scan,
    )
    policy = apply_exit_rule(
        episodes,
        stop_loss_pct=FIXED_STOP_LOSS_PCT,
        trail_pct=FIXED_TRAIL_PCT,
        max_hold_minutes=FIXED_MAX_HOLD_MINUTES,
        policy_name="fixed_downstream_teacher",
    )
    states = states.merge(
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
    if states.resolved.isna().any():
        raise ValueError(
            "Request278 state/outcome merge lost pullback rows"
        )

    baseline_columns = feature_columns(states)
    states["state_t"] = pd.to_numeric(
        states.entry_t,
        errors="raise",
    ).astype("int64")
    states = attach_minute_features(
        states,
        raw_scan,
    )
    minute_columns = tuple(
        name
        for name in CURRENT_MINUTE_FEATURES
        if name in states
        and pd.to_numeric(
            states[name],
            errors="coerce",
        ).notna().any()
    )
    enriched_columns = tuple(
        dict.fromkeys([
            *baseline_columns,
            *minute_columns,
        ])
    )
    return states, policy, enriched_columns


def predict_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
) -> pd.DataFrame:
    reg, _, severe = fit_triplet(
        train,
        columns,
        seed,
    )
    out = test.copy()
    x = feature_x(out, columns)
    out["predicted_return_pct"] = reg.predict(x)
    out["severe_probability"] = (
        severe.predict_proba(x)[:, 1]
    )
    return out


def admission_mask(
    predictions: pd.DataFrame,
    *,
    risk_threshold: float,
    value_threshold: float,
) -> pd.Series:
    risk = pd.to_numeric(
        predictions.severe_probability,
        errors="coerce",
    )
    value = pd.to_numeric(
        predictions.predicted_return_pct,
        errors="coerce",
    )
    return (
        risk.lt(float(risk_threshold))
        & value.ge(float(value_threshold))
    )


def decision_metrics(
    policy: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    risk_threshold: float | None,
    value_threshold: float | None,
) -> dict:
    decisions = policy.copy()

    admission = predictions.loc[:, [
        "trading_day",
        "ticker",
        "hot_t",
        "predicted_return_pct",
        "severe_probability",
    ]].copy()
    if (
        risk_threshold is None
        or value_threshold is None
    ):
        admission["admitted"] = True
    else:
        admission["admitted"] = admission_mask(
            admission,
            risk_threshold=float(risk_threshold),
            value_threshold=float(value_threshold),
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
        decisions.loc[
            trade_mask,
            "trade_return_pct",
        ],
        errors="coerce",
    )
    decisions.loc[
        ~decisions.decision_resolved,
        "decision_return_pct",
    ] = np.nan

    resolved = decisions.loc[
        decisions.decision_resolved
    ].copy()
    trades = decisions.loc[
        trade_mask
    ].copy()
    daily = resolved.groupby(
        resolved.trading_day.astype(str),
        sort=True,
    ).decision_return_pct.mean()

    trade_returns = pd.to_numeric(
        trades.trade_return_pct,
        errors="coerce",
    )

    return {
        "candidate_episodes": int(len(decisions)),
        "pullback_triggers": int(
            decisions.entered.astype(bool).sum()
        ),
        "admissions": int(
            decisions.admitted.sum()
        ),
        "resolution_or_cash_rate": (
            float(
                decisions.decision_resolved.mean()
            )
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
        "trade_count": int(
            trade_returns.notna().sum()
        ),
        "trade_mean_pct": (
            float(trade_returns.mean())
            if trade_returns.notna().any()
            else None
        ),
        "trade_positive_rate": (
            float(
                trade_returns.gt(0).mean()
            )
            if trade_returns.notna().any()
            else None
        ),
        "trade_severe_loss_rate_le_minus2": (
            float(
                trade_returns.le(-2.0).mean()
            )
            if trade_returns.notna().any()
            else None
        ),
        "by_day": {
            str(day): {
                "candidates": int(len(part)),
                "admissions": int(
                    part.admitted.sum()
                ),
                "candidate_mean_pct": (
                    float(
                        pd.to_numeric(
                            part.loc[
                                part.decision_resolved,
                                "decision_return_pct",
                            ],
                            errors="coerce",
                        ).mean()
                    )
                ),
            }
            for day, part in decisions.groupby(
                decisions.trading_day.astype(str),
                sort=True,
            )
        },
    }


def gain_vs(
    candidate: dict,
    baseline: dict,
) -> float | None:
    left = candidate.get("candidate_mean_pct")
    right = baseline.get("candidate_mean_pct")
    if left is None or right is None:
        return None
    return float(left - right)


def oof_checks(
    metrics: dict,
    gain: float | None,
) -> dict:
    return {
        "min_admissions": (
            metrics["admissions"]
            >= MIN_OOF_ADMISSIONS
        ),
        "resolution": (
            metrics["resolution_or_cash_rate"]
            >= MIN_OOF_RESOLUTION
        ),
        "candidate_mean_positive": (
            metrics["candidate_mean_pct"] is not None
            and metrics["candidate_mean_pct"] > 0
        ),
        "day_balanced_mean_positive": (
            metrics[
                "day_balanced_candidate_mean_pct"
            ] is not None
            and metrics[
                "day_balanced_candidate_mean_pct"
            ] > 0
        ),
        "positive_days": (
            metrics["positive_candidate_mean_days"]
            >= MIN_OOF_POSITIVE_DAYS
        ),
        "trade_mean_positive": (
            metrics["trade_mean_pct"] is not None
            and metrics["trade_mean_pct"] > 0
        ),
        "trade_positive_rate": (
            metrics["trade_positive_rate"] is not None
            and metrics["trade_positive_rate"]
            >= MIN_OOF_TRADE_POSITIVE_RATE
        ),
        "severe_loss": (
            metrics[
                "trade_severe_loss_rate_le_minus2"
            ] is not None
            and metrics[
                "trade_severe_loss_rate_le_minus2"
            ] <= MAX_OOF_SEVERE_RATE
        ),
        "gain_vs_raw": (
            gain is not None
            and gain >= MIN_OOF_GAIN_VS_RAW
        ),
    }


def test_checks(
    metrics: dict,
    gain: float | None,
) -> dict:
    return {
        "min_admissions": (
            metrics["admissions"]
            >= MIN_TEST_ADMISSIONS
        ),
        "resolution": (
            metrics["resolution_or_cash_rate"]
            >= MIN_TEST_RESOLUTION
        ),
        "candidate_mean_positive": (
            metrics["candidate_mean_pct"] is not None
            and metrics["candidate_mean_pct"] > 0
        ),
        "day_balanced_mean_positive": (
            metrics[
                "day_balanced_candidate_mean_pct"
            ] is not None
            and metrics[
                "day_balanced_candidate_mean_pct"
            ] > 0
        ),
        "positive_days": (
            metrics["positive_candidate_mean_days"]
            >= MIN_TEST_POSITIVE_DAYS
        ),
        "trade_mean_positive": (
            metrics["trade_mean_pct"] is not None
            and metrics["trade_mean_pct"] > 0
        ),
        "trade_positive_rate": (
            metrics["trade_positive_rate"] is not None
            and metrics["trade_positive_rate"]
            >= MIN_TEST_TRADE_POSITIVE_RATE
        ),
        "severe_loss": (
            metrics[
                "trade_severe_loss_rate_le_minus2"
            ] is not None
            and metrics[
                "trade_severe_loss_rate_le_minus2"
            ] <= MAX_TEST_SEVERE_RATE
        ),
        "gain_vs_raw": (
            gain is not None
            and gain >= MIN_TEST_GAIN_VS_RAW
        ),
    }


def evaluate(
    first_hot_path: Path,
    scan_path: Path,
    output_path: Path,
    rows_output: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    raw_scan = pd.read_parquet(scan_path)

    dev_states, dev_policy, columns = prepare(
        first_hot,
        raw_scan,
        CROSSFIT_DAYS,
    )

    oof_parts = []
    for fold_index, held_day in enumerate(
        CROSSFIT_DAYS
    ):
        train = dev_states.loc[
            ~dev_states.trading_day.astype(str).eq(
                held_day
            )
        ].copy()
        held = dev_states.loc[
            dev_states.trading_day.astype(str).eq(
                held_day
            )
        ].copy()
        prediction = predict_models(
            train,
            held,
            columns,
            seed=20261457 + fold_index * 10,
        )
        prediction["held_out_day"] = held_day
        oof_parts.append(prediction)

    oof = pd.concat(
        oof_parts,
        ignore_index=True,
    )
    raw_oof = decision_metrics(
        dev_policy,
        oof,
        risk_threshold=None,
        value_threshold=None,
    )

    grid = []
    passing = []
    for risk_reject in RISK_REJECT_FRACTIONS:
        risk_threshold = float(
            np.quantile(
                pd.to_numeric(
                    oof.severe_probability,
                    errors="coerce",
                ).dropna().to_numpy(float),
                1.0 - float(risk_reject),
            )
        )
        for value_retain in VALUE_RETAIN_FRACTIONS:
            value_threshold = float(
                np.quantile(
                    pd.to_numeric(
                        oof.predicted_return_pct,
                        errors="coerce",
                    ).dropna().to_numpy(float),
                    1.0 - float(value_retain),
                )
            )
            metrics = decision_metrics(
                dev_policy,
                oof,
                risk_threshold=risk_threshold,
                value_threshold=value_threshold,
            )
            gain = gain_vs(
                metrics,
                raw_oof,
            )
            checks = oof_checks(
                metrics,
                gain,
            )
            row = {
                "risk_reject_fraction": float(
                    risk_reject
                ),
                "value_retain_fraction": float(
                    value_retain
                ),
                "oof_risk_threshold": (
                    risk_threshold
                ),
                "oof_value_threshold": (
                    value_threshold
                ),
                "metrics": metrics,
                "candidate_mean_gain_vs_raw_pct": (
                    gain
                ),
                "checks": checks,
                "passes": bool(
                    all(checks.values())
                ),
            }
            grid.append(row)
            if row["passes"]:
                passing.append(row)

    chosen = None
    if passing:
        passing.sort(
            key=lambda row: (
                -(
                    row["metrics"][
                        "candidate_mean_pct"
                    ]
                    or -999.0
                ),
                row["metrics"][
                    "trade_severe_loss_rate_le_minus2"
                ],
                -row["metrics"]["admissions"],
            )
        )
        chosen = passing[0]

    final_thresholds = None
    raw_test = None
    test_metrics = None
    test_gain = None
    dev_test_checks = None
    dev_test_gate_pass = False

    if chosen is not None:
        reg, _, severe = fit_triplet(
            dev_states,
            columns,
            20261507,
        )
        dev_x = feature_x(
            dev_states,
            columns,
        )
        final_risk_scores = (
            severe.predict_proba(dev_x)[:, 1]
        )
        final_value_scores = reg.predict(dev_x)

        final_risk_threshold = float(
            np.quantile(
                final_risk_scores,
                1.0
                - float(
                    chosen[
                        "risk_reject_fraction"
                    ]
                ),
            )
        )
        final_value_threshold = float(
            np.quantile(
                final_value_scores,
                1.0
                - float(
                    chosen[
                        "value_retain_fraction"
                    ]
                ),
            )
        )
        final_thresholds = {
            "risk_threshold": (
                final_risk_threshold
            ),
            "value_threshold": (
                final_value_threshold
            ),
            "source": (
                "quantiles of final May5-8 model "
                "scores using OOF-selected fractions"
            ),
        }

        test_states, test_policy, test_columns = prepare(
            first_hot,
            raw_scan,
            TEST_DAYS,
        )
        if tuple(test_columns) != tuple(columns):
            missing = [
                name
                for name in columns
                if name not in test_states
            ]
            if missing:
                raise ValueError(
                    "Request278 test feature mismatch: "
                    f"{missing}"
                )
        test_x = feature_x(
            test_states,
            columns,
        )
        test_pred = test_states.copy()
        test_pred["predicted_return_pct"] = (
            reg.predict(test_x)
        )
        test_pred["severe_probability"] = (
            severe.predict_proba(test_x)[:, 1]
        )

        raw_test = decision_metrics(
            test_policy,
            test_pred,
            risk_threshold=None,
            value_threshold=None,
        )
        test_metrics = decision_metrics(
            test_policy,
            test_pred,
            risk_threshold=final_risk_threshold,
            value_threshold=final_value_threshold,
        )
        test_gain = gain_vs(
            test_metrics,
            raw_test,
        )
        dev_test_checks = test_checks(
            test_metrics,
            test_gain,
        )
        dev_test_gate_pass = bool(
            all(dev_test_checks.values())
        )

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "promotion_eligible": False,
        "opens_new_dates": False,
        "crossfit_days": list(
            CROSSFIT_DAYS
        ),
        "development_test_days": list(
            TEST_DAYS
        ),
        "policy_family": (
            "risk veto then predicted-return rank admission"
        ),
        "risk_reject_fractions": list(
            RISK_REJECT_FRACTIONS
        ),
        "value_retain_fractions": list(
            VALUE_RETAIN_FRACTIONS
        ),
        "fixed_downstream_policy": {
            "entry": "first causal 2% pullback",
            "hard_stop_loss_pct": (
                FIXED_STOP_LOSS_PCT
            ),
            "trailing_drawdown_pct": (
                FIXED_TRAIL_PCT
            ),
            "max_hold_minutes": (
                FIXED_MAX_HOLD_MINUTES
            ),
            "missing_deadline": "unresolved",
        },
        "oof_raw_pullback_baseline": (
            raw_oof
        ),
        "oof_grid": grid,
        "oof_gate_pass": chosen is not None,
        "chosen_oof_policy": chosen,
        "final_thresholds": final_thresholds,
        "development_raw_pullback_baseline": (
            raw_test
        ),
        "development_metrics": (
            test_metrics
        ),
        "development_gain_vs_raw_pct": (
            test_gain
        ),
        "development_checks": (
            dev_test_checks
        ),
        "development_gate_pass": (
            dev_test_gate_pass
        ),
        "interpretation": (
            "The policy is selected only from leave-one-day-out "
            "May5-8 predictions. May11-20 is touched only after "
            "a positive OOF economic gate. Completed-minute "
            "features are strictly lagged as audited in "
            "Request277. June15-19 remains unopened."
        ),
        "next_boundary": (
            "restore recurrent HOLD/EXIT around the risk-first admitted set"
            if dev_test_gate_pass
            else (
                "redesign candidate-level state and regime context; "
                "do not open June15-19"
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
