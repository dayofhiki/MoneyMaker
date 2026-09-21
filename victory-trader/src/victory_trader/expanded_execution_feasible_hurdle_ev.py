from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .expanded_anchor_hurdle_ev import (
    POLICIES as V24_POLICIES,
    _metrics,
    calibration_diagnostics,
    hurdle_diagnostics,
    load_fold,
    magnitude_diagnostics,
    probability_diagnostics,
    score_anchors,
    select_policies,
    train_magnitude_model,
    train_probability_model,
)
from .expanded_episode_viability_rank import _first_eligible_rows
from .state_multi_source_value import _coverage_row
from .state_rank_turn import day_cluster_bootstrap


ORDER_UNIT_DOLLARS = 1_000.0
MAX_ORDER_SHARE_5M = 0.01
MIN_DOLLAR_VOLUME_5M = ORDER_UNIT_DOLLARS / MAX_ORDER_SHARE_5M
MIN_TRANSACTIONS_5M = 50.0
MIN_ACTIVE_MINUTE_FRACTION_15M = 0.80
BOOTSTRAP_SAMPLES = 10_000
POLICY_MAP = {
    "earliest_eligible_15m_cap1": "feasible_earliest_15m_cap1",
    "probability_half_15m_cap1": "feasible_probability_half_15m_cap1",
    "hurdle_ev_15m_cap1": "feasible_hurdle_ev_15m_cap1",
}
POLICIES = tuple(POLICY_MAP[name] for name in V24_POLICIES)
PRIMARY = "feasible_hurdle_ev_15m_cap1"


def execution_feasible_mask(frame: pd.DataFrame) -> pd.Series:
    dollar_volume = pd.to_numeric(
        frame["dollar_volume_5m"], errors="coerce"
    )
    transactions = pd.to_numeric(
        frame["transactions_5m"], errors="coerce"
    )
    active = pd.to_numeric(
        frame["active_minute_fraction_15m"], errors="coerce"
    )
    return (
        dollar_volume.ge(MIN_DOLLAR_VOLUME_5M)
        & transactions.ge(MIN_TRANSACTIONS_5M)
        & active.ge(MIN_ACTIVE_MINUTE_FRACTION_15M)
    )


def execution_feasible_anchors(frame: pd.DataFrame) -> pd.DataFrame:
    anchors = _first_eligible_rows(frame).copy()
    return anchors.loc[execution_feasible_mask(anchors)].copy()


def _rename_policies(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "policy" in result.columns:
        result["policy"] = result["policy"].replace(POLICY_MAP)
    return result


def run_fold(dataset_paths: dict[str, Path], evaluation_month: str):
    fit, calibration, evaluation, provenance = load_fold(
        dataset_paths, evaluation_month
    )

    fit_before = len(fit)
    calibration_before = len(calibration)
    evaluation_all_anchors = _first_eligible_rows(evaluation)
    evaluation_before = len(evaluation_all_anchors)

    fit = fit.loc[execution_feasible_mask(fit)].copy()
    calibration = calibration.loc[
        execution_feasible_mask(calibration)
    ].copy()
    evaluation_anchors = execution_feasible_anchors(evaluation)

    provenance = provenance.copy()
    provenance["fit_anchors_before_gate"] = fit_before
    provenance["fit_anchors_after_gate"] = len(fit)
    provenance["calibration_anchors_before_gate"] = calibration_before
    provenance["calibration_anchors_after_gate"] = len(calibration)
    provenance["evaluation_anchors_before_gate"] = evaluation_before
    provenance["evaluation_anchors_after_gate"] = len(evaluation_anchors)

    print(
        "execution-feasible fold rows",
        {
            "fit": (fit_before, len(fit)),
            "calibration": (calibration_before, len(calibration)),
            "evaluation": (evaluation_before, len(evaluation_anchors)),
        },
        flush=True,
    )

    probability_model = train_probability_model(fit, calibration)
    win_model = train_magnitude_model(
        fit, calibration, positive=True
    )
    loss_model = train_magnitude_model(
        fit, calibration, positive=False
    )

    scored = score_anchors(
        evaluation_anchors,
        probability_model,
        win_model,
        loss_model,
    )
    trades, attempts, paths = select_policies(scored)
    trades = _rename_policies(trades)
    attempts = _rename_policies(attempts)
    paths = _rename_policies(paths)

    if not trades.empty:
        trades["month"] = evaluation_month
    if not attempts.empty:
        attempts["month"] = evaluation_month
    paths["month"] = evaluation_month

    diagnostics = pd.DataFrame(
        [
            {
                **probability_diagnostics(
                    scored, month=evaluation_month
                ),
                **magnitude_diagnostics(
                    scored,
                    month=evaluation_month,
                    positive=True,
                ),
                **magnitude_diagnostics(
                    scored,
                    month=evaluation_month,
                    positive=False,
                ),
                **hurdle_diagnostics(
                    scored, month=evaluation_month
                ),
                **calibration_diagnostics(
                    calibration,
                    win_model,
                    loss_model,
                    month=evaluation_month,
                ),
                "execution_feasible_rate": (
                    len(evaluation_anchors) / evaluation_before
                    if evaluation_before
                    else np.nan
                ),
                "min_dollar_volume_5m": MIN_DOLLAR_VOLUME_5M,
                "min_transactions_5m": MIN_TRANSACTIONS_5M,
                "min_active_minute_fraction_15m":
                    MIN_ACTIVE_MINUTE_FRACTION_15M,
            }
        ]
    )

    details = pd.DataFrame(
        [
            _metrics(
                trades,
                month=evaluation_month,
                policy=policy,
            )
            for policy in POLICIES
        ]
    )
    coverage = pd.DataFrame(
        [_coverage_row(evaluation_anchors, evaluation_month)]
    )
    bootstrap = day_cluster_bootstrap(
        trades,
        policy=PRIMARY,
        samples=BOOTSTRAP_SAMPLES,
    )
    comparisons = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "primary": PRIMARY,
                "comparator": "feasible_earliest_15m_cap1",
            },
            {
                "month": evaluation_month,
                "primary": PRIMARY,
                "comparator":
                    "feasible_probability_half_15m_cap1",
            },
        ]
    )
    return (
        details,
        trades,
        attempts,
        diagnostics,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    )


def render_report(
    details,
    attempts,
    diagnostics,
    coverage,
    provenance,
    bootstrap,
):
    reason_summary = (
        attempts.groupby(["policy", "evaluation_reason"])
        .size()
        .rename("attempts")
        .reset_index()
        if not attempts.empty
        else pd.DataFrame()
    )
    return "\n".join(
        [
            "=== MoneyMaker Execution-Feasible Hurdle EV v2.6 ===",
            (
                "candidate_universe=first eligible anchor must pass "
                "fixed execution-feasibility gate"
            ),
            (
                f"gate=dollar_volume_5m>={MIN_DOLLAR_VOLUME_5M:.0f}; "
                f"transactions_5m>={MIN_TRANSACTIONS_5M:.0f}; "
                "active_minute_fraction_15m>="
                f"{MIN_ACTIVE_MINUTE_FRACTION_15M:.2f}"
            ),
            (
                "economic rationale=$1000 order <=1% of observed 5m "
                "dollar volume plus continuity/activity floor"
            ),
            (
                "model=exact v2.4 hurdle decomposition retrained only "
                "inside gated universe"
            ),
            "primary=feasible hurdle EV > 0",
            "no later-anchor fallthrough",
            "evaluation=strict past-only; April 2026+ sealed",
            "",
            "=== Date and gate provenance ===",
            provenance.to_string(index=False),
            "",
            "=== Diagnostics ===",
            diagnostics.to_string(index=False),
            "",
            "=== Policies ===",
            details.to_string(index=False),
            "",
            "=== Attempt paths ===",
            reason_summary.to_string(index=False),
            "",
            "=== External-data coverage inside gated universe ===",
            coverage.to_string(index=False),
            "",
            "=== Primary pooled day bootstrap ===",
            str(bootstrap),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--dataset must be LABEL=PATH"
        )
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "victory_trader.expanded_execution_feasible_hurdle_ev"
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
    )
    parser.add_argument("--evaluation-month", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--attempts-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--comparisons-csv", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--provenance-csv", type=Path, required=True)
    args = parser.parse_args()

    (
        details,
        trades,
        attempts,
        diag,
        comparisons,
        coverage,
        provenance,
        bootstrap,
    ) = run_fold(dict(args.dataset), args.evaluation_month)

    report = render_report(
        details,
        attempts,
        diag,
        coverage,
        provenance,
        bootstrap,
    )
    print(report, flush=True)

    for path in (
        args.report,
        args.details_csv,
        args.trades_csv,
        args.attempts_csv,
        args.diagnostics_csv,
        args.comparisons_csv,
        args.coverage_csv,
        args.provenance_csv,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    attempts.to_csv(args.attempts_csv, index=False)
    diag.to_csv(args.diagnostics_csv, index=False)
    comparisons.to_csv(args.comparisons_csv, index=False)
    coverage.to_csv(args.coverage_csv, index=False)
    provenance.to_csv(args.provenance_csv, index=False)
    pd.DataFrame([bootstrap]).to_csv(
        args.report.with_name(
            args.report.stem + "-bootstrap.csv"
        ),
        index=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
