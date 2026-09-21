from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .expanded_opportunity_shared_q import (
    predict_opportunity,
    train_opportunity_model,
)
from .expanded_path_aware_continuation import (
    KEYS,
    MINUTE_MS,
    PATH_FEATURES,
    ContinuationModel,
    build_continuation_rows,
    continuation_feature_frame,
    day_bootstrap,
    diagnostics,
)
from .expanded_supply_hurdle_ev import load_fold


PATH_TRANSITION_LAGS = (1, 2, 3, 5, 8, 13)
RANDOM_SEED = 20261043


def _groupers(frame: pd.DataFrame) -> list[pd.Series]:
    return [frame[key].astype(str) for key in KEYS]


def _exact_lag_mask(frame: pd.DataFrame, lag_minutes: int) -> pd.Series:
    timestamp = pd.to_numeric(frame["t"], errors="coerce")
    lagged_t = timestamp.groupby(_groupers(frame), sort=False).shift(lag_minutes)
    return (timestamp - lagged_t).eq(lag_minutes * MINUTE_MS)


def path_transition_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact causal changes in the frozen v3.5 position-path state.

    Row shifts are admitted only when the timestamp difference equals the
    preregistered minute lag, so halts and missing bars are never compressed.
    """
    result = pd.DataFrame(index=frame.index)
    groupers = _groupers(frame)
    timestamp = pd.to_numeric(frame["t"], errors="coerce")

    for lag in PATH_TRANSITION_LAGS:
        lagged_t = timestamp.groupby(groupers, sort=False).shift(lag)
        exact = (timestamp - lagged_t).eq(lag * MINUTE_MS)
        for column in PATH_FEATURES:
            current = pd.to_numeric(frame[column], errors="coerce")
            lagged = current.groupby(groupers, sort=False).shift(lag)
            result[f"path_transition_{lag}m_{column}"] = (
                current - lagged
            ).where(exact)
    return result


def transition_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = continuation_feature_frame(frame).copy()
    transitions = path_transition_features(frame)
    for column in transitions:
        result[column] = transitions[column]
    return result


def transition_lag_coverage(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    rows = []
    total = int(len(frame))
    for lag in PATH_TRANSITION_LAGS:
        exact = _exact_lag_mask(frame, lag)
        exact_rows = int(exact.sum())
        rows.append(
            {
                "month": month,
                "lag_minutes": lag,
                "candidate_rows": total,
                "exact_lag_rows": exact_rows,
                "exact_lag_coverage": float(exact_rows / total)
                if total
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def train_transition_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> ContinuationModel:
    fit = fit.loc[fit["hold_advantage_pct"].notna()].copy()
    calibration = calibration.loc[
        calibration["hold_advantage_pct"].notna()
    ].copy()
    y_fit = fit["hold_advantage_pct"].gt(0).astype(int)
    y_cal = calibration["hold_advantage_pct"].gt(0).astype(int)
    if y_fit.nunique() < 2 or y_cal.nunique() < 2:
        raise ValueError("transition continuation requires both classes")

    x_fit = transition_feature_frame(fit)
    columns = [column for column in x_fit if x_fit[column].notna().any()]
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=RANDOM_SEED,
    )
    model.fit(x_fit.loc[:, columns], y_fit)

    raw = model.predict_proba(
        transition_feature_frame(calibration).reindex(columns=columns)
    )[:, 1]
    raw = np.clip(raw, 1e-6, 1 - 1e-6)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    platt.fit(np.log(raw / (1 - raw)).reshape(-1, 1), y_cal)
    return ContinuationModel(model, tuple(columns), platt)


def predict_transition_continuation(
    frame: pd.DataFrame,
    fitted: ContinuationModel,
) -> np.ndarray:
    raw = fitted.model.predict_proba(
        transition_feature_frame(frame).reindex(
            columns=fitted.feature_columns
        )
    )[:, 1]
    raw = np.clip(raw, 1e-6, 1 - 1e-6)
    return fitted.platt.predict_proba(
        np.log(raw / (1 - raw)).reshape(-1, 1)
    )[:, 1]


def _month_of(frame: pd.DataFrame) -> pd.Series:
    return frame["trading_day"].astype(str).str.slice(0, 7)


def run_fold(
    anchor_paths: dict[str, Path],
    state_paths: dict[str, Path],
    evaluation_month: str,
):
    fit, calibration, evaluation, provenance = load_fold(
        anchor_paths,
        evaluation_month,
    )
    opportunity = train_opportunity_model(fit, calibration)

    partitions = []
    for name, frame in (
        ("fit", fit),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        item = frame.copy()
        item["_partition"] = name
        item["opportunity_probability"] = predict_opportunity(
            item,
            opportunity,
        )
        partitions.append(item)
    anchors = pd.concat(partitions, ignore_index=True)

    sequence_parts = []
    for month, group in anchors.groupby(_month_of(anchors), sort=True):
        if month not in state_paths:
            raise ValueError(f"missing state panel for {month}")
        sequence_parts.append(
            build_continuation_rows(
                pd.read_parquet(state_paths[month]),
                group,
            )
        )
    sequence = pd.concat(sequence_parts, ignore_index=True)
    fit_rows = sequence.loc[
        sequence["_partition"].eq("fit")
    ].reset_index(drop=True)
    cal_rows = sequence.loc[
        sequence["_partition"].eq("calibration")
    ].reset_index(drop=True)
    eval_rows = sequence.loc[
        sequence["_partition"].eq("evaluation")
    ].reset_index(drop=True)

    model = train_transition_model(fit_rows, cal_rows)
    eval_rows["hold_probability"] = predict_transition_continuation(
        eval_rows,
        model,
    )

    diag = diagnostics(eval_rows, evaluation_month)
    bootstrap = day_bootstrap(eval_rows, evaluation_month)
    lag_coverage = transition_lag_coverage(eval_rows, evaluation_month)

    primary = diag.loc[
        diag["pool"].eq("opportunity_gate")
        & diag["bucket"].eq("overall")
    ].iloc[0]
    boot = bootstrap.iloc[0]
    success = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "auc_above_half": bool(
                    primary.get("auc", np.nan) > 0.5
                ),
                "policy_mean_positive": bool(
                    primary.get(
                        "policy_incremental_mean_pct",
                        np.nan,
                    )
                    > 0
                ),
                "day_mean_positive": bool(
                    primary.get(
                        "day_balanced_policy_mean_pct",
                        np.nan,
                    )
                    > 0
                ),
                "coverage_at_least_90pct": bool(
                    primary.get("evaluation_coverage", 0) >= 0.9
                ),
                "bootstrap_low_positive": bool(
                    boot.get("ci_low_pct", np.nan) > 0
                ),
            }
        ]
    )
    success["all_frozen_checks_pass"] = success.iloc[:, 1:].all(axis=1)
    return diag, eval_rows, bootstrap, lag_coverage, provenance, success


def render_report(outputs: tuple[pd.DataFrame, ...]) -> str:
    diagnostics_frame, _scored, bootstrap, lag_coverage, provenance, success = (
        outputs
    )
    lines = [
        "=== MoneyMaker Path-Transition Persistence Continuation v3.7 ===",
        "diagnostic only: unchanged v3.5 one-step HOLD-vs-EXIT classification",
        "only change=frozen exact-lag deltas of all ten position-path features",
        "lag grid=1/2/3/5/8/13m; no gap compression; HOLD iff calibrated P>0.5",
        "April 2026+ sealed",
    ]
    for title, frame in (
        ("Diagnostics", diagnostics_frame),
        ("Day bootstrap", bootstrap),
        ("Transition lag coverage", lag_coverage),
        ("Provenance", provenance),
        ("Frozen checks", success),
    ):
        lines.extend(("", f"=== {title} ===", frame.to_string(index=False)))
    return "\n".join(lines)


def _parse(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-month", required=True)
    parser.add_argument("--anchor", action="append", type=_parse, required=True)
    parser.add_argument("--state", action="append", type=_parse, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--scored-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-csv", type=Path, required=True)
    parser.add_argument("--lag-coverage-csv", type=Path, required=True)
    parser.add_argument("--provenance-csv", type=Path, required=True)
    parser.add_argument("--success-csv", type=Path, required=True)
    args = parser.parse_args()

    outputs = run_fold(
        dict(args.anchor),
        dict(args.state),
        args.evaluation_month,
    )
    report = render_report(outputs)
    print(report, flush=True)
    for path, frame in zip(
        (
            args.diagnostics_csv,
            args.scored_csv,
            args.bootstrap_csv,
            args.lag_coverage_csv,
            args.provenance_csv,
            args.success_csv,
        ),
        outputs,
        strict=True,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
