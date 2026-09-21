from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .expanded_opportunity_shared_q import (
    predict_opportunity,
    train_opportunity_model,
)
from .expanded_path_aware_continuation import (
    BOOTSTRAP_SAMPLES,
    KEYS,
    PHASES,
    build_continuation_rows,
    continuation_feature_frame,
)
from .expanded_supply_hurdle_ev import load_fold


RANDOM_SEED = 20261042


@dataclass(frozen=True)
class ExpectedContinuationModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    additive_bias_pct: float
    selected_tail_correction_pct: float
    selected_calibration_days: int
    mean_daily_optimism_pct: float
    optimism_se_pct: float
    fit_clip_low_pct: float
    fit_clip_high_pct: float

    @property
    def hold_enabled(self) -> bool:
        return (
            self.selected_calibration_days >= 5
            and np.isfinite(self.selected_tail_correction_pct)
        )


def _selected_tail_calibration(
    calibration: pd.DataFrame,
    raw_prediction: np.ndarray,
) -> tuple[float, float, int, float, float]:
    actual = pd.to_numeric(
        calibration["hold_advantage_pct"], errors="coerce"
    ).to_numpy(dtype=float)
    raw = np.asarray(raw_prediction, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(raw)
    if not valid.any():
        return 0.0, np.inf, 0, np.nan, np.nan

    bias = float(np.mean(actual[valid] - raw[valid]))
    adjusted = raw + bias
    selected = valid & (adjusted > 0)
    if not selected.any():
        return bias, np.inf, 0, np.nan, np.nan

    selected_frame = pd.DataFrame(
        {
            "day": calibration.loc[selected, "trading_day"].astype(str).to_numpy(),
            "optimism": adjusted[selected] - actual[selected],
        }
    )
    daily = selected_frame.groupby("day", sort=True)["optimism"].mean()
    days = int(len(daily))
    if days < 5:
        return bias, np.inf, days, float(daily.mean()), np.nan

    mean_optimism = float(daily.mean())
    se = float(daily.std(ddof=1) / np.sqrt(days))
    correction = float(max(0.0, mean_optimism + 1.645 * se))
    return bias, correction, days, mean_optimism, se


def train_expected_continuation_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> ExpectedContinuationModel:
    fit = fit.loc[fit["hold_advantage_pct"].notna()].copy()
    calibration = calibration.loc[
        calibration["hold_advantage_pct"].notna()
    ].copy()
    if fit.empty or calibration.empty:
        raise ValueError("expected continuation requires fit and calibration rows")

    x_fit = continuation_feature_frame(fit)
    columns = [column for column in x_fit if x_fit[column].notna().any()]
    y_fit = pd.to_numeric(fit["hold_advantage_pct"], errors="coerce")
    finite_fit = y_fit[np.isfinite(y_fit)]
    if finite_fit.empty:
        raise ValueError("expected continuation fit target is empty")

    clip_low = float(finite_fit.quantile(0.005))
    clip_high = float(finite_fit.quantile(0.995))
    clipped = y_fit.clip(lower=clip_low, upper=clip_high)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=RANDOM_SEED,
    )
    model.fit(x_fit.loc[:, columns], clipped)

    raw_cal = model.predict(
        continuation_feature_frame(calibration).reindex(columns=columns)
    )
    bias, correction, days, mean_optimism, se = _selected_tail_calibration(
        calibration,
        raw_cal,
    )
    return ExpectedContinuationModel(
        model=model,
        feature_columns=tuple(columns),
        additive_bias_pct=bias,
        selected_tail_correction_pct=correction,
        selected_calibration_days=days,
        mean_daily_optimism_pct=mean_optimism,
        optimism_se_pct=se,
        fit_clip_low_pct=clip_low,
        fit_clip_high_pct=clip_high,
    )


def predict_expected_continuation(
    frame: pd.DataFrame,
    fitted: ExpectedContinuationModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = fitted.model.predict(
        continuation_feature_frame(frame).reindex(
            columns=fitted.feature_columns
        )
    )
    bias_adjusted = raw + fitted.additive_bias_pct
    if fitted.hold_enabled:
        adjusted = bias_adjusted - fitted.selected_tail_correction_pct
    else:
        adjusted = np.full(len(frame), np.nan, dtype=float)
    return raw, bias_adjusted, adjusted


def _safe_corr(
    actual: pd.Series,
    prediction: pd.Series,
    *,
    method: str,
) -> float:
    valid = actual.notna() & prediction.notna()
    if int(valid.sum()) < 2:
        return np.nan
    if actual.loc[valid].nunique() < 2 or prediction.loc[valid].nunique() < 2:
        return np.nan
    return float(actual.loc[valid].corr(prediction.loc[valid], method=method))


def _diagnostic_row(
    frame: pd.DataFrame,
    *,
    month: str,
    pool: str,
    bucket: str,
) -> dict[str, object]:
    evaluated = frame.loc[frame["hold_advantage_pct"].notna()].copy()
    if evaluated.empty:
        return {
            "month": month,
            "pool": pool,
            "bucket": bucket,
            "candidate_rows": int(len(frame)),
            "evaluated_rows": 0,
        }

    actual = pd.to_numeric(evaluated["hold_advantage_pct"], errors="coerce")
    raw = pd.to_numeric(evaluated["raw_expected_advantage_pct"], errors="coerce")
    adjusted = pd.to_numeric(
        evaluated["adjusted_expected_advantage_pct"], errors="coerce"
    )
    selected = adjusted.gt(0)
    action_gain = actual.where(selected, 0.0)
    daily = (
        pd.DataFrame(
            {
                "day": evaluated["trading_day"].astype(str),
                "gain": action_gain,
            }
        )
        .groupby("day", sort=True)["gain"]
        .mean()
    )

    return {
        "month": month,
        "pool": pool,
        "bucket": bucket,
        "candidate_rows": int(len(frame)),
        "evaluated_rows": int(len(evaluated)),
        "evaluation_coverage": float(len(evaluated) / len(frame))
        if len(frame)
        else np.nan,
        "raw_prediction_mean_pct": float(raw.mean()),
        "adjusted_prediction_mean_pct": float(adjusted.mean())
        if adjusted.notna().any()
        else np.nan,
        "pearson": _safe_corr(actual, adjusted, method="pearson"),
        "spearman": _safe_corr(actual, adjusted, method="spearman"),
        "selected_hold_rows": int(selected.sum()),
        "selected_hold_rate": float(selected.mean()),
        "always_hold_mean_pct": float(actual.mean()),
        "policy_incremental_mean_pct": float(action_gain.mean()),
        "day_balanced_policy_mean_pct": float(daily.mean()),
        "selected_predicted_mean_pct": float(adjusted.loc[selected].mean())
        if selected.any()
        else np.nan,
        "selected_realized_mean_pct": float(actual.loc[selected].mean())
        if selected.any()
        else np.nan,
    }


def diagnostics(scored: pd.DataFrame, month: str) -> pd.DataFrame:
    scored = scored.reset_index(drop=True)
    pools = {
        "all_anchors": pd.Series(True, index=scored.index),
        "opportunity_gate": pd.to_numeric(
            scored["opportunity_probability"], errors="coerce"
        ).gt(0.5),
    }
    buckets = {
        "overall": pd.Series(True, index=scored.index),
        "held_1_2m": scored["minutes_held"].between(1, 2),
        "held_3_5m": scored["minutes_held"].between(3, 5),
        "held_6_10m": scored["minutes_held"].between(6, 10),
        "held_11_20m": scored["minutes_held"].between(11, 20),
        "held_21_29m": scored["minutes_held"].between(21, 29),
    }
    for phase in PHASES:
        buckets[f"phase_{phase}"] = scored["phase"].astype(str).eq(phase)

    rows = []
    for pool, pool_mask in pools.items():
        for bucket, bucket_mask in buckets.items():
            mask = (pool_mask & bucket_mask).to_numpy(dtype=bool)
            rows.append(
                _diagnostic_row(
                    scored.loc[mask],
                    month=month,
                    pool=pool,
                    bucket=bucket,
                )
            )
    result = pd.DataFrame(rows)
    overall = result.loc[
        result["pool"].eq("all_anchors") & result["bucket"].eq("overall"),
        "candidate_rows",
    ]
    if len(overall) != 1 or int(overall.iloc[0]) != len(scored):
        raise AssertionError("diagnostic denominator differs from scored rows")
    return result


def day_bootstrap(scored: pd.DataFrame, month: str) -> pd.DataFrame:
    selected = scored.loc[
        pd.to_numeric(scored["opportunity_probability"], errors="coerce").gt(0.5)
        & scored["hold_advantage_pct"].notna()
    ].copy()
    adjusted = pd.to_numeric(
        selected["adjusted_expected_advantage_pct"], errors="coerce"
    )
    gain = pd.to_numeric(
        selected["hold_advantage_pct"], errors="coerce"
    ).where(adjusted.gt(0), 0.0)
    daily = (
        pd.DataFrame(
            {
                "day": selected["trading_day"].astype(str),
                "gain": gain,
            }
        )
        .groupby("day", sort=True)["gain"]
        .mean()
        .dropna()
        .to_numpy()
    )
    if not len(daily):
        return pd.DataFrame([{"month": month, "days": 0}])

    rng = np.random.default_rng(RANDOM_SEED)
    means = daily[
        rng.integers(0, len(daily), size=(BOOTSTRAP_SAMPLES, len(daily)))
    ].mean(axis=1)
    return pd.DataFrame(
        [
            {
                "month": month,
                "days": int(len(daily)),
                "day_balanced_mean_pct": float(daily.mean()),
                "ci_low_pct": float(np.quantile(means, 0.025)),
                "ci_high_pct": float(np.quantile(means, 0.975)),
            }
        ]
    )


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
        item["opportunity_probability"] = predict_opportunity(item, opportunity)
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

    model = train_expected_continuation_model(fit_rows, cal_rows)
    raw, bias_adjusted, adjusted = predict_expected_continuation(
        eval_rows,
        model,
    )
    eval_rows["raw_expected_advantage_pct"] = raw
    eval_rows["bias_adjusted_expected_advantage_pct"] = bias_adjusted
    eval_rows["adjusted_expected_advantage_pct"] = adjusted

    diag = diagnostics(eval_rows, evaluation_month)
    for column, value in (
        ("calibration_additive_bias_pct", model.additive_bias_pct),
        (
            "selected_tail_correction_pct",
            model.selected_tail_correction_pct,
        ),
        ("selected_calibration_days", model.selected_calibration_days),
        ("mean_daily_optimism_pct", model.mean_daily_optimism_pct),
        ("optimism_se_pct", model.optimism_se_pct),
        ("fit_clip_low_pct", model.fit_clip_low_pct),
        ("fit_clip_high_pct", model.fit_clip_high_pct),
    ):
        diag[column] = value

    bootstrap = day_bootstrap(eval_rows, evaluation_month)
    primary = diag.loc[
        diag["pool"].eq("opportunity_gate")
        & diag["bucket"].eq("overall")
    ].iloc[0]
    boot = bootstrap.iloc[0]
    has_selection = int(primary.get("selected_hold_rows", 0)) > 0
    success = pd.DataFrame(
        [
            {
                "month": evaluation_month,
                "spearman_positive": bool(
                    has_selection
                    and primary.get("spearman", np.nan) > 0
                ),
                "policy_mean_positive": bool(
                    has_selection
                    and primary.get(
                        "policy_incremental_mean_pct",
                        np.nan,
                    )
                    > 0
                ),
                "day_mean_positive": bool(
                    has_selection
                    and primary.get(
                        "day_balanced_policy_mean_pct",
                        np.nan,
                    )
                    > 0
                ),
                "coverage_at_least_90pct": bool(
                    primary.get("evaluation_coverage", 0) >= 0.9
                ),
                "bootstrap_low_positive": bool(
                    has_selection and boot.get("ci_low_pct", np.nan) > 0
                ),
            }
        ]
    )
    success["all_frozen_checks_pass"] = success.iloc[:, 1:].all(axis=1)
    return diag, eval_rows, bootstrap, provenance, success


def render_report(outputs: tuple[pd.DataFrame, ...]) -> str:
    diagnostics_frame, _scored, bootstrap, provenance, success = outputs
    lines = [
        "=== MoneyMaker Path-Aware Expected Continuation Value v3.6 ===",
        "diagnostic only: unchanged v3.5 state and exact one-minute geometry",
        "target=arithmetic HOLD-vs-EXIT incremental BASE value",
        "train-only 0.5/99.5 winsorization; additive calibration",
        "v1.8 selected-tail optimism allowance; HOLD iff adjusted EV > 0",
        "April 2026+ sealed",
    ]
    for title, frame in (
        ("Diagnostics", diagnostics_frame),
        ("Day bootstrap", bootstrap),
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
