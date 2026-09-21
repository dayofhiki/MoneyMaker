from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .execution_costs import DEFAULT_EXECUTION_SCENARIOS, modeled_sell_fill
from .expanded_filing_semantics_hurdle_ev import (
    filing_semantic_split_supply_feature_frame,
)
from .expanded_opportunity_shared_q import (
    predict_opportunity,
    train_opportunity_model,
)
from .expanded_supply_hurdle_ev import load_fold
from .state_sequence_enrichment import SEQUENCE_FEATURES, enrich_state_sequence


MINUTE_MS = 60_000
MAX_HOLD_MINUTES = 30
HOLD_THRESHOLD = 0.5
BOOTSTRAP_SAMPLES = 10_000
KEYS = ["trading_day", "ticker"]
PHASES = (
    "consolidation",
    "failure",
    "impulse",
    "other",
    "pullback_above_vwap",
    "pullback_below_vwap",
    "reclaim",
)
BASE_SCENARIO = next(s for s in DEFAULT_EXECUTION_SCENARIOS if s.name == "base")


@dataclass(frozen=True)
class ContinuationModel:
    model: HistGradientBoostingClassifier
    feature_columns: tuple[str, ...]
    platt: LogisticRegression


def continuation_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = filing_semantic_split_supply_feature_frame(frame).copy()
    for column in SEQUENCE_FEATURES:
        result[column] = pd.to_numeric(frame[column], errors="coerce")
    result["continuation_minutes_held"] = pd.to_numeric(
        frame["minutes_held"], errors="coerce"
    )
    phase = frame["phase"].astype(str)
    for value in PHASES:
        result[f"continuation_phase_{value}"] = phase.eq(value).astype(float)
    return result


def build_continuation_rows(
    state: pd.DataFrame,
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    """Build causal post-entry EXIT-now versus HOLD-one-minute examples.

    A state bar at t is observable at t+1 minute. EXIT uses the exact open at
    t+1; HOLD uses the exact open at t+2. Missing timestamps are never skipped.
    """
    if anchors.duplicated(KEYS).any():
        raise ValueError("anchors must contain one row per ticker-day")
    required = set(KEYS + ["t", "o"])
    if not required.issubset(state.columns):
        raise ValueError("state panel requires trading_day/ticker/t/o")

    enriched = enrich_state_sequence(state)
    anchor_columns = [c for c in anchors.columns if c not in KEYS + ["t"]]
    anchor_data = anchors.loc[:, KEYS + ["t"] + anchor_columns].rename(
        columns={"t": "anchor_t"}
    )
    duplicate_payload = [c for c in anchor_columns if c in enriched.columns]
    anchor_data = anchor_data.drop(columns=duplicate_payload)
    rows = enriched.merge(anchor_data, on=KEYS, how="inner", validate="many_to_one")
    rows["minutes_held"] = (
        pd.to_numeric(rows["t"], errors="coerce")
        - pd.to_numeric(rows["anchor_t"], errors="coerce")
    ) / MINUTE_MS
    rows = rows.loc[
        rows["minutes_held"].ge(1)
        & rows["minutes_held"].lt(MAX_HOLD_MINUTES)
    ].copy()

    opens = enriched.loc[:, KEYS + ["t", "o"]].copy()
    opens["t"] = pd.to_numeric(opens["t"], errors="coerce")
    rows["exit_reference_t"] = pd.to_numeric(rows["t"], errors="coerce") + MINUTE_MS
    rows["hold_reference_t"] = pd.to_numeric(rows["t"], errors="coerce") + 2 * MINUTE_MS
    exit_open = opens.rename(columns={"t": "exit_reference_t", "o": "exit_open"})
    hold_open = opens.rename(columns={"t": "hold_reference_t", "o": "hold_open"})
    rows = rows.merge(exit_open, on=KEYS + ["exit_reference_t"], how="left", validate="many_to_one")
    rows = rows.merge(hold_open, on=KEYS + ["hold_reference_t"], how="left", validate="many_to_one")

    exit_valid = pd.to_numeric(rows["exit_open"], errors="coerce").gt(0)
    hold_valid = pd.to_numeric(rows["hold_open"], errors="coerce").gt(0)
    rows["evaluation_reason"] = np.select(
        [~exit_valid, exit_valid & ~hold_valid],
        ["missing_exact_exit_open", "missing_exact_hold_open"],
        default="evaluated",
    )
    rows["exit_sell_fill"] = np.nan
    rows["hold_sell_fill"] = np.nan
    valid = exit_valid & hold_valid
    rows.loc[valid, "exit_sell_fill"] = rows.loc[valid, "exit_open"].map(
        lambda x: modeled_sell_fill(float(x), BASE_SCENARIO)
    )
    rows.loc[valid, "hold_sell_fill"] = rows.loc[valid, "hold_open"].map(
        lambda x: modeled_sell_fill(float(x), BASE_SCENARIO)
    )
    rows["hold_advantage_pct"] = np.where(
        valid,
        (rows["hold_sell_fill"] / rows["exit_sell_fill"] - 1.0) * 100.0,
        np.nan,
    )
    rows["hold_wins"] = rows["hold_advantage_pct"].gt(0).where(valid)
    return rows


def train_continuation_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> ContinuationModel:
    fit = fit.loc[fit["hold_advantage_pct"].notna()].copy()
    calibration = calibration.loc[calibration["hold_advantage_pct"].notna()].copy()
    y_fit = fit["hold_advantage_pct"].gt(0).astype(int)
    y_cal = calibration["hold_advantage_pct"].gt(0).astype(int)
    if y_fit.nunique() < 2 or y_cal.nunique() < 2:
        raise ValueError("continuation fit/calibration requires both classes")
    x_fit = continuation_feature_frame(fit)
    columns = [c for c in x_fit if x_fit[c].notna().any()]
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=20261041,
    )
    model.fit(x_fit.loc[:, columns], y_fit)
    raw = model.predict_proba(
        continuation_feature_frame(calibration).reindex(columns=columns)
    )[:, 1]
    raw = np.clip(raw, 1e-6, 1 - 1e-6)
    platt = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=1000, random_state=20261041
    )
    platt.fit(np.log(raw / (1 - raw)).reshape(-1, 1), y_cal)
    return ContinuationModel(model, tuple(columns), platt)


def predict_continuation(frame: pd.DataFrame, fitted: ContinuationModel) -> np.ndarray:
    raw = fitted.model.predict_proba(
        continuation_feature_frame(frame).reindex(columns=fitted.feature_columns)
    )[:, 1]
    raw = np.clip(raw, 1e-6, 1 - 1e-6)
    return fitted.platt.predict_proba(
        np.log(raw / (1 - raw)).reshape(-1, 1)
    )[:, 1]


def _safe_auc(y: pd.Series, p: pd.Series) -> float:
    return float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan


def _diagnostic_row(frame: pd.DataFrame, *, month: str, pool: str, bucket: str) -> dict[str, object]:
    evaluated = frame.loc[frame["hold_advantage_pct"].notna()].copy()
    if evaluated.empty:
        return {"month": month, "pool": pool, "bucket": bucket, "candidate_rows": len(frame), "evaluated_rows": 0}
    y = evaluated["hold_advantage_pct"].gt(0).astype(int)
    p = pd.to_numeric(evaluated["hold_probability"], errors="coerce")
    action_gain = pd.to_numeric(evaluated["hold_advantage_pct"], errors="coerce").where(
        p.gt(HOLD_THRESHOLD), 0.0
    )
    daily = pd.DataFrame({"day": evaluated["trading_day"].astype(str), "gain": action_gain}).groupby("day")["gain"].mean()
    return {
        "month": month,
        "pool": pool,
        "bucket": bucket,
        "candidate_rows": int(len(frame)),
        "evaluated_rows": int(len(evaluated)),
        "evaluation_coverage": float(len(evaluated) / len(frame)) if len(frame) else np.nan,
        "hold_base_rate": float(y.mean()),
        "auc": _safe_auc(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p.clip(1e-9, 1 - 1e-9), labels=[0, 1])),
        "predicted_hold_rate": float(p.gt(HOLD_THRESHOLD).mean()),
        "always_hold_mean_pct": float(evaluated["hold_advantage_pct"].mean()),
        "policy_incremental_mean_pct": float(action_gain.mean()),
        "day_balanced_policy_mean_pct": float(daily.mean()),
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
            rows.append(_diagnostic_row(scored.loc[mask], month=month, pool=pool, bucket=bucket))
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
    gain = selected["hold_advantage_pct"].where(
        selected["hold_probability"].gt(HOLD_THRESHOLD), 0.0
    )
    daily = pd.DataFrame({"day": selected["trading_day"].astype(str), "gain": gain}).groupby("day")["gain"].mean().dropna().to_numpy()
    if not len(daily):
        return pd.DataFrame([{"month": month, "days": 0}])
    rng = np.random.default_rng(20261041)
    means = daily[rng.integers(0, len(daily), size=(BOOTSTRAP_SAMPLES, len(daily)))].mean(axis=1)
    return pd.DataFrame([{
        "month": month,
        "days": int(len(daily)),
        "day_balanced_mean_pct": float(daily.mean()),
        "ci_low_pct": float(np.quantile(means, 0.025)),
        "ci_high_pct": float(np.quantile(means, 0.975)),
    }])


def _month_of(frame: pd.DataFrame) -> pd.Series:
    return frame["trading_day"].astype(str).str.slice(0, 7)


def run_fold(anchor_paths: dict[str, Path], state_paths: dict[str, Path], evaluation_month: str):
    fit, calibration, evaluation, provenance = load_fold(anchor_paths, evaluation_month)
    opportunity = train_opportunity_model(fit, calibration)
    partitions = []
    for name, frame in (("fit", fit), ("calibration", calibration), ("evaluation", evaluation)):
        item = frame.copy()
        item["_partition"] = name
        item["opportunity_probability"] = predict_opportunity(item, opportunity)
        partitions.append(item)
    anchors = pd.concat(partitions, ignore_index=True)
    sequence_parts = []
    for month, group in anchors.groupby(_month_of(anchors), sort=True):
        if month not in state_paths:
            raise ValueError(f"missing state panel for {month}")
        sequence_parts.append(build_continuation_rows(pd.read_parquet(state_paths[month]), group))
    sequence = pd.concat(sequence_parts, ignore_index=True)
    fit_rows = sequence.loc[sequence["_partition"].eq("fit")].reset_index(drop=True)
    cal_rows = sequence.loc[
        sequence["_partition"].eq("calibration")
    ].reset_index(drop=True)
    eval_rows = sequence.loc[
        sequence["_partition"].eq("evaluation")
    ].reset_index(drop=True)
    model = train_continuation_model(fit_rows, cal_rows)
    eval_rows["hold_probability"] = predict_continuation(eval_rows, model)
    diag = diagnostics(eval_rows, evaluation_month)
    bootstrap = day_bootstrap(eval_rows, evaluation_month)
    primary = diag.loc[
        diag["pool"].eq("opportunity_gate") & diag["bucket"].eq("overall")
    ].iloc[0]
    boot = bootstrap.iloc[0]
    success = pd.DataFrame([{
        "month": evaluation_month,
        "auc_above_half": bool(primary.get("auc", np.nan) > 0.5),
        "policy_mean_positive": bool(primary.get("policy_incremental_mean_pct", np.nan) > 0),
        "day_mean_positive": bool(primary.get("day_balanced_policy_mean_pct", np.nan) > 0),
        "coverage_at_least_90pct": bool(primary.get("evaluation_coverage", 0) >= 0.9),
        "bootstrap_low_positive": bool(boot.get("ci_low_pct", np.nan) > 0),
    }])
    success["all_frozen_checks_pass"] = success.iloc[:, 1:].all(axis=1)
    return diag, eval_rows, bootstrap, provenance, success


def render_report(outputs: tuple[pd.DataFrame, ...]) -> str:
    diagnostics_frame, _scored, bootstrap, provenance, success = outputs
    lines = [
        "=== MoneyMaker One-Step Continuation Observability v3.4 ===",
        "diagnostic only: conceptual entry at frozen first anchor",
        "EXIT=exact next open; HOLD=exact open one minute later; BASE future sell friction only",
        "post-entry states held 1-29 minutes; no missing-bar fallback; April 2026+ sealed",
    ]
    for title, frame in (("Diagnostics", diagnostics_frame), ("Day bootstrap", bootstrap), ("Provenance", provenance), ("Frozen checks", success)):
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
    outputs = run_fold(dict(args.anchor), dict(args.state), args.evaluation_month)
    report = render_report(outputs)
    print(report, flush=True)
    for path, frame in zip((args.diagnostics_csv, args.scored_csv, args.bootstrap_csv, args.provenance_csv, args.success_csv), outputs, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
