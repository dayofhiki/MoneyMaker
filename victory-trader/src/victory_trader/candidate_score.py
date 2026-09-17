from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import load_event_dataset


ALPHA_FEATURES = (
    "prior_15m_low_rebound_pct",
    "volatility_15m_pct",
    "signal_bar_range_pct",
    "trailing_return_5m_pct",
)
LIQUIDITY_FEATURES = (
    "dollar_volume_5m",
    "transactions_5m",
    "active_minute_fraction_15m",
)
HORIZONS = (1, 2, 5, 10, 15, 30, 60)
TRAIN_FRACTION = 0.70
LIQUIDITY_FLOOR_QUANTILE = 0.20
ALPHA_SCORE_QUANTILE = 0.80
MIN_ALPHA_FEATURES = 3
MIN_TRAIN_ROWS_PER_THRESHOLD = 50


@dataclass(frozen=True)
class CandidateRule:
    train_dates: tuple[str, ...]
    holdout_dates: tuple[str, ...]
    thresholds: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": "candidate-v1",
            "design": {
                "alpha_features": list(ALPHA_FEATURES),
                "alpha_direction": "lower_is_better",
                "liquidity_features": list(LIQUIDITY_FEATURES),
                "train_fraction": TRAIN_FRACTION,
                "liquidity_floor_quantile": LIQUIDITY_FLOOR_QUANTILE,
                "alpha_score_quantile": ALPHA_SCORE_QUANTILE,
                "min_alpha_features": MIN_ALPHA_FEATURES,
                "min_train_rows_per_threshold": MIN_TRAIN_ROWS_PER_THRESHOLD,
                "weighting": "equal_weight_empirical_percentile",
            },
            "train_dates": list(self.train_dates),
            "holdout_dates": list(self.holdout_dates),
            "thresholds": self.thresholds,
        }


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _split_dates(frame: pd.DataFrame) -> tuple[set[str], set[str]]:
    dates = sorted(frame["trading_day"].astype(str).unique())
    if len(dates) < 4:
        raise ValueError("candidate score needs at least four trading days")
    split = int(len(dates) * TRAIN_FRACTION)
    split = min(max(split, 2), len(dates) - 2)
    return set(dates[:split]), set(dates[split:])


def _favorable_percentile(value: float, reference: list[float]) -> float:
    if not reference or pd.isna(value):
        return float("nan")
    array = np.asarray(reference, dtype=float)
    # Lower values are favorable. A training-minimum-like value approaches 1.
    rank = np.searchsorted(array, float(value), side="left")
    return float(1.0 - rank / len(array))


def _alpha_score_row(row: pd.Series, references: dict[str, list[float]]) -> float:
    scores = []
    for feature in ALPHA_FEATURES:
        value = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        score = _favorable_percentile(float(value), references.get(feature, []))
        if pd.notna(score):
            scores.append(score)
    if len(scores) < MIN_ALPHA_FEATURES:
        return float("nan")
    return float(np.mean(scores))


def fit_candidate_rule(frame: pd.DataFrame) -> CandidateRule:
    required = {"trading_day", "threshold_pct", "entry_price"} | set(ALPHA_FEATURES) | set(
        LIQUIDITY_FEATURES
    )
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing candidate-rule columns: {sorted(missing)}")

    work = _numeric(frame, ("threshold_pct",) + ALPHA_FEATURES + LIQUIDITY_FEATURES)
    work["trading_day"] = work["trading_day"].astype(str)
    train_dates, holdout_dates = _split_dates(work)
    train = work.loc[work["trading_day"].isin(train_dates)].copy()

    thresholds: dict[str, dict[str, object]] = {}
    for threshold, group in train.groupby("threshold_pct", sort=True):
        base = group.loc[pd.to_numeric(group["entry_price"], errors="coerce").notna()].copy()
        if len(base) < MIN_TRAIN_ROWS_PER_THRESHOLD:
            continue

        liquidity_cuts: dict[str, float] = {}
        for feature in LIQUIDITY_FEATURES:
            values = base[feature].dropna()
            if len(values) < MIN_TRAIN_ROWS_PER_THRESHOLD:
                break
            liquidity_cuts[feature] = float(values.quantile(LIQUIDITY_FLOOR_QUANTILE))
        if len(liquidity_cuts) != len(LIQUIDITY_FEATURES):
            continue

        references: dict[str, list[float]] = {}
        for feature in ALPHA_FEATURES:
            values = sorted(base[feature].dropna().astype(float).tolist())
            if len(values) >= MIN_TRAIN_ROWS_PER_THRESHOLD:
                references[feature] = values
        if len(references) < MIN_ALPHA_FEATURES:
            continue

        scores = base.apply(lambda row: _alpha_score_row(row, references), axis=1)
        valid_scores = scores.dropna()
        if len(valid_scores) < MIN_TRAIN_ROWS_PER_THRESHOLD:
            continue
        alpha_cut = float(valid_scores.quantile(ALPHA_SCORE_QUANTILE))

        thresholds[str(float(threshold))] = {
            "train_executable_n": int(len(base)),
            "liquidity_cuts": liquidity_cuts,
            "alpha_references": references,
            "alpha_score_cut": alpha_cut,
        }

    if not thresholds:
        raise ValueError("no thresholds had enough training data to fit candidate rule")

    return CandidateRule(
        train_dates=tuple(sorted(train_dates)),
        holdout_dates=tuple(sorted(holdout_dates)),
        thresholds=thresholds,
    )


def apply_candidate_rule(frame: pd.DataFrame, rule: CandidateRule) -> pd.DataFrame:
    work = _numeric(frame, ("threshold_pct",) + ALPHA_FEATURES + LIQUIDITY_FEATURES)
    result = work.copy()
    result["alpha_score_v1"] = np.nan
    result["liquidity_gate_v1"] = False
    result["alpha_selected_v1"] = False
    result["candidate_v1"] = False
    result["candidate_rule_eligible_v1"] = False

    for threshold_key, params in rule.thresholds.items():
        threshold = float(threshold_key)
        mask = result["threshold_pct"].eq(threshold)
        if not mask.any():
            continue
        references = params["alpha_references"]
        liquidity_cuts = params["liquidity_cuts"]
        alpha_cut = float(params["alpha_score_cut"])

        subset = result.loc[mask]
        scores = subset.apply(lambda row: _alpha_score_row(row, references), axis=1)
        liquid = pd.Series(True, index=subset.index)
        for feature, cut in liquidity_cuts.items():
            liquid &= pd.to_numeric(subset[feature], errors="coerce").ge(float(cut))
        alpha_selected = scores.ge(alpha_cut)

        result.loc[mask, "alpha_score_v1"] = scores
        result.loc[mask, "liquidity_gate_v1"] = liquid
        result.loc[mask, "alpha_selected_v1"] = alpha_selected
        result.loc[mask, "candidate_v1"] = liquid & alpha_selected
        result.loc[mask, "candidate_rule_eligible_v1"] = True

    return result


def _matched_lift(
    baseline: pd.DataFrame,
    selected: pd.DataFrame,
    target: str,
) -> float:
    if selected.empty:
        return float("nan")
    cell_cols = ["trading_day", "threshold_pct"]
    means = baseline.groupby(cell_cols)[target].mean()
    index = pd.MultiIndex.from_frame(selected[cell_cols])
    matched = means.reindex(index).to_numpy(dtype=float)
    return float((selected[target].to_numpy(dtype=float) - matched).mean())


def evaluate_candidate_rule(frame: pd.DataFrame, rule: CandidateRule) -> pd.DataFrame:
    scored = apply_candidate_rule(frame, rule)
    scored["trading_day"] = scored["trading_day"].astype(str)
    test = scored.loc[scored["trading_day"].isin(rule.holdout_dates)].copy()
    executable = pd.to_numeric(test["entry_price"], errors="coerce").notna()

    stages = {
        "baseline": pd.Series(True, index=test.index),
        "liquidity_only": test["liquidity_gate_v1"].astype(bool),
        "alpha_only": test["alpha_selected_v1"].astype(bool),
        "candidate_combined": test["candidate_v1"].astype(bool),
    }

    rows: list[dict[str, float | int | str]] = []
    for horizon in HORIZONS:
        targets = {
            "gross": f"return_{horizon}m_pct",
            "base": f"return_{horizon}m_base_net_return_pct",
            "stress": f"return_{horizon}m_stress_net_return_pct",
        }
        missing = [column for column in targets.values() if column not in test.columns]
        if missing:
            raise ValueError(f"dataset missing candidate targets: {missing}")

        for stage, stage_mask in stages.items():
            selected = test.loc[executable & stage_mask].copy()
            row: dict[str, float | int | str] = {
                "horizon_min": horizon,
                "stage": stage,
                "selected_executable_n": int(len(selected)),
                "selected_days": int(selected["trading_day"].nunique()) if len(selected) else 0,
                "selection_rate_of_executable": float(len(selected) / executable.sum())
                if executable.any()
                else float("nan"),
            }

            gross_observed = pd.to_numeric(selected[targets["gross"]], errors="coerce").notna()
            row["observed_outcomes_n"] = int(gross_observed.sum())
            row["observed_given_entry_rate"] = float(gross_observed.mean()) if len(selected) else float("nan")

            for scenario, target in targets.items():
                test[target] = pd.to_numeric(test[target], errors="coerce")
                selected[target] = pd.to_numeric(selected[target], errors="coerce")
                baseline = test.loc[executable & test[target].notna()].copy()
                observed = selected.loc[selected[target].notna()].copy()
                row[f"{scenario}_mean_pct"] = (
                    float(observed[target].mean()) if len(observed) else float("nan")
                )
                row[f"{scenario}_median_pct"] = (
                    float(observed[target].median()) if len(observed) else float("nan")
                )
                row[f"{scenario}_positive_rate"] = (
                    float((observed[target] > 0).mean()) if len(observed) else float("nan")
                )
                row[f"{scenario}_p05_pct"] = (
                    float(observed[target].quantile(0.05)) if len(observed) else float("nan")
                )
                row[f"{scenario}_matched_lift_pct"] = _matched_lift(
                    baseline, observed, target
                )
            rows.append(row)

    return pd.DataFrame(rows)


def render_candidate_report(
    frame: pd.DataFrame,
) -> tuple[str, CandidateRule, pd.DataFrame]:
    rule = fit_candidate_rule(frame)
    evaluation = evaluate_candidate_rule(frame, rule)

    combined = evaluation.loc[evaluation["stage"].eq("candidate_combined")].copy()
    table_columns = [
        "horizon_min",
        "selected_executable_n",
        "selected_days",
        "selection_rate_of_executable",
        "observed_given_entry_rate",
        "gross_mean_pct",
        "gross_matched_lift_pct",
        "base_mean_pct",
        "base_matched_lift_pct",
        "stress_mean_pct",
        "stress_matched_lift_pct",
        "base_positive_rate",
        "base_p05_pct",
    ]

    diagnostic = evaluation.loc[
        evaluation["horizon_min"].isin([5, 10, 15]),
        [
            "horizon_min",
            "stage",
            "selected_executable_n",
            "observed_given_entry_rate",
            "gross_mean_pct",
            "base_mean_pct",
            "base_matched_lift_pct",
        ],
    ]

    report = "\n".join(
        [
            "=== MoneyMaker Candidate Score v1 ===",
            f"alpha_features={','.join(ALPHA_FEATURES)}",
            f"liquidity_features={','.join(LIQUIDITY_FEATURES)}",
            f"train_dates={rule.train_dates[0]}..{rule.train_dates[-1]} ({len(rule.train_dates)} days)",
            f"holdout_dates={rule.holdout_dates[0]}..{rule.holdout_dates[-1]} ({len(rule.holdout_dates)} days)",
            f"eligible_thresholds={','.join(rule.thresholds.keys())}",
            "rule=equal-weight lower-is-better empirical alpha percentiles; top-20% alpha score; each liquidity metric above its train 20th percentile",
            "",
            "=== Combined candidate holdout across horizons ===",
            combined.loc[:, table_columns].to_string(index=False),
            "",
            "=== Stagewise diagnostic at 5/10/15m ===",
            diagnostic.to_string(index=False),
            "",
            "Lock rule: no feature weights, quantiles, or feature membership may be tuned after viewing the next-month sample.",
            "Interpretation: March holdout is an internal development check, not out-of-sample proof. The next independent test is February 2026.",
        ]
    )
    return report, rule, evaluation


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m victory_trader.candidate_score")
    parser.add_argument("path", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rule-json", type=Path, required=True)
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    args = parser.parse_args()

    frame = load_event_dataset(args.path)
    report, rule, evaluation = render_candidate_report(frame)
    print(report)

    for path in (args.report, args.rule_json, args.evaluation_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    args.rule_json.write_text(
        json.dumps(rule.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluation.to_csv(args.evaluation_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
