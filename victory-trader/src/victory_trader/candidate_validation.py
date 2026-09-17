from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analytics import load_event_dataset
from .candidate_score import CandidateRule, HORIZONS, apply_candidate_rule


def load_candidate_rule(path: Path) -> CandidateRule:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "candidate-v1":
        raise ValueError(f"unsupported candidate rule version: {payload.get('version')}")
    return CandidateRule(
        train_dates=tuple(str(value) for value in payload["train_dates"]),
        holdout_dates=tuple(str(value) for value in payload["holdout_dates"]),
        thresholds=dict(payload["thresholds"]),
    )


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


def evaluate_external_month(frame: pd.DataFrame, rule: CandidateRule) -> pd.DataFrame:
    scored = apply_candidate_rule(frame, rule)
    scored["trading_day"] = scored["trading_day"].astype(str)
    executable = pd.to_numeric(scored["entry_price"], errors="coerce").notna()

    stages = {
        "baseline": pd.Series(True, index=scored.index),
        "liquidity_only": scored["liquidity_gate_v1"].astype(bool),
        "alpha_only": scored["alpha_selected_v1"].astype(bool),
        "candidate_combined": scored["candidate_v1"].astype(bool),
    }

    rows: list[dict[str, float | int | str]] = []
    for horizon in HORIZONS:
        targets = {
            "gross": f"return_{horizon}m_pct",
            "base": f"return_{horizon}m_base_net_return_pct",
            "stress": f"return_{horizon}m_stress_net_return_pct",
        }
        missing = [target for target in targets.values() if target not in scored.columns]
        if missing:
            raise ValueError(f"dataset missing external validation targets: {missing}")

        for target in targets.values():
            scored[target] = pd.to_numeric(scored[target], errors="coerce")

        for stage, stage_mask in stages.items():
            selected = scored.loc[executable & stage_mask].copy()
            gross_observed = selected[targets["gross"]].notna()
            row: dict[str, float | int | str] = {
                "horizon_min": horizon,
                "stage": stage,
                "selected_executable_n": int(len(selected)),
                "selected_days": int(selected["trading_day"].nunique()) if len(selected) else 0,
                "selection_rate_of_executable": float(len(selected) / executable.sum())
                if executable.any()
                else float("nan"),
                "observed_outcomes_n": int(gross_observed.sum()),
                "observed_given_entry_rate": float(gross_observed.mean())
                if len(selected)
                else float("nan"),
            }
            for scenario, target in targets.items():
                baseline = scored.loc[executable & scored[target].notna()].copy()
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


def summarize_thresholds(frame: pd.DataFrame, rule: CandidateRule) -> pd.DataFrame:
    scored = apply_candidate_rule(frame, rule)
    scored["threshold_pct"] = pd.to_numeric(scored["threshold_pct"], errors="coerce")
    scored["entry_price"] = pd.to_numeric(scored["entry_price"], errors="coerce")
    rows = []
    for threshold, group in scored.groupby("threshold_pct", sort=True):
        executable = group["entry_price"].notna()
        rows.append(
            {
                "threshold_pct": float(threshold),
                "events": int(len(group)),
                "executable_entries": int(executable.sum()),
                "rule_eligible_events": int(group["candidate_rule_eligible_v1"].sum()),
                "liquidity_pass_executable": int(
                    (executable & group["liquidity_gate_v1"]).sum()
                ),
                "alpha_pass_executable": int(
                    (executable & group["alpha_selected_v1"]).sum()
                ),
                "candidate_executable": int(
                    (executable & group["candidate_v1"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def render_external_validation(
    frame: pd.DataFrame,
    rule: CandidateRule,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    evaluation = evaluate_external_month(frame, rule)
    thresholds = summarize_thresholds(frame, rule)
    combined = evaluation.loc[evaluation["stage"].eq("candidate_combined")].copy()
    diagnostic = evaluation.loc[
        evaluation["horizon_min"].isin([5, 10, 15]),
        [
            "horizon_min",
            "stage",
            "selected_executable_n",
            "observed_given_entry_rate",
            "gross_mean_pct",
            "base_mean_pct",
            "stress_mean_pct",
            "base_matched_lift_pct",
        ],
    ]

    report = "\n".join(
        [
            "=== MoneyMaker Candidate v1 External Validation ===",
            f"dataset_days={frame['trading_day'].astype(str).min()}..{frame['trading_day'].astype(str).max()}",
            f"locked_rule_train={rule.train_dates[0]}..{rule.train_dates[-1]}",
            "rule_status=FROZEN; no refit or threshold changes on this dataset",
            "",
            "=== Rule coverage by event threshold ===",
            thresholds.to_string(index=False),
            "",
            "=== Combined candidate external performance ===",
            combined[
                [
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
            ].to_string(index=False),
            "",
            "=== Stagewise external diagnostic at 5/10/15m ===",
            diagnostic.to_string(index=False),
            "",
            "Decision rule: do not alter candidate-v1 from this month alone. Compare direction, lift, observability, and net economics with March before any revision.",
        ]
    )
    return report, evaluation, thresholds


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.candidate_validation"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--rule-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evaluation-csv", type=Path, required=True)
    parser.add_argument("--thresholds-csv", type=Path, required=True)
    args = parser.parse_args()

    frame = load_event_dataset(args.dataset)
    rule = load_candidate_rule(args.rule_json)
    report, evaluation, thresholds = render_external_validation(frame, rule)
    print(report)

    for path in (args.report, args.evaluation_csv, args.thresholds_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    evaluation.to_csv(args.evaluation_csv, index=False)
    thresholds.to_csv(args.thresholds_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
