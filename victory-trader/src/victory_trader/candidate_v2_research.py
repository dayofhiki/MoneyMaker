from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import load_event_dataset
from .execution_costs import DEFAULT_EXECUTION_SCENARIOS, net_round_trip_return_pct


ALPHA_FAMILIES = {
    "core2": ("volatility_15m_pct", "signal_bar_range_pct"),
    "core3": (
        "volatility_15m_pct",
        "signal_bar_range_pct",
        "prior_15m_low_rebound_pct",
    ),
}
LIQUIDITY_FEATURES = (
    "dollar_volume_5m",
    "transactions_5m",
    "active_minute_fraction_15m",
)
SELECTION_QUANTILES = (0.80, 0.90, 0.95)
BASE_COST_CEILINGS_PCT = (None, 1.50, 1.00)
HORIZONS = (5, 10, 15)
ELIGIBLE_THRESHOLDS = (10.0, 20.0, 30.0, 50.0)
LIQUIDITY_FLOOR_QUANTILE = 0.20
MIN_TRAIN_ROWS_PER_THRESHOLD = 50
MIN_HELDOUT_OBSERVED = 40

BASE_SCENARIO = next(s for s in DEFAULT_EXECUTION_SCENARIOS if s.name == "base")


@dataclass(frozen=True)
class V2Config:
    alpha_family: str
    selection_quantile: float
    base_cost_ceiling_pct: float | None

    @property
    def key(self) -> str:
        cap = "none" if self.base_cost_ceiling_pct is None else f"{self.base_cost_ceiling_pct:.2f}"
        pct = int(round((1.0 - self.selection_quantile) * 100))
        return f"{self.alpha_family}_top{pct}_cost{cap}"


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _favorable_percentile(value: float, reference: list[float]) -> float:
    if pd.isna(value) or not reference:
        return float("nan")
    array = np.asarray(reference, dtype=float)
    rank = np.searchsorted(array, float(value), side="left")
    return float(1.0 - rank / len(array))


def _alpha_score_row(
    row: pd.Series,
    features: tuple[str, ...],
    references: dict[str, list[float]],
) -> float:
    scores = []
    for feature in features:
        value = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
        if pd.isna(value):
            return float("nan")
        score = _favorable_percentile(float(value), references.get(feature, []))
        if pd.isna(score):
            return float("nan")
        scores.append(score)
    return float(np.mean(scores))


def modeled_base_zero_return_cost_pct(entry_price: float) -> float:
    if pd.isna(entry_price) or float(entry_price) <= 0:
        return float("nan")
    net = net_round_trip_return_pct(float(entry_price), 0.0, BASE_SCENARIO)
    return float(-net)


def _fit_threshold_params(
    train: pd.DataFrame,
    config: V2Config,
) -> dict[str, dict[str, object]]:
    features = ALPHA_FAMILIES[config.alpha_family]
    params: dict[str, dict[str, object]] = {}

    for threshold in ELIGIBLE_THRESHOLDS:
        group = train.loc[
            train["threshold_pct"].eq(threshold)
            & pd.to_numeric(train["entry_price"], errors="coerce").notna()
        ].copy()
        if len(group) < MIN_TRAIN_ROWS_PER_THRESHOLD:
            continue

        liquidity_cuts: dict[str, float] = {}
        for feature in LIQUIDITY_FEATURES:
            values = group[feature].dropna().astype(float)
            if len(values) < MIN_TRAIN_ROWS_PER_THRESHOLD:
                break
            liquidity_cuts[feature] = float(values.quantile(LIQUIDITY_FLOOR_QUANTILE))
        if len(liquidity_cuts) != len(LIQUIDITY_FEATURES):
            continue

        references: dict[str, list[float]] = {}
        for feature in features:
            values = sorted(group[feature].dropna().astype(float).tolist())
            if len(values) < MIN_TRAIN_ROWS_PER_THRESHOLD:
                break
            references[feature] = values
        if len(references) != len(features):
            continue

        scores = group.apply(
            lambda row: _alpha_score_row(row, features, references),
            axis=1,
        ).dropna()
        if len(scores) < MIN_TRAIN_ROWS_PER_THRESHOLD:
            continue

        params[str(threshold)] = {
            "liquidity_cuts": liquidity_cuts,
            "alpha_references": references,
            "alpha_score_cut": float(scores.quantile(config.selection_quantile)),
            "train_n": int(len(group)),
        }
    return params


def apply_v2_config(
    frame: pd.DataFrame,
    params: dict[str, dict[str, object]],
    config: V2Config,
) -> pd.DataFrame:
    features = ALPHA_FAMILIES[config.alpha_family]
    work = _numeric(
        frame,
        ("threshold_pct", "entry_price") + features + LIQUIDITY_FEATURES,
    )
    out = work.copy()
    out["candidate_v2_score"] = np.nan
    out["candidate_v2_liquid"] = False
    out["candidate_v2_cost_ok"] = False
    out["candidate_v2_selected"] = False
    out["base_zero_return_cost_pct"] = out["entry_price"].apply(
        modeled_base_zero_return_cost_pct
    )

    for threshold_key, threshold_params in params.items():
        threshold = float(threshold_key)
        mask = out["threshold_pct"].eq(threshold)
        if not mask.any():
            continue

        subset = out.loc[mask]
        references = threshold_params["alpha_references"]
        scores = subset.apply(
            lambda row: _alpha_score_row(row, features, references),
            axis=1,
        )

        liquid = pd.Series(True, index=subset.index)
        for feature, cut in threshold_params["liquidity_cuts"].items():
            liquid &= subset[feature].ge(float(cut))

        if config.base_cost_ceiling_pct is None:
            cost_ok = subset["base_zero_return_cost_pct"].notna()
        else:
            cost_ok = subset["base_zero_return_cost_pct"].le(
                float(config.base_cost_ceiling_pct)
            )

        alpha_ok = scores.ge(float(threshold_params["alpha_score_cut"]))
        selected = liquid & cost_ok & alpha_ok

        out.loc[mask, "candidate_v2_score"] = scores
        out.loc[mask, "candidate_v2_liquid"] = liquid
        out.loc[mask, "candidate_v2_cost_ok"] = cost_ok
        out.loc[mask, "candidate_v2_selected"] = selected

    return out


def _matched_lift(
    baseline: pd.DataFrame,
    selected: pd.DataFrame,
    target: str,
) -> float:
    if selected.empty:
        return float("nan")
    cell_cols = ["trading_day", "threshold_pct"]
    means = baseline.groupby(cell_cols)[target].mean()
    idx = pd.MultiIndex.from_frame(selected[cell_cols])
    matched = means.reindex(idx).to_numpy(dtype=float)
    return float((selected[target].to_numpy(dtype=float) - matched).mean())


def _evaluate_one(
    holdout: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    month: str,
    config: V2Config,
) -> list[dict[str, object]]:
    executable = pd.to_numeric(scored["entry_price"], errors="coerce").notna()
    selected_mask = executable & scored["candidate_v2_selected"].astype(bool)
    rows: list[dict[str, object]] = []

    for horizon in HORIZONS:
        targets = {
            "gross": f"return_{horizon}m_pct",
            "base": f"return_{horizon}m_base_net_return_pct",
            "stress": f"return_{horizon}m_stress_net_return_pct",
        }
        row: dict[str, object] = {
            "month": month,
            "config": config.key,
            "alpha_family": config.alpha_family,
            "selection_quantile": config.selection_quantile,
            "base_cost_ceiling_pct": config.base_cost_ceiling_pct,
            "horizon_min": horizon,
            "executable_n": int(executable.sum()),
            "selected_n": int(selected_mask.sum()),
            "selection_rate": float(selected_mask.sum() / executable.sum())
            if executable.any()
            else float("nan"),
        }

        for target in targets.values():
            scored[target] = pd.to_numeric(scored[target], errors="coerce")

        selected = scored.loc[selected_mask].copy()
        gross_obs = selected[targets["gross"]].notna()
        row["observed_n"] = int(gross_obs.sum())
        row["observed_given_entry_rate"] = (
            float(gross_obs.mean()) if len(selected) else float("nan")
        )

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
                baseline,
                observed,
                target,
            )
        rows.append(row)
    return rows


def run_leave_one_month_out(
    monthly_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if len(monthly_frames) < 3:
        raise ValueError("candidate v2 research requires at least three development months")

    all_features = tuple(
        sorted(set(feature for family in ALPHA_FAMILIES.values() for feature in family))
    )
    prepared: dict[str, pd.DataFrame] = {}
    for month, frame in monthly_frames.items():
        required = {
            "trading_day",
            "threshold_pct",
            "entry_price",
        } | set(all_features) | set(LIQUIDITY_FEATURES)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{month} missing v2 columns: {sorted(missing)}")
        item = _numeric(
            frame,
            ("threshold_pct", "entry_price") + all_features + LIQUIDITY_FEATURES,
        )
        item["trading_day"] = item["trading_day"].astype(str)
        prepared[month] = item

    configs = [
        V2Config(family, quantile, cap)
        for family in ALPHA_FAMILIES
        for quantile in SELECTION_QUANTILES
        for cap in BASE_COST_CEILINGS_PCT
    ]

    rows: list[dict[str, object]] = []
    for holdout_month, holdout in prepared.items():
        train = pd.concat(
            [frame for month, frame in prepared.items() if month != holdout_month],
            ignore_index=True,
        )
        for config in configs:
            params = _fit_threshold_params(train, config)
            scored = apply_v2_config(holdout, params, config)
            rows.extend(
                _evaluate_one(
                    holdout,
                    scored,
                    month=holdout_month,
                    config=config,
                )
            )
    return pd.DataFrame(rows)


def summarize_configs(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "config",
        "alpha_family",
        "selection_quantile",
        "base_cost_ceiling_pct",
        "horizon_min",
    ]
    for keys, group in details.groupby(group_cols, dropna=False, sort=False):
        (
            config,
            alpha_family,
            selection_quantile,
            base_cost_ceiling_pct,
            horizon_min,
        ) = keys
        valid = group.loc[group["observed_n"].ge(MIN_HELDOUT_OBSERVED)].copy()
        rows.append(
            {
                "config": config,
                "alpha_family": alpha_family,
                "selection_quantile": float(selection_quantile),
                "base_cost_ceiling_pct": (
                    None
                    if pd.isna(base_cost_ceiling_pct)
                    else float(base_cost_ceiling_pct)
                ),
                "horizon_min": int(horizon_min),
                "months_tested": int(len(valid)),
                "min_observed_n": int(valid["observed_n"].min()) if len(valid) else 0,
                "median_selection_rate": float(valid["selection_rate"].median())
                if len(valid)
                else float("nan"),
                "gross_positive_months": int((valid["gross_mean_pct"] > 0).sum()),
                "base_positive_months": int((valid["base_mean_pct"] > 0).sum()),
                "base_lift_positive_months": int(
                    (valid["base_matched_lift_pct"] > 0).sum()
                ),
                "stress_lift_positive_months": int(
                    (valid["stress_matched_lift_pct"] > 0).sum()
                ),
                "median_gross_mean_pct": float(valid["gross_mean_pct"].median())
                if len(valid)
                else float("nan"),
                "median_base_mean_pct": float(valid["base_mean_pct"].median())
                if len(valid)
                else float("nan"),
                "worst_base_mean_pct": float(valid["base_mean_pct"].min())
                if len(valid)
                else float("nan"),
                "median_base_lift_pct": float(
                    valid["base_matched_lift_pct"].median()
                )
                if len(valid)
                else float("nan"),
                "median_stress_mean_pct": float(valid["stress_mean_pct"].median())
                if len(valid)
                else float("nan"),
                "median_observed_given_entry_rate": float(
                    valid["observed_given_entry_rate"].median()
                )
                if len(valid)
                else float("nan"),
            }
        )
    summary = pd.DataFrame(rows)
    summary["robust"] = (
        summary["months_tested"].eq(len(details["month"].unique()))
        & summary["base_lift_positive_months"].eq(len(details["month"].unique()))
        & summary["gross_positive_months"].ge(2)
    )
    return summary.sort_values(
        [
            "robust",
            "base_positive_months",
            "median_base_mean_pct",
            "worst_base_mean_pct",
            "median_base_lift_pct",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def choose_candidate(summary: pd.DataFrame) -> pd.Series | None:
    eligible = summary.loc[summary["robust"]].copy()
    if eligible.empty:
        return None
    return eligible.iloc[0]


def render_report(
    details: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    winner = choose_candidate(summary)
    top = summary.head(15).copy()
    columns = [
        "config",
        "horizon_min",
        "months_tested",
        "min_observed_n",
        "median_selection_rate",
        "gross_positive_months",
        "base_positive_months",
        "base_lift_positive_months",
        "median_gross_mean_pct",
        "median_base_mean_pct",
        "worst_base_mean_pct",
        "median_base_lift_pct",
        "median_stress_mean_pct",
        "median_observed_given_entry_rate",
        "robust",
    ]

    lines = [
        "=== MoneyMaker Candidate v2 Development ===",
        "method=leave-one-month-out across development months; each held-out month is scored using cuts/references fit only on the other months",
        "alpha_families=core2(volatility_15m,signal_bar_range); core3(+prior_15m_low_rebound)",
        "selection_grid=top20%,top10%,top5%",
        "liquidity_gate=fixed 20th-percentile floors for dollar_volume_5m,transactions_5m,active_minute_fraction_15m",
        "execution_gate=modeled base zero-return friction <= none/1.50%/1.00%",
        "optimization_horizons=5m,10m,15m",
        "robust_requirement=positive base matched lift in every held-out month and positive gross mean in >=2 months",
        "",
        "=== Top development configurations ===",
        top.loc[:, columns].to_string(index=False),
        "",
    ]
    if winner is None:
        lines += [
            "=== Candidate v2 decision ===",
            "NO ROBUST WINNER. Do not freeze a v2 rule from this grid.",
        ]
    else:
        config_details = details.loc[
            details["config"].eq(winner["config"])
            & details["horizon_min"].eq(winner["horizon_min"])
        ].copy()
        lines += [
            "=== Candidate v2 provisional winner ===",
            winner.loc[columns].to_string(),
            "",
            "=== Winner month-by-month ===",
            config_details[
                [
                    "month",
                    "selected_n",
                    "selection_rate",
                    "observed_n",
                    "observed_given_entry_rate",
                    "gross_mean_pct",
                    "base_mean_pct",
                    "stress_mean_pct",
                    "base_matched_lift_pct",
                    "base_p05_pct",
                ]
            ].to_string(index=False),
            "",
            "This is a development winner only. Freeze it before testing December 2025.",
        ]
    return "\n".join(lines)


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("dataset label must not be empty")
    return label, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m victory_trader.candidate_v2_research")
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
        help="Development dataset as LABEL=PATH; repeat for each month",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    monthly_frames = {
        label: load_event_dataset(path)
        for label, path in args.dataset
    }
    details = run_leave_one_month_out(monthly_frames)
    summary = summarize_configs(details)
    report = render_report(details, summary)
    print(report)

    for path in (args.report, args.details_csv, args.summary_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
