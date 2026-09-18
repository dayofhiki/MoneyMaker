from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .state_short_volume_enrichment import SHORT_VOLUME_FEATURES
from .state_value_model import _eligible


RELATIVE_SHORT_VOLUME_FEATURES = (
    "runner_other_mean_short_ratio_latest_prior",
    "runner_short_ratio_latest_minus_other_mean",
    "runner_short_ratio_latest_percentile",
    "runner_other_mean_short_ratio_mean5_prior",
    "runner_short_ratio_mean5_minus_other_mean",
    "runner_short_ratio_mean5_percentile",
    "runner_short_volume_latest_vs_mean5_minus_other_mean",
    "runner_finra_total_volume_latest_vs_mean5_minus_other_mean",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _leave_one_out_mean(
    values: pd.Series,
    group_sum: pd.Series,
    group_count: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    own_valid = numeric.notna().astype(float)
    denominator = group_count - own_valid
    numerator = group_sum - numeric.fillna(0.0)
    return (numerator / denominator).where(denominator > 0)


def _percentile_with_min_two(
    values: pd.Series,
    group_size: pd.Series,
    keys: list[str],
) -> pd.Series:
    ranked = values.groupby(
        [values.index.map(lambda idx: idx)],
        sort=False,
    )
    del ranked
    frame = pd.DataFrame(
        {
            "_value": pd.to_numeric(values, errors="coerce"),
            "_group_size": group_size,
        },
        index=values.index,
    )
    # Caller supplies exact (day, timestamp) grouping via temporary columns.
    raise RuntimeError("internal helper should not be called directly")


def enrich_relative_short_volume_context(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "trading_day",
        "t",
        "ticker",
        *SHORT_VOLUME_FEATURES,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "state panel missing short-volume context columns: "
            f"{sorted(missing)}"
        )

    result = frame.copy()
    keys = ["trading_day", "t"]
    result["trading_day"] = result["trading_day"].astype(str)
    result["t"] = pd.to_numeric(result["t"], errors="coerce")

    latest = _numeric(result, "short_ratio_latest_prior")
    mean5 = _numeric(result, "short_ratio_mean5_prior")
    short_vs5 = _numeric(result, "short_volume_latest_vs_mean5")
    total_vs5 = _numeric(result, "finra_total_volume_latest_vs_mean5")

    temporary = pd.DataFrame(
        {
            "trading_day": result["trading_day"],
            "t": result["t"],
            "_latest": latest,
            "_mean5": mean5,
            "_short_vs5": short_vs5,
            "_total_vs5": total_vs5,
        },
        index=result.index,
    )
    grouped = temporary.groupby(keys, sort=False, dropna=False)
    group_size = grouped["t"].transform("size").astype(float)

    specs = {
        "_latest": (
            "runner_other_mean_short_ratio_latest_prior",
            "runner_short_ratio_latest_minus_other_mean",
        ),
        "_mean5": (
            "runner_other_mean_short_ratio_mean5_prior",
            "runner_short_ratio_mean5_minus_other_mean",
        ),
        "_short_vs5": (
            None,
            "runner_short_volume_latest_vs_mean5_minus_other_mean",
        ),
        "_total_vs5": (
            None,
            "runner_finra_total_volume_latest_vs_mean5_minus_other_mean",
        ),
    }

    for source, (mean_output, diff_output) in specs.items():
        values = temporary[source]
        sums = grouped[source].transform("sum")
        counts = grouped[source].transform("count").astype(float)
        other_mean = _leave_one_out_mean(values, sums, counts)
        if mean_output is not None:
            result[mean_output] = other_mean
        result[diff_output] = values - other_mean

    for source, output in (
        ("_latest", "runner_short_ratio_latest_percentile"),
        ("_mean5", "runner_short_ratio_mean5_percentile"),
    ):
        valid_counts = grouped[source].transform("count").astype(float)
        ranks = grouped[source].rank(method="average", pct=True)
        result[output] = ranks.where(valid_counts >= 2)

    for column in RELATIVE_SHORT_VOLUME_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["runner_context_count"] = group_size
    return result


def context_coverage(frame: pd.DataFrame, month: str) -> dict[str, object]:
    scoreable = frame.loc[_eligible(frame)].copy()
    counts = pd.to_numeric(
        scoreable["runner_context_count"], errors="coerce"
    )
    row: dict[str, object] = {
        "month": month,
        "scoreable_rows": int(len(scoreable)),
        "median_runner_context_count": float(counts.median()),
        "rows_ge_2_runners": float(counts.ge(2).mean()),
        "rows_ge_3_runners": float(counts.ge(3).mean()),
        "rows_ge_5_runners": float(counts.ge(5).mean()),
    }
    for column in RELATIVE_SHORT_VOLUME_FEATURES:
        row[f"{column}_coverage"] = float(
            pd.to_numeric(scoreable[column], errors="coerce")
            .notna()
            .mean()
        )
    return row


def success_check(coverage: pd.DataFrame) -> dict[str, bool]:
    if coverage.empty:
        checks = {
            "percentile_coverage_at_least_60pct_every_month": False,
            "difference_coverage_at_least_60pct_every_month": False,
            "median_runner_count_at_least_2_every_month": False,
        }
    else:
        checks = {
            "percentile_coverage_at_least_60pct_every_month": bool(
                coverage[
                    "runner_short_ratio_latest_percentile_coverage"
                ].ge(0.60).all()
            ),
            "difference_coverage_at_least_60pct_every_month": bool(
                coverage[
                    "runner_short_ratio_latest_minus_other_mean_coverage"
                ].ge(0.60).all()
            ),
            "median_runner_count_at_least_2_every_month": bool(
                coverage["median_runner_context_count"].ge(2.0).all()
            ),
        }
    checks["all_pass"] = all(checks.values())
    return checks


def render_report(
    coverage: pd.DataFrame,
    checks: dict[str, bool],
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker Relative Short-Volume Context Probe v1.0 ===",
            "comparison_set=exact contemporaneous runner rows at (trading_day,t)",
            "outcome_blind=true",
            "",
            "=== Coverage ===",
            coverage.to_string(index=False),
            "",
            "=== Pre-registered coverage check ===",
            pd.DataFrame(
                [
                    {"criterion": key, "pass": value}
                    for key, value in checks.items()
                ]
            ).to_string(index=False),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.state_short_volume_context"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, path in args.dataset:
        frame = pd.read_parquet(path)
        enriched = enrich_relative_short_volume_context(frame)
        rows.append(context_coverage(enriched, label))
        enriched.to_parquet(
            args.output_dir / f"state-{label}.parquet",
            index=False,
            compression="zstd",
        )

    coverage = pd.DataFrame(rows)
    checks = success_check(coverage)
    report = render_report(coverage, checks)
    print(report)

    args.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.coverage_csv, index=False)
    args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
