from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import load_event_dataset
from .edge_discovery import summarize_chronological_feature_holdout
from .extended_edge_discovery import (
    DISCOVERY_FEATURES,
    _availability_alias,
    _gross_alias,
    summarize_outcome_missingness_reasons,
)


DEFAULT_HORIZONS = (1, 2, 5, 10, 15, 30, 60)


def _bh_qvalues(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = pd.to_numeric(p_values, errors="coerce").dropna()
    if valid.empty:
        return result
    ordered = valid.sort_values()
    m = len(ordered)
    raw = ordered.to_numpy(dtype=float) * m / np.arange(1, m + 1, dtype=float)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


def collect_multi_horizon_holdouts(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for horizon in horizons:
        for target_kind, target_frame, scenario in (
            ("gross", _gross_alias(frame, horizon), "gross"),
            ("base_net", frame, "base"),
            ("availability", _availability_alias(frame, horizon), "availability"),
        ):
            result = summarize_chronological_feature_holdout(
                target_frame,
                horizon_min=horizon,
                scenario=scenario,
                features=DISCOVERY_FEATURES,
            )
            if result.empty:
                continue
            result = result.copy()
            result.insert(0, "target_kind", target_kind)
            result.insert(1, "horizon_min", horizon)
            rows.append(result)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    combined["global_train_q_value_bh"] = np.nan
    for target_kind, indices in combined.groupby("target_kind").groups.items():
        combined.loc[indices, "global_train_q_value_bh"] = _bh_qvalues(
            combined.loc[indices, "train_p_value"]
        )
    return combined


def summarize_cross_horizon_stability(
    holdouts: pd.DataFrame,
    *,
    target_kind: str,
) -> pd.DataFrame:
    subset = holdouts.loc[holdouts["target_kind"] == target_kind].copy()
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for feature, group in subset.groupby("feature", sort=False):
        usable = group.loc[group["enough_sample"].eq(1)].copy()
        if usable.empty:
            continue
        directions = usable["train_direction"].value_counts()
        dominant = str(directions.index[0])
        direction_consistency = float(directions.iloc[0] / len(usable))
        train_sig = usable["global_train_q_value_bh"].le(0.05)
        positive_lift = pd.to_numeric(
            usable["test_adjusted_lift_pct"], errors="coerce"
        ).gt(0)
        positive_mean = pd.to_numeric(usable["test_mean_pct"], errors="coerce").gt(0)
        robust = train_sig & positive_lift

        rows.append(
            {
                "feature": feature,
                "target_kind": target_kind,
                "horizons_tested": len(usable),
                "dominant_direction": dominant,
                "direction_consistency": direction_consistency,
                "global_train_significant_horizons": int(train_sig.sum()),
                "positive_holdout_lift_horizons": int(positive_lift.sum()),
                "positive_holdout_mean_horizons": int(positive_mean.sum()),
                "robust_horizons": int(robust.sum()),
                "median_holdout_lift_pct": float(
                    pd.to_numeric(
                        usable["test_adjusted_lift_pct"], errors="coerce"
                    ).median()
                ),
                "worst_holdout_lift_pct": float(
                    pd.to_numeric(
                        usable["test_adjusted_lift_pct"], errors="coerce"
                    ).min()
                ),
                "best_holdout_lift_pct": float(
                    pd.to_numeric(
                        usable["test_adjusted_lift_pct"], errors="coerce"
                    ).max()
                ),
                "median_test_mean_pct": float(
                    pd.to_numeric(usable["test_mean_pct"], errors="coerce").median()
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        [
            "robust_horizons",
            "positive_holdout_lift_horizons",
            "direction_consistency",
            "median_holdout_lift_pct",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def summarize_horizon_coverage(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    total = len(frame)
    executable = pd.to_numeric(frame["entry_price"], errors="coerce").notna()
    for horizon in horizons:
        target = f"return_{horizon}m_pct"
        observed = pd.to_numeric(frame[target], errors="coerce").notna()
        overall, _ = summarize_outcome_missingness_reasons(
            frame, horizon_min=horizon
        )
        counts = (
            dict(zip(overall["reason"], overall["n"], strict=True))
            if not overall.empty
            else {}
        )
        rows.append(
            {
                "horizon_min": horizon,
                "events": total,
                "executable_entries": int(executable.sum()),
                "observed_outcomes": int(observed.sum()),
                "observed_rate": float(observed.mean()) if total else float("nan"),
                "observed_given_entry_rate": float(observed.sum() / executable.sum())
                if executable.any()
                else float("nan"),
                "entry_missing_bar": int(counts.get("entry_missing_bar", 0)),
                "entry_session_close": int(counts.get("entry_session_close", 0)),
                "horizon_missing_bar": int(counts.get("horizon_missing_bar", 0)),
                "horizon_session_close": int(counts.get("horizon_session_close", 0)),
            }
        )
    return pd.DataFrame(rows)


def _table(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False) if not frame.empty else "No eligible rows."


def render_multi_horizon_report(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    holdouts = collect_multi_horizon_holdouts(frame, horizons=horizons)
    gross_stability = summarize_cross_horizon_stability(
        holdouts, target_kind="gross"
    )
    net_stability = summarize_cross_horizon_stability(
        holdouts, target_kind="base_net"
    )
    availability_stability = summarize_cross_horizon_stability(
        holdouts, target_kind="availability"
    )
    coverage = summarize_horizon_coverage(frame, horizons=horizons)

    report = "\n".join(
        [
            "=== MoneyMaker Multi-Horizon Discovery ===",
            f"horizons={','.join(str(h) for h in horizons)}",
            f"pre_specified_features={len(DISCOVERY_FEATURES)}",
            "",
            "=== Horizon coverage / exact-bar availability ===",
            _table(coverage),
            "",
            "=== GROSS alpha cross-horizon stability ===",
            _table(gross_stability),
            "",
            "=== BASE-NET cross-horizon stability ===",
            _table(net_stability),
            "",
            "=== EXECUTABILITY / availability cross-horizon stability ===",
            _table(availability_stability),
            "",
            "Interpretation rule 1: alpha priority comes from GROSS stability; BASE-NET-only signals may be execution-cost mechanics.",
            "Interpretation rule 2: global_train_q_value_bh corrects across every feature x horizon test within each target family.",
            "Interpretation rule 3: robust_horizons requires both globally significant training evidence and positive chronological holdout lift.",
            "Interpretation rule 4: a single attractive horizon is not promoted unless behavior repeats across horizons and later dates.",
        ]
    )
    stability = pd.concat(
        [gross_stability, net_stability, availability_stability],
        ignore_index=True,
    )
    return report, holdouts, stability


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.multi_horizon_discovery"
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--stability-csv", type=Path, required=True)
    args = parser.parse_args()

    frame = load_event_dataset(args.path)
    report, holdouts, stability = render_multi_horizon_report(frame)
    print(report)

    for path in (args.output, args.details_csv, args.stability_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    holdouts.to_csv(args.details_csv, index=False)
    stability.to_csv(args.stability_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
