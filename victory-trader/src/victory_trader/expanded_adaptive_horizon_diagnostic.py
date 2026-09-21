from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HORIZONS = (1, 2, 5, 10, 15, 30)
PRIMARY_POLICY = "filing_semantic_split_supply_hurdle_ev_15m_cap1"
COMPARATOR_POLICY = "split_supply_hurdle_ev_15m_cap1"
BROAD_POLICY = "feasible_earliest_15m_cap1"
POLICIES = (BROAD_POLICY, COMPARATOR_POLICY, PRIMARY_POLICY)


def _return_column(horizon: int, scenario: str) -> str:
    if scenario == "gross":
        return f"buy_return_{horizon}m_pct"
    return f"buy_return_{horizon}m_{scenario}_net_return_pct"


def _day_balanced(frame: pd.DataFrame, values: pd.Series) -> float:
    valid = pd.DataFrame(
        {"trading_day": frame["trading_day"].astype(str), "value": values}
    ).dropna()
    if valid.empty:
        return float("nan")
    return float(valid.groupby("trading_day")["value"].mean().mean())


def _safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _safe_quantile(values: pd.Series, q: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else float("nan")


JOIN_KEYS = ("trading_day", "ticker", "t")


def _return_columns() -> tuple[str, ...]:
    return tuple(
        _return_column(horizon, scenario)
        for horizon in HORIZONS
        for scenario in ("gross", "base", "stress")
    )


def _normalize_join_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["trading_day"] = result["trading_day"].astype(str)
    result["ticker"] = result["ticker"].astype(str)
    result["t"] = pd.to_numeric(result["t"], errors="raise").astype("int64")
    return result


def _attach_state_labels(
    frame: pd.DataFrame,
    state_paths: list[Path],
) -> pd.DataFrame:
    if not state_paths:
        return frame

    needed = [column for column in _return_columns() if column not in frame.columns]
    if not needed:
        return frame

    label_parts: list[pd.DataFrame] = []
    for path in state_paths:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema.names)
        required = set(JOIN_KEYS) | set(needed)
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"state panel {path} missing v3.1 columns: {missing}")
        label_parts.append(pd.read_parquet(path, columns=[*JOIN_KEYS, *needed]))

    labels = _normalize_join_keys(pd.concat(label_parts, ignore_index=True))
    if labels.duplicated(list(JOIN_KEYS)).any():
        raise ValueError("duplicate state-panel keys in v3.1 label source")

    left = _normalize_join_keys(frame)
    merged = left.merge(
        labels,
        how="left",
        on=list(JOIN_KEYS),
        validate="many_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        examples = merged.loc[unmatched, list(JOIN_KEYS)].head(10).to_dict("records")
        raise ValueError(f"v3.1 state-label join missed trade keys: {examples}")
    return merged.drop(columns="_merge")


def _validate(frame: pd.DataFrame) -> None:
    required = {"policy", *JOIN_KEYS, *_return_columns()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"v3.1 input missing required columns: {missing}")


def load_trades(paths: list[Path], state_paths: list[Path] | None = None) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("no v3.0 trade files supplied")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame = frame.loc[frame["policy"].isin(POLICIES)].copy()
    frame = _attach_state_labels(frame, state_paths or [])
    _validate(frame)
    frame["month"] = frame["trading_day"].astype(str).str.slice(0, 7)
    key = ["policy", *JOIN_KEYS]
    duplicated = frame.duplicated(key, keep=False)
    if duplicated.any():
        raise ValueError(
            "duplicate v3.0 evaluated trade keys: "
            f"{frame.loc[duplicated, key].head(10).to_dict('records')}"
        )
    return frame


def horizon_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (month, policy), group in frame.groupby(["month", "policy"], sort=True):
        for horizon in HORIZONS:
            gross = pd.to_numeric(
                group[_return_column(horizon, "gross")], errors="coerce"
            )
            base = pd.to_numeric(
                group[_return_column(horizon, "base")], errors="coerce"
            )
            stress = pd.to_numeric(
                group[_return_column(horizon, "stress")], errors="coerce"
            )
            valid = base.notna() & gross.notna()
            if not valid.any():
                continue
            g = group.loc[valid]
            gross_v = gross.loc[valid]
            base_v = base.loc[valid]
            stress_v = stress.loc[valid]
            rows.append(
                {
                    "month": month,
                    "policy": policy,
                    "horizon_min": horizon,
                    "evaluated": int(valid.sum()),
                    "gross_mean_pct": _safe_mean(gross_v),
                    "base_mean_pct": _safe_mean(base_v),
                    "stress_mean_pct": _safe_mean(stress_v),
                    "modeled_base_cost_drag_mean_pct": _safe_mean(
                        gross_v - base_v
                    ),
                    "base_median_pct": _safe_quantile(base_v, 0.5),
                    "base_p05_pct": _safe_quantile(base_v, 0.05),
                    "base_positive_rate": float((base_v > 0).mean()),
                    "day_balanced_base_mean_pct": _day_balanced(g, base_v),
                    "gross_positive_base_nonpositive_rate": float(
                        ((gross_v > 0) & (base_v <= 0)).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def path_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (month, policy), group in frame.groupby(["month", "policy"], sort=True):
        base_matrix = pd.DataFrame(
            {
                horizon: pd.to_numeric(
                    group[_return_column(horizon, "base")], errors="coerce"
                )
                for horizon in HORIZONS
            },
            index=group.index,
        )
        gross_matrix = pd.DataFrame(
            {
                horizon: pd.to_numeric(
                    group[_return_column(horizon, "gross")], errors="coerce"
                )
                for horizon in HORIZONS
            },
            index=group.index,
        )
        base15 = base_matrix[15]
        gross15 = gross_matrix[15]
        valid15 = base15.notna() & gross15.notna()
        if not valid15.any():
            continue

        available_counts = base_matrix.notna().sum(axis=1)
        oracle_base = base_matrix.max(axis=1, skipna=True)
        oracle_gross = gross_matrix.max(axis=1, skipna=True)
        any_base_positive = (base_matrix > 0).any(axis=1)
        any_gross_positive = (gross_matrix > 0).any(axis=1)

        fixed15_positive = valid15 & (base15 > 0)
        horizon_miss = valid15 & (base15 <= 0) & any_base_positive
        candidate_miss = valid15 & (base15 <= 0) & (~any_base_positive)
        nonpositive15 = valid15 & (base15 <= 0)

        oracle_valid = valid15 & oracle_base.notna()
        oracle_day_balanced = _day_balanced(
            group.loc[oracle_valid], oracle_base.loc[oracle_valid]
        )

        best_horizon_counts = {h: 0 for h in HORIZONS}
        for idx in group.index[oracle_valid]:
            row = base_matrix.loc[idx]
            finite = row.dropna()
            if finite.empty:
                continue
            best_value = finite.max()
            best_horizon = min(
                int(h) for h, value in finite.items() if value == best_value
            )
            best_horizon_counts[best_horizon] += 1

        n = int(valid15.sum())
        nonpositive_n = int(nonpositive15.sum())
        row: dict[str, object] = {
            "month": month,
            "policy": policy,
            "trades_with_15m": n,
            "trades_with_all_horizons": int(
                (available_counts.loc[valid15] == len(HORIZONS)).sum()
            ),
            "mean_available_horizons": float(
                available_counts.loc[valid15].mean()
            ),
            "fixed15_gross_mean_pct": _safe_mean(gross15.loc[valid15]),
            "fixed15_base_mean_pct": _safe_mean(base15.loc[valid15]),
            "oracle_gross_mean_pct": _safe_mean(oracle_gross.loc[oracle_valid]),
            "oracle_base_mean_pct": _safe_mean(oracle_base.loc[oracle_valid]),
            "oracle_base_day_balanced_mean_pct": oracle_day_balanced,
            "oracle_base_lift_vs_15m_pct": _safe_mean(
                oracle_base.loc[oracle_valid] - base15.loc[oracle_valid]
            ),
            "fixed15_positive_rate": float(fixed15_positive.sum() / n),
            "horizon_miss_rate": float(horizon_miss.sum() / n),
            "candidate_miss_rate": float(candidate_miss.sum() / n),
            "horizon_miss_rate_among_15m_nonpositive": (
                float(horizon_miss.sum() / nonpositive_n)
                if nonpositive_n
                else float("nan")
            ),
            "any_base_positive_rate": float(
                any_base_positive.loc[valid15].mean()
            ),
            "any_gross_positive_rate": float(
                any_gross_positive.loc[valid15].mean()
            ),
            "any_gross_positive_no_base_positive_rate": float(
                (
                    any_gross_positive.loc[valid15]
                    & (~any_base_positive.loc[valid15])
                ).mean()
            ),
            "fixed15_gross_positive_base_nonpositive_rate": float(
                ((gross15.loc[valid15] > 0) & (base15.loc[valid15] <= 0)).mean()
            ),
        }
        for horizon in HORIZONS:
            row[f"oracle_best_{horizon}m_rate"] = (
                best_horizon_counts[horizon] / n if n else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def render_report(
    horizons: pd.DataFrame, paths: pd.DataFrame
) -> str:
    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "(no rows)"
        return frame.to_string(index=False)

    primary_horizons = horizons.loc[horizons["policy"].eq(PRIMARY_POLICY)]
    primary_paths = paths.loc[paths["policy"].eq(PRIMARY_POLICY)]
    return "\n".join(
        [
            "=== MoneyMaker Adaptive-Horizon Diagnostic v3.1 ===",
            "scope=January-March 2026 development results from frozen v3.0 evaluated anchors",
            "horizons=1,2,5,10,15,30 minutes; future outcomes are diagnostic only",
            "oracle=per-trade hindsight ceiling, never an executable policy",
            "question=separate bad-candidate failures from fixed-15m horizon failures",
            "",
            "=== All policies: fixed-horizon gross/base/stress ===",
            table(horizons),
            "",
            "=== All policies: path / horizon opportunity decomposition ===",
            table(paths),
            "",
            "=== Primary v3.0 semantic tail only ===",
            table(primary_horizons),
            "",
            table(primary_paths),
            "",
            "Interpretation:",
            "- horizon_miss: fixed 15m BASE <= 0 but at least one other frozen horizon BASE > 0.",
            "- candidate_miss: fixed 15m BASE <= 0 and no frozen horizon has positive BASE.",
            "- oracle statistics use future returns and measure only an upper bound on horizon-choice value.",
            "- no horizon, threshold, feature, or policy is promoted by this diagnostic.",
        ]
    )


def run_diagnostic(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    horizons = horizon_summary(frame)
    paths = path_diagnostics(frame)
    report = render_report(horizons, paths)
    return horizons, paths, report


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.expanded_adaptive_horizon_diagnostic"
    )
    parser.add_argument("trades", nargs="+", type=Path)
    parser.add_argument("--state", nargs="+", type=Path, default=[])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--horizon-csv", type=Path, required=True)
    parser.add_argument("--path-csv", type=Path, required=True)
    args = parser.parse_args()

    frame = load_trades(args.trades, args.state)
    horizons, paths, report = run_diagnostic(frame)
    print(report)

    for path in (args.report, args.horizon_csv, args.path_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    horizons.to_csv(args.horizon_csv, index=False)
    paths.to_csv(args.path_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
