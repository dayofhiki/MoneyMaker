"""Economic diagnostics for first-HOT entry episodes.

This module contains only deterministic labeling/summarization helpers. It does
not select a trading policy or open a new evaluation block by itself.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .attention_replay import MINUTE_MS
from .execution_costs import (
    DEFAULT_EXECUTION_SCENARIOS,
    ExecutionScenario,
    net_round_trip_return_pct,
)

HORIZONS = (1, 2, 5, 10, 15, 30)


def first_hot_events(trace: pd.DataFrame) -> pd.DataFrame:
    hot = trace.loc[trace["state"].astype(str).eq("hot")].copy()
    if hot.empty:
        return hot
    return (
        hot.sort_values(["trading_day", "ticker", "t"], kind="stable")
        .groupby(["trading_day", "ticker"], sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def _price_lookup(scan: pd.DataFrame) -> dict[tuple[str, str, int], dict[str, float]]:
    needed = ["trading_day", "ticker", "t", "o"]
    missing = set(needed) - set(scan.columns)
    if missing:
        raise ValueError(f"scan missing entry-economics columns: {sorted(missing)}")
    result: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in scan.itertuples(index=False):
        opening = pd.to_numeric(pd.Series([getattr(row, "o")]), errors="coerce").iloc[0]
        if pd.isna(opening) or float(opening) <= 0:
            continue
        item = {"o": float(opening)}
        if hasattr(row, "return_from_previous_close_pct"):
            value = pd.to_numeric(
                pd.Series([getattr(row, "return_from_previous_close_pct")]),
                errors="coerce",
            ).iloc[0]
            item["return_from_previous_close_pct"] = (
                float(value) if pd.notna(value) else np.nan
            )
        result[
            (str(row.trading_day), str(row.ticker).upper(), int(row.t))
        ] = item
    return result


def promotion_hot_events(trace: pd.DataFrame) -> pd.DataFrame:
    promoted = trace.loc[
        trace["state"].astype(str).eq("hot")
        & trace.get("reason", pd.Series(index=trace.index, dtype=object))
        .astype(str)
        .eq("learned_hot_promote")
    ].copy()
    if promoted.empty:
        return promoted

    promoted = promoted.sort_values(
        ["trading_day", "ticker", "t"], kind="stable"
    ).reset_index(drop=True)
    promoted["promotion_index"] = (
        promoted.groupby(["trading_day", "ticker"], sort=False).cumcount() + 1
    )
    previous_t = promoted.groupby(
        ["trading_day", "ticker"], sort=False
    )["t"].shift()
    promoted["minutes_since_previous_promotion"] = (
        pd.to_numeric(promoted["t"], errors="coerce")
        - pd.to_numeric(previous_t, errors="coerce")
    ) / MINUTE_MS
    promoted["is_repromotion"] = promoted["promotion_index"].gt(1)
    return promoted


def label_hot_event_economics(
    events: pd.DataFrame,
    scan: pd.DataFrame,
    *,
    horizons: Iterable[int] = HORIZONS,
    scenarios: tuple[ExecutionScenario, ...] = DEFAULT_EXECUTION_SCENARIOS,
) -> pd.DataFrame:
    lookup = _price_lookup(scan)
    rows: list[dict[str, object]] = []

    passthrough = (
        "attention_score",
        "learned_hot_score",
        "reason",
        "promotion_index",
        "minutes_since_previous_promotion",
        "is_repromotion",
    )

    for event in events.itertuples(index=False):
        day = str(event.trading_day)
        ticker = str(event.ticker).upper()
        decision_t = int(event.t)
        entry_key = (day, ticker, decision_t + MINUTE_MS)
        entry = lookup.get(entry_key)
        record: dict[str, object] = {
            "trading_day": day,
            "ticker": ticker,
            "hot_t": decision_t,
            "entry_reference_t": decision_t + MINUTE_MS,
            "entry_price": entry["o"] if entry is not None else np.nan,
            "entry_reference_available": entry is not None,
            "entry_return_from_previous_close_pct": (
                entry.get("return_from_previous_close_pct", np.nan)
                if entry is not None
                else np.nan
            ),
        }
        for column in passthrough:
            if hasattr(event, column):
                record[column] = getattr(event, column)

        oracle_base: list[tuple[int, float]] = []
        if entry is not None:
            entry_price = float(entry["o"])
            for minute in range(1, 31):
                exit_key = (
                    day,
                    ticker,
                    decision_t + (minute + 1) * MINUTE_MS,
                )
                exit_row = lookup.get(exit_key)
                if exit_row is None:
                    continue
                gross = (float(exit_row["o"]) / entry_price - 1.0) * 100.0
                base = net_round_trip_return_pct(
                    entry_price,
                    gross,
                    next(s for s in scenarios if s.name == "base"),
                )
                oracle_base.append((minute, base))

            for horizon in horizons:
                exit_key = (
                    day,
                    ticker,
                    decision_t + (int(horizon) + 1) * MINUTE_MS,
                )
                exit_row = lookup.get(exit_key)
                gross_key = f"return_{horizon}m_gross_pct"
                record[gross_key] = np.nan
                for scenario in scenarios:
                    record[f"return_{horizon}m_{scenario.name}_net_pct"] = np.nan
                if exit_row is None:
                    continue
                gross = (float(exit_row["o"]) / entry_price - 1.0) * 100.0
                record[gross_key] = gross
                for scenario in scenarios:
                    record[f"return_{horizon}m_{scenario.name}_net_pct"] = (
                        net_round_trip_return_pct(
                            entry_price,
                            gross,
                            scenario,
                        )
                    )

        if oracle_base:
            best_minute, best_base = max(oracle_base, key=lambda pair: pair[1])
            record["oracle_best_base_pct"] = float(best_base)
            record["oracle_best_minute"] = int(best_minute)
            record["oracle_base_positive"] = bool(best_base > 0)
        else:
            record["oracle_best_base_pct"] = np.nan
            record["oracle_best_minute"] = np.nan
            record["oracle_base_positive"] = pd.NA
        rows.append(record)

    return pd.DataFrame(rows)


def label_first_hot_economics(
    trace: pd.DataFrame,
    scan: pd.DataFrame,
    *,
    horizons: Iterable[int] = HORIZONS,
    scenarios: tuple[ExecutionScenario, ...] = DEFAULT_EXECUTION_SCENARIOS,
) -> pd.DataFrame:
    return label_hot_event_economics(
        first_hot_events(trace),
        scan,
        horizons=horizons,
        scenarios=scenarios,
    )


def summarize_hot_economics(
    labeled: pd.DataFrame,
    *,
    horizons: Iterable[int] = HORIZONS,
) -> dict[str, object]:
    episodes = int(len(labeled))
    valid_entry = labeled["entry_reference_available"].fillna(False).astype(bool)
    summary: dict[str, object] = {
        "first_hot_episodes": episodes,
        "valid_entry_references": int(valid_entry.sum()),
        "entry_reference_coverage": (
            float(valid_entry.mean()) if episodes else None
        ),
    }

    horizon_metrics: dict[str, object] = {}
    for horizon in horizons:
        gross = pd.to_numeric(
            labeled[f"return_{horizon}m_gross_pct"], errors="coerce"
        )
        base = pd.to_numeric(
            labeled[f"return_{horizon}m_base_net_pct"], errors="coerce"
        )
        stress = pd.to_numeric(
            labeled[f"return_{horizon}m_stress_net_pct"], errors="coerce"
        )
        valid = base.notna()
        horizon_metrics[str(horizon)] = {
            "observed": int(valid.sum()),
            "coverage": float(valid.mean()) if episodes else None,
            "gross_mean_pct": float(gross.loc[valid].mean()) if valid.any() else None,
            "base_mean_pct": float(base.loc[valid].mean()) if valid.any() else None,
            "base_median_pct": float(base.loc[valid].median()) if valid.any() else None,
            "base_positive_rate": (
                float(base.loc[valid].gt(0).mean()) if valid.any() else None
            ),
            "base_p05_pct": (
                float(base.loc[valid].quantile(0.05)) if valid.any() else None
            ),
            "stress_mean_pct": (
                float(stress.loc[valid].mean()) if valid.any() else None
            ),
        }
    summary["fixed_horizons"] = horizon_metrics

    oracle = pd.to_numeric(labeled["oracle_best_base_pct"], errors="coerce")
    valid_oracle = oracle.notna()
    best_minute = pd.to_numeric(labeled["oracle_best_minute"], errors="coerce")
    summary["oracle_30m"] = {
        "evaluable": int(valid_oracle.sum()),
        "coverage": float(valid_oracle.mean()) if episodes else None,
        "base_mean_pct": (
            float(oracle.loc[valid_oracle].mean()) if valid_oracle.any() else None
        ),
        "base_median_pct": (
            float(oracle.loc[valid_oracle].median()) if valid_oracle.any() else None
        ),
        "base_positive_rate": (
            float(oracle.loc[valid_oracle].gt(0).mean())
            if valid_oracle.any()
            else None
        ),
        "base_p05_pct": (
            float(oracle.loc[valid_oracle].quantile(0.05))
            if valid_oracle.any()
            else None
        ),
        "median_best_minute": (
            float(best_minute.loc[valid_oracle].median())
            if valid_oracle.any()
            else None
        ),
    }

    by_day: dict[str, object] = {}
    for day, group in labeled.groupby("trading_day", sort=True):
        day_oracle = pd.to_numeric(group["oracle_best_base_pct"], errors="coerce")
        valid = day_oracle.notna()
        by_day[str(day)] = {
            "episodes": int(len(group)),
            "entry_coverage": float(
                group["entry_reference_available"].fillna(False).astype(bool).mean()
            ),
            "oracle_evaluable": int(valid.sum()),
            "oracle_base_mean_pct": (
                float(day_oracle.loc[valid].mean()) if valid.any() else None
            ),
            "oracle_base_positive_rate": (
                float(day_oracle.loc[valid].gt(0).mean()) if valid.any() else None
            ),
        }
    summary["by_day"] = by_day
    return summary
