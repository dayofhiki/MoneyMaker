"""Request266: causal best-later-exit opportunity ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .causal_minute_controller_rebuild import causal_execution_scan
from .causal_opportunity_ceiling import (
    fit_candidate_models,
)
from .full_hot_fixed_policy_value import (
    attach_fixed_value,
    event_lookup,
    xframe,
)
from .learned_pullback_entry import TEST_DAYS, build_episode_states
from .recurrent_wait_entry_action_value import _base_return

REQUEST_ID = 266
ENTRY_WINDOW_MINUTES = 10
EXIT_CAP_MINUTES = 30


def attach_best_later_exit(
    states: pd.DataFrame,
    causal_scan: pd.DataFrame,
) -> pd.DataFrame:
    lookup = event_lookup(causal_scan)
    result = states.copy()
    result["best_later_exit_value_pct"] = np.nan
    result["best_later_exit_t"] = np.nan

    for index, row in result.iterrows():
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        hot_t = int(row["hot_t"])
        state_t = int(row["state_t"])
        entry = float(row["state_price"])
        cap_t = hot_t + EXIT_CAP_MINUTES * 60_000
        future = [
            (int(t), float(price))
            for t, price in lookup.get((day, ticker), [])
            if state_t < int(t) <= cap_t and float(price) > 0
        ]
        if not future:
            continue
        values = np.asarray(
            [_base_return(entry, price) for _, price in future],
            dtype=float,
        )
        finite = np.isfinite(values)
        if not finite.any():
            continue
        valid_indices = np.flatnonzero(finite)
        best_local = valid_indices[int(np.argmax(values[finite]))]
        result.at[index, "best_later_exit_value_pct"] = float(
            values[best_local]
        )
        result.at[index, "best_later_exit_t"] = int(
            future[best_local][0]
        )
    return result


def episode_ceiling(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in states.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        ordered = group.sort_values("state_t")
        fixed = pd.to_numeric(
            ordered.enter_value_pct,
            errors="coerce",
        )
        dynamic = pd.to_numeric(
            ordered.best_later_exit_value_pct,
            errors="coerce",
        )
        usable_dynamic = ordered.loc[dynamic.notna()].copy()

        first_dynamic = (
            float(usable_dynamic.iloc[0].best_later_exit_value_pct)
            if len(usable_dynamic)
            else np.nan
        )

        drawdown = pd.to_numeric(
            ordered.drawdown_from_running_high_pct,
            errors="coerce",
        )
        pullback = ordered.loc[drawdown.le(-2.0)]
        if len(pullback):
            pb = pullback.iloc[0]
            pullback_entered = True
            pb_dynamic = pd.to_numeric(
                pd.Series([pb.best_later_exit_value_pct]),
                errors="coerce",
            ).iloc[0]
        else:
            pullback_entered = False
            pb_dynamic = 0.0

        rows.append(
            {
                "trading_day": str(key[0]),
                "ticker": str(key[1]).upper(),
                "hot_t": int(key[2]),
                "first_entry_best_exit_pct": first_dynamic,
                "pullback_entered": pullback_entered,
                "pullback_2_best_exit_pct": (
                    float(pb_dynamic)
                    if pd.notna(pb_dynamic)
                    else np.nan
                ),
                "fixed_exit_oracle_pct": (
                    float(fixed.max())
                    if fixed.notna().any()
                    else np.nan
                ),
                "joint_entry_exit_oracle_pct": (
                    float(dynamic.max())
                    if dynamic.notna().any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def metric(series: pd.Series) -> dict:
    value = pd.to_numeric(series, errors="coerce")
    valid = value.dropna()
    return {
        "coverage": float(value.notna().mean()) if len(value) else 0.0,
        "rows": int(len(valid)),
        "mean_pct": float(valid.mean()) if len(valid) else None,
        "median_pct": float(valid.median()) if len(valid) else None,
        "positive_rate": (
            float(valid.gt(0).mean()) if len(valid) else None
        ),
        "severe_loss_rate_le_minus2": (
            float(valid.le(-2).mean()) if len(valid) else None
        ),
    }


def report(frame: pd.DataFrame) -> dict:
    return {
        "episodes": int(len(frame)),
        "first_entry_best_exit": metric(
            frame.first_entry_best_exit_pct
        ),
        "pullback_2_best_exit": metric(
            frame.pullback_2_best_exit_pct
        ),
        "fixed_exit_oracle": metric(
            frame.fixed_exit_oracle_pct
        ),
        "joint_entry_exit_oracle": metric(
            frame.joint_entry_exit_oracle_pct
        ),
        "pullback_entry_rate": float(
            frame.pullback_entered.mean()
        ) if len(frame) else 0.0,
    }


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
) -> int:
    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    causal_scan = causal_execution_scan(scan)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, causal_scan)

    reg, cls, columns, fit_support = fit_candidate_models(first_hot)
    del reg
    test_hot = first_hot.loc[
        first_hot.trading_day.astype(str).isin(TEST_DAYS)
    ].copy()
    probability = cls.predict_proba(
        xframe(test_hot, columns)
    )[:, 1]
    test_hot["candidate_probability"] = probability
    top_threshold = float(np.quantile(probability, 0.80))
    top_hot = test_hot.loc[
        test_hot.candidate_probability.ge(top_threshold)
    ].copy()

    all_states = build_episode_states(test_hot, causal_scan)
    all_states = attach_best_later_exit(all_states, causal_scan)
    all_episodes = episode_ceiling(all_states)

    top_keys = set(
        zip(
            top_hot.trading_day.astype(str),
            top_hot.ticker.astype(str).str.upper(),
            pd.to_numeric(top_hot.t, errors="coerce").astype("int64"),
        )
    )
    top_mask = [
        (str(day), str(ticker).upper(), int(hot_t)) in top_keys
        for day, ticker, hot_t in all_episodes.loc[
            :, ["trading_day", "ticker", "hot_t"]
        ].itertuples(index=False, name=None)
    ]
    top_episodes = all_episodes.loc[top_mask].copy()

    all_report = report(all_episodes)
    top_report = report(top_episodes)
    fixed_mean = all_report["fixed_exit_oracle"]["mean_pct"]
    dynamic_mean = all_report["joint_entry_exit_oracle"]["mean_pct"]
    top_dynamic = top_report["joint_entry_exit_oracle"]["mean_pct"]

    exit_contract_bottleneck = bool(
        fixed_mean is not None
        and fixed_mean <= 0
        and dynamic_mean is not None
        and dynamic_mean > 0
    )
    top20_dynamic_strong = bool(
        top_dynamic is not None and top_dynamic >= 1.0
    )

    if exit_contract_bottleneck and top20_dynamic_strong:
        diagnosis = "candidate_signal_good_exit_contract_major_bottleneck"
    elif exit_contract_bottleneck:
        diagnosis = "exit_contract_major_bottleneck"
    elif dynamic_mean is not None and dynamic_mean <= 0:
        diagnosis = "attention_population_lacks_price_expansion"
    else:
        diagnosis = "mixed_opportunity_structure"

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "causal_reference_contract": True,
        "candidate_fit_support": fit_support,
        "top20_probability_threshold": top_threshold,
        "all_full_hot": all_report,
        "candidate_top20": top_report,
        "exit_contract_bottleneck": exit_contract_bottleneck,
        "top20_dynamic_oracle_strong": top20_dynamic_strong,
        "diagnosis": diagnosis,
        "interpretation": (
            "Best-later-exit and joint entry/exit results are hindsight "
            "diagnostic ceilings only. They are not deployable policies."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-hot", type=Path, required=True)
    parser.add_argument("--opportunity-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.first_hot,
        args.opportunity_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
