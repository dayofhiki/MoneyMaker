from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .state_action_risk import SEVERE_LOSS_PCT
from .state_action_value import (
    HORIZONS,
    _target_column,
    train_direct_ev_model,
)
from .state_entry_ranker import (
    EPISODE_KEYS,
    EV_GATE_PCT,
    _rank_diagnostics,
    _select_first_trades as _ranker_select_first_trades,
    score_entry_actions,
    train_episode_rank_model,
)


MINUTE_MS = 60_000
MIN_MONTH_TRADES = 15
BOOTSTRAP_SAMPLES = 10_000
POLICIES = ("earliest_ev_cap1", "rank_turn_cap1")


def _select_rank_turn_trades(scored: pd.DataFrame) -> pd.DataFrame:
    trades: list[dict[str, object]] = []

    for _, group in scored.groupby(EPISODE_KEYS, sort=False):
        ordered = group.sort_values("t", kind="stable")
        armed = False
        previous_t: int | None = None
        previous_score = np.nan

        for _, row in ordered.iterrows():
            ev = pd.to_numeric(
                pd.Series(
                    [row.get("rank_action_predicted_base_ev_pct", np.nan)]
                ),
                errors="coerce",
            ).iloc[0]
            score = pd.to_numeric(
                pd.Series([row.get("predicted_rank_score", np.nan)]),
                errors="coerce",
            ).iloc[0]
            t_value = pd.to_numeric(
                pd.Series([row.get("t", np.nan)]), errors="coerce"
            ).iloc[0]

            qualified = (
                pd.notna(ev)
                and float(ev) >= EV_GATE_PCT
                and pd.notna(score)
                and pd.notna(t_value)
            )
            if not qualified:
                armed = False
                previous_t = None
                previous_score = np.nan
                continue

            t = int(t_value)
            current_score = float(score)

            if (
                not armed
                or previous_t is None
                or t - previous_t != MINUTE_MS
            ):
                armed = True
                previous_t = t
                previous_score = current_score
                continue

            if current_score > float(previous_score):
                previous_t = t
                previous_score = current_score
                continue

            # First causal non-increase after a consecutive qualified state is
            # the attempted entry. The attempt is consumed even if unfillable.
            horizon = int(row["rank_action_horizon_min"])
            entry = pd.to_numeric(
                pd.Series([row.get("entry_price", np.nan)]),
                errors="coerce",
            ).iloc[0]
            realized = pd.to_numeric(
                pd.Series([row.get(_target_column(horizon), np.nan)]),
                errors="coerce",
            ).iloc[0]

            if (
                pd.notna(entry)
                and float(entry) > 0
                and pd.notna(realized)
            ):
                item = row.to_dict()
                item["action_horizon_min"] = horizon
                item["selected_predicted_base_ev_pct"] = float(ev)
                item["selected_predicted_rank_score"] = current_score
                item["realized_base_net_return_pct"] = float(realized)
                item["realized_gross_return_pct"] = float(
                    row[f"buy_return_{horizon}m_pct"]
                )
                item["realized_stress_net_return_pct"] = float(
                    row[
                        f"buy_return_{horizon}m_stress_net_return_pct"
                    ]
                )
                trades.append(item)
            break

    return pd.DataFrame(trades)


def _select_baseline(scored: pd.DataFrame) -> pd.DataFrame:
    return _ranker_select_first_trades(
        scored,
        policy="earliest_ev_cap1",
        rank_gate=0.0,
    )


def _metrics(
    trades: pd.DataFrame,
    *,
    month: str,
    policy: str,
) -> dict[str, object]:
    if trades.empty:
        return {"month": month, "policy": policy, "trades": 0}

    base = pd.to_numeric(
        trades["realized_base_net_return_pct"], errors="coerce"
    )
    gross = pd.to_numeric(
        trades["realized_gross_return_pct"], errors="coerce"
    )
    stress = pd.to_numeric(
        trades["realized_stress_net_return_pct"], errors="coerce"
    )
    daily = trades.assign(_base=base).groupby("trading_day")["_base"].mean()
    horizons = pd.to_numeric(trades["action_horizon_min"], errors="coerce")

    return {
        "month": month,
        "policy": policy,
        "trades": int(len(trades)),
        "ticker_day_episodes": int(
            trades[EPISODE_KEYS].drop_duplicates().shape[0]
        ),
        "days": int(trades["trading_day"].nunique()),
        "trades_per_day": float(
            len(trades) / trades["trading_day"].nunique()
        ),
        "predicted_base_ev_mean_pct": float(
            pd.to_numeric(
                trades["selected_predicted_base_ev_pct"], errors="coerce"
            ).mean()
        ),
        "predicted_rank_mean": float(
            pd.to_numeric(
                trades["selected_predicted_rank_score"], errors="coerce"
            ).mean()
        ),
        "gross_mean_pct": float(gross.mean()),
        "base_mean_pct": float(base.mean()),
        "base_median_pct": float(base.median()),
        "base_positive_rate": float(base.gt(0).mean()),
        "actual_severe_loss_rate": float(base.le(SEVERE_LOSS_PCT).mean()),
        "base_p05_pct": float(base.quantile(0.05)),
        "stress_mean_pct": float(stress.mean()),
        "day_balanced_base_mean_pct": float(daily.mean()),
        "worst_day_mean_pct": float(daily.min()),
        "action_5m_rate": float(horizons.eq(5).mean()),
        "action_10m_rate": float(horizons.eq(10).mean()),
        "action_15m_rate": float(horizons.eq(15).mean()),
        "median_entry_minute": float(
            pd.to_numeric(
                trades["minutes_since_10pct_cross"], errors="coerce"
            ).median()
        ),
    }


def _common_episode_comparison(
    baseline: pd.DataFrame,
    turned: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    if baseline.empty or turned.empty:
        return {"month": month, "common_episodes": 0}

    columns = EPISODE_KEYS + [
        "t",
        "realized_base_net_return_pct",
        "minutes_since_10pct_cross",
    ]
    merged = baseline.loc[:, columns].merge(
        turned.loc[:, columns],
        on=EPISODE_KEYS,
        suffixes=("_baseline", "_turn"),
    )
    if merged.empty:
        return {"month": month, "common_episodes": 0}

    delta = (
        merged["realized_base_net_return_pct_turn"]
        - merged["realized_base_net_return_pct_baseline"]
    )
    minute_delta = (
        merged["minutes_since_10pct_cross_turn"]
        - merged["minutes_since_10pct_cross_baseline"]
    )
    return {
        "month": month,
        "common_episodes": int(len(merged)),
        "turn_minus_baseline_base_mean_pct": float(delta.mean()),
        "turn_better_rate": float(delta.gt(0).mean()),
        "turn_later_rate": float(minute_delta.gt(0).mean()),
        "median_entry_delay_minutes": float(minute_delta.median()),
    }


def run_lomo(
    monthly_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    comparisons: list[dict[str, object]] = []

    for holdout_month, holdout in monthly_frames.items():
        train = pd.concat(
            [
                frame
                for month, frame in monthly_frames.items()
                if month != holdout_month
            ],
            ignore_index=True,
        )
        direct_models = {
            horizon: train_direct_ev_model(train, horizon)
            for horizon in HORIZONS
        }
        rank_models = {
            horizon: train_episode_rank_model(train, horizon)
            for horizon in HORIZONS
        }
        scored = score_entry_actions(holdout, direct_models, rank_models)
        diagnostic_frames.append(
            _rank_diagnostics(scored, month=holdout_month)
        )

        baseline = _select_baseline(scored)
        turned = _select_rank_turn_trades(scored)
        selected = {
            "earliest_ev_cap1": baseline,
            "rank_turn_cap1": turned,
        }

        for policy in POLICIES:
            trades = selected[policy]
            metric_rows.append(
                _metrics(
                    trades,
                    month=holdout_month,
                    policy=policy,
                )
            )
            if not trades.empty:
                item = trades.copy()
                item["month"] = holdout_month
                item["policy"] = policy
                trade_frames.append(item)

        comparisons.append(
            _common_episode_comparison(
                baseline,
                turned,
                month=holdout_month,
            )
        )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(),
        pd.concat(diagnostic_frames, ignore_index=True),
        pd.DataFrame(comparisons),
    )


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, group in details.groupby("policy", sort=True):
        valid = group.loc[group["trades"].ge(MIN_MONTH_TRADES)].copy()
        rows.append(
            {
                "policy": policy,
                "months_tested": int(len(valid)),
                "min_trades": int(valid["trades"].min())
                if len(valid)
                else 0,
                "base_positive_months": int(
                    valid["base_mean_pct"].gt(0).sum()
                ),
                "day_positive_months": int(
                    valid["day_balanced_base_mean_pct"].gt(0).sum()
                ),
                "median_base_mean_pct": float(
                    valid["base_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
                "worst_base_mean_pct": float(
                    valid["base_mean_pct"].min()
                )
                if len(valid)
                else np.nan,
                "median_day_balanced_base_mean_pct": float(
                    valid["day_balanced_base_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
                "median_base_p05_pct": float(
                    valid["base_p05_pct"].median()
                )
                if len(valid)
                else np.nan,
                "median_stress_mean_pct": float(
                    valid["stress_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "base_positive_months",
            "day_positive_months",
            "median_base_mean_pct",
            "worst_base_mean_pct",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def day_cluster_bootstrap(
    trades: pd.DataFrame,
    *,
    policy: str = "rank_turn_cap1",
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, float | int]:
    selected = trades.loc[trades["policy"].eq(policy)].copy()
    if selected.empty:
        return {
            "days": 0,
            "day_balanced_mean_pct": np.nan,
            "ci_low_pct": np.nan,
            "ci_high_pct": np.nan,
        }

    daily = (
        selected.assign(
            _base=pd.to_numeric(
                selected["realized_base_net_return_pct"], errors="coerce"
            )
        )
        .groupby("trading_day")["_base"]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    rng = np.random.default_rng(20261201)
    indices = rng.integers(0, len(daily), size=(samples, len(daily)))
    means = daily[indices].mean(axis=1)
    return {
        "days": int(len(daily)),
        "day_balanced_mean_pct": float(daily.mean()),
        "ci_low_pct": float(np.quantile(means, 0.025)),
        "ci_high_pct": float(np.quantile(means, 0.975)),
    }


def success_check(
    details: pd.DataFrame,
    diagnostics: pd.DataFrame,
    bootstrap: dict[str, float | int],
) -> dict[str, bool]:
    baseline = details.loc[
        details["policy"].eq("earliest_ev_cap1")
    ].set_index("month")
    turned = details.loc[
        details["policy"].eq("rank_turn_cap1")
    ].set_index("month")
    months = sorted(set(baseline.index) & set(turned.index))

    enough = (
        len(months) == 3
        and turned.loc[months, "trades"].ge(MIN_MONTH_TRADES).all()
    )
    base_positive = (
        len(months) == 3
        and turned.loc[months, "base_mean_pct"].gt(0).all()
    )
    day_positive = (
        len(months) == 3
        and turned.loc[
            months, "day_balanced_base_mean_pct"
        ].gt(0).all()
    )
    improves_day = (
        len(months) == 3
        and (
            turned.loc[months, "day_balanced_base_mean_pct"]
            > baseline.loc[months, "day_balanced_base_mean_pct"]
        ).all()
    )
    tail_not_worse = (
        len(months) == 3
        and (
            turned.loc[months, "base_p05_pct"]
            >= baseline.loc[months, "base_p05_pct"]
        ).all()
        and (
            turned.loc[months, "stress_mean_pct"]
            >= baseline.loc[months, "stress_mean_pct"]
        ).all()
    )

    rank_positive = True
    for month in months:
        month_diag = diagnostics.loc[diagnostics["month"].eq(month)]
        if set(month_diag["horizon_min"]) != set(HORIZONS):
            rank_positive = False
            break
        if not month_diag["global_spearman"].gt(0).all():
            rank_positive = False
            break
    rank_positive = bool(rank_positive and len(months) == 3)

    bootstrap_positive = bool(
        np.isfinite(float(bootstrap["ci_low_pct"]))
        and float(bootstrap["ci_low_pct"]) > 0
    )

    checks = {
        "enough_trades_every_month": bool(enough),
        "base_positive_every_month": bool(base_positive),
        "day_positive_every_month": bool(day_positive),
        "improves_day_every_month": bool(improves_day),
        "tail_and_stress_not_worse_every_month": bool(tail_not_worse),
        "rank_spearman_positive_all_months_horizons": rank_positive,
        "pooled_day_bootstrap_lower_bound_positive": bootstrap_positive,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def render_report(
    details: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    comparisons: pd.DataFrame,
    bootstrap: dict[str, float | int],
    checks: dict[str, bool],
) -> str:
    coverage = details.pivot(
        index="month", columns="policy", values="trades"
    ).reset_index()
    if {
        "earliest_ev_cap1",
        "rank_turn_cap1",
    }.issubset(coverage.columns):
        coverage["turn_trade_coverage_vs_baseline"] = (
            coverage["rank_turn_cap1"]
            / coverage["earliest_ev_cap1"].replace(0, np.nan)
        )

    return "\n".join(
        [
            "=== MoneyMaker State Rank-Turn Stop v0.6 ===",
            "buy_gate=direct base EV >= 0.50%",
            "rank_score=max predicted episode rank among EV-qualified actions",
            "turn_rule=arm first qualified minute; enter current state at first consecutive 1m non-increase; reset on qualification/time gap",
            "features=models unchanged from State Entry Ranker v0.3",
            "evaluation=leave-one-month-out January-March 2026",
            "NOTE=development diagnostic only; no fresh month consumed",
            "",
            "=== Cross-month summary ===",
            summary.to_string(index=False),
            "",
            "=== Month-by-month ===",
            details.to_string(index=False),
            "",
            "=== Trade coverage ===",
            coverage.to_string(index=False),
            "",
            "=== Holdout rank diagnostics ===",
            diagnostics.to_string(index=False),
            "",
            "=== Common-episode comparison ===",
            comparisons.to_string(index=False),
            "",
            "=== Pooled day-cluster bootstrap ===",
            pd.DataFrame([bootstrap]).to_string(index=False),
            "",
            "=== Pre-registered success check ===",
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
        prog="python -m victory_trader.state_rank_turn"
    )
    parser.add_argument(
        "--dataset", action="append", type=_parse_dataset, required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--comparisons-csv", type=Path, required=True)
    args = parser.parse_args()

    monthly = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    details, trades, diagnostics, comparisons = run_lomo(monthly)
    summary = summarize(details)
    bootstrap = day_cluster_bootstrap(trades)
    checks = success_check(details, diagnostics, bootstrap)
    report = render_report(
        details,
        summary,
        diagnostics,
        comparisons,
        bootstrap,
        checks,
    )
    print(report)

    for path in (
        args.report,
        args.details_csv,
        args.trades_csv,
        args.summary_csv,
        args.diagnostics_csv,
        args.comparisons_csv,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    diagnostics.to_csv(args.diagnostics_csv, index=False)
    comparisons.to_csv(args.comparisons_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
