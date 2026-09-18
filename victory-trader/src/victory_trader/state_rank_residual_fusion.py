from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

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
from .state_rank_turn import day_cluster_bootstrap, summarize


MIN_MONTH_TRADES = 15
BOOTSTRAP_SAMPLES = 10_000
POLICIES = ("earliest_ev_cap1", "rank_fused_ev_cap1")


@dataclass(frozen=True)
class RankResidualCorrection:
    horizon: int
    model: IsotonicRegression
    training_rows: int
    residual_rank_spearman: float
    residual_mean_pct: float


def _fit_base_models(
    train: pd.DataFrame,
) -> tuple[dict[int, object], dict[int, object]]:
    direct_models = {
        horizon: train_direct_ev_model(train, horizon)
        for horizon in HORIZONS
    }
    rank_models = {
        horizon: train_episode_rank_model(train, horizon)
        for horizon in HORIZONS
    }
    return direct_models, rank_models


def _fit_rank_residual_corrections(
    crossfit_scores: list[tuple[str, pd.DataFrame]],
    *,
    holdout_month: str,
) -> tuple[dict[int, RankResidualCorrection], pd.DataFrame]:
    corrections: dict[int, RankResidualCorrection] = {}
    diagnostic_rows: list[dict[str, object]] = []

    for horizon in HORIZONS:
        pieces: list[pd.DataFrame] = []
        direct_column = f"predicted_base_ev_{horizon}m_pct"
        rank_column = f"predicted_episode_rank_{horizon}m"
        target_column = _target_column(horizon)

        for source_month, scored in crossfit_scores:
            piece = pd.DataFrame(
                {
                    "source_month": source_month,
                    "direct_ev": pd.to_numeric(
                        scored[direct_column], errors="coerce"
                    ),
                    "rank_score": pd.to_numeric(
                        scored[rank_column], errors="coerce"
                    ),
                    "realized": pd.to_numeric(
                        scored[target_column], errors="coerce"
                    ),
                }
            )
            pieces.append(piece)

        training = pd.concat(pieces, ignore_index=True)
        training = training.loc[
            np.isfinite(training["direct_ev"])
            & np.isfinite(training["rank_score"])
            & np.isfinite(training["realized"])
        ].copy()
        if len(training) < 2 or training["rank_score"].nunique() < 2:
            raise ValueError(
                "insufficient cross-fitted rank-residual training rows "
                f"for {horizon}m"
            )

        training["residual"] = (
            training["realized"] - training["direct_ev"]
        )
        model = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
        )
        model.fit(
            training["rank_score"].to_numpy(dtype=float),
            training["residual"].to_numpy(dtype=float),
        )
        spearman = training["rank_score"].corr(
            training["residual"],
            method="spearman",
        )
        fitted = RankResidualCorrection(
            horizon=horizon,
            model=model,
            training_rows=int(len(training)),
            residual_rank_spearman=float(spearman),
            residual_mean_pct=float(training["residual"].mean()),
        )
        corrections[horizon] = fitted

        rank_min = float(training["rank_score"].min())
        rank_median = float(training["rank_score"].median())
        rank_max = float(training["rank_score"].max())
        predicted = model.predict(
            np.asarray([rank_min, rank_median, rank_max], dtype=float)
        )
        diagnostic_rows.append(
            {
                "holdout_month": holdout_month,
                "horizon_min": horizon,
                "training_rows": int(len(training)),
                "source_months": ",".join(
                    sorted(training["source_month"].astype(str).unique())
                ),
                "residual_rank_spearman": float(spearman),
                "residual_mean_pct": float(training["residual"].mean()),
                "rank_min": rank_min,
                "rank_median": rank_median,
                "rank_max": rank_max,
                "correction_at_rank_min_pct": float(predicted[0]),
                "correction_at_rank_median_pct": float(predicted[1]),
                "correction_at_rank_max_pct": float(predicted[2]),
            }
        )

    return corrections, pd.DataFrame(diagnostic_rows)


def score_rank_fused_actions(
    frame: pd.DataFrame,
    direct_models: dict[int, object],
    rank_models: dict[int, object],
    corrections: dict[int, RankResidualCorrection],
) -> pd.DataFrame:
    scored = score_entry_actions(frame, direct_models, rank_models)

    fused_columns: list[str] = []
    for horizon in HORIZONS:
        direct_column = f"predicted_base_ev_{horizon}m_pct"
        rank_column = f"predicted_episode_rank_{horizon}m"
        fused_column = f"predicted_fused_ev_{horizon}m_pct"

        direct = pd.to_numeric(
            scored[direct_column], errors="coerce"
        ).to_numpy(dtype=float)
        rank = pd.to_numeric(
            scored[rank_column], errors="coerce"
        ).to_numpy(dtype=float)
        fused = np.full(len(scored), np.nan, dtype=float)
        mask = np.isfinite(direct) & np.isfinite(rank)
        if mask.any():
            adjustment = corrections[horizon].model.predict(rank[mask])
            fused[mask] = direct[mask] + adjustment
        scored[fused_column] = fused
        fused_columns.append(fused_column)

    fused_matrix = scored.loc[:, fused_columns].to_numpy(dtype=float)
    finite = np.isfinite(fused_matrix)
    safe = np.where(finite, fused_matrix, -np.inf)
    best_index = np.argmax(safe, axis=1)
    rows = np.arange(len(scored))
    horizon_array = np.asarray(HORIZONS, dtype=int)
    has_action = finite.any(axis=1)

    scored["predicted_best_fused_ev_pct"] = np.where(
        has_action,
        safe[rows, best_index],
        np.nan,
    )
    scored["predicted_best_fused_horizon_min"] = np.where(
        has_action,
        horizon_array[best_index],
        0,
    )
    return scored


def _select_fused_trades(scored: pd.DataFrame) -> pd.DataFrame:
    signals = scored.loc[
        pd.to_numeric(
            scored["predicted_best_fused_ev_pct"], errors="coerce"
        ).ge(EV_GATE_PCT)
    ].copy()
    if signals.empty:
        return signals

    trades: list[dict[str, object]] = []
    for _, group in signals.groupby(EPISODE_KEYS, sort=False):
        row = group.sort_values("t", kind="stable").iloc[0]
        horizon = int(row["predicted_best_fused_horizon_min"])
        entry = pd.to_numeric(
            pd.Series([row.get("entry_price", np.nan)]), errors="coerce"
        ).iloc[0]
        realized = pd.to_numeric(
            pd.Series([row.get(_target_column(horizon), np.nan)]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(entry) or float(entry) <= 0 or pd.isna(realized):
            continue

        item = row.to_dict()
        item["action_horizon_min"] = horizon
        item["selected_predicted_base_ev_pct"] = float(
            row["predicted_best_fused_ev_pct"]
        )
        item["selected_direct_ev_pct"] = float(
            row[f"predicted_base_ev_{horizon}m_pct"]
        )
        item["selected_predicted_rank_score"] = float(
            row[f"predicted_episode_rank_{horizon}m"]
        )
        item["realized_base_net_return_pct"] = float(realized)
        item["realized_gross_return_pct"] = float(
            row[f"buy_return_{horizon}m_pct"]
        )
        item["realized_stress_net_return_pct"] = float(
            row[f"buy_return_{horizon}m_stress_net_return_pct"]
        )
        trades.append(item)

    return pd.DataFrame(trades)


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
        "predicted_gate_ev_mean_pct": float(
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
    fused: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    if baseline.empty or fused.empty:
        return {"month": month, "common_episodes": 0}

    columns = EPISODE_KEYS + [
        "t",
        "realized_base_net_return_pct",
        "minutes_since_10pct_cross",
    ]
    merged = baseline.loc[:, columns].merge(
        fused.loc[:, columns],
        on=EPISODE_KEYS,
        suffixes=("_baseline", "_fused"),
    )
    if merged.empty:
        return {"month": month, "common_episodes": 0}

    delta = (
        merged["realized_base_net_return_pct_fused"]
        - merged["realized_base_net_return_pct_baseline"]
    )
    minute_delta = (
        merged["minutes_since_10pct_cross_fused"]
        - merged["minutes_since_10pct_cross_baseline"]
    )
    return {
        "month": month,
        "common_episodes": int(len(merged)),
        "fused_minus_baseline_base_mean_pct": float(delta.mean()),
        "fused_better_rate": float(delta.gt(0).mean()),
        "fused_later_rate": float(minute_delta.gt(0).mean()),
        "fused_earlier_rate": float(minute_delta.lt(0).mean()),
        "median_entry_delay_minutes": float(minute_delta.median()),
    }


def run_lomo(
    monthly_frames: dict[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if len(monthly_frames) != 3:
        raise ValueError("State Rank-Residual Fusion v0.7 requires 3 months")

    single_month_models = {
        month: _fit_base_models(frame)
        for month, frame in monthly_frames.items()
    }

    metric_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    rank_diagnostics: list[pd.DataFrame] = []
    fusion_diagnostics: list[pd.DataFrame] = []
    comparisons: list[dict[str, object]] = []

    for holdout_month, holdout in monthly_frames.items():
        training_months = [
            month for month in monthly_frames if month != holdout_month
        ]
        crossfit_scores: list[tuple[str, pd.DataFrame]] = []
        for target_month in training_months:
            source_month = next(
                month
                for month in training_months
                if month != target_month
            )
            direct_models, rank_models = single_month_models[source_month]
            crossfit_scores.append(
                (
                    target_month,
                    score_entry_actions(
                        monthly_frames[target_month],
                        direct_models,
                        rank_models,
                    ),
                )
            )

        corrections, fusion_diag = _fit_rank_residual_corrections(
            crossfit_scores,
            holdout_month=holdout_month,
        )
        fusion_diagnostics.append(fusion_diag)

        train = pd.concat(
            [monthly_frames[month] for month in training_months],
            ignore_index=True,
        )
        direct_models, rank_models = _fit_base_models(train)
        scored = score_rank_fused_actions(
            holdout,
            direct_models,
            rank_models,
            corrections,
        )
        rank_diagnostics.append(
            _rank_diagnostics(scored, month=holdout_month)
        )

        baseline = _ranker_select_first_trades(
            scored,
            policy="earliest_ev_cap1",
            rank_gate=0.0,
        )
        fused = _select_fused_trades(scored)
        selected = {
            "earliest_ev_cap1": baseline,
            "rank_fused_ev_cap1": fused,
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
                fused,
                month=holdout_month,
            )
        )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(),
        pd.concat(rank_diagnostics, ignore_index=True),
        pd.concat(fusion_diagnostics, ignore_index=True),
        pd.DataFrame(comparisons),
    )


def success_check(
    details: pd.DataFrame,
    rank_diagnostics: pd.DataFrame,
    bootstrap: dict[str, float | int],
) -> dict[str, bool]:
    baseline = details.loc[
        details["policy"].eq("earliest_ev_cap1")
    ].set_index("month")
    fused = details.loc[
        details["policy"].eq("rank_fused_ev_cap1")
    ].set_index("month")
    months = sorted(set(baseline.index) & set(fused.index))

    enough = (
        len(months) == 3
        and fused.loc[months, "trades"].ge(MIN_MONTH_TRADES).all()
    )
    base_positive = (
        len(months) == 3
        and fused.loc[months, "base_mean_pct"].gt(0).all()
    )
    day_positive = (
        len(months) == 3
        and fused.loc[
            months, "day_balanced_base_mean_pct"
        ].gt(0).all()
    )
    improves_day = (
        len(months) == 3
        and (
            fused.loc[months, "day_balanced_base_mean_pct"]
            > baseline.loc[months, "day_balanced_base_mean_pct"]
        ).all()
    )
    tail_not_worse = (
        len(months) == 3
        and (
            fused.loc[months, "base_p05_pct"]
            >= baseline.loc[months, "base_p05_pct"]
        ).all()
        and (
            fused.loc[months, "stress_mean_pct"]
            >= baseline.loc[months, "stress_mean_pct"]
        ).all()
    )

    rank_positive = True
    for month in months:
        month_diag = rank_diagnostics.loc[
            rank_diagnostics["month"].eq(month)
        ]
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
    rank_diagnostics: pd.DataFrame,
    fusion_diagnostics: pd.DataFrame,
    comparisons: pd.DataFrame,
    bootstrap: dict[str, float | int],
    checks: dict[str, bool],
) -> str:
    coverage = details.pivot(
        index="month", columns="policy", values="trades"
    ).reset_index()
    if {
        "earliest_ev_cap1",
        "rank_fused_ev_cap1",
    }.issubset(coverage.columns):
        coverage["fused_trade_coverage_vs_baseline"] = (
            coverage["rank_fused_ev_cap1"]
            / coverage["earliest_ev_cap1"].replace(0, np.nan)
        )

    return "\n".join(
        [
            "=== MoneyMaker State Rank-Residual Fusion v0.7 ===",
            "buy_gate=fused or direct base EV >= 0.50%",
            "fusion=direct EV + cross-fitted isotonic residual correction(rank)",
            "rank_threshold=none",
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
            rank_diagnostics.to_string(index=False),
            "",
            "=== Cross-fitted residual-fusion diagnostics ===",
            fusion_diagnostics.to_string(index=False),
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
        prog="python -m victory_trader.state_rank_residual_fusion"
    )
    parser.add_argument(
        "--dataset", action="append", type=_parse_dataset, required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--rank-diagnostics-csv", type=Path, required=True)
    parser.add_argument("--fusion-diagnostics-csv", type=Path, required=True)
    parser.add_argument("--comparisons-csv", type=Path, required=True)
    args = parser.parse_args()

    monthly = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    (
        details,
        trades,
        rank_diagnostics,
        fusion_diagnostics,
        comparisons,
    ) = run_lomo(monthly)
    summary = summarize(details)
    bootstrap = day_cluster_bootstrap(
        trades,
        policy="rank_fused_ev_cap1",
        samples=BOOTSTRAP_SAMPLES,
    )
    checks = success_check(details, rank_diagnostics, bootstrap)
    report = render_report(
        details,
        summary,
        rank_diagnostics,
        fusion_diagnostics,
        comparisons,
        bootstrap,
        checks,
    )
    print(report)

    paths = (
        args.report,
        args.details_csv,
        args.trades_csv,
        args.summary_csv,
        args.rank_diagnostics_csv,
        args.fusion_diagnostics_csv,
        args.comparisons_csv,
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    rank_diagnostics.to_csv(args.rank_diagnostics_csv, index=False)
    fusion_diagnostics.to_csv(args.fusion_diagnostics_csv, index=False)
    comparisons.to_csv(args.comparisons_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
