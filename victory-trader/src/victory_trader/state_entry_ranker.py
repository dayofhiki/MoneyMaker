from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .state_action_value import (
    HORIZONS,
    _label_eligible,
    _sample_training,
    _target_column,
    action_feature_frame,
    score_actions,
    train_direct_ev_model,
)
from .state_action_risk import SEVERE_LOSS_PCT
from .state_sequence_enrichment import SEQUENCE_FEATURES
from .state_value_model import (
    BOOLEAN_FEATURES,
    LOG_FEATURES,
    RAW_FEATURES,
    _chronological_fit_calibration_split,
    _eligible,
)


EV_GATE_PCT = 0.50
RANK_GATE_QUANTILE = 0.75
EPISODE_KEYS = ["trading_day", "ticker"]
MIN_EPISODE_LABELS = 8
MIN_MONTH_TRADES = 15
MIN_CALIBRATION_RANK_CANDIDATES = 15
POLICIES = ("earliest_ev_cap1", "rank_top25_cap1")
EXTERNAL_FEATURES = (
    "short_ratio_latest_prior",
    "short_ratio_mean5_prior",
    "short_ratio_mean20_prior",
    "short_ratio_latest_minus_mean5",
    "short_ratio_mean5_minus_mean20",
    "short_volume_latest_vs_mean5",
    "finra_total_volume_latest_vs_mean5",
    "short_volume_latest_age_days",
    "eight_k_unique_filings_30d",
    "eight_k_disclosures_30d",
    "eight_k_has_filing_7d",
    "eight_k_latest_filing_age_days",
    "eight_k_distinct_primary_categories_30d",
    "eight_k_primary_capital_and_financing_count_30d",
    "eight_k_primary_leadership_and_governance_count_30d",
    "eight_k_primary_shareholder_activity_count_30d",
    "eight_k_primary_strategic_transactions_count_30d",
    "eight_k_primary_financial_results_count_30d",
    "eight_k_primary_operations_and_strategy_count_30d",
    "eight_k_primary_risk_events_count_30d",
    "eight_k_primary_regulatory_and_compliance_count_30d",
    "short_interest_latest",
    "short_interest_avg_daily_volume_latest",
    "short_interest_days_to_cover_latest",
    "short_interest_change_pct",
    "short_interest_avg_daily_volume_change_pct",
    "short_interest_days_to_cover_change",
    "short_interest_publication_age_days",
    "short_interest_reports_available",
)

MODEL_INPUT_COLUMNS = tuple(
    dict.fromkeys(
        [
            *EPISODE_KEYS,
            "t",
            "c",
            "previous_close",
            "entry_price",
            "active_minute_fraction_15m",
            *RAW_FEATURES,
            *BOOLEAN_FEATURES,
            *LOG_FEATURES,
            *SEQUENCE_FEATURES,
            *EXTERNAL_FEATURES,
            *(
                column
                for horizon in HORIZONS
                for column in (
                    f"buy_return_{horizon}m_pct",
                    f"buy_return_{horizon}m_base_net_return_pct",
                    f"buy_return_{horizon}m_stress_net_return_pct",
                )
            ),
            "eight_k_query_complete",
            "short_interest_query_complete",
            "short_ratio_latest_prior",
            "short_interest_latest",
        ]
    )
)


def read_model_panel(path: str | Path) -> pd.DataFrame:
    """Project Parquet inputs to columns consumed by the frozen v2.2 audit."""
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    columns = [column for column in MODEL_INPUT_COLUMNS if column in available]
    return pd.read_parquet(path, columns=columns)


def _coverage_row(frame: pd.DataFrame, month: str) -> dict[str, object]:
    scoreable = frame.loc[_eligible(frame)].copy()
    ticker_days = scoreable[
        [
            "ticker",
            "trading_day",
            "eight_k_query_complete",
            "short_interest_query_complete",
        ]
    ].drop_duplicates(["ticker", "trading_day"])
    return {
        "month": month,
        "scoreable_rows": int(len(scoreable)),
        "scoreable_ticker_days": int(len(ticker_days)),
        "short_volume_latest_coverage": float(
            pd.to_numeric(
                scoreable["short_ratio_latest_prior"], errors="coerce"
            )
            .notna()
            .mean()
        ),
        "short_interest_latest_coverage": float(
            pd.to_numeric(
                scoreable["short_interest_latest"], errors="coerce"
            )
            .notna()
            .mean()
        ),
        "eight_k_query_complete_coverage": float(
            pd.to_numeric(
                ticker_days["eight_k_query_complete"], errors="coerce"
            )
            .eq(1.0)
            .mean()
        ),
        "short_interest_query_complete_coverage": float(
            pd.to_numeric(
                ticker_days["short_interest_query_complete"],
                errors="coerce",
            )
            .eq(1.0)
            .mean()
        ),
    }


def day_cluster_bootstrap(
    trades: pd.DataFrame,
    *,
    policy: str,
    samples: int = 10_000,
) -> dict[str, float | int]:
    selected = trades.loc[trades["policy"].eq(policy)].copy()
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
    if len(daily) < 2:
        return {
            "days": int(len(daily)),
            "day_balanced_mean_pct": float(np.mean(daily))
            if len(daily)
            else np.nan,
            "ci_low_pct": np.nan,
            "ci_high_pct": np.nan,
        }
    rng = np.random.default_rng(20261201)
    indices = rng.integers(0, len(daily), size=(samples, len(daily)))
    means = daily[indices].mean(axis=1)
    return {
        "days": int(len(daily)),
        "day_balanced_mean_pct": float(daily.mean()),
        "ci_low_pct": float(np.quantile(means, 0.025)),
        "ci_high_pct": float(np.quantile(means, 0.975)),
    }


@dataclass(frozen=True)
class EpisodeRankModel:
    horizon: int
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]


def episode_percentile_target(
    frame: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    target = pd.to_numeric(frame[_target_column(horizon)], errors="coerce")
    keys = [frame[column].astype(str) for column in EPISODE_KEYS]
    counts = target.groupby(keys, sort=False).transform("count")
    ranks = target.groupby(keys, sort=False).rank(
        method="average",
        pct=True,
    )
    return ranks.where(counts.ge(MIN_EPISODE_LABELS))


def train_episode_rank_model(
    train: pd.DataFrame,
    horizon: int,
) -> EpisodeRankModel:
    scoreable = train.loc[_eligible(train)].copy()
    fit_period, _ = _chronological_fit_calibration_split(scoreable)
    labeled = fit_period.loc[_label_eligible(fit_period, horizon)].copy()
    labeled["_episode_rank_target"] = episode_percentile_target(
        labeled,
        horizon,
    )
    labeled = labeled.loc[labeled["_episode_rank_target"].notna()].copy()
    fit = _sample_training(labeled)
    if fit.empty:
        raise ValueError(f"no training labels for {horizon}m episode ranker")

    features = action_feature_frame(fit)
    columns = [
        column for column in features.columns if features[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable episode-rank features")

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=2.0,
        random_state=20261020 + horizon,
    )
    model.fit(
        features.loc[:, columns],
        pd.to_numeric(fit["_episode_rank_target"], errors="coerce"),
    )
    return EpisodeRankModel(
        horizon=horizon,
        model=model,
        feature_columns=tuple(columns),
    )


def score_episode_ranks(
    frame: pd.DataFrame,
    models: dict[int, EpisodeRankModel],
) -> pd.DataFrame:
    scored = frame.loc[_eligible(frame)].copy()
    features = action_feature_frame(scored)
    for horizon, fitted in models.items():
        scored[f"predicted_episode_rank_{horizon}m"] = fitted.model.predict(
            features.reindex(columns=fitted.feature_columns)
        )
    return scored


def score_entry_actions(
    frame: pd.DataFrame,
    direct_models: dict[int, object],
    rank_models: dict[int, EpisodeRankModel],
) -> pd.DataFrame:
    scored = score_actions(frame, direct_models)
    rank_scored = score_episode_ranks(frame, rank_models)

    ev_columns: list[str] = []
    rank_columns: list[str] = []
    for horizon in HORIZONS:
        ev_column = f"predicted_base_ev_{horizon}m_pct"
        rank_column = f"predicted_episode_rank_{horizon}m"
        scored[rank_column] = rank_scored[rank_column].reindex(scored.index)
        ev_columns.append(ev_column)
        rank_columns.append(rank_column)

    ev_matrix = scored.loc[:, ev_columns].to_numpy(dtype=float)
    rank_matrix = scored.loc[:, rank_columns].to_numpy(dtype=float)
    eligible_actions = np.isfinite(ev_matrix) & (ev_matrix >= EV_GATE_PCT)
    masked_ranks = np.where(eligible_actions, rank_matrix, -np.inf)
    best_rank_index = np.argmax(masked_ranks, axis=1)
    rows = np.arange(len(scored))
    horizon_array = np.asarray(HORIZONS, dtype=int)
    has_rank_action = eligible_actions.any(axis=1)

    scored["rank_action_horizon_min"] = np.where(
        has_rank_action,
        horizon_array[best_rank_index],
        0,
    )
    scored["predicted_rank_score"] = np.where(
        has_rank_action,
        rank_matrix[rows, best_rank_index],
        np.nan,
    )
    scored["rank_action_predicted_base_ev_pct"] = np.where(
        has_rank_action,
        ev_matrix[rows, best_rank_index],
        np.nan,
    )
    return scored


def fit_rank_gate(
    train: pd.DataFrame,
    direct_models: dict[int, object],
    rank_models: dict[int, EpisodeRankModel],
) -> float:
    scoreable = train.loc[_eligible(train)].copy()
    _, calibration = _chronological_fit_calibration_split(scoreable)
    scored = score_entry_actions(calibration, direct_models, rank_models)
    candidates = pd.to_numeric(
        scored["predicted_rank_score"], errors="coerce"
    ).dropna()
    if len(candidates) < MIN_CALIBRATION_RANK_CANDIDATES:
        raise ValueError(
            "fewer than 15 calibration rank candidates: "
            f"observed {len(candidates)}"
        )
    return float(candidates.quantile(RANK_GATE_QUANTILE))


def _policy_signal_mask(
    scored: pd.DataFrame,
    *,
    policy: str,
    rank_gate: float,
) -> pd.Series:
    if policy == "earliest_ev_cap1":
        return pd.to_numeric(
            scored["predicted_best_base_ev_pct"], errors="coerce"
        ).ge(EV_GATE_PCT)
    if policy == "rank_top25_cap1":
        return (
            pd.to_numeric(
                scored["rank_action_predicted_base_ev_pct"],
                errors="coerce",
            ).ge(EV_GATE_PCT)
            & pd.to_numeric(
                scored["predicted_rank_score"], errors="coerce"
            ).ge(rank_gate)
        )
    raise ValueError(f"unknown entry-ranker policy: {policy}")


def _select_first_trades(
    scored: pd.DataFrame,
    *,
    policy: str,
    rank_gate: float,
    attempt_records: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    signals = scored.loc[
        _policy_signal_mask(scored, policy=policy, rank_gate=rank_gate)
    ].copy()
    if signals.empty:
        return signals

    trades: list[dict[str, object]] = []
    for _, group in signals.groupby(EPISODE_KEYS, sort=False):
        row = group.sort_values("t", kind="stable").iloc[0]
        if policy == "earliest_ev_cap1":
            horizon = int(row["predicted_best_horizon_min"])
            selected_ev = float(row["predicted_best_base_ev_pct"])
            selected_rank = float(
                row[f"predicted_episode_rank_{horizon}m"]
            )
        else:
            horizon = int(row["rank_action_horizon_min"])
            selected_ev = float(row["rank_action_predicted_base_ev_pct"])
            selected_rank = float(row["predicted_rank_score"])

        entry = pd.to_numeric(
            pd.Series([row.get("entry_price", np.nan)]), errors="coerce"
        ).iloc[0]
        realized = pd.to_numeric(
            pd.Series([row.get(_target_column(horizon), np.nan)]),
            errors="coerce",
        ).iloc[0]
        gross = pd.to_numeric(
            pd.Series([row.get(f"buy_return_{horizon}m_pct", np.nan)]),
            errors="coerce",
        ).iloc[0]
        stress = pd.to_numeric(
            pd.Series(
                [
                    row.get(
                        f"buy_return_{horizon}m_stress_net_return_pct",
                        np.nan,
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]
        flags = {
            "entry_missing": bool(pd.isna(entry)),
            "entry_invalid": bool(
                pd.notna(entry) and (not np.isfinite(entry) or entry <= 0)
            ),
            "gross_label_missing": bool(pd.isna(gross)),
            "base_label_missing": bool(pd.isna(realized)),
            "stress_label_missing": bool(pd.isna(stress)),
            "return_nonfinite": bool(
                any(
                    pd.notna(value) and not np.isfinite(value)
                    for value in (gross, realized, stress)
                )
            ),
        }
        if flags["entry_missing"]:
            reason = "entry_missing"
        elif flags["entry_invalid"]:
            reason = "entry_invalid"
        elif flags["gross_label_missing"]:
            reason = "gross_label_missing"
        elif flags["base_label_missing"] or flags["stress_label_missing"]:
            reason = "scenario_label_missing"
        elif flags["return_nonfinite"]:
            reason = "return_nonfinite"
        else:
            reason = "evaluated"
        if attempt_records is not None:
            attempt_records.append(
                {
                    "trading_day": row.get("trading_day"),
                    "ticker": row.get("ticker"),
                    "decision_t": row.get("t"),
                    "action_horizon_min": horizon,
                    "entry_price": entry,
                    "selected_predicted_base_ev_pct": selected_ev,
                    "selected_predicted_rank_score": selected_rank,
                    "evaluation_reason": reason,
                    **flags,
                }
            )
        if reason != "evaluated":
            # The first signal was the attempted entry. Do not use future label
            # availability to fall through to a later signal in the episode.
            continue

        item = row.to_dict()
        item["action_horizon_min"] = horizon
        item["selected_predicted_base_ev_pct"] = selected_ev
        item["selected_predicted_rank_score"] = selected_rank
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
    rank_gate: float,
) -> dict[str, object]:
    if trades.empty:
        return {
            "month": month,
            "policy": policy,
            "rank_gate": rank_gate,
            "trades": 0,
        }

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
        "rank_gate": rank_gate,
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


def _rank_diagnostics(
    scored: pd.DataFrame,
    *,
    month: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        labeled = scored.loc[_label_eligible(scored, horizon)].copy()
        labeled["_target_rank"] = episode_percentile_target(
            labeled,
            horizon,
        )
        labeled = labeled.loc[labeled["_target_rank"].notna()].copy()
        prediction = pd.to_numeric(
            labeled[f"predicted_episode_rank_{horizon}m"], errors="coerce"
        )
        target_rank = pd.to_numeric(
            labeled["_target_rank"], errors="coerce"
        )
        direct_prediction = pd.to_numeric(
            labeled[f"predicted_base_ev_{horizon}m_pct"], errors="coerce"
        )
        base_target = pd.to_numeric(
            labeled[_target_column(horizon)], errors="coerce"
        )
        episode_correlations = []
        for _, group in labeled.groupby(EPISODE_KEYS, sort=False):
            if len(group) < MIN_EPISODE_LABELS:
                continue
            corr = group[f"predicted_episode_rank_{horizon}m"].corr(
                group["_target_rank"], method="spearman"
            )
            if pd.notna(corr):
                episode_correlations.append(float(corr))
        rows.append(
            {
                "month": month,
                "horizon_min": horizon,
                "rows": int(len(labeled)),
                "episodes": int(
                    labeled[EPISODE_KEYS].drop_duplicates().shape[0]
                ),
                "global_spearman": float(
                    prediction.corr(target_rank, method="spearman")
                ),
                "direct_ev_spearman": float(
                    direct_prediction.corr(base_target, method="spearman")
                ),
                "episode_median_spearman": float(
                    np.median(episode_correlations)
                )
                if episode_correlations
                else np.nan,
                "episode_positive_spearman_rate": float(
                    np.mean(np.asarray(episode_correlations) > 0)
                )
                if episode_correlations
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _common_episode_comparison(
    baseline: pd.DataFrame,
    ranked: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    if baseline.empty or ranked.empty:
        return {"month": month, "common_episodes": 0}
    columns = EPISODE_KEYS + [
        "t",
        "realized_base_net_return_pct",
        "minutes_since_10pct_cross",
    ]
    merged = baseline.loc[:, columns].merge(
        ranked.loc[:, columns],
        on=EPISODE_KEYS,
        suffixes=("_baseline", "_rank"),
    )
    if merged.empty:
        return {"month": month, "common_episodes": 0}
    delta = (
        merged["realized_base_net_return_pct_rank"]
        - merged["realized_base_net_return_pct_baseline"]
    )
    minute_delta = (
        merged["minutes_since_10pct_cross_rank"]
        - merged["minutes_since_10pct_cross_baseline"]
    )
    return {
        "month": month,
        "common_episodes": int(len(merged)),
        "rank_minus_baseline_base_mean_pct": float(delta.mean()),
        "rank_better_rate": float(delta.gt(0).mean()),
        "rank_later_rate": float(minute_delta.gt(0).mean()),
        "median_entry_delay_minutes": float(minute_delta.median()),
    }


def evaluation_folds(
    monthly_frames: dict[str, pd.DataFrame],
    mode: str,
    *,
    evaluation_min_month: str | None = None,
    evaluation_max_month: str | None = None,
):
    labels = sorted(monthly_frames)
    for holdout_month in labels:
        if evaluation_min_month is not None and holdout_month < evaluation_min_month:
            continue
        if evaluation_max_month is not None and holdout_month > evaluation_max_month:
            continue
        training_months = [
            month
            for month in labels
            if month != holdout_month and (mode == "lomo" or month < holdout_month)
        ]
        if mode == "forward" and len(training_months) < 2:
            continue
        holdout = monthly_frames[holdout_month]
        if mode == "forward":
            last_training_day = max(
                monthly_frames[month]["trading_day"].astype(str).max()
                for month in training_months
            )
            if last_training_day >= holdout["trading_day"].astype(str).min():
                raise ValueError(
                    "forward training overlaps or follows evaluation dates"
                )
        yield holdout_month, training_months, holdout


def run_evaluation(
    monthly_frames: dict[str, pd.DataFrame],
    *,
    mode: str = "lomo",
    evaluation_min_month: str | None = None,
    evaluation_max_month: str | None = None,
    audit: bool = False,
):
    metric_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    comparisons: list[dict[str, object]] = []
    attempt_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for holdout_month, training_months, holdout in evaluation_folds(
        monthly_frames,
        mode,
        evaluation_min_month=evaluation_min_month,
        evaluation_max_month=evaluation_max_month,
    ):
        train = pd.concat(
            [monthly_frames[month] for month in training_months],
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
        rank_gate = fit_rank_gate(train, direct_models, rank_models)
        scored = score_entry_actions(holdout, direct_models, rank_models)
        coverage_rows.append(_coverage_row(holdout, holdout_month))
        diagnostic_frames.append(
            _rank_diagnostics(scored, month=holdout_month)
        )

        selected: dict[str, pd.DataFrame] = {}
        for policy in POLICIES:
            policy_attempts: list[dict[str, object]] = []
            trades = _select_first_trades(
                scored,
                policy=policy,
                rank_gate=rank_gate,
                attempt_records=policy_attempts,
            )
            attempt_rows.extend(
                {"month": holdout_month, "policy": policy, **row}
                for row in policy_attempts
            )
            path_rows.append(
                {
                    "month": holdout_month,
                    "policy": policy,
                    "attempted": len(policy_attempts),
                    "evaluated": sum(
                        row["evaluation_reason"] == "evaluated"
                        for row in policy_attempts
                    ),
                    "unevaluable": sum(
                        row["evaluation_reason"] != "evaluated"
                        for row in policy_attempts
                    ),
                }
            )
            selected[policy] = trades
            metric_rows.append(
                _metrics(
                    trades,
                    month=holdout_month,
                    policy=policy,
                    rank_gate=rank_gate,
                )
            )
            if not trades.empty:
                item = trades.copy()
                item["month"] = holdout_month
                item["policy"] = policy
                item["rank_gate"] = rank_gate
                trade_frames.append(item)
        comparisons.append(
            _common_episode_comparison(
                selected["earliest_ev_cap1"],
                selected["rank_top25_cap1"],
                month=holdout_month,
            )
        )
        del train, direct_models, rank_models, scored
        gc.collect()

    core = (
        pd.DataFrame(metric_rows),
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(),
        pd.concat(diagnostic_frames, ignore_index=True),
        pd.DataFrame(comparisons),
    )
    if not audit:
        return core
    return (
        *core,
        pd.DataFrame(attempt_rows),
        pd.DataFrame(path_rows),
        pd.DataFrame(coverage_rows),
    )


def run_lomo(
    monthly_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_evaluation(monthly_frames, mode="lomo")


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
                "worst_base_mean_pct": float(valid["base_mean_pct"].min())
                if len(valid)
                else np.nan,
                "median_day_balanced_base_mean_pct": float(
                    valid["day_balanced_base_mean_pct"].median()
                )
                if len(valid)
                else np.nan,
                "median_base_p05_pct": float(valid["base_p05_pct"].median())
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


def render_report(
    details: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    evaluation: str = "lomo",
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker State Entry Ranker v0.3 ===",
            "absolute_gate=calibrated direct base EV >= 0.50%",
            "rank_target=within-ticker-day percentile of realized base-net return per action horizon",
            "rank_gate=train-only calibration score 75th percentile",
            "policies=first EV-qualified entry vs first EV+rank-qualified entry; one attempt per ticker-day",
            "features=point-in-time state + breadth + 36 causal ticker-local lags",
            f"evaluation={evaluation}; forward uses strictly earlier months only",
            "NOTE=development diagnostic only; no fresh month consumed",
            "",
            "=== Cross-month summary ===",
            summary.to_string(index=False),
            "",
            "=== Month-by-month ===",
            details.to_string(index=False),
            "",
            "=== Holdout rank diagnostics ===",
            diagnostics.to_string(index=False),
            "",
            "=== Common-episode comparison ===",
            comparisons.to_string(index=False),
        ]
    )


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m victory_trader.state_entry_ranker"
    )
    parser.add_argument(
        "--dataset", action="append", type=_parse_dataset, required=True
    )
    parser.add_argument("--evaluation", choices=("lomo", "forward"), default="lomo")
    parser.add_argument("--evaluation-min-month")
    parser.add_argument("--evaluation-max-month")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--comparisons-csv", type=Path, required=True)
    parser.add_argument("--attempts-csv", type=Path)
    parser.add_argument("--paths-csv", type=Path)
    parser.add_argument("--coverage-csv", type=Path)
    args = parser.parse_args()

    dataset_paths = dict(args.dataset)
    monthly: dict[str, pd.DataFrame] = {}
    for label in sorted(dataset_paths):
        if (
            args.evaluation == "forward"
            and args.evaluation_max_month is not None
            and label > args.evaluation_max_month
        ):
            continue
        monthly[label] = read_model_panel(dataset_paths[label])
    (
        details,
        trades,
        diagnostics,
        comparisons,
        attempts,
        paths,
        coverage,
    ) = run_evaluation(
        monthly,
        mode=args.evaluation,
        evaluation_min_month=args.evaluation_min_month,
        evaluation_max_month=args.evaluation_max_month,
        audit=True,
    )
    summary = summarize(details)
    report = render_report(
        details,
        summary,
        diagnostics,
        comparisons,
        evaluation=args.evaluation,
    )
    attempt_summary = (
        attempts.groupby(
            ["month", "policy", "evaluation_reason"], dropna=False
        )
        .size()
        .rename("attempts")
        .reset_index()
        if not attempts.empty
        else pd.DataFrame()
    )
    ranked = (
        trades.loc[trades["policy"].eq("rank_top25_cap1")]
        if not trades.empty
        else pd.DataFrame()
    )
    bootstrap = (
        day_cluster_bootstrap(
            ranked,
            policy="rank_top25_cap1",
            samples=10_000,
        )
        if not ranked.empty and ranked["trading_day"].nunique() >= 2
        else {
            "bootstrap_available": False,
            "reason": "fewer than two independent evaluated trading days",
        }
    )
    report += "\n\n=== Attempt evaluation reasons ===\n"
    report += attempt_summary.to_string(index=False)
    report += "\n\n=== Attempt paths ===\n" + paths.to_string(index=False)
    report += "\n\n=== External-data coverage ===\n"
    report += coverage.to_string(index=False)
    report += "\n\n=== Pooled trading-day bootstrap ===\n"
    report += str(bootstrap)
    print(report)

    for path in (
        args.report,
        args.details_csv,
        args.trades_csv,
        args.summary_csv,
        args.diagnostics_csv,
        args.comparisons_csv,
        args.attempts_csv,
        args.paths_csv,
        args.coverage_csv,
    ):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    diagnostics.to_csv(args.diagnostics_csv, index=False)
    comparisons.to_csv(args.comparisons_csv, index=False)
    if args.attempts_csv is not None:
        attempts.to_csv(args.attempts_csv, index=False)
    if args.paths_csv is not None:
        paths.to_csv(args.paths_csv, index=False)
    if args.coverage_csv is not None:
        coverage.to_csv(args.coverage_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
