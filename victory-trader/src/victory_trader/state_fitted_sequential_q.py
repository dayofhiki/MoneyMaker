from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .state_action_risk import SEVERE_LOSS_PCT
from .state_multi_source_value import (
    _coverage_row,
    multi_source_action_feature_frame,
)
from .state_pairwise_buy_wait import (
    EPISODE_KEYS,
    GROSS_COLUMN,
    HOLD_MINUTES,
    MIN_MONTH_TRADES,
    STRESS_COLUMN,
    TARGET_COLUMN,
)
from .state_rank_turn import day_cluster_bootstrap
from .state_value_model import _eligible


MINUTE_MS = 60_000
CHECKPOINT_OFFSETS = (0, 5, 10)
WAIT_MS = 5 * MINUTE_MS
TARGET_WINSOR_LOW = 0.005
TARGET_WINSOR_HIGH = 0.995
BOOTSTRAP_SAMPLES = 10_000
POLICIES = ("earliest_eligible_10m_cap1", "fitted_sequential_q_cap1")


@dataclass(frozen=True)
class QModel:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    target_low: float
    target_high: float


@dataclass(frozen=True)
class SequentialModels:
    q10_buy: QModel
    q5_buy: QModel
    q5_wait: QModel
    q0_buy: QModel
    q0_wait: QModel


def split_training_days(frame: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
    days = sorted(frame["trading_day"].astype(str).unique())
    if len(days) < 6:
        raise ValueError("not enough training days for three-way sequential split")
    parts = np.array_split(np.asarray(days, dtype=object), 3)
    if any(len(part) == 0 for part in parts):
        raise ValueError("empty sequential training partition")
    return tuple(set(map(str, part.tolist())) for part in parts)  # type: ignore[return-value]


def checkpoint_frames(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    scoreable = frame.loc[_eligible(frame)].copy()
    scoreable["_t_num"] = pd.to_numeric(scoreable["t"], errors="coerce")
    rows: dict[int, list[pd.Series]] = {offset: [] for offset in CHECKPOINT_OFFSETS}

    for _, group in scoreable.groupby(EPISODE_KEYS, sort=False):
        ordered = group.sort_values("_t_num", kind="stable")
        if ordered.empty:
            continue
        first = ordered.iloc[0]
        first_t = int(first["_t_num"])
        by_t = {int(row["_t_num"]): row for _, row in ordered.iterrows()}
        for offset in CHECKPOINT_OFFSETS:
            row = by_t.get(first_t + offset * MINUTE_MS)
            if row is not None:
                rows[offset].append(row)

    output: dict[int, pd.DataFrame] = {}
    for offset, items in rows.items():
        if items:
            out = pd.DataFrame(items).drop(columns=["_t_num"], errors="ignore")
            output[offset] = out.reset_index(drop=True)
        else:
            output[offset] = scoreable.iloc[0:0].drop(
                columns=["_t_num"], errors="ignore"
            ).copy()
    return output


def _fit_q_model(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    seed: int,
) -> QModel:
    y = pd.to_numeric(target, errors="coerce")
    valid = y.notna()
    fit = frame.loc[valid].copy()
    y = y.loc[valid].astype(float)
    if fit.empty:
        raise ValueError("no Q-model training rows")

    features = multi_source_action_feature_frame(fit)
    columns = [
        column for column in features.columns if features[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable Q-model features")

    low = float(y.quantile(TARGET_WINSOR_LOW))
    high = float(y.quantile(TARGET_WINSOR_HIGH))
    clipped = y.clip(lower=low, upper=high)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(features.loc[:, columns], clipped)
    return QModel(
        model=model,
        feature_columns=tuple(columns),
        target_low=low,
        target_high=high,
    )


def _predict_q(frame: pd.DataFrame, fitted: QModel) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    features = multi_source_action_feature_frame(frame).reindex(
        columns=fitted.feature_columns
    )
    return fitted.model.predict(features).astype(float)


def _episode_map(frame: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    result: dict[tuple[str, str], pd.Series] = {}
    for _, row in frame.iterrows():
        key = (str(row["trading_day"]), str(row["ticker"]))
        result[key] = row
    return result


def _realized_buy_value(row: pd.Series | None) -> float:
    if row is None:
        return np.nan
    value = pd.to_numeric(
        pd.Series([row.get(TARGET_COLUMN, np.nan)]), errors="coerce"
    ).iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _choose_mid_action(q_buy: float, q_wait: float) -> str:
    # Exact-tie order from pre-registration: SKIP, WAIT, BUY.
    best = max(float(q_buy), float(q_wait), 0.0)
    if best == 0.0:
        return "SKIP"
    if float(q_wait) == best:
        return "WAIT"
    return "BUY"


def _stage10_realized_value(
    row10: pd.Series | None,
    model10: QModel,
) -> float:
    if row10 is None:
        return 0.0
    one = pd.DataFrame([row10])
    q_buy = float(_predict_q(one, model10)[0])
    if q_buy <= 0.0:
        return 0.0
    return _realized_buy_value(row10)


def _stage5_realized_value(
    row5: pd.Series | None,
    row10: pd.Series | None,
    models: SequentialModels | tuple[QModel, QModel, QModel],
) -> float:
    if row5 is None:
        return 0.0

    if isinstance(models, SequentialModels):
        q5_buy_model = models.q5_buy
        q5_wait_model = models.q5_wait
        q10_buy_model = models.q10_buy
    else:
        q5_buy_model, q5_wait_model, q10_buy_model = models

    one = pd.DataFrame([row5])
    q_buy = float(_predict_q(one, q5_buy_model)[0])
    q_wait = float(_predict_q(one, q5_wait_model)[0])
    action = _choose_mid_action(q_buy, q_wait)

    if action == "SKIP":
        return 0.0
    if action == "BUY":
        return _realized_buy_value(row5)
    return _stage10_realized_value(row10, q10_buy_model)


def _wait_targets_stage5(
    stage5: pd.DataFrame,
    stage10: pd.DataFrame,
    model10: QModel,
) -> pd.Series:
    map10 = _episode_map(stage10)
    values: list[float] = []
    for _, row5 in stage5.iterrows():
        key = (str(row5["trading_day"]), str(row5["ticker"]))
        values.append(_stage10_realized_value(map10.get(key), model10))
    return pd.Series(values, index=stage5.index, dtype=float)


def _wait_targets_stage0(
    stage0: pd.DataFrame,
    stage5: pd.DataFrame,
    stage10: pd.DataFrame,
    q5_buy: QModel,
    q5_wait: QModel,
    q10_buy: QModel,
) -> pd.Series:
    map5 = _episode_map(stage5)
    map10 = _episode_map(stage10)
    values: list[float] = []
    downstream = (q5_buy, q5_wait, q10_buy)
    for _, row0 in stage0.iterrows():
        key = (str(row0["trading_day"]), str(row0["ticker"]))
        values.append(
            _stage5_realized_value(
                map5.get(key),
                map10.get(key),
                downstream,
            )
        )
    return pd.Series(values, index=stage0.index, dtype=float)


def train_sequential_models(train: pd.DataFrame) -> SequentialModels:
    days_a, days_b, days_c = split_training_days(train)
    checkpoints = checkpoint_frames(train)

    stage10_a = checkpoints[10].loc[
        checkpoints[10]["trading_day"].astype(str).isin(days_a)
    ].copy()
    q10_buy = _fit_q_model(
        stage10_a,
        pd.to_numeric(stage10_a[TARGET_COLUMN], errors="coerce"),
        seed=2026092710,
    )

    stage5_b = checkpoints[5].loc[
        checkpoints[5]["trading_day"].astype(str).isin(days_b)
    ].copy()
    stage10_b = checkpoints[10].loc[
        checkpoints[10]["trading_day"].astype(str).isin(days_b)
    ].copy()
    q5_buy = _fit_q_model(
        stage5_b,
        pd.to_numeric(stage5_b[TARGET_COLUMN], errors="coerce"),
        seed=2026092705,
    )
    wait5_target = _wait_targets_stage5(stage5_b, stage10_b, q10_buy)
    q5_wait = _fit_q_model(
        stage5_b,
        wait5_target,
        seed=2026092755,
    )

    stage0_c = checkpoints[0].loc[
        checkpoints[0]["trading_day"].astype(str).isin(days_c)
    ].copy()
    stage5_c = checkpoints[5].loc[
        checkpoints[5]["trading_day"].astype(str).isin(days_c)
    ].copy()
    stage10_c = checkpoints[10].loc[
        checkpoints[10]["trading_day"].astype(str).isin(days_c)
    ].copy()
    q0_buy = _fit_q_model(
        stage0_c,
        pd.to_numeric(stage0_c[TARGET_COLUMN], errors="coerce"),
        seed=2026092700,
    )
    wait0_target = _wait_targets_stage0(
        stage0_c,
        stage5_c,
        stage10_c,
        q5_buy,
        q5_wait,
        q10_buy,
    )
    q0_wait = _fit_q_model(
        stage0_c,
        wait0_target,
        seed=2026092750,
    )

    return SequentialModels(
        q10_buy=q10_buy,
        q5_buy=q5_buy,
        q5_wait=q5_wait,
        q0_buy=q0_buy,
        q0_wait=q0_wait,
    )


def _attempt_trade(
    row: pd.Series,
    *,
    decision_score: float,
) -> dict[str, object] | None:
    entry = pd.to_numeric(
        pd.Series([row.get("entry_price", np.nan)]), errors="coerce"
    ).iloc[0]
    realized = pd.to_numeric(
        pd.Series([row.get(TARGET_COLUMN, np.nan)]), errors="coerce"
    ).iloc[0]
    gross = pd.to_numeric(
        pd.Series([row.get(GROSS_COLUMN, np.nan)]), errors="coerce"
    ).iloc[0]
    stress = pd.to_numeric(
        pd.Series([row.get(STRESS_COLUMN, np.nan)]), errors="coerce"
    ).iloc[0]

    if (
        pd.isna(entry)
        or float(entry) <= 0
        or pd.isna(realized)
        or pd.isna(gross)
        or pd.isna(stress)
    ):
        return None

    item = row.to_dict()
    item["action_horizon_min"] = HOLD_MINUTES
    item["selected_decision_score"] = float(decision_score)
    item["realized_base_net_return_pct"] = float(realized)
    item["realized_gross_return_pct"] = float(gross)
    item["realized_stress_net_return_pct"] = float(stress)
    return item


def select_sequential_policy(
    frame: pd.DataFrame,
    models: SequentialModels,
) -> tuple[pd.DataFrame, dict[str, int]]:
    checkpoints = checkpoint_frames(frame)
    map0 = _episode_map(checkpoints[0])
    map5 = _episode_map(checkpoints[5])
    map10 = _episode_map(checkpoints[10])

    trades: list[dict[str, object]] = []
    paths = {
        "episodes": len(map0),
        "buy_at_0": 0,
        "buy_at_5": 0,
        "buy_at_10": 0,
        "skip_at_0": 0,
        "skip_at_5": 0,
        "skip_at_10": 0,
        "skip_missing_checkpoint": 0,
        "buy_unfillable": 0,
    }

    for key, row0 in map0.items():
        one0 = pd.DataFrame([row0])
        q0_buy = float(_predict_q(one0, models.q0_buy)[0])
        q0_wait = float(_predict_q(one0, models.q0_wait)[0])
        action0 = _choose_mid_action(q0_buy, q0_wait)

        if action0 == "SKIP":
            paths["skip_at_0"] += 1
            continue
        if action0 == "BUY":
            trade = _attempt_trade(row0, decision_score=q0_buy)
            if trade is None:
                paths["buy_unfillable"] += 1
            else:
                paths["buy_at_0"] += 1
                trades.append(trade)
            continue

        row5 = map5.get(key)
        if row5 is None:
            paths["skip_missing_checkpoint"] += 1
            continue

        one5 = pd.DataFrame([row5])
        q5_buy = float(_predict_q(one5, models.q5_buy)[0])
        q5_wait = float(_predict_q(one5, models.q5_wait)[0])
        action5 = _choose_mid_action(q5_buy, q5_wait)

        if action5 == "SKIP":
            paths["skip_at_5"] += 1
            continue
        if action5 == "BUY":
            trade = _attempt_trade(row5, decision_score=q5_buy)
            if trade is None:
                paths["buy_unfillable"] += 1
            else:
                paths["buy_at_5"] += 1
                trades.append(trade)
            continue

        row10 = map10.get(key)
        if row10 is None:
            paths["skip_missing_checkpoint"] += 1
            continue

        q10_buy = float(_predict_q(pd.DataFrame([row10]), models.q10_buy)[0])
        if q10_buy <= 0.0:
            paths["skip_at_10"] += 1
            continue
        trade = _attempt_trade(row10, decision_score=q10_buy)
        if trade is None:
            paths["buy_unfillable"] += 1
        else:
            paths["buy_at_10"] += 1
            trades.append(trade)

    return pd.DataFrame(trades), paths


def select_earliest(frame: pd.DataFrame) -> pd.DataFrame:
    checkpoints = checkpoint_frames(frame)
    trades: list[dict[str, object]] = []
    for _, row in checkpoints[0].iterrows():
        trade = _attempt_trade(row, decision_score=0.0)
        if trade is not None:
            trades.append(trade)
    return pd.DataFrame(trades)


def _metrics(
    trades: pd.DataFrame,
    *,
    month: str,
    policy: str,
) -> dict[str, object]:
    if trades.empty:
        return {"month": month, "policy": policy, "trades": 0}

    base = pd.to_numeric(trades["realized_base_net_return_pct"], errors="coerce")
    gross = pd.to_numeric(trades["realized_gross_return_pct"], errors="coerce")
    stress = pd.to_numeric(
        trades["realized_stress_net_return_pct"], errors="coerce"
    )
    daily = trades.assign(_base=base).groupby("trading_day")["_base"].mean()

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
        "decision_score_mean": float(
            pd.to_numeric(
                trades["selected_decision_score"], errors="coerce"
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
        "median_entry_minute": float(
            pd.to_numeric(
                trades["minutes_since_10pct_cross"], errors="coerce"
            ).median()
        ),
    }


def _spearman(prediction: np.ndarray, actual: pd.Series) -> float:
    p = pd.Series(prediction, index=actual.index, dtype=float)
    y = pd.to_numeric(actual, errors="coerce")
    valid = p.notna() & y.notna()
    if valid.sum() < 2:
        return np.nan
    return float(p.loc[valid].corr(y.loc[valid], method="spearman"))


def holdout_diagnostics(
    frame: pd.DataFrame,
    models: SequentialModels,
    *,
    month: str,
) -> dict[str, object]:
    checkpoints = checkpoint_frames(frame)
    stage0 = checkpoints[0]
    stage5 = checkpoints[5]
    stage10 = checkpoints[10]

    q0_buy = _predict_q(stage0, models.q0_buy)
    q0_wait = _predict_q(stage0, models.q0_wait)
    actual_buy = pd.to_numeric(stage0[TARGET_COLUMN], errors="coerce")

    wait_actual = _wait_targets_stage0(
        stage0,
        stage5,
        stage10,
        models.q5_buy,
        models.q5_wait,
        models.q10_buy,
    )
    return {
        "month": month,
        "rows": int(len(stage0)),
        "q0_buy_spearman": _spearman(q0_buy, actual_buy),
        "q0_wait_spearman": _spearman(q0_wait, wait_actual),
        "q0_buy_predicted_mean": float(np.nanmean(q0_buy)),
        "q0_wait_predicted_mean": float(np.nanmean(q0_wait)),
        "q0_buy_actual_mean": float(actual_buy.mean()),
        "q0_wait_actual_mean": float(wait_actual.mean()),
    }


def common_episode_comparison(
    baseline: pd.DataFrame,
    policy: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    if baseline.empty or policy.empty:
        return {"month": month, "common_episodes": 0}
    columns = EPISODE_KEYS + [
        "t",
        "realized_base_net_return_pct",
        "minutes_since_10pct_cross",
    ]
    merged = baseline.loc[:, columns].merge(
        policy.loc[:, columns],
        on=EPISODE_KEYS,
        suffixes=("_baseline", "_policy"),
    )
    if merged.empty:
        return {"month": month, "common_episodes": 0}

    delta = (
        merged["realized_base_net_return_pct_policy"]
        - merged["realized_base_net_return_pct_baseline"]
    )
    delay = (
        merged["minutes_since_10pct_cross_policy"]
        - merged["minutes_since_10pct_cross_baseline"]
    )
    return {
        "month": month,
        "common_episodes": int(len(merged)),
        "policy_minus_baseline_base_mean_pct": float(delta.mean()),
        "policy_better_rate": float(delta.gt(0).mean()),
        "policy_later_rate": float(delay.gt(0).mean()),
        "median_entry_delay_minutes": float(delay.median()),
    }


def run_lomo(
    monthly_frames: dict[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    metric_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []

    for holdout_month, holdout in monthly_frames.items():
        coverage_rows.append(_coverage_row(holdout, holdout_month))
        train = pd.concat(
            [
                frame
                for month, frame in monthly_frames.items()
                if month != holdout_month
            ],
            ignore_index=True,
        )
        models = train_sequential_models(train)
        diagnostic_rows.append(
            holdout_diagnostics(holdout, models, month=holdout_month)
        )

        baseline = select_earliest(holdout)
        policy, paths = select_sequential_policy(holdout, models)
        selected = {
            "earliest_eligible_10m_cap1": baseline,
            "fitted_sequential_q_cap1": policy,
        }

        for policy_name, trades in selected.items():
            metric_rows.append(
                _metrics(trades, month=holdout_month, policy=policy_name)
            )
            if not trades.empty:
                item = trades.copy()
                item["month"] = holdout_month
                item["policy"] = policy_name
                trade_frames.append(item)

        path_rows.append({"month": holdout_month, **paths})
        comparison_rows.append(
            common_episode_comparison(
                baseline,
                policy,
                month=holdout_month,
            )
        )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(),
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(comparison_rows),
        pd.DataFrame(coverage_rows),
        pd.DataFrame(path_rows),
    )


def success_check(
    details: pd.DataFrame,
    diagnostics: pd.DataFrame,
    coverage: pd.DataFrame,
    bootstrap: dict[str, float | int],
) -> dict[str, bool]:
    baseline = details.loc[
        details["policy"].eq("earliest_eligible_10m_cap1")
    ].set_index("month")
    policy = details.loc[
        details["policy"].eq("fitted_sequential_q_cap1")
    ].set_index("month")
    months = sorted(set(baseline.index) & set(policy.index))

    checks = {
        "enough_trades_every_month": bool(
            len(months) == 3
            and policy.loc[months, "trades"].ge(MIN_MONTH_TRADES).all()
        ),
        "base_positive_every_month": bool(
            len(months) == 3
            and policy.loc[months, "base_mean_pct"].gt(0).all()
        ),
        "day_positive_every_month": bool(
            len(months) == 3
            and policy.loc[
                months, "day_balanced_base_mean_pct"
            ].gt(0).all()
        ),
        "improves_day_vs_earliest_every_month": bool(
            len(months) == 3
            and (
                policy.loc[months, "day_balanced_base_mean_pct"]
                > baseline.loc[months, "day_balanced_base_mean_pct"]
            ).all()
        ),
        "tail_and_stress_not_worse_every_month": bool(
            len(months) == 3
            and (
                policy.loc[months, "base_p05_pct"]
                >= baseline.loc[months, "base_p05_pct"]
            ).all()
            and (
                policy.loc[months, "stress_mean_pct"]
                >= baseline.loc[months, "stress_mean_pct"]
            ).all()
        ),
        "q0_buy_and_wait_spearman_positive_every_month": bool(
            len(diagnostics) == 3
            and diagnostics["q0_buy_spearman"].gt(0).all()
            and diagnostics["q0_wait_spearman"].gt(0).all()
        ),
        "external_coverage_pass_every_month": bool(
            len(coverage) == 3
            and coverage["short_volume_latest_coverage"].ge(0.90).all()
            and coverage["short_interest_latest_coverage"].ge(0.90).all()
            and coverage["eight_k_query_complete_coverage"].eq(1.0).all()
        ),
        "pooled_day_bootstrap_lower_bound_positive": bool(
            np.isfinite(float(bootstrap["ci_low_pct"]))
            and float(bootstrap["ci_low_pct"]) > 0
        ),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def render_report(
    details: pd.DataFrame,
    diagnostics: pd.DataFrame,
    comparisons: pd.DataFrame,
    coverage: pd.DataFrame,
    paths: pd.DataFrame,
    bootstrap: dict[str, float | int],
    checks: dict[str, bool],
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker State Fitted Sequential Action Value v1.7 ===",
            "actions=BUY10m / WAIT5m / SKIP",
            "training=chronological thirds A(+10), B(+5), C(stage0)",
            "decision_rule=argmax fitted Q versus SKIP value 0",
            "external_features=frozen v1.3-v1.6 multi-source set",
            "evaluation=leave-one-month-out January-March 2026",
            "NOTE=adaptive development diagnostic only; no fresh month consumed",
            "",
            "=== External feature coverage ===",
            coverage.to_string(index=False),
            "",
            "=== Holdout Q diagnostics ===",
            diagnostics.to_string(index=False),
            "",
            "=== Month-by-month policy results ===",
            details.to_string(index=False),
            "",
            "=== Sequential policy path counts ===",
            paths.to_string(index=False),
            "",
            "=== Common-episode comparison ===",
            comparisons.to_string(index=False),
            "",
            "=== Pooled policy day-cluster bootstrap ===",
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
        prog="python -m victory_trader.state_fitted_sequential_q"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--comparisons-csv", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--paths-csv", type=Path, required=True)
    args = parser.parse_args()

    monthly = {label: pd.read_parquet(path) for label, path in args.dataset}
    details, trades, diagnostics, comparisons, coverage, paths = run_lomo(monthly)
    bootstrap = day_cluster_bootstrap(
        trades,
        policy="fitted_sequential_q_cap1",
        samples=BOOTSTRAP_SAMPLES,
    )
    checks = success_check(details, diagnostics, coverage, bootstrap)
    report = render_report(
        details,
        diagnostics,
        comparisons,
        coverage,
        paths,
        bootstrap,
        checks,
    )
    print(report)

    for path in (
        args.report,
        args.details_csv,
        args.trades_csv,
        args.diagnostics_csv,
        args.comparisons_csv,
        args.coverage_csv,
        args.paths_csv,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    diagnostics.to_csv(args.diagnostics_csv, index=False)
    comparisons.to_csv(args.comparisons_csv, index=False)
    coverage.to_csv(args.coverage_csv, index=False)
    paths.to_csv(args.paths_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
