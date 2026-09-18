from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, log_loss

from .state_action_risk import SEVERE_LOSS_PCT
from .state_action_value import _sample_training
from .state_multi_source_value import (
    _coverage_row,
    multi_source_action_feature_frame,
)
from .state_pairwise_buy_wait import (
    CHECKPOINT_OFFSETS_MIN,
    EPISODE_KEYS,
    HOLD_MINUTES,
    MIN_MONTH_TRADES,
    TARGET_COLUMN,
    GROSS_COLUMN,
    STRESS_COLUMN,
    WAIT_MS,
)
from .state_rank_turn import day_cluster_bootstrap
from .state_value_model import _chronological_fit_calibration_split, _eligible


ACTION_BUY = 0
ACTION_WAIT = 1
ACTION_SKIP = 2
ACTION_NAMES = {
    ACTION_BUY: "BUY",
    ACTION_WAIT: "WAIT",
    ACTION_SKIP: "SKIP",
}
BOOTSTRAP_SAMPLES = 10_000
POLICIES = ("earliest_eligible_10m_cap1", "buy_wait_skip_cap1")


def three_action_label(frame: pd.DataFrame) -> pd.Series:
    work = frame.loc[:, EPISODE_KEYS + ["t", TARGET_COLUMN]].copy()
    work["_original_index"] = work.index
    work["_t_num"] = pd.to_numeric(work["t"], errors="coerce")
    work["_buy"] = pd.to_numeric(work[TARGET_COLUMN], errors="coerce")

    future = work.loc[:, EPISODE_KEYS + ["_t_num", "_buy"]].copy()
    future["_t_num"] = future["_t_num"] - WAIT_MS
    future = future.rename(columns={"_buy": "_wait"})

    merged = work.merge(
        future,
        on=EPISODE_KEYS + ["_t_num"],
        how="left",
        validate="one_to_one",
    )
    valid = merged["_buy"].notna() & merged["_wait"].notna()

    label = np.full(len(merged), np.nan, dtype=float)
    buy = merged["_buy"].to_numpy(dtype=float)
    wait = merged["_wait"].to_numpy(dtype=float)
    valid_arr = valid.to_numpy(dtype=bool)

    skip_mask = valid_arr & (buy <= 0.0) & (wait <= 0.0)
    buy_mask = valid_arr & (buy > 0.0) & (buy >= wait)
    wait_mask = valid_arr & ~(skip_mask | buy_mask)

    label[skip_mask] = ACTION_SKIP
    label[buy_mask] = ACTION_BUY
    label[wait_mask] = ACTION_WAIT

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    original = merged["_original_index"].to_numpy()
    result.loc[original[valid_arr]] = label[valid_arr]
    return result


@dataclass(frozen=True)
class BuyWaitSkipModel:
    model: HistGradientBoostingClassifier
    feature_columns: tuple[str, ...]


def train_action_model(train: pd.DataFrame) -> BuyWaitSkipModel:
    scoreable = train.loc[_eligible(train)].copy()
    scoreable["_action_label"] = three_action_label(scoreable)
    fit_period, _ = _chronological_fit_calibration_split(scoreable)

    fit = fit_period.loc[
        pd.to_numeric(fit_period["_action_label"], errors="coerce").notna()
    ].copy()
    fit = _sample_training(fit)
    if fit.empty:
        raise ValueError("no three-action training labels")

    features = multi_source_action_feature_frame(fit)
    columns = [
        column for column in features.columns if features[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable three-action features")

    y = pd.to_numeric(fit["_action_label"], errors="coerce").astype(int)
    if set(y.unique()) != {ACTION_BUY, ACTION_WAIT, ACTION_SKIP}:
        raise ValueError("three-action training fold is missing a class")

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=2.0,
        random_state=20260926,
    )
    model.fit(features.loc[:, columns], y)
    return BuyWaitSkipModel(model=model, feature_columns=tuple(columns))


def score_actions(
    frame: pd.DataFrame,
    fitted: BuyWaitSkipModel,
) -> pd.DataFrame:
    scored = frame.loc[_eligible(frame)].copy()
    features = multi_source_action_feature_frame(scored).reindex(
        columns=fitted.feature_columns
    )
    prediction = fitted.model.predict(features).astype(int)
    probabilities = fitted.model.predict_proba(features)

    scored["predicted_policy_action"] = prediction
    for class_index, action in enumerate(fitted.model.classes_):
        scored[f"predicted_action_prob_{ACTION_NAMES[int(action)].lower()}"] = (
            probabilities[:, class_index]
        )
    return scored


def action_diagnostics(
    scored: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    labels = three_action_label(scored)
    predicted = pd.to_numeric(
        scored["predicted_policy_action"], errors="coerce"
    )
    valid = labels.notna() & predicted.notna()
    y = labels.loc[valid].astype(int)
    p = predicted.loc[valid].astype(int)

    if len(y) == 0:
        return {"month": month, "rows": 0}

    probability_columns = [
        "predicted_action_prob_buy",
        "predicted_action_prob_wait",
        "predicted_action_prob_skip",
    ]
    probabilities = scored.loc[valid, probability_columns].to_numpy(dtype=float)

    row: dict[str, object] = {
        "month": month,
        "rows": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
    }
    for action, name in ACTION_NAMES.items():
        actual_mask = y.eq(action)
        row[f"actual_{name.lower()}_rate"] = float(actual_mask.mean())
        row[f"predicted_{name.lower()}_rate"] = float(p.eq(action).mean())
        row[f"{name.lower()}_recall"] = (
            float(p.loc[actual_mask].eq(action).mean())
            if actual_mask.any()
            else np.nan
        )
    return row


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


def _checkpoint_rows(
    group: pd.DataFrame,
) -> tuple[pd.Series, dict[int, pd.Series]]:
    ordered = group.sort_values("t", kind="stable")
    first = ordered.iloc[0]
    first_t = int(first["t"])
    by_t = {int(row["t"]): row for _, row in ordered.iterrows()}
    checkpoints: dict[int, pd.Series] = {}
    for offset in CHECKPOINT_OFFSETS_MIN:
        target_t = first_t + offset * 60_000
        if target_t in by_t:
            checkpoints[offset] = by_t[target_t]
    return first, checkpoints


def select_policy(
    scored: pd.DataFrame,
    *,
    policy: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    trades: list[dict[str, object]] = []
    path_counts = {
        "episodes": 0,
        "buy_at_0": 0,
        "buy_at_5": 0,
        "buy_at_10": 0,
        "skip_predicted": 0,
        "skip_wait_end": 0,
        "skip_missing_checkpoint": 0,
        "buy_unfillable": 0,
    }

    for _, group in scored.groupby(EPISODE_KEYS, sort=False):
        path_counts["episodes"] += 1
        first, checkpoints = _checkpoint_rows(group)

        if policy == "earliest_eligible_10m_cap1":
            probability = pd.to_numeric(
                pd.Series([first.get("predicted_action_prob_buy", np.nan)]),
                errors="coerce",
            ).iloc[0]
            trade = _attempt_trade(first, decision_score=float(probability))
            if trade is not None:
                trades.append(trade)
            else:
                path_counts["buy_unfillable"] += 1
            continue

        if policy != "buy_wait_skip_cap1":
            raise ValueError(f"unknown policy: {policy}")

        finished = False
        for offset in CHECKPOINT_OFFSETS_MIN:
            row = checkpoints.get(offset)
            if row is None:
                path_counts["skip_missing_checkpoint"] += 1
                finished = True
                break

            action = int(row["predicted_policy_action"])
            if action == ACTION_SKIP:
                path_counts["skip_predicted"] += 1
                finished = True
                break

            if action == ACTION_WAIT:
                if offset == CHECKPOINT_OFFSETS_MIN[-1]:
                    path_counts["skip_wait_end"] += 1
                    finished = True
                    break
                continue

            if action != ACTION_BUY:
                raise ValueError(f"unknown predicted action: {action}")

            probability = float(row["predicted_action_prob_buy"])
            trade = _attempt_trade(row, decision_score=probability)
            if trade is not None:
                trades.append(trade)
                path_counts[f"buy_at_{offset}"] += 1
            else:
                path_counts["buy_unfillable"] += 1
            finished = True
            break

        if not finished:
            path_counts["skip_wait_end"] += 1

    return pd.DataFrame(trades), path_counts


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
    diagnostics: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
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
        fitted = train_action_model(train)
        scored = score_actions(holdout, fitted)
        diagnostics.append(action_diagnostics(scored, month=holdout_month))

        selected: dict[str, pd.DataFrame] = {}
        for policy in POLICIES:
            trades, paths = select_policy(scored, policy=policy)
            selected[policy] = trades
            metric_rows.append(
                _metrics(trades, month=holdout_month, policy=policy)
            )
            if policy == "buy_wait_skip_cap1":
                path_rows.append({"month": holdout_month, **paths})
            if not trades.empty:
                item = trades.copy()
                item["month"] = holdout_month
                item["policy"] = policy
                trade_frames.append(item)

        comparisons.append(
            common_episode_comparison(
                selected["earliest_eligible_10m_cap1"],
                selected["buy_wait_skip_cap1"],
                month=holdout_month,
            )
        )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(),
        pd.DataFrame(diagnostics),
        pd.DataFrame(comparisons),
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
        details["policy"].eq("buy_wait_skip_cap1")
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
        "balanced_accuracy_above_random_every_month": bool(
            len(diagnostics) == 3
            and diagnostics["balanced_accuracy"].gt(1.0 / 3.0).all()
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
            "=== MoneyMaker State BUY-WAIT-SKIP Policy v1.6 ===",
            "actions=BUY10m / WAIT5m / SKIP",
            "decision_rule=argmax predicted class probability",
            "checkpoints=0,+5,+10 minutes from first eligible state",
            "absolute_EV_gate=none",
            "external_features=frozen v1.3-v1.5 multi-source set",
            "evaluation=leave-one-month-out January-March 2026",
            "NOTE=adaptive development diagnostic only; no fresh month consumed",
            "",
            "=== External feature coverage ===",
            coverage.to_string(index=False),
            "",
            "=== Holdout action diagnostics ===",
            diagnostics.to_string(index=False),
            "",
            "=== Month-by-month policy results ===",
            details.to_string(index=False),
            "",
            "=== Pairwise policy path counts ===",
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
        prog="python -m victory_trader.state_buy_wait_skip"
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

    monthly = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    details, trades, diagnostics, comparisons, coverage, paths = run_lomo(monthly)
    bootstrap = day_cluster_bootstrap(
        trades,
        policy="buy_wait_skip_cap1",
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
