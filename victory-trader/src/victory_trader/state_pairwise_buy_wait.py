from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)

from .state_action_risk import SEVERE_LOSS_PCT
from .state_action_value import _sample_training
from .state_multi_source_value import (
    _coverage_row,
    multi_source_action_feature_frame,
)
from .state_rank_turn import day_cluster_bootstrap
from .state_value_model import (
    MINUTE_MS,
    _chronological_fit_calibration_split,
    _eligible,
)


EPISODE_KEYS = ["trading_day", "ticker"]
HOLD_MINUTES = 10
WAIT_MINUTES = 5
WAIT_MS = WAIT_MINUTES * MINUTE_MS
CHECKPOINT_OFFSETS_MIN = (0, 5, 10)
PAIRWISE_THRESHOLD = 0.50
MIN_MONTH_TRADES = 15
BOOTSTRAP_SAMPLES = 10_000
TARGET_COLUMN = "buy_return_10m_base_net_return_pct"
GROSS_COLUMN = "buy_return_10m_pct"
STRESS_COLUMN = "buy_return_10m_stress_net_return_pct"
POLICIES = (
    "earliest_eligible_10m_cap1",
    "fixed_wait5_10m_cap1",
    "fixed_wait10_10m_cap1",
    "pairwise_wait_cap1",
)


def pairwise_label(frame: pd.DataFrame) -> pd.Series:
    """1 when buying now beats buying exactly five minutes later.

    Future values are label-only. The future row's features are never joined
    into the current feature frame.
    """
    work = frame.loc[:, EPISODE_KEYS + ["t", TARGET_COLUMN]].copy()
    work["_original_index"] = work.index
    work["_t_num"] = pd.to_numeric(work["t"], errors="coerce")
    work["_now"] = pd.to_numeric(work[TARGET_COLUMN], errors="coerce")

    future = work.loc[:, EPISODE_KEYS + ["_t_num", "_now"]].copy()
    future["_t_num"] = future["_t_num"] - WAIT_MS
    future = future.rename(columns={"_now": "_wait5"})

    merged = work.merge(
        future,
        on=EPISODE_KEYS + ["_t_num"],
        how="left",
        validate="one_to_one",
    )
    valid = merged["_now"].notna() & merged["_wait5"].notna()

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    labels = merged["_now"].gt(merged["_wait5"]).astype(float)
    result.loc[merged.loc[valid, "_original_index"].to_numpy()] = (
        labels.loc[valid].to_numpy(dtype=float)
    )
    return result


class PairwisePreferenceModel:
    def __init__(
        self,
        model: HistGradientBoostingClassifier,
        feature_columns: tuple[str, ...],
    ) -> None:
        self.model = model
        self.feature_columns = feature_columns


def train_pairwise_model(train: pd.DataFrame) -> PairwisePreferenceModel:
    scoreable = train.loc[_eligible(train)].copy()
    scoreable["_pairwise_label"] = pairwise_label(scoreable)
    fit_period, _ = _chronological_fit_calibration_split(scoreable)

    fit = fit_period.loc[
        pd.to_numeric(
            fit_period["_pairwise_label"], errors="coerce"
        ).notna()
    ].copy()
    fit = _sample_training(fit)
    if fit.empty:
        raise ValueError("no pairwise training labels")

    features = multi_source_action_feature_frame(fit)
    columns = [
        column for column in features.columns if features[column].notna().any()
    ]
    if not columns:
        raise ValueError("no usable pairwise features")

    y = pd.to_numeric(fit["_pairwise_label"], errors="coerce").astype(int)
    if y.nunique() < 2:
        raise ValueError("pairwise training fold has only one class")

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=2.0,
        random_state=20260925,
    )
    model.fit(features.loc[:, columns], y)

    return PairwisePreferenceModel(
        model=model,
        feature_columns=tuple(columns),
    )


def score_pairwise(
    frame: pd.DataFrame,
    fitted: PairwisePreferenceModel,
) -> pd.DataFrame:
    scored = frame.loc[_eligible(frame)].copy()
    features = multi_source_action_feature_frame(scored).reindex(
        columns=fitted.feature_columns
    )
    probability = fitted.model.predict_proba(features)[:, 1]
    scored["predicted_now_beats_wait5_prob"] = probability
    return scored


def pairwise_diagnostics(
    scored: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    labels = pairwise_label(scored)
    probability = pd.to_numeric(
        scored["predicted_now_beats_wait5_prob"], errors="coerce"
    )
    valid = labels.notna() & probability.notna()
    y = labels.loc[valid].astype(int)
    p = probability.loc[valid].astype(float)

    if len(y) == 0:
        return {
            "month": month,
            "rows": 0,
        }

    prediction = p.ge(PAIRWISE_THRESHOLD).astype(int)
    auc = (
        float(roc_auc_score(y, p))
        if y.nunique() >= 2
        else np.nan
    )
    return {
        "month": month,
        "rows": int(len(y)),
        "now_positive_rate": float(y.mean()),
        "predicted_now_rate": float(prediction.mean()),
        "roc_auc": auc,
        "balanced_accuracy": float(
            balanced_accuracy_score(y, prediction)
        ),
        "log_loss": float(
            log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])
        ),
        "mean_probability": float(p.mean()),
    }


def _attempt_trade(row: pd.Series, *, decision_score: float) -> dict[str, object] | None:
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


def _checkpoint_rows(group: pd.DataFrame) -> tuple[pd.Series, dict[int, pd.Series]]:
    ordered = group.sort_values("t", kind="stable")
    first = ordered.iloc[0]
    first_t = int(first["t"])
    by_t = {
        int(row["t"]): row
        for _, row in ordered.iterrows()
    }
    checkpoints: dict[int, pd.Series] = {}
    for offset in CHECKPOINT_OFFSETS_MIN:
        target_t = first_t + offset * MINUTE_MS
        if target_t in by_t:
            checkpoints[offset] = by_t[target_t]
    return first, checkpoints


def select_policy_trades(
    scored: pd.DataFrame,
    *,
    policy: str,
) -> pd.DataFrame:
    trades: list[dict[str, object]] = []

    for _, group in scored.groupby(EPISODE_KEYS, sort=False):
        first, checkpoints = _checkpoint_rows(group)

        selected: pd.Series | None = None
        score = np.nan

        if policy == "earliest_eligible_10m_cap1":
            selected = first
        elif policy == "fixed_wait5_10m_cap1":
            selected = checkpoints.get(5)
        elif policy == "fixed_wait10_10m_cap1":
            if 5 in checkpoints:
                selected = checkpoints.get(10)
        elif policy == "pairwise_wait_cap1":
            for offset in CHECKPOINT_OFFSETS_MIN:
                row = checkpoints.get(offset)
                if row is None:
                    selected = None
                    break
                probability = pd.to_numeric(
                    pd.Series(
                        [row.get("predicted_now_beats_wait5_prob", np.nan)]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.isna(probability):
                    selected = None
                    break
                if float(probability) >= PAIRWISE_THRESHOLD:
                    selected = row
                    score = float(probability)
                    break
            # Three consecutive WAIT decisions mean SKIP.
        else:
            raise ValueError(f"unknown policy: {policy}")

        if selected is None:
            continue

        if not np.isfinite(score):
            score = float(
                pd.to_numeric(
                    pd.Series(
                        [
                            selected.get(
                                "predicted_now_beats_wait5_prob",
                                np.nan,
                            )
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]
            )

        trade = _attempt_trade(selected, decision_score=score)
        # A BUY decision consumes the attempt even when unfillable.
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
    entry_minute = pd.to_numeric(
        trades["minutes_since_10pct_cross"], errors="coerce"
    )

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
        "median_entry_minute": float(entry_minute.median()),
    }


def common_episode_comparison(
    baseline: pd.DataFrame,
    pairwise: pd.DataFrame,
    *,
    month: str,
) -> dict[str, object]:
    if baseline.empty or pairwise.empty:
        return {"month": month, "common_episodes": 0}

    columns = EPISODE_KEYS + [
        "t",
        "realized_base_net_return_pct",
        "minutes_since_10pct_cross",
    ]
    merged = baseline.loc[:, columns].merge(
        pairwise.loc[:, columns],
        on=EPISODE_KEYS,
        suffixes=("_baseline", "_pairwise"),
    )
    if merged.empty:
        return {"month": month, "common_episodes": 0}

    delta = (
        merged["realized_base_net_return_pct_pairwise"]
        - merged["realized_base_net_return_pct_baseline"]
    )
    delay = (
        merged["minutes_since_10pct_cross_pairwise"]
        - merged["minutes_since_10pct_cross_baseline"]
    )
    return {
        "month": month,
        "common_episodes": int(len(merged)),
        "pairwise_minus_baseline_base_mean_pct": float(delta.mean()),
        "pairwise_better_rate": float(delta.gt(0).mean()),
        "pairwise_later_rate": float(delay.gt(0).mean()),
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
]:
    metric_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

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
        fitted = train_pairwise_model(train)
        scored = score_pairwise(holdout, fitted)
        diagnostics.append(
            pairwise_diagnostics(scored, month=holdout_month)
        )

        selected = {
            policy: select_policy_trades(scored, policy=policy)
            for policy in POLICIES
        }

        for policy, trades in selected.items():
            metric_rows.append(
                _metrics(trades, month=holdout_month, policy=policy)
            )
            if not trades.empty:
                item = trades.copy()
                item["month"] = holdout_month
                item["policy"] = policy
                trade_frames.append(item)

        comparisons.append(
            common_episode_comparison(
                selected["earliest_eligible_10m_cap1"],
                selected["pairwise_wait_cap1"],
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
    pairwise = details.loc[
        details["policy"].eq("pairwise_wait_cap1")
    ].set_index("month")
    months = sorted(set(baseline.index) & set(pairwise.index))

    checks = {
        "enough_trades_every_month": bool(
            len(months) == 3
            and pairwise.loc[months, "trades"].ge(MIN_MONTH_TRADES).all()
        ),
        "base_positive_every_month": bool(
            len(months) == 3
            and pairwise.loc[months, "base_mean_pct"].gt(0).all()
        ),
        "day_positive_every_month": bool(
            len(months) == 3
            and pairwise.loc[
                months, "day_balanced_base_mean_pct"
            ].gt(0).all()
        ),
        "improves_day_vs_earliest_every_month": bool(
            len(months) == 3
            and (
                pairwise.loc[months, "day_balanced_base_mean_pct"]
                > baseline.loc[months, "day_balanced_base_mean_pct"]
            ).all()
        ),
        "tail_and_stress_not_worse_every_month": bool(
            len(months) == 3
            and (
                pairwise.loc[months, "base_p05_pct"]
                >= baseline.loc[months, "base_p05_pct"]
            ).all()
            and (
                pairwise.loc[months, "stress_mean_pct"]
                >= baseline.loc[months, "stress_mean_pct"]
            ).all()
        ),
        "pairwise_auc_above_half_every_month": bool(
            len(diagnostics) == 3
            and diagnostics["roc_auc"].gt(0.50).all()
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
    bootstrap: dict[str, float | int],
    checks: dict[str, bool],
) -> str:
    return "\n".join(
        [
            "=== MoneyMaker State Pairwise Buy-Wait Preference v1.5 ===",
            "action=BUY 10m versus WAIT exactly 5m",
            "primary_threshold=P(NOW beats WAIT5) >= 0.50",
            "decision_offsets=0,+5,+10 minutes from first eligible state",
            "absolute_EV_gate=none",
            "external_features=frozen v1.3/v1.4 multi-source set",
            "evaluation=leave-one-month-out January-March 2026",
            "NOTE=adaptive development diagnostic only; no fresh month consumed",
            "",
            "=== External feature coverage ===",
            coverage.to_string(index=False),
            "",
            "=== Holdout pairwise diagnostics ===",
            diagnostics.to_string(index=False),
            "",
            "=== Month-by-month policy results ===",
            details.to_string(index=False),
            "",
            "=== Common-episode comparison ===",
            comparisons.to_string(index=False),
            "",
            "=== Pooled pairwise day-cluster bootstrap ===",
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
        prog="python -m victory_trader.state_pairwise_buy_wait"
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
    args = parser.parse_args()

    monthly = {
        label: pd.read_parquet(path)
        for label, path in args.dataset
    }
    details, trades, diagnostics, comparisons, coverage = run_lomo(monthly)
    bootstrap = day_cluster_bootstrap(
        trades,
        policy="pairwise_wait_cap1",
        samples=BOOTSTRAP_SAMPLES,
    )
    checks = success_check(details, diagnostics, coverage, bootstrap)
    report = render_report(
        details,
        diagnostics,
        comparisons,
        coverage,
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
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    args.report.write_text(report + "\n", encoding="utf-8")
    details.to_csv(args.details_csv, index=False)
    trades.to_csv(args.trades_csv, index=False)
    diagnostics.to_csv(args.diagnostics_csv, index=False)
    comparisons.to_csv(args.comparisons_csv, index=False)
    coverage.to_csv(args.coverage_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
