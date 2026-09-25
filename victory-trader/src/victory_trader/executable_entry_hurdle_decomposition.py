"""Request 198: executable-entry hurdle decomposition.

FIT-only models, chronological calibration-only diagnostics. No fresh Request-178
outcomes are evaluated. Separates entry fillability, terminal resolvability,
right-tail capture, positive economics, and closed-return ranking.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .causal_second_risk_compatible_entry import (
    RULES,
    SecondStore,
    prepare_states,
)
from .future_cost_cover_state_observability import (
    _day_weights,
    _safe_ap,
    _safe_auc,
    _safe_spearman,
)
from .market_regime_position_value import build_market_regime
from .shadow_entry_repricing_decomposition import FEATURES

REQUEST_ID = 198
CLASSIFIER_SEED = 20261110
REGRESSION_SEED = 20261120
QUANTILES = (0.80, 0.90, 0.95)
MIN_CLASS_ROWS = 120
MIN_REG_ROWS = 120


@dataclass(frozen=True)
class ClassifierHead:
    model: HistGradientBoostingClassifier
    columns: tuple[str, ...]


@dataclass(frozen=True)
class RegressorHead:
    model: HistGradientBoostingRegressor
    columns: tuple[str, ...]
    low: float
    high: float


def _feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in FEATURES
        if column in frame.columns
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    )


def _x(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def add_hurdle_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    status = result["replay_status"].astype(str)
    result["entry_fillable"] = ~status.eq("entry_unavailable")
    result["terminal_closed"] = status.eq("closed").where(
        result["entry_fillable"]
    )
    partial = result.get(
        "partial_take_reached",
        pd.Series(False, index=result.index),
    )
    result["partial_take_captured"] = (
        partial.fillna(False).astype(bool).where(result["entry_fillable"])
    )
    net = pd.to_numeric(result["policy_net_return_pct"], errors="coerce")
    result["positive_net"] = net.gt(0).where(status.eq("closed"))
    return result


def fit_classifier(
    frame: pd.DataFrame,
    target: str,
    *,
    seed: int,
) -> ClassifierHead:
    y = pd.to_numeric(frame[target], errors="coerce")
    valid = y.notna()
    train = frame.loc[valid].copy()
    actual = y.loc[valid].astype(int)
    if len(train) < MIN_CLASS_ROWS or actual.nunique() < 2:
        raise ValueError(
            f"request 198 insufficient classifier support for {target}: "
            f"rows={len(train)}, classes={actual.nunique()}"
        )
    columns = _feature_columns(train)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _x(train, columns),
        actual,
        sample_weight=_day_weights(train),
    )
    return ClassifierHead(model=model, columns=columns)


def fit_regressor(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> RegressorHead:
    y = pd.to_numeric(frame["policy_net_return_pct"], errors="coerce")
    valid = frame["replay_status"].eq("closed") & y.notna()
    train = frame.loc[valid].copy()
    actual = y.loc[valid].astype(float)
    if len(train) < MIN_REG_ROWS:
        raise ValueError(
            f"request 198 insufficient closed return support: {len(train)}"
        )
    low, high = np.quantile(actual.to_numpy(dtype=float), [0.005, 0.995])
    columns = _feature_columns(train)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _x(train, columns),
        actual.clip(lower=low, upper=high),
        sample_weight=_day_weights(train),
    )
    return RegressorHead(
        model=model,
        columns=columns,
        low=float(low),
        high=float(high),
    )


def predict_classifier(frame: pd.DataFrame, head: ClassifierHead) -> np.ndarray:
    return head.model.predict_proba(_x(frame, head.columns))[:, 1]


def predict_regressor(frame: pd.DataFrame, head: RegressorHead) -> np.ndarray:
    return head.model.predict(_x(frame, head.columns))


def _classifier_metrics(
    frame: pd.DataFrame,
    *,
    target: str,
    score: str,
) -> dict[str, object]:
    y = pd.to_numeric(frame[target], errors="coerce")
    s = pd.to_numeric(frame[score], errors="coerce")
    valid = y.notna() & s.notna()
    part = frame.loc[valid].copy()
    y = y.loc[valid]
    s = s.loc[valid]
    by_day: dict[str, object] = {}
    for day, group in part.groupby(part["trading_day"].astype(str), sort=True):
        idx = group.index
        by_day[str(day)] = {
            "rows": int(len(group)),
            "prevalence": float(y.loc[idx].mean()) if len(group) else None,
            "auc": _safe_auc(y.loc[idx], s.loc[idx]),
            "average_precision": _safe_ap(y.loc[idx], s.loc[idx]),
        }
    return {
        "rows": int(valid.sum()),
        "prevalence": float(y.mean()) if int(valid.sum()) else None,
        "auc": _safe_auc(y, s),
        "average_precision": _safe_ap(y, s),
        "by_day": by_day,
    }


def _regression_metrics(
    frame: pd.DataFrame,
    *,
    score: str,
) -> dict[str, object]:
    y = pd.to_numeric(frame["policy_net_return_pct"], errors="coerce")
    s = pd.to_numeric(frame[score], errors="coerce")
    valid = frame["replay_status"].eq("closed") & y.notna() & s.notna()
    part = frame.loc[valid].copy()
    y = y.loc[valid]
    s = s.loc[valid]
    by_day: dict[str, object] = {}
    for day, group in part.groupby(part["trading_day"].astype(str), sort=True):
        idx = group.index
        by_day[str(day)] = {
            "rows": int(len(group)),
            "spearman": _safe_spearman(y.loc[idx], s.loc[idx]),
        }
    return {
        "rows": int(valid.sum()),
        "spearman": _safe_spearman(y, s),
        "by_day": by_day,
    }


def _day_balanced_closed_mean(frame: pd.DataFrame) -> float | None:
    net = pd.to_numeric(frame["policy_net_return_pct"], errors="coerce")
    valid = frame["replay_status"].eq("closed") & net.notna()
    work = frame.loc[valid].copy()
    if work.empty:
        return None
    work["_net"] = net.loc[valid]
    daily = work.groupby(work["trading_day"].astype(str), sort=True)["_net"].mean()
    return float(daily.mean()) if len(daily) else None


def _top_groups(frame: pd.DataFrame) -> dict[str, object]:
    score = pd.to_numeric(frame["composite_score"], errors="coerce")
    outputs: dict[str, object] = {}
    for quantile in QUANTILES:
        cutoff = float(score.dropna().quantile(quantile))
        part = frame.loc[score.ge(cutoff)].copy()
        status = part["replay_status"].astype(str)
        filled = ~status.eq("entry_unavailable")
        closed = status.eq("closed")
        net = pd.to_numeric(part["policy_net_return_pct"], errors="coerce")
        outputs[f"p{int(quantile * 100)}"] = {
            "score_cutoff": cutoff,
            "states": int(len(part)),
            "episodes": int(
                part.loc[:, ["trading_day", "ticker", "hot_t"]]
                .drop_duplicates()
                .shape[0]
            ),
            "entry_fill_rate": float(filled.mean()) if len(part) else None,
            "closed_all_state_rate": float(closed.mean()) if len(part) else None,
            "closed_given_fill_rate": (
                float(closed.loc[filled].mean()) if int(filled.sum()) else None
            ),
            "positive_rate_among_closed": (
                float(net.loc[closed].gt(0).mean()) if int(closed.sum()) else None
            ),
            "day_balanced_net_return_pct": _day_balanced_closed_mean(part),
            "unresolved_rate": float(status.eq("unresolved").mean()) if len(part) else None,
            "ambiguous_rate": float(status.eq("ambiguous").mean()) if len(part) else None,
            "entry_unavailable_rate": (
                float(status.eq("entry_unavailable").mean()) if len(part) else None
            ),
        }
    return outputs


def evaluate_rule(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    seed_offset: int,
) -> dict[str, object]:
    fit = add_hurdle_labels(fit)
    calibration = add_hurdle_labels(calibration)

    heads = {
        "fill": fit_classifier(
            fit,
            "entry_fillable",
            seed=CLASSIFIER_SEED + seed_offset,
        ),
        "closed": fit_classifier(
            fit.loc[fit["entry_fillable"].fillna(False)].copy(),
            "terminal_closed",
            seed=CLASSIFIER_SEED + 10 + seed_offset,
        ),
        "partial": fit_classifier(
            fit.loc[fit["entry_fillable"].fillna(False)].copy(),
            "partial_take_captured",
            seed=CLASSIFIER_SEED + 20 + seed_offset,
        ),
        "positive": fit_classifier(
            fit.loc[fit["replay_status"].eq("closed")].copy(),
            "positive_net",
            seed=CLASSIFIER_SEED + 30 + seed_offset,
        ),
    }
    reg = fit_regressor(
        fit,
        seed=REGRESSION_SEED + seed_offset,
    )

    calibration["p_fill"] = predict_classifier(calibration, heads["fill"])
    calibration["p_closed"] = predict_classifier(calibration, heads["closed"])
    calibration["p_partial"] = predict_classifier(calibration, heads["partial"])
    calibration["p_positive"] = predict_classifier(calibration, heads["positive"])
    calibration["predicted_net"] = predict_regressor(calibration, reg)
    calibration["composite_score"] = (
        calibration["p_fill"]
        * calibration["p_closed"]
        * calibration["p_positive"]
    )

    fillable = calibration["entry_fillable"].fillna(False)
    closed = calibration["replay_status"].eq("closed")

    return {
        "heads": {
            "entry_fillability": _classifier_metrics(
                calibration,
                target="entry_fillable",
                score="p_fill",
            ),
            "terminal_resolvability_given_fill": _classifier_metrics(
                calibration.loc[fillable].copy(),
                target="terminal_closed",
                score="p_closed",
            ),
            "partial_take_capture_given_fill": _classifier_metrics(
                calibration.loc[fillable].copy(),
                target="partial_take_captured",
                score="p_partial",
            ),
            "positive_net_given_closed": _classifier_metrics(
                calibration.loc[closed].copy(),
                target="positive_net",
                score="p_positive",
            ),
            "closed_return_regression": _regression_metrics(
                calibration,
                score="predicted_net",
            ),
        },
        "composite_top_groups": _top_groups(calibration),
        "calibration_status_counts": {
            str(k): int(v)
            for k, v in calibration["replay_status"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_path)
    calibration_positions = pd.read_parquet(calibration_path)
    scan = pd.read_parquet(history_scan_path)

    fit_days = set(fit_positions["trading_day"].astype(str))
    cal_days = set(calibration_positions["trading_day"].astype(str))
    fit_scan = scan.loc[scan["trading_day"].astype(str).isin(fit_days)].copy()
    cal_scan = scan.loc[scan["trading_day"].astype(str).isin(cal_days)].copy()

    store = SecondStore()
    _, fit_by_rule = prepare_states(fit_positions, fit_scan, store)
    _, cal_by_rule = prepare_states(calibration_positions, cal_scan, store)

    diagnostics: dict[str, object] = {}
    for index, rule_name in enumerate(RULES):
        diagnostics[rule_name] = evaluate_rule(
            fit_by_rule[rule_name],
            cal_by_rule[rule_name],
            seed_offset=index,
        )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "fresh_evaluated": False,
        "promotion_eligible": False,
        "training_partition": "request171_fit_only",
        "evaluation_partition": "request171_calibration_only",
        "diagnostics": diagnostics,
        "halt_feed_by_day": dict(sorted(store.halt_status.items())),
        "second_client_stats": store.client.stats.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--history-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
