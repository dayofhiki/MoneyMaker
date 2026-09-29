"""Request248: fixed-horizon position-signal audit, no controller changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from .causal_tradability_admission import (
    TRAIN_DAYS,
    fetch_second_paths,
    fit_admission,
    label_tradability,
    probability,
)
from .chronological_action_value import (
    BASE,
    CAL_DAYS,
    CAUSAL_SOURCE_FEATURES,
    FIT_DAYS,
    KEYS,
    PATH_FEATURES,
    build_states,
    features,
)
from .config import load_settings
from .execution_costs import modeled_sell_fill
from .massive_client import MassiveClient
from .second_execution_reconstruction import (
    EXPIRY_MS,
    PRIMARY_LATENCY_MS,
    first_observed_open,
)

REQUEST_ID = 248
VALUE_TRAIN_DAYS = ["2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08"]
TEST_DAYS = CAL_DAYS
HORIZONS = (1, 2, 3, 5)
ADMISSION_THRESHOLD = 0.60
MIN_TEST_ROWS = 250
MIN_SIGN_AUC = 0.55
MIN_SPEARMAN = 0.05
MIN_SPREAD = 0.30


def continuation(now_ref: float, later_ref: float) -> float:
    now_sell = modeled_sell_fill(float(now_ref), BASE)
    later_sell = modeled_sell_fill(float(later_ref), BASE)
    if not (
        np.isfinite(now_sell)
        and np.isfinite(later_sell)
        and now_sell > 0
    ):
        return np.nan
    return float((later_sell / now_sell - 1) * 100)


def horizon_targets(
    states: pd.DataFrame,
    probs: np.ndarray,
    paths: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    horizon: int,
) -> pd.DataFrame:
    work = states.reset_index(drop=True).copy()
    work["tradability_probability"] = probs
    rows = []
    for key, part in work.groupby(KEYS, sort=False):
        part = part.sort_values("state_t")
        ids = list(part.index)
        times, opens = paths.get(
            (str(key[0]), str(key[1]).upper()),
            (np.array([], dtype=np.int64), np.array([], dtype=float)),
        )
        for pos, idx in enumerate(ids):
            later_pos = pos + horizon
            if later_pos >= len(ids) or bool(work.at[idx, "terminal"]):
                continue
            later_idx = ids[later_pos]
            _, now_ref = first_observed_open(
                times,
                opens,
                decision_t=int(work.at[idx, "state_t"]),
                latency_ms=PRIMARY_LATENCY_MS,
                expiry_ms=EXPIRY_MS,
            )
            _, later_ref = first_observed_open(
                times,
                opens,
                decision_t=int(work.at[later_idx, "state_t"]),
                latency_ms=PRIMARY_LATENCY_MS,
                expiry_ms=EXPIRY_MS,
            )
            record = work.loc[idx].to_dict()
            record["target"] = (
                continuation(now_ref, later_ref)
                if now_ref is not None and later_ref is not None
                else np.nan
            )
            rows.append(record)
    return pd.DataFrame(rows)


def model_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        c
        for c in dict.fromkeys(
            [*CAUSAL_SOURCE_FEATURES, *PATH_FEATURES, "tradability_probability"]
        )
        if c in frame
        and pd.to_numeric(frame[c], errors="coerce").notna().any()
    )


def fit_model(train: pd.DataFrame):
    valid = np.isfinite(pd.to_numeric(train.target, errors="coerce"))
    train = train.loc[valid].copy()
    cols = model_columns(train)
    y = train.target.to_numpy(float)
    low, high = np.quantile(y, [0.005, 0.995])
    sizes = train.groupby(KEYS)["state_t"].transform("size").to_numpy(float)
    episodes_per_day = train[KEYS].drop_duplicates().groupby("trading_day").size()
    weights = 1 / sizes / train.trading_day.map(episodes_per_day).to_numpy(float)
    weights /= weights.mean()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=140,
        max_leaf_nodes=7,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261348,
    )
    model.fit(
        features(train, cols),
        np.clip(y, low, high),
        sample_weight=weights,
    )
    return model, cols, len(train)


def score(model, cols: tuple[str, ...], test: pd.DataFrame) -> dict:
    valid = np.isfinite(pd.to_numeric(test.target, errors="coerce"))
    evaluated = test.loc[valid].copy()
    evaluated = evaluated.loc[
        evaluated.tradability_probability >= ADMISSION_THRESHOLD
    ].copy()
    y = evaluated.target.to_numpy(float)
    pred = (
        model.predict(features(evaluated, cols))
        if len(evaluated)
        else np.array([])
    )
    auc = (
        roc_auc_score(y > 0, pred)
        if len(evaluated) and len(np.unique(y > 0)) == 2
        else None
    )
    spearman = (
        float(pd.Series(pred).corr(pd.Series(y), method="spearman"))
        if len(evaluated) > 1
        else None
    )
    spread = top_mean = top_positive_rate = None
    if len(evaluated):
        q25, q75 = np.quantile(pred, [0.25, 0.75])
        top = y[pred >= q75]
        bottom = y[pred <= q25]
        if len(top) and len(bottom):
            spread = float(np.mean(top) - np.mean(bottom))
        if len(top):
            top_mean = float(np.mean(top))
            top_positive_rate = float(np.mean(top > 0))
    return {
        "rows": int(len(evaluated)),
        "sign_auc": None if auc is None else float(auc),
        "spearman": spearman,
        "top_bottom_spread_pct": spread,
        "top_quartile_mean_pct": top_mean,
        "top_quartile_positive_rate": top_positive_rate,
    }


def evaluate(
    fit_path: Path,
    test_path: Path,
    output: Path,
) -> int:
    fit = build_states(pd.read_parquet(fit_path))
    test = build_states(pd.read_parquet(test_path))
    if sorted(fit.trading_day.astype(str).unique()) != FIT_DAYS:
        raise ValueError("fit dates changed")
    if sorted(test.trading_day.astype(str).unique()) != TEST_DAYS:
        raise ValueError("test dates changed")

    all_states = pd.concat([fit, test], ignore_index=True)
    client = MassiveClient(
        load_settings().massive_api_key,
        cache_dir=Path("data/cache/request248-second-bars"),
        request_interval_seconds=0.05,
    )
    paths = fetch_second_paths(all_states, client)
    labeled = label_tradability(all_states, paths)
    early = labeled.loc[
        labeled.trading_day.astype(str).isin(TRAIN_DAYS)
    ].copy()
    admission_model, admission_columns = fit_admission(early)

    train_states = fit.loc[
        fit.trading_day.astype(str).isin(VALUE_TRAIN_DAYS)
    ].reset_index(drop=True)
    train_probs = probability(
        admission_model,
        admission_columns,
        train_states,
    )
    test_probs = probability(
        admission_model,
        admission_columns,
        test,
    )

    reports = {}
    for horizon in HORIZONS:
        training_targets = horizon_targets(
            train_states,
            train_probs,
            paths,
            horizon,
        )
        test_targets = horizon_targets(
            test,
            test_probs,
            paths,
            horizon,
        )
        model, cols, train_rows = fit_model(training_targets)
        report = score(model, cols, test_targets)
        report["train_rows"] = int(train_rows)
        report["passes"] = bool(
            report["rows"] >= MIN_TEST_ROWS
            and (report["sign_auc"] or 0) >= MIN_SIGN_AUC
            and (report["spearman"] or 0) >= MIN_SPEARMAN
            and (report["top_bottom_spread_pct"] or 0) >= MIN_SPREAD
        )
        reports[str(horizon)] = report

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "horizons": reports,
        "signal_horizon_found": any(
            report["passes"] for report in reports.values()
        ),
        "api_stats": client.stats.to_dict(),
        "interpretation": (
            "Diagnostic only. Each horizon is fixed before outcomes; "
            "no row chooses its best future exit."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-positions", type=Path, required=True)
    parser.add_argument("--calibration-positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
