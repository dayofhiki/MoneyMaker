"""Request257: rich causal post-HOT state representation diagnostic."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score

from .config import load_settings
from .learned_pullback_entry import (
    CONTROLLER_TRAIN_DAYS,
    TEST_DAYS,
    _weights,
    build_episode_states,
    controller_columns,
    controller_x,
    fit_candidate_model,
)
from .massive_client import MassiveClient
from .rich_second_position_value import (
    RICH_SECOND_FEATURES,
    rich_second_features,
)
from .second_path_attention_probe import (
    BASELINE_FEATURES as MINUTE_DYNAMIC_FEATURES,
    SECOND_CACHE_DIR,
    SECOND_FEATURES,
    _annotate_scan,
    _second_frame,
    second_path_features,
)

REQUEST_ID = 257
TRAIN_CANDIDATE_FRACTION = 0.20
TEST_CANDIDATE_FRACTION = 0.05
PULLBACK_TRIGGER_PCT = 2.0
MIN_TRAIN_PULLBACK_ROWS = 150
MIN_TEST_PULLBACK_ROWS = 100
MIN_RICH_SECOND_COVERAGE = 0.80
MIN_RICH_VALUE_SPEARMAN = 0.10
MIN_VALUE_SPEARMAN_GAIN = 0.05
MIN_RICH_POSITIVE_AUC = 0.60
MIN_POSITIVE_AUC_GAIN = 0.03
MIN_RICH_SEVERE_AUC = 0.60

SECOND_DYNAMIC_FEATURES = tuple(
    dict.fromkeys([*SECOND_FEATURES, *RICH_SECOND_FEATURES])
)
CURRENT_MINUTE_FEATURES = tuple(
    f"current_{name}" for name in MINUTE_DYNAMIC_FEATURES
)
CURRENT_SECOND_FEATURES = tuple(
    f"current_{name}" for name in SECOND_DYNAMIC_FEATURES
)
DELTA_MINUTE_FEATURES = tuple(
    f"delta_{name}" for name in CURRENT_MINUTE_FEATURES
)
DELTA_SECOND_FEATURES = tuple(
    f"delta_{name}" for name in CURRENT_SECOND_FEATURES
)


@dataclass(frozen=True)
class RepresentationModels:
    value: HistGradientBoostingRegressor
    positive: HistGradientBoostingClassifier
    severe: HistGradientBoostingClassifier
    columns: tuple[str, ...]


def score_and_select_candidates(
    first_hot: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    model, columns = fit_candidate_model(first_hot)
    scored = first_hot.copy()
    scored["candidate_probability"] = model.predict_proba(
        controller_x(scored, columns)
    )[:, 1]
    source = scored.loc[
        scored.trading_day.astype(str).isin(CONTROLLER_TRAIN_DAYS),
        "candidate_probability",
    ]
    if len(source) < 500:
        raise ValueError("Request257 candidate threshold support too small")
    train_threshold = float(
        np.quantile(
            source.to_numpy(float),
            1.0 - TRAIN_CANDIDATE_FRACTION,
        )
    )
    test_threshold = float(
        np.quantile(
            source.to_numpy(float),
            1.0 - TEST_CANDIDATE_FRACTION,
        )
    )
    return scored, train_threshold, test_threshold


def attach_minute_features(
    states: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    annotated = _annotate_scan(scan)
    available = [
        name
        for name in MINUTE_DYNAMIC_FEATURES
        if name in annotated.columns
    ]
    dynamic = annotated.loc[
        :,
        ["trading_day", "ticker", "t", *available],
    ].copy()
    dynamic = dynamic.rename(
        columns={
            "t": "state_t",
            **{
                name: f"current_{name}"
                for name in available
            },
        }
    )
    result = states.merge(
        dynamic,
        on=["trading_day", "ticker", "state_t"],
        how="left",
        validate="many_to_one",
    )
    return result


def attach_second_features(
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    if states.empty:
        return states.copy(), {
            "ticker_days": 0,
            "rows": 0,
            "any_second_coverage": 0.0,
        }

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SECOND_CACHE_DIR / "request257",
        request_interval_seconds=0.02,
    )
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    records: list[dict[str, float]] = []

    for row in states.loc[
        :, ["trading_day", "ticker", "state_t"]
    ].to_dict("records"):
        day = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        key = (day, ticker)
        if key not in cache:
            payload = client.second_bars_range(
                ticker,
                date.fromisoformat(day),
                date.fromisoformat(day),
                adjusted=False,
            )
            cache[key] = _second_frame(payload)
        seconds = cache[key]
        features = {}
        features.update(
            second_path_features(seconds, int(row["state_t"]))
        )
        features.update(
            rich_second_features(seconds, int(row["state_t"]))
        )
        records.append(
            {
                f"current_{name}": features.get(name, np.nan)
                for name in SECOND_DYNAMIC_FEATURES
            }
        )

    dynamic = pd.DataFrame(records, index=states.index)
    result = pd.concat(
        [states.reset_index(drop=True), dynamic.reset_index(drop=True)],
        axis=1,
    )
    available = [
        name for name in CURRENT_SECOND_FEATURES
        if name in result.columns
    ]
    coverage = (
        float(result.loc[:, available].notna().any(axis=1).mean())
        if available
        else 0.0
    )
    audit = {
        "ticker_days": int(len(cache)),
        "rows": int(len(result)),
        "any_second_coverage": coverage,
        "feature_coverage": {
            name: float(
                pd.to_numeric(result[name], errors="coerce")
                .notna()
                .mean()
            )
            for name in available
        },
        "api_stats": client.stats.to_dict(),
    }
    return result, audit


def attach_dynamic_deltas(states: pd.DataFrame) -> pd.DataFrame:
    result = states.copy()
    dynamic = [
        name
        for name in [
            *CURRENT_MINUTE_FEATURES,
            *CURRENT_SECOND_FEATURES,
        ]
        if name in result.columns
    ]
    for name in dynamic:
        result[f"delta_{name}"] = np.nan

    for _, group in result.groupby(
        ["trading_day", "ticker", "hot_t"],
        sort=False,
    ):
        ordered = group.sort_values("state_t")
        if ordered.empty:
            continue
        first = ordered.iloc[0]
        for name in dynamic:
            anchor = pd.to_numeric(
                pd.Series([first.get(name)]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(anchor):
                continue
            values = pd.to_numeric(
                ordered[name],
                errors="coerce",
            )
            result.loc[
                ordered.index,
                f"delta_{name}",
            ] = values - float(anchor)
    return result


def pullback_rows(states: pd.DataFrame) -> pd.DataFrame:
    drawdown = pd.to_numeric(
        states.drawdown_from_running_high_pct,
        errors="coerce",
    )
    target = pd.to_numeric(
        states.enter_value_pct,
        errors="coerce",
    )
    return states.loc[
        drawdown.le(-PULLBACK_TRIGGER_PCT)
        & target.notna()
    ].copy()


def usable(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        name
        for name in dict.fromkeys(columns)
        if name in frame
        and pd.to_numeric(frame[name], errors="coerce")
        .notna()
        .any()
    )


def fit_representation(
    train: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
) -> RepresentationModels:
    if len(train) < MIN_TRAIN_PULLBACK_ROWS:
        raise ValueError(
            f"Request257 only {len(train)} train pullback rows"
        )
    target = pd.to_numeric(
        train.enter_value_pct,
        errors="coerce",
    ).to_numpy(float)
    low, high = np.quantile(target, [0.005, 0.995])
    weights = _weights(train)
    value = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )
    positive = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed + 1,
    )
    severe = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed + 2,
    )
    x = controller_x(train, columns)
    value.fit(
        x,
        np.clip(target, low, high),
        sample_weight=weights,
    )
    positive.fit(
        x,
        target > 0,
        sample_weight=weights,
    )
    severe.fit(
        x,
        target <= -2.0,
        sample_weight=weights,
    )
    return RepresentationModels(value, positive, severe, columns)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(y) < 20 or len(np.unique(y)) != 2:
        return None
    return float(roc_auc_score(y.astype(int), score))


def representation_report(
    test: pd.DataFrame,
    fitted: RepresentationModels,
) -> dict:
    target = pd.to_numeric(
        test.enter_value_pct,
        errors="coerce",
    )
    x = controller_x(test, fitted.columns)
    value_prediction = fitted.value.predict(x)
    positive_probability = fitted.positive.predict_proba(x)[:, 1]
    severe_probability = fitted.severe.predict_proba(x)[:, 1]
    spearman = pd.Series(value_prediction).corr(
        pd.Series(target.to_numpy(float)),
        method="spearman",
    )
    return {
        "rows": int(len(test)),
        "feature_count": len(fitted.columns),
        "value_spearman": (
            None if pd.isna(spearman) else float(spearman)
        ),
        "positive_auc": safe_auc(
            target.gt(0).to_numpy(bool),
            positive_probability,
        ),
        "severe_auc": safe_auc(
            target.le(-2).to_numpy(bool),
            severe_probability,
        ),
        "target_mean_pct": float(target.mean()),
        "target_positive_rate": float(target.gt(0).mean()),
        "target_severe_rate": float(target.le(-2).mean()),
    }


def evaluate(
    first_hot_path: Path,
    opportunity_scan_path: Path,
    output_path: Path,
) -> int:
    from .full_hot_fixed_policy_value import attach_fixed_value

    first_hot = pd.read_parquet(first_hot_path)
    scan = pd.read_parquet(opportunity_scan_path)
    first_hot = first_hot.drop(
        columns=[
            "fixed_first_watch_value_pct",
            "fixed_entry_elapsed_minutes",
        ],
        errors="ignore",
    )
    first_hot = attach_fixed_value(first_hot, scan)

    scored, train_threshold, test_threshold = score_and_select_candidates(
        first_hot
    )
    train_selected = scored.loc[
        scored.trading_day.astype(str).isin(CONTROLLER_TRAIN_DAYS)
        & scored.candidate_probability.ge(train_threshold)
    ].copy()
    test_selected = scored.loc[
        scored.trading_day.astype(str).isin(TEST_DAYS)
        & scored.candidate_probability.ge(test_threshold)
    ].copy()

    train_states = build_episode_states(train_selected, scan)
    test_states = build_episode_states(test_selected, scan)
    combined = pd.concat(
        [
            train_states.assign(_split="train"),
            test_states.assign(_split="test"),
        ],
        ignore_index=True,
    )
    combined = attach_minute_features(combined, scan)
    combined, second_audit = attach_second_features(combined)
    combined = attach_dynamic_deltas(combined)
    train_states = combined.loc[
        combined._split.eq("train")
    ].drop(columns="_split").copy()
    test_states = combined.loc[
        combined._split.eq("test")
    ].drop(columns="_split").copy()

    train_pullback = pullback_rows(train_states)
    test_pullback = pullback_rows(test_states)

    baseline_columns = controller_columns(train_pullback)
    minute_columns = usable(
        train_pullback,
        [
            *baseline_columns,
            *CURRENT_MINUTE_FEATURES,
            *DELTA_MINUTE_FEATURES,
        ],
    )
    rich_columns = usable(
        train_pullback,
        [
            *minute_columns,
            *CURRENT_SECOND_FEATURES,
            *DELTA_SECOND_FEATURES,
        ],
    )

    baseline = fit_representation(
        train_pullback,
        baseline_columns,
        20261360,
    )
    minute = fit_representation(
        train_pullback,
        minute_columns,
        20261363,
    )
    rich = fit_representation(
        train_pullback,
        rich_columns,
        20261366,
    )
    reports = {
        "baseline": representation_report(
            test_pullback,
            baseline,
        ),
        "minute": representation_report(
            test_pullback,
            minute,
        ),
        "rich": representation_report(
            test_pullback,
            rich,
        ),
    }

    b = reports["baseline"]
    r = reports["rich"]
    value_gain = (
        r["value_spearman"] - b["value_spearman"]
        if r["value_spearman"] is not None
        and b["value_spearman"] is not None
        else None
    )
    positive_gain = (
        r["positive_auc"] - b["positive_auc"]
        if r["positive_auc"] is not None
        and b["positive_auc"] is not None
        else None
    )
    checks = {
        "train_pullback_rows": (
            len(train_pullback) >= MIN_TRAIN_PULLBACK_ROWS
        ),
        "test_pullback_rows": (
            len(test_pullback) >= MIN_TEST_PULLBACK_ROWS
        ),
        "rich_second_coverage": (
            second_audit["any_second_coverage"]
            >= MIN_RICH_SECOND_COVERAGE
        ),
        "rich_value_spearman": (
            r["value_spearman"] is not None
            and r["value_spearman"] >= MIN_RICH_VALUE_SPEARMAN
        ),
        "value_spearman_gain": (
            value_gain is not None
            and value_gain >= MIN_VALUE_SPEARMAN_GAIN
        ),
        "rich_positive_auc": (
            r["positive_auc"] is not None
            and r["positive_auc"] >= MIN_RICH_POSITIVE_AUC
        ),
        "positive_auc_gain": (
            positive_gain is not None
            and positive_gain >= MIN_POSITIVE_AUC_GAIN
        ),
        "rich_severe_auc": (
            r["severe_auc"] is not None
            and r["severe_auc"] >= MIN_RICH_SEVERE_AUC
        ),
        "severe_auc_nonlower": (
            r["severe_auc"] is not None
            and b["severe_auc"] is not None
            and r["severe_auc"] >= b["severe_auc"]
        ),
    }

    result = {
        "request_id": REQUEST_ID,
        "development_only": True,
        "opens_new_dates": False,
        "promotion_eligible": False,
        "training_candidate_fraction": TRAIN_CANDIDATE_FRACTION,
        "test_candidate_fraction": TEST_CANDIDATE_FRACTION,
        "train_candidate_threshold": train_threshold,
        "test_candidate_threshold": test_threshold,
        "pullback_trigger_pct": PULLBACK_TRIGGER_PCT,
        "support": {
            "train_candidate_episodes": int(len(train_selected)),
            "test_candidate_episodes": int(len(test_selected)),
            "train_states": int(len(train_states)),
            "test_states": int(len(test_states)),
            "train_pullback_rows": int(len(train_pullback)),
            "test_pullback_rows": int(len(test_pullback)),
        },
        "second_data_audit": second_audit,
        "representations": reports,
        "rich_minus_baseline": {
            "value_spearman": value_gain,
            "positive_auc": positive_gain,
            "severe_auc": (
                r["severe_auc"] - b["severe_auc"]
                if r["severe_auc"] is not None
                and b["severe_auc"] is not None
                else None
            ),
        },
        "checks": checks,
        "representation_gate_pass": bool(all(checks.values())),
        "interpretation": (
            "All minute and one-second features are causal completed-bar "
            "aggregates. Historical one-second aggregates are not tick trades "
            "or NBBO. May11-20 is development-only."
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
