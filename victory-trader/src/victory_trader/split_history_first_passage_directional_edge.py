"""Request 205: strictly-prior split-history first-passage edge."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .causal_first_passage_directional_edge import (
    MODEL_SEED,
    add_first_passage,
    classifier_metrics,
    prepare_directional_states,
)
from .causal_second_risk_compatible_entry import RULES, SecondStore
from .config import load_settings
from .expanded_split_history_enrichment import (
    LOOKBACK_DAYS,
    _consolidation,
    _prepare_events,
)
from .future_cost_cover_state_observability import _day_weights
from .massive_client import MassiveClient
from .premarket_first_passage_directional_edge import (
    BASE_COLUMNS,
    MIN_AUC_UPLIFT,
    MIN_DAY_WINS,
    MIN_P90_PROXY_UPLIFT_PCT,
    MIN_PRIMARY_AUC,
    _day_balanced_proxy,
    top_groups,
)

REQUEST_ID = 205
SPLIT_CACHE_DIR = Path("data/cache/massive-rest-request205-splits")

SPLIT_FEATURES = (
    "split_any_730d",
    "reverse_split_count_730d",
    "reverse_split_count_365d",
    "forward_split_count_730d",
    "stock_dividend_count_730d",
    "split_log1p_days_since_latest_reverse",
    "split_log_latest_reverse_consolidation",
    "split_log_max_reverse_consolidation",
)


@dataclass(frozen=True)
class Head:
    model: HistGradientBoostingClassifier
    columns: tuple[str, ...]


def add_split_features(
    frame: pd.DataFrame,
    events_by_ticker: dict[str, list[dict[str, object]]],
) -> pd.DataFrame:
    result = frame.copy()
    rows: list[dict[str, object]] = []

    cache: dict[tuple[str, str], dict[str, object]] = {}
    for row in result.to_dict("records"):
        day_text = str(row["trading_day"])
        ticker = str(row["ticker"]).upper()
        key = (day_text, ticker)
        if key not in cache:
            trading_day = pd.Timestamp(day_text).date()
            start = trading_day - timedelta(days=LOOKBACK_DAYS)
            records = [
                item
                for item in events_by_ticker.get(ticker, [])
                if start <= item["_execution_date"] < trading_day
            ]
            strict_prior = all(
                item["_execution_date"] < trading_day
                for item in records
            )
            reverse = [
                item
                for item in records
                if str(
                    item.get("adjustment_type", "")
                ).lower()
                == "reverse_split"
            ]
            reverse_365 = [
                item
                for item in reverse
                if (
                    trading_day
                    - item["_execution_date"]
                ).days
                <= 365
            ]
            forward = [
                item
                for item in records
                if str(
                    item.get("adjustment_type", "")
                ).lower()
                == "forward_split"
            ]
            stock_dividend = [
                item
                for item in records
                if str(
                    item.get("adjustment_type", "")
                ).lower()
                == "stock_dividend"
            ]

            latest_reverse = (
                reverse[-1] if reverse else None
            )
            latest_factor = (
                _consolidation(latest_reverse)
                if latest_reverse is not None
                else np.nan
            )
            factors = [
                value
                for value in (
                    _consolidation(item)
                    for item in reverse
                )
                if np.isfinite(value)
            ]
            days_since = (
                float(
                    (
                        trading_day
                        - latest_reverse["_execution_date"]
                    ).days
                )
                if latest_reverse is not None
                else np.nan
            )

            cache[key] = {
                "split_history_query_success": 1.0,
                "split_history_strict_prior": float(
                    strict_prior
                ),
                "split_any_730d": float(
                    bool(records)
                ),
                "reverse_split_count_730d": float(
                    len(reverse)
                ),
                "reverse_split_count_365d": float(
                    len(reverse_365)
                ),
                "forward_split_count_730d": float(
                    len(forward)
                ),
                "stock_dividend_count_730d": float(
                    len(stock_dividend)
                ),
                "split_log1p_days_since_latest_reverse": (
                    float(np.log1p(days_since))
                    if np.isfinite(days_since)
                    and days_since >= 0
                    else np.nan
                ),
                "split_log_latest_reverse_consolidation": (
                    float(np.log(latest_factor))
                    if np.isfinite(latest_factor)
                    and latest_factor > 0
                    else np.nan
                ),
                "split_log_max_reverse_consolidation": (
                    float(np.log(max(factors)))
                    if factors
                    and max(factors) > 0
                    else np.nan
                ),
            }
        rows.append(cache[key])

    enriched = pd.DataFrame(
        rows,
        index=result.index,
    )
    for column in enriched.columns:
        result[column] = enriched[column]
    return result


def _x(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(
        pd.to_numeric,
        errors="coerce",
    )


def fit_head(
    frame: pd.DataFrame,
    *,
    include_split: bool,
    seed: int,
) -> Head:
    y = pd.to_numeric(
        frame["take_before_stop"],
        errors="coerce",
    )
    valid = y.notna()
    train = frame.loc[valid].copy()
    actual = y.loc[valid].astype(int)
    if len(train) < 500 or actual.nunique() < 2:
        raise ValueError(
            "request 205 insufficient first-passage training support"
        )
    candidates = (
        [*BASE_COLUMNS, *SPLIT_FEATURES]
        if include_split
        else list(BASE_COLUMNS)
    )
    columns = tuple(
        column
        for column in candidates
        if column in train.columns
        and pd.to_numeric(
            train[column],
            errors="coerce",
        ).notna().any()
    )
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
    return Head(
        model=model,
        columns=columns,
    )


def predict(
    frame: pd.DataFrame,
    head: Head,
) -> np.ndarray:
    return head.model.predict_proba(
        _x(frame, head.columns)
    )[:, 1]


def coverage(
    frame: pd.DataFrame,
) -> dict[str, object]:
    unique = frame.drop_duplicates(
        ["ticker", "trading_day"]
    )
    reverse = pd.to_numeric(
        unique["reverse_split_count_730d"],
        errors="coerce",
    ).gt(0)
    return {
        "ticker_days": int(len(unique)),
        "query_success": float(
            pd.to_numeric(
                unique["split_history_query_success"],
                errors="coerce",
            ).eq(1.0).mean()
        ),
        "strict_prior": bool(
            pd.to_numeric(
                unique["split_history_strict_prior"],
                errors="coerce",
            ).eq(1.0).all()
        ),
        "any_split_rate_730d": float(
            pd.to_numeric(
                unique["split_any_730d"],
                errors="coerce",
            ).eq(1.0).mean()
        ),
        "reverse_split_rate_730d": float(
            reverse.mean()
        ),
        "reverse_split_ticker_days": int(
            reverse.sum()
        ),
    }


def evaluate_rule(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    second_store: SecondStore,
    *,
    rule_name: str,
    seed_offset: int,
) -> dict[str, object]:
    take_pct = RULES[rule_name].take_pct
    fit_labeled = add_first_passage(
        fit,
        second_store,
        take_pct=take_pct,
    )
    cal_labeled = add_first_passage(
        calibration,
        second_store,
        take_pct=take_pct,
    )

    base = fit_head(
        fit_labeled,
        include_split=False,
        seed=MODEL_SEED + seed_offset,
    )
    split = fit_head(
        fit_labeled,
        include_split=True,
        seed=MODEL_SEED + 70 + seed_offset,
    )
    cal_labeled["base_score"] = predict(
        cal_labeled,
        base,
    )
    cal_labeled["split_score"] = predict(
        cal_labeled,
        split,
    )

    base_metrics = classifier_metrics(
        cal_labeled,
        "base_score",
    )
    split_metrics = classifier_metrics(
        cal_labeled,
        "split_score",
    )
    base_auc = base_metrics["auc"]
    split_auc = split_metrics["auc"]
    auc_uplift = (
        float(split_auc) - float(base_auc)
        if base_auc is not None
        and split_auc is not None
        else None
    )

    day_wins = 0
    for day, split_day in split_metrics["by_day"].items():
        base_day = base_metrics["by_day"].get(day, {})
        if (
            split_day.get("auc") is not None
            and base_day.get("auc") is not None
            and float(split_day["auc"])
            > float(base_day["auc"])
        ):
            day_wins += 1

    top = top_groups(
        cal_labeled,
        "split_score",
    )
    all_proxy = _day_balanced_proxy(
        cal_labeled
    )
    p90_proxy = top["p90"][
        "day_balanced_barrier_proxy_pct"
    ]
    proxy_uplift = (
        float(p90_proxy) - float(all_proxy)
        if p90_proxy is not None
        and all_proxy is not None
        else None
    )

    cov = coverage(cal_labeled)
    signal_pass = bool(
        split_auc is not None
        and float(split_auc) >= MIN_PRIMARY_AUC
        and auc_uplift is not None
        and float(auc_uplift) >= MIN_AUC_UPLIFT
        and day_wins >= MIN_DAY_WINS
        and proxy_uplift is not None
        and float(proxy_uplift)
        >= MIN_P90_PROXY_UPLIFT_PCT
        and float(cov["query_success"]) == 1.0
        and bool(cov["strict_prior"])
    )

    return {
        "take_pct": float(take_pct),
        "base": base_metrics,
        "split": split_metrics,
        "split_minus_base_auc": auc_uplift,
        "split_auc_day_wins": int(day_wins),
        "all_state_day_balanced_barrier_proxy_pct": (
            all_proxy
        ),
        "split_top_groups": top,
        "p90_proxy_uplift_pct": proxy_uplift,
        "development_signal_pass": signal_pass,
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(fit_path)
    cal_positions = pd.read_parquet(
        calibration_path
    )
    scan = pd.read_parquet(
        history_scan_path
    )

    fit_days = set(
        fit_positions["trading_day"].astype(str)
    )
    cal_days = set(
        cal_positions["trading_day"].astype(str)
    )
    fit_scan = scan.loc[
        scan["trading_day"].astype(str).isin(fit_days)
    ].copy()
    cal_scan = scan.loc[
        scan["trading_day"].astype(str).isin(cal_days)
    ].copy()

    second_store = SecondStore()
    fit = prepare_directional_states(
        fit_positions,
        fit_scan,
        second_store,
    )
    cal = prepare_directional_states(
        cal_positions,
        cal_scan,
        second_store,
    )

    all_days = [
        pd.Timestamp(str(day)).date()
        for day in pd.concat(
            [
                fit["trading_day"],
                cal["trading_day"],
            ],
            ignore_index=True,
        ).astype(str).unique()
    ]
    query_start = min(all_days) - timedelta(
        days=LOOKBACK_DAYS
    )
    query_end = max(all_days) - timedelta(
        days=1
    )

    settings = load_settings()
    split_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SPLIT_CACHE_DIR,
        request_interval_seconds=0.05,
    )
    records = split_client.splits_market(
        execution_date_gte=query_start,
        execution_date_lte=query_end,
    )
    events = _prepare_events(records)
    fit = add_split_features(
        fit,
        events,
    )
    cal = add_split_features(
        cal,
        events,
    )

    diagnostics: dict[str, object] = {}
    for i, rule_name in enumerate(RULES):
        diagnostics[rule_name] = evaluate_rule(
            fit,
            cal,
            second_store,
            rule_name=rule_name,
            seed_offset=i,
        )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "fresh_evaluated": False,
        "promotion_eligible": False,
        "training_partition": (
            "request171_fit_only"
        ),
        "evaluation_partition": (
            "request171_calibration_only"
        ),
        "split_features": list(
            SPLIT_FEATURES
        ),
        "fit_split_coverage": coverage(fit),
        "calibration_split_coverage": coverage(cal),
        "diagnostics": diagnostics,
        "development_signal_pass": bool(
            any(
                bool(
                    item[
                        "development_signal_pass"
                    ]
                )
                for item in diagnostics.values()
            )
        ),
        "split_market_records": int(len(records)),
        "split_query_start": query_start.isoformat(),
        "split_query_end": query_end.isoformat(),
        "split_client_stats": (
            split_client.stats.to_dict()
        ),
        "second_client_stats": (
            second_store.client.stats.to_dict()
        ),
        "halt_feed_by_day": dict(
            sorted(
                second_store.halt_status.items()
            )
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--history-scan",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return evaluate(
        args.fit,
        args.calibration,
        args.history_scan,
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
