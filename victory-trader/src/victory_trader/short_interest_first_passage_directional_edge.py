"""Request 203: publication-safe short-interest first-passage edge.

FIT-only training, chronological calibration-only evaluation. Request-178 stays
sealed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
from .state_short_interest_enrichment import (
    SHORT_INTEREST_FEATURES,
    enrich_frame,
    fetch_short_interest_history,
)

REQUEST_ID = 203
MIN_LATEST_COVERAGE = 0.80
SHORT_INTEREST_CACHE_DIR = Path(
    "data/cache/massive-rest-request203-short-interest"
)


@dataclass(frozen=True)
class Head:
    model: HistGradientBoostingClassifier
    columns: tuple[str, ...]


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
    include_short_interest: bool,
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
            "request 203 insufficient first-passage training support"
        )

    candidate_columns = (
        [*BASE_COLUMNS, *SHORT_INTEREST_FEATURES]
        if include_short_interest
        else list(BASE_COLUMNS)
    )
    columns = tuple(
        column
        for column in candidate_columns
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
    latest = pd.to_numeric(
        frame["short_interest_latest"],
        errors="coerce",
    )
    reports = pd.to_numeric(
        frame["short_interest_reports_available"],
        errors="coerce",
    )
    dtc = pd.to_numeric(
        frame["short_interest_days_to_cover_latest"],
        errors="coerce",
    )
    query_complete = pd.to_numeric(
        frame["short_interest_query_complete"],
        errors="coerce",
    )

    # Publication-safe enrichment only exposes records whose official
    # publication date is strictly before the event day. The query-complete
    # marker confirms that zero-record tickers are genuine misses rather than
    # skipped requests.
    return {
        "rows": int(len(frame)),
        "query_complete_coverage": float(
            query_complete.eq(1.0).mean()
        ),
        "latest_record_coverage": float(
            latest.notna().mean()
        ),
        "two_record_coverage": float(
            reports.ge(2.0).mean()
        ),
        "days_to_cover_coverage": float(
            dtc.notna().mean()
        ),
        "publication_safe_by_construction": True,
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
        include_short_interest=False,
        seed=MODEL_SEED + seed_offset,
    )
    short = fit_head(
        fit_labeled,
        include_short_interest=True,
        seed=MODEL_SEED + 50 + seed_offset,
    )
    cal_labeled["base_score"] = predict(
        cal_labeled,
        base,
    )
    cal_labeled["short_interest_score"] = predict(
        cal_labeled,
        short,
    )

    base_metrics = classifier_metrics(
        cal_labeled,
        "base_score",
    )
    short_metrics = classifier_metrics(
        cal_labeled,
        "short_interest_score",
    )
    base_auc = base_metrics["auc"]
    short_auc = short_metrics["auc"]
    auc_uplift = (
        float(short_auc) - float(base_auc)
        if base_auc is not None
        and short_auc is not None
        else None
    )

    day_wins = 0
    for day, short_day in short_metrics["by_day"].items():
        base_day = base_metrics["by_day"].get(day, {})
        if (
            short_day.get("auc") is not None
            and base_day.get("auc") is not None
            and float(short_day["auc"])
            > float(base_day["auc"])
        ):
            day_wins += 1

    top = top_groups(
        cal_labeled,
        "short_interest_score",
    )
    all_proxy = _day_balanced_proxy(cal_labeled)
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
        short_auc is not None
        and float(short_auc) >= MIN_PRIMARY_AUC
        and auc_uplift is not None
        and float(auc_uplift) >= MIN_AUC_UPLIFT
        and day_wins >= MIN_DAY_WINS
        and proxy_uplift is not None
        and float(proxy_uplift)
        >= MIN_P90_PROXY_UPLIFT_PCT
        and float(cov["latest_record_coverage"])
        >= MIN_LATEST_COVERAGE
        and bool(cov["publication_safe_by_construction"])
    )

    return {
        "take_pct": float(take_pct),
        "base": base_metrics,
        "short_interest": short_metrics,
        "short_interest_minus_base_auc": auc_uplift,
        "short_interest_auc_day_wins": int(day_wins),
        "all_state_day_balanced_barrier_proxy_pct": (
            all_proxy
        ),
        "short_interest_top_groups": top,
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
    cal_positions = pd.read_parquet(calibration_path)
    scan = pd.read_parquet(history_scan_path)

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

    settings = load_settings()
    short_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=SHORT_INTEREST_CACHE_DIR,
        request_interval_seconds=0.05,
    )
    history, queried = fetch_short_interest_history(
        {
            "fit": fit,
            "calibration": cal,
        },
        short_client,
    )
    fit = enrich_frame(
        fit,
        history,
        queried,
    )
    cal = enrich_frame(
        cal,
        history,
        queried,
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
        "training_partition": "request171_fit_only",
        "evaluation_partition": (
            "request171_calibration_only"
        ),
        "short_interest_features": list(
            SHORT_INTEREST_FEATURES
        ),
        "fit_short_interest_coverage": coverage(fit),
        "calibration_short_interest_coverage": (
            coverage(cal)
        ),
        "diagnostics": diagnostics,
        "development_signal_pass": bool(
            any(
                bool(item["development_signal_pass"])
                for item in diagnostics.values()
            )
        ),
        "second_client_stats": (
            second_store.client.stats.to_dict()
        ),
        "short_interest_client_stats": (
            short_client.stats.to_dict()
        ),
        "halt_feed_by_day": dict(
            sorted(second_store.halt_status.items())
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
