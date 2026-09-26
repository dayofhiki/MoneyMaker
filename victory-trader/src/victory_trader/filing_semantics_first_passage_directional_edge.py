"""Request 204: strictly-prior 8-K filing semantics first-passage edge."""

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
from .expanded_filing_semantics_enrichment import (
    enrich_frame,
    fetch_history,
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

REQUEST_ID = 204
FILING_CACHE_DIR = Path("data/cache/massive-rest-request204-filings")

SEMANTIC_FEATURES = (
    "filing_semantic_log1p_equity_supply_30d",
    "filing_semantic_log1p_equity_supply_7d",
    "filing_semantic_log1p_debt_30d",
    "filing_semantic_log1p_debt_7d",
    "filing_semantic_log1p_operating_30d",
    "filing_semantic_log1p_operating_7d",
    "filing_semantic_log1p_adverse_30d",
    "filing_semantic_log1p_adverse_7d",
    "filing_semantic_log1p_total_30d",
    "filing_semantic_log1p_latest_age_days",
    "filing_semantic_supply_minus_operating_balance",
)


@dataclass(frozen=True)
class Head:
    model: HistGradientBoostingClassifier
    columns: tuple[str, ...]


def add_semantic_transforms(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    raw: dict[str, pd.Series] = {}
    for bucket, source in (
        ("equity_supply_30d", "filing_equity_supply_accessions_30d"),
        ("equity_supply_7d", "filing_equity_supply_accessions_7d"),
        ("debt_30d", "filing_debt_accessions_30d"),
        ("debt_7d", "filing_debt_accessions_7d"),
        ("operating_30d", "filing_operating_catalyst_accessions_30d"),
        ("operating_7d", "filing_operating_catalyst_accessions_7d"),
        ("adverse_30d", "filing_adverse_accessions_30d"),
        ("adverse_7d", "filing_adverse_accessions_7d"),
        ("total_30d", "filing_semantic_accessions_30d"),
    ):
        values = pd.to_numeric(
            result[source],
            errors="coerce",
        ).clip(lower=0)
        raw[bucket] = values
        result[f"filing_semantic_log1p_{bucket}"] = np.log1p(
            values
        )

    age = pd.to_numeric(
        result["filing_latest_semantic_age_days"],
        errors="coerce",
    )
    result["filing_semantic_log1p_latest_age_days"] = np.log1p(
        age.where(age >= 0)
    )
    result["filing_semantic_supply_minus_operating_balance"] = (
        raw["equity_supply_30d"] - raw["operating_30d"]
    ) / (
        raw["equity_supply_30d"]
        + raw["operating_30d"]
        + 1.0
    )
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
    include_semantics: bool,
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
            "request 204 insufficient first-passage training support"
        )

    candidate_columns = (
        [*BASE_COLUMNS, *SEMANTIC_FEATURES]
        if include_semantics
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


def semantic_coverage(
    frame: pd.DataFrame,
) -> dict[str, object]:
    unique = frame.drop_duplicates(
        ["ticker", "trading_day"]
    )
    return {
        "ticker_days": int(len(unique)),
        "query_success": float(
            unique["filing_semantics_query_success"]
            .eq(1.0)
            .mean()
        ),
        "strict_prior": bool(
            unique["filing_semantics_strict_prior"]
            .eq(1.0)
            .all()
        ),
        "semantic_30d_rate": float(
            unique["filing_semantic_accessions_30d"]
            .gt(0)
            .mean()
        ),
        "equity_supply_30d_rate": float(
            unique["filing_equity_supply_accessions_30d"]
            .gt(0)
            .mean()
        ),
        "debt_30d_rate": float(
            unique["filing_debt_accessions_30d"]
            .gt(0)
            .mean()
        ),
        "operating_30d_rate": float(
            unique["filing_operating_catalyst_accessions_30d"]
            .gt(0)
            .mean()
        ),
        "adverse_30d_rate": float(
            unique["filing_adverse_accessions_30d"]
            .gt(0)
            .mean()
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
        include_semantics=False,
        seed=MODEL_SEED + seed_offset,
    )
    semantic = fit_head(
        fit_labeled,
        include_semantics=True,
        seed=MODEL_SEED + 60 + seed_offset,
    )
    cal_labeled["base_score"] = predict(
        cal_labeled,
        base,
    )
    cal_labeled["semantic_score"] = predict(
        cal_labeled,
        semantic,
    )

    base_metrics = classifier_metrics(
        cal_labeled,
        "base_score",
    )
    semantic_metrics = classifier_metrics(
        cal_labeled,
        "semantic_score",
    )
    base_auc = base_metrics["auc"]
    semantic_auc = semantic_metrics["auc"]
    auc_uplift = (
        float(semantic_auc) - float(base_auc)
        if base_auc is not None
        and semantic_auc is not None
        else None
    )

    day_wins = 0
    for day, semantic_day in semantic_metrics["by_day"].items():
        base_day = base_metrics["by_day"].get(day, {})
        if (
            semantic_day.get("auc") is not None
            and base_day.get("auc") is not None
            and float(semantic_day["auc"])
            > float(base_day["auc"])
        ):
            day_wins += 1

    top = top_groups(
        cal_labeled,
        "semantic_score",
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

    cov = semantic_coverage(cal_labeled)
    signal_pass = bool(
        semantic_auc is not None
        and float(semantic_auc) >= MIN_PRIMARY_AUC
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
        "filing_semantics": semantic_metrics,
        "filing_semantics_minus_base_auc": (
            auc_uplift
        ),
        "filing_semantics_auc_day_wins": int(
            day_wins
        ),
        "all_state_day_balanced_barrier_proxy_pct": (
            all_proxy
        ),
        "filing_semantics_top_groups": top,
        "p90_proxy_uplift_pct": proxy_uplift,
        "development_signal_pass": signal_pass,
    }


def evaluate(
    fit_path: Path,
    calibration_path: Path,
    history_scan_path: Path,
    output_path: Path,
) -> int:
    fit_positions = pd.read_parquet(
        fit_path
    )
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

    settings = load_settings()
    filing_client = MassiveClient(
        settings.massive_api_key,
        cache_dir=FILING_CACHE_DIR,
        request_interval_seconds=0.05,
    )
    history, queried = fetch_history(
        {
            "fit": fit,
            "calibration": cal,
        },
        filing_client,
    )
    fit = add_semantic_transforms(
        enrich_frame(
            fit,
            history,
            queried,
        )
    )
    cal = add_semantic_transforms(
        enrich_frame(
            cal,
            history,
            queried,
        )
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
        "semantic_features": list(
            SEMANTIC_FEATURES
        ),
        "fit_filing_coverage": (
            semantic_coverage(fit)
        ),
        "calibration_filing_coverage": (
            semantic_coverage(cal)
        ),
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
        "second_client_stats": (
            second_store.client.stats.to_dict()
        ),
        "filing_client_stats": (
            filing_client.stats.to_dict()
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
