"""Request 213: point-in-time supply admission composed with Request-208 timing.

Development-only diagnostic on the already-opened Request-178 dates.
Compare:
1) Request-208 transition timing without admission,
2) Request-209 state-only semantic-EV admission + Request-208 timing,
3) the same admission architecture augmented only with strictly-prior D-1
   point-in-time share-supply features.

No low-float/share threshold is tuned. Admission is always predicted BASE EV > 0.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression

from .config import load_settings
from .decomposed_cost_aware_entry_surplus import EPISODE_KEYS, FRESH_DAYS
from .massive_client import MassiveClient
from .opportunity_admission_transition_timing import (
    CLASSIFIER_SEED,
    NONPOSITIVE_SEED,
    POSITIVE_SEED,
    HurdleAdmissionModel,
    _admitted_timing_policy,
    attach_episode_opportunity,
    minute1_states,
    predict_admission_ev,
    train_admission_model,
)
from .pullback_turn_transition_entry import (
    MODEL_FEATURES,
    attach_transition_features,
    predict as predict_timing,
    train_model as train_timing_model,
)
from .recurrent_wait_entry_action_value import build_watch_states
from .relative_recurrent_entry_timing import (
    _matched_difference,
    _metrics,
    _policy_trades,
    attach_relative_advantage,
)
from .wait_reobserve_entry_confirmation import _day_weights, _feature_frame

REQUEST_ID = 213
SUPPLY_MODEL_SEED_OFFSET = 101
BOOTSTRAP_SEED = 20261120
BOOTSTRAP_SAMPLES = 10_000
MIN_REQUEST_SUCCESS = 0.95
MIN_WEIGHTED_COVERAGE = 0.80
MIN_ADMISSION_RATE = 0.05
MAX_ADMISSION_RATE = 0.70
MIN_TRADES = 30
MIN_IMPROVED_DAYS = 4

RAW_SUPPLY_COLUMNS = (
    "supply_weighted_shares_outstanding_prior",
    "supply_share_class_shares_outstanding_prior",
    "supply_provider_market_cap_prior",
)
SUPPLY_FEATURES = (
    "supply_log_weighted_shares_prior",
    "supply_log_share_class_shares_prior",
    "supply_log_implied_market_cap_prior",
    "supply_weighted_share_turnover_1m",
    "supply_share_class_turnover_1m",
    "supply_market_cap_turnover_1m",
    "supply_share_class_to_weighted_ratio",
)


def _positive_number(value: object) -> float:
    numeric = pd.to_numeric(
        pd.Series([value]), errors="coerce"
    ).iloc[0]
    if (
        pd.isna(numeric)
        or not np.isfinite(float(numeric))
        or float(numeric) <= 0
    ):
        return np.nan
    return float(numeric)


def enrich_supply(
    frame: pd.DataFrame,
    client: MassiveClient,
) -> pd.DataFrame:
    result = frame.copy()
    result["supply_query_date"] = ""
    result["supply_query_success"] = 0.0
    for column in RAW_SUPPLY_COLUMNS:
        result[column] = np.nan
    for column in SUPPLY_FEATURES:
        result[column] = np.nan

    cache: dict[tuple[str, str], dict[str, object]] = {}
    for index, row in result.iterrows():
        day = pd.Timestamp(str(row["trading_day"])).date()
        ticker = str(row["ticker"]).upper()
        query_day = day - timedelta(days=1)
        if query_day >= day:
            raise ValueError("request 213 supply query is not strictly prior")
        key = (query_day.isoformat(), ticker)

        if key not in cache:
            success = True
            payload: dict[str, object] = {}
            try:
                response = client.ticker_details(ticker, query_day)
                raw = response.get("results") or {}
                if isinstance(raw, dict):
                    payload = raw
                else:
                    success = False
            except Exception:
                success = False
            cache[key] = {
                "success": success,
                "weighted": _positive_number(
                    payload.get("weighted_shares_outstanding")
                ),
                "share_class": _positive_number(
                    payload.get("share_class_shares_outstanding")
                ),
                "market_cap": _positive_number(
                    payload.get("market_cap")
                ),
            }

        info = cache[key]
        weighted = float(info["weighted"])
        share_class = float(info["share_class"])
        current_price = _positive_number(row.get("current_price"))
        return_prev = pd.to_numeric(
            pd.Series(
                [row.get("return_from_previous_close_pct")]
            ),
            errors="coerce",
        ).iloc[0]
        previous_close = np.nan
        if (
            _positive_number(current_price) == current_price
            and pd.notna(return_prev)
            and np.isfinite(float(return_prev))
        ):
            denom = 1.0 + float(return_prev) / 100.0
            if denom > 0:
                previous_close = float(current_price / denom)

        log_volume = pd.to_numeric(
            pd.Series([row.get("log_minute_volume")]),
            errors="coerce",
        ).iloc[0]
        minute_volume = (
            float(np.expm1(float(log_volume)))
            if pd.notna(log_volume)
            and np.isfinite(float(log_volume))
            else np.nan
        )
        implied_cap = (
            weighted * previous_close
            if np.isfinite(weighted)
            and np.isfinite(previous_close)
            else np.nan
        )
        dollar_volume = (
            minute_volume * current_price
            if np.isfinite(minute_volume)
            and np.isfinite(current_price)
            else np.nan
        )

        result.at[index, "supply_query_date"] = query_day.isoformat()
        result.at[index, "supply_query_success"] = float(
            bool(info["success"])
        )
        result.at[
            index, "supply_weighted_shares_outstanding_prior"
        ] = weighted
        result.at[
            index, "supply_share_class_shares_outstanding_prior"
        ] = share_class
        result.at[
            index, "supply_provider_market_cap_prior"
        ] = float(info["market_cap"])

        result.at[index, "supply_log_weighted_shares_prior"] = (
            float(np.log(weighted))
            if np.isfinite(weighted) and weighted > 0
            else np.nan
        )
        result.at[
            index, "supply_log_share_class_shares_prior"
        ] = (
            float(np.log(share_class))
            if np.isfinite(share_class) and share_class > 0
            else np.nan
        )
        result.at[
            index, "supply_log_implied_market_cap_prior"
        ] = (
            float(np.log(implied_cap))
            if np.isfinite(implied_cap) and implied_cap > 0
            else np.nan
        )
        result.at[
            index, "supply_weighted_share_turnover_1m"
        ] = (
            float(minute_volume / weighted)
            if np.isfinite(minute_volume)
            and np.isfinite(weighted)
            and weighted > 0
            else np.nan
        )
        result.at[
            index, "supply_share_class_turnover_1m"
        ] = (
            float(minute_volume / share_class)
            if np.isfinite(minute_volume)
            and np.isfinite(share_class)
            and share_class > 0
            else np.nan
        )
        result.at[
            index, "supply_market_cap_turnover_1m"
        ] = (
            float(dollar_volume / implied_cap)
            if np.isfinite(dollar_volume)
            and np.isfinite(implied_cap)
            and implied_cap > 0
            else np.nan
        )
        result.at[
            index, "supply_share_class_to_weighted_ratio"
        ] = (
            float(share_class / weighted)
            if np.isfinite(share_class)
            and np.isfinite(weighted)
            and weighted > 0
            else np.nan
        )

    return result


def supply_coverage(frame: pd.DataFrame) -> dict[str, object]:
    query = pd.to_datetime(
        frame["supply_query_date"], errors="coerce"
    )
    trading = pd.to_datetime(
        frame["trading_day"], errors="coerce"
    )
    weighted = pd.to_numeric(
        frame["supply_weighted_shares_outstanding_prior"],
        errors="coerce",
    )
    share_class = pd.to_numeric(
        frame["supply_share_class_shares_outstanding_prior"],
        errors="coerce",
    )
    return {
        "rows": int(len(frame)),
        "request_success": float(
            pd.to_numeric(
                frame["supply_query_success"], errors="coerce"
            )
            .eq(1.0)
            .mean()
        ),
        "weighted_shares_coverage": float(
            weighted.notna().mean()
        ),
        "share_class_coverage": float(
            share_class.notna().mean()
        ),
        "strict_prior": bool(
            query.notna().all()
            and trading.notna().all()
            and query.lt(trading).all()
        ),
    }


def _fit_regressor(
    frame: pd.DataFrame,
    target: pd.Series,
    columns: tuple[str, ...],
    *,
    seed: int,
) -> HistGradientBoostingRegressor:
    y = target.astype(float)
    if len(frame) < 100:
        raise ValueError(
            "request 213 needs >=100 rows per magnitude branch"
        )
    low, high = np.quantile(
        y.to_numpy(dtype=float), [0.005, 0.995]
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(
        _feature_frame(frame, columns),
        y.clip(lower=low, upper=high),
        sample_weight=_day_weights(frame),
    )
    return model


def train_supply_admission(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
) -> HurdleAdmissionModel:
    target = pd.to_numeric(
        fit["episode_best_entry_3m_base_pct"],
        errors="coerce",
    )
    valid = target.notna()
    train = fit.loc[valid].copy()
    y = target.loc[valid].astype(float)
    if len(train) < 300:
        raise ValueError(
            "request 213 needs >=300 supply admission fit rows"
        )

    candidate_columns = tuple(
        dict.fromkeys([*MODEL_FEATURES, *SUPPLY_FEATURES])
    )
    columns = tuple(
        column
        for column in candidate_columns
        if column in train.columns
        and pd.to_numeric(
            train[column], errors="coerce"
        ).notna().any()
    )
    label = y.gt(0).astype(int)
    if label.nunique() < 2:
        raise ValueError(
            "request 213 supply admission fit has one class"
        )

    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=CLASSIFIER_SEED
        + SUPPLY_MODEL_SEED_OFFSET,
    )
    classifier.fit(
        _feature_frame(train, columns),
        label,
        sample_weight=_day_weights(train),
    )

    pos_mask = y.gt(0)
    neg_mask = ~pos_mask
    positive = _fit_regressor(
        train.loc[pos_mask],
        y.loc[pos_mask],
        columns,
        seed=POSITIVE_SEED + SUPPLY_MODEL_SEED_OFFSET,
    )
    nonpositive = _fit_regressor(
        train.loc[neg_mask],
        y.loc[neg_mask],
        columns,
        seed=NONPOSITIVE_SEED
        + SUPPLY_MODEL_SEED_OFFSET,
    )

    cal_target = pd.to_numeric(
        calibration["episode_best_entry_3m_base_pct"],
        errors="coerce",
    )
    cal_valid = cal_target.notna()
    cal = calibration.loc[cal_valid].copy()
    cal_y = cal_target.loc[cal_valid].astype(float)
    if len(cal) < 150 or cal_y.gt(0).nunique() < 2:
        raise ValueError(
            "request 213 needs valid two-class calibration"
        )

    features = _feature_frame(cal, columns)
    raw_p = np.clip(
        classifier.predict_proba(features)[:, 1],
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(
        raw_p / (1.0 - raw_p)
    ).reshape(-1, 1)
    platt = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=CLASSIFIER_SEED
        + SUPPLY_MODEL_SEED_OFFSET,
    )
    platt.fit(logits, cal_y.gt(0).astype(int))

    pos_cal = cal.loc[cal_y.gt(0)].copy()
    neg_cal = cal.loc[~cal_y.gt(0)].copy()
    if len(pos_cal) < 50 or len(neg_cal) < 50:
        raise ValueError(
            "request 213 needs >=50 calibration rows per branch"
        )
    pos_raw = positive.predict(
        _feature_frame(pos_cal, columns)
    )
    neg_raw = nonpositive.predict(
        _feature_frame(neg_cal, columns)
    )
    positive_offset = float(
        pd.to_numeric(
            pos_cal["episode_best_entry_3m_base_pct"],
            errors="coerce",
        ).mean()
        - float(np.mean(pos_raw))
    )
    nonpositive_offset = float(
        pd.to_numeric(
            neg_cal["episode_best_entry_3m_base_pct"],
            errors="coerce",
        ).mean()
        - float(np.mean(neg_raw))
    )
    return HurdleAdmissionModel(
        classifier=classifier,
        platt=platt,
        positive=positive,
        nonpositive=nonpositive,
        columns=columns,
        positive_offset=positive_offset,
        nonpositive_offset=nonpositive_offset,
    )


def admitted_keys(
    first: pd.DataFrame,
    fitted: HurdleAdmissionModel,
) -> tuple[set[tuple[str, str, int]], dict[str, float]]:
    expected, probability, positive, nonpositive = (
        predict_admission_ev(first, fitted)
    )
    mask = expected > 0
    selected = first.loc[mask]
    keys = {
        (
            str(row.trading_day),
            str(row.ticker),
            int(row.hot_t),
        )
        for row in selected.itertuples(index=False)
    }
    return keys, {
        "admitted": int(mask.sum()),
        "rate": float(mask.mean()) if len(mask) else 0.0,
        "ev_mean_pct": (
            float(np.mean(expected)) if len(expected) else np.nan
        ),
        "positive_probability_mean": (
            float(np.mean(probability))
            if len(probability)
            else np.nan
        ),
        "positive_magnitude_mean_pct": (
            float(np.mean(positive))
            if len(positive)
            else np.nan
        ),
        "nonpositive_magnitude_mean_pct": (
            float(np.mean(nonpositive))
            if len(nonpositive)
            else np.nan
        ),
    }


def _zero_adjusted_daily(
    first: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.Series:
    base = first.loc[:, EPISODE_KEYS].copy()
    realized = trades.loc[
        :, EPISODE_KEYS + ["realized_base_return_pct"]
    ].copy()
    merged = base.merge(
        realized,
        on=EPISODE_KEYS,
        how="left",
        validate="one_to_one",
    )
    merged["realized_base_return_pct"] = pd.to_numeric(
        merged["realized_base_return_pct"],
        errors="coerce",
    ).fillna(0.0)
    return (
        merged.groupby(
            merged["trading_day"].astype(str),
            sort=True,
        )["realized_base_return_pct"]
        .mean()
    )


def _difference_bootstrap(
    candidate: pd.Series,
    comparator: pd.Series,
) -> dict[str, object]:
    days = sorted(
        set(candidate.index.astype(str))
        & set(comparator.index.astype(str))
    )
    values = np.asarray(
        [
            float(candidate.loc[day])
            - float(comparator.loc[day])
            for day in days
        ],
        dtype=float,
    )
    if len(values) < 2:
        return {
            "days": len(values),
            "mean_difference_pct": (
                float(values.mean())
                if len(values)
                else None
            ),
            "ci_low_pct": None,
            "ci_high_pct": None,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_SAMPLES, len(values)),
    )
    draws = values[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "days": int(len(values)),
        "mean_difference_pct": float(values.mean()),
        "ci_low_pct": float(low),
        "ci_high_pct": float(high),
        "improved_days": int((values > 0).sum()),
        "by_day": {
            day: float(candidate.loc[day])
            - float(comparator.loc[day])
            for day in days
        },
    }


def evaluate(
    fit_positions_path: Path,
    calibration_positions_path: Path,
    fresh_dir: Path,
    output_path: Path,
    rows_output_path: Path,
    cache_dir: Path,
) -> int:
    fit = attach_transition_features(
        attach_episode_opportunity(
            attach_relative_advantage(
                build_watch_states(
                    pd.read_parquet(fit_positions_path)
                )
            )
        )
    )
    calibration = attach_transition_features(
        attach_episode_opportunity(
            attach_relative_advantage(
                build_watch_states(
                    pd.read_parquet(
                        calibration_positions_path
                    )
                )
            )
        )
    )

    timing_model = train_timing_model(fit, calibration)
    baseline_admission = train_admission_model(
        minute1_states(fit),
        minute1_states(calibration),
    )

    settings = load_settings()
    client = MassiveClient(
        settings.massive_api_key,
        cache_dir=cache_dir,
        request_interval_seconds=0.05,
    )
    fit_first = enrich_supply(
        minute1_states(fit), client
    )
    calibration_first = enrich_supply(
        minute1_states(calibration), client
    )
    supply_admission = train_supply_admission(
        fit_first, calibration_first
    )

    paths = sorted(fresh_dir.glob("*-positions.parquet"))
    if len(paths) != len(FRESH_DAYS):
        raise ValueError(
            "request 213 requires complete request 178 shards"
        )
    fresh_positions = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )
    fresh = attach_transition_features(
        attach_episode_opportunity(
            attach_relative_advantage(
                build_watch_states(fresh_positions)
            )
        )
    )
    if sorted(
        fresh["trading_day"].astype(str).unique()
    ) != list(FRESH_DAYS):
        raise ValueError(
            "request 213 fresh days differ from request 178"
        )

    fresh[
        "predicted_relative_advantage_pct"
    ] = predict_timing(fresh, timing_model)
    fresh_first = enrich_supply(
        minute1_states(fresh), client
    )

    baseline_keys, baseline_pred = admitted_keys(
        fresh_first, baseline_admission
    )
    supply_keys, supply_pred = admitted_keys(
        fresh_first, supply_admission
    )

    ungated, ungated_counts = _policy_trades(
        fresh, recurrent=True
    )
    baseline_gated, baseline_counts = (
        _admitted_timing_policy(fresh, baseline_keys)
    )
    supply_gated, supply_counts = (
        _admitted_timing_policy(fresh, supply_keys)
    )
    ungated["policy"] = "request208_ungated"
    baseline_gated["policy"] = "state_admission_request209"
    supply_gated["policy"] = "supply_admission_request213"

    ungated_metrics = _metrics(ungated)
    baseline_metrics = _metrics(baseline_gated)
    supply_metrics = _metrics(supply_gated)

    zero_ungated = _zero_adjusted_daily(
        fresh_first, ungated
    )
    zero_baseline = _zero_adjusted_daily(
        fresh_first, baseline_gated
    )
    zero_supply = _zero_adjusted_daily(
        fresh_first, supply_gated
    )
    supply_minus_ungated = _difference_bootstrap(
        zero_supply, zero_ungated
    )
    supply_minus_baseline = _difference_bootstrap(
        zero_supply, zero_baseline
    )

    matched_supply_vs_ungated = _matched_difference(
        supply_gated, ungated
    )
    coverage = {
        "fit": supply_coverage(fit_first),
        "calibration": supply_coverage(calibration_first),
        "fresh": supply_coverage(fresh_first),
    }
    fresh_cov = coverage["fresh"]

    supply_day = supply_metrics.get(
        "day_balanced_mean_pct"
    )
    ungated_day = ungated_metrics.get(
        "day_balanced_mean_pct"
    )
    baseline_day = baseline_metrics.get(
        "day_balanced_mean_pct"
    )
    supply_severe = supply_metrics.get(
        "severe_loss_rate"
    )
    ungated_severe = ungated_metrics.get(
        "severe_loss_rate"
    )
    supply_rate = float(supply_pred["rate"])
    supply_trades = int(supply_metrics.get("trades", 0))

    gate = bool(
        fresh_cov["request_success"]
        >= MIN_REQUEST_SUCCESS
        and fresh_cov["weighted_shares_coverage"]
        >= MIN_WEIGHTED_COVERAGE
        and fresh_cov["strict_prior"]
        and MIN_ADMISSION_RATE
        <= supply_rate
        <= MAX_ADMISSION_RATE
        and supply_trades >= MIN_TRADES
        and supply_day is not None
        and float(supply_day) > 0
        and ungated_day is not None
        and float(supply_day) > float(ungated_day)
        and baseline_day is not None
        and float(supply_day) > float(baseline_day)
        and supply_minus_ungated.get(
            "mean_difference_pct"
        ) is not None
        and float(
            supply_minus_ungated[
                "mean_difference_pct"
            ]
        ) > 0
        and supply_minus_ungated.get(
            "ci_low_pct"
        ) is not None
        and float(
            supply_minus_ungated["ci_low_pct"]
        ) > 0
        and int(
            supply_minus_ungated.get(
                "improved_days", 0
            )
        ) >= MIN_IMPROVED_DAYS
        and supply_severe is not None
        and ungated_severe is not None
        and float(supply_severe)
        <= float(ungated_severe)
    )

    result = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "opens_new_dates": False,
        "reuses_request178_dates": True,
        "promotion_eligible": False,
        "architecture": (
            "strict-prior D-1 supply admission -> "
            "Request208 recurrent timing -> fixed 3m exit"
        ),
        "supply_features": list(SUPPLY_FEATURES),
        "supply_coverage": coverage,
        "network_stats": client.stats.to_dict(),
        "baseline_state_admission_prediction": (
            baseline_pred
        ),
        "supply_admission_prediction": supply_pred,
        "ungated_request208": ungated_metrics,
        "baseline_state_admission": baseline_metrics,
        "supply_admission": supply_metrics,
        "zero_adjusted_daily": {
            "ungated_request208": {
                str(k): float(v)
                for k, v in zero_ungated.items()
            },
            "baseline_state_admission": {
                str(k): float(v)
                for k, v in zero_baseline.items()
            },
            "supply_admission": {
                str(k): float(v)
                for k, v in zero_supply.items()
            },
        },
        "zero_adjusted_supply_minus_ungated": (
            supply_minus_ungated
        ),
        "zero_adjusted_supply_minus_state_admission": (
            supply_minus_baseline
        ),
        "matched_executed_supply_minus_ungated": (
            matched_supply_vs_ungated
        ),
        "path_counts": {
            "ungated": ungated_counts,
            "state_admission": baseline_counts,
            "supply_admission": supply_counts,
        },
        "frozen_gate": {
            "min_request_success": MIN_REQUEST_SUCCESS,
            "min_weighted_coverage": (
                MIN_WEIGHTED_COVERAGE
            ),
            "strict_prior_required": True,
            "min_admission_rate": MIN_ADMISSION_RATE,
            "max_admission_rate": MAX_ADMISSION_RATE,
            "min_trades": MIN_TRADES,
            "selected_day_balanced_base_must_be_positive": True,
            "must_beat_ungated_request208": True,
            "must_beat_state_only_admission": True,
            "zero_adjusted_bootstrap_low_must_be_positive": True,
            "min_improved_days": MIN_IMPROVED_DAYS,
            "severe_loss_no_worse_than_ungated": True,
        },
        "development_gate_pass": gate,
        "promotion_gate_pass": False,
    }

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    rows_output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.concat(
        [ungated, baseline_gated, supply_gated],
        ignore_index=True,
    ).to_parquet(
        rows_output_path,
        index=False,
        compression="zstd",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-positions", type=Path, required=True
    )
    parser.add_argument(
        "--calibration-positions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fresh-dir", type=Path, required=True
    )
    parser.add_argument(
        "--output", type=Path, required=True
    )
    parser.add_argument(
        "--rows-output", type=Path, required=True
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/massive-rest"),
    )
    args = parser.parse_args()
    return evaluate(
        args.fit_positions,
        args.calibration_positions,
        args.fresh_dir,
        args.output,
        args.rows_output,
        args.cache_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
